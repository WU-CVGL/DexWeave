"""Replay complete packaged HUMOTO/GRAB trajectories with the original Rerun renderer."""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import numpy as np
import pinocchio as pin
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation


from motion_io import URDF, scan_sequences, load_motion, robot_trajectory, object_mesh_path, _smart_load_mesh


def _build_recording_camera(points: np.ndarray) -> rrb.EyeControls3D:
    mins, maxs = np.min(points, axis=0), np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, [0.8, 0.8, 1.0])
    extent = float(max(np.linalg.norm(span[:2]), span[2], 1.0))
    target = center + [0.0, 0.0, 0.18 * extent]
    position = center + np.array([-2.6, -1.8, 1.2]) * extent
    return rrb.EyeControls3D(position=position, look_target=target, eye_up=[0.0, 0.0, 1.0], speed=max(1.0, 1.8 * extent))


class RerunVisualizer:
    def __init__(self) -> None:
        self.model = pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.visual_model = pin.buildGeomFromUrdf(self.model, str(URDF), pin.GeometryType.VISUAL, package_dirs=[str(URDF.parent)])

    def log_robot_static(self) -> None:
        for geom in self.visual_model.geometryObjects:
            mesh_path = Path(geom.meshPath)
            if not mesh_path.is_file():
                raise FileNotFoundError(mesh_path)
            mesh = _smart_load_mesh(mesh_path)
            mesh.apply_scale(np.asarray(geom.meshScale, dtype=float))
            mesh.apply_transform(geom.placement.homogeneous)
            joint_name = self.model.names[geom.parentJoint]
            color = np.asarray(geom.meshColor, dtype=np.float32).copy()
            if mesh_path.parent.name in ("RH56DFX_L", "RH56DFX_R"):
                color[:3] = 0.35 if any(part in mesh_path.stem for part in ("rubber", "pad", "tip")) else 1.0
            else:
                color[:3] = 0.22 + 0.78 * color[:3]
            rr.log(
                f"world/robot/{joint_name}/geometry_{geom.name}",
                rr.Mesh3D(vertex_positions=mesh.vertices, vertex_normals=mesh.vertex_normals,
                          triangle_indices=mesh.faces.astype(np.uint32), albedo_factor=color),
                static=True,
            )

    def log_saved_scene_objects_static(self, motion: dict, dataset: str) -> list[str]:
        paths = []
        for index, (name, saved_path) in enumerate(zip(motion["object_names"], motion["object_mesh_paths"])):
            mesh = _smart_load_mesh(object_mesh_path(str(saved_path), dataset), auto_unit_scale=dataset == "HUMOTO")
            path = f"world/objects/{index}_{rr.escape_entity_path_part(str(name))}"
            color = (0.80, 0.79, 0.76) if name == "table" else colorsys.hsv_to_rgb((0.12 + index / len(motion["object_names"])) % 1.0, 0.20, 0.90)
            rr.log(
                f"{path}/mesh",
                rr.Mesh3D(vertex_positions=mesh.vertices, vertex_normals=mesh.vertex_normals,
                          triangle_indices=mesh.faces.astype(np.uint32), albedo_factor=[*color, 1.0]),
                static=True,
            )
            paths.append(path)
        return paths

    def run(self, entry: dict, save_rrd: Path | None) -> None:
        motion = load_motion(entry)
        trajectory = robot_trajectory(self.model, motion)
        frames = len(trajectory)
        points = np.concatenate([motion["root_pos"], motion["object_pos"][motion["object_active"]]], axis=0)
        blueprint = rrb.Blueprint(
            rrb.Spatial3DView(origin="/world", name=f"{entry['dataset']} / {entry['sequence']}", background=[245, 244, 238],
                              eye_controls=_build_recording_camera(points), line_grid=True),
            rrb.TimePanel(state="collapsed", timeline="frame", fps=float(motion["fps"])),
            collapse_panels=True,
        )
        if save_rrd is not None:
            save_rrd = save_rrd.expanduser().resolve()
            if save_rrd.suffix.lower() != ".rrd" or save_rrd.exists():
                raise ValueError("--save-rrd must name a new .rrd file")
            save_rrd.parent.mkdir(parents=True, exist_ok=True)
        rr.init("DexWeave_HUMOTO_GRAB", spawn=save_rrd is None, strict=True)
        try:
            if save_rrd is not None:
                rr.save(save_rrd)
            rr.send_blueprint(blueprint, make_active=True, make_default=True)
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            rr.log("world/origin", rr.Arrows3D(vectors=[[0.6, 0, 0], [0, 0.6, 0], [0, 0, 0.6]],
                                             colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]), static=True)
            self.log_robot_static()
            object_paths = self.log_saved_scene_objects_static(motion, entry["dataset"])
            quats = motion["object_quat"][..., [1, 2, 3, 0]]
            rotations = Rotation.from_quat(quats.reshape(-1, 4)).as_matrix().reshape(frames, len(object_paths), 3, 3)
            for frame, q in enumerate(trajectory):
                rr.set_time("frame", sequence=frame)
                rr.set_time("time", duration=frame * float(motion["dt"]))
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacements(self.model, self.data)
                for joint in range(1, self.model.njoints):
                    pose = self.data.oMi[joint]
                    rr.log(f"world/robot/{self.model.names[joint]}", rr.Transform3D(translation=pose.translation, mat3x3=pose.rotation))
                for index, path in enumerate(object_paths):
                    scale = float(motion["object_demo_scale"][frame, index]) if motion["object_active"][frame, index] else 0.0
                    rr.log(path, rr.Transform3D(translation=motion["object_pos"][frame, index],
                                               mat3x3=rotations[frame, index], scale=scale))
            rr.get_global_data_recording().flush()
        finally:
            rr.disconnect()
        print(f"Recorded all {frames} frames: {entry['dataset']}/{entry['sequence']}")
        if save_rrd is not None:
            print(save_rrd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("HUMOTO", "GRAB"), required=True)
    parser.add_argument("--sequence", help="Exact ID, including GRAB subject, e.g. s9/wineglass_pass_1")
    parser.add_argument("--list", action="store_true", help="List sequence IDs in the selected dataset")
    parser.add_argument("--save-rrd", type=Path, help="Save all frames to a new RRD instead of opening the viewer")
    args = parser.parse_args()
    entries = [entry for entry in scan_sequences() if entry["dataset"] == args.dataset]
    if args.list:
        if args.sequence or args.save_rrd:
            parser.error("--list cannot be combined with --sequence or --save-rrd")
        for entry in entries:
            print(f"{entry['sequence']}\t{entry['frames']} frames")
        return
    if not args.sequence:
        parser.error("--sequence is required unless --list is used")
    selected = [entry for entry in entries if entry["sequence"] == args.sequence]
    if len(selected) != 1:
        parser.error("Sequence not found; use --list to see exact IDs")
    RerunVisualizer().run(selected[0], args.save_rrd)


if __name__ == "__main__":
    main()
