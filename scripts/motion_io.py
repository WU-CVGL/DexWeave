"""Shared loading for the fixed G1/RH56DFX release; no evaluation-result dependency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "resources/g1/g1_rh56dfx.urdf"


def package_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if ROOT not in path.parents:
        raise ValueError(f"Path leaves the package: {relative}")
    return path


def scan_sequences() -> list[dict]:
    entries = []
    for path in sorted((ROOT / "data").rglob("*_opt.npz")):
        relative = path.relative_to(ROOT / "data")
        dataset, sequence = relative.parts[0], Path(*relative.parts[1:-1])
        if dataset not in ("HUMOTO", "GRAB") or len(sequence.parts) != (1 if dataset == "HUMOTO" else 2) or path.name != f"{sequence.name}_opt.npz":
            raise ValueError(f"Invalid sequence path: {relative}")
        with np.load(path, allow_pickle=False) as saved:
            frames, fps = int(saved["retimed_frame_count"]), float(saved["fps"])
        entries.append({"dataset": dataset, "sequence": sequence.as_posix(), "motion": path.relative_to(ROOT).as_posix(), "debug": path.with_name(f"{sequence.name}_val.npz").relative_to(ROOT).as_posix(), "frames": frames, "fps": fps})
    if not entries:
        raise ValueError(f"No *_opt.npz sequences found in {ROOT / 'data'}")
    return entries


def load_motion(entry: dict) -> dict:
    path = package_path(entry["motion"])
    with np.load(path, allow_pickle=False) as saved:
        motion = {key: saved[key] for key in saved.files}
    frames = int(entry["frames"])
    objects = len(motion["object_names"])
    joints = len(motion["joint_names"])
    if frames <= 0 or objects <= 0:
        raise ValueError("A recording requires frames and scene objects")
    for key, expected in (("source_dataset", entry["dataset"]), ("source_seq_name", entry["sequence"])):
        if str(motion[key]) != expected:
            raise ValueError(f"Saved {key} does not match the sequence directory")
    if str(motion["quaternion_order"]) != "wxyz":
        raise ValueError("This release stores quaternions in wxyz order")
    if int(motion["frame_start"]) != 0 or int(motion["frame_cap"]) != -1 or int(motion["retimed_frame_count"]) != frames:
        raise ValueError("Expected a complete sequence, starting at frame zero")
    if not np.isclose(float(motion["fps"]), 50.0) or not np.isclose(float(motion["dt"]), 0.02):
        raise ValueError("Expected the release's 50 Hz time grid")
    shapes = {
        "root_pos": (frames, 3), "root_quat": (frames, 4),
        "joint_pos": (frames, joints), "object_pos": (frames, objects, 3),
        "object_quat": (frames, objects, 4), "object_demo_scale": (frames, objects),
    }
    for key, shape in shapes.items():
        if motion[key].shape != shape or not np.all(np.isfinite(motion[key])):
            raise ValueError(f"{key} must be finite with shape {shape}")
    for key in ("root_quat", "object_quat"):
        if not np.allclose(np.linalg.norm(motion[key], axis=-1), 1.0, atol=1e-5, rtol=0.0):
            raise ValueError(f"{key} contains non-unit quaternions")
    active = motion["object_active"]
    if active.dtype.kind != "b" or active.shape != (frames, objects):
        raise ValueError("object_active must be a boolean (T, N) array")
    if np.any(motion["object_demo_scale"][active] <= 0.0):
        raise ValueError("Active objects must have positive display scales")
    for key in ("joint_names", "object_names"):
        names = motion[key]
        if names.ndim != 1 or len(set(names.tolist())) != len(names) or any(not str(name) for name in names):
            raise ValueError(f"{key} must contain unique, nonempty names")
    if motion["object_mesh_paths"].shape != (objects,):
        raise ValueError("Every object must have a mesh reference")
    return motion


def _smart_load_mesh(path: Path, *, auto_unit_scale: bool = True) -> trimesh.Trimesh:
    """Keep the original mesh frame and the renderer's HUMOTO mm-to-m rule."""
    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0 or not np.all(np.isfinite(mesh.vertices)):
        raise ValueError(f"Invalid triangle mesh: {path}")
    if auto_unit_scale and np.linalg.norm(mesh.extents) > 5.0:
        mesh.apply_scale(0.001)
    return mesh


def robot_trajectory(model, motion: dict) -> np.ndarray:
    names = motion["joint_names"].tolist()
    joints = [model.joints[index] for index in range(2, model.njoints)]
    model_names = [str(model.names[index]) for index in range(2, model.njoints)]
    if set(names) != set(model_names) or any(joint.nq != 1 for joint in joints):
        raise ValueError("Saved joint_names do not match the packaged robot")
    trajectory = np.zeros((len(motion["root_pos"]), model.nq), dtype=float)
    trajectory[:, :3] = motion["root_pos"]
    trajectory[:, 3:7] = motion["root_quat"][:, [1, 2, 3, 0]]
    for joint, name in zip(joints, model_names):
        trajectory[:, joint.idx_q] = motion["joint_pos"][:, names.index(name)]
    return trajectory


def object_mesh_path(saved_path: str, dataset: str) -> Path:
    matches = list((ROOT / "resources/objects" / dataset).rglob(Path(str(saved_path)).name))
    if len(matches) != 1:
        raise ValueError(f"Expected one {Path(str(saved_path)).name} in resources/objects/{dataset}; found {len(matches)}. See README for object-mesh setup.")
    return matches[0]
