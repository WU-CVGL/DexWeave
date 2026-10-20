"""Evaluate complete HUMOTO/GRAB sequences; write results to evaluatioin/ by default."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import sys

# Each batch slot owns one process; numerical libraries must not multiply that limit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import hppfcl
import mujoco
import numpy as np
import pinocchio as pin
import trimesh
from scipy.spatial.transform import Rotation

from motion_io import ROOT, URDF, scan_sequences, load_motion, object_mesh_path, package_path, robot_trajectory


CONTACT_MARGIN_M = 5e-4
DETECTION_DISTANCE_M = 0.01
QUERY_TOLERANCE_M = 1e-6
FOOT_POINT = np.array([0.14, 0.0, -0.03])
FOOT_FRAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


def rh56_finger_identity(name: str) -> tuple[str, str] | None:
    """Return the semantic side/finger for one current-RH56 link, joint, mesh, or primitive name."""
    for fragment in reversed(str(name or "").strip().lower().split(":")):
        parts = fragment.rsplit("/", 1)[-1].split(".", 1)[0].split("_")
        if (
            len(parts) >= 2
            and parts[0] in {"left", "right"}
            and parts[1] in {"thumb", "index", "middle", "ring", "little"}
        ):
            return (
                "L" if parts[0] == "left" else "R",
                "pinky" if parts[1] == "little" else parts[1],
            )
    return None


def _link_group(name: str) -> str:
    """Map a link name to one physical link assembly."""
    n = name.lower()
    # Fingers before generic base handling, e.g. left_thumb_1.
    if rh56_finger_identity(n) is not None or any(
        k in n for k in ['thumb', 'index', 'middle', 'ring', 'pinky', 'little']
    ):
        return 'finger'
    # Pelvis / base
    if (
        'pelvis' in n
        or 'pelvis_contour' in n
        or n == 'base_link'
        or n.endswith('/base_link')
        or 'imu_in_pelvis' in n
    ):
        return 'pelvis'
    # Head
    if 'head' in n:
        return 'head'
    # Torso / waist / fixed torso-mounted accessories
    if any(k in n for k in ['torso', 'waist', 'logo', 'd435', 'mid360', 'imu_in_torso']):
        return 'torso'
    # Left leg chain
    if ('left' in n or n.startswith('l_')) and any(k in n for k in ['hip', 'thigh']):
        return 'l_hip'
    if ('left' in n or n.startswith('l_')) and 'knee' in n:
        return 'l_knee'
    if ('left' in n or n.startswith('l_')) and any(k in n for k in ['ankle', 'foot']):
        return 'l_foot'
    # Right leg chain
    if ('right' in n or n.startswith('r_')) and any(k in n for k in ['hip', 'thigh']):
        return 'r_hip'
    if ('right' in n or n.startswith('r_')) and 'knee' in n:
        return 'r_knee'
    if ('right' in n or n.startswith('r_')) and any(k in n for k in ['ankle', 'foot']):
        return 'r_foot'
    # Left arm chain
    if ('left' in n or n.startswith('l_')) and 'shoulder' in n:
        return 'l_shoulder'
    if ('left' in n or n.startswith('l_')) and 'elbow' in n:
        return 'l_elbow'
    if ('left' in n or n.startswith('l_')) and any(k in n for k in ['wrist', 'hand']):
        return 'l_hand'
    # Right arm chain
    if ('right' in n or n.startswith('r_')) and 'shoulder' in n:
        return 'r_shoulder'
    if ('right' in n or n.startswith('r_')) and 'elbow' in n:
        return 'r_elbow'
    if ('right' in n or n.startswith('r_')) and any(k in n for k in ['wrist', 'hand']):
        return 'r_hand'
    return n


_ADJACENT_MECHANICAL_GROUPS = frozenset(
    frozenset(pair)
    for pair in (
        ('pelvis', 'torso'),
        ('torso', 'head'),
        ('pelvis', 'l_hip'),
        ('pelvis', 'r_hip'),
        ('l_hip', 'l_knee'),
        ('r_hip', 'r_knee'),
        ('l_knee', 'l_foot'),
        ('r_knee', 'r_foot'),
        ('l_shoulder', 'l_elbow'),
        ('r_shoulder', 'r_elbow'),
        ('l_elbow', 'l_hand'),
        ('r_elbow', 'r_hand'),
    )
)


def mujoco_self_collision_pairs(model, geom_ids) -> frozenset[tuple[int, int]]:
    """Return non-structural robot self-collision pairs from MuJoCo geometry."""
    selected = tuple(sorted({int(geom_id) for geom_id in geom_ids}))
    if any(geom_id < 0 or geom_id >= int(model.ngeom) for geom_id in selected):
        raise ValueError("Self-collision geom id is outside the MuJoCo model")
    body_cluster = np.full((int(model.nbody),), -1, dtype=int)
    cluster_parent: dict[int, int] = {}
    body_cluster[0] = 0
    for body_id in range(1, int(model.nbody)):
        parent_body = int(model.body_parentid[body_id])
        parent_cluster = int(body_cluster[parent_body])
        if parent_cluster < 0:
            raise ValueError(f"MuJoCo body topology is not parent-before-child at body {body_id}")
        if int(model.body_jntnum[body_id]) == 0:
            body_cluster[body_id] = parent_cluster
        else:
            body_cluster[body_id] = body_id
            cluster_parent[body_id] = parent_cluster
    cluster_names = {
        cluster: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cluster) or ""
        for cluster in set(int(value) for value in body_cluster if int(value) > 0)
    }
    cluster_finger = {
        cluster: rh56_finger_identity(name) for cluster, name in cluster_names.items()
    }
    cluster_group = {cluster: _link_group(name) for cluster, name in cluster_names.items()}
    pairs = set()
    for slot, geom_a in enumerate(selected):
        cluster_a = int(body_cluster[int(model.geom_bodyid[geom_a])])
        for geom_b in selected[slot + 1 :]:
            cluster_b = int(body_cluster[int(model.geom_bodyid[geom_b])])
            finger_a, finger_b = cluster_finger.get(cluster_a), cluster_finger.get(cluster_b)
            group_a, group_b = cluster_group.get(cluster_a), cluster_group.get(cluster_b)
            same_hand_fingers = (
                finger_a is not None and finger_b is not None and finger_a[0] == finger_b[0]
            )
            adjacent_mechanical_groups = (
                finger_a is None
                and finger_b is None
                and (
                    group_a == group_b
                    or frozenset((group_a, group_b)) in _ADJACENT_MECHANICAL_GROUPS
                )
            )
            if (
                cluster_a == cluster_b
                or cluster_parent.get(cluster_a) == cluster_b
                or cluster_parent.get(cluster_b) == cluster_a
                or same_hand_fingers
                or adjacent_mechanical_groups
            ):
                continue
            pairs.add((geom_a, geom_b))
    return frozenset(pairs)


def mujoco_compiled_convex(model: mujoco.MjModel, geom_id: int):
    """Use MuJoCo mesh_graph hulls directly; mesh_face may still be non-convex."""

    geom_id = int(geom_id)
    if geom_id < 0 or geom_id >= int(model.ngeom):
        raise IndexError("MuJoCo geom id is out of range")
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        raise TypeError("Material soft geometry requires a MuJoCo compiled mesh geom")
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0 or mesh_id >= int(model.nmesh):
        raise ValueError("MuJoCo mesh geom has an invalid mesh id")
    vertex_start, vertex_count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    raw_vertices = np.asarray(
        model.mesh_vert[vertex_start : vertex_start + vertex_count], dtype=float
    ).reshape(vertex_count, 3)
    graph_address = int(model.mesh_graphadr[mesh_id])
    graph = np.asarray(model.mesh_graph, dtype=np.int64).reshape(-1)
    if vertex_count < 4 or not np.all(np.isfinite(raw_vertices)):
        raise ValueError("MuJoCo compiled mesh vertices are invalid")
    if graph_address < 0 or graph_address + 2 > graph.size:
        raise ValueError("MuJoCo mesh has no valid compiled collision graph")
    hull_vertex_count, hull_face_count = (
        int(value) for value in graph[graph_address : graph_address + 2]
    )
    if hull_vertex_count < 4 or hull_face_count < 4:
        raise ValueError("MuJoCo compiled collision graph is not a three-dimensional convex hull")
    graph_length = 2 + 3 * hull_vertex_count + 6 * hull_face_count
    graph_end = graph_address + graph_length
    if graph_end > graph.size:
        raise ValueError("MuJoCo compiled collision graph is truncated")
    later_addresses = np.asarray(model.mesh_graphadr, dtype=np.int64).reshape(-1)
    later_addresses = later_addresses[later_addresses > graph_address]
    stored_graph_end = int(np.min(later_addresses)) if later_addresses.size else int(graph.size)
    if graph_end != stored_graph_end:
        raise ValueError("MuJoCo compiled collision graph length does not match its stored segment")
    segment = graph[graph_address:graph_end]
    vertex_edge_addresses = segment[2 : 2 + hull_vertex_count]
    vertex_global_ids = segment[2 + hull_vertex_count : 2 + 2 * hull_vertex_count]
    edge_local_ids = segment[
        2 + 2 * hull_vertex_count : 2 + 3 * hull_vertex_count + 3 * hull_face_count
    ]
    face_global_ids = segment[2 + 3 * hull_vertex_count + 3 * hull_face_count :].reshape(
        hull_face_count, 3
    )
    if (
        vertex_edge_addresses[0] != 0
        or np.any(np.diff(vertex_edge_addresses) <= 0)
        or np.any(vertex_edge_addresses < 0)
        or np.any(vertex_edge_addresses >= edge_local_ids.size)
    ):
        raise ValueError("MuJoCo compiled collision graph has invalid vertex-edge addresses")
    edge_ends = np.concatenate(
        (vertex_edge_addresses[1:], np.asarray((edge_local_ids.size,), dtype=np.int64))
    )
    graph_neighbours = []
    for vertex_index, (edge_start, edge_end) in enumerate(
        zip(vertex_edge_addresses.tolist(), edge_ends.tolist())
    ):
        neighbours = edge_local_ids[int(edge_start) : int(edge_end)]
        if (
            neighbours.size < 2
            or int(neighbours[-1]) != -1
            or np.any(neighbours[:-1] < 0)
            or np.any(neighbours[:-1] >= hull_vertex_count)
            or np.any(neighbours[:-1] == vertex_index)
            or np.unique(neighbours[:-1]).size != neighbours.size - 1
        ):
            raise ValueError("MuJoCo compiled collision graph has an invalid hull adjacency list")
        graph_neighbours.append(frozenset(int(value) for value in neighbours[:-1]))
    if any(
        vertex_index not in graph_neighbours[neighbour]
        for vertex_index, neighbours in enumerate(graph_neighbours)
        for neighbour in neighbours
    ):
        raise ValueError("MuJoCo compiled collision graph adjacency is not symmetric")
    if (
        np.unique(vertex_global_ids).size != hull_vertex_count
        or np.any(vertex_global_ids < 0)
        or np.any(vertex_global_ids >= vertex_count)
    ):
        raise ValueError("MuJoCo compiled collision graph has invalid hull vertex ids")
    if np.any(face_global_ids < 0) or np.any(face_global_ids >= vertex_count):
        raise ValueError("MuJoCo compiled collision graph has out-of-range hull face ids")
    inverse_vertex_ids = np.full((vertex_count,), -1, dtype=np.int64)
    inverse_vertex_ids[vertex_global_ids] = np.arange(hull_vertex_count, dtype=np.int64)
    faces = inverse_vertex_ids[face_global_ids]
    vertices = raw_vertices[vertex_global_ids]
    if (
        np.any(faces < 0)
        or np.unique(face_global_ids).size != hull_vertex_count
        or np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0)
    ):
        raise ValueError("MuJoCo compiled collision graph has invalid hull faces")
    directed_edges = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0)
    edge_incidence: dict[tuple[int, int], list[tuple[int, int]]] = {}
    face_neighbours = [set() for _ in range(hull_vertex_count)]
    for first, second in directed_edges.tolist():
        first, second = int(first), int(second)
        edge_incidence.setdefault(tuple(sorted((first, second))), []).append((first, second))
        face_neighbours[first].add(second)
        face_neighbours[second].add(first)
    if any(
        len(orientations) != 2 or orientations[0] != orientations[1][::-1]
        for orientations in edge_incidence.values()
    ):
        raise ValueError(
            "MuJoCo compiled collision graph faces are not a closed oriented two-manifold"
        )
    if hull_vertex_count - len(edge_incidence) + hull_face_count != 2 or any(
        frozenset(neighbours) != graph_neighbours[index]
        for index, neighbours in enumerate(face_neighbours)
    ):
        raise ValueError("MuJoCo compiled collision graph topology is inconsistent")
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    points, triangles = hppfcl.StdVec_Vec3s(), hppfcl.StdVec_Triangle()
    for vertex in vertices:
        points.append(np.asarray(vertex, dtype=float))
    for face in faces:
        triangles.append(hppfcl.Triangle(int(face[0]), int(face[1]), int(face[2])))
    shape = hppfcl.Convex(points, triangles)
    shape.computeLocalAABB()
    shape_lower = np.asarray(shape.aabb_local.min_, dtype=float).reshape(3)
    shape_upper = np.asarray(shape.aabb_local.max_, dtype=float).reshape(3)
    shape_volume = float(shape.computeVolume())
    shape_center = np.asarray(shape.computeCOM(), dtype=float).reshape(3)
    if (
        not np.all(np.isfinite(shape_lower))
        or not np.all(np.isfinite(shape_upper))
        or np.any(shape_lower > shape_upper)
        or not np.isfinite(shape_volume)
        or shape_volume <= 0.0
        or not np.all(np.isfinite(shape_center))
    ):
        raise ValueError(
            "HPP-FCL rejected the compiled collision graph's finite convex volume contract"
        )
    if any(
        frozenset(int(value) for value in shape.neighbors(vertex_index))
        != graph_neighbours[vertex_index]
        for vertex_index in range(hull_vertex_count)
    ):
        raise ValueError(
            "HPP-FCL convex adjacency does not preserve MuJoCo's compiled collision graph"
        )
    return shape


def mujoco_collision_shape(model, geom_id: int):
    geom_type = int(model.geom_type[int(geom_id)])
    size = np.asarray(model.geom_size[int(geom_id)], dtype=float)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        return mujoco_compiled_convex(model, geom_id)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return hppfcl.Sphere(float(size[0]))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return hppfcl.Box(2.0 * np.asarray(size[:3], dtype=float))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return hppfcl.Capsule(float(size[0]), 2.0 * float(size[1]))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return hppfcl.Cylinder(float(size[0]), 2.0 * float(size[1]))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return hppfcl.Ellipsoid(np.asarray(size[:3], dtype=float))
    geom_name = (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or f"geom_{int(geom_id)}"
    )
    raise ValueError(
        f"Compiled robot collision does not support URDF geom {geom_name!r} of MuJoCo type {geom_type}"
    )


def load_collision_mesh(path, dataset):
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.all(np.isfinite(mesh.vertices)):
        raise ValueError(f"Invalid collision mesh: {path}")
    if dataset == "HUMOTO" and np.linalg.norm(mesh.extents) > 5.0:
        mesh.apply_scale(0.001)
    vertices, inverse = np.unique(np.asarray(mesh.vertices, dtype=float), axis=0, return_inverse=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=inverse[np.asarray(mesh.faces)], process=False)
    mesh.update_faces(mesh.nondegenerate_faces() & mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=True)
    return mesh


class GeometryEvaluator:
    def __init__(self):
        xml = URDF.with_name("g1_rh56dfx_mujoco.xml")
        reference = mujoco.MjModel.from_xml_path(str(xml))
        spec = mujoco.MjSpec.from_file(str(xml))
        spec.worldbody.add_geom(name="metric_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0.0, 0.0, 0.05), contype=1, conaffinity=1, margin=CONTACT_MARGIN_M)
        self.model = spec.compile()
        self.ground = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "metric_ground")
        if self.ground != 0 or self.model.ngeom != reference.ngeom + 1 or self.model.nq != reference.nq:
            raise ValueError("Adding the ground changed the collision model layout")
        for field in ("geom_type", "geom_bodyid", "geom_dataid", "geom_size", "geom_pos", "geom_quat"):
            if not np.array_equal(getattr(self.model, field)[1:], getattr(reference, field)):
                raise ValueError(f"Adding the ground changed {field}")
        self.data = mujoco.MjData(self.model)
        self.pin_model = pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())
        self.pin_data = self.pin_model.createData()
        self.foot_ids = [self.pin_model.getFrameId(name) for name in FOOT_FRAMES]
        if any(index >= self.pin_model.nframes for index in self.foot_ids):
            raise ValueError("The packaged G1 foot frames are missing")
        self.geoms = np.array([i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] > 0 and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])], dtype=int)
        self.model.geom_margin[self.geoms] = np.maximum(self.model.geom_margin[self.geoms], CONTACT_MARGIN_M)
        self.self_pairs = mujoco_self_collision_pairs(self.model, self.geoms)
        self.shapes = {int(i): mujoco_collision_shape(self.model, int(i)) for i in self.geoms}
        for shape in self.shapes.values():
            shape.computeLocalAABB()
        self.centers = np.array([(s.aabb_local.min_ + s.aabb_local.max_) * 0.5 for s in self.shapes.values()])
        self.extents = np.array([(s.aabb_local.max_ - s.aabb_local.min_) * 0.5 for s in self.shapes.values()])
        self.result = hppfcl.CollisionResult()
        self.robot_transform = hppfcl.Transform3f()

    def evaluate(self, motion, dataset):
        frames = len(motion["root_pos"])
        q_pin = robot_trajectory(self.pin_model, motion)
        names = motion["joint_names"].tolist()
        model_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, self.model.njnt)]
        if set(names) != set(model_names) or self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("MuJoCo joints do not match the complete saved configuration")
        q_mj = np.zeros((frames, self.model.nq))
        q_mj[:, :3], q_mj[:, 3:7] = motion["root_pos"], motion["root_quat"]
        for joint, name in enumerate(model_names, 1):
            if self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Expected scalar revolute joint: {name}")
            q_mj[:, self.model.jnt_qposadr[joint]] = motion["joint_pos"][:, names.index(name)]
        enabled = motion["object_collision_enabled"]
        if enabled.dtype.kind != "b" or enabled.shape != motion["object_active"].shape:
            raise ValueError("object_collision_enabled must be a boolean (T, N) array")
        rotations = Rotation.from_quat(motion["object_quat"][..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(frames, -1, 3, 3)
        objects = {}
        for i, saved_path in enumerate(motion["object_mesh_paths"]):
            if np.any(motion["object_active"][:, i] & enabled[:, i]):
                objects[i] = load_collision_mesh(object_mesh_path(str(saved_path), dataset), dataset)
        cache = {}
        depths = {name: np.zeros(frames) for name in ("ground", "object", "self")}
        toes = [np.zeros((frames, 3)), np.zeros((frames, 3))]
        for frame in range(frames):
            pin.forwardKinematics(self.pin_model, self.pin_data, q_pin[frame])
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            for toe, foot_id in zip(toes, self.foot_ids):
                pose = self.pin_data.oMf[foot_id]
                toe[frame] = pose.translation + pose.rotation @ FOOT_POINT
            self.data.qpos[:] = q_mj[frame]
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_collision(self.model, self.data)
            for contact in self.data.contact[:self.data.ncon]:
                a, b = int(contact.geom1), int(contact.geom2)
                group = "ground" if (a == self.ground and b in self.shapes) or (b == self.ground and a in self.shapes) else "self" if tuple(sorted((a, b))) in self.self_pairs else None
                if group and -contact.dist > depths[group][frame]:
                    depths[group][frame] = -float(contact.dist)
            geom_rotations = self.data.geom_xmat[self.geoms].reshape(-1, 3, 3)
            centers = self.data.geom_xpos[self.geoms] + np.einsum("nij,nj->ni", geom_rotations, self.centers)
            extents = np.einsum("nij,nj->ni", np.abs(geom_rotations), self.extents)
            for i, mesh in objects.items():
                if not (motion["object_active"][frame, i] and enabled[frame, i]):
                    continue
                scale = float(motion["object_demo_scale"][frame, i])
                key = (i, scale)
                if key not in cache:
                    vertices = np.ascontiguousarray(mesh.vertices * scale, dtype=float)
                    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
                    bvh = hppfcl.BVHModelOBBRSS()
                    bvh.beginModel(len(faces), len(vertices))
                    bvh.addVertices(vertices)
                    bvh.addTriangles(faces)
                    if bvh.endModel() != 0:
                        raise RuntimeError("Failed to build the original object triangle BVH")
                    request = hppfcl.CollisionRequest()
                    request.enable_contact = True
                    request.security_margin = DETECTION_DISTANCE_M
                    request.gjk_tolerance = request.epa_tolerance = QUERY_TOLERANCE_M
                    request.num_max_contacts = len(faces)
                    cache[key] = bvh, request
                bvh, request = cache[key]
                rotation, position = rotations[frame, i], motion["object_pos"][frame, i]
                obj_center = position + rotation @ (mesh.bounds.mean(axis=0) * scale)
                obj_extent = np.abs(rotation) @ ((mesh.bounds[1] - mesh.bounds[0]) * scale * 0.5)
                separation = np.maximum(np.maximum(obj_center - obj_extent - (centers + extents), centers - extents - (obj_center + obj_extent)), 0.0)
                candidates = self.geoms[np.linalg.norm(separation, axis=1) < DETECTION_DISTANCE_M]
                obj_transform = hppfcl.Transform3f(rotation, position)
                for geom in candidates:
                    self.robot_transform.setTransform(self.data.geom_xmat[geom].reshape(3, 3), self.data.geom_xpos[geom])
                    self.result.clear()
                    hppfcl.collide(self.shapes[int(geom)], self.robot_transform, bvh, obj_transform, request, self.result)
                    contacts = self.result.getContacts()
                    distances = np.array([contact.penetration_depth for contact in contacts])
                    faces = np.array([contact.b2 for contact in contacts], dtype=int)
                    if not np.all(np.isfinite(distances)) or np.any(distances == -np.finfo(float).max) or np.any((faces < 0) | (faces >= len(mesh.faces))):
                        raise FloatingPointError(f"Invalid collision distance at frame {frame}, geom {geom}, object {i}")
                    minimum = float(np.min(distances)) if distances.size else DETECTION_DISTANCE_M
                    if -minimum > depths["object"][frame]:
                        depths["object"][frame] = -minimum
        return depths, toes


SOURCE_FOOT_STATE_CONTRACT = "foot_state_utils.raw_source_foot_trajectory_v1"
SOURCE_SOLE_STICK_SCHEDULE_CONTRACT = "foot_state_utils.sole_stick_episode_v2"
SOURCE_INTERACTION_HAND_METRIC_CONTRACT = "source_true_hand_interaction_frames_v1"
FOOT_SPEED_THRESHOLD_M_PER_S = 0.3
PENETRATION_COMPONENT_TOLERANCES_M = {"ground": 0.01, "object": 0.01, "self": 0.01}
METRIC_KEYS = ("penetration_duration", "penetration_max_depth_cm", "footskating_duration", "footskating_max_velocity_m_per_s", "primary_finger_distance_mm", "secondary_finger_distance_mm", "palm_orientation_deg")
AGGREGATION_KEYS = ("penetration_frame_count", "penetration_depth_sum_cm", "footskating_frame_count", "footskating_speed_sum_m_per_s")
SIDES = ("L", "R")


def validate_sole_stick_episode_tracks(
    active, episode, frame_count: int, *, side: str
) -> tuple[np.ndarray, np.ndarray]:
    active_raw = np.asarray(active)
    episode_raw = np.asarray(episode)
    if active_raw.dtype.kind != "b" or episode_raw.dtype.kind not in "iu":
        raise ValueError(
            f"{side}-foot sole-stick active/episode tracks must use bool/integer dtypes"
        )
    active_arr = np.asarray(active_raw, dtype=bool).reshape(-1)
    episode_arr = np.asarray(episode_raw, dtype=np.int64).reshape(-1)
    expected_count = int(frame_count)
    if active_arr.shape != (expected_count,) or episode_arr.shape != (expected_count,):
        raise ValueError(
            f"{side}-foot sole-stick tracks must have shape {(expected_count,)}, got {active_arr.shape} and {episode_arr.shape}"
        )
    expected_episode = np.full((expected_count,), -1, dtype=np.int64)
    episode_id = -1
    for frame_index in np.flatnonzero(active_arr):
        if frame_index == 0 or not active_arr[frame_index - 1]:
            episode_id += 1
        expected_episode[frame_index] = episode_id
    if not np.array_equal(episode_arr, expected_episode):
        raise ValueError(
            f"{side}-foot sole-stick episode ids are not the contiguous episodes implied by the active track"
        )
    return active_arr.copy(), episode_arr.copy()


def _xy_speed_m_per_s(positions: np.ndarray, dt: float, first_value: float) -> np.ndarray:
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Foot point trajectory must have shape (T, >=2), got {points.shape}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Foot speed requires a positive finite dt, got {dt}")
    if points.shape[0] <= 1:
        return np.full((points.shape[0],), float(first_value), dtype=float)
    return np.concatenate(
        [[float(first_value)], np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1) / float(dt)]
    )


def _same_contact_episode_steps(active: np.ndarray, episode: np.ndarray) -> np.ndarray:
    active_arr = np.asarray(active, dtype=bool).reshape(-1)
    episode_arr = np.asarray(episode, dtype=int).reshape(-1)
    if active_arr.shape != episode_arr.shape:
        raise ValueError(
            f"Contact active and episode arrays must share a shape, got {active_arr.shape} and {episode_arr.shape}"
        )
    valid = np.zeros(active_arr.shape, dtype=bool)
    if active_arr.size > 1:
        valid[1:] = active_arr[1:] & active_arr[:-1] & (episode_arr[1:] == episode_arr[:-1])
    return valid


def _foot_skating_metrics(
    source_l_contact: np.ndarray,
    source_r_contact: np.ndarray,
    source_l_episode: np.ndarray,
    source_r_episode: np.ndarray,
    robot_l_toe: np.ndarray,
    robot_r_toe: np.ndarray,
    dt: float,
) -> tuple[float | None, np.ndarray]:
    robot_l_speed = _xy_speed_m_per_s(robot_l_toe, dt, 0.0)
    robot_r_speed = _xy_speed_m_per_s(robot_r_toe, dt, 0.0)
    source_l_sticking = _same_contact_episode_steps(source_l_contact, source_l_episode)
    source_r_sticking = _same_contact_episode_steps(source_r_contact, source_r_episode)
    threshold = FOOT_SPEED_THRESHOLD_M_PER_S
    l_sliding = source_l_sticking & (robot_l_speed > threshold)
    r_sliding = source_r_sticking & (robot_r_speed > threshold)
    source_sticking_count = int(np.sum(source_l_sticking | source_r_sticking))
    if source_sticking_count <= 0:
        return None, np.empty((0,), dtype=float)
    values = np.maximum(robot_l_speed * l_sliding, robot_r_speed * r_sliding)
    values = values[values > 0.0]
    return float(values.size / source_sticking_count), values


def _aggregate_penetration_depths(
    component_depths: dict[str, np.ndarray],
    *,
    component_tolerances_m: dict[str, float] = PENETRATION_COMPONENT_TOLERANCES_M,
) -> tuple[float, np.ndarray]:
    component_names = ("ground", "object", "self")
    missing = [name for name in component_names if name not in component_depths]
    if missing:
        raise ValueError(f"Penetration components are missing required tracks: {missing}")
    if set(component_tolerances_m) != set(component_names):
        raise ValueError(f"Penetration component tolerances must define exactly {component_names}")
    tolerances = np.asarray(
        [float(component_tolerances_m[name]) for name in component_names], dtype=float
    )
    if not np.all(np.isfinite(tolerances)) or np.any(tolerances <= 0.0):
        raise ValueError("Penetration component tolerances must be finite and positive")
    components = [
        np.asarray(component_depths[name], dtype=float).reshape(-1) for name in component_names
    ]
    if components[0].size <= 0 or len({values.shape for values in components}) != 1:
        raise ValueError(
            "Ground, object, and self-collision penetration tracks must share a non-empty shape"
        )
    if any(not np.all(np.isfinite(values)) or np.any(values < 0.0) for values in components):
        raise ValueError("Penetration depth tracks must contain finite non-negative values")
    frame_depths = np.maximum.reduce(components)
    violating = np.logical_or.reduce(
        [values > tolerance for values, tolerance in zip(components, tolerances)]
    )
    penetrating_depths = frame_depths[violating]
    return float(penetrating_depths.size / frame_depths.size), penetrating_depths


def _scalar(data: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got shape {value.shape}")
    return value.reshape(-1)[0].item()


def read_interaction_hand_metrics(
    debug_path: Path
) -> tuple[dict[str, float | None], int]:
    prefix = "final_target"
    stems = {
        "primary_finger_distance_mm": "primary_tip_mean_mm",
        "secondary_finger_distance_mm": "secondary_tip_mean_mm",
        "palm_orientation_deg": "palm_normal_deg",
    }
    per_side = {key: {} for key in stems}
    frame_counts = set()
    with np.load(debug_path, allow_pickle=False) as debug:
        if (
            _scalar(debug, f"{prefix}_hand_metric_contract")
            != SOURCE_INTERACTION_HAND_METRIC_CONTRACT
        ):
            raise ValueError(f"Expected original-source interaction hand metrics in {debug_path}")
        for side in SIDES:
            active = np.asarray(debug[f"{prefix}_frame_hand_interaction_active_{side}"])
            if active.ndim != 1 or not active.size or active.dtype != np.bool_:
                raise ValueError(
                    f"{side} hand interaction mask must be a nonempty boolean frame array"
                )
            frame_counts.add(active.size)
            for key, stem in stems.items():
                values = np.asarray(
                    debug[f"{prefix}_frame_{stem}_source_target_{side}"], dtype=float
                )
                if (
                    values.shape != active.shape
                    or not np.all(np.isfinite(values[active]))
                    or np.any(values[active] < 0.0)
                ):
                    raise ValueError(
                        f"Invalid or missing {side} {key} values on interaction frames in {debug_path}"
                    )
                if key == "palm_orientation_deg" and np.any(values[active] > 180.0):
                    raise ValueError(f"{side} palm-normal error exceeds 180 degrees")
                per_side[key][side] = float(np.mean(values[active])) if np.any(active) else None
    if len(frame_counts) != 1:
        raise ValueError("Left and right hand metrics have different frame counts")
    metrics = {
        key: (
            float(np.mean(available))
            if (available := [value for value in sides.values() if value is not None])
            else None
        )
        for key, sides in per_side.items()
    }
    return metrics, int(next(iter(frame_counts)))


def load_metric_inputs(entry, motion):
    frames = entry["frames"]
    if str(motion["source_foot_state_contract"]) != SOURCE_FOOT_STATE_CONTRACT or str(motion["source_foot_sole_stick_contract"]) != SOURCE_SOLE_STICK_SCHEDULE_CONTRACT:
        raise ValueError("The saved foot-state contract is not supported by this evaluator")
    left, right = [validate_sole_stick_episode_tracks(motion[f"source_{side}_foot_sole_stick_active"], motion[f"source_{side}_foot_sole_stick_episode_id"], frames, side=side) for side in ("l", "r")]
    path = package_path(entry["debug"])
    with np.load(path, allow_pickle=False) as saved:
        for key in ("source_dataset", "source_seq_name", "retime_source_indices", "frame_start", "frame_cap", "retimed_frame_count", "dt"):
            if not np.array_equal(saved[key], motion[key]):
                raise ValueError(f"opt/val mismatch: {key}")
        # The training opt serializes transforms as float32; val retains float64.
        for key in ("world_canonical_heading_transform", "solver_world_from_source"):
            if not np.array_equal(saved[key].astype(motion[key].dtype), motion[key]):
                raise ValueError(f"opt/val coordinate transform mismatch: {key}")
    hand, hand_frames = read_interaction_hand_metrics(path)
    if hand_frames != frames:
        raise ValueError("Hand diagnostics do not cover the complete trajectory")
    return (left[0], right[0], left[1], right[1]), hand


def compute_metrics(depths, toes, schedule, hand, dataset, dt):
    tolerances = {**PENETRATION_COMPONENT_TOLERANCES_M, "object": 0.02 if dataset == "HUMOTO" else 0.01}
    penetration, depths_m = _aggregate_penetration_depths(depths, component_tolerances_m=tolerances)
    skating, speeds = _foot_skating_metrics(*schedule, *toes, dt)
    metrics = {
        "penetration_duration": penetration,
        "penetration_max_depth_cm": float(np.max(depths_m) * 100.0) if depths_m.size else 0.0,
        "footskating_duration": skating,
        "footskating_max_velocity_m_per_s": float(np.mean(speeds)) if speeds.size else (0.0 if skating is not None else None),
        **hand,
    }
    # Counts and sums preserve frame-pooled means during resume without storing per-frame arrays.
    aggregation = {"penetration_frame_count": int(depths_m.size), "penetration_depth_sum_cm": float(np.sum(depths_m * 100.0)), "footskating_frame_count": int(speeds.size), "footskating_speed_sum_m_per_s": float(np.sum(speeds))}
    return {"metrics": metrics, "aggregation": aggregation}


def aggregate_metrics(payloads):
    metrics = {}
    for key in METRIC_KEYS:
        if key == "penetration_max_depth_cm":
            count = sum(p["aggregation"]["penetration_frame_count"] for p in payloads)
            total = sum(p["aggregation"]["penetration_depth_sum_cm"] for p in payloads)
            metrics[key] = total / count if count else (0.0 if payloads else None)
        elif key == "footskating_max_velocity_m_per_s":
            applicable = [p for p in payloads if p["metrics"]["footskating_duration"] is not None]
            count = sum(p["aggregation"]["footskating_frame_count"] for p in applicable)
            total = sum(p["aggregation"]["footskating_speed_sum_m_per_s"] for p in applicable)
            metrics[key] = total / count if count else (0.0 if applicable else None)
        else:
            values = [p["metrics"][key] for p in payloads if p["metrics"][key] is not None]
            metrics[key] = float(np.mean(values)) if values else None
        if metrics[key] is not None and not np.isfinite(metrics[key]):
            raise ValueError(f"Non-finite summary values: {key}")
    return metrics


GEOMETRY = None


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate_one(entry, output, resume):
    global GEOMETRY
    motion = load_motion(entry)
    schedule, hand = load_metric_inputs(entry, motion)
    target = output / entry["dataset"] / "metrics" / f"{entry['sequence']}.json"
    identity = {"dataset": entry["dataset"], "sequence": entry["sequence"], "frame_count": entry["frames"]}
    if target.exists():
        saved = json.loads(target.read_text(encoding="utf-8"))
        if not resume or any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Existing result does not match this sequence; choose a new output directory: {target}")
        if set(saved) != set(identity) | {"metrics", "aggregation"} or set(saved["metrics"]) != set(METRIC_KEYS) or set(saved["aggregation"]) != set(AGGREGATION_KEYS):
            raise ValueError(f"Existing result has missing or unexpected fields: {target}")
        return saved, "reused"
    if GEOMETRY is None:
        GEOMETRY = GeometryEvaluator()
    depths, toes = GEOMETRY.evaluate(motion, entry["dataset"])
    payload = {**identity, **compute_metrics(depths, toes, schedule, hand, entry["dataset"], float(motion["dt"]))}
    write_json(target, payload)
    return payload, "saved"


def write_message(message, log):
    print(message, flush=True)
    print(message, file=log, flush=True)


def format_summary(summary):
    rows = (
        ("Primary finger distance", "mm", "primary_finger_distance_mm"),
        ("Secondary finger distance", "mm", "secondary_finger_distance_mm"),
        ("Palm orientation", "deg", "palm_orientation_deg"),
        ("Penetration duration", "ratio", "penetration_duration"),
        ("Penetration depth (pooled)", "cm", "penetration_max_depth_cm"),
        ("Foot skating duration", "ratio", "footskating_duration"),
        ("Foot speed (pooled)", "m/s", "footskating_max_velocity_m_per_s"),
    )
    lines = [
        f"\n{summary['dataset']}  |  Sequences: {summary['evaluated_sequence_count']}/{summary['requested_sequence_count']}  |  Frames: {summary['evaluated_frame_count']:,}",
        f"┌{'─' * 29}┬{'─' * 7}┬{'─' * 14}┐",
        f"│ {'Metric':<27} │ {'Unit':<5} │ {'Value':>12} │",
        f"├{'─' * 29}┼{'─' * 7}┼{'─' * 14}┤",
    ]
    for label, unit, key in rows:
        value = "N/A" if summary["metrics"][key] is None else f"{summary['metrics'][key]:.6f}"
        lines.append(f"│ {label:<27} │ {unit:<5} │ {value:>12} │")
    lines.append(f"└{'─' * 29}┴{'─' * 7}┴{'─' * 14}┘")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("HUMOTO", "GRAB"), help="Omit to evaluate both datasets")
    parser.add_argument("--sequence", action="append", help="Exact sequence ID; repeat to select multiple sequences from --dataset")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent sequence processes (default: 1)")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluatioin", help="Output directory (default: <package>/evaluatioin)")
    parser.add_argument("--resume", action="store_true", help="Reuse complete results for matching sequences; use a new output after changing inputs or code")
    
    args = parser.parse_args()
    if args.jobs < 1 or (args.sequence and not args.dataset):
        parser.error("--jobs must be positive; --sequence requires --dataset")
    
    catalog = scan_sequences()
    selected = [e for e in catalog if (not args.dataset or e["dataset"] == args.dataset) and (not args.sequence or e["sequence"] in args.sequence)]
    if not selected or (args.sequence and set(args.sequence) != {e["sequence"] for e in selected}):
        parser.error("No matching selection, or one or more sequence IDs were not found")
    
    output = args.output.expanduser().resolve()
    protected = [ROOT / name for name in ("data", "assets", "resources", "scripts", "licenses")]
    if output == ROOT or output.is_file() or any(output == path or path in output.parents for path in protected):
        parser.error("--output must be a result directory, not the package root or an input directory")
    
    targets = [output / e["dataset"] / "metrics" / f"{e['sequence']}.json" for e in selected]
    if not args.resume and any(path.exists() for path in targets):
        parser.error("Some selected outputs already exist; use --resume or choose a new output directory")
    
    write_json(output / "run.json", {"jobs": args.jobs, "selection": [{"dataset": e["dataset"], "sequence": e["sequence"], "frames": e["frames"]} for e in selected]})
    by_dataset = {dataset: [e for e in selected if e["dataset"] == dataset] for dataset in sorted({e["dataset"] for e in selected})}
    completed, failures = [], []
    with ExitStack() as stack:
        logs, progress = {}, {dataset: 0 for dataset in by_dataset}
        for dataset, entries in by_dataset.items():
            path = output / dataset / "result.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            logs[dataset] = stack.enter_context(path.open("w", encoding="utf-8"))
            print("$ " + shlex.join([sys.executable, *sys.argv]) + "\n", file=logs[dataset], flush=True)
            write_message(f"Evaluating {dataset}: {len(entries)} complete sequences / {sum(e['frames'] for e in entries)} frames; jobs={args.jobs}", logs[dataset])
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(evaluate_one, entry, output, args.resume): entry for entry in selected}
            for future in as_completed(futures):
                entry = futures[future]
                dataset = entry["dataset"]
                progress[dataset] += 1
                prefix = f"[{progress[dataset]}/{len(by_dataset[dataset])}]"
                try:
                    payload, status = future.result()
                except Exception as error:
                    failures.append({"dataset": dataset, "sequence": entry["sequence"], "error": f"{type(error).__name__}: {error}"})
                    message = f"{prefix} FAILED {dataset}/{entry['sequence']}: {error}"
                else:
                    completed.append(payload)
                    message = f"{prefix} {status} {dataset}/{entry['sequence']} ({entry['frames']} frames)"
                write_message(message, logs[dataset])
        for dataset, expected in by_dataset.items():
            payloads = sorted((p for p in completed if p["dataset"] == dataset), key=lambda p: p["sequence"])
            summary = {"dataset": dataset, "requested_sequence_count": len(expected), "evaluated_sequence_count": len(payloads), "evaluated_frame_count": sum(p["frame_count"] for p in payloads), "metrics": aggregate_metrics(payloads), "failures": [{"sequence": f["sequence"], "error": f["error"]} for f in failures if f["dataset"] == dataset]}
            write_json(output / dataset / "summary.json", summary)
            write_message(format_summary(summary), logs[dataset])
            write_message(f"Completed {len(payloads)}/{len(expected)} {dataset} sequences. Results: {output / dataset}", logs[dataset])
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
