"""Retarget this measured X/Z/Y-channel BVH to the 15-body AMP humanoid.

Run with --help. Geometry is scaled to the target leg length; this is a
kinematic reference, not a reconstruction of contact forces or muscle activity.
"""
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp
from motion_tools.amp_motion import (MOTIONS, MJCF, load_motion, make_motion,
                                    forward_kinematics)


def read_bvh(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    hierarchy, motion = text.split("MOTION", 1)
    tokens = re.findall(r"[{}]|[^\s{}]+", hierarchy)
    nodes, cursor, channel_count = [], 1, 0

    def parse_node(parent):
        nonlocal cursor, channel_count
        kind = tokens[cursor]
        cursor += 1
        if kind == "End":
            if tokens[cursor] != "Site":
                raise ValueError("Malformed BVH End Site")
            cursor += 1
            name = nodes[parent]["name"] + f"_end_{len(nodes)}"
        else:
            name = tokens[cursor]
            cursor += 1
        index = len(nodes)
        nodes.append({"name": name, "parent": parent, "channels": [], "start": channel_count})
        if tokens[cursor] != "{":
            raise ValueError("Expected BVH opening brace")
        cursor += 1
        while tokens[cursor] != "}":
            token = tokens[cursor]
            if token == "OFFSET":
                nodes[index]["offset"] = np.array(tokens[cursor + 1:cursor + 4], dtype=float)
                cursor += 4
            elif token == "CHANNELS":
                count = int(tokens[cursor + 1])
                nodes[index]["channels"] = tokens[cursor + 2:cursor + 2 + count]
                channel_count += count
                cursor += 2 + count
            elif token in ("JOINT", "End"):
                parse_node(index)
            else:
                raise ValueError(f"Unexpected BVH token {token}")
        cursor += 1
    parse_node(-1)
    lines = motion.strip().splitlines()
    frames = int(lines[0].split(":")[1])
    dt = float(lines[1].split(":")[1])
    data = np.fromstring(" ".join(lines[2:]), sep=" ")
    if data.size != frames * channel_count:
        raise ValueError("BVH frame/channel count mismatch")
    return nodes, data.reshape(frames, channel_count), dt


def bvh_global(nodes, data):
    count = len(data)
    matrices = np.empty((count, len(nodes), 3, 3))
    positions = np.empty((count, len(nodes), 3))
    for i, node in enumerate(nodes):
        r = np.tile(np.eye(3), (count, 1, 1))
        p = np.tile(node["offset"], (count, 1))
        for k, channel in enumerate(node["channels"]):
            values = data[:, node["start"] + k]
            axis = "XYZ".index(channel[0].upper())
            if channel.lower().endswith("rotation"):
                v = np.zeros((count, 3))
                v[:, axis] = np.deg2rad(values)
                # BVH intrinsic rotations compose in the listed channel order.
                r = r @ Rotation.from_rotvec(v).as_matrix()
            elif channel.lower().endswith("position"):
                p[:, axis] += values
            else:
                raise ValueError(f"Unsupported channel {channel}")
        parent = node["parent"]
        if parent < 0:
            matrices[:, i], positions[:, i] = r, p
        else:
            matrices[:, i] = matrices[:, parent] @ r
            positions[:, i] = positions[:, parent] + np.einsum("fij,fj->fi", matrices[:, parent], p)
    return positions, matrices


def foot_minimum_height(positions, matrices, tree, xml):
    corners = []
    for name in ("right_foot", "left_foot"):
        index = list(tree["node_names"]).index(name)
        body = xml.find(f".//body[@name='{name}']")
        for geom in body.findall("geom"):
            if geom.get("type") != "box":
                continue
            center = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            half = np.fromstring(geom.get("size"), sep=" ")
            vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half + center
            xyz = np.einsum("fij,vj->fvi", matrices[:, index], vertices) + positions[:, index, None]
            corners.append(xyz[:, :, 2].min(axis=1))
    return np.min(corners, axis=0)


def retarget(nodes, data, dt, tree, xml, fps=60, floor_clearance=0.005):
    names = {n["name"]: i for i, n in enumerate(nodes)}
    mapping = ["Hips", "Spine3", "Head", "RightArm", "RightForeArm", "RightForeArm",
               "LeftArm", "LeftForeArm", "LeftForeArm", "RightUpLeg", "RightLeg", "RightFoot",
               "LeftUpLeg", "LeftLeg", "LeftFoot"]
    if list(tree["node_names"]) != ["pelvis", "torso", "head", "right_upper_arm", "right_lower_arm", "right_hand",
                                   "left_upper_arm", "left_lower_arm", "left_hand", "right_thigh", "right_shin",
                                   "right_foot", "left_thigh", "left_shin", "left_foot"]:
        raise ValueError("This mapping requires the repository's 15-body AMP skeleton")
    missing = set(mapping + ["LeftLeg", "LeftFoot"]) - names.keys()
    if missing:
        raise ValueError(f"Missing source BVH joints: {sorted(missing)}")
    source_pos, source_rot = bvh_global(nodes, data)
    basis = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    offsets = tree["local_translation"]["arr"]
    source_leg = sum(np.linalg.norm(nodes[names[n]]["offset"]) for n in ("LeftLeg", "LeftFoot"))
    target_leg = sum(np.linalg.norm(offsets[i]) for i in (13, 14))
    scale = target_leg / source_leg
    root = source_pos[:, names["Hips"]] @ basis.T * scale
    desired = np.empty((len(data), 15, 3, 3))
    for j, name in enumerate(mapping):
        alignment = np.eye(3)
        if j in (3, 4, 5, 6, 7, 8):
            alignment = Rotation.from_euler("x", -90 if j < 6 else 90, degrees=True).as_matrix()
        desired[:, j] = basis @ source_rot[:, names[name]] @ basis.T @ alignment
    # Remove the initial heading only; preserve turns measured in the recording.
    heading = np.arctan2(desired[0, 0, 1, 0], desired[0, 0, 0, 0])
    yaw = Rotation.from_euler("z", -heading).as_matrix()
    desired = yaw @ desired
    root = root @ yaw.T
    root[:, :2] -= root[0, :2]
    parents = tree["parent_indices"]["arr"]
    local = np.empty_like(desired)
    for j, parent in enumerate(parents):
        local[:, j] = desired[:, j] if parent < 0 else desired[:, parent].transpose(0, 2, 1) @ desired[:, j]
    # Project hinge joints onto Y, retaining the bend magnitude of the source.
    # A forearm Z channel cannot be treated as an AMP elbow Z action.
    for upper, lower, sign in ((3, 4, -1), (6, 7, -1), (9, 10, 1), (12, 13, 1)):
        bend = np.arccos(np.clip(local[:, lower, 2, 2], -1., 1.)) * sign
        hinge = Rotation.from_rotvec(np.column_stack([bend * 0, bend, bend * 0])).as_matrix()
        # Align the projected lower-bone direction to its measured direction;
        # the resulting upper twist is chosen nearest the original orientation.
        a, b = -hinge[:, :, 2], -local[:, lower, :, 2]
        cross = np.cross(a, b)
        norm = np.linalg.norm(cross, axis=1)
        angle = np.arctan2(norm, np.sum(a * b, axis=1))
        adjustment = Rotation.from_rotvec(cross / np.maximum(norm[:, None], 1e-12) * angle[:, None]).as_matrix()
        old_upper = local[:, upper].copy()
        local[:, upper] = old_upper @ adjustment
        local[:, lower] = hinge
        # Preserve the measured foot orientation after changing the shin frame.
        if lower in (10, 13):
            old_lower_global = desired[:, lower]
            new_lower_global = desired[:, parents[upper]] @ local[:, upper] @ hinge
            local[:, lower + 1] = new_lower_global.transpose(0, 2, 1) @ old_lower_global @ local[:, lower + 1]
    local[:, [5, 8]] = np.eye(3)  # The target hands are fixed bodies.
    source_times = np.arange(len(data)) * dt
    times = np.arange(int(np.floor(source_times[-1] * fps)) + 1) / fps
    rotation = np.empty((len(times), 15, 4))
    for j in range(15):
        rotation[:, j] = Slerp(source_times, Rotation.from_matrix(local[:, j]))(times).as_quat()
    root = np.column_stack([np.interp(times, source_times, root[:, axis]) for axis in range(3)])
    # MotionLib uses exponential-map components for spherical joints. Apply
    # the same convention to joint limits rather than clipping BVH Euler angles.
    clipped_counts = {}
    for j in range(1, 15):
        body = xml.find(f".//body[@name='{tree['node_names'][j]}']")
        joints = body.findall("joint")
        if not joints:
            continue
        v = Rotation.from_quat(rotation[:, j]).as_rotvec()
        before = v.copy()
        for joint in joints:
            axis = np.argmax(np.abs(np.fromstring(joint.get("axis"), sep=" ")))
            low, high = np.deg2rad(np.fromstring(joint.get("range"), sep=" "))
            v[:, axis] = np.clip(v[:, axis], low, high)
        changed = np.any(np.abs(v - before) > 1e-7, axis=1)
        clipped_counts[str(tree["node_names"][j])] = int(changed.sum())
        rotation[:, j] = Rotation.from_rotvec(v).as_quat()
    positions, matrices = forward_kinematics(rotation, root, tree)
    old_height = foot_minimum_height(positions, matrices, tree, xml)
    correction = floor_clearance - old_height
    root[:, 2] += correction
    speed = gaussian_filter1d(np.linalg.norm(np.gradient(root[:, :2], axis=0) * fps, axis=1), fps * .15)
    moving = speed > .25
    changes = np.diff(np.r_[False, moving, False].astype(int))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    if not len(starts):
        raise ValueError("No walking segment found above 0.25 m/s")
    longest = np.argmax(ends - starts)
    start, end = int(starts[longest]), int(ends[longest])
    info = {"source_frames": len(data), "source_fps": 1 / dt,
            "source_axes": "Z up, +X right, +Y forward (specific to this recording)",
            "target_axes": "Z up, +X forward, +Y left", "source_unit_to_target_meter": scale,
            "target_leg_length_m": target_leg, "fps": fps, "full_frames": len(times),
            "walking_frames": [start, end], "walking_seconds_in_source": [start / fps, (end - 1) / fps],
            "floor_clearance_m": floor_clearance,
            "root_height_correction_m": [float(correction.min()), float(correction.max())],
            "joint_limit_clipped_frames": clipped_counts,
            "notes": ["No measured muscle activations/contact forces are inferred.",
                      "Hands are fixed to the forearms; source toe rotations are discarded.",
                      "Per-frame root height makes the lowest foot box sit on a flat floor.",
                      "The walking clip includes acceleration/deceleration and is not a seamless gait loop."]}
    full = make_motion(rotation, root, tree, fps)
    cropped_root = root[start:end].copy()
    cropped_root[:, :2] -= cropped_root[0, :2]
    walk = make_motion(rotation[start:end], cropped_root, tree, fps)
    return full, walk, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", type=Path, default=MOTIONS / "P002_WALK_T01.bvh")
    parser.add_argument("--reference", type=Path, default=MOTIONS / "amp_humanoid_walk_175.npy")
    parser.add_argument("--model", type=Path, default=MJCF / "amp_humanoid_175_muscle.xml")
    parser.add_argument("--output-dir", type=Path, default=MOTIONS)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()
    if args.fps < 2:
        parser.error("--fps must be at least 2")
    nodes, data, dt = read_bvh(args.bvh)
    tree = load_motion(args.reference)["skeleton_tree"]
    # Avoid silently generating motion for a different morphology than the MJCF.
    xml = ET.parse(args.model).getroot()
    for name, offset in zip(tree["node_names"], tree["local_translation"]["arr"]):
        body = xml.find(f".//body[@name='{name}']")
        if body is None or not np.allclose(np.fromstring(body.get("pos", "0 0 0"), sep=" "), offset, atol=1e-6):
            raise ValueError(f"Reference skeleton and target MJCF differ at {name}")
    full, walk, info = retarget(nodes, data, dt, tree, xml, args.fps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, motion in (("amp", full), ("walk_amp", walk)):
        path = args.output_dir / f"{args.bvh.stem}_{suffix}.npy"
        np.save(path, motion)
        print(f"Saved {path}: {len(motion['rotation']['arr'])} frames @ {args.fps} Hz")
    info.update({"source": args.bvh.name, "reference": args.reference.name, "model": args.model.name})
    (args.output_dir / f"{args.bvh.stem}_retarget.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
