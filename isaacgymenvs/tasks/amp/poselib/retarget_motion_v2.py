import argparse
import json
import os

import torch

from poselib.core.rotation3d import (
    quat_from_angle_axis,
    quat_identity,
    quat_inverse,
    quat_mul_norm,
    quat_normalize,
)
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from poselib.visualization.common import plot_skeleton_motion_interactive, plot_skeleton_state


def _resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def _quat_to_hinge_xyz_chain(q):
    """Decompose Qx(qx) * Qy(qy) * Qz(qz) into nested hinge angles."""
    q = quat_normalize(q)
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]

    t0 = 2.0 * (w * x - y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    ex = torch.atan2(t0, t1)

    t2 = 2.0 * (x * z + w * y)
    t2 = torch.clamp(t2, -1.0, 1.0)
    ey = torch.asin(t2)

    t3 = 2.0 * (w * z - x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    ez = torch.atan2(t3, t4)

    return torch.stack(
        [_normalize_angle(ex), _normalize_angle(ey), _normalize_angle(ez)], dim=-1
    )


def _quat_to_single_axis_angle(q, axis_idx):
    """Extract a signed pure-axis hinge angle without acos near zero."""
    q = quat_normalize(q)
    signed = 2.0 * torch.atan2(q[..., axis_idx], q[..., 3])
    return _normalize_angle(signed)


def _axis_tensor(device, dim, value):
    axis = torch.zeros((dim, 3), device=device, dtype=torch.float32)
    axis[:, value] = 1.0
    return axis


def _center_quaternion_sequence(q):
    """Remove a sequence's mean local orientation while preserving motion."""
    q = quat_normalize(q)
    mean_q = quat_normalize(torch.mean(q, dim=0, keepdim=True))
    centered_q = quat_mul_norm(quat_inverse(mean_q).expand_as(q), q)
    return centered_q, mean_q


def _format_angle_range_deg(q):
    xyz_deg = torch.rad2deg(_quat_to_hinge_xyz_chain(q))
    return {
        "min": xyz_deg.min(dim=0).values.detach().cpu().numpy(),
        "mean": xyz_deg.mean(dim=0).detach().cpu().numpy(),
        "max": xyz_deg.max(dim=0).values.detach().cpu().numpy(),
    }


def _build_target_local_rotation(
    source_motion, target_tpose, center_neck_reference=False
):
    src_tree = source_motion.skeleton_tree
    tgt_tree = target_tpose.skeleton_tree

    src_ids = src_tree._node_indices
    tgt_ids = tgt_tree._node_indices
    src_rot = source_motion.local_rotation
    device = src_rot.device
    num_frames = src_rot.shape[0]
    num_tgt_nodes = len(tgt_tree.node_names)

    required_source_nodes = [
        "pelvis",
        "torso",
        "head",
        "right_upper_arm",
        "right_lower_arm",
        "right_hand",
        "left_upper_arm",
        "left_lower_arm",
        "left_hand",
        "right_thigh",
        "right_shin",
        "right_foot",
        "left_thigh",
        "left_shin",
        "left_foot",
    ]
    required_target_nodes = [
        "pelvis",
        "abdomen_x_link",
        "abdomen_y_link",
        "torso",
        "neck_x_link",
        "neck_y_link",
        "head",
        "right_shoulder_x_link",
        "right_shoulder_y_link",
        "right_upper_arm",
        "right_lower_arm",
        "right_hand",
        "left_shoulder_x_link",
        "left_shoulder_y_link",
        "left_upper_arm",
        "left_lower_arm",
        "left_hand",
        "right_hip_x_link",
        "right_hip_y_link",
        "right_thigh",
        "right_shin",
        "right_ankle_x_link",
        "right_ankle_y_link",
        "right_foot",
        "left_hip_x_link",
        "left_hip_y_link",
        "left_thigh",
        "left_shin",
        "left_ankle_x_link",
        "left_ankle_y_link",
        "left_foot",
    ]
    missing_source = [name for name in required_source_nodes if name not in src_ids]
    missing_target = [name for name in required_target_nodes if name not in tgt_ids]
    if missing_source:
        raise KeyError(f"Source motion is missing expected nodes: {missing_source}")
    if missing_target:
        raise KeyError(f"Target T-pose is missing expected nodes: {missing_target}")

    target_rot = quat_identity([num_frames, num_tgt_nodes]).to(device)

    shared_copy_names = ["pelvis", "right_hand", "left_hand"]
    for name in shared_copy_names:
        target_rot[:, tgt_ids[name], :] = src_rot[:, src_ids[name], :]

    y_axis = _axis_tensor(device, num_frames, 1)

    right_elbow_angle = _quat_to_single_axis_angle(src_rot[:, src_ids["right_lower_arm"], :], 1)
    left_elbow_angle = _quat_to_single_axis_angle(src_rot[:, src_ids["left_lower_arm"], :], 1)
    right_knee_angle = _quat_to_single_axis_angle(src_rot[:, src_ids["right_shin"], :], 1)
    left_knee_angle = _quat_to_single_axis_angle(src_rot[:, src_ids["left_shin"], :], 1)

    target_rot[:, tgt_ids["right_lower_arm"], :] = quat_from_angle_axis(right_elbow_angle, y_axis)
    target_rot[:, tgt_ids["left_lower_arm"], :] = quat_from_angle_axis(left_elbow_angle, y_axis)
    target_rot[:, tgt_ids["right_shin"], :] = quat_from_angle_axis(right_knee_angle, y_axis)
    target_rot[:, tgt_ids["left_shin"], :] = quat_from_angle_axis(left_knee_angle, y_axis)

    abdomen_xyz = _quat_to_hinge_xyz_chain(src_rot[:, src_ids["torso"], :])
    source_neck_q = src_rot[:, src_ids["head"], :]
    if center_neck_reference:
        before = _format_angle_range_deg(source_neck_q)
        source_neck_q, neck_mean_q = _center_quaternion_sequence(source_neck_q)
        after = _format_angle_range_deg(source_neck_q)
        print("[neck compensation] removed mean local quaternion:", neck_mean_q[0].cpu().numpy())
        print("[neck compensation] XYZ deg before:", before)
        print("[neck compensation] XYZ deg after: ", after)
    neck_xyz = _quat_to_hinge_xyz_chain(source_neck_q)
    right_shoulder_xyz = _quat_to_hinge_xyz_chain(src_rot[:, src_ids["right_upper_arm"], :])
    left_shoulder_xyz = _quat_to_hinge_xyz_chain(src_rot[:, src_ids["left_upper_arm"], :])
    right_hip_xyz = _quat_to_hinge_xyz_chain(src_rot[:, src_ids["right_thigh"], :])
    left_hip_xyz = _quat_to_hinge_xyz_chain(src_rot[:, src_ids["left_thigh"], :])
    right_ankle_xyz = _quat_to_hinge_xyz_chain(src_rot[:, src_ids["right_foot"], :])
    left_ankle_xyz = _quat_to_hinge_xyz_chain(src_rot[:, src_ids["left_foot"], :])

    x_axis = _axis_tensor(device, num_frames, 0)
    z_axis = _axis_tensor(device, num_frames, 2)

    target_rot[:, tgt_ids["abdomen_x_link"], :] = quat_from_angle_axis(abdomen_xyz[:, 0], x_axis)
    target_rot[:, tgt_ids["abdomen_y_link"], :] = quat_from_angle_axis(abdomen_xyz[:, 1], y_axis)
    target_rot[:, tgt_ids["torso"], :] = quat_from_angle_axis(abdomen_xyz[:, 2], z_axis)

    target_rot[:, tgt_ids["neck_x_link"], :] = quat_from_angle_axis(neck_xyz[:, 0], x_axis)
    target_rot[:, tgt_ids["neck_y_link"], :] = quat_from_angle_axis(neck_xyz[:, 1], y_axis)
    target_rot[:, tgt_ids["head"], :] = quat_from_angle_axis(neck_xyz[:, 2], z_axis)

    target_rot[:, tgt_ids["right_shoulder_x_link"], :] = quat_from_angle_axis(right_shoulder_xyz[:, 0], x_axis)
    target_rot[:, tgt_ids["right_shoulder_y_link"], :] = quat_from_angle_axis(right_shoulder_xyz[:, 1], y_axis)
    target_rot[:, tgt_ids["right_upper_arm"], :] = quat_from_angle_axis(right_shoulder_xyz[:, 2], z_axis)

    target_rot[:, tgt_ids["left_shoulder_x_link"], :] = quat_from_angle_axis(left_shoulder_xyz[:, 0], x_axis)
    target_rot[:, tgt_ids["left_shoulder_y_link"], :] = quat_from_angle_axis(left_shoulder_xyz[:, 1], y_axis)
    target_rot[:, tgt_ids["left_upper_arm"], :] = quat_from_angle_axis(left_shoulder_xyz[:, 2], z_axis)

    target_rot[:, tgt_ids["right_hip_x_link"], :] = quat_from_angle_axis(right_hip_xyz[:, 0], x_axis)
    target_rot[:, tgt_ids["right_hip_y_link"], :] = quat_from_angle_axis(right_hip_xyz[:, 1], y_axis)
    target_rot[:, tgt_ids["right_thigh"], :] = quat_from_angle_axis(right_hip_xyz[:, 2], z_axis)

    target_rot[:, tgt_ids["left_hip_x_link"], :] = quat_from_angle_axis(left_hip_xyz[:, 0], x_axis)
    target_rot[:, tgt_ids["left_hip_y_link"], :] = quat_from_angle_axis(left_hip_xyz[:, 1], y_axis)
    target_rot[:, tgt_ids["left_thigh"], :] = quat_from_angle_axis(left_hip_xyz[:, 2], z_axis)

    target_rot[:, tgt_ids["right_ankle_x_link"], :] = quat_from_angle_axis(right_ankle_xyz[:, 0], x_axis)
    target_rot[:, tgt_ids["right_ankle_y_link"], :] = quat_from_angle_axis(right_ankle_xyz[:, 1], y_axis)
    target_rot[:, tgt_ids["right_foot"], :] = quat_from_angle_axis(right_ankle_xyz[:, 2], z_axis)

    target_rot[:, tgt_ids["left_ankle_x_link"], :] = quat_from_angle_axis(left_ankle_xyz[:, 0], x_axis)
    target_rot[:, tgt_ids["left_ankle_y_link"], :] = quat_from_angle_axis(left_ankle_xyz[:, 1], y_axis)
    target_rot[:, tgt_ids["left_foot"], :] = quat_from_angle_axis(left_ankle_xyz[:, 2], z_axis)

    return target_rot


def _trim_motion(local_rotation, root_translation, frame_beg, frame_end):
    if frame_beg == -1:
        frame_beg = 0
    if frame_end == -1:
        frame_end = local_rotation.shape[0]

    local_rotation = local_rotation[frame_beg:frame_end, ...]
    root_translation = root_translation[frame_beg:frame_end, ...]
    return local_rotation, root_translation


def _ground_motion(target_tree, local_rotation, root_translation, root_height_offset):
    temp_state = SkeletonState.from_rotation_and_root_translation(
        target_tree,
        local_rotation,
        root_translation,
        is_local=True,
    )
    global_translation = temp_state.global_translation
    min_h = torch.min(global_translation[..., 2])
    root_translation = root_translation.clone()
    root_translation[:, 2] += -min_h
    root_translation[:, 2] += root_height_offset
    return root_translation


def run_retarget(config_path: str, visualize: bool = False) -> None:
    config_abs_path = os.path.abspath(config_path)
    with open(config_abs_path, "r") as f:
        cfg = json.load(f)

    config_dir = os.path.dirname(config_abs_path)
    source_motion_path = _resolve_path(config_dir, cfg["source_motion"])
    target_motion_path = _resolve_path(config_dir, cfg["target_motion_path"])
    target_tpose_path = _resolve_path(config_dir, cfg["target_tpose"])

    os.makedirs(os.path.dirname(target_motion_path), exist_ok=True)

    source_motion = SkeletonMotion.from_file(source_motion_path)
    target_tpose = SkeletonState.from_file(target_tpose_path)

    if visualize:
        plot_skeleton_state(target_tpose)
        plot_skeleton_motion_interactive(source_motion)

    target_local_rotation = _build_target_local_rotation(
        source_motion,
        target_tpose,
        center_neck_reference=cfg.get("center_neck_reference", False),
    )
    root_translation = source_motion.root_translation.clone()

    target_local_rotation, root_translation = _trim_motion(
        target_local_rotation,
        root_translation,
        cfg.get("trim_frame_beg", 0),
        cfg.get("trim_frame_end", -1),
    )

    root_translation = _ground_motion(
        target_tpose.skeleton_tree,
        target_local_rotation,
        root_translation,
        cfg.get("root_height_offset", 0.0),
    )

    target_state = SkeletonState.from_rotation_and_root_translation(
        target_tpose.skeleton_tree,
        target_local_rotation,
        root_translation,
        is_local=True,
    )
    target_motion = SkeletonMotion.from_skeleton_state(target_state, fps=source_motion.fps)
    target_motion.to_file(target_motion_path)

    print(f"Saved v2 retargeted motion to: {target_motion_path}")
    print(f"Source motion: {source_motion_path}")
    print(f"Target tpose: {target_tpose_path}")
    print(f"Frames: {target_motion.local_rotation.shape[0]}")
    print(f"Target skeleton nodes: {len(target_motion.skeleton_tree.node_names)}")

    if visualize:
        plot_skeleton_motion_interactive(target_motion)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert old AMP humanoid motion into the hinge-chain v2 skeleton motion."
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to the v2 motion retarget config json.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_retarget(config_path=args.config, visualize=args.visualize)
