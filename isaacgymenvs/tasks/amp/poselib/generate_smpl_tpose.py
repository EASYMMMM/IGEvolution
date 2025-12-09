# Copyright (c) 2018-2025, NVIDIA Corporation and contributors
# SPDX-License-Identifier: BSD-3-Clause
#
# 该脚本用于从 MJCF 文件中导出 SMPL Humanoid 的 T-Pose（SkeletonState）。
# 默认会将手臂抬至近似 T 字姿势，并保存为 npy 文件，供动作重定向使用。

import argparse
import os
import torch

from poselib.core.rotation3d import quat_mul, quat_from_angle_axis
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState


def _normalize_path(path: str) -> str:
    """将相对路径转换为绝对路径（相对于当前脚本所在目录）。"""
    if os.path.isabs(path):
        return path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(root_dir, path))


def _rotate_joint(local_rotation: torch.Tensor,
                  skeleton: SkeletonTree,
                  joint_name: str,
                  axis,
                  angle_deg: float) -> None:
    """对指定关节施加额外的轴角旋转（角度单位：度）。"""
    if joint_name not in skeleton.node_names:
        raise ValueError(f"Joint '{joint_name}' not found in skeleton.")

    joint_id = skeleton.index(joint_name)
    base_rot = local_rotation[joint_id]
    delta_rot = quat_from_angle_axis(
        angle=torch.tensor([angle_deg], dtype=torch.float32),
        axis=torch.tensor(axis, dtype=torch.float32),
        degree=True,
    )
    local_rotation[joint_id] = quat_mul(delta_rot, base_rot)


def export_smpl_tpose(mjcf_path: str, output_path: str, lift_arms: bool = False) -> None:
    """从 MJCF 导出 SkeletonState，并保存为 npy 文件。"""
    mjcf_abs_path = _normalize_path(mjcf_path)
    output_abs_path = _normalize_path(output_path)

    skeleton = SkeletonTree.from_mjcf(mjcf_abs_path)
    tpose_state = SkeletonState.zero_pose(skeleton)

    local_rotation = tpose_state.local_rotation.clone()

    if lift_arms:
        # 将手臂旋转至类似 T Pose 的姿态
        _rotate_joint(local_rotation, skeleton, "L_Shoulder", axis=[1.0, 0.0, 0.0], angle_deg=90.0)
        _rotate_joint(local_rotation, skeleton, "R_Shoulder", axis=[1.0, 0.0, 0.0], angle_deg=-90.0)

    # 更新姿态并写入文件
    tpose_state = SkeletonState.from_rotation_and_root_translation(
        skeleton,
        local_rotation=local_rotation,
        root_translation=tpose_state.root_translation,
        is_local=True,
    )

    os.makedirs(os.path.dirname(output_abs_path), exist_ok=True)
    tpose_state.to_file(output_abs_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SMPL humanoid T-pose SkeletonState from MJCF.")
    parser.add_argument(
        "--mjcf",
        default="../../../../assets/mjcf/smpl_humanoid_s1.xml",
        help="MJCF 文件路径（相对路径以脚本所在目录为基准）。",
    )
    parser.add_argument(
        "--output",
        default="data/smpl_humanoid_s1_tpose.npy",
        help="导出的 T-Pose npy 文件路径（相对路径以脚本所在目录为基准）。",
    )
    parser.add_argument(
        "--lift-arms",
        action="store_true",
        help="若指定，则在导出的 T Pose 中额外抬高手臂（适用于 MJCF 默认姿态非 T Pose 的情况）。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    export_smpl_tpose(
        mjcf_path=args.mjcf,
        output_path=args.output,
        lift_arms=args.lift_arms,
    )


if __name__ == "__main__":
    main()

