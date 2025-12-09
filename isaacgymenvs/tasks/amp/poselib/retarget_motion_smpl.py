# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.

import argparse
import json
import os
import numpy as np
import torch

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

def _resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))

def run_retarget(config_path: str, visualize: bool = False) -> None:
    config_abs_path = os.path.abspath(config_path)
    with open(config_abs_path, "r") as f:
        retarget_data = json.load(f)

    config_dir = os.path.dirname(config_abs_path)

    print("1. Loading Data...")
    source_motion_path = _resolve_path(config_dir, retarget_data["source_motion"])
    target_motion_path = _resolve_path(config_dir, retarget_data["target_motion_path"])
    source_tpose_path = _resolve_path(config_dir, retarget_data["source_tpose"])
    target_tpose_path = _resolve_path(config_dir, retarget_data["target_tpose"])

    os.makedirs(os.path.dirname(target_motion_path), exist_ok=True)

    # 加载 Source Motion
    source_motion = SkeletonMotion.from_file(source_motion_path)
    device = source_motion.tensor.device 
    print(f"   Motion loaded on device: {device}")

    # 加载 T-Poses
    target_tpose = SkeletonState.from_file(target_tpose_path)
    source_tpose = SkeletonState.from_file(source_tpose_path)

    # [修复] 处理 T-Pose 可能存在的 Batch 维度 [1, Joints, 4] -> [Joints, 4]
    if len(target_tpose.tensor.shape) == 2 and target_tpose.tensor.shape[0] == 1:
        target_tpose.tensor = target_tpose.tensor.squeeze(0)
    if len(source_tpose.tensor.shape) == 2 and source_tpose.tensor.shape[0] == 1:
        source_tpose.tensor = source_tpose.tensor.squeeze(0)

    # [修复] 确保所有数据在同一设备
    if str(device) != 'cpu':
        target_tpose.tensor = target_tpose.tensor.to(device)
        source_tpose.tensor = source_tpose.tensor.to(device)
        # 关键：移动骨架定义的偏移量
        target_tpose.skeleton_tree._local_translation = target_tpose.skeleton_tree._local_translation.to(device)
        source_tpose.skeleton_tree._local_translation = source_tpose.skeleton_tree._local_translation.to(device)

    # [验证] 打印 T-Pose 的一些骨骼长度，确保它是对的
    tpose_offsets = target_tpose.skeleton_tree.local_translation
    print(f"   Target T-Pose Bone(1) Length: {torch.norm(tpose_offsets[1]):.4f}")

    print("2. Calculating Rotations (Retargeting)...")
    rotation_to_target_skeleton = torch.tensor(retarget_data["rotation"], device=device, dtype=torch.float32)

    # 核心重定向计算
    # 这一步计算出的 target_motion 可能带有错误的骨骼长度，但旋转是对的
    target_motion = source_motion.retarget_to_by_tpose(
        joint_mapping=retarget_data["joint_mapping"],
        source_tpose=source_tpose,
        target_tpose=target_tpose,
        rotation_to_target_skeleton=rotation_to_target_skeleton,
        scale_to_target_skeleton=retarget_data["scale"],
    )

    # 裁剪帧
    frame_beg = retarget_data.get("trim_frame_beg", 0)
    frame_end = retarget_data.get("trim_frame_end", -1)
    if frame_beg == -1: frame_beg = 0
    if frame_end == -1: frame_end = target_motion.local_rotation.shape[0]

    local_rotation = target_motion.local_rotation[frame_beg:frame_end, ...].clone()
    root_translation = target_motion.root_translation[frame_beg:frame_end, ...].clone()

    # ====================================================================
    # 【关键步骤 3】: 骨架移植手术 (Skeleton Transplant)
    # 抛弃计算过程中可能被篡改的骨架，强制使用 Target T-Pose 的骨架
    # ====================================================================
    print("3. Fixing Skeleton Structure...")
    
    # 使用真理：T-Pose 的骨架树
    final_skeleton_tree = target_tpose.skeleton_tree

    # 重新计算自动贴地 (Auto Grounding)
    # 我们必须用“正确的骨架 + 计算出的旋转”来重新计算脚的位置
    temp_state = SkeletonState.from_rotation_and_root_translation(
        final_skeleton_tree, local_rotation, root_translation, is_local=True
    )
    global_translation = temp_state.global_translation
    # 假设 Z 轴向上，找到所有帧、所有关节的最低点
    min_h = torch.min(global_translation[..., 2])
    
    # 应用高度修正
    root_translation[:, 2] += -min_h
    root_translation[:, 2] += retarget_data.get("root_height_offset", 0.0)

    # ====================================================================
    # 【关键步骤 4】: 构建最终文件
    # 组合：[正确的骨架] + [重定向的旋转] + [修正后的根位置]
    # ====================================================================
    final_motion = SkeletonMotion.from_skeleton_state(
        SkeletonState.from_rotation_and_root_translation(
            final_skeleton_tree, local_rotation, root_translation, is_local=True
        ),
        fps=target_motion.fps
    )

    # 保存
    final_motion.to_file(target_motion_path)
    print(f"SUCCESS: Saved retargeted motion to: {target_motion_path}")
    
    # [验证] 检查保存后的文件骨骼长度是否和 T-Pose 一致
    final_len = torch.norm(final_motion.skeleton_tree.local_translation[1])
    print(f"   Final Motion Bone(1) Length:  {final_len:.4f}")

    if visualize:
        print("Visualizing...")
        try:
            plot_skeleton_motion_interactive(final_motion)
        except Exception as e:
            print(f"Visualization failed (likely headless environment): {e}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Retarget motion from source skeleton to target skeleton using T-poses."
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Retarget config file path.",
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