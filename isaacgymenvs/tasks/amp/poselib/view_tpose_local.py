import argparse
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 强制使用 TkAgg 防止崩溃
matplotlib.use('TkAgg')

try:
    from poselib.skeleton.skeleton3d import SkeletonState
except ImportError:
    print("Error: Could not import poselib.")
    sys.exit(1)

def run_viewer(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    print(f"Loading T-Pose: {file_path} ...")
    try:
        tpose = SkeletonState.from_file(file_path)
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # 获取全局坐标
    global_pos = tpose.global_translation.cpu().numpy()
    
    # ========================================================
    # 【修复点】: 处理 Batch 维度
    # 如果形状是 [1, Joints, 3]，我们需要取 [0] 变成 [Joints, 3]
    # ========================================================
    if global_pos.ndim == 3:
        global_pos = global_pos[0]
    
    parents = tpose.skeleton_tree.parent_indices.cpu().numpy()
    node_names = tpose.skeleton_tree.node_names

    print(f"Joint count: {global_pos.shape[0]}") # 这次应该显示 22

    # 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.manager.set_window_title(f"T-Pose Viewer - {os.path.basename(file_path)}")

    # 计算包围盒
    min_bound = global_pos.min(axis=0)
    max_bound = global_pos.max(axis=0)
    mid = (min_bound + max_bound) / 2
    max_range = (max_bound - min_bound).max() / 2.0

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(min_bound[2], max_bound[2] + 0.5)

    # 画骨骼连线
    for i, p in enumerate(parents):
        if p == -1: continue
        ax.plot(
            [global_pos[p, 0], global_pos[i, 0]],
            [global_pos[p, 1], global_pos[i, 1]],
            [global_pos[p, 2], global_pos[i, 2]],
            color='blue', linewidth=2
        )

    # 画关节红点
    ax.scatter(global_pos[:, 0], global_pos[:, 1], global_pos[:, 2], s=30, c='red')

    # 显示名字
    for i, name in enumerate(node_names):
        ax.text(global_pos[i, 0], global_pos[i, 1], global_pos[i, 2], name, fontsize=9)

    print("Displaying T-Pose. Rotate with mouse.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to tpose .npy file')
    args = parser.parse_args()
    
    run_viewer(args.file)