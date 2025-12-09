import argparse
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 强制使用 TkAgg 后端
matplotlib.use('TkAgg')

try:
    from poselib.skeleton.skeleton3d import SkeletonMotion
except ImportError:
    print("Error: Could not import poselib.")
    sys.exit(1)

def run_viewer(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    print(f"Loading motion: {file_path} ...")
    motion = SkeletonMotion.from_file(file_path)

    # 获取数据
    global_pos = motion.global_translation.cpu().numpy()
    parents = motion.skeleton_tree.parent_indices.cpu().numpy()
    node_names = motion.skeleton_tree.node_names # 获取关节名称
    
    num_frames = global_pos.shape[0]

    # 设置绘图
    fig = plt.figure(figsize=(12, 10)) # 窗口大一点方便看字
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.manager.set_window_title(f"Label Viewer - {os.path.basename(file_path)}")

    # 计算包围盒
    all_pos = global_pos.reshape(-1, 3)
    min_bound = all_pos.min(axis=0)
    max_bound = all_pos.max(axis=0)
    mid = (min_bound + max_bound) / 2
    max_range = (max_bound - min_bound).max() / 2.0

    # 存储文本对象
    texts = []

    def init():
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(0, max_bound[2] + 0.5)
        return []

    def update(frame):
        ax.clear()
        init()
        ax.set_title(f"Frame: {frame}/{num_frames}")

        curr_pos = global_pos[frame]

        # 画连线
        for i, p in enumerate(parents):
            if p == -1: continue
            ax.plot(
                [curr_pos[p, 0], curr_pos[i, 0]],
                [curr_pos[p, 1], curr_pos[i, 1]],
                [curr_pos[p, 2], curr_pos[i, 2]],
                color='blue', alpha=0.5
            )
        
        # 画红点
        ax.scatter(curr_pos[:, 0], curr_pos[:, 1], curr_pos[:, 2], s=20, c='red')

        # 画标签 (名字)
        for i, name in enumerate(node_names):
            # 为了防止字重叠，只显示主要关节，或者你可以把下面if去掉显示全部
            # 这里默认显示全部，字体设小一点
            ax.text(
                curr_pos[i, 0], curr_pos[i, 1], curr_pos[i, 2], 
                name, 
                fontsize=8, 
                color='black'
            )

    print("Starting animation with labels...")
    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=100) # 速度放慢点方便看
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to .npy file')
    args = parser.parse_args()
    run_viewer(args.file)