import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg') # 离线模式
import matplotlib.pyplot as plt
import numpy as np

try:
    from poselib.skeleton.skeleton3d import SkeletonMotion
except ImportError:
    sys.path.append(os.getcwd())
    from poselib.skeleton.skeleton3d import SkeletonMotion

def run_viewer_2d(file_path):
    print(f"Loading: {file_path}")
    motion = SkeletonMotion.from_file(file_path)
    
    # [Frames, Joints, 3]
    global_pos = motion.global_translation.cpu().numpy()
    parents = motion.skeleton_tree.parent_indices.cpu().numpy()
    
    # 假设 Z 是向上，Y 是向前
    
    print("Generating 2D Side-View Frames...")
    save_dir = "frames_2d"
    os.makedirs(save_dir, exist_ok=True)
    
    # 画前 60 帧，每隔 2 帧画一次
    for frame in range(0, min(60, global_pos.shape[0]), 2):
        plt.figure(figsize=(6, 6))
        
        curr_pos = global_pos[frame]
        
        # 画骨骼 (查看 Y-Z 平面，即侧视图)
        # x轴: Y (前后), y轴: Z (上下)
        for i, p in enumerate(parents):
            if p == -1: continue
            plt.plot(
                [curr_pos[p, 1], curr_pos[i, 1]], 
                [curr_pos[p, 2], curr_pos[i, 2]], 
                color='blue', linewidth=2
            )
            
        plt.scatter(curr_pos[:, 1], curr_pos[:, 2], c='red', s=20)
        
        plt.title(f"Side View Frame {frame}")
        plt.xlabel("Y (Forward)")
        plt.ylabel("Z (Up)")
        plt.ylim(0, 2.0) # 固定高度
        plt.xlim(-1, 1)  # 根据需要调整
        plt.grid(True)
        
        filename = os.path.join(save_dir, f"frame_{frame:03d}.png")
        plt.savefig(filename)
        plt.close()
        
    print(f"Done! Saved 2D frames to folder '{save_dir}/'.")
    print("Please download the folder or images to view.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    args = parser.parse_args()
    run_viewer_2d(args.file)