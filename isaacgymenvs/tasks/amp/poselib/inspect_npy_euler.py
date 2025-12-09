import torch
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# 尝试导入 poselib
try:
    from poselib.skeleton.skeleton3d import SkeletonMotion
except ImportError:
    sys.path.append(os.getcwd())
    try:
        from poselib.skeleton.skeleton3d import SkeletonMotion
    except ImportError:
        print("Error: poselib not found.")
        sys.exit(1)

def inspect_frame(file_path, frame_idx=0):
    print(f"Loading: {file_path}")
    motion = SkeletonMotion.from_file(file_path)
    
    # 获取第 0 帧的局部旋转 (Quaternions) [Joints, 4]
    # 注意：poselib 的四元数顺序通常是 [x, y, z, w]
    local_rot = motion.local_rotation[frame_idx]
    
    # 获取根节点位置
    root_pos = motion.root_translation[frame_idx]
    
    print("\n" + "="*80)
    print(f"📂 NPY FILE RAW DATA (Frame {frame_idx})")
    print("="*80)
    print(f"1. NPY Root Pos: {root_pos.numpy()}")
    
    # XML 顺序对照 (假设 smpl_importer.py 是对的)
    # index 1: L_Hip
    # index 2: L_Knee
    # index 4: R_Hip
    
    joints_to_check = {
        "L_Hip": 1,
        "L_Knee": 2,
        "R_Hip": 4
    }
    
    print("2. NPY Rotations -> Euler Conversions:")
    
    for name, idx in joints_to_check.items():
        quat = local_rot[idx].numpy() # [x, y, z, w]
        
        # 使用 Scipy 转换四元数为欧拉角
        # SMPL 的 XML 里关节顺序是 x, y, z，所以我们尝试 'xyz' 顺序
        r = R.from_quat(quat)
        euler_xyz = r.as_euler('xyz', degrees=False)
        euler_zyx = r.as_euler('zyx', degrees=False) # 备选测试
        
        print(f"   Joint: {name} (Index {idx})")
        print(f"     Quaternion: {quat}")
        print(f"     Euler (XYZ): {euler_xyz}  <-- 对应 XML 的 x,y,z 轴")
        print(f"     Euler (ZYX): {euler_zyx}  <-- 另一种常见顺序")
        print("-" * 40)

if __name__ == "__main__":
    # 替换为你的 motion 文件路径
    inspect_frame("smpl_humanoid_walk.npy")