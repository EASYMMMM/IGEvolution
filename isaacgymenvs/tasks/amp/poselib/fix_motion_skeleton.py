import argparse
import torch
import sys
import os
import shutil

# 尝试导入 poselib
try:
    from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
except ImportError:
    sys.path.append(os.getcwd())
    from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState

def fix_skeleton(tpose_path, motion_path):
    print("="*80)
    print("💀 SKELETON STRUCTURE FIXER")
    print("="*80)

    # 1. 加载真理 (T-Pose)
    print(f"1. Loading Geometry Truth (T-Pose): {tpose_path}")
    try:
        tpose = SkeletonState.from_file(tpose_path)
        # 获取正确的局部偏移量 [Joints, 3]
        correct_offsets = tpose.skeleton_tree.local_translation
        # 处理可能的 Batch 维度
        if correct_offsets.dim() == 3:
            correct_offsets = correct_offsets[0]
        
        print(f"   Target Skeleton has {correct_offsets.shape[0]} joints.")
        print(f"   Correct Chest Bone Length: {torch.norm(correct_offsets[9]):.4f} m") # 假设 Chest 是 idx 9
    except Exception as e:
        print(f"Error loading T-Pose: {e}")
        return

    # 2. 加载歪掉的运动文件
    print(f"2. Loading Broken Motion: {motion_path}")
    try:
        motion = SkeletonMotion.from_file(motion_path)
        # 检查当前的错误长度
        bad_offsets = motion.skeleton_tree.local_translation
        if bad_offsets.dim() == 3: bad_offsets = bad_offsets[0]
        print(f"   Current (Bad) Chest Bone Length: {torch.norm(bad_offsets[9]):.4f} m")
    except Exception as e:
        print(f"Error loading Motion: {e}")
        return

    # 3. 执行外科手术 (Overwrite)
    print("3. Overwriting skeleton structure...")
    
    # 确保在同一个设备上
    if str(motion.tensor.device) != 'cpu':
        correct_offsets = correct_offsets.to(motion.tensor.device)

    # 【核心操作】直接内存覆盖
    # 注意：这里必须修改 _local_translation，这是 poselib 内部存储偏移量的变量
    motion.skeleton_tree._local_translation[:] = correct_offsets[:]

    # 4. 验证修复结果
    new_offsets = motion.skeleton_tree.local_translation
    if new_offsets.dim() == 3: new_offsets = new_offsets[0]
    new_len = torch.norm(new_offsets[9])
    print(f"   New Chest Bone Length: {new_len:.4f} m")

    if abs(new_len - torch.norm(correct_offsets[9])) < 1e-5:
        print("   ✅ Structure matches T-Pose perfectly.")
    else:
        print("   ❌ Fix failed somehow.")
        return

    # 5. 保存
    backup_path = motion_path + ".bak"
    shutil.copy(motion_path, backup_path)
    print(f"   Backup created at: {backup_path}")
    
    motion.to_file(motion_path)
    print(f"4. Saved FIXED motion to: {motion_path}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tpose', type=str, help='Path to smpl_tpose.npy')
    parser.add_argument('motion', type=str, help='Path to motion.npy to fix')
    args = parser.parse_args()
    
    fix_skeleton(args.tpose, args.motion)