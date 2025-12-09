import argparse
import torch
import sys
import os

# 尝试导入 poselib
try:
    from poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion
except ImportError:
    sys.path.append(os.getcwd())
    try:
        from poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion
    except ImportError:
        print("Error: poselib not found.")
        sys.exit(1)

def load_skeleton_data(file_path):
    print(f"Loading: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
        
    try:
        # 尝试作为 Motion 读取
        obj = SkeletonMotion.from_file(file_path)
        sk_tree = obj.skeleton_tree
        file_type = "Motion"
    except:
        try:
            # 尝试作为 State (T-Pose) 读取
            obj = SkeletonState.from_file(file_path)
            sk_tree = obj.skeleton_tree
            file_type = "T-Pose"
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
            
    # 获取局部平移 (骨骼偏移量)
    # local_translation 形状可能是 [Joints, 3] 或 [1, Joints, 3]
    offsets = sk_tree.local_translation
    if offsets.dim() == 3:
        offsets = offsets[0]
        
    return sk_tree.node_names, offsets, file_type

def compare(tpose_path, motion_path):
    print("-" * 100)
    names_t, offsets_t, type_t = load_skeleton_data(tpose_path)
    names_m, offsets_m, type_m = load_skeleton_data(motion_path)
    print("-" * 100)

    # 检查关节数量
    if len(names_t) != len(names_m):
        print(f"CRITICAL ERROR: Joint counts do not match!")
        print(f"T-Pose: {len(names_t)}, Motion: {len(names_m)}")
        return

    print(f"{'ID':<3} | {'Joint Name':<15} | {'T-Pose Len':<10} | {'Motion Len':<10} | {'Diff (m)':<10} | {'Status'}")
    print("-" * 100)

    mismatch_count = 0
    
    for i, name in enumerate(names_t):
        # 确保名字对应
        if names_m[i] != name:
            print(f"Error: Joint name mismatch at index {i}: {name} vs {names_m[i]}")
            break
            
        # 计算长度 (L2 Norm)
        len_t = torch.norm(offsets_t[i]).item()
        len_m = torch.norm(offsets_m[i]).item()
        
        # 计算差异
        diff = abs(len_t - len_m)
        
        # 向量本身的差异 (不仅仅是长度，还要看方向是否变了)
        vec_diff = torch.norm(offsets_t[i] - offsets_m[i]).item()
        
        status = "OK"
        if vec_diff > 1e-4: # 误差大于 0.1 毫米
            status = "MISMATCH!"
            mismatch_count += 1
            
        print(f"{i:<3} | {name:<15} | {len_t:.4f}     | {len_m:.4f}     | {diff:.6f}   | {status}")
        
        # 打印具体的向量值，如果出错的话
        if status == "MISMATCH!":
            print(f"    -> T-Pose Vec: {offsets_t[i].tolist()}")
            print(f"    -> Motion Vec: {offsets_m[i].tolist()}")

    print("-" * 100)
    if mismatch_count == 0:
        print("✅ SUCCESS: Skeleton structure is IDENTICAL.")
        print("The motion file creates exactly the same skeleton dimensions as the T-Pose.")
        print("If it looks weird visually, it is due to joint ROTATIONS (Pose), not bone lengths.")
    else:
        print(f"❌ FAILURE: Found {mismatch_count} joints with mismatched structure.")
        print("The retargeting script failed to preserve the skeleton structure.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare skeleton structure between two npy files.")
    parser.add_argument('tpose', type=str, help='Path to the standard T-Pose npy file')
    parser.add_argument('motion', type=str, help='Path to the generated motion npy file')
    args = parser.parse_args()
    
    compare(args.tpose, args.motion)