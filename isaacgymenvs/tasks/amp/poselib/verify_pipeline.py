import argparse
import torch
import sys
import os
import numpy as np

# 尝试导入 poselib
try:
    from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
except ImportError:
    sys.path.append(os.getcwd()) # 尝试当前目录
    try:
        from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
    except ImportError:
        print("Error: poselib library not found.")
        sys.exit(1)

def load_data(source_type, file_path):
    # print(f"   Loading {source_type}...")
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    try:
        if source_type == "XML":
            # XML 解析器读取
            sk_tree = SkeletonTree.from_mjcf(file_path)
            offsets = sk_tree.local_translation
        
        elif source_type == "T-Pose":
            # State 读取
            state = SkeletonState.from_file(file_path)
            sk_tree = state.skeleton_tree
            offsets = sk_tree.local_translation
            
        elif source_type == "Motion":
            # Motion 读取
            motion = SkeletonMotion.from_file(file_path)
            sk_tree = motion.skeleton_tree
            offsets = sk_tree.local_translation

        # 统一处理 Batch 维度: [1, J, 3] -> [J, 3]
        if offsets.dim() == 3:
            offsets = offsets.squeeze(0)
            
        return sk_tree.node_names, offsets
    
    except Exception as e:
        print(f"Error parsing {source_type}: {e}")
        sys.exit(1)

def verify(xml_path, tpose_path, motion_path):
    print("=" * 120)
    print("🔍  PIPELINE INTEGRITY CHECKER (XML <-> T-Pose <-> Motion)")
    print("=" * 120)
    print(f"1. XML File:    {xml_path}")
    print(f"2. T-Pose File: {tpose_path}")
    print(f"3. Motion File: {motion_path}")
    print("-" * 120)

    names_x, offsets_x = load_data("XML", xml_path)
    names_t, offsets_t = load_data("T-Pose", tpose_path)
    names_m, offsets_m = load_data("Motion", motion_path)

    # 检查关节数量
    if not (len(names_x) == len(names_t) == len(names_m)):
        print("❌ CRITICAL ERROR: Joint counts do not match!")
        print(f"   XML: {len(names_x)}, T-Pose: {len(names_t)}, Motion: {len(names_m)}")
        return

    # 表头
    # A = XML vs T-Pose, B = T-Pose vs Motion
    print(f"{'Joint Name':<15} | {'XML Len':<8} | {'T-Pose Len':<10} | {'Motion Len':<10} | {'XML vs TP':<9} | {'TP vs Mot':<9} | {'Status'}")
    print("-" * 120)
    
    error_count = 0
    
    for i, name in enumerate(names_x):
        # 名字对齐检查
        if names_t[i] != name or names_m[i] != name:
            print(f"{name:<15} | NAME MISMATCH AT INDEX {i}")
            error_count += 1
            continue

        vec_x = offsets_x[i]
        vec_t = offsets_t[i]
        vec_m = offsets_m[i]

        # 计算长度
        len_x = torch.norm(vec_x).item()
        len_t = torch.norm(vec_t).item()
        len_m = torch.norm(vec_m).item()

        # 计算向量差异 (不仅仅是长度，还有方向)
        diff_xt = torch.norm(vec_x - vec_t).item() # XML vs T-Pose
        diff_tm = torch.norm(vec_t - vec_m).item() # T-Pose vs Motion

        # 判定状态
        status = "OK"
        if diff_xt > 1e-4:
            status = "ERR:XML-TP"
            error_count += 1
        elif diff_tm > 1e-4:
            status = "ERR:TP-Mot"
            error_count += 1
        
        # 打印行
        print(f"{name:<15} | {len_x:.4f}   | {len_t:.4f}     | {len_m:.4f}     | {diff_xt:.6f}  | {diff_tm:.6f}  | {status}")

    print("-" * 120)
    if error_count == 0:
        print("✅ ALL CHECKS PASSED: The pipeline is perfectly consistent.")
        print("   Structure: XML == T-Pose == Motion")
    else:
        print(f"❌ CHECK FAILED: Found {error_count} inconsistencies.")
        print("   ERR:XML-TP  -> Need to fix T-Pose generation (smpl_importer.py)")
        print("   ERR:TP-Mot  -> Need to fix Motion generation (fix_motion_skeleton.py)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('xml', type=str, help='XML file')
    parser.add_argument('tpose', type=str, help='T-Pose NPY file')
    parser.add_argument('motion', type=str, help='Motion NPY file')
    args = parser.parse_args()
    
    verify(args.xml, args.tpose, args.motion)