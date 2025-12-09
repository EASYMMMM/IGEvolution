import argparse
import torch
import sys
import os
import numpy as np

# 尝试导入 poselib
try:
    from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
except ImportError:
    sys.path.append(os.getcwd())
    try:
        from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
    except ImportError:
        print("Error: poselib not found. Please run this script in the correct directory.")
        sys.exit(1)

def get_skeleton_data(source_type, file_path):
    print(f"Loading {source_type}: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    try:
        if source_type == "XML":
            # 使用 poselib 官方解析器读取 XML
            # 这是最权威的，因为它代表了物理引擎/训练代码眼中的 XML 结构
            sk_tree = SkeletonTree.from_mjcf(file_path)
            offsets = sk_tree.local_translation
        
        elif source_type == "NPY":
            # 读取 T-Pose 文件
            state = SkeletonState.from_file(file_path)
            sk_tree = state.skeleton_tree
            offsets = sk_tree.local_translation
            
            # 处理 Batch 维度 [1, J, 3] -> [J, 3]
            if offsets.dim() == 3:
                offsets = offsets.squeeze(0)
                
        return sk_tree.node_names, offsets
    
    except Exception as e:
        print(f"Error parsing {source_type}: {e}")
        sys.exit(1)

def compare(xml_path, npy_path):
    print("=" * 100)
    print("⚖️  SKELETON INTEGRITY CHECKER")
    print("=" * 100)
    
    names_xml, offsets_xml = get_skeleton_data("XML", xml_path)
    names_npy, offsets_npy = get_skeleton_data("NPY", npy_path)
    
    print("-" * 100)

    # 1. 检查关节名称和数量是否一致
    if len(names_xml) != len(names_npy):
        print(f"❌ CRITICAL ERROR: Joint counts mismatch!")
        print(f"   XML Joints: {len(names_xml)}")
        print(f"   NPY Joints: {len(names_npy)}")
        # 打印差异
        set_xml = set(names_xml)
        set_npy = set(names_npy)
        print(f"   In XML but not NPY: {set_xml - set_npy}")
        print(f"   In NPY but not XML: {set_npy - set_xml}")
        return

    # 2. 逐个关节对比偏移量 (Bone Length / Offset)
    print(f"{'Joint Name':<20} | {'XML Offset (x,y,z)':<25} | {'NPY Offset (x,y,z)':<25} | {'Diff':<8} | {'Status'}")
    print("-" * 100)
    
    mismatch_count = 0
    
    for i, name in enumerate(names_xml):
        # 按照名字找对应的索引（防止顺序不一致）
        if name not in names_npy:
            print(f"{name:<20} | {'MISSING IN NPY':<50} | ❌")
            mismatch_count += 1
            continue
            
        idx_npy = names_npy.index(name)
        
        vec_xml = offsets_xml[i]
        vec_npy = offsets_npy[idx_npy]
        
        # 计算向量距离
        diff = torch.norm(vec_xml - vec_npy).item()
        
        # 格式化字符串
        str_xml = f"[{vec_xml[0]:.3f}, {vec_xml[1]:.3f}, {vec_xml[2]:.3f}]"
        str_npy = f"[{vec_npy[0]:.3f}, {vec_npy[1]:.3f}, {vec_npy[2]:.3f}]"
        
        status = "OK"
        if diff > 1e-4: # 误差容忍度 0.1mm
            status = "MISMATCH"
            mismatch_count += 1
        
        print(f"{name:<20} | {str_xml:<25} | {str_npy:<25} | {diff:.4f}   | {status}")

    print("-" * 100)
    if mismatch_count == 0:
        print("✅ VERIFICATION PASSED: The T-Pose NPY perfectly matches the XML model structure.")
        print("   Conclusion: The skeleton dimensions in the NPY file are correct.")
    else:
        print(f"❌ VERIFICATION FAILED: Found {mismatch_count} mismatches.")
        print("   Conclusion: The T-Pose NPY does NOT represent the XML model correctly.")
        print("   Action: You need to regenerate the T-Pose NPY from the XML.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('xml', type=str, help='Path to .xml file')
    parser.add_argument('npy', type=str, help='Path to .npy t-pose file')
    args = parser.parse_args()
    
    compare(args.xml, args.npy)