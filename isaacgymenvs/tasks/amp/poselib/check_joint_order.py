import os
import sys
from isaacgym import gymapi

# 尝试导入 poselib
try:
    from poselib.skeleton.skeleton3d import SkeletonMotion
except ImportError:
    sys.path.append(os.getcwd())
    try:
        from poselib.skeleton.skeleton3d import SkeletonMotion
    except ImportError:
        print("Error: Could not import poselib.")
        sys.exit(1)

def check_order(xml_path, npy_path):
    # 1. 处理路径 (关键修复：转换为绝对路径)
    xml_abs_path = os.path.abspath(xml_path)
    npy_abs_path = os.path.abspath(npy_path)
    
    if not os.path.exists(xml_abs_path):
        print(f"Error: XML file not found at: {xml_abs_path}")
        return
    if not os.path.exists(npy_abs_path):
        print(f"Error: NPY file not found at: {npy_abs_path}")
        return

    print(f"Checking Alignment:\nXML: {xml_abs_path}\nNPY: {npy_abs_path}")
    print("-" * 60)

    # 2. 初始化 Isaac Gym (只用于解析 XML)
    gym = gymapi.acquire_gym()
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, gymapi.SimParams())
    
    if sim is None:
        print("Error: Failed to create sim.")
        return

    asset_root = os.path.dirname(xml_abs_path)
    asset_file = os.path.basename(xml_abs_path)
    
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    
    print(f"Loading Asset from: Root='{asset_root}', File='{asset_file}'")
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    
    if asset is None:
        print("❌ Failed to load XML asset! Please check the file content.")
        return

    # 获取 Body 名称 (物理引擎解析出的刚体列表)
    # 这是 Isaac Gym 内部认为的骨骼顺序
    gym_body_names = gym.get_asset_rigid_body_names(asset)

    # 3. 获取 NPY 的顺序
    try:
        motion = SkeletonMotion.from_file(npy_abs_path)
        npy_node_names = motion.skeleton_tree.node_names
    except Exception as e:
        print(f"❌ Failed to load NPY file: {e}")
        return

    # 4. 对比分析
    print(f"\n{'Index':<5} | {'XML Body Name (Physics Order)':<35} | {'NPY Node Name (Data Order)':<35} | {'Match?'}")
    print("-" * 90)
    
    # XML Body 0 通常是 World 或者 Pelvis (取决于是否有 freejoint)
    # 我们取两者的最小长度进行对比
    min_len = min(len(gym_body_names), len(npy_node_names))
    
    mismatch_flag = False
    
    for i in range(min_len):
        xml_name = gym_body_names[i]
        npy_name = npy_node_names[i]
        
        # 检查名称是否一致
        is_match = "YES" if (xml_name == npy_name) else "NO <---"
        if xml_name != npy_name:
            mismatch_flag = True
        
        print(f"{i:<5} | {xml_name:<35} | {npy_name:<35} | {is_match}")

    print("-" * 90)
    if mismatch_flag:
        print("❌ CRITICAL: ORDER MISMATCH DETECTED!")
        print("Isaac Gym applies rotations based on the XML order (Left column).")
        print("But your .npy file provides data in a different order (Right column).")
        print("Result: The robot is receiving the wrong data for its joints (e.g., Left Arm data applied to Right Leg).")
        print("Solution: Update your 'smpl_importer.py' node_names list to match the XML order exactly.")
    else:
        print("✅ Order is CORRECT.")
        print("The skeleton hierarchy matches perfectly.")

if __name__ == "__main__":
    # 直接使用当前目录下的文件名
    check_order("smpl_0_humanoid.xml", "smpl_humanoid_walk.npy")