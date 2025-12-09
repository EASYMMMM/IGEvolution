import torch
import os
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState

# =============================================================================
# 1. 关节名称 (XML 深度优先顺序)
# =============================================================================
node_names = [
    "Pelvis",       # 0
    "L_Hip",        # 1
    "L_Knee",       # 2
    "L_Ankle",      # 3
    "R_Hip",        # 4
    "R_Knee",       # 5
    "R_Ankle",      # 6
    "Torso",        # 7
    "Spine",        # 8
    "Chest",        # 9
    "Neck",         # 10
    "Head",         # 11
    "L_Thorax",     # 12
    "L_Shoulder",   # 13
    "L_Elbow",      # 14
    "L_Wrist",      # 15
    "L_Hand",       # 16
    "R_Thorax",     # 17
    "R_Shoulder",   # 18
    "R_Elbow",      # 19
    "R_Wrist",      # 20
    "R_Hand"        # 21
]

# =============================================================================
# 2. 父子关系索引
# =============================================================================
parent_indices = [
    -1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 10, 9, 12, 13, 14, 15, 9, 17, 18, 19, 20
]

# =============================================================================
# 3. 局部偏移量 (Local Translation)
# 数据来源: smpl_humanoid_0.xml 的 <body> pos 属性
# =============================================================================
local_translation = [
    [-0.0022, -0.2408, 0.0286], # Pelvis (Root)
    [-0.0177, 0.0586, -0.0823], # L_Hip
    [0.008, 0.0435, -0.3865],   # L_Knee
    [-0.0374, -0.0148, -0.4269],# L_Ankle
    [-0.0135, -0.0603, -0.0905],# R_Hip
    [-0.0048, -0.0433, -0.3837],# R_Knee
    [-0.0346, 0.0191, -0.42],   # R_Ankle
    [-0.0384, 0.0044, 0.1244],  # Torso
    [0.0268, 0.0045, 0.138],    # Spine
    [0.0029, -0.0023, 0.056],   # Chest
    [-0.0335, -0.0134, 0.2116], # Neck
    [0.0504, 0.0101, 0.0889],   # Head
    [-0.0189, 0.0717, 0.114],   # L_Thorax
    [-0.019, 0.1229, 0.0452],   # L_Shoulder
    [-0.0229, 0.2553, -0.0156], # L_Elbow
    [-0.0074, 0.2657, 0.0127],  # L_Wrist
    [-0.0156, 0.0867, -0.0106],  # L_Hand
    [-0.0237, -0.083, 0.1125],  # R_Thorax
    [-0.0085, -0.1132, 0.0469], # R_Shoulder
    [-0.0313, -0.2601, -0.0144],# R_Elbow
    [-0.006, -0.2691, 0.0068],  # R_Wrist
    [-0.0101, -0.0888, -0.0087] # R_Hand
]

def generate():
    # 转换为 Tensor
    local_trans_tensor = torch.tensor(local_translation, dtype=torch.float32)
    parent_indices_tensor = torch.tensor(parent_indices, dtype=torch.long)
    
    # 构建骨架树
    skeleton_tree = SkeletonTree(
        node_names=node_names,
        parent_indices=parent_indices_tensor,
        local_translation=local_trans_tensor
    )

    # 初始旋转 Identity
    local_rot_tensor = torch.zeros((len(node_names), 4), dtype=torch.float32)
    local_rot_tensor[:, 3] = 1.0

    # 根节点位移设为 0 (Pelvis 的绝对位置已包含在 offsets[0] 中)
    root_trans = torch.zeros((1, 3), dtype=torch.float32)
    
    tpose_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree, 
        local_rot_tensor.unsqueeze(0), 
        root_trans, 
        is_local=True
    )

    # 输出路径
    save_path = "data/smpl2_tpose.npy"
    os.makedirs("data", exist_ok=True)
    tpose_state.to_file(save_path)
    
    print("="*60)
    print(f"SUCCESS: Generated T-Pose at {save_path}")
    print(f"Root Pos: {local_translation[0]}")
    print("="*60)

if __name__ == "__main__":
    generate()