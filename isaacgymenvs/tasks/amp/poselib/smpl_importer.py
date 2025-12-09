import torch
import os
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState

# 1. 关节名称
node_names = [
    "Pelvis", 
    "L_Hip", "L_Knee", "L_Ankle",
    "R_Hip", "R_Knee", "R_Ankle",
    "Torso", "Spine", "Chest", 
    "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"
]

# 2. 父节点索引
parent_indices = [
    -1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 10, 9, 12, 13, 14, 15, 9, 17, 18, 19, 20
]

# 3. 局部偏移量 (Local Translation) - 来自你的 XML
# 【关键修改】：第一行 Pelvis 不再是 [0,0,0]，而是填入 XML 的 pos
local_translation = [
    [-0.0018, -0.2233, 0.0282], # Pelvis (修正：填入 XML pos)
    [-0.0068, 0.0695, -0.0914], # L_Hip
    [-0.0045, 0.0343, -0.3752], # L_Knee
    [-0.0437, -0.0136, -0.398], # L_Ankle
    [-0.0043, -0.0677, -0.0905], # R_Hip
    [-0.0089, -0.0383, -0.3826], # R_Knee
    [-0.0423, 0.0158, -0.3984], # R_Ankle
    [-0.0267, -0.0025, 0.109],  # Torso
    [0.0011, 0.0055, 0.1352],   # Spine
    [0.0254, 0.0015, 0.0529],   # Chest
    [-0.0429, -0.0028, 0.2139], # Neck
    [0.0513, 0.0052, 0.065],    # Head
    [-0.0341, 0.0788, 0.1217],  # L_Thorax
    [-0.0089, 0.091, 0.0305],   # L_Shoulder
    [-0.0275, 0.2596, -0.0128], # L_Elbow
    [-0.0012, 0.2492, 0.009],   # L_Wrist
    [-0.0149, 0.084, -0.0082],  # L_Hand
    [-0.0386, -0.0818, 0.1188], # R_Thorax
    [-0.0091, -0.096, 0.0326],  # R_Shoulder
    [-0.0214, -0.2537, -0.0133],# R_Elbow
    [-0.0056, -0.2553, 0.0078], # R_Wrist
    [-0.0103, -0.0846, -0.0061] # R_Hand
]

def generate():
    local_trans_tensor = torch.tensor(local_translation, dtype=torch.float32)
    parent_indices_tensor = torch.tensor(parent_indices, dtype=torch.long)
    
    # 构建 SkeletonTree
    skeleton_tree = SkeletonTree(
        node_names=node_names,
        parent_indices=parent_indices_tensor,
        local_translation=local_trans_tensor
    )

    # 初始旋转 Identity
    local_rot_tensor = torch.zeros((len(node_names), 4), dtype=torch.float32)
    local_rot_tensor[:, 3] = 1.0

    # 根节点位移设为 0 (因为偏移量已经写在 local_translation[0] 里了)
    root_trans = torch.zeros((1, 3), dtype=torch.float32)
    
    tpose_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree, 
        local_rot_tensor.unsqueeze(0), 
        root_trans, 
        is_local=True
    )

    save_path = "data/smpl_tpose_manual.npy"
    os.makedirs("data", exist_ok=True)
    tpose_state.to_file(save_path)
    print(f"SUCCESS: Generated manual T-Pose at {save_path}")

if __name__ == "__main__":
    generate()