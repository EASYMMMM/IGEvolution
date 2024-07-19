# 验证四元数关于x-z平面对称
import torch

def my_quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def quat_to_tan_norm(q):
    # type: (Tensor) -> Tensor
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = my_quat_rotate(q, ref_tan)
    
    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = my_quat_rotate(q, ref_norm)
    
    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan

t = torch.tensor([[ 0.389,  0.131, 0.593,  0.693],[ 0.360,  0.197, 0.464,  0.785]])

rot_obs = quat_to_tan_norm(t)
print('rot obs:',rot_obs)

rot_obs[:,1] =  -rot_obs[:,1]
rot_obs[:,4] =  -rot_obs[:,4]

print('mirrored rot obs:',rot_obs)

import numpy as np

# 定义镜像矩阵
observation_permutation_humanoid = np.array([-0.0001, 1, -2, -3, 4, -5, -10, 11, -12, 13, -6,
                                             7, -8, 9, -21, 22, -23, 24, -25, 26, -27, -14, 
                                             15, -16, 17, -18, 19, -20])
observation_permutation_srl = np.array([-32, 33, 34, 35, -28, 29, 30, 31])

# 拼接矩阵
combined_observation_permutation = np.concatenate((observation_permutation_humanoid, observation_permutation_srl))

# 输出拼接后的矩阵
print(combined_observation_permutation)
obs_dim = combined_observation_permutation.shape[0]
print(obs_dim)
obs_perm_mat = torch.zeros((obs_dim, obs_dim), dtype=torch.float32 )
for i, perm in enumerate(combined_observation_permutation):
    obs_perm_mat[i, int(abs(perm))] = np.sign(perm)

print('matrix: ', obs_perm_mat)