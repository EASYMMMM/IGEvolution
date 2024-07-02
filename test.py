import torch

# 定义 compute_srl_reward 函数
def compute_humanoid_reward(obs_buf, dof_force_tensor, action):
    # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
    # reward = torch.ones_like(obs_buf[:, 0])
    velocity  = obs_buf[:,7]  # vx
    target_velocity = 1.4
    velocity_penalty = - torch.where(velocity < target_velocity, (target_velocity - velocity)**2, torch.zeros_like(velocity))

    # 14-28 包括髋关节+膝关节+踝关节
    # torque_usage =  torch.sum(dof_force_tensor[:,14:28] ** 2, dim=1)
    # v1.2.1力矩使用惩罚（假设action代表施加的力矩）
    # torque_reward = - 0.1 *  torque_usage # 惩罚力矩的平方和
    # v1.2.2指数衰减
    # torque_reward = 2*torch.exp(-0.1 * torque_usage)  # 指数衰减，0.1为衰减系数
    # v1.5.12 比例惩罚，力矩绝对值超过100
    torque_threshold = 100
    torque_usage   = dof_force_tensor[:, 14:28]
    torque_penalty = torch.where(torch.abs(torque_usage) > torque_threshold, 
                                 (torch.abs(torque_usage) - torque_threshold) / torque_threshold, 
                                 torch.zeros_like(torque_usage))
    torque_reward  = - torch.sum(torque_penalty, dim=1)
    
    # reward = -velocity_penalty + torque_reward
    reward = velocity_penalty + torque_reward

    return reward, velocity_penalty, torque_reward


# 定义样本数据
obs_buf = torch.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # 一些观察数据，其中速度为1.0
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0]   # 速度为1.5
], dtype=torch.float32)

dof_force_tensor = torch.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 105.0, 0.0, 0.0, 0.0, 0.0, 110.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 超过100的力矩
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 95.0, 0.0, 0.0, 0.0, 0.0, 85.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 未超过100的力矩
], dtype=torch.float32)

action = torch.tensor([
    [0.0] * 28,
    [0.0] * 28
], dtype=torch.float32)

# 调用compute_humanoid_reward函数
reward, velocity_penalty, torque_penalty = compute_humanoid_reward(obs_buf, dof_force_tensor, action)

# 打印结果
print("Reward:", reward)
print("Velocity Penalty:", velocity_penalty)
print("Torque Penalty:", torque_penalty)