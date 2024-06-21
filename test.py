import torch

# 定义 compute_srl_reward 函数
def compute_srl_reward(obs_buf, dof_force_tensor, action):
    reward = torch.ones_like(obs_buf[:, 0])
    # 14-28 包括髋关节+膝关节+踝关节
    torque_usage = torch.sum(action[:, 14:28] ** 2, dim=1)
    print('torque usage:')
    print(torque_usage)
    # v1.2.1力矩使用惩罚（假设action代表施加的力矩）
    torque_reward = -0.1 * torque_usage  # 惩罚力矩的平方和
    print('torque_reward:')
    print(torque_reward)
    # v1.2.2指数衰减
    torque_reward_decay = torch.exp(-0.1 * torque_usage)  # 指数衰减，0.1为衰减系数
    # r = 0*reward + velocity_reward - torque_cost
    print('torque_reward_decay:')
    print(torque_reward_decay)
    r = torque_reward
    return r

# 随机初始化张量
batch_size = 1
action_dim = 40
obs_buf = torch.rand((batch_size, 50))  # 假设 obs_buf 的第二维度是 50
dof_force_tensor = torch.rand((batch_size, 40))  # 假设 dof_force_tensor 的维度是 (400, 40)
action = torch.rand((batch_size, action_dim))

# 计算奖励
reward = compute_srl_reward(obs_buf, dof_force_tensor, action)

# 打印结果
print("Action:",action)
print("Reward shape:", reward.shape)
print("Reward:", reward)