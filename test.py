import torch

# 创建一个16*20的action张量
actions = torch.randn(16, 20)  # 随机生成16个环境中的20个动作向量

def compute_reward(actions):
    # 计算每个环境动作的2范数的平方
    torque_penalty = torch.sum(actions[:,:15] ** 2, dim=1)  # 对动作维度求和

    # 这里假设还有其他奖励逻辑，暂时以示例形式表示
    performance_reward = torch.randn(16)  # 随机生成一些性能奖励，仅作为示例

    # 融合奖励和惩罚
    # 注意：这里的-0.001是惩罚系数，可能需要根据实际情况调整
    total_reward = performance_reward - 0.001 * torque_penalty
    print(total_reward)
    return total_reward

# 计算并获取奖励
rewards = compute_reward(actions)
print(actions[1,:15])
print(rewards.shape)  # 验证最终得到的尺寸是否是（16）