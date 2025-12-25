import torch
import numpy as np

class SimpleCurveGenerator:
    def __init__(self, num_envs, device, dt, episode_duration, 
                 speed_mean=1.0, 
                 turn_speed_max=0.5,  # 最大角速度 (rad/s)，越小弯越大
                 num_turns=3):        # 一条轨迹包含的转向次数
        
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.episode_duration = episode_duration
        
        # 计算总的时间步数
        self.num_steps = int(episode_duration / dt) + 1
        
        self.speed_mean = speed_mean
        self.turn_speed_max = turn_speed_max
        self.num_turns = num_turns

        # z轴设置为0，后续可以填充地形高度
        self._traj_pos = torch.zeros((self.num_envs, self.num_steps, 3), device=self.device, dtype=torch.float)
        
        # 记录每条轨迹的生成的实际速度
        self._traj_speed = torch.zeros((self.num_envs, self.num_steps), device=self.device, dtype=torch.float)

    def reset(self, env_ids, init_pos, init_rot=None):
        """
        为指定的 env_ids 生成新的随机曲线轨迹
        init_pos: [len(env_ids), 3] 机器人的初始位置，确保轨迹从机器人脚下开始
        """
        n = len(env_ids)
        if n == 0:
            return

        # 假设速度恒定
        # shape: [n, num_steps]
        base_speed = self.speed_mean
        random_speed = (torch.rand((n, self.num_steps), device=self.device) - 0.5) * 0.2 # +/- 0.1 m/s 波动
        speed = base_speed + random_speed
        self._traj_speed[env_ids] = speed

        num_keyframes = self.num_turns + 2 # 起点 + 终点 + 中间拐点
        
        # 随机生成关键点的角速度，范围 [-max, +max]
        key_omega = (torch.rand((n, num_keyframes), device=self.device) - 0.5) * 2 * self.turn_speed_max
        
        key_omega = key_omega.unsqueeze(1) # [n, 1, k]
        omega = torch.nn.functional.interpolate(key_omega, size=self.num_steps, mode='linear', align_corners=True)
        omega = omega.squeeze(1) # [n, num_steps]

        # 积分得到航向角
        d_theta = omega * self.dt
        
        # === 根据机器人朝向初始化 ===
        if init_rot is not None:
            # 从四元数计算 Yaw (Heading)
            # Quat format: [x, y, z, w]
            x, y, z, w = init_rot[:, 0], init_rot[:, 1], init_rot[:, 2], init_rot[:, 3]
            # atan2(2(wz + xy), 1 - 2(y^2 + z^2))
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            init_heading = yaw.unsqueeze(-1) # [n, 1]
        else:
            # 随机朝向 (旧逻辑)
            init_heading = (torch.rand(n, device=self.device) * 2 * np.pi).unsqueeze(-1)
        # ==================================
        
        # 累加得到每一步的绝对航向
        heading = torch.cumsum(d_theta, dim=-1) + init_heading

        # 积分得到位置
        dx = speed * torch.cos(heading) * self.dt
        dy = speed * torch.sin(heading) * self.dt

        # 累加
        x_traj = torch.cumsum(dx, dim=-1)
        y_traj = torch.cumsum(dy, dim=-1)

        # 平移轨迹到初始位置
        # 当前 x_traj[0] 不是 0，需要减去第一个点，再加上 init_pos
        x_traj = x_traj - x_traj[:, 0:1] + init_pos[:, 0:1]
        y_traj = y_traj - y_traj[:, 0:1] + init_pos[:, 1:2]

        # 6. 存入 Buffer
        self._traj_pos[env_ids, :, 0] = x_traj
        self._traj_pos[env_ids, :, 1] = y_traj
        self._traj_pos[env_ids, :, 2] = init_pos[:, 2:3] # Z轴保持和初始高度一致(或0)

    def get_position(self, env_ids, time):
        """
        根据当前时间 t (秒) 获取全局坐标目标点
        用于计算 Reward
        """
        # 将时间转换为索引
        step_indices = (time / self.dt).long()
        # 限制不超过最大步数
        step_indices = torch.clamp(step_indices, 0, self.num_steps - 1)
        
        # 通过高级索引获取 [n, 3]
        target_pos = self._traj_pos[env_ids, step_indices, :]
        return target_pos

    def get_observation_points(self, env_ids, current_time, future_times):
        """
        获取未来几个时刻的轨迹点，用于构建 Observation
        current_time: scalar 或者 [n]
        future_times: list or tensor, e.g. [0.5, 1.0, 1.5] 代表未来多少秒
        """
        n = len(env_ids)
        num_samples = len(future_times)
        
        # shape: [n, num_samples]
        if isinstance(current_time, torch.Tensor) and current_time.dim() > 0:
            base_t = current_time.unsqueeze(-1)
        else:
            base_t = torch.full((n, 1), current_time, device=self.device)
            
        offsets = torch.tensor(future_times, device=self.device).unsqueeze(0) # [1, num_samples]
        target_times = base_t + offsets
        
        # 转换为索引
        target_indices = (target_times / self.dt).long()
        target_indices = torch.clamp(target_indices, 0, self.num_steps - 1)
        
        # 取出数据: _traj_pos shape [N_total, T, 3]
        # 我们需要取出 [env_ids, target_indices, :]
        # 先选出对应的环境行
        selected_traj = self._traj_pos[env_ids] # [n, T, 3]
        
        batch_idx = torch.arange(n, device=self.device).unsqueeze(-1).expand(-1, num_samples)
        
        # [n, num_samples, 3]
        sample_pos = selected_traj[batch_idx, target_indices, :]
        
        return sample_pos