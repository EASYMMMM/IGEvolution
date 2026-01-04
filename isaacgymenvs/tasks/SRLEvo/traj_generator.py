import torch
import numpy as np
import torch.nn.functional as F

class SimpleCurveGenerator:
    def __init__(
        self,
        num_envs,
        device,
        dt,
        episode_duration,
        speed_mean=1.0,
        turn_speed_max=0.5,   # 最大角速度 (rad/s)
        num_turns=3,          # 一条轨迹包含的转向次数
        train_stage=2,        # 1: 原地站立，2: 直线行走，3: 曲线行走
    ):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.episode_duration = episode_duration

        # 预计算总步数
        self.num_steps = int(episode_duration / dt) + 1

        self.speed_mean = float(speed_mean)
        self.turn_speed_max = float(turn_speed_max)
        self.num_turns = int(num_turns)
        self.train_stage = int(train_stage)

        # 核心数据存储：预先分配显存
        # [num_envs, num_steps, 3] -> (x, y, z)
        self._traj_pos = torch.zeros((self.num_envs, self.num_steps, 3),
                                     device=self.device, dtype=torch.float)
        # [num_envs, num_steps]
        self._traj_speed = torch.zeros((self.num_envs, self.num_steps),
                                       device=self.device, dtype=torch.float)

        # 【优化1】预先生成环境索引，供 get_observation_points 复用
        self.all_env_indices = torch.arange(self.num_envs, device=self.device)

        # 【优化2】缓存 future_times 的 tensor 版本，避免每一步重复 cpu->gpu copy
        self._cached_future_times = None
        self._cached_offsets = None

    def set_stage(self, stage: int):
        """训练中动态切换 stage。"""
        self.train_stage = int(stage)

    @torch.no_grad()
    def reset(self, env_ids, init_pos, init_rot=None):
        """
        为指定 env_ids 生成轨迹
        init_pos: [n, 3]
        init_rot: [n, 4] (x,y,z,w)
        """
        n = len(env_ids)
        if n == 0:
            return

        # ---------------------------------------------------------
        # 1. 计算初始朝向 (Yaw)
        # ---------------------------------------------------------
        if init_rot is not None:
            # quaternion (x,y,z,w) -> yaw
            x, y, z, w = init_rot[:, 0], init_rot[:, 1], init_rot[:, 2], init_rot[:, 3]
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            init_heading = yaw.unsqueeze(-1)  # [n, 1]
        else:
            init_heading = (torch.rand(n, device=self.device) * 2 * np.pi).unsqueeze(-1)

        # ---------------------------------------------------------
        # 2. Stage 1: 原地站立 (直接返回，不做后续计算)
        # ---------------------------------------------------------
        if self.train_stage == 1:
            self._traj_speed[env_ids] = 0.0
            self._traj_pos[env_ids] = init_pos.unsqueeze(1)
            return

        # ---------------------------------------------------------
        # 3. 速度生成 (Stage 2 & 3 通用)
        # ---------------------------------------------------------
        base_speed = self.speed_mean
        random_speed = (torch.rand((n, self.num_steps), device=self.device) - 0.5) * 0.2
        speed = base_speed + random_speed

        self._traj_speed[env_ids] = speed

        # ---------------------------------------------------------
        # 4. Heading 计算 (合并 Stage 2 & 3 逻辑)
        # ---------------------------------------------------------
        if self.train_stage == 2:
            # Stage 2: 直线，heading 恒定
            heading = init_heading.expand(-1, self.num_steps)

        else:
            # Stage 3: 曲线
            num_keyframes = self.num_turns + 2

            if self.turn_speed_max <= 0:
                key_omega = torch.zeros((n, num_keyframes), device=self.device)
            else:
                key_omega = (torch.rand((n, num_keyframes), device=self.device) - 0.5) * 2 * self.turn_speed_max

            # 插值: [n, num_keyframes] -> [n, num_steps]
            key_omega = key_omega.unsqueeze(1)
            omega = F.interpolate(key_omega, size=self.num_steps, mode='linear', align_corners=True).squeeze(1)

            # 积分: d_theta = omega * dt
            d_theta = omega * self.dt
            heading = torch.cumsum(d_theta, dim=-1) + init_heading  # [n, num_steps]

        # ---------------------------------------------------------
        # 5. 轨迹积分 (Stage 2 & 3 通用)
        # ---------------------------------------------------------
        dx = speed * torch.cos(heading) * self.dt
        dy = speed * torch.sin(heading) * self.dt

        x_traj = torch.cumsum(dx, dim=-1)
        y_traj = torch.cumsum(dy, dim=-1)

        # 平移到 init_pos 为起点
        x_traj = x_traj - x_traj[:, 0:1] + init_pos[:, 0:1]
        y_traj = y_traj - y_traj[:, 0:1] + init_pos[:, 1:2]

        # 批量赋值
        self._traj_pos[env_ids, :, 0] = x_traj
        self._traj_pos[env_ids, :, 1] = y_traj
        self._traj_pos[env_ids, :, 2] = init_pos[:, 2:3]

    @torch.no_grad()
    def get_position(self, env_ids, time):
        """根据当前时间获取目标位置"""
        step_indices = (time / self.dt).long()
        step_indices = torch.clamp(step_indices, 0, self.num_steps - 1)
        return self._traj_pos[env_ids, step_indices, :]

    @torch.no_grad()
    def get_observation_points(self, env_ids, current_time, future_times):
        """
        高度优化的观测获取函数
        future_times: list of floats, e.g. [0.1, 0.2, 0.3]
        """
        n = len(env_ids)

        # 缓存 offsets Tensor，避免每帧 CPU->GPU 拷贝
        if self._cached_future_times != future_times:
            self._cached_future_times = future_times
            self._cached_offsets = torch.tensor(future_times, device=self.device).unsqueeze(0)

        offsets = self._cached_offsets  # [1, k]

        # 处理时间索引
        if isinstance(current_time, torch.Tensor) and current_time.dim() > 0:
            base_t = current_time.unsqueeze(-1)  # [n, 1]
        else:
            base_t = current_time  # scalar

        target_times = base_t + offsets  # [n, k]
        target_indices = (target_times / self.dt).long()
        target_indices = torch.clamp(target_indices, 0, self.num_steps - 1)

        # 高级索引取值
        if n == self.num_envs:
            batch_idx = self.all_env_indices.unsqueeze(-1)  # [N, 1]
        else:
            if isinstance(env_ids, torch.Tensor):
                batch_idx = env_ids.unsqueeze(-1)
            else:
                batch_idx = torch.tensor(env_ids, device=self.device).unsqueeze(-1)

        return self._traj_pos[batch_idx, target_indices, :]