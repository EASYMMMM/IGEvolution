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
        turn_speed_max=0.5,
        num_turns=4,
        train_stage=4,
        is_test=False 
    ):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.episode_duration = episode_duration
        self.is_test = is_test 

        # 预计算总步数
        self.num_steps = int(episode_duration / dt) + 1

        self.speed_mean = float(speed_mean)
        self.turn_speed_max = float(turn_speed_max)
        self.num_turns = int(num_turns)
        self.train_stage = int(train_stage)

        # 显存分配
        self._traj_pos = torch.zeros((self.num_envs, self.num_steps, 3), device=self.device, dtype=torch.float)
        self._traj_speed = torch.zeros((self.num_envs, self.num_steps), device=self.device, dtype=torch.float)
        
        self.all_env_indices = torch.arange(self.num_envs, device=self.device)
        self._cached_future_times = None
        self._cached_offsets = None

        self.action_map = {0: "站立", 1: "直行", 2: "左转", 3: "右转"}

    def set_stage(self, stage: int):
        self.train_stage = int(stage)

    @torch.no_grad()
    def reset(self, env_ids, init_pos, init_rot=None):
        n = len(env_ids)
        if n == 0:
            return

        # 1. 计算初始朝向
        if init_rot is not None:
            x, y, z, w = init_rot[:, 0], init_rot[:, 1], init_rot[:, 2], init_rot[:, 3]
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            init_heading = yaw.unsqueeze(-1)
        else:
            init_heading = (torch.rand(n, device=self.device) * 2 * np.pi).unsqueeze(-1)

        # ================= Stage 1: 原地站立 =================
        if self.train_stage == 1:
            self._traj_speed[env_ids] = 0.0
            self._traj_pos[env_ids] = init_pos.unsqueeze(1)
            return

        # ================= Stage 4: 状态转移混合模式 =================
        elif self.train_stage == 4:
            # === Step A: 动态计算段数 ===
            avg_duration = 6.0 
            num_segments = int(self.episode_duration / avg_duration) + 1
            num_segments = max(num_segments, 3) 

            # === Step B: 基于规则生成状态序列 ===
            seg_types_list = []
            
            # 1. 初始化第一个状态
            first_probs = torch.tensor([0.2, 0.4, 0.2, 0.2], device=self.device).repeat(n, 1)
            current_state = torch.multinomial(first_probs, 1).squeeze(-1) 
            seg_types_list.append(current_state)

            # 2. 循环生成后续状态
            for i in range(1, num_segments):
                prev_state = seg_types_list[-1]
                next_probs = torch.zeros((n, 4), device=self.device)
                
                mask_was_stand = (prev_state == 0)
                mask_was_move = (prev_state > 0)
                
                # 规则 1: 站立后 -> 禁止站立
                if mask_was_stand.any():
                    next_probs[mask_was_stand, 0] = 0.0
                    next_probs[mask_was_stand, 1] = 0.50
                    next_probs[mask_was_stand, 2] = 0.25
                    next_probs[mask_was_stand, 3] = 0.25
                
                # 规则 2: 移动后 -> 30% 站立
                if mask_was_move.any():
                    next_probs[mask_was_move, 0] = 0.30
                    next_probs[mask_was_move, 1] = 0.35
                    next_probs[mask_was_move, 2] = 0.175
                    next_probs[mask_was_move, 3] = 0.175
                
                current_state = torch.multinomial(next_probs, 1).squeeze(-1)
                seg_types_list.append(current_state)
            
            seg_types = torch.stack(seg_types_list, dim=1)
            
            # === Step C: 生成时长 ===
            raw_duration = torch.zeros((n, num_segments), device=self.device, dtype=torch.float)
            
            mask_stand = (seg_types == 0)
            mask_move = (seg_types > 0)
            
            if mask_stand.any():
                raw_duration[mask_stand] = torch.rand(mask_stand.sum(), device=self.device) * 5.0 + 5.0
            
            if mask_move.any():
                raw_duration[mask_move] = torch.rand(mask_move.sum(), device=self.device) * 4.0 + 2.0
            
            total_raw = raw_duration.sum(dim=1, keepdim=True)
            scale_factor = self.episode_duration / total_raw
            final_duration = raw_duration * scale_factor 
            
            seg_counts = (final_duration / self.dt).long()
            current_sums = seg_counts.sum(dim=1)
            diffs = self.num_steps - current_sums
            seg_counts[:, -1] += diffs
            seg_counts = torch.clamp(seg_counts, min=1)
            seg_counts[:, -1] += self.num_steps - seg_counts.sum(dim=1)

            # === Step D: 构建关键值 ===
            base_v = self.speed_mean
            
            key_speed = torch.zeros_like(seg_types, dtype=torch.float)
            key_omega = torch.zeros_like(seg_types, dtype=torch.float)
            
            key_speed[mask_move] = base_v
            
            rand_omegas = torch.rand(seg_types.shape, device=self.device) * 0.3 + 0.15
            mask_left = (seg_types == 2)
            key_omega[mask_left] = rand_omegas[mask_left]
            mask_right = (seg_types == 3)
            key_omega[mask_right] = -rand_omegas[mask_right]

            # === Step E: 插值与强制覆盖 (核心修改) ===
            flat_counts = seg_counts.view(-1)
            flat_speed = key_speed.view(-1)
            flat_omega = key_omega.view(-1)
            
            # 1. 正常插值
            expanded_speed = torch.repeat_interleave(flat_speed, flat_counts)
            expanded_omega = torch.repeat_interleave(flat_omega, flat_counts)
            
            raw_speed = expanded_speed.view(n, self.num_steps)
            raw_omega = expanded_omega.view(n, self.num_steps)

            # 2. 正常平滑 (为了行走的自然过渡)
            kernel_size = 60 
            padding = kernel_size // 2
            smoothing_kernel = torch.ones((1, 1, kernel_size), device=self.device) / kernel_size
            
            speed_in = raw_speed.unsqueeze(1)
            speed_smoothed = F.conv1d(speed_in, smoothing_kernel, padding=padding).squeeze(1)
            speed = speed_smoothed[:, :self.num_steps]

            omega_in = raw_omega.unsqueeze(1)
            omega_smoothed = F.conv1d(omega_in, smoothing_kernel, padding=padding).squeeze(1)
            omega = omega_smoothed[:, :self.num_steps]

            # 3. 【核心修改】强制覆盖：实现“Stage 1 式”的绝对静止
            # 我们需要把 seg_types 也展开，对应到每一个时间步
            flat_types = seg_types.view(-1)
            expanded_types = torch.repeat_interleave(flat_types, flat_counts).view(n, self.num_steps)

            # 找出所有属于“站立片段”的时间步
            mask_stand_step = (expanded_types == 0)

            # 强制将这些步的速度和角速度设为绝对 0.0
            # 这会切断平滑带来的“拖尾”，让目标点瞬间停死，就像 Stage 1 一样
            speed[mask_stand_step] = 0.0
            omega[mask_stand_step] = 0.0
            
            self._traj_speed[env_ids] = speed
            
            d_theta = omega * self.dt
            heading = torch.cumsum(d_theta, dim=-1) + init_heading
            
            # === G. 打印日志 (仅在 Test 模式) ===
            if self.is_test:
                first_idx = 0 
                env_counts_list = seg_counts[first_idx].flatten().cpu().tolist()
                env_types_list = seg_types[first_idx].flatten().cpu().tolist()
                env_omegas_list = key_omega[first_idx].flatten().cpu().tolist()
                
                log_str = ">>> [Stage4] "
                for i, (t_type, t_count) in enumerate(zip(env_types_list, env_counts_list)):
                    t_type = int(t_type)
                    duration = t_count * self.dt
                    
                    info = f"{self.action_map.get(t_type, '未知')}"
                    if (t_type == 2 or t_type == 3) and abs(env_omegas_list[i]) > 0.001:
                        radius = self.speed_mean / abs(env_omegas_list[i])
                        info += f"(R~{radius:.1f}m)"
                        
                    log_str += f"[{info} {duration:.1f}s] -> "
                print(log_str[:-4])

        # ================= Stage 2 & 3: 旧逻辑 =================
        else:
            base_speed = self.speed_mean
            random_speed = (torch.rand((n, self.num_steps), device=self.device) - 0.5) * 0.2
            speed = base_speed + random_speed
            self._traj_speed[env_ids] = speed

            if self.train_stage == 2:
                heading = init_heading.expand(-1, self.num_steps)
            else:
                num_keyframes = self.num_turns + 2
                eff_turn_speed = self.turn_speed_max if self.turn_speed_max > 0.01 else 0.5
                key_omega = (torch.rand((n, num_keyframes), device=self.device) - 0.5) * 2 * eff_turn_speed
                key_omega = key_omega.unsqueeze(1)
                omega = F.interpolate(key_omega, size=self.num_steps, mode='linear', align_corners=True).squeeze(1)
                d_theta = omega * self.dt
                heading = torch.cumsum(d_theta, dim=-1) + init_heading

        # ================= 公共积分 =================
        dx = self._traj_speed[env_ids] * torch.cos(heading) * self.dt
        dy = self._traj_speed[env_ids] * torch.sin(heading) * self.dt

        x_traj = torch.cumsum(dx, dim=-1)
        y_traj = torch.cumsum(dy, dim=-1)

        x_traj = x_traj - x_traj[:, 0:1] + init_pos[:, 0:1]
        y_traj = y_traj - y_traj[:, 0:1] + init_pos[:, 1:2]

        self._traj_pos[env_ids, :, 0] = x_traj
        self._traj_pos[env_ids, :, 1] = y_traj
        self._traj_pos[env_ids, :, 2] = init_pos[:, 2:3]

    @torch.no_grad()
    def get_position(self, env_ids, time):
        step_indices = (time / self.dt).long()
        step_indices = torch.clamp(step_indices, 0, self.num_steps - 1)
        return self._traj_pos[env_ids, step_indices, :]

    @torch.no_grad()
    def get_observation_points(self, env_ids, current_time, future_times):
        n = len(env_ids)
        if self._cached_future_times != future_times:
            self._cached_future_times = future_times
            self._cached_offsets = torch.tensor(future_times, device=self.device).unsqueeze(0)
        offsets = self._cached_offsets

        if isinstance(current_time, torch.Tensor) and current_time.dim() > 0:
            base_t = current_time.unsqueeze(-1)
        else:
            base_t = current_time

        target_times = base_t + offsets
        target_indices = (target_times / self.dt).long()
        target_indices = torch.clamp(target_indices, 0, self.num_steps - 1)

        if n == self.num_envs:
            batch_idx = self.all_env_indices.unsqueeze(-1)
        else:
            if isinstance(env_ids, torch.Tensor):
                batch_idx = env_ids.unsqueeze(-1)
            else:
                batch_idx = torch.tensor(env_ids, device=self.device).unsqueeze(-1)

        return self._traj_pos[batch_idx, target_indices, :]