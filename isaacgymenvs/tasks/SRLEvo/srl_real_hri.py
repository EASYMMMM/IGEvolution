from enum import Enum
import numpy as np
import torch
import os

from gym import spaces

from isaacgym import gymapi
from isaacgym import gymtorch

from isaacgymenvs.tasks.SRLEvo.srl_real_hri_base import SRL_Real_HRI_Base, dof_to_obs, compute_srl_reward
from isaacgymenvs.tasks.amp.utils_amp import gym_util
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, calc_heading_quat_inv, quat_to_tan_norm, my_quat_rotate, exp_map_to_quat


NUM_AMP_OBS_PER_STEP = 13 + 52 + 28 + 12 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]


class SRL_Real_HRI(SRL_Real_HRI_Base):

    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        print("============================ ")
        print('HUMANOID AMP SRL TEST')
        print("============================ ")
        # Minimal omni-walk migration: match the humanoid-only task layout before base init requests obs sizes.
        self._traj_sample_times = [0.5, 1.0, 1.5]
        self._num_traj_points = len(self._traj_sample_times)
        self._traj_obs_dim = self._num_traj_points * 2 + 1 + 1 + 2 + 2
        self._num_state_hist_steps = cfg["env"].get("numStateHistSteps", 5)
        self._humanoid_state_obs_dim = 111
        self.cfg = cfg

        state_init = cfg["env"]["stateInit"]
        self._state_init = SRL_Real_HRI.StateInit[state_init] # 初始化方式 （随机重启）
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Minimal omni-walk migration: base init reuses the old 6D traj command layout,
        # so restore the humanoid-only omni command width here before networks/checkpoints use it.
        self._traj_obs_dim = self._num_traj_points * 2 + 1 + 1 + 2 + 2

        motion_file = cfg['env'].get('motion_file', "amp_humanoid_backflip.npy")
        motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets/amp/motions/" + motion_file)
        self._load_motion(motion_file_path)

        self.num_amp_obs = self._num_amp_obs_steps * NUM_AMP_OBS_PER_STEP  # AMP输入为2个时间步长的观测，(s_t,s_t+1)

        self._amp_obs_space = spaces.Box(np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)  # torch.Size([num_envs, 2, NUM_AMP_OBS_PER_STEP]) ([20, 2, 105])
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0] # s_t  torch.Size([num_envs, 105])
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:] # s_t+1  torch.Size([num_envs, 1, 105])

        # Minimal omni-walk migration: humanoid stacked-state buffer and standing-phase cache.
        self._state_hist_obs_buf = torch.zeros(
            (self.num_envs, self._num_state_hist_steps, self._humanoid_state_obs_dim),
            device=self.device,
            dtype=torch.float
        )
        self._pelvis_body_id = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles[0], "pelvis"
        )
        self._head_body_id = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles[0], "head"
        )
        self._left_foot_body_id = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles[0], "left_foot"
        )
        self._right_foot_body_id = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles[0], "right_foot"
        )
        self._stand_heading_dir = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self._prev_is_standing_phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._standing_phase_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._stand_enter_thres = 0.01
        self._stand_exit_thres = 0.03
        self._stand_pos_tol = 0.08

        self._amp_obs_demo_buf = None

        return

    def get_humanoid_obs_size(self):
        # Minimal omni-walk migration: humanoid obs = stacked base state + single-frame omni command.
        return self._humanoid_state_obs_dim * self._num_state_hist_steps + self._traj_obs_dim

    def _update_hist_state_obs(self, env_ids=None):
        if self._num_state_hist_steps <= 1:
            return

        if env_ids is None:
            for i in reversed(range(self._num_state_hist_steps - 1)):
                self._state_hist_obs_buf[:, i + 1] = self._state_hist_obs_buf[:, i]
        else:
            for i in reversed(range(self._num_state_hist_steps - 1)):
                self._state_hist_obs_buf[env_ids, i + 1] = self._state_hist_obs_buf[env_ids, i]
        return

    def _calc_current_heading_xy(self, root_rot):
        num_envs = root_rot.shape[0]
        forward_local = torch.zeros((num_envs, 3), device=root_rot.device, dtype=root_rot.dtype)
        forward_local[:, 0] = 1.0

        forward_world = my_quat_rotate(root_rot, forward_local)
        forward_xy = forward_world[:, 0:2]
        forward_xy = torch.nn.functional.normalize(forward_xy, dim=-1, eps=1e-6)
        return forward_xy

    def _update_standing_heading_cache(self, is_standing_phase, root_rot):
        current_heading_xy = self._calc_current_heading_xy(root_rot)
        entering_standing = torch.logical_and(
            is_standing_phase,
            torch.logical_not(self._prev_is_standing_phase)
        )

        if torch.any(entering_standing):
            self._stand_heading_dir[entering_standing] = current_heading_xy[entering_standing]

        self._prev_is_standing_phase[:] = is_standing_phase
        return current_heading_xy

    def _reset_standing_heading(self, env_ids):
        if len(env_ids) == 0:
            return

        root_rot = self._root_states[env_ids, 3:7]
        current_heading_xy = self._calc_current_heading_xy(root_rot)
        self._stand_heading_dir[env_ids] = current_heading_xy

        current_time = self.progress_buf[env_ids] * self.control_dt
        target_pos = self._traj_gen.get_position(env_ids, current_time)
        future_traj_points = self._traj_gen.get_observation_points(env_ids, current_time, self._traj_sample_times)
        target_pos_05s = future_traj_points[:, 0, 0:2]
        target_dist = torch.norm(target_pos_05s - target_pos[..., 0:2], dim=-1)

        init_standing = target_dist < self._stand_enter_thres
        self._standing_phase_state[env_ids] = init_standing
        self._prev_is_standing_phase[env_ids] = init_standing
        return

    def _update_standing_phase_state(self, target_dist):
        enter_mask = target_dist < self._stand_enter_thres
        exit_mask = target_dist > self._stand_exit_thres

        self._standing_phase_state = torch.where(
            enter_mask,
            torch.ones_like(self._standing_phase_state),
            torch.where(
                exit_mask,
                torch.zeros_like(self._standing_phase_state),
                self._standing_phase_state
            )
        )
        return self._standing_phase_state

    def _update_srl_task_commands(self, env_ids=None):
        if env_ids is None:
            env_ids_flat = torch.arange(self.num_envs, device=self.device)
            progress_buf = self.progress_buf
            root_rot = self._root_states[:, 3:7]
        else:
            env_ids_flat = env_ids
            progress_buf = self.progress_buf[env_ids]
            root_rot = self._root_states[env_ids, 3:7]

        current_time = progress_buf * self.control_dt
        target_pos = self._traj_gen.get_position(env_ids_flat, current_time)
        future_traj_points = self._traj_gen.get_observation_points(
            env_ids_flat, current_time, self._traj_sample_times
        )

        target_pos_05s = future_traj_points[:, 0, 0:2]
        traj_delta_now = target_pos_05s - target_pos[:, 0:2]
        target_dist = torch.norm(traj_delta_now, dim=-1)

        if env_ids is None:
            prev_state = self._standing_phase_state
            prev_is_standing = self._prev_is_standing_phase
        else:
            prev_state = self._standing_phase_state[env_ids_flat]
            prev_is_standing = self._prev_is_standing_phase[env_ids_flat]

        enter_mask = target_dist < self._stand_enter_thres
        exit_mask = target_dist > self._stand_exit_thres
        is_standing_phase = torch.where(
            enter_mask,
            torch.ones_like(prev_state),
            torch.where(
                exit_mask,
                torch.zeros_like(prev_state),
                prev_state
            )
        )

        current_heading_xy = self._calc_current_heading_xy(root_rot)
        entering_standing = torch.logical_and(
            is_standing_phase,
            torch.logical_not(prev_is_standing)
        )
        if torch.any(entering_standing):
            self._stand_heading_dir[env_ids_flat[entering_standing]] = current_heading_xy[entering_standing]

        saved_heading_xy = self._stand_heading_dir[env_ids_flat]

        walk_dir_now = torch.nn.functional.normalize(traj_delta_now, dim=-1, eps=1e-6)
        future_delta = future_traj_points[:, 1, 0:2] - future_traj_points[:, 0, 0:2]
        walk_dir_future = torch.nn.functional.normalize(future_delta, dim=-1, eps=1e-6)

        walk_yaw = torch.atan2(walk_dir_now[:, 1], walk_dir_now[:, 0])
        future_yaw = torch.atan2(walk_dir_future[:, 1], walk_dir_future[:, 0])
        yaw_delta = torch.atan2(torch.sin(future_yaw - walk_yaw), torch.cos(future_yaw - walk_yaw))
        walk_ang_vel_z = yaw_delta / (self._traj_sample_times[1] - self._traj_sample_times[0])
        walk_speed = target_dist / self._traj_sample_times[0]

        stand_yaw = torch.atan2(saved_heading_xy[:, 1], saved_heading_xy[:, 0])
        zero_cmd = torch.zeros_like(walk_speed)

        target_vel_x = torch.where(is_standing_phase, zero_cmd, walk_speed)
        target_ang_vel_z = torch.where(is_standing_phase, zero_cmd, walk_ang_vel_z)
        target_yaw = torch.where(is_standing_phase, stand_yaw, walk_yaw)

        self._standing_phase_state[env_ids_flat] = is_standing_phase
        self._prev_is_standing_phase[env_ids_flat] = is_standing_phase
        self.target_vel_x[env_ids_flat] = target_vel_x
        self.target_ang_vel_z[env_ids_flat] = target_ang_vel_z
        self.target_yaw[env_ids_flat] = target_yaw
        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        if self.viewer and self.debug_viz:
            self._draw_debug_traj()
        return

    def _draw_debug_traj(self):
        self.gym.clear_lines(self.viewer)
        num_debug_envs = min(self.num_envs, 5)
        for i in range(num_debug_envs):
            traj = self._traj_gen._traj_pos[i]
            points = traj[::10].cpu().numpy()
            lines = np.zeros((len(points) - 1, 6), dtype=np.float32)
            for j in range(len(points) - 1):
                lines[j, :3] = points[j]
                lines[j, 3:] = points[j + 1]
            colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32).repeat(len(lines), axis=0)
            self.gym.add_lines(self.viewer, self.envs[i], len(lines), lines, colors)
        return

    def get_num_amp_obs(self):
        return self.num_amp_obs

    @property
    def dof_names(self):
        return self._dof_names

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)

    def fetch_amp_obs_demo(self, num_samples):
        dt = self.control_dt
        motion_ids = self._motion_lib.sample_motions(num_samples)

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
            
        motion_times0 = self._motion_lib.sample_time(motion_ids)
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps])
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -dt * np.arange(0, self._num_amp_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
                                      self._local_root_obs)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)

        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())
        return amp_obs_demo_flat

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)
        return

    def _load_motion(self, motion_file):
        self._motion_lib = MotionLib(motion_file=motion_file, 
                                     num_dofs=self.num_dof,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return

    def reset_idx(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._reset_actors(env_ids)
        for env_id in env_ids:
            self.srl_obs_buffer[env_id] = 0
            self.srl_obs_mirrored_buffer[env_id] = 0
            self.teacher_srl_obs_buffer[env_id] = 0
        self.srl_tau2_ema[env_ids] = 0.0
        self.srl_peak_ratio_window[env_ids] = 0.0

        root_pos = self._root_states[env_ids, 0:3]
        root_rot = self._root_states[env_ids, 3:7]
        self._traj_gen.reset(env_ids, root_pos, init_rot=root_rot)
        # Minimal omni-walk migration: reset standing heading cache together with trajectory labels.
        self._reset_standing_heading(env_ids)
        self._refresh_sim_tensors()

        srl_end_body_pos = self._rigid_body_pos[env_ids][:, self._srl_end_ids, :]
        self.prev_srl_end_body_pos[env_ids] = srl_end_body_pos.clone()
        self._compute_observations(env_ids)
        self._init_amp_obs(env_ids)
        return

    def _compute_humanoid_obs(self, env_ids=None):
        base_obs_with_traj = super()._compute_humanoid_obs(env_ids)
        base_obs = base_obs_with_traj[:, :-self._num_traj_points * 2]

        if env_ids is None:
            env_ids_flat = torch.arange(self.num_envs, device=self.device)
            current_time = self.progress_buf * self.control_dt
            root_pos = self._root_states[:, 0:3]
            root_rot = self._root_states[:, 3:7]
        else:
            env_ids_flat = env_ids
            current_time = self.progress_buf[env_ids] * self.control_dt
            root_pos = self._root_states[env_ids, 0:3]
            root_rot = self._root_states[env_ids, 3:7]

        target_pos = self._traj_gen.get_position(env_ids_flat, current_time)
        traj_points_world = self._traj_gen.get_observation_points(
            env_ids_flat, current_time, self._traj_sample_times
        )

        # Minimal omni-walk migration: sync standing hysteresis inside obs to avoid one-step command/reward drift.
        traj_delta = traj_points_world[:, 0, 0:2] - target_pos[:, 0:2]
        target_dist_flat = torch.norm(traj_delta, dim=-1)

        if env_ids is None:
            prev_state = self._standing_phase_state
            prev_is_standing = self._prev_is_standing_phase
        else:
            prev_state = self._standing_phase_state[env_ids_flat]
            prev_is_standing = self._prev_is_standing_phase[env_ids_flat]

        enter_mask = target_dist_flat < self._stand_enter_thres
        exit_mask = target_dist_flat > self._stand_exit_thres
        is_standing_phase = torch.where(
            enter_mask,
            torch.ones_like(prev_state),
            torch.where(
                exit_mask,
                torch.zeros_like(prev_state),
                prev_state
            )
        )

        current_heading_xy = self._calc_current_heading_xy(root_rot)
        entering_standing = torch.logical_and(
            is_standing_phase,
            torch.logical_not(prev_is_standing)
        )
        if torch.any(entering_standing):
            self._stand_heading_dir[env_ids_flat[entering_standing]] = current_heading_xy[entering_standing]

        self._standing_phase_state[env_ids_flat] = is_standing_phase
        self._prev_is_standing_phase[env_ids_flat] = is_standing_phase
        saved_heading_xy = self._stand_heading_dir[env_ids_flat]

        heading_inv = calc_heading_quat_inv(root_rot)
        num_samples = self._num_traj_points
        heading_inv_expand = heading_inv.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 4)
        delta_pos = traj_points_world - root_pos.unsqueeze(1)
        delta_pos_flat = delta_pos.reshape(-1, 3)
        local_traj_points = my_quat_rotate(heading_inv_expand, delta_pos_flat)
        local_traj_points = local_traj_points.view(root_pos.shape[0], num_samples, 3)
        traj_obs = local_traj_points[..., 0:2].reshape(root_pos.shape[0], -1)

        cmd_speed = target_dist_flat.unsqueeze(-1) / self._traj_sample_times[0]
        standness = (self._stand_exit_thres - target_dist_flat.unsqueeze(-1)) / (
            self._stand_exit_thres - self._stand_enter_thres + 1e-6
        )
        standness = torch.clamp(standness, 0.0, 1.0)

        walk_dir = torch.nn.functional.normalize(traj_delta, dim=-1, eps=1e-6)
        walk_yaw_cos = torch.sum(current_heading_xy * walk_dir, dim=-1, keepdim=True)
        walk_yaw_sin = (
            current_heading_xy[:, 0:1] * walk_dir[:, 1:2]
            - current_heading_xy[:, 1:2] * walk_dir[:, 0:1]
        )
        stand_yaw_cos = torch.sum(current_heading_xy * saved_heading_xy, dim=-1, keepdim=True)
        stand_yaw_sin = (
            current_heading_xy[:, 0:1] * saved_heading_xy[:, 1:2]
            - current_heading_xy[:, 1:2] * saved_heading_xy[:, 0:1]
        )

        cmd_obs = torch.cat([
            traj_obs,
            cmd_speed,
            standness,
            walk_yaw_cos,
            walk_yaw_sin,
            stand_yaw_cos,
            stand_yaw_sin,
        ], dim=-1)

        # Minimal omni-walk migration: only stack humanoid state, keep command as single-frame.
        if self._num_state_hist_steps > 1:
            if env_ids is None:
                self._update_hist_state_obs()
                self._state_hist_obs_buf[:, 0] = base_obs

                reset_mask = (self.progress_buf == 0)
                if torch.any(reset_mask):
                    self._state_hist_obs_buf[reset_mask] = base_obs[reset_mask].unsqueeze(1).repeat(
                        1, self._num_state_hist_steps, 1
                    )
                hist_obs = self._state_hist_obs_buf.reshape(self.num_envs, -1)
            else:
                self._update_hist_state_obs(env_ids)
                self._state_hist_obs_buf[env_ids, 0] = base_obs

                reset_mask = (self.progress_buf[env_ids] == 0)
                if torch.any(reset_mask):
                    reset_env_ids = env_ids[reset_mask]
                    self._state_hist_obs_buf[reset_env_ids] = base_obs[reset_mask].unsqueeze(1).repeat(
                        1, self._num_state_hist_steps, 1
                    )
                hist_obs = self._state_hist_obs_buf[env_ids].reshape(len(env_ids), -1)
        else:
            hist_obs = base_obs

        return torch.cat([hist_obs, cmd_obs], dim=-1)

    def _compute_observations(self, env_ids=None):
        self._update_srl_task_commands(env_ids)
        super()._compute_observations(env_ids)
        return

    def _compute_reward(self, actions):
        load_cell_sensor = self._virtual_load_cell_from_dof(self._dof_pos, self._dof_vel)
        srl_end_body_pos = self._rigid_body_pos[:, self._srl_end_ids, :]
        to_target = self.targets - self._initial_root_states[:, 0:3]
        srl_root_pos = self.srl_root_states[:, 0:3]
        clearance_reward = self.compute_foot_clearance_reward()

        srl_peak_cost, srl_thermal_cost, srl_power_cost = self._compute_srl_motor_costs()
        srl_motor_cost = (self.srl_peak_cost_scale * srl_peak_cost
                          + self.srl_thermal_cost_scale * srl_thermal_cost
                          + self.srl_power_cost_scale * srl_power_cost)

        env_ids = torch.arange(self.num_envs, device=self.device)
        current_time = self.progress_buf * self.control_dt
        target_pos = self._traj_gen.get_position(env_ids, current_time)
        root_pos = self._root_states[:, 0:3]
        root_rot = self._root_states[:, 3:7]
        future_traj_points = self._traj_gen.get_observation_points(
            env_ids, current_time, self._traj_sample_times
        )
        target_pos_05s = future_traj_points[:, 0, 0:2]
        traj_delta = target_pos_05s - target_pos[:, 0:2]
        target_dist = torch.norm(traj_delta, dim=-1)
        implied_target_speed = target_dist / 0.5

        is_standing_phase = self._update_standing_phase_state(target_dist)
        pos_err_xy = target_pos[:, 0:2] - root_pos[:, 0:2]
        dist_sq = torch.sum(pos_err_xy ** 2, dim=-1)
        pos_reward = torch.exp(-2.0 * dist_sq)

        current_heading_xy = self._update_standing_heading_cache(is_standing_phase, root_rot)
        walk_dir = torch.nn.functional.normalize(traj_delta, dim=-1, eps=1e-6)
        root_vel_xy = self._root_states[:, 7:9]
        current_vel_proj = torch.sum(root_vel_xy * walk_dir, dim=-1)
        vel_reward = torch.exp(-2.0 * (current_vel_proj - implied_target_speed) ** 2)
        walk_heading_alignment = torch.sum(current_heading_xy * walk_dir, dim=-1)
        walk_heading_reward = torch.clamp(walk_heading_alignment, min=0.0, max=1.0)

        saved_heading_xy = self._stand_heading_dir
        stand_heading_dot = torch.sum(current_heading_xy * saved_heading_xy, dim=-1)
        stand_heading_dot = torch.clamp(stand_heading_dot, min=-1.0, max=1.0)
        stand_heading_cross = (
            current_heading_xy[:, 0] * saved_heading_xy[:, 1]
            - current_heading_xy[:, 1] * saved_heading_xy[:, 0]
        )
        stand_heading_err = torch.atan2(stand_heading_cross, stand_heading_dot)
        stand_heading_penalty = stand_heading_err ** 2

        stand_pos_err = torch.norm(pos_err_xy, dim=-1)
        stand_pos_reward = torch.where(
            stand_pos_err < self._stand_pos_tol,
            torch.ones_like(stand_pos_err),
            torch.exp(-20.0 * (stand_pos_err - self._stand_pos_tol) ** 2)
        )

        root_vel_stand = self._root_states[:, 7:10]
        stand_speed_xy = torch.norm(root_vel_stand[:, 0:2], dim=-1)
        stand_vel_excess = torch.clamp(stand_speed_xy - 0.15, min=0.0)
        stand_vel_reward = torch.exp(-8.0 * stand_vel_excess ** 2)

        stand_yaw_rate = torch.abs(self._root_states[:, 12])
        stand_yaw_rate_excess = torch.clamp(stand_yaw_rate - 0.20, min=0.0)
        stand_yaw_rate_reward = torch.exp(-12.0 * stand_yaw_rate_excess ** 2)

        dof_abs_err = torch.abs(self._dof_pos - self._initial_dof_pos)
        joint_excess = torch.clamp(dof_abs_err - 0.20, min=0.0)
        stand_joint_penalty = torch.sqrt(torch.mean(joint_excess ** 2, dim=-1) + 1e-8)

        pelvis_pos = self._rigid_body_pos[:, self._pelvis_body_id, :]
        head_pos = self._rigid_body_pos[:, self._head_body_id, :]
        trunk_vec = head_pos - pelvis_pos
        trunk_dir = trunk_vec / torch.norm(trunk_vec, dim=-1, keepdim=True).clamp(min=1e-6)
        trunk_upright_cos = torch.clamp(trunk_dir[:, 2], min=-1.0, max=1.0)
        stand_trunk_reward = torch.exp(-8.0 * (1.0 - trunk_upright_cos))

        left_foot_pos = self._rigid_body_pos[:, self._left_foot_body_id, :]
        right_foot_pos = self._rigid_body_pos[:, self._right_foot_body_id, :]
        foot_delta_xy = left_foot_pos[:, 0:2] - right_foot_pos[:, 0:2]
        foot_dist_xy = torch.norm(foot_delta_xy, dim=-1)
        foot_dist_excess = torch.clamp(foot_dist_xy - (0.22 + 0.10), min=0.0)
        stand_foot_spacing_reward = torch.exp(-10.0 * foot_dist_excess ** 2)

        left_foot_height_excess = torch.clamp(left_foot_pos[:, 2] - (0.06 + 0.025), min=0.0)
        right_foot_height_excess = torch.clamp(right_foot_pos[:, 2] - (0.06 + 0.025), min=0.0)
        stand_foot_ground_reward = torch.exp(
            -80.0 * (left_foot_height_excess ** 2 + right_foot_height_excess ** 2)
        )

        stand_reward = (
            0.35 * stand_pos_reward
            - 0.50 * stand_heading_penalty
            + 0.15 * stand_vel_reward
            + 0.15 * stand_yaw_rate_reward
            + 0.20 * stand_trunk_reward
            + 0.08 * stand_foot_spacing_reward
            + 0.07 * stand_foot_ground_reward
            - 0.70 * stand_joint_penalty
        )
        stand_reward = torch.clamp(stand_reward, min=0.0)

        # Minimal omni-walk migration: keep the humanoid solo-training stage labels in MARL.
        if self.train_stage == 1:
            self.rew_buf[:] = stand_reward
            self.humanoid_task_rew_buf[:] = stand_reward
            self.extras["amp_mask"] = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.float
            )
        elif self.train_stage == 4:
            walk_reward = pos_reward + vel_reward + walk_heading_reward
            stand_reward_s4 = stand_reward * 3.0
            humanoid_reward = torch.where(is_standing_phase, stand_reward_s4, walk_reward)
            humanoid_task_reward = torch.where(is_standing_phase, stand_reward, walk_reward)
            self.rew_buf[:] = humanoid_reward
            self.humanoid_task_rew_buf[:] = humanoid_task_reward
            self.extras["amp_mask"] = (~is_standing_phase).float()
        else:
            walk_reward = pos_reward + vel_reward + walk_heading_reward
            self.rew_buf[:] = walk_reward
            self.humanoid_task_rew_buf[:] = walk_reward

        self.srl_rew_buf[:]  = compute_srl_reward(
            self.srl_obs_buf[:],
            clearance_reward,
            to_target,
            self.progress_buf,
            self.phase_buf,
            self.actions[:,-self.srl_actions_num:],
            srl_end_body_pos,
            srl_root_pos,
            self.potentials,
            self.prev_potentials,
            self._srl_termination_height,
            self.death_cost,
            self.max_episode_length,
            self.gait_period,
            self.humanoid_task_rew_buf,
            srl_obs_num=self.srl_obs_num,
            alive_reward_scale=self.alive_reward_scale,
            humanoid_share_reward_scale=self.humanoid_share_reward_scale,
            progress_reward_scale=self.progress_reward_scale,
            torques_cost_scale=self.torques_cost_scale,
            dof_acc_cost_scale=self.dof_acc_cost_scale,
            dof_vel_cost_scale=self.dof_vel_cost_scale,
            dof_pos_cost_scale=self.dof_pos_cost_scale,
            contact_force_cost_scale=self.contact_force_cost_scale,
            no_fly_penalty_scale=self.no_fly_penalty_scale,
            vel_tracking_reward_scale=self.vel_tracking_reward_scale,
            tracking_ang_vel_reward_scale=self.tracking_ang_vel_reward_scale,
            gait_similarity_penalty_scale=self.gait_similarity_penalty_scale,
            pelvis_height_reward_scale=self.pelvis_height_reward_scale,
            orientation_reward_scale=self.orientation_reward_scale,
            clearance_penalty_scale=self.clearance_penalty_scale,
            lateral_distance_penalty_scale=self.lateral_distance_penalty_scale,
            actions_rate_scale=self.actions_rate_scale,
            actions_smoothness_scale=self.actions_smoothness_scale,
            srl_motor_cost=srl_motor_cost,
            srl_motor_cost_scale=self.srl_motor_cost_scale,
        )
        return

    def _compute_reset(self):
        super()._compute_reset()

        current_time = self.progress_buf * self.control_dt
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_pos = self._traj_gen.get_position(env_ids, current_time)
        root_pos = self._root_states[..., 0:3]

        # Minimal omni-walk migration: use the same sensible trajectory-deviation reset scale as humanoid solo training.
        dist_sq = torch.sum((target_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1)
        has_strayed = (dist_sq > (1.5 * 1.5)) & (self.progress_buf > 10)
        strayed_tensor = has_strayed.to(dtype=torch.long)
        self.reset_buf = self.reset_buf | strayed_tensor
        self._terminate_buf = self._terminate_buf | strayed_tensor
        return

    def _reset_actors(self, env_ids):
        # super()._reset_actors(env_ids)
        to_target = self.targets[env_ids] - self._initial_root_states[env_ids, 0:3]
        to_target[:, self.up_axis_idx] = 0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.control_dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        srl_end_body_pos = self._rigid_body_pos[:, self._srl_end_ids, :].clone()
        self.prev_srl_end_body_pos[env_ids] = srl_end_body_pos[env_ids,:,:].clone()

        if (self._state_init == SRL_Real_HRI.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == SRL_Real_HRI.StateInit.Start
              or self._state_init == SRL_Real_HRI.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == SRL_Real_HRI.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

        return
    
    def _reset_default(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        if (self._state_init == SRL_Real_HRI.StateInit.Random
            or self._state_init == SRL_Real_HRI.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == SRL_Real_HRI.StateInit.Start):
            motion_times = np.zeros(num_envs)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.control_dt
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps - 1])
        motion_times = np.expand_dims(motion_times, axis=-1)
        time_steps = -dt * (np.arange(0, self._num_amp_obs_steps - 1) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
                                      self._local_root_obs)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._root_states[env_ids, 0:3]   = root_pos
        self._root_states[env_ids, 3:7]   = root_rot
        self._root_states[env_ids, 7:10]  = root_vel
        self._root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        # MLY: SRL init
        self._dof_pos[env_ids, -self.srl_actions_num:] = self._initial_dof_pos[env_ids,-self.srl_actions_num:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), 
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return

    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations(self._root_states, self._dof_pos, self._dof_vel, key_body_pos,
                                                                self._local_root_obs)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._root_states[env_ids], self._dof_pos[env_ids], 
                                                                    self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                    self._local_root_obs)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def dof_to_obs_amp(pose):
    # type: (Tensor) -> Tensor
    #dof_obs_size = 64
    #dof_offsets = [0, 3, 6, 9, 12, 13, 16, 19, 20, 23, 24, 27, 30, 31, 34]
    '''
    仅计算amp的obs
    '''
    dof_obs_size = 52
    dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28] # humanoid 
    
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset:(dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs

@torch.jit.script
def build_amp_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs_amp(dof_pos)

    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel[:,0:28], flat_local_key_pos), dim=-1)
    return obs
