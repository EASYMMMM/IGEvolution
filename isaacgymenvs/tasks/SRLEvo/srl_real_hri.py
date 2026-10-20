from enum import Enum
import numpy as np
import torch
import os

from gym import spaces

from isaacgym import gymapi
from isaacgym import gymtorch

from isaacgymenvs.tasks.SRLEvo.srl_real_hri_base import SRL_Real_HRI_Base, compute_srl_reward
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
        # Match the pretrained HumanoidAMPMuscleFlat policy interface exactly:
        # 111 proprioceptive values plus three future local XY targets.
        self._traj_sample_times = [0.5, 1.0, 1.5]
        self._num_traj_points = len(self._traj_sample_times)
        self._traj_obs_dim = self._num_traj_points * 2
        self._humanoid_state_obs_dim = 111
        self.cfg = cfg

        state_init = cfg["env"]["stateInit"]
        self._state_init = SRL_Real_HRI.StateInit[state_init] # 初始化方式 （随机重启）
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._max_traj_distance = float(cfg["env"].get("maxTrajectoryDistance", 3.0))
        if self._max_traj_distance <= 0:
            raise ValueError("maxTrajectoryDistance must be positive")
        compliance_cfg = cfg["env"].get("compliance", {})
        self._humanoid_reward_weights = tuple(
            float(compliance_cfg.get(key, default)) for key, default in (
                ("position_reward_weight", 1.0),
                ("velocity_reward_weight", 1.0),
                ("heading_reward_weight", 1.0),
                ("activation_cost_weight", 0.0),
            )
        )

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        self.forward_vec_local = torch.tensor(
            [1.0, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)

        motion_file = cfg['env'].get('motion_file', "amp_humanoid_backflip.npy")
        motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets/amp/motions/" + motion_file)
        self._load_motion(motion_file_path)

        self.num_amp_obs = self._num_amp_obs_steps * NUM_AMP_OBS_PER_STEP  # AMP输入为2个时间步长的观测，(s_t,s_t+1)

        self._amp_obs_space = spaces.Box(np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)  # torch.Size([num_envs, 2, NUM_AMP_OBS_PER_STEP]) ([20, 2, 105])
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0] # s_t  torch.Size([num_envs, 105])
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:] # s_t+1  torch.Size([num_envs, 1, 105])

        # Standing caches are retained for SRL command generation, but they are
        # not part of the 117D humanoid policy observation.
        self._stand_heading_dir = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self._prev_is_standing_phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._standing_phase_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._stand_enter_thres = 0.01
        self._stand_exit_thres = 0.03

        self._amp_obs_demo_buf = None

        return

    def get_humanoid_obs_size(self):
        obs_size = self._humanoid_state_obs_dim + self._traj_obs_dim
        configured_size = int(self.cfg["env"].get("humanoid_obs_num", obs_size))
        if configured_size != obs_size:
            raise ValueError(
                f"HumanoidAMPMuscleFlat checkpoint requires {obs_size}D observations, "
                f"but env.humanoid_obs_num={configured_size}."
            )
        return obs_size

    def _calc_current_heading_xy(self, root_rot):
        num_envs = root_rot.shape[0]
        forward_local = torch.zeros((num_envs, 3), device=root_rot.device, dtype=root_rot.dtype)
        forward_local[:, 0] = 1.0

        forward_world = my_quat_rotate(root_rot, forward_local)
        forward_xy = forward_world[:, 0:2]
        forward_xy = torch.nn.functional.normalize(forward_xy, dim=-1, eps=1e-6)
        return forward_xy

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

        if self.srl_teacher_command_clamp:
            target_vel_x = torch.clamp(
                target_vel_x,
                min=float(self.srl_teacher_vel_x_range[0]),
                max=float(self.srl_teacher_vel_x_range[1]),
            )
            target_ang_vel_z = torch.clamp(
                target_ang_vel_z,
                min=float(self.srl_teacher_ang_vel_z_range[0]),
                max=float(self.srl_teacher_ang_vel_z_range[1]),
            )
            self.target_pelvis_height[env_ids_flat] = torch.clamp(
                self.target_pelvis_height[env_ids_flat],
                min=float(self.srl_teacher_pelvis_height_range[0]),
                max=float(self.srl_teacher_pelvis_height_range[1]),
            )

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
                                     num_dofs=self.humanoid_dof_num,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return

    def reset_idx(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._reset_actors(env_ids)
        # Match standalone Real Bot reset semantics. In particular, the first
        # stacked observation of a new episode must not contain actions left by
        # the preceding episode.
        self.raw_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.srl_filtered_actions[env_ids] = 0.0
        for env_id in env_ids:
            self.srl_obs_buffer[env_id] = 0
            self.srl_obs_mirrored_buffer[env_id] = 0
            self.srl_full_obs_buffer[env_id] = 0
            self.teacher_srl_obs_buffer[env_id] = 0
        self.srl_tau2_ema[env_ids] = 0.0
        self.srl_peak_ratio_window[env_ids] = 0.0

        # Sync reset states before rebuilding the episode trajectory.
        self._refresh_sim_tensors()
        root_pos = self._root_states[env_ids, 0:3]
        root_rot = self._root_states[env_ids, 3:7]
        self._traj_gen.reset(env_ids, root_pos, init_rot=root_rot)
        # Minimal omni-walk migration: reset standing heading cache together with trajectory labels.
        self._reset_standing_heading(env_ids)

        srl_end_body_pos = self._rigid_body_pos[env_ids][:, self._srl_end_ids, :]
        self.prev_srl_end_body_pos[env_ids] = srl_end_body_pos.clone()
        self._compute_observations(env_ids)
        self._init_amp_obs(env_ids)
        return

    def _compute_humanoid_obs(self, env_ids=None):
        # The base implementation already matches HumanoidAMPMuscleFlat:
        # 111D body state (human DOFs only) + 6D local future trajectory.
        return super()._compute_humanoid_obs(env_ids)

    def _compute_observations(self, env_ids=None):
        self._update_srl_task_commands(env_ids)
        super()._compute_observations(env_ids)
        return

    def _compute_reward(self, actions):
        """HumanoidAMPMuscleFlat task reward plus the unchanged SRL reward."""
        srl_end_body_pos = self._rigid_body_pos[:, self._srl_end_ids, :]
        to_target = self.targets - self._initial_root_states[:, 0:3]
        srl_root_pos = self.srl_root_states[:, 0:3]
        clearance_reward = self.compute_foot_clearance_reward()

        srl_peak_cost, srl_thermal_cost, srl_power_cost = self._compute_srl_motor_costs()
        srl_motor_cost = (
            self.srl_peak_cost_scale * srl_peak_cost
            + self.srl_thermal_cost_scale * srl_thermal_cost
            + self.srl_power_cost_scale * srl_power_cost
        )

        env_ids = torch.arange(self.num_envs, device=self.device)
        current_time = self.progress_buf * self.control_dt
        target_pos = self._traj_gen.get_position(env_ids, current_time)
        future = self._traj_gen.get_observation_points(
            env_ids, current_time, self._traj_sample_times
        )
        nominal_velocity = (
            future[:, 0, :2] - target_pos[:, :2]
        ) / self._traj_sample_times[0]

        position_error = target_pos[:, :2] - self._root_states[:, :2]
        pos_reward = torch.exp(-2.0 * torch.sum(position_error ** 2, dim=-1))
        velocity_error = nominal_velocity - self._root_states[:, 7:9]
        vel_reward = torch.exp(-2.0 * torch.sum(velocity_error ** 2, dim=-1))

        speed = torch.linalg.vector_norm(nominal_velocity, dim=-1)
        heading = nominal_velocity / speed[:, None].clamp_min(1e-6)
        forward = my_quat_rotate(self._root_states[:, 3:7], self.forward_vec_local)
        heading_reward = torch.clamp(
            torch.sum(forward[:, :2] * heading, dim=-1), 0.0, 1.0
        )
        heading_reward = torch.where(
            speed > 1e-6, heading_reward, torch.ones_like(heading_reward)
        )

        activation_cost = torch.mean(self.muscle_actuator.a ** 2, dim=-1)
        wp, wv, wh, wa = self._humanoid_reward_weights
        if self.train_stage == 1:
            angular_penalty = self._root_states[:, 12] ** 2
            joint_penalty = torch.sum(
                (
                    self._dof_pos[:, :self.humanoid_dof_num]
                    - self._initial_dof_pos[:, :self.humanoid_dof_num]
                ) ** 2,
                dim=-1,
            )
            humanoid_reward = (
                pos_reward
                * vel_reward
                * torch.exp(-angular_penalty - joint_penalty - wa * activation_cost)
            )
            self.extras["amp_mask"] = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.float
            )
        else:
            humanoid_reward = (
                wp * pos_reward + wv * vel_reward + wh * heading_reward
                - wa * activation_cost
            )
            self.extras["amp_mask"] = torch.ones(
                self.num_envs, device=self.device, dtype=torch.float
            )

        self.rew_buf[:] = humanoid_reward
        self.humanoid_task_rew_buf[:] = humanoid_reward
        self.extras["humanoid_task_rewards"] = {
            "position_mean": pos_reward.mean(),
            "velocity_mean": vel_reward.mean(),
            "heading_mean": heading_reward.mean(),
            "activation_cost_mean": activation_cost.mean(),
        }

        self.srl_rew_buf[:] = compute_srl_reward(
            self.srl_full_obs_buf[:],
            clearance_reward,
            to_target,
            self.progress_buf,
            self.phase_buf,
            self.actions[:, -self.srl_actions_num:],
            srl_end_body_pos,
            srl_root_pos,
            self.potentials,
            self.prev_potentials,
            self._srl_termination_height,
            self.death_cost,
            self.max_episode_length,
            self.gait_period,
            self.humanoid_task_rew_buf,
            srl_obs_num=self.srl_full_obs_num,
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

        dist_sq = torch.sum((target_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1)
        has_strayed = (dist_sq > self._max_traj_distance ** 2) & (self.progress_buf > 10)
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
        self._reset_srl_action_filter(env_ids)
        self.muscle_actuator.a[env_ids] = 0.0

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

        # Match HumanoidAMPMuscleFlat reference-state placement in every
        # sub-environment, including its measured-motion height offset.
        root_pos[:, 0] = self._initial_root_states[env_ids, 0]
        root_pos[:, 1] = self._initial_root_states[env_ids, 1]
        # root_pos[:, 2] += 0.14

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
        
        # MotionLib now exposes the same 28 human DOFs as the standalone task.
        # Attachment and SRL joints keep their configured joint-task defaults.
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._dof_pos[env_ids, :self.humanoid_dof_num] = dof_pos
        self._dof_vel[env_ids, :self.humanoid_dof_num] = dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), 
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self._reset_srl_action_filter(env_ids)
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
            self._curr_amp_obs_buf[:] = build_amp_observations(self._root_states, self._dof_pos[:, :self.humanoid_dof_num], self._dof_vel[:, :self.humanoid_dof_num], key_body_pos,
                                                                self._local_root_obs)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._root_states[env_ids], self._dof_pos[env_ids, :self.humanoid_dof_num],
                                                                    self._dof_vel[env_ids, :self.humanoid_dof_num], key_body_pos[env_ids],
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
