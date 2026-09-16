import collections
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np


def quat_to_euler_xyz(quat):
    """Convert MuJoCo quaternion [w, x, y, z] to [yaw, pitch, roll]."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([yaw_z, pitch_y, roll_x], dtype=np.float32)


@dataclass
class EnvConfig:
    model_path: str = "checkpoints/SRL_Real_Vir_s4_v2.pth"
    xml_path: str = "mjcf/srl_real_v1/srl_real_bot_v1.xml"
    dt: float = 0.005
    decimation: int = 3
    gait_period: float = 54.0
    frame_stack: int = 5
    target_vel_x: float = 1.0
    target_ang_vel_z: float = 0.0
    target_height: float = 1.0
    default_dof_pos: tuple = (0.0, -0.1, 0.2, 0.0, -0.1, 0.2)
    kp: tuple = (120.0, 210.0, 280.0, 120.0, 210.0, 280.0)
    kd: tuple = (20.0, 25.0, 40.0, 20.0, 25.0, 40.0)
    action_scale: tuple = (0.71, 0.71, 0.71, 0.71, 0.71, 0.71)
    action_delay_alpha: float = 0.0
    max_torques: tuple = (150.0, 150.0, 150.0, 150.0, 150.0, 150.0)
    gear_ratio: float = 450.0
    max_episode_steps: int = 5000
    termination_height: float = 0.70
    death_cost: float = -1.0
    clip_obs: float = 10.0
    clip_actions: float = 1.0
    foot_clearance: float = 0.2
    target_point_x: float = 1000.0
    alive_reward_scale: float = 0     # HACK
    progress_reward_scale: float = 0  # HACK
    torques_cost_scale: float = 5e-4
    dof_acc_cost_scale: float = 0.5
    dof_vel_cost_scale: float = 1.0
    dof_pos_cost_scale: float = 0.2
    no_fly_penalty_scale: float = 10.0
    vel_tracking_reward_scale: float = 6.0
    tracking_ang_vel_reward_scale: float = 2.0
    gait_similarity_penalty_scale: float = 10.0
    pelvis_height_reward_scale: float = 5.0  # HACK
    orientation_reward_scale: float = 3.0  # HACK
    clearance_penalty_scale: float = 50.0
    lateral_distance_penalty_scale: float = 30.0
    actions_rate_scale: float = 0.3
    actions_smoothness_scale: float = 0.6
    srl_motor_cost_scale: float = 0.5  # HACK
    srl_max_effort: float = 150.0
    srl_rated_nm: float = 60.0
    srl_peak_nm: float = 180.0
    srl_peak_start_ratio: float = 0.7
    srl_thermal_start: float = 0.7
    srl_peak_cost_scale: float = 0.5
    srl_thermal_cost_scale: float = 0.8
    srl_power_cost_scale: float = 0.3
    srl_rated_w: float = 1100.0
    srl_power_start_ratio: float = 0.6
    srl_thermal_tau_s: float = 2.0
    srl_peak_window_tau_s: float = 0.3
    base_wobble_penalty_scale: float = 1.0  # HACK
    base_wobble_yaw_weight: float = 0.8  
    base_wobble_pitch_weight: float = 1.0
    base_wobble_roll_weight: float = 1.2
    base_wobble_wx_weight: float = 0.5
    base_wobble_wy_weight: float = 0.5
    base_wobble_wz_weight: float = 0.2
    proxy_kx: float = 80.0
    proxy_cx: float = 12.0
    proxy_kz: float = 180.0
    proxy_cz: float = 20.0
    proxy_kt: float = 60.0
    proxy_ct: float = 8.0
    proxy_fz_bias: float = 50.0
    proxy_fx_gain: float = 7.0
    proxy_ramp_steps: float = 2000.0
    proxy_dx_limit: float = 0.25
    proxy_dz_limit: float = 0.25
    proxy_dth_limit: float = 0.6
    proxy_vx_limit: float = 2.0
    proxy_vz_limit: float = 2.0
    proxy_vth_limit: float = 4.0
    proxy_fx_limit: float = 150.0
    proxy_fz_min: float = 0.0
    proxy_fz_max: float = 350.0
    proxy_ty_limit: float = 80.0
    proxy_fx_penalty_scale: float = 0.001
    proxy_fz_low: float = 60.0
    proxy_fz_high: float = 260.0
    proxy_fz_penalty_scale: float = 0.001
    proxy_tau_penalty_scale: float = 0.001
    proxy_state_penalty_scale: float = 0.05
    proxy_force_smooth_scale: float = 0.0002
    render: bool = False


class SRLMujocoVirEnv:
    """Single-environment MuJoCo wrapper for PPO finetuning."""

    obs_dim = 153
    act_dim = 6

    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config or EnvConfig()
        self.control_dt = self.cfg.dt * self.cfg.decimation

        self.default_dof_pos = np.asarray(self.cfg.default_dof_pos, dtype=np.float32)
        self.kp = np.asarray(self.cfg.kp, dtype=np.float32)
        self.kd = np.asarray(self.cfg.kd, dtype=np.float32)
        self.action_scale = np.asarray(self.cfg.action_scale, dtype=np.float32)
        self.max_torques = np.asarray(self.cfg.max_torques, dtype=np.float32)

        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.model.jnt_stiffness[:] = 0.0
        self.model.dof_damping[:] = 0.0
        self.data = mujoco.MjData(self.model)
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")

        self.last_action = np.zeros(self.act_dim, dtype=np.float32)
        self.last_applied_torques = np.zeros(self.act_dim, dtype=np.float32)
        self.proxy_state = np.zeros(6, dtype=np.float32)
        self.proxy_force = np.zeros(3, dtype=np.float32)
        self.prev_proxy_force = np.zeros(3, dtype=np.float32)
        self.srl_peak_ratio_window = 0.0
        self.srl_tau2_ema = 0.0
        self.prev_local_lin_vel = np.zeros(3, dtype=np.float32)
        self.prev_local_ang_vel = np.zeros(3, dtype=np.float32)
        self.prev_srl_end_body_pos = np.zeros((2, 3), dtype=np.float32)
        self.target_point = np.array([self.cfg.target_point_x, 0.0, 0.0], dtype=np.float32)
        self.potential = 0.0
        self.prev_potential = 0.0
        self.rl_step_counter = 0
        self.obs_history = collections.deque(maxlen=self.cfg.frame_stack)
        self._srl_thermal_gamma = float(np.exp(-self.control_dt / self.cfg.srl_thermal_tau_s))
        self._srl_peak_decay = float(np.exp(-self.control_dt / self.cfg.srl_peak_window_tau_s))

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = self.cfg.target_height
        self.data.qpos[7:] = self.default_dof_pos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.last_action[:] = 0.0
        self.last_applied_torques[:] = 0.0
        self.proxy_state[:] = 0.0
        self.proxy_force[:] = 0.0
        self.prev_proxy_force[:] = 0.0
        self.srl_peak_ratio_window = 0.0
        self.srl_tau2_ema = 0.0
        self.rl_step_counter = 0

        root_rot_mat = self.data.xmat[self.base_id].reshape(3, 3)
        self.prev_local_lin_vel[:] = root_rot_mat.T @ self.data.qvel[0:3]
        self.prev_local_ang_vel[:] = root_rot_mat.T @ self.data.qvel[3:6]
        self.prev_srl_end_body_pos[:] = self._get_srl_end_body_pos()
        self.potential = self._compute_potential()
        self.prev_potential = self.potential

        initial_obs = self._get_single_frame_obs()
        self.obs_history.clear()
        for _ in range(self.cfg.frame_stack):
            self.obs_history.append(initial_obs.copy())

        return self._get_stacked_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.cfg.clip_actions, self.cfg.clip_actions)

        filtered_action = (
            (1.0 - self.cfg.action_delay_alpha) * action
            + self.cfg.action_delay_alpha * self.last_action
        )
        self.last_action = filtered_action.copy()

        self._update_proxy_human()

        target_pos = self.default_dof_pos + self.action_scale * filtered_action
        for _ in range(self.cfg.decimation):
            curr_pos = self.data.qpos[7:]
            curr_vel = self.data.qvel[6:]
            torques = self.kp * (target_pos - curr_pos) - self.kd * curr_vel
            torques = np.clip(torques, -self.max_torques, self.max_torques)
            self.last_applied_torques[:] = torques.astype(np.float32)
            self.data.ctrl[:] = torques / self.cfg.gear_ratio
            mujoco.mj_step(self.model, self.data)

        self.rl_step_counter += 1
        obs = self._get_stacked_obs()
        reward, terminated, truncated, reward_info = self._compute_reward(obs, filtered_action)
        info = self._get_info()
        info["raw_action"] = action
        info["filtered_action"] = filtered_action
        info["proxy_state"] = self.proxy_state.copy()
        info["proxy_force"] = self.proxy_force.copy()
        info.update(reward_info)
        return obs, reward, terminated, truncated, info

    def _get_single_frame_obs(self):
        root_h = self.data.qpos[2]

        root_rot_mat = self.data.xmat[self.base_id].reshape(3, 3)
        local_lin_vel = root_rot_mat.T @ self.data.qvel[0:3]
        local_ang_vel = root_rot_mat.T @ self.data.qvel[3:6]

        quat = self.data.qpos[3:7]
        euler = quat_to_euler_xyz(quat)
        current_yaw, current_pitch, current_roll = euler[0], euler[1], euler[2]
        yaw_err = np.arctan2(np.sin(-current_yaw), np.cos(-current_yaw))
        euler_err = np.array([yaw_err, -current_pitch, -current_roll], dtype=np.float32)

        dof_pos = self.data.qpos[7:]
        srl_dof_obs = dof_pos - self.default_dof_pos

        dof_vel = self.data.qvel[6:]
        srl_dof_vel = dof_vel * 0.05

        phase_t = (2 * np.pi / self.cfg.gait_period) * (self.rl_step_counter % self.cfg.gait_period)
        sin_phase = np.sin(phase_t)
        cos_phase = np.cos(phase_t)

        obs = np.concatenate(
            [
                np.array([root_h], dtype=np.float32),
                local_lin_vel.astype(np.float32),
                local_ang_vel.astype(np.float32),
                euler_err,
                srl_dof_obs.astype(np.float32),
                srl_dof_vel.astype(np.float32),
                self.last_action.astype(np.float32),
                np.array([sin_phase, cos_phase], dtype=np.float32),
            ]
        )
        return np.clip(obs, -self.cfg.clip_obs, self.cfg.clip_obs)

    def _get_stacked_obs(self):
        current_obs = self._get_single_frame_obs()
        self.obs_history.append(current_obs)
        flat_history = np.concatenate(list(self.obs_history)[::-1])
        task_cmd = np.array(
            [self.cfg.target_vel_x, self.cfg.target_ang_vel_z, self.cfg.target_height],
            dtype=np.float32,
        )
        return np.concatenate([flat_history, task_cmd]).astype(np.float32)

    def _update_proxy_human(self):
        root_rot_mat = self.data.xmat[self.base_id].reshape(3, 3)
        current_local_lin_vel = root_rot_mat.T @ self.data.qvel[0:3]
        current_local_ang_vel = root_rot_mat.T @ self.data.qvel[3:6]

        local_acc = (current_local_lin_vel - self.prev_local_lin_vel) / self.control_dt
        local_ang_acc = (current_local_ang_vel - self.prev_local_ang_vel) / self.control_dt
        self.prev_local_lin_vel[:] = current_local_lin_vel
        self.prev_local_ang_vel[:] = current_local_ang_vel

        dx, dz, dth, vx, vz, vth = self.proxy_state
        ax = local_acc[0] - self.cfg.proxy_cx * vx - self.cfg.proxy_kx * dx
        az = local_acc[2] - self.cfg.proxy_cz * vz - self.cfg.proxy_kz * dz
        ath = local_ang_acc[1] - self.cfg.proxy_ct * vth - self.cfg.proxy_kt * dth

        vx += ax * self.control_dt
        vz += az * self.control_dt
        vth += ath * self.control_dt
        dx += vx * self.control_dt
        dz += vz * self.control_dt
        dth += vth * self.control_dt

        dx = np.clip(dx, -self.cfg.proxy_dx_limit, self.cfg.proxy_dx_limit)
        dz = np.clip(dz, -self.cfg.proxy_dz_limit, self.cfg.proxy_dz_limit)
        dth = np.clip(dth, -self.cfg.proxy_dth_limit, self.cfg.proxy_dth_limit)
        vx = np.clip(vx, -self.cfg.proxy_vx_limit, self.cfg.proxy_vx_limit)
        vz = np.clip(vz, -self.cfg.proxy_vz_limit, self.cfg.proxy_vz_limit)
        vth = np.clip(vth, -self.cfg.proxy_vth_limit, self.cfg.proxy_vth_limit)
        self.proxy_state[:] = [dx, dz, dth, vx, vz, vth]

        fx_local = self.cfg.proxy_fx_gain * (-self.cfg.proxy_kx * dx - self.cfg.proxy_cx * vx)
        fz_local = self.cfg.proxy_fz_bias - self.cfg.proxy_kz * dz - self.cfg.proxy_cz * vz
        ty_local = -self.cfg.proxy_kt * dth - self.cfg.proxy_ct * vth

        fx_local = np.clip(fx_local, -self.cfg.proxy_fx_limit, self.cfg.proxy_fx_limit)
        fz_local = np.clip(fz_local, self.cfg.proxy_fz_min, self.cfg.proxy_fz_max)
        ty_local = np.clip(ty_local, -self.cfg.proxy_ty_limit, self.cfg.proxy_ty_limit)
        self.prev_proxy_force[:] = self.proxy_force
        self.proxy_force[:] = [fx_local, fz_local, ty_local]

        alpha = np.clip(self.rl_step_counter / self.cfg.proxy_ramp_steps, 0.0, 1.0)
        force_local_vec = np.array([alpha * fx_local, 0.0, alpha * fz_local], dtype=np.float32)
        torque_local_vec = np.array([0.0, alpha * ty_local, 0.0], dtype=np.float32)

        force_global = root_rot_mat @ force_local_vec
        torque_global = root_rot_mat @ torque_local_vec

        self.data.xfrc_applied[:] = 0.0
        self.data.xfrc_applied[self.base_id, :3] = force_global
        self.data.xfrc_applied[self.base_id, 3:] = torque_global

    def _get_info(self):
        return {
            "step": self.rl_step_counter,
            "root_height": float(self.data.qpos[2]),
            "target_vel_x": float(self.cfg.target_vel_x),
        }

    def _get_srl_end_body_pos(self):
        return np.stack(
            [
                self.data.xpos[self.left_foot_id].copy(),
                self.data.xpos[self.right_foot_id].copy(),
            ],
            axis=0,
        ).astype(np.float32)

    def _compute_potential(self):
        torso_position = self.data.qpos[0:3].copy()
        to_target = self.target_point - torso_position
        to_target[2] = 0.0
        return -float(np.linalg.norm(to_target)) / self.control_dt

    def _compute_clearance_penalty(self):
        curr = self._get_srl_end_body_pos()
        prev = self.prev_srl_end_body_pos.copy()
        self.prev_srl_end_body_pos[:] = curr

        pz = curr[:, 2]
        dx = curr[:, 0] - prev[:, 0]
        dy = curr[:, 1] - prev[:, 1]
        v_xy = np.sqrt(dx * dx + dy * dy) / self.control_dt
        v_xy = np.where(v_xy < 0.8, 0.0, v_xy)
        this_term = (self.cfg.foot_clearance - pz) ** 2 * v_xy
        return float(np.sum(this_term)), curr

    def _compute_srl_motor_costs(self):
        tau = self.last_applied_torques.astype(np.float32)
        tau_abs = np.abs(tau)

        tau_ratio_peak = tau_abs / float(self.cfg.srl_peak_nm)
        peak_ratio_inst = float(np.max(tau_ratio_peak))
        self.srl_peak_ratio_window = max(
            peak_ratio_inst,
            self.srl_peak_ratio_window * self._srl_peak_decay,
        )
        peak_cost = max(
            (self.srl_peak_ratio_window - self.cfg.srl_peak_start_ratio)
            / (1.0 - self.cfg.srl_peak_start_ratio),
            0.0,
        ) ** 2

        tau_ratio_rated = tau_abs / float(self.cfg.srl_rated_nm)
        tau2_mean = float(np.mean(tau_ratio_rated ** 2))
        g = self._srl_thermal_gamma
        self.srl_tau2_ema = g * self.srl_tau2_ema + (1.0 - g) * tau2_mean
        thermal_cost = max(
            (self.srl_tau2_ema - self.cfg.srl_thermal_start)
            / (1.0 - self.cfg.srl_thermal_start),
            0.0,
        ) ** 2

        qd = self.data.qvel[6:6 + self.act_dim].astype(np.float32)
        p_abs = np.abs(tau * qd)
        p_inst = float(np.max(p_abs))
        p_ratio = p_inst / float(self.cfg.srl_rated_w)
        power_cost = max(
            (p_ratio - self.cfg.srl_power_start_ratio)
            / (1.0 - self.cfg.srl_power_start_ratio),
            0.0,
        ) ** 2

        return peak_cost, thermal_cost, power_cost

    def _compute_reward(self, obs, actions):
        progress = self.rl_step_counter
        target_vel_x = float(self.cfg.target_vel_x)
        target_ang_vel_z = float(self.cfg.target_ang_vel_z)
        target_pelvis_height = float(self.cfg.target_height)
        warmup = np.clip(progress / 10.0, 0.0, 1.0)

        # --- Termination handling ---
        current_potential = self._compute_potential()
        alive_reward_coef = 4.0 if target_vel_x < 0.1 else 1.0
        alive_reward = alive_reward_coef

        progress_reward_coef = 0.0 if target_vel_x < 0.1 else 1.0
        progress_reward = progress_reward_coef * (current_potential - self.prev_potential)
        progress_reward *= warmup

        # --- Pelvis velocity ---
        root_vel = obs[1:4]
        root_target_vel = np.array([target_vel_x, 0.0, 0.0], dtype=np.float32)
        vel_error_vec = root_vel - root_target_vel
        vel_tracking_reward = float(np.exp(-4.0 * np.linalg.norm(vel_error_vec)))

        # --- Torques cost ---
        torques_cost = float(np.sum(actions ** 2))

        # --- DOF deviation cost ---
        srl_dof_pos = obs[10:16].copy()
        srl_dof_pos[0] *= 3.0
        srl_dof_pos[3] *= 3.0
        dof_pos_cost = float(np.sum(srl_dof_pos ** 2))

        # --- DOF velocity cost ---
        dof_vel = obs[16:16 + self.act_dim]
        dof_vel_cost = float(np.sum(dof_vel ** 2))

        # --- DOF acceleration cost ---
        frame = 30
        dof_vel_prev_raw = obs[16 + frame:16 + frame + self.act_dim]
        dof_vel_prev = warmup * dof_vel_prev_raw + (1.0 - warmup) * dof_vel
        dof_acc = dof_vel - dof_vel_prev
        dof_acc_reward_raw = float(np.exp(-2.0 * np.sum(dof_acc ** 2)))
        dof_acc_reward = warmup * dof_acc_reward_raw + (1.0 - warmup) * 1.0

        # --- Action Smooth ---
        actions_prev_raw = obs[22 + frame:22 + frame + self.act_dim]
        actions_prev_prev_raw = obs[22 + 2 * frame:22 + 2 * frame + self.act_dim]
        actions_prev = warmup * actions_prev_raw + (1.0 - warmup) * actions
        actions_prev_prev = warmup * actions_prev_prev_raw + (1.0 - warmup) * actions_prev
        actions_rate = warmup * float(np.sum((actions - actions_prev) ** 2))
        actions_smoothness = warmup * float(np.sum((actions - 2.0 * actions_prev + actions_prev_prev) ** 2))

        # --- Pelvis Orientation ---
        euler_err = obs[7:10]
        angle_diff = ((euler_err + np.pi) % (2 * np.pi)) - np.pi
        yaw = angle_diff[0]
        rp = angle_diff[1:3]
        ori_cost = 0.8 * yaw * yaw + float(np.sum(rp * rp))
        orientation_reward = float(np.exp(-8.0 * ori_cost))

        # --- Pelvis height ---
        pelvis_height = float(obs[0])
        pelvis_height_error = pelvis_height - target_pelvis_height
        pelvis_height_reward = float(np.exp(-6.0 * (3.0 * pelvis_height_error) ** 2))

        # --- Pelvis angular rate ---
        w = obs[4:7]
        wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
        wz_err = wz - target_ang_vel_z
        ang_vel_cost = 3.0 * (wx * wx + wy * wy) + 2.0 * (wz_err * wz_err)
        ang_vel_tracking_reward = float(np.exp(-2.0 * ang_vel_cost))
        base_wobble_penalty = (
            self.cfg.base_wobble_yaw_weight * (yaw * yaw)
            + self.cfg.base_wobble_pitch_weight * (float(rp[0]) * float(rp[0]))
            + self.cfg.base_wobble_roll_weight * (float(rp[1]) * float(rp[1]))
            + self.cfg.base_wobble_wx_weight * (wx * wx)
            + self.cfg.base_wobble_wy_weight * (wy * wy)
            + self.cfg.base_wobble_wz_weight * (wz_err * wz_err)
        )

        clearance_penalty, srl_end_body_pos = self._compute_clearance_penalty()
        srl_root_pos = self.data.qpos[0:3].copy().astype(np.float32)
        left_foot_height = float(srl_end_body_pos[0, 2])
        right_foot_height = float(srl_end_body_pos[1, 2])
        contact_threshold = 0.055
        no_feet_on_ground = (left_foot_height > contact_threshold) and (right_foot_height > contact_threshold)
        no_fly_penalty_scale = self.cfg.no_fly_penalty_scale * (5.0 if target_vel_x < 0.1 else 1.0)
        no_fly_penalty = (no_fly_penalty_scale if no_feet_on_ground else 0.0) * warmup

        local_srl_end_body_pos = srl_end_body_pos - srl_root_pos[None, :]
        lateral_distance = abs(float(local_srl_end_body_pos[0, 1] - local_srl_end_body_pos[1, 1]))
        below_violation = max(0.25 - lateral_distance, 0.0)
        above_violation = max(lateral_distance - 0.85, 0.0)
        feet_lateral_penalty = (below_violation + above_violation) * warmup

        phase_t = (2 * np.pi / self.cfg.gait_period) * float(progress % self.cfg.gait_period)
        phase_left = phase_t
        phase_right = (phase_t + np.pi) % (2 * np.pi)
        expect_stancing_left = 1.0 if np.sin(phase_left) > -0.2 else 0.0
        expect_stancing_right = 1.0 if np.sin(phase_right) > -0.2 else 0.0
        expect_flying_left = 1.0 if np.sin(phase_left) < -0.7 else 0.0
        expect_flying_right = 1.0 if np.sin(phase_right) < -0.7 else 0.0
        is_contact_left = 1.0 if left_foot_height < contact_threshold else 0.0
        is_contact_right = 1.0 if right_foot_height < contact_threshold else 0.0
        stance_miss_left = expect_stancing_left * (1.0 - is_contact_left)
        stance_miss_right = expect_stancing_right * (1.0 - is_contact_right)
        flying_miss_left = expect_flying_left * is_contact_left
        flying_miss_right = expect_flying_right * is_contact_right
        gait_phase_penalty = self.cfg.gait_similarity_penalty_scale * (
            stance_miss_left + stance_miss_right + flying_miss_left + flying_miss_right
        )
        gait_phase_penalty *= warmup

        dx, dz, dth = float(self.proxy_state[0]), float(self.proxy_state[1]), float(self.proxy_state[2])
        fx, fz, tau_pitch = map(float, self.proxy_force)
        prev_fx, prev_fz, prev_tau = map(float, self.prev_proxy_force)
        proxy_state_penalty = dx * dx + dz * dz + dth * dth
        proxy_fx_penalty = abs(fx)
        proxy_tau_penalty = abs(tau_pitch)
        proxy_fz_penalty = max(self.cfg.proxy_fz_low - fz, 0.0) + max(fz - self.cfg.proxy_fz_high, 0.0)
        proxy_force_smooth_penalty = (
            (fx - prev_fx) ** 2 + (fz - prev_fz) ** 2 + (tau_pitch - prev_tau) ** 2
        )
        peak_cost, thermal_cost, power_cost = self._compute_srl_motor_costs()
        srl_motor_cost = (
            self.cfg.srl_peak_cost_scale * peak_cost
            + self.cfg.srl_thermal_cost_scale * thermal_cost
            + self.cfg.srl_power_cost_scale * power_cost
        )

        reward = (
            self.cfg.alive_reward_scale * alive_reward
            + self.cfg.progress_reward_scale * progress_reward
            + self.cfg.vel_tracking_reward_scale * vel_tracking_reward
            + self.cfg.tracking_ang_vel_reward_scale * ang_vel_tracking_reward
            + self.cfg.orientation_reward_scale * orientation_reward
            + self.cfg.pelvis_height_reward_scale * pelvis_height_reward
            - self.cfg.torques_cost_scale * torques_cost
            - self.cfg.dof_pos_cost_scale * dof_pos_cost
            - self.cfg.dof_vel_cost_scale * dof_vel_cost
            + self.cfg.dof_acc_cost_scale * dof_acc_reward
            - self.cfg.actions_rate_scale * actions_rate
            - self.cfg.actions_smoothness_scale * actions_smoothness
            - self.cfg.base_wobble_penalty_scale * base_wobble_penalty
            - no_fly_penalty
            - gait_phase_penalty
            - self.cfg.clearance_penalty_scale * clearance_penalty
            - self.cfg.lateral_distance_penalty_scale * feet_lateral_penalty
            - self.cfg.srl_motor_cost_scale * srl_motor_cost
            - self.cfg.proxy_state_penalty_scale * proxy_state_penalty
            - self.cfg.proxy_fx_penalty_scale * proxy_fx_penalty
            - self.cfg.proxy_fz_penalty_scale * proxy_fz_penalty
            - self.cfg.proxy_tau_penalty_scale * proxy_tau_penalty
            - self.cfg.proxy_force_smooth_scale * proxy_force_smooth_penalty
        )

        terminated = bool(pelvis_height < self.cfg.termination_height)
        truncated = bool(progress >= self.cfg.max_episode_steps)
        if terminated:
            reward = self.cfg.death_cost

        self.prev_potential = current_potential
        self.potential = current_potential

        reward_info = {
            "reward_total": float(reward),
            "reward_alive": float(alive_reward),
            "reward_progress": float(progress_reward),
            "reward_vel_tracking": float(vel_tracking_reward),
            "reward_ang_vel_tracking": float(ang_vel_tracking_reward),
            "reward_orientation": float(orientation_reward),
            "reward_pelvis_height": float(pelvis_height_reward),
            "reward_dof_acc": float(dof_acc_reward),
            "penalty_torques": float(torques_cost),
            "penalty_dof_pos": float(dof_pos_cost),
            "penalty_dof_vel": float(dof_vel_cost),
            "penalty_actions_rate": float(actions_rate),
            "penalty_actions_smoothness": float(actions_smoothness),
            "penalty_base_wobble": float(base_wobble_penalty),
            "penalty_no_fly": float(no_fly_penalty),
            "penalty_gait_phase": float(gait_phase_penalty),
            "penalty_clearance": float(clearance_penalty),
            "penalty_lateral": float(feet_lateral_penalty),
            "penalty_srl_motor": float(srl_motor_cost),
            "penalty_srl_motor_peak": float(peak_cost),
            "penalty_srl_motor_thermal": float(thermal_cost),
            "penalty_srl_motor_power": float(power_cost),
            "penalty_proxy_state": float(proxy_state_penalty),
            "penalty_proxy_fx": float(proxy_fx_penalty),
            "penalty_proxy_fz": float(proxy_fz_penalty),
            "penalty_proxy_tau": float(proxy_tau_penalty),
            "penalty_proxy_smooth": float(proxy_force_smooth_penalty),
        }
        return float(reward), terminated, truncated, reward_info
