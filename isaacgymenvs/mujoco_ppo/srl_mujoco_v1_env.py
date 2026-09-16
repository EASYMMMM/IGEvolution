import collections
from dataclasses import dataclass
from typing import Optional, Tuple, Union

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


def _as_float_array(value, length):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = np.full(length, float(arr), dtype=np.float32)
    return arr


@dataclass
class WalkEnvConfig:
    xml_path: str = "mjcf/srl_real_v1/srl_real_bot_v1.xml"
    dt: float = 0.005
    decimation: int = 3
    gait_period: float = 54.0
    frame_stack: int = 5

    target_vel_x: float = 1.0
    target_ang_vel_z: float = 0.0
    target_height: float = 1.0
    target_yaw: float = 0.0
    target_point_x: float = 1000.0

    default_dof_pos: Tuple[float, ...] = (0.0, -0.1, 0.2, 0.0, -0.1, 0.2)
    #kp: Tuple[float, ...] = (98.0, 176.4, 259.7, 98.0, 176.4, 259.7)
    #kd: Tuple[float, ...] = (25.5, 29.58, 48.96, 25.5, 29.58, 48.96)
    kp: Tuple[float, ...] = (114.0, 199.5, 266.0, 114.0, 199.5, 266.0)
    kd: Tuple[float, ...] = (22.0, 27.5, 44.0, 22.0, 27.5, 44.0)
    action_scale: Tuple[float, ...] = (0.71, 0.71, 0.71, 0.71, 0.71, 0.71)
    action_delay_alpha: float = 0.0
    clip_actions: float = 1.0
    clip_obs: float = 10.0

    max_torques: Tuple[float, ...] = (150.0, 150.0, 150.0, 150.0, 150.0, 150.0)
    gear_ratio: float = 450.0
    max_torque_step: float = 0.0
    dof_vel_filter_alpha: float = 1.0
    base_w_damp_x: float = 0.0
    base_w_damp_y: float = 0.0
    passive_dof_damping: Union[float, Tuple[float, ...]] = 0.0
    passive_joint_stiffness: Union[float, Tuple[float, ...]] = 0.0
    passive_armature: Optional[Tuple[float, ...]] = None
    torque_update_mode: str = "physics"  # "physics" matches current run_mujoco_v1.py.

    max_episode_steps: int = 5000
    termination_height: float = 0.70
    death_cost: float = -5.0
    foot_contact_height: float = 0.055
    foot_clearance: float = 0.2

    alive_reward_scale: float = 1.0
    progress_reward_scale: float = 0.0
    torques_cost_scale: float = 5.0e-4
    dof_acc_cost_scale: float = 0.5
    dof_vel_cost_scale: float = 1.0
    dof_pos_cost_scale: float = 0.2
    no_fly_penalty_scale: float = 10.0
    vel_tracking_reward_scale: float = 6.0
    tracking_ang_vel_reward_scale: float = 2.0
    gait_similarity_penalty_scale: float = 10.0
    pelvis_height_reward_scale: float = 5.0
    orientation_reward_scale: float = 3.0
    clearance_penalty_scale: float = 50.0
    lateral_distance_penalty_scale: float = 30.0
    actions_rate_scale: float = 0.3
    actions_smoothness_scale: float = 0.6
    srl_motor_cost_scale: float = 0.0

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

    base_wobble_penalty_scale: float = 8.0
    base_ang_acc_penalty_scale: float = 0.001
    yaw_drift_penalty_scale: float = 1.0
    foot_impact_penalty_scale: float = 0.0
    foot_force_threshold_bw: float = 1.8
    foot_force_penalty_power: float = 2.0


class SRLMujocoWalkEnv:
    """No-virtual-human MuJoCo environment for SRL walking finetuning.

    This environment is intended to mirror run_mujoco_v1.py rather than the
    proxy-human/vir training setup in srl_mujoco_env.py.
    """

    obs_dim = 153
    act_dim = 6

    def __init__(self, config: Optional[WalkEnvConfig] = None):
        self.cfg = config or WalkEnvConfig()
        self.control_dt = self.cfg.dt * self.cfg.decimation

        self.default_dof_pos = np.asarray(self.cfg.default_dof_pos, dtype=np.float32)
        self.kp = np.asarray(self.cfg.kp, dtype=np.float32)
        self.kd = np.asarray(self.cfg.kd, dtype=np.float32)
        self.action_scale = np.asarray(self.cfg.action_scale, dtype=np.float32)
        self.max_torques = np.asarray(self.cfg.max_torques, dtype=np.float32)

        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.model.opt.timestep = self.cfg.dt
        self.model.jnt_stiffness[:] = _as_float_array(
            self.cfg.passive_joint_stiffness, self.model.njnt
        )
        self.model.dof_damping[:] = _as_float_array(
            self.cfg.passive_dof_damping, self.model.nv
        )
        if self.cfg.passive_armature is not None:
            self.model.dof_armature[6:] = np.asarray(self.cfg.passive_armature, dtype=np.float32)

        self.data = mujoco.MjData(self.model)
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.left_foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_contact")
        self.right_foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_contact")
        if min(self.floor_geom_id, self.left_foot_geom_id, self.right_foot_geom_id) < 0:
            raise RuntimeError("Missing floor/foot contact geom names required for foot impact reward.")
        self.body_weight = float(np.sum(self.model.body_mass) * 9.81)

        self.last_action = np.zeros(self.act_dim, dtype=np.float32)
        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.prev_prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.prev_torques = np.zeros(self.act_dim, dtype=np.float32)
        self.last_applied_torques = np.zeros(self.act_dim, dtype=np.float32)
        self.filtered_dof_vel = np.zeros(self.act_dim, dtype=np.float32)
        self.prev_local_ang_vel = np.zeros(3, dtype=np.float32)
        self.prev_srl_end_body_pos = np.zeros((2, 3), dtype=np.float32)
        self.target_point = np.array([self.cfg.target_point_x, 0.0, 0.0], dtype=np.float32)
        self.potential = 0.0
        self.prev_potential = 0.0
        self.srl_peak_ratio_window = 0.0
        self.srl_tau2_ema = 0.0
        self.last_left_foot_force_max = 0.0
        self.last_right_foot_force_max = 0.0
        self._srl_thermal_gamma = float(np.exp(-self.control_dt / self.cfg.srl_thermal_tau_s))
        self._srl_peak_decay = float(np.exp(-self.control_dt / self.cfg.srl_peak_window_tau_s))
        self.rl_step_counter = 0
        self.obs_history = collections.deque(maxlen=self.cfg.frame_stack)

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
        self.prev_action[:] = 0.0
        self.prev_prev_action[:] = 0.0
        self.prev_torques[:] = 0.0
        self.last_applied_torques[:] = 0.0
        self.filtered_dof_vel[:] = 0.0
        self.srl_peak_ratio_window = 0.0
        self.srl_tau2_ema = 0.0
        self.last_left_foot_force_max = 0.0
        self.last_right_foot_force_max = 0.0
        self.rl_step_counter = 0

        root_rot_mat = self.data.xmat[self.base_id].reshape(3, 3)
        self.prev_local_ang_vel[:] = root_rot_mat.T @ self.data.qvel[3:6]
        self.prev_srl_end_body_pos[:] = self._get_srl_end_body_pos()
        self.potential = self._compute_potential()
        self.prev_potential = self.potential

        initial_obs = self._get_single_frame_obs()
        self.obs_history.clear()
        for _ in range(self.cfg.frame_stack):
            self.obs_history.append(initial_obs.copy())

        return self._get_stacked_obs(append_current=False), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.cfg.clip_actions, self.cfg.clip_actions)

        filtered_action = (
            (1.0 - self.cfg.action_delay_alpha) * action
            + self.cfg.action_delay_alpha * self.last_action
        ).astype(np.float32)

        self.prev_prev_action[:] = self.prev_action
        self.prev_action[:] = self.last_action
        self.last_action[:] = filtered_action
        self.data.xfrc_applied[:] = 0.0

        target_pos = self.default_dof_pos + self.action_scale * filtered_action
        torque_rate_sum = 0.0
        left_foot_force_max = 0.0
        right_foot_force_max = 0.0

        if self.cfg.torque_update_mode == "control":
            torques = self._compute_torques(target_pos)
            torque_rate_sum += float(np.sum((torques - self.prev_torques) ** 2))
            self.prev_torques[:] = torques
            self.last_applied_torques[:] = torques
            self.data.ctrl[:] = torques / self.cfg.gear_ratio
            for _ in range(self.cfg.decimation):
                mujoco.mj_step(self.model, self.data)
                left_force, right_force = self._get_foot_contact_forces()
                left_foot_force_max = max(left_foot_force_max, left_force)
                right_foot_force_max = max(right_foot_force_max, right_force)
        elif self.cfg.torque_update_mode == "physics":
            for _ in range(self.cfg.decimation):
                torques = self._compute_torques(target_pos)
                torque_rate_sum += float(np.sum((torques - self.prev_torques) ** 2))
                self.prev_torques[:] = torques
                self.last_applied_torques[:] = torques
                self.data.ctrl[:] = torques / self.cfg.gear_ratio
                mujoco.mj_step(self.model, self.data)
                left_force, right_force = self._get_foot_contact_forces()
                left_foot_force_max = max(left_foot_force_max, left_force)
                right_foot_force_max = max(right_foot_force_max, right_force)
        else:
            raise ValueError(f"Unsupported torque_update_mode: {self.cfg.torque_update_mode}")

        self.last_left_foot_force_max = left_foot_force_max
        self.last_right_foot_force_max = right_foot_force_max

        self.rl_step_counter += 1
        obs = self._get_stacked_obs()
        reward, terminated, truncated, reward_info = self._compute_reward(
            obs,
            filtered_action,
            torque_rate_sum / max(self.cfg.decimation, 1),
        )

        info = self._get_info()
        info["raw_action"] = action.copy()
        info["filtered_action"] = filtered_action.copy()
        info["last_torques"] = self.last_applied_torques.copy()
        info.update(reward_info)
        return obs, reward, terminated, truncated, info

    def _compute_torques(self, target_pos):
        curr_pos = self.data.qpos[7:]
        curr_vel = self.data.qvel[6:]
        alpha = self.cfg.dof_vel_filter_alpha
        self.filtered_dof_vel[:] = (
            alpha * curr_vel + (1.0 - alpha) * self.filtered_dof_vel
        )

        torques_raw = self.kp * (target_pos - curr_pos) - self.kd * self.filtered_dof_vel

        if self.cfg.base_w_damp_x != 0.0 or self.cfg.base_w_damp_y != 0.0:
            root_rot_mat = self.data.xmat[self.base_id].reshape(3, 3)
            base_w = root_rot_mat.T @ self.data.qvel[3:6]
            torques_raw += np.array(
                [
                    -self.cfg.base_w_damp_x * base_w[0],
                    -self.cfg.base_w_damp_y * base_w[1],
                    0.0,
                    -self.cfg.base_w_damp_x * base_w[0],
                    -self.cfg.base_w_damp_y * base_w[1],
                    0.0,
                ],
                dtype=np.float32,
            )

        torques_raw = np.clip(torques_raw, -self.max_torques, self.max_torques)
        if self.cfg.max_torque_step > 0.0:
            delta_tau = torques_raw - self.prev_torques
            delta_tau = np.clip(delta_tau, -self.cfg.max_torque_step, self.cfg.max_torque_step)
            torques = self.prev_torques + delta_tau
        else:
            torques = torques_raw
        return np.clip(torques, -self.max_torques, self.max_torques).astype(np.float32)

    def _get_single_frame_obs(self):
        root_h = self.data.qpos[2]

        root_rot_mat = self.data.xmat[self.base_id].reshape(3, 3)
        local_lin_vel = root_rot_mat.T @ self.data.qvel[0:3]
        local_ang_vel = root_rot_mat.T @ self.data.qvel[3:6]

        euler = quat_to_euler_xyz(self.data.qpos[3:7])
        yaw_err = self.cfg.target_yaw - euler[0]
        yaw_err = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))
        euler_err = np.array([yaw_err, -euler[1], -euler[2]], dtype=np.float32)

        srl_dof_obs = self.data.qpos[7:] - self.default_dof_pos
        srl_dof_vel = self.data.qvel[6:] * 0.05

        phase_t = (2 * np.pi / self.cfg.gait_period) * (
            self.rl_step_counter % self.cfg.gait_period
        )
        obs = np.concatenate(
            [
                np.array([root_h], dtype=np.float32),
                local_lin_vel.astype(np.float32),
                local_ang_vel.astype(np.float32),
                euler_err.astype(np.float32),
                srl_dof_obs.astype(np.float32),
                srl_dof_vel.astype(np.float32),
                self.last_action.astype(np.float32),
                np.array([np.sin(phase_t), np.cos(phase_t)], dtype=np.float32),
            ]
        )
        return np.clip(obs, -self.cfg.clip_obs, self.cfg.clip_obs)

    def _get_stacked_obs(self, append_current=True):
        if append_current:
            self.obs_history.append(self._get_single_frame_obs())
        flat_history = np.concatenate(list(self.obs_history)[::-1])
        task_cmd = np.array(
            [self.cfg.target_vel_x, self.cfg.target_ang_vel_z, self.cfg.target_height],
            dtype=np.float32,
        )
        return np.concatenate([flat_history, task_cmd]).astype(np.float32)

    def _get_srl_end_body_pos(self):
        return np.stack(
            [
                self.data.xpos[self.left_foot_id].copy(),
                self.data.xpos[self.right_foot_id].copy(),
            ],
            axis=0,
        ).astype(np.float32)

    def _get_foot_contact_forces(self):
        left_force = 0.0
        right_force = 0.0
        force6 = np.zeros(6, dtype=np.float64)

        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 != self.floor_geom_id and geom2 != self.floor_geom_id:
                continue

            other_geom = geom2 if geom1 == self.floor_geom_id else geom1
            if other_geom != self.left_foot_geom_id and other_geom != self.right_foot_geom_id:
                continue

            mujoco.mj_contactForce(self.model, self.data, contact_idx, force6)
            normal_force = max(float(force6[0]), 0.0)
            if other_geom == self.left_foot_geom_id:
                left_force += normal_force
            else:
                right_force += normal_force

        return left_force, right_force

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
        clearance_penalty = float(np.sum((self.cfg.foot_clearance - pz) ** 2 * v_xy))
        return clearance_penalty, curr

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

        srl_motor_cost = (
            self.cfg.srl_peak_cost_scale * peak_cost
            + self.cfg.srl_thermal_cost_scale * thermal_cost
            + self.cfg.srl_power_cost_scale * power_cost
        )
        return srl_motor_cost, peak_cost, thermal_cost, power_cost

    def _compute_reward(self, obs, action, torque_rate):
        root_h = float(obs[0])
        local_vel = obs[1:4]
        local_ang_vel = obs[4:7]
        euler_err = obs[7:10]
        srl_dof_pos = obs[10:16].copy()
        srl_dof_vel_scaled = obs[16:22]

        target_vel_x = float(self.cfg.target_vel_x)
        target_ang_vel_z = float(self.cfg.target_ang_vel_z)
        target_pelvis_height = float(self.cfg.target_height)
        warmup = float(np.clip(self.rl_step_counter / 10.0, 0.0, 1.0))

        alive_reward = 4.0 if target_vel_x < 0.1 else 1.0
        current_potential = self._compute_potential()
        progress_reward = 0.0 if target_vel_x < 0.1 else current_potential - self.prev_potential
        progress_reward *= warmup
        self.prev_potential = current_potential

        target_vel = np.array([target_vel_x, 0.0, 0.0], dtype=np.float32)
        vel_error_vec = local_vel - target_vel
        vel_tracking_reward = float(np.exp(-4.0 * np.linalg.norm(vel_error_vec)))

        torques_cost = float(np.sum(action ** 2))

        srl_dof_pos[0] *= 3.0
        srl_dof_pos[3] *= 3.0
        dof_pos_cost = float(np.sum(srl_dof_pos ** 2))

        dof_vel_cost = float(np.sum(srl_dof_vel_scaled ** 2))

        frame = 30
        act_dim = self.act_dim
        dof_vel_prev_raw = obs[16 + frame:16 + frame + act_dim]
        dof_vel_prev = warmup * dof_vel_prev_raw + (1.0 - warmup) * srl_dof_vel_scaled
        dof_acc = srl_dof_vel_scaled - dof_vel_prev
        dof_acc_reward_raw = float(np.exp(-2.0 * np.sum(dof_acc ** 2)))
        dof_acc_reward = warmup * dof_acc_reward_raw + (1.0 - warmup)

        actions_prev_raw = obs[22 + frame:22 + frame + act_dim]
        actions_prev_prev_raw = obs[22 + 2 * frame:22 + 2 * frame + act_dim]
        actions_prev = warmup * actions_prev_raw + (1.0 - warmup) * action
        actions_prev_prev = warmup * actions_prev_prev_raw + (1.0 - warmup) * actions_prev
        actions_rate = warmup * float(np.sum((action - actions_prev) ** 2))
        actions_smoothness = warmup * float(np.sum((action - 2.0 * actions_prev + actions_prev_prev) ** 2))

        angle_diff = ((euler_err + np.pi) % (2.0 * np.pi)) - np.pi
        yaw = float(angle_diff[0])
        pitch_err = float(angle_diff[1])
        roll_err = float(angle_diff[2])
        ori_cost = 0.2 * yaw * yaw + pitch_err * pitch_err + roll_err * roll_err
        orientation_reward = float(np.exp(-8.0 * ori_cost))

        pelvis_height_error = root_h - target_pelvis_height
        pelvis_height_reward = float(np.exp(-6.0 * (3.0 * pelvis_height_error) ** 2))

        wx, wy, wz = map(float, local_ang_vel)
        wz_err = wz - target_ang_vel_z
        ang_vel_cost = 3.0 * (wx * wx + wy * wy) + 0.5 * (wz_err * wz_err)
        ang_vel_tracking_reward = float(np.exp(-2.0 * ang_vel_cost))
        yaw_drift_penalty = warmup * float(wz_err * wz_err)

        clearance_penalty, srl_end_body_pos = self._compute_clearance_penalty()
        clearance_penalty *= warmup

        left_foot_height = float(srl_end_body_pos[0, 2])
        right_foot_height = float(srl_end_body_pos[1, 2])
        no_feet_on_ground = (
            left_foot_height > self.cfg.foot_contact_height
            and right_foot_height > self.cfg.foot_contact_height
        )
        no_fly_coef = 5.0 if target_vel_x < 0.1 else 1.0
        no_fly_penalty = (
            self.cfg.no_fly_penalty_scale * no_fly_coef if no_feet_on_ground else 0.0
        )
        no_fly_penalty *= warmup

        srl_root_pos = self.data.qpos[0:3].copy().astype(np.float32)
        local_srl_end_body_pos = srl_end_body_pos - srl_root_pos[None, :]
        lateral_distance = abs(float(local_srl_end_body_pos[0, 1] - local_srl_end_body_pos[1, 1]))
        below_violation = max(0.25 - lateral_distance, 0.0)
        above_violation = max(lateral_distance - 0.85, 0.0)
        feet_lateral_penalty = (below_violation + above_violation) * warmup

        phase_t = (2.0 * np.pi / self.cfg.gait_period) * float(self.rl_step_counter % self.cfg.gait_period)
        phase_left = phase_t
        phase_right = (phase_t + np.pi) % (2.0 * np.pi)
        expect_stancing_left = 1.0 if np.sin(phase_left) > -0.2 else 0.0
        expect_stancing_right = 1.0 if np.sin(phase_right) > -0.2 else 0.0
        expect_flying_left = 1.0 if np.sin(phase_left) < -0.7 else 0.0
        expect_flying_right = 1.0 if np.sin(phase_right) < -0.7 else 0.0
        is_contact_left = 1.0 if left_foot_height < self.cfg.foot_contact_height else 0.0
        is_contact_right = 1.0 if right_foot_height < self.cfg.foot_contact_height else 0.0
        stance_miss_left = expect_stancing_left * (1.0 - is_contact_left)
        stance_miss_right = expect_stancing_right * (1.0 - is_contact_right)
        flying_miss_left = expect_flying_left * is_contact_left
        flying_miss_right = expect_flying_right * is_contact_right
        gait_phase_penalty = self.cfg.gait_similarity_penalty_scale * (
            stance_miss_left + stance_miss_right + flying_miss_left + flying_miss_right
        )
        gait_phase_penalty *= warmup

        srl_motor_cost, peak_cost, thermal_cost, power_cost = self._compute_srl_motor_costs()

        root_rot_mat = self.data.xmat[self.base_id].reshape(3, 3)
        current_local_ang_vel = root_rot_mat.T @ self.data.qvel[3:6]
        base_ang_acc = (current_local_ang_vel - self.prev_local_ang_vel) / self.control_dt
        self.prev_local_ang_vel[:] = current_local_ang_vel
        base_ang_acc_penalty = float(base_ang_acc[0] ** 2 + base_ang_acc[1] ** 2)
        base_wobble_penalty = pitch_err * pitch_err + roll_err * roll_err + 0.5 * (wx * wx + wy * wy)
        left_foot_force_bw = self.last_left_foot_force_max / max(self.body_weight, 1e-6)
        right_foot_force_bw = self.last_right_foot_force_max / max(self.body_weight, 1e-6)
        left_foot_force_excess = max(left_foot_force_bw - self.cfg.foot_force_threshold_bw, 0.0)
        right_foot_force_excess = max(right_foot_force_bw - self.cfg.foot_force_threshold_bw, 0.0)
        foot_impact_penalty = warmup * float(
            left_foot_force_excess ** self.cfg.foot_force_penalty_power
            + right_foot_force_excess ** self.cfg.foot_force_penalty_power
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
            - no_fly_penalty
            - gait_phase_penalty
            - self.cfg.clearance_penalty_scale * clearance_penalty
            - self.cfg.lateral_distance_penalty_scale * feet_lateral_penalty
            - self.cfg.srl_motor_cost_scale * srl_motor_cost
            - self.cfg.base_wobble_penalty_scale * base_wobble_penalty
            - self.cfg.base_ang_acc_penalty_scale * base_ang_acc_penalty
            - self.cfg.yaw_drift_penalty_scale * yaw_drift_penalty
            - self.cfg.foot_impact_penalty_scale * foot_impact_penalty
        )

        terminated = bool(root_h < self.cfg.termination_height)
        truncated = bool(self.rl_step_counter >= self.cfg.max_episode_steps)
        if terminated:
            reward = self.cfg.death_cost

        info = {
            "reward_total": float(reward),
            "reward_alive": float(alive_reward),
            "reward_progress": float(progress_reward),
            "reward_vel_tracking": vel_tracking_reward,
            "reward_ang_vel_tracking": ang_vel_tracking_reward,
            "reward_orientation": orientation_reward,
            "reward_pelvis_height": pelvis_height_reward,
            "reward_height": pelvis_height_reward,
            "reward_dof_acc": dof_acc_reward,
            "penalty_torques": torques_cost,
            "penalty_dof_pos": dof_pos_cost,
            "penalty_dof_vel": dof_vel_cost,
            "penalty_actions_rate": actions_rate,
            "penalty_actions_smoothness": actions_smoothness,
            "penalty_no_fly": no_fly_penalty,
            "penalty_gait_phase": gait_phase_penalty,
            "penalty_clearance": clearance_penalty,
            "penalty_lateral": feet_lateral_penalty,
            "penalty_srl_motor": srl_motor_cost,
            "penalty_srl_motor_peak": peak_cost,
            "penalty_srl_motor_thermal": thermal_cost,
            "penalty_srl_motor_power": power_cost,
            "penalty_base_wobble": base_wobble_penalty,
            "penalty_base_ang_acc": base_ang_acc_penalty,
            "penalty_yaw_drift": yaw_drift_penalty,
            "penalty_foot_impact": foot_impact_penalty,
            "left_foot_force_max": float(self.last_left_foot_force_max),
            "right_foot_force_max": float(self.last_right_foot_force_max),
            "left_foot_force_bw": float(left_foot_force_bw),
            "right_foot_force_bw": float(right_foot_force_bw),
            "yaw": yaw,
            "pitch": -pitch_err,
            "roll": -roll_err,
            "wx": wx,
            "wy": wy,
            "wz": wz,
            "vel_x": float(local_vel[0]),
            "root_height": root_h,
            "penalty_torque_rate_debug": float(torque_rate),
        }
        return float(reward), terminated, truncated, info

    def _get_info(self):
        root_rot_mat = self.data.xmat[self.base_id].reshape(3, 3)
        local_vel = root_rot_mat.T @ self.data.qvel[0:3]
        local_ang_vel = root_rot_mat.T @ self.data.qvel[3:6]
        euler = quat_to_euler_xyz(self.data.qpos[3:7])
        return {
            "step": self.rl_step_counter,
            "root_height": float(self.data.qpos[2]),
            "target_vel_x": float(self.cfg.target_vel_x),
            "vel_x": float(local_vel[0]),
            "yaw": float(euler[0]),
            "pitch": float(euler[1]),
            "roll": float(euler[2]),
            "wx": float(local_ang_vel[0]),
            "wy": float(local_ang_vel[1]),
            "wz": float(local_ang_vel[2]),
        }


# Compatibility names for training scripts that expect EnvConfig.
EnvConfig = WalkEnvConfig
