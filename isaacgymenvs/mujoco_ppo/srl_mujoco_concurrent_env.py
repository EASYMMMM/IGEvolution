from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import mujoco
import numpy as np

from mujoco_ppo.srl_mujoco_v3_env import SRLMujocoV3Env, V3WalkEnvConfig


class JointwiseSecondOrderLowPass:
    """Second-order Butterworth filter with one cutoff per policy joint."""

    def __init__(self, cutoff_hz, sample_dt, initial_value):
        self.sample_dt = float(sample_dt)
        self.configure(cutoff_hz)
        self.reset(initial_value)

    def configure(self, cutoff_hz):
        cutoff = np.asarray(cutoff_hz, dtype=np.float64)
        if cutoff.ndim != 1 or np.any(cutoff <= 0.0):
            raise ValueError("joint filter cutoffs must be a positive vector")
        sample_freq = 1.0 / self.sample_dt
        cutoff = np.minimum(cutoff, 0.499 * sample_freq)
        k = np.tan(math.pi * cutoff / sample_freq)
        norm = 1.0 / (1.0 + math.sqrt(2.0) * k + k * k)
        self.b0 = k * k * norm
        self.b1 = 2.0 * self.b0
        self.b2 = self.b0.copy()
        self.a1 = 2.0 * (k * k - 1.0) * norm
        self.a2 = (1.0 - math.sqrt(2.0) * k + k * k) * norm
        self.cutoff_hz = cutoff.astype(np.float32)

    def reset(self, value):
        value = np.asarray(value, dtype=np.float32).copy()
        if value.shape != self.cutoff_hz.shape:
            raise ValueError(
                f"filter state shape {value.shape} does not match cutoffs "
                f"{self.cutoff_hz.shape}"
            )
        self.x1 = value.copy()
        self.x2 = value.copy()
        self.y1 = value.copy()
        self.y2 = value.copy()

    def apply(self, value):
        value = np.asarray(value, dtype=np.float32)
        output = (
            self.b0 * value
            + self.b1 * self.x1
            + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2
        )
        self.x2[:] = self.x1
        self.x1[:] = value
        self.y2[:] = self.y1
        self.y1[:] = output
        return output.astype(np.float32)

    def overwrite_last_output(self, value):
        self.y1[:] = np.asarray(value, dtype=np.float32)


@dataclass
class ConcurrentWalkEnvConfig(V3WalkEnvConfig):
    """Concurrent-only extensions to the shared V3 walking environment."""

    initial_pose_randomization_enable: bool = False
    initial_pose_root_height_range: tuple[float, float] = (0.85, 1.14)
    initial_pose_ground_clearance: float = 0.005
    initial_pose_low_hip_y_offset: float = 0.165
    initial_pose_low_knee_offset: float = 0.285
    initial_pose_high_hip_y_offset: float = 0.035
    initial_pose_high_knee_offset: float = -0.305
    initial_pose_hip_y_asymmetry_max: float = 0.02
    initial_pose_knee_asymmetry_max: float = 0.02
    initial_pose_foot_height_tolerance: float = 0.015
    initial_pose_joint_limit_margin: float = 0.01
    initial_pose_max_attempts: int = 32
    startup_support_hold_until_unloaded: bool = False
    startup_support_post_unload_hold_time: float = 0.0
    foot_lateral_velocity_penalty_scale: float = 0.0
    foot_lateral_velocity_deadband: float = 0.25
    foot_lateral_channel_penalty_scale: float = 0.0
    foot_lateral_channel_center: float = 0.17
    foot_lateral_channel_half_width: float = 0.07
    foot_lateral_channel_turn_relaxation: float = 0.08
    foot_lateral_channel_recovery_relaxation: float = 0.08
    foot_lateral_channel_max_relaxation: float = 0.12
    action_filter_jointwise_enable: bool = False
    action_filter_hip_x_cutoff_hz_range: tuple[float, float] = (8.0, 10.0)
    action_filter_hip_y_cutoff_hz_range: tuple[float, float] = (5.0, 7.0)
    action_filter_knee_cutoff_hz_range: tuple[float, float] = (6.0, 9.0)
    hip_x_velocity_penalty_scale: float = 0.0
    hip_x_velocity_deadband: float = 0.2
    hip_x_velocity_recovery_roll_threshold: float = 0.08
    hip_x_velocity_recovery_rate_threshold: float = 0.4
    hip_x_velocity_recovery_scale: float = 0.25


class SRLMujocoConcurrentEnv(SRLMujocoV3Env):
    """V3 walking task with the supervision required by the 137-D actor.

    The public observation remains the deployable 133-D observation.  The
    training-only estimator history and target are exposed through ``info``.
    """

    estimator_frame_dim = 26
    estimator_history_len = 10
    estimator_target_dim = 4
    concurrent_actor_obs_dim = 137

    def __init__(self, cfg: ConcurrentWalkEnvConfig, estimator_history_len: int = 10):
        self.estimator_history_len = int(estimator_history_len)
        if self.estimator_history_len <= 0:
            raise ValueError("estimator_history_len must be positive")
        if cfg.startup_support_post_unload_hold_time < 0.0:
            raise ValueError("startup_support_post_unload_hold_time must be >= 0")
        height_range = tuple(float(value) for value in cfg.initial_pose_root_height_range)
        if len(height_range) != 2 or height_range[0] <= 0.0 or height_range[0] > height_range[1]:
            raise ValueError("initial_pose_root_height_range must be an increasing positive pair")
        for name in (
            "initial_pose_hip_y_asymmetry_max",
            "initial_pose_knee_asymmetry_max",
            "initial_pose_ground_clearance",
            "initial_pose_foot_height_tolerance",
            "initial_pose_joint_limit_margin",
        ):
            if float(getattr(cfg, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if int(cfg.initial_pose_max_attempts) <= 0:
            raise ValueError("initial_pose_max_attempts must be positive")
        for name in (
            "hip_x_velocity_penalty_scale",
            "hip_x_velocity_deadband",
            "hip_x_velocity_recovery_roll_threshold",
            "hip_x_velocity_recovery_rate_threshold",
        ):
            if float(getattr(cfg, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 <= float(cfg.hip_x_velocity_recovery_scale) <= 1.0:
            raise ValueError("hip_x_velocity_recovery_scale must be in [0, 1]")
        self.estimator_history = deque(maxlen=self.estimator_history_len)
        super().__init__(cfg)
        self.action_filter_cutoff_hz_by_joint = self._nominal_filter_cutoffs()
        if self.cfg.action_filter_jointwise_enable:
            self.target_filter = JointwiseSecondOrderLowPass(
                self.action_filter_cutoff_hz_by_joint,
                self.control_dt,
                self.default_dof_pos,
            )
        self.initial_pose_dof_pos = self.default_dof_pos.copy()
        self.initial_pose_offsets = np.zeros(self.act_dim, dtype=np.float32)
        self.initial_pose_root_height = float(self.cfg.root_height)
        self.initial_pose_target_root_height = float(self.cfg.root_height)
        self.initial_pose_foot_clearances = np.zeros(2, dtype=np.float32)
        self.initial_pose_randomization_accepted = False

    @staticmethod
    def _validate_cutoff_range(name, value_range):
        if len(value_range) != 2:
            raise ValueError(f"{name} must contain two values")
        low, high = map(float, value_range)
        if low <= 0.0 or high < low:
            raise ValueError(f"{name} must be a positive increasing pair")
        return low, high

    def _nominal_filter_cutoffs(self):
        if not self.cfg.action_filter_jointwise_enable:
            return np.full(
                self.act_dim, float(self.cfg.action_filter_cutoff_hz),
                dtype=np.float32,
            )
        ranges = (
            self._validate_cutoff_range(
                "action_filter_hip_x_cutoff_hz_range",
                self.cfg.action_filter_hip_x_cutoff_hz_range,
            ),
            self._validate_cutoff_range(
                "action_filter_hip_y_cutoff_hz_range",
                self.cfg.action_filter_hip_y_cutoff_hz_range,
            ),
            self._validate_cutoff_range(
                "action_filter_knee_cutoff_hz_range",
                self.cfg.action_filter_knee_cutoff_hz_range,
            ),
        )
        centers = [0.5 * (low + high) for low, high in ranges]
        return np.asarray(centers + centers, dtype=np.float32)

    def _sample_filter_cutoffs(self):
        nominal = self._nominal_filter_cutoffs()
        if not self.cfg.action_filter_jointwise_enable:
            return nominal
        progress = (
            float(np.clip(self.episode_dr_progress, 0.0, 1.0))
            if self.cfg.domain_randomization_enable else 0.0
        )
        sampled = nominal.copy()
        groups = (
            ((0, 3), self.cfg.action_filter_hip_x_cutoff_hz_range),
            ((1, 4), self.cfg.action_filter_hip_y_cutoff_hz_range),
            ((2, 5), self.cfg.action_filter_knee_cutoff_hz_range),
        )
        for indices, value_range in groups:
            low, high = map(float, value_range)
            center = 0.5 * (low + high)
            scaled_low = center + progress * (low - center)
            scaled_high = center + progress * (high - center)
            sampled[list(indices)] = self.rng.uniform(
                scaled_low, scaled_high, size=len(indices)
            )
        return sampled

    def _reset_randomized_params(self):
        super()._reset_randomized_params()
        cutoffs = self._sample_filter_cutoffs()
        self.action_filter_cutoff_hz_by_joint = cutoffs.astype(
            np.float32, copy=True
        )
        if self.cfg.action_filter_jointwise_enable:
            self.target_filter.configure(cutoffs)
        for label, indices in (
            ("hip_x", (0, 3)), ("hip_y", (1, 4)), ("knee", (2, 5))
        ):
            values = cutoffs[list(indices)]
            self.dr_sampled_params[f"action_filter_{label}_cutoff_hz_min"] = (
                float(np.min(values))
            )
            self.dr_sampled_params[f"action_filter_{label}_cutoff_hz_max"] = (
                float(np.max(values))
            )

    def _foot_ground_distances(self):
        return np.asarray(
            [
                mujoco.mj_geomDistance(
                    self.model, self.data, geom_id, self.floor_geom_id, 1.0, None
                )
                for geom_id in (self.left_foot_geom_id, self.right_foot_geom_id)
            ],
            dtype=np.float64,
        )

    def _contact_root_height(self, baseline_qpos, pose):
        self.data.qpos[:] = baseline_qpos
        self.data.qpos[self.srl_qpos_indices] = pose
        mujoco.mj_forward(self.model, self.data)
        clearance = float(self.cfg.initial_pose_ground_clearance)
        return float(self.data.qpos[2] + clearance - np.min(self._foot_ground_distances()))

    def _pose_for_root_height(self, baseline_qpos, target_height):
        default = self.default_dof_pos.astype(np.float64)
        default_height = self._contact_root_height(baseline_qpos, default)
        if target_height <= default_height:
            endpoint = default.copy()
            endpoint[[1, 4]] += float(self.cfg.initial_pose_low_hip_y_offset)
            endpoint[[2, 5]] += float(self.cfg.initial_pose_low_knee_offset)
        else:
            endpoint = default.copy()
            endpoint[[1, 4]] += float(self.cfg.initial_pose_high_hip_y_offset)
            endpoint[[2, 5]] += float(self.cfg.initial_pose_high_knee_offset)
        endpoint = np.clip(
            endpoint,
            self.joint_low + self.cfg.initial_pose_joint_limit_margin,
            self.joint_high - self.cfg.initial_pose_joint_limit_margin,
        )
        endpoint_height = self._contact_root_height(baseline_qpos, endpoint)
        lower = min(default_height, endpoint_height)
        upper = max(default_height, endpoint_height)
        if not lower <= target_height <= upper:
            return None

        lo, hi = 0.0, 1.0
        increasing = endpoint_height > default_height
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            pose = default + mid * (endpoint - default)
            height = self._contact_root_height(baseline_qpos, pose)
            if (height < target_height) == increasing:
                lo = mid
            else:
                hi = mid
        return default + 0.5 * (lo + hi) * (endpoint - default)

    def _randomize_initial_pose(self):
        self.initial_pose_dof_pos = self.default_dof_pos.copy()
        self.initial_pose_offsets.fill(0.0)
        self.initial_pose_root_height = float(self.data.qpos[2])
        self.initial_pose_target_root_height = float(self.data.qpos[2])
        self.initial_pose_foot_clearances.fill(0.0)
        self.initial_pose_randomization_accepted = False
        if not self.cfg.initial_pose_randomization_enable:
            return False

        baseline_qpos = self.data.qpos.copy()
        progress = (
            float(np.clip(self.episode_dr_progress, 0.0, 1.0))
            if self.cfg.domain_randomization_enable else 0.0
        )
        margin = float(self.cfg.initial_pose_joint_limit_margin)
        tolerance = float(self.cfg.initial_pose_foot_height_tolerance)
        default_height = self._contact_root_height(baseline_qpos, self.default_dof_pos)
        configured_low, configured_high = self.cfg.initial_pose_root_height_range
        low = default_height + progress * (float(configured_low) - default_height)
        high = default_height + progress * (float(configured_high) - default_height)

        for _ in range(int(self.cfg.initial_pose_max_attempts)):
            target_height = float(self.rng.uniform(low, high))
            symmetric_pose = self._pose_for_root_height(
                baseline_qpos, target_height
            )
            if symmetric_pose is None:
                continue
            asymmetry = float(self.rng.uniform(-1.0, 1.0)) * progress
            hip_asymmetry = asymmetry * self.cfg.initial_pose_hip_y_asymmetry_max
            knee_asymmetry = asymmetry * self.cfg.initial_pose_knee_asymmetry_max
            candidate = symmetric_pose.copy()
            candidate[[1, 4]] += (hip_asymmetry, -hip_asymmetry)
            candidate[[2, 5]] += (knee_asymmetry, -knee_asymmetry)
            if np.any(candidate < self.joint_low + margin):
                continue
            if np.any(candidate > self.joint_high - margin):
                continue

            self.data.qpos[:] = baseline_qpos
            self.data.qpos[self.srl_qpos_indices] = candidate
            self.data.qpos[2] = target_height
            mujoco.mj_forward(self.model, self.data)
            distances = self._foot_ground_distances()
            self.data.qpos[2] += float(
                self.cfg.initial_pose_ground_clearance - np.min(distances)
            )
            mujoco.mj_forward(self.model, self.data)
            clearances = self._foot_ground_distances()
            if np.max(clearances) - np.min(clearances) > tolerance:
                continue
            range_epsilon = 1e-6
            if not (
                low - range_epsilon
                <= float(self.data.qpos[2])
                <= high + range_epsilon
            ):
                continue

            self.initial_pose_dof_pos = candidate.astype(np.float32, copy=True)
            self.initial_pose_offsets = (
                self.initial_pose_dof_pos - self.default_dof_pos
            )
            self.initial_pose_root_height = float(self.data.qpos[2])
            self.initial_pose_target_root_height = target_height
            self.initial_pose_foot_clearances = clearances.astype(np.float32)
            self.initial_pose_randomization_accepted = True
            break
        else:
            self.data.qpos[:] = baseline_qpos
            mujoco.mj_forward(self.model, self.data)

        if not self.initial_pose_randomization_accepted:
            return False

        initial_pose = self.initial_pose_dof_pos
        self.raw_target_pos[:] = initial_pose
        self.target_pos[:] = initial_pose
        self.previous_cycle_target_pos[:] = initial_pose
        self.target_filter.reset(initial_pose)
        self._reset_action_delay_queues(target_pos=initial_pose)
        self.prev_local_ang_vel[:] = self.data.qvel[3:6]
        self.prev_srl_end_body_pos[:] = self._get_srl_end_body_pos()
        self.potential = self._compute_potential()
        self.prev_potential = self.potential
        return True

    def _rebuild_initial_observation(self):
        initial_clean_obs = self._get_clean_single_frame_obs()
        initial_actor_obs = self._add_actor_obs_noise(initial_clean_obs)
        self.obs_history.clear()
        self.clean_obs_history.clear()
        for _ in range(self.cfg.frame_stack):
            self.obs_history.append(initial_actor_obs.copy())
            self.clean_obs_history.append(initial_clean_obs.copy())

        full_noisy_obs = self._get_stacked_obs(self.obs_history)
        actor_obs = self._actor_obs_from_full(full_noisy_obs)
        info = self._get_info()
        info["startup_command_scale"] = self._startup_command_scale(0)
        info["critic_obs"] = self.get_critic_obs()
        info["actor_obs_mirrored"] = self.mirror_actor_obs(actor_obs)
        self._add_startup_support_info(info)
        return actor_obs, info

    def _support_command_scale(self, policy_step: int) -> float:
        if not (
            self.cfg.startup_support_hold_until_unloaded
            and self.startup_support_mode != "none"
        ):
            return super()._support_command_scale(policy_step)

        elapsed = max(int(policy_step), 0) * self.control_dt
        release_time = (
            self.startup_support_unload_time
            + self.cfg.startup_support_post_unload_hold_time
        )
        if elapsed < release_time:
            return 0.0

        ramp_time = float(self.cfg.startup_support_command_ramp_time)
        if ramp_time <= 0.0:
            return 1.0
        return float(np.clip((elapsed - release_time) / ramp_time, 0.0, 1.0))

    @staticmethod
    def _yaw_rotation(root_rotation):
        yaw = float(np.arctan2(root_rotation[1, 0], root_rotation[0, 0]))
        cosine, sine = np.cos(yaw), np.sin(yaw)
        return np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _foot_lateral_metrics(self):
        foot_positions = self._get_srl_end_body_pos()
        root_position = self.data.xpos[self.base_id]
        root_rotation = self.data.xmat[self.base_id].reshape(3, 3)
        yaw_rotation = self._yaw_rotation(root_rotation)
        yaw_local_positions = (foot_positions - root_position) @ yaw_rotation

        velocities = []
        root_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
            self.base_id, root_velocity, 0,
        )
        for body_id, foot_position in zip(
            (self.left_foot_id, self.right_foot_id), foot_positions
        ):
            foot_velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                body_id, foot_velocity, 0,
            )
            arm = foot_position - root_position
            relative_world = (
                foot_velocity[3:] - root_velocity[3:]
                - np.cross(root_velocity[:3], arm)
            )
            velocities.append(relative_world @ yaw_rotation)

        lateral_speeds = [abs(float(velocity[1])) for velocity in velocities]
        contacts = [
            float(foot_positions[0, 2] < self.cfg.foot_contact_height),
            float(foot_positions[1, 2] < self.cfg.foot_contact_height),
        ]
        excess = [
            max(speed - self.cfg.foot_lateral_velocity_deadband, 0.0)
            for speed in lateral_speeds
        ]
        penalty = sum(
            (1.0 - contact) * value * value
            for contact, value in zip(contacts, excess)
        )

        root_yaw_velocity = root_velocity[3:] @ yaw_rotation
        turn_relaxation = (
            self.cfg.foot_lateral_channel_turn_relaxation
            * abs(float(self.cfg.target_ang_vel_z))
        )
        recovery_relaxation = (
            self.cfg.foot_lateral_channel_recovery_relaxation
            * abs(float(root_yaw_velocity[1]))
        )
        relaxation = min(
            turn_relaxation + recovery_relaxation,
            self.cfg.foot_lateral_channel_max_relaxation,
        )
        half_width = self.cfg.foot_lateral_channel_half_width + relaxation
        lateral_y = [float(yaw_local_positions[0, 1]), float(yaw_local_positions[1, 1])]
        centers = [
            self.cfg.foot_lateral_channel_center,
            -self.cfg.foot_lateral_channel_center,
        ]
        channel_excess = [
            max(abs(position - center) - half_width, 0.0)
            for position, center in zip(lateral_y, centers)
        ]
        channel_penalty = sum(
            (1.0 - contact) * value * value
            for contact, value in zip(contacts, channel_excess)
        )
        return {
            "lateral_speeds": lateral_speeds,
            "velocity_penalty": float(penalty),
            "yaw_local_y": lateral_y,
            "channel_half_width": float(half_width),
            "channel_penalty": float(channel_penalty),
            "channel_violation": float(any(value > 0.0 for value in channel_excess)),
        }

    def _compute_reward(self, obs, action):
        reward, terminated, truncated, info = super()._compute_reward(obs, action)
        metrics = self._foot_lateral_metrics()
        hip_x_velocity = self._get_srl_qvel()[[0, 3]]
        hip_x_excess = np.maximum(
            np.abs(hip_x_velocity) - self.cfg.hip_x_velocity_deadband, 0.0
        )
        roll_recovery = (
            abs(float(info.get("roll", 0.0)))
            >= self.cfg.hip_x_velocity_recovery_roll_threshold
            or abs(float(obs[4]))
            >= self.cfg.hip_x_velocity_recovery_rate_threshold
            or np.any(np.abs(self.horizontal_force_pulse) > 0.0)
        )
        recovery_scale = (
            self.cfg.hip_x_velocity_recovery_scale if roll_recovery else 1.0
        )
        hip_x_velocity_penalty = float(
            recovery_scale * np.sum(hip_x_excess * hip_x_excess)
        )
        reward -= (
            self.cfg.foot_lateral_velocity_penalty_scale
            * metrics["velocity_penalty"]
            + self.cfg.foot_lateral_channel_penalty_scale
            * metrics["channel_penalty"]
            + self.cfg.hip_x_velocity_penalty_scale
            * hip_x_velocity_penalty
        )
        info.update({
            "reward_total": float(reward),
            "left_foot_lateral_speed": metrics["lateral_speeds"][0],
            "right_foot_lateral_speed": metrics["lateral_speeds"][1],
            "penalty_foot_lateral_velocity": metrics["velocity_penalty"],
            "left_foot_yaw_local_y": metrics["yaw_local_y"][0],
            "right_foot_yaw_local_y": metrics["yaw_local_y"][1],
            "foot_lateral_channel_half_width": metrics["channel_half_width"],
            "penalty_foot_lateral_channel": metrics["channel_penalty"],
            "foot_lateral_channel_violation": metrics["channel_violation"],
            "left_hip_x_velocity": float(hip_x_velocity[0]),
            "right_hip_x_velocity": float(hip_x_velocity[1]),
            "penalty_hip_x_velocity": hip_x_velocity_penalty,
            "hip_x_velocity_recovery_active": float(roll_recovery),
            "hip_x_velocity_recovery_scale": float(recovery_scale),
            "action_filter_cutoff_hz_by_joint": (
                self.action_filter_cutoff_hz_by_joint.copy()
            ),
        })
        return float(reward), terminated, truncated, info

    @staticmethod
    def _estimator_frame(actor_obs: np.ndarray) -> np.ndarray:
        frame = np.asarray(actor_obs[:26], dtype=np.float32)
        if frame.shape != (26,):
            raise ValueError(f"expected a 133-D deployable observation, got {actor_obs.shape}")
        return frame.copy()

    def _estimator_target(self) -> np.ndarray:
        # Current clean frame: root height and body-frame vx/vy/vz.
        return np.asarray(self.clean_obs_history[-1][:4], dtype=np.float32).copy()

    def _add_concurrent_info(self, info, actor_obs):
        result = dict(info)
        result["estimator_history"] = np.asarray(
            self.estimator_history, dtype=np.float32
        ).copy()
        result["estimator_target"] = self._estimator_target()
        result["deployable_obs"] = np.asarray(actor_obs, dtype=np.float32).copy()
        result["initial_pose_dof_pos"] = self.initial_pose_dof_pos.copy()
        result["initial_pose_offsets"] = self.initial_pose_offsets.copy()
        result["initial_pose_root_height"] = float(self.initial_pose_root_height)
        result["initial_pose_target_root_height"] = float(
            self.initial_pose_target_root_height
        )
        result["initial_pose_foot_clearances"] = (
            self.initial_pose_foot_clearances.copy()
        )
        result["initial_pose_randomization_accepted"] = bool(
            self.initial_pose_randomization_accepted
        )
        result["action_filter_cutoff_hz_by_joint"] = (
            self.action_filter_cutoff_hz_by_joint.copy()
        )
        return result

    def reset(self, seed=None):
        actor_obs, info = super().reset(seed=seed)
        if self._randomize_initial_pose():
            self._reset_startup_tether()
            self._reset_startup_no_retreat()
            actor_obs, info = self._rebuild_initial_observation()
        frame = self._estimator_frame(actor_obs)
        self.estimator_history.clear()
        self.estimator_history.extend(
            frame.copy() for _ in range(self.estimator_history_len)
        )
        return actor_obs, self._add_concurrent_info(info, actor_obs)

    def step(self, action):
        actor_obs, reward, terminated, truncated, info = super().step(action)
        self.estimator_history.append(self._estimator_frame(actor_obs))
        return (
            actor_obs,
            reward,
            terminated,
            truncated,
            self._add_concurrent_info(info, actor_obs),
        )
