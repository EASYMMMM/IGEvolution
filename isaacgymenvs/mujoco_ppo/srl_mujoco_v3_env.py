from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from mujoco_ppo.srl_mujoco_v2_env import SRLMujocoV2Env, V2WalkEnvConfig


class IsaacTaskCommandGenerator:
    """Stateful Stage 0-3 command generator shared by training and viewers."""

    def __init__(
        self,
        stage=0,
        control_dt=0.015,
        target_vel_x=1.0,
        target_ang_vel_z=0.0,
        target_height=1.0,
        target_yaw=0.0,
        velocity_change_period=600,
        height_change_period=430,
        turn_change_period=750,
        turn_duration_steps=60,
    ):
        self.stage = int(stage)
        self.control_dt = float(control_dt)
        self.initial = (
            float(target_vel_x),
            float(target_ang_vel_z),
            float(target_height),
            float(target_yaw),
        )
        self.velocity_change_period = int(velocity_change_period)
        self.height_change_period = int(height_change_period)
        self.turn_change_period = int(turn_change_period)
        self.turn_duration_steps = int(turn_duration_steps)
        self.reset()

    def reset(self):
        (self.target_vel_x, self.target_ang_vel_z,
         self.target_height, self.target_yaw) = self.initial
        self.turn_end_step = -1

    def update(self, progress_step, rng):
        progress_step = int(progress_step)
        if self.stage <= 0:
            return

        velocity_period = max(self.velocity_change_period, 1)
        if (progress_step - 1) % velocity_period == 0:
            self.target_vel_x = float(
                rng.choice(np.array([0.0, 0.6, 0.8, 1.0, 1.2, 1.4]))
            )

        if self.stage >= 2:
            height_period = max(self.height_change_period, 1)
            if progress_step == 1:
                self.target_height = 1.0
            elif (progress_step - 1) % height_period == 0:
                self.target_height = float(
                    rng.choice(
                        np.array([0.80, 0.85, 0.90, 0.95, 0.95,
                                  1.00, 1.00, 1.00, 1.05])
                    )
                )

        if self.stage < 3:
            return

        if progress_step == self.turn_end_step:
            self.target_ang_vel_z = 0.0

        turn_period = max(self.turn_change_period, 1)
        if progress_step >= turn_period + 1 and (progress_step - 1) % turn_period == 0:
            yaw_delta = float(
                rng.choice(
                    np.deg2rad(
                        np.array([0.0, 0.0, -10.0, 10.0, -20.0, 20.0,
                                  -30.0, 30.0, -45.0, 45.0])
                    )
                )
            )
            duration = max(self.turn_duration_steps, 1)
            self.target_yaw += yaw_delta
            self.target_ang_vel_z = yaw_delta / (duration * self.control_dt)
            self.turn_end_step = progress_step + duration

    @property
    def values(self):
        return (
            self.target_vel_x,
            self.target_ang_vel_z,
            self.target_height,
            self.target_yaw,
        )


@dataclass
class V3WalkEnvConfig(V2WalkEnvConfig):
    """MuJoCo environment matching the 133D actor / 153D critic Isaac setup."""

    dof_vel_obs_scale: float = 0.1
    action_filter_cutoff_hz: float = 10.0
    effort_limits: Tuple[float, ...] = (
        90.0,
        90.0,
        350.0,
        90.0,
        90.0,
        350.0,
    )
    task_training_stage: int = 0
    startup_inplace_time: float = 0.8
    startup_ramp_time: float = 0.8
    startup_support_enable: bool = False
    startup_support_mode_probabilities: Tuple[float, ...] = (0.4, 0.4, 0.1, 0.1)
    startup_support_fraction_range: Tuple[float, float] = (0.2, 0.8)
    startup_support_unload_time_range: Tuple[float, float] = (0.8, 3.0)
    startup_support_fast_unload_time_range: Tuple[float, float] = (0.25, 0.8)
    startup_support_residual_fraction_range: Tuple[float, float] = (0.02, 0.10)
    startup_support_residual_hold_time_range: Tuple[float, float] = (0.3, 1.0)
    startup_support_final_unload_time_range: Tuple[float, float] = (0.3, 1.0)
    startup_support_command_gate_enable: bool = False
    startup_support_command_gate_threshold: float = 0.15
    startup_support_command_ramp_time: float = 0.8
    startup_support_fluctuation_enable: bool = False
    startup_support_noise_std_range: Tuple[float, float] = (0.02, 0.04)
    startup_support_noise_tau_range: Tuple[float, float] = (0.5, 1.5)
    startup_support_noise_limit: float = 0.06
    startup_support_actual_max_fraction: float = 0.5
    startup_tether_enable: bool = False
    startup_tether_stiffness_1mg_range: Tuple[float, float] = (100.0, 250.0)
    startup_tether_damping_range: Tuple[float, float] = (5.0, 25.0)
    startup_tether_deadzone: float = 0.01
    startup_tether_force_limit_fraction: float = 0.10
    startup_tether_scale_with_support: bool = True
    startup_no_retreat_enable: bool = False
    startup_no_retreat_duration: float = 0.0
    startup_retreat_penalty_scale: float = 0.0
    startup_negative_vx_penalty_scale: float = 0.0
    startup_action_magnitude_penalty_scale: float = 0.0
    startup_action_rate_penalty_scale: float = 0.0
    startup_max_retreat_distance: float = 0.0
    velocity_change_period: int = 600
    height_change_period: int = 430
    turn_change_period: int = 750
    turn_duration_steps: int = 60


class SRLMujocoV3Env(SRLMujocoV2Env):
    """Asymmetric-observation v3 environment.

    The actor receives 26 values per frame for five frames, followed by the
    three task commands (133D).  The critic receives the clean 30-value frame
    stack plus the same commands (153D).  Rewards are still computed by the v2
    environment from the clean full observation.
    """

    full_frame_dim = 30
    actor_frame_dim = 26
    command_dim = 3
    actor_obs_dim = 133
    critic_obs_dim = 153
    obs_dim = actor_obs_dim
    state_dim = critic_obs_dim
    act_dim = 6

    # root height and local root linear velocity xyz are privileged.
    actor_frame_indices = np.arange(4, full_frame_dim, dtype=np.int64)
    action_mirror_indices = np.array([3, 4, 5, 0, 1, 2], dtype=np.int64)

    def __init__(self, config: Optional[V3WalkEnvConfig] = None):
        super().__init__(config or V3WalkEnvConfig())
        if min(self.cfg.startup_inplace_time, self.cfg.startup_ramp_time) < 0.0:
            raise ValueError("Startup inplace and ramp times must be >= 0.")
        self._validate_startup_support_config()
        self._validate_startup_no_retreat_config()
        self.startup_support_mode = "none"
        self.startup_support_initial_fraction = 0.0
        self.startup_support_unload_time = 0.0
        self.startup_support_residual_fraction = 0.0
        self.startup_support_residual_hold_time = 0.0
        self.startup_support_final_unload_time = 0.0
        self.startup_support_current_fraction = 0.0
        self.startup_support_nominal_fraction = 0.0
        self.startup_support_noise_state = 0.0
        self.startup_support_noise_std = 0.0
        self.startup_support_noise_tau = 1.0
        self.startup_support_command_release_step = None
        self.startup_support_command_scale = 1.0
        self.startup_tether_anchor_xy = np.zeros(2, dtype=np.float64)
        self.startup_tether_stiffness_1mg = 0.0
        self.startup_tether_damping = 0.0
        self.startup_tether_displacement_xy = np.zeros(2, dtype=np.float64)
        self.startup_tether_force_xy = np.zeros(2, dtype=np.float64)
        self.startup_tether_force_limited = False
        self.startup_tether_command_was_stationary = True
        self.startup_origin_xy = np.zeros(2, dtype=np.float64)
        self.startup_forward_xy = np.array([1.0, 0.0], dtype=np.float64)
        self.startup_max_retreat_distance = 0.0
        self.task_commands = IsaacTaskCommandGenerator(
            stage=self.cfg.task_training_stage,
            control_dt=self.control_dt,
            target_vel_x=self.cfg.target_vel_x,
            target_ang_vel_z=self.cfg.target_ang_vel_z,
            target_height=self.cfg.target_height,
            target_yaw=self.cfg.target_yaw,
            velocity_change_period=self.cfg.velocity_change_period,
            height_change_period=self.cfg.height_change_period,
            turn_change_period=self.cfg.turn_change_period,
            turn_duration_steps=self.cfg.turn_duration_steps,
        )
        expected_actor_dim = self.actor_frame_dim * self.cfg.frame_stack + self.command_dim
        expected_critic_dim = self.full_frame_dim * self.cfg.frame_stack + self.command_dim
        if expected_actor_dim != self.actor_obs_dim or expected_critic_dim != self.critic_obs_dim:
            raise ValueError(
                "V3 observation layout requires frame_stack=5: "
                f"actor={expected_actor_dim}, critic={expected_critic_dim}"
            )

    def _actor_obs_from_full(self, full_obs: np.ndarray) -> np.ndarray:
        full_obs = np.asarray(full_obs, dtype=np.float32)
        frame_values = full_obs[: self.full_frame_dim * self.cfg.frame_stack]
        commands = full_obs[-self.command_dim :]
        frames = frame_values.reshape(self.cfg.frame_stack, self.full_frame_dim)
        actor_frames = frames[:, self.actor_frame_indices]
        actor_obs = np.concatenate((actor_frames.reshape(-1), commands), axis=0)
        return np.clip(actor_obs, -self.cfg.clip_obs, self.cfg.clip_obs).astype(np.float32)

    def mirror_actor_obs(self, actor_obs: np.ndarray) -> np.ndarray:
        """Mirror a 133D actor observation using the IsaacGym convention."""
        actor_obs = np.asarray(actor_obs, dtype=np.float32)
        frames = actor_obs[: self.actor_frame_dim * self.cfg.frame_stack].reshape(
            self.cfg.frame_stack, self.actor_frame_dim
        ).copy()

        # Frame layout after removing privileged root height/linear velocity:
        # angular velocity, euler error, q, qd, previous action, sin/cos phase.
        frames[:, 0] *= -1.0
        frames[:, 2] *= -1.0
        frames[:, 3] *= -1.0
        frames[:, 5] *= -1.0
        frames[:, 6:12] = frames[:, 6:12][:, self.action_mirror_indices]
        frames[:, 12:18] = frames[:, 12:18][:, self.action_mirror_indices]
        frames[:, 18:24] = frames[:, 18:24][:, self.action_mirror_indices]
        frames[:, 24:26] *= -1.0

        commands = actor_obs[-self.command_dim :].copy()
        commands[1] *= -1.0
        mirrored = np.concatenate((frames.reshape(-1), commands), axis=0)
        return np.clip(mirrored, -self.cfg.clip_obs, self.cfg.clip_obs).astype(np.float32)

    @classmethod
    def mirror_actions(cls, actions):
        """Swap left and right action triplets for NumPy arrays or tensors."""
        return actions[..., [3, 4, 5, 0, 1, 2]]

    def _reset_task_commands(self):
        self.task_commands.stage = int(self.cfg.task_training_stage)
        self.task_commands.reset()
        self._sync_task_commands(command_scale=0.0)

    def _update_task_commands(self, progress_step: int):
        self.task_commands.stage = int(self.cfg.task_training_stage)
        self.task_commands.update(progress_step, self.rng)
        policy_step = progress_step - 1
        legacy_scale = self._legacy_startup_command_scale(policy_step)
        support_scale = self._support_command_scale(policy_step)
        self.startup_support_command_scale = support_scale
        self._sync_task_commands(command_scale=legacy_scale * support_scale)

    def _legacy_startup_command_scale(self, policy_step: int) -> float:
        inplace_steps = int(round(self.cfg.startup_inplace_time / self.control_dt))
        ramp_steps = int(round(self.cfg.startup_ramp_time / self.control_dt))
        if policy_step < inplace_steps:
            return 0.0
        if ramp_steps <= 0:
            return 1.0
        ramp_step = policy_step - inplace_steps + 1
        return float(np.clip(ramp_step / ramp_steps, 0.0, 1.0))

    def _support_command_scale(self, policy_step: int) -> float:
        if (
            not self.cfg.startup_support_command_gate_enable
            or self.startup_support_mode == "none"
        ):
            return 1.0

        elapsed = max(int(policy_step), 0) * self.control_dt
        support_fraction = self._startup_support_fraction_at(elapsed)
        if support_fraction > self.cfg.startup_support_command_gate_threshold:
            self.startup_support_command_release_step = None
            return 0.0

        if self.startup_support_command_release_step is None:
            self.startup_support_command_release_step = max(int(policy_step), 0)
        ramp_time = float(self.cfg.startup_support_command_ramp_time)
        if ramp_time <= 0.0:
            return 1.0
        ramp_steps = max(int(round(ramp_time / self.control_dt)), 1)
        elapsed_steps = max(
            int(policy_step) - self.startup_support_command_release_step + 1, 0
        )
        return float(np.clip(elapsed_steps / ramp_steps, 0.0, 1.0))

    def _startup_command_scale(self, policy_step: int) -> float:
        """Combined scale retained for diagnostics and compatibility."""
        return float(
            self._legacy_startup_command_scale(policy_step)
            * self.startup_support_command_scale
        )

    def _sync_task_commands(self, command_scale: float = 1.0):
        (target_vel_x, target_ang_vel_z,
         self.cfg.target_height, self.cfg.target_yaw) = self.task_commands.values
        self.cfg.target_vel_x = float(command_scale) * target_vel_x
        self.cfg.target_ang_vel_z = float(command_scale) * target_ang_vel_z

    @staticmethod
    def _cosine_decay(start: float, end: float, elapsed: float, duration: float) -> float:
        if duration <= 0.0 or elapsed >= duration:
            return float(end)
        phase = float(np.clip(elapsed / duration, 0.0, 1.0))
        blend = 0.5 * (1.0 + np.cos(np.pi * phase))
        return float(end + (start - end) * blend)

    @staticmethod
    def _validate_range(name: str, value_range, *, lower_bound=0.0, upper_bound=None):
        if len(value_range) != 2:
            raise ValueError(f"{name} must contain two values.")
        lo, hi = map(float, value_range)
        if lo < lower_bound or hi < lo or (upper_bound is not None and hi > upper_bound):
            raise ValueError(f"Invalid {name}: {value_range}")

    def _validate_startup_support_config(self):
        probabilities = np.asarray(
            self.cfg.startup_support_mode_probabilities, dtype=np.float64
        )
        if probabilities.shape != (4,) or np.any(probabilities < 0.0):
            raise ValueError(
                "startup_support_mode_probabilities must contain four non-negative values."
            )
        if not np.isfinite(probabilities).all() or probabilities.sum() <= 0.0:
            raise ValueError("Startup support mode probabilities must have a positive sum.")
        self._validate_range(
            "startup_support_fraction_range",
            self.cfg.startup_support_fraction_range,
            upper_bound=1.0,
        )
        self._validate_range(
            "startup_support_unload_time_range", self.cfg.startup_support_unload_time_range
        )
        self._validate_range(
            "startup_support_fast_unload_time_range",
            self.cfg.startup_support_fast_unload_time_range,
        )
        self._validate_range(
            "startup_support_residual_fraction_range",
            self.cfg.startup_support_residual_fraction_range,
            upper_bound=1.0,
        )
        self._validate_range(
            "startup_support_residual_hold_time_range",
            self.cfg.startup_support_residual_hold_time_range,
        )
        self._validate_range(
            "startup_support_final_unload_time_range",
            self.cfg.startup_support_final_unload_time_range,
        )
        threshold = float(self.cfg.startup_support_command_gate_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "startup_support_command_gate_threshold must be in [0, 1]."
            )
        if self.cfg.startup_support_command_ramp_time < 0.0:
            raise ValueError("startup_support_command_ramp_time must be >= 0.")
        self._validate_range(
            "startup_support_noise_std_range",
            self.cfg.startup_support_noise_std_range,
        )
        self._validate_range(
            "startup_support_noise_tau_range",
            self.cfg.startup_support_noise_tau_range,
            lower_bound=1e-6,
        )
        if float(self.cfg.startup_support_noise_limit) < 0.0:
            raise ValueError("startup_support_noise_limit must be >= 0.")
        actual_max = float(self.cfg.startup_support_actual_max_fraction)
        if not 0.0 <= actual_max <= 1.0:
            raise ValueError(
                "startup_support_actual_max_fraction must be in [0, 1]."
            )
        self._validate_range(
            "startup_tether_stiffness_1mg_range",
            self.cfg.startup_tether_stiffness_1mg_range,
        )
        self._validate_range(
            "startup_tether_damping_range",
            self.cfg.startup_tether_damping_range,
        )
        if float(self.cfg.startup_tether_deadzone) < 0.0:
            raise ValueError("startup_tether_deadzone must be >= 0.")
        if float(self.cfg.startup_tether_force_limit_fraction) < 0.0:
            raise ValueError(
                "startup_tether_force_limit_fraction must be >= 0."
            )

    def _validate_startup_no_retreat_config(self):
        non_negative = {
            "startup_no_retreat_duration": self.cfg.startup_no_retreat_duration,
            "startup_retreat_penalty_scale": self.cfg.startup_retreat_penalty_scale,
            "startup_negative_vx_penalty_scale": (
                self.cfg.startup_negative_vx_penalty_scale
            ),
            "startup_action_magnitude_penalty_scale": (
                self.cfg.startup_action_magnitude_penalty_scale
            ),
            "startup_action_rate_penalty_scale": (
                self.cfg.startup_action_rate_penalty_scale
            ),
            "startup_max_retreat_distance": self.cfg.startup_max_retreat_distance,
        }
        for name, value in non_negative.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be >= 0.")

    def _reset_startup_no_retreat(self):
        self.startup_origin_xy[:] = self.data.xpos[self.base_id, :2]
        root_rot_mat = self.data.xmat[self.base_id].reshape(3, 3)
        forward_xy = root_rot_mat[:2, 0].astype(np.float64)
        norm = float(np.linalg.norm(forward_xy))
        if norm > 1e-8:
            self.startup_forward_xy[:] = forward_xy / norm
        else:
            self.startup_forward_xy[:] = (1.0, 0.0)
        self.startup_max_retreat_distance = 0.0

    def _startup_no_retreat_active(self) -> bool:
        return bool(
            self.cfg.startup_no_retreat_enable
            and self.cfg.startup_no_retreat_duration > 0.0
            and self.rl_step_counter * self.control_dt
            <= self.cfg.startup_no_retreat_duration
        )

    def _startup_retreat_distance(self) -> float:
        displacement_xy = (
            self.data.xpos[self.base_id, :2] - self.startup_origin_xy
        )
        forward_displacement = float(
            np.dot(displacement_xy, self.startup_forward_xy)
        )
        return max(-forward_displacement, 0.0)

    def _reset_startup_support(self):
        self.startup_support_mode = "none"
        self.startup_support_initial_fraction = 0.0
        self.startup_support_unload_time = 0.0
        self.startup_support_residual_fraction = 0.0
        self.startup_support_residual_hold_time = 0.0
        self.startup_support_final_unload_time = 0.0
        self.startup_support_current_fraction = 0.0
        self.startup_support_nominal_fraction = 0.0
        self.startup_support_noise_state = 0.0
        self.startup_support_noise_std = 0.0
        self.startup_support_noise_tau = 1.0
        self.startup_support_command_release_step = None
        self.startup_support_command_scale = 1.0
        if not self.cfg.startup_support_enable:
            return

        modes = np.asarray(("none", "smooth", "fast", "residual"), dtype=object)
        probabilities = np.asarray(
            self.cfg.startup_support_mode_probabilities, dtype=np.float64
        )
        probabilities /= probabilities.sum()
        self.startup_support_mode = str(self.rng.choice(modes, p=probabilities))
        if self.startup_support_mode == "none":
            return

        self.startup_support_initial_fraction = float(
            self.rng.uniform(*self.cfg.startup_support_fraction_range)
        )
        if self.startup_support_mode == "fast":
            time_range = self.cfg.startup_support_fast_unload_time_range
        else:
            time_range = self.cfg.startup_support_unload_time_range
        self.startup_support_unload_time = float(self.rng.uniform(*time_range))

        if self.startup_support_mode == "residual":
            residual = float(
                self.rng.uniform(*self.cfg.startup_support_residual_fraction_range)
            )
            self.startup_support_residual_fraction = min(
                residual, self.startup_support_initial_fraction
            )
            self.startup_support_residual_hold_time = float(
                self.rng.uniform(*self.cfg.startup_support_residual_hold_time_range)
            )
            self.startup_support_final_unload_time = float(
                self.rng.uniform(*self.cfg.startup_support_final_unload_time_range)
            )
            if self.cfg.startup_support_fluctuation_enable:
                self.startup_support_noise_std = float(
                    self.rng.uniform(*self.cfg.startup_support_noise_std_range)
                )
                self.startup_support_noise_tau = float(
                    self.rng.uniform(*self.cfg.startup_support_noise_tau_range)
                )
        self.startup_support_current_fraction = (
            self.startup_support_initial_fraction
        )
        self.startup_support_nominal_fraction = (
            self.startup_support_initial_fraction
        )

    def _startup_support_residual_plateau_active(self, elapsed: float) -> bool:
        if self.startup_support_mode != "residual":
            return False
        plateau_start = self.startup_support_unload_time
        plateau_end = plateau_start + self.startup_support_residual_hold_time
        return plateau_start <= elapsed < plateau_end

    def _update_startup_support_fraction(self, elapsed: float) -> float:
        nominal = self._startup_support_fraction_at(elapsed)
        self.startup_support_nominal_fraction = nominal
        if not (
            self.cfg.startup_support_fluctuation_enable
            and self._startup_support_residual_plateau_active(elapsed)
        ):
            self.startup_support_noise_state = 0.0
            self.startup_support_current_fraction = nominal
            return nominal

        tau = max(float(self.startup_support_noise_tau), 1e-6)
        alpha = float(np.exp(-self.control_dt / tau))
        innovation_scale = float(
            self.startup_support_noise_std * np.sqrt(max(1.0 - alpha * alpha, 0.0))
        )
        self.startup_support_noise_state = float(
            alpha * self.startup_support_noise_state
            + innovation_scale * self.rng.normal()
        )
        noise_limit = float(self.cfg.startup_support_noise_limit)
        self.startup_support_noise_state = float(
            np.clip(self.startup_support_noise_state, -noise_limit, noise_limit)
        )
        stable_cap = min(
            float(self.cfg.startup_support_actual_max_fraction),
            float(self.startup_support_initial_fraction),
        )
        actual = float(
            np.clip(nominal + self.startup_support_noise_state, 0.0, stable_cap)
        )
        self.startup_support_current_fraction = actual
        return actual

    def _reset_startup_tether(self):
        self.startup_tether_anchor_xy[:] = self.data.xpos[self.base_id, :2]
        self.startup_tether_stiffness_1mg = 0.0
        self.startup_tether_damping = 0.0
        self.startup_tether_displacement_xy[:] = 0.0
        self.startup_tether_force_xy[:] = 0.0
        self.startup_tether_force_limited = False
        self.startup_tether_command_was_stationary = (
            abs(float(self.cfg.target_vel_x)) <= 1e-6
            and abs(float(self.cfg.target_ang_vel_z)) <= 1e-6
        )
        if not (
            self.cfg.startup_tether_enable
            and self.startup_support_mode != "none"
        ):
            return
        self.startup_tether_stiffness_1mg = float(
            self.rng.uniform(*self.cfg.startup_tether_stiffness_1mg_range)
        )
        self.startup_tether_damping = float(
            self.rng.uniform(*self.cfg.startup_tether_damping_range)
        )

    def _startup_tether_force(self, support_fraction: float) -> np.ndarray:
        current_xy = self.data.xpos[self.base_id, :2].astype(np.float64)
        self.startup_tether_displacement_xy[:] = current_xy - self.startup_tether_anchor_xy
        self.startup_tether_force_xy[:] = 0.0
        self.startup_tether_force_limited = False
        if not (
            self.cfg.startup_tether_enable
            and self.startup_support_mode != "none"
            and support_fraction > 0.0
        ):
            return self.startup_tether_force_xy

        # The horizontal tether represents the rig holding the robot near its
        # start point while stepping in place. It must not oppose commanded
        # locomotion or turning.
        command_is_stationary = (
            abs(float(self.cfg.target_vel_x)) <= 1e-6
            and abs(float(self.cfg.target_ang_vel_z)) <= 1e-6
        )
        if not command_is_stationary:
            self.startup_tether_command_was_stationary = False
            return self.startup_tether_force_xy

        if not self.startup_tether_command_was_stationary:
            self.startup_tether_anchor_xy[:] = current_xy
            self.startup_tether_displacement_xy[:] = 0.0
            self.startup_tether_command_was_stationary = True

        displacement = self.startup_tether_displacement_xy.copy()
        distance = float(np.linalg.norm(displacement))
        deadzone = float(self.cfg.startup_tether_deadzone)
        if distance <= deadzone:
            displacement[:] = 0.0
        elif distance > 1e-12:
            displacement *= (distance - deadzone) / distance

        if self.cfg.startup_tether_scale_with_support:
            stiffness_scale = float(support_fraction)
            damping_scale = float(np.sqrt(support_fraction))
        else:
            stiffness_scale = 1.0
            damping_scale = 1.0
        stiffness = self.startup_tether_stiffness_1mg * stiffness_scale
        damping = self.startup_tether_damping * damping_scale
        velocity_xy = self.data.qvel[:2].astype(np.float64)
        force = -stiffness * displacement - damping * velocity_xy

        force_limit = (
            float(self.cfg.startup_tether_force_limit_fraction) * self.body_weight
        )
        force_norm = float(np.linalg.norm(force))
        if force_limit > 0.0 and force_norm > force_limit:
            force *= force_limit / force_norm
            self.startup_tether_force_limited = True
        self.startup_tether_force_xy[:] = force
        return self.startup_tether_force_xy

    def _startup_support_fraction_at(self, elapsed: float) -> float:
        if self.startup_support_mode == "none":
            return 0.0
        if self.startup_support_mode != "residual":
            return self._cosine_decay(
                self.startup_support_initial_fraction,
                0.0,
                elapsed,
                self.startup_support_unload_time,
            )

        if elapsed < self.startup_support_unload_time:
            return self._cosine_decay(
                self.startup_support_initial_fraction,
                self.startup_support_residual_fraction,
                elapsed,
                self.startup_support_unload_time,
            )
        elapsed -= self.startup_support_unload_time
        if elapsed < self.startup_support_residual_hold_time:
            return self.startup_support_residual_fraction
        elapsed -= self.startup_support_residual_hold_time
        return self._cosine_decay(
            self.startup_support_residual_fraction,
            0.0,
            elapsed,
            self.startup_support_final_unload_time,
        )

    def _before_physics_step(self, physics_step):
        super()._before_physics_step(physics_step)
        if physics_step == 0:
            elapsed = self.rl_step_counter * self.control_dt
            self._update_startup_support_fraction(elapsed)
        fraction = self.startup_support_current_fraction
        tether_force_xy = self._startup_tether_force(fraction)
        # The parent assigns the DR force pulse on every physics substep, so
        # adding the tether here cannot accumulate across mj_step calls.
        self.data.xfrc_applied[self.base_id, :2] += tether_force_xy
        self.data.xfrc_applied[self.base_id, 2] = fraction * self.body_weight

    def _add_startup_support_info(self, info):
        info["startup_support_enabled"] = bool(self.cfg.startup_support_enable)
        info["startup_support_mode"] = self.startup_support_mode
        info["startup_support_initial_fraction"] = float(
            self.startup_support_initial_fraction
        )
        info["startup_support_fraction"] = float(
            self.startup_support_current_fraction
        )
        info["startup_support_nominal_fraction"] = float(
            self.startup_support_nominal_fraction
        )
        info["startup_support_residual_fraction"] = float(
            self.startup_support_residual_fraction
        )
        info["startup_support_noise"] = float(
            self.startup_support_current_fraction
            - self.startup_support_nominal_fraction
        )
        info["startup_support_noise_std"] = float(
            self.startup_support_noise_std
        )
        info["startup_support_noise_tau"] = float(
            self.startup_support_noise_tau
        )
        info["startup_support_force_n"] = float(
            self.startup_support_current_fraction * self.body_weight
        )
        info["startup_support_unload_time"] = float(
            self.startup_support_unload_time
        )
        info["startup_support_command_gate_enabled"] = bool(
            self.cfg.startup_support_command_gate_enable
        )
        info["startup_support_command_gate_threshold"] = float(
            self.cfg.startup_support_command_gate_threshold
        )
        info["startup_support_command_scale"] = float(
            self.startup_support_command_scale
        )
        info["startup_tether_enabled"] = bool(
            self.cfg.startup_tether_enable
            and self.startup_support_mode != "none"
        )
        info["startup_tether_stiffness_1mg"] = float(
            self.startup_tether_stiffness_1mg
        )
        info["startup_tether_damping"] = float(
            self.startup_tether_damping
        )
        info["startup_tether_displacement_xy"] = (
            self.startup_tether_displacement_xy.astype(np.float32).copy()
        )
        info["startup_tether_displacement_norm"] = float(
            np.linalg.norm(self.startup_tether_displacement_xy)
        )
        info["startup_tether_force_xy"] = (
            self.startup_tether_force_xy.astype(np.float32).copy()
        )
        info["startup_tether_force_norm"] = float(
            np.linalg.norm(self.startup_tether_force_xy)
        )
        info["startup_tether_force_limited"] = bool(
            self.startup_tether_force_limited
        )
        info["startup_no_retreat_enabled"] = bool(
            self.cfg.startup_no_retreat_enable
        )
        info["startup_no_retreat_active"] = self._startup_no_retreat_active()
        info["startup_max_retreat_distance"] = float(
            self.startup_max_retreat_distance
        )
        info.setdefault("startup_retreat_distance", 0.0)
        info.setdefault("startup_retreat_terminated", False)
        return info

    def _compute_reward(self, obs, action):
        reward, terminated, truncated, info = super()._compute_reward(obs, action)
        active = self._startup_no_retreat_active()
        retreat_distance = self._startup_retreat_distance() if active else 0.0
        if active:
            self.startup_max_retreat_distance = max(
                self.startup_max_retreat_distance, retreat_distance
            )

        negative_vx_penalty = 0.0
        action_magnitude_penalty = 0.0
        action_rate_penalty = 0.0
        retreat_penalty = 0.0
        if active:
            negative_vx_penalty = max(-float(info.get("vel_x", 0.0)), 0.0) ** 2
            action_magnitude_penalty = float(np.sum(np.asarray(action) ** 2))
            action_rate_penalty = float(info.get("penalty_actions_rate", 0.0))
            retreat_penalty = retreat_distance * retreat_distance
            reward -= (
                self.cfg.startup_retreat_penalty_scale * retreat_penalty
                + self.cfg.startup_negative_vx_penalty_scale * negative_vx_penalty
                + self.cfg.startup_action_magnitude_penalty_scale
                * action_magnitude_penalty
                + self.cfg.startup_action_rate_penalty_scale * action_rate_penalty
            )

        retreat_terminated = bool(
            active
            and self.cfg.startup_max_retreat_distance > 0.0
            and self.startup_max_retreat_distance
            > self.cfg.startup_max_retreat_distance
        )
        if retreat_terminated and not terminated:
            applied_penalty = float(self.cfg.termination_penalty)
            reward += applied_penalty
            info["penalty_termination"] = float(
                info.get("penalty_termination", 0.0) + applied_penalty
            )
            terminated = True

        info.update(
            {
                "reward_total": float(reward),
                "startup_no_retreat_active": bool(active),
                "startup_retreat_distance": float(retreat_distance),
                "startup_max_retreat_distance": float(
                    self.startup_max_retreat_distance
                ),
                "startup_retreat_terminated": bool(retreat_terminated),
                "penalty_startup_retreat": float(retreat_penalty),
                "penalty_startup_negative_vx": float(negative_vx_penalty),
                "penalty_startup_action_magnitude": float(
                    action_magnitude_penalty
                ),
                "penalty_startup_action_rate": float(action_rate_penalty),
            }
        )
        return float(reward), bool(terminated), truncated, info

    def get_critic_obs(self) -> np.ndarray:
        """Return the clean privileged state used only by the critic."""
        return self._get_stacked_obs(self.clean_obs_history).astype(np.float32, copy=True)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_task_commands()
        self._reset_startup_support()
        legacy_scale = self._legacy_startup_command_scale(0)
        support_scale = self._support_command_scale(0)
        self.startup_support_command_scale = support_scale
        self._sync_task_commands(command_scale=legacy_scale * support_scale)
        # The seed was applied above so support sampling, phase sampling and DR
        # all share one deterministic per-environment RNG stream.
        full_noisy_obs, info = super().reset(seed=None)
        self._reset_startup_tether()
        self._reset_startup_no_retreat()
        actor_obs = self._actor_obs_from_full(full_noisy_obs)
        info = dict(info)
        info["startup_command_scale"] = float(legacy_scale * support_scale)
        info["critic_obs"] = self.get_critic_obs()
        info["actor_obs_mirrored"] = self.mirror_actor_obs(actor_obs)
        self._add_startup_support_info(info)
        return actor_obs, info

    def step(self, action):
        self._update_task_commands(self.rl_step_counter + 1)
        full_noisy_obs, reward, terminated, truncated, info = super().step(action)
        actor_obs = self._actor_obs_from_full(full_noisy_obs)
        info = dict(info)
        info["startup_command_scale"] = self._startup_command_scale(
            self.rl_step_counter - 1
        )
        info["critic_obs"] = self.get_critic_obs()
        info["actor_obs_mirrored"] = self.mirror_actor_obs(actor_obs)
        self._add_startup_support_info(info)
        return (
            actor_obs,
            reward,
            terminated,
            truncated,
            info,
        )
