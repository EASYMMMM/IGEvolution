import collections
import math
from dataclasses import dataclass
from typing import Optional, Tuple

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


class SecondOrderLowPass:
    """Second-order Butterworth low-pass used by srl_real_bot_v2.py."""

    def __init__(self, cutoff_hz: float, sample_dt: float, initial_value):
        sample_freq = 1.0 / sample_dt
        cutoff_hz = min(float(cutoff_hz), 0.499 * sample_freq)
        k = math.tan(math.pi * cutoff_hz / sample_freq)
        norm = 1.0 / (1.0 + math.sqrt(2.0) * k + k * k)
        self.b0 = float(k * k * norm)
        self.b1 = float(2.0 * self.b0)
        self.b2 = float(self.b0)
        self.a1 = float(2.0 * (k * k - 1.0) * norm)
        self.a2 = float((1.0 - math.sqrt(2.0) * k + k * k) * norm)
        self.reset(initial_value)

    def reset(self, value):
        value = np.asarray(value, dtype=np.float32).copy()
        self.x1 = value.copy()
        self.x2 = value.copy()
        self.y1 = value.copy()
        self.y2 = value.copy()

    def apply(self, x):
        x = np.asarray(x, dtype=np.float32)
        y = (
            self.b0 * x
            + self.b1 * self.x1
            + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2
        )
        self.x2[:] = self.x1
        self.x1[:] = x
        self.y2[:] = self.y1
        self.y1[:] = y
        return y.astype(np.float32)

    def overwrite_last_output(self, y):
        self.y1[:] = np.asarray(y, dtype=np.float32)


@dataclass
class V2WalkEnvConfig:
    """MuJoCo v2 SRL environment.

    This is intentionally separate from srl_mujoco_v1_env.py.  v2 follows the
    IsaacGym srl_real_bot_v2.py control surface: default joint angles come from
    config, actions map to PD targets through XML joint limits, and an optional
    second-order target low-pass filter is applied before the actuator backend.
    """

    xml_path: str = "mjcf/srl_real_v1/srl_real_bot_v2_pos.xml"
    dt: float = 0.005
    decimation: int = 3
    gait_period: float = 54.0
    frame_stack: int = 5

    target_vel_x: float = 1.0
    target_ang_vel_z: float = 0.0
    target_height: float = 1.0
    root_height: float = 1.1
    target_yaw: float = 0.0
    target_point_x: float = 1000.0

    default_dof_pos: Tuple[float, ...] = (0.0, -0.55, -0.3, 0.0, -0.55, -0.3)
    kp: Tuple[float, ...] = (250.0, 300.0, 300.0, 250.0, 300.0, 300.0)
    kd: Tuple[float, ...] = (18.0, 22.0, 14.0, 18.0, 22.0, 14.0)
    effort_limits: Tuple[float, ...] = (120.0, 120.0, 400.0, 120.0, 120.0, 400.0)
    gear_ratio: float = 450.0

    action_scale: float = 1.0
    action_filter_enable: bool = True
    action_filter_cutoff_hz: float = 8.0
    pd_tracking_error_limits: Optional[Tuple[float, ...]] = None
    pd_target_change_limit: Optional[float] = None
    clip_actions: float = 1.0
    clip_obs: float = 10.0
    dof_vel_obs_scale: float = 0.05
    control_backend: str = "position_target"  # "position_target" or "torque_pd"

    passive_dof_damping: float = 1.0
    passive_joint_stiffness: float = 0.0
    passive_armature: Optional[Tuple[float, ...]] = None

    max_episode_steps: int = 5000
    termination_height: float = 0.70
    termination_penalty: float = -25.0
    foot_contact_height: float = 0.055
    foot_clearance: float = 0.2

    alive_reward_scale: float = 1.0
    progress_reward_scale: float = 1.0
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
    lateral_min_distance: float = 0.25
    lateral_max_distance: float = 0.85
    lateral_symmetry_weight: float = 0.0
    lateral_target_distance: float = 0.0
    lateral_target_weight: float = 0.0
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

    base_wobble_penalty_scale: float = 0.0
    pitch_wobble_penalty_scale: float = 0.0
    roll_wobble_penalty_scale: float = 0.0
    roll_rate_weight: float = 0.3
    base_ang_acc_penalty_scale: float = 0.0
    yaw_drift_penalty_scale: float = 0.0
    foot_impact_penalty_scale: float = 0.0
    foot_force_threshold_bw: float = 1.8
    foot_force_penalty_power: float = 2.0
    randomize_initial_phase: bool = True
    initial_phase: Optional[int] = None

    domain_randomization_enable: bool = False
    dr_friction_range: Tuple[float, float] = (1.0, 1.0)
    dr_kp_range: Tuple[float, float] = (1.0, 1.0)
    dr_kp_joint_range: Tuple[float, float] = (1.0, 1.0)
    dr_kd_range: Tuple[float, float] = (1.0, 1.0)
    dr_kd_joint_range: Tuple[float, float] = (1.0, 1.0)
    dr_effort_range: Tuple[float, float] = (1.0, 1.0)
    dr_effort_joint_range: Tuple[float, float] = (1.0, 1.0)
    dr_gear_range: Tuple[float, float] = (1.0, 1.0)
    dr_mass_range: Tuple[float, float] = (1.0, 1.0)
    dr_mass_link_range: Tuple[float, float] = (1.0, 1.0)
    dr_inertia_range: Tuple[float, float] = (1.0, 1.0)
    dr_inertia_link_range: Tuple[float, float] = (1.0, 1.0)
    dr_damping_range: Tuple[float, float] = (1.0, 1.0)
    dr_damping_joint_range: Tuple[float, float] = (1.0, 1.0)
    dr_frictionloss_range: Tuple[float, float] = (1.0, 1.0)
    dr_frictionloss_joint_range: Tuple[float, float] = (1.0, 1.0)
    dr_armature_range: Tuple[float, float] = (1.0, 1.0)
    dr_armature_joint_range: Tuple[float, float] = (1.0, 1.0)
    dr_foot_friction_range: Tuple[float, float] = (1.0, 1.0)
    dr_foot_solref_timeconst_range: Tuple[float, float] = (1.0, 1.0)
    dr_foot_solref_dampratio_range: Tuple[float, float] = (1.0, 1.0)
    dr_foot_solimp_width_range: Tuple[float, float] = (1.0, 1.0)
    dr_foot_radius_range: Tuple[float, float] = (1.0, 1.0)
    dr_base_com_shift: float = 0.0
    dr_link_com_shift: float = 0.0
    dr_gravity_std: float = 0.0
    dr_joint_limit_std: float = 0.0
    dr_stratified_sampling_enable: bool = False
    dr_scenario_probabilities: Tuple[float, ...] = (0.45, 0.15, 0.15, 0.05, 0.10, 0.10)
    dr_hard_longitudinal_gravity_sigma_range: Tuple[float, float] = (1.5, 2.5)
    dr_hard_lateral_gravity_sigma_range: Tuple[float, float] = (1.0, 2.0)
    dr_combined_longitudinal_gravity_sigma_range: Tuple[float, float] = (2.0, 3.0)
    obs_noise_std: float = 0.0
    obs_noise_dof_pos_std: float = 0.0
    obs_noise_dof_vel_std: float = 0.0
    action_noise_std: float = 0.0
    # Legacy control-period delay. One step equals dt * decimation.
    action_delay_max_steps: int = 0
    # Preferred delay setting. One step equals one MuJoCo physics timestep.
    # When omitted, action_delay_max_steps is converted automatically.
    action_delay_max_physics_steps: Optional[int] = None
    velocity_perturbation_enable: bool = False
    velocity_perturbation_prob: float = 1.0 / 500.0
    velocity_perturbation_range: Tuple[float, float] = (0.2, 0.8)
    horizontal_force_perturbation_enable: bool = False
    horizontal_force_perturbation_prob: float = 1.0 / 1000.0
    horizontal_force_perturbation_range: Tuple[float, float] = (20.0, 80.0)
    horizontal_force_perturbation_duration_range: Tuple[float, float] = (0.05, 0.20)


class SRLMujocoV2Env:
    """No-human MuJoCo v2 environment for v2 policy finetuning."""

    obs_dim = 153
    act_dim = 6

    def __init__(self, config: Optional[V2WalkEnvConfig] = None):
        self.cfg = config or V2WalkEnvConfig()
        if self.cfg.control_backend not in ("position_target", "torque_pd"):
            raise ValueError(f"Unsupported control_backend: {self.cfg.control_backend}")

        self.control_dt = self.cfg.dt * self.cfg.decimation
        self.rng = np.random.default_rng()
        self.dr_progress = 1.0
        self.episode_dr_progress = 1.0
        self.dr_scenario = "normal"
        self.dr_scenario_probabilities = np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.default_dof_pos = np.asarray(self.cfg.default_dof_pos, dtype=np.float32)
        self.nominal_kp = np.asarray(self.cfg.kp, dtype=np.float32)
        self.nominal_kd = np.asarray(self.cfg.kd, dtype=np.float32)
        self.nominal_effort_limits = np.asarray(self.cfg.effort_limits, dtype=np.float32)
        self.nominal_gear_ratio = float(self.cfg.gear_ratio)
        self.kp = self.nominal_kp.copy()
        self.kd = self.nominal_kd.copy()
        self.effort_limits = self.nominal_effort_limits.copy()
        self.gear_ratio = self.nominal_gear_ratio

        if bool(getattr(self.cfg, "fixed_base", False)):
            spec = mujoco.MjSpec()
            spec.from_file(self.cfg.xml_path)
            base_spec = spec.find_body("base_link")
            if base_spec is None:
                raise RuntimeError("Missing base_link required for fixed-base mode.")
            base_spec.pos = np.array(
                [0.0, 0.0, float(self.cfg.root_height)], dtype=np.float64
            )
            weld = spec.add_equality()
            weld.name = "fixed_base_weld"
            weld.type = mujoco.mjtEq.mjEQ_WELD
            weld.objtype = mujoco.mjtObj.mjOBJ_BODY
            weld.name1 = "base_link"
            weld.solref = np.array(
                [float(getattr(self.cfg, "fixed_base_timeconst", 0.001)), 1.0],
                dtype=np.float64,
            )
            weld.solimp = np.array(
                [0.99, 0.999, 0.001, 0.5, 2.0], dtype=np.float64
            )
            self.model = spec.compile()
        else:
            self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.model.opt.timestep = self.cfg.dt
        self.srl_joint_ids = self._get_srl_joint_ids()
        self.srl_qpos_indices = self.model.jnt_qposadr[self.srl_joint_ids].astype(
            np.int32, copy=True
        )
        self.srl_dof_indices = self.model.jnt_dofadr[self.srl_joint_ids].astype(
            np.int32, copy=True
        )
        # Only the six motor joints belong to the policy. Extra passive joints
        # (for example the v4 connector slides) keep their XML stiffness and
        # damping instead of being folded into the 6D action/state vectors.
        self.model.jnt_stiffness[self.srl_joint_ids] = _as_float_array(
            self.cfg.passive_joint_stiffness, self.act_dim
        )
        # Passive damping applies only to the six actuated joints. Applying it
        # to the floating-base DOFs would add nonphysical world-frame drag.
        self.model.dof_damping[:6] = 0.0
        self.model.dof_damping[self.srl_dof_indices] = _as_float_array(
            self.cfg.passive_dof_damping, self.act_dim
        )
        if self.cfg.passive_armature is not None:
            self.model.dof_armature[self.srl_dof_indices] = _as_float_array(
                self.cfg.passive_armature, self.act_dim
            )
        self._store_nominal_model_params()

        self.data = mujoco.MjData(self.model)
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.left_foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_contact")
        self.right_foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_contact")
        if min(self.floor_geom_id, self.left_foot_geom_id, self.right_foot_geom_id) < 0:
            raise RuntimeError("Missing floor/foot contact geom names required for foot impact reward.")
        self.body_weight = float(
            np.sum(self.model.body_mass) * np.linalg.norm(self.model.opt.gravity)
        )

        self.joint_low, self.joint_high = self._get_joint_ranges()
        self.pd_low, self.pd_high = self._make_pd_action_bounds()

        self.raw_action = np.zeros(self.act_dim, dtype=np.float32)
        self.filtered_action = np.zeros(self.act_dim, dtype=np.float32)
        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.prev_prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.raw_target_pos = self.default_dof_pos.copy()
        self.target_pos = self.default_dof_pos.copy()
        self.last_applied_torques = np.zeros(self.act_dim, dtype=np.float32)
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
        self.target_filter = SecondOrderLowPass(
            self.cfg.action_filter_cutoff_hz, self.control_dt, self.default_dof_pos
        )
        self.pd_tracking_error_limits = None
        if self.cfg.pd_tracking_error_limits is not None:
            limits = np.asarray(self.cfg.pd_tracking_error_limits, dtype=np.float32)
            if limits.shape != (self.act_dim,) or np.any(limits <= 0.0):
                raise ValueError(
                    "pd_tracking_error_limits must contain six positive values."
                )
            self.pd_tracking_error_limits = limits
        self.pd_target_change_limit = self.cfg.pd_target_change_limit
        if (
            self.pd_target_change_limit is not None
            and float(self.pd_target_change_limit) <= 0.0
        ):
            raise ValueError("pd_target_change_limit must be positive when enabled.")
        self.pd_target_change_clip_mask = np.zeros(self.act_dim, dtype=bool)
        self.pd_tracking_error_clip_mask = np.zeros(self.act_dim, dtype=bool)
        self.rl_step_counter = 0
        self.phase_counter = 0
        self.action_delay_physics_steps = 0
        self.action_delay_steps = 0.0
        self.horizontal_force_pulse = np.zeros(2, dtype=np.float64)
        self.horizontal_force_pulse_remaining_steps = 0
        self.horizontal_force_pulse_event_count = 0
        self.horizontal_force_pulse_last_duration_steps = 0
        self._validate_horizontal_force_perturbation_config()
        self.previous_cycle_target_pos = self.default_dof_pos.copy()
        self.previous_cycle_control_action = np.zeros(self.act_dim, dtype=np.float32)
        self.delayed_target_queue = collections.deque()
        self.delayed_control_action_queue = collections.deque()
        self.dr_sampled_params = {}
        self.obs_history = collections.deque(maxlen=self.cfg.frame_stack)
        self.clean_obs_history = collections.deque(maxlen=self.cfg.frame_stack)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Keep every DR component at one curriculum strength for this episode.
        self.episode_dr_progress = self.dr_progress
        self._reset_dr_scenario()
        self._reset_randomized_params()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = self.cfg.root_height
        self.data.qpos[self.srl_qpos_indices] = self.default_dof_pos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.raw_action[:] = 0.0
        self.filtered_action[:] = 0.0
        self.prev_action[:] = 0.0
        self.prev_prev_action[:] = 0.0
        self.raw_target_pos[:] = self.default_dof_pos
        self.target_pos[:] = self.default_dof_pos
        self.previous_cycle_target_pos[:] = self.default_dof_pos
        self.previous_cycle_control_action[:] = 0.0
        self.last_applied_torques[:] = 0.0
        self.srl_peak_ratio_window = 0.0
        self.srl_tau2_ema = 0.0
        self.last_left_foot_force_max = 0.0
        self.last_right_foot_force_max = 0.0
        self.target_filter.reset(self.default_dof_pos)
        self.pd_target_change_clip_mask[:] = False
        self.pd_tracking_error_clip_mask[:] = False
        self.rl_step_counter = 0
        self.horizontal_force_pulse[:] = 0.0
        self.horizontal_force_pulse_remaining_steps = 0
        self.horizontal_force_pulse_event_count = 0
        self.horizontal_force_pulse_last_duration_steps = 0
        if self.cfg.initial_phase is not None:
            self.phase_counter = int(self.cfg.initial_phase) % max(
                int(self.cfg.gait_period), 1
            )
        elif self.cfg.randomize_initial_phase:
            self.phase_counter = int(self.rng.integers(0, max(int(self.cfg.gait_period), 1)))
        else:
            self.phase_counter = 0
        self._reset_action_delay()
        self._reset_action_delay_queues()

        self.prev_local_ang_vel[:] = self.data.qvel[3:6]
        self.prev_srl_end_body_pos[:] = self._get_srl_end_body_pos()
        self.potential = self._compute_potential()
        self.prev_potential = self.potential

        initial_clean_obs = self._get_clean_single_frame_obs()
        initial_actor_obs = self._add_actor_obs_noise(initial_clean_obs)
        self.obs_history.clear()
        self.clean_obs_history.clear()
        for _ in range(self.cfg.frame_stack):
            self.obs_history.append(initial_actor_obs.copy())
            self.clean_obs_history.append(initial_clean_obs.copy())

        return self._get_stacked_obs(self.obs_history), self._get_info()

    def step(self, action):
        issued_action = np.clip(
            np.asarray(action, dtype=np.float32),
            -self.cfg.clip_actions,
            self.cfg.clip_actions,
        )
        control_action = issued_action.copy()
        action_noise_std = 0.0
        if self.cfg.domain_randomization_enable:
            action_noise_std = self.cfg.action_noise_std * self.episode_dr_progress
        if action_noise_std > 0.0:
            control_action += self.rng.normal(
                0.0, action_noise_std, self.act_dim
            ).astype(np.float32)
            control_action = np.clip(
                control_action, -self.cfg.clip_actions, self.cfg.clip_actions
            )

        # Keep the policy-issued command in observation history. Delay and
        # actuator noise belong to the plant, not to the actor observation.
        self.prev_prev_action[:] = self.prev_action
        self.prev_action[:] = self.raw_action
        self.raw_action[:] = issued_action

        self.raw_target_pos[:] = self._action_to_pd_targets(control_action)
        if self.cfg.action_filter_enable:
            target_pos = self.target_filter.apply(self.raw_target_pos)
        else:
            target_pos = self.raw_target_pos.copy()
        target_pos = np.clip(target_pos, self.pd_low, self.pd_high).astype(np.float32)
        target_pos = self._apply_pd_target_safety_limits(target_pos)
        if self.cfg.action_filter_enable:
            self.target_filter.overwrite_last_output(target_pos)
        self.target_pos[:] = target_pos

        self.filtered_action[:] = self._pd_targets_to_action(self.target_pos)
        self.data.xfrc_applied[:] = 0.0
        self._maybe_apply_velocity_perturbation()
        self._maybe_start_horizontal_force_pulse()

        left_foot_force_max = 0.0
        right_foot_force_max = 0.0

        last_control_action = control_action
        for physics_step in range(self.cfg.decimation):
            self._before_physics_step(physics_step)
            if self.action_delay_physics_steps > 0:
                self.delayed_target_queue.append(self.target_pos.copy())
                self.delayed_control_action_queue.append(control_action.copy())
                applied_target_pos = self.delayed_target_queue.popleft()
                last_control_action = self.delayed_control_action_queue.popleft()
            else:
                applied_target_pos = self.target_pos
                last_control_action = control_action.copy()
            if self.cfg.control_backend == "position_target":
                self.data.ctrl[:] = applied_target_pos
                self.last_applied_torques[:] = self._estimate_pd_torques(applied_target_pos)
            else:
                torques = self._estimate_pd_torques(applied_target_pos)
                self.last_applied_torques[:] = torques
                self.data.ctrl[:] = torques / self.gear_ratio

            mujoco.mj_step(self.model, self.data)
            left_force, right_force = self._get_foot_contact_forces()
            left_foot_force_max = max(left_foot_force_max, left_force)
            right_foot_force_max = max(right_foot_force_max, right_force)

        self.previous_cycle_target_pos[:] = self.target_pos
        self.previous_cycle_control_action[:] = control_action

        self.last_left_foot_force_max = left_foot_force_max
        self.last_right_foot_force_max = right_foot_force_max
        self.rl_step_counter += 1
        self.phase_counter = (self.phase_counter + 1) % max(int(self.cfg.gait_period), 1)

        clean_frame = self._get_clean_single_frame_obs()
        actor_frame = self._add_actor_obs_noise(clean_frame)
        self.clean_obs_history.append(clean_frame)
        self.obs_history.append(actor_frame)
        clean_obs = self._get_stacked_obs(self.clean_obs_history)
        obs = self._get_stacked_obs(self.obs_history)
        reward, terminated, truncated, reward_info = self._compute_reward(clean_obs, self.raw_action)
        info = self._get_info()
        info["raw_action"] = issued_action.copy()
        info["control_action"] = last_control_action.copy()
        info["noisy_control_action"] = control_action.copy()
        info["filtered_action"] = self.filtered_action.copy()
        info["raw_target_pos"] = self.raw_target_pos.copy()
        info["target_pos"] = self.target_pos.copy()
        info["pd_target_change_clip_mask"] = self.pd_target_change_clip_mask.copy()
        info["pd_tracking_error_clip_mask"] = self.pd_tracking_error_clip_mask.copy()
        info["pd_target_change_clip_fraction"] = float(
            np.mean(self.pd_target_change_clip_mask)
        )
        info["pd_tracking_error_clip_fraction"] = float(
            np.mean(self.pd_tracking_error_clip_mask)
        )
        info["last_torques"] = self.last_applied_torques.copy()
        info.update(reward_info)
        return obs, reward, terminated, truncated, info

    def _apply_pd_target_safety_limits(self, target_pos):
        target = np.asarray(target_pos, dtype=np.float32).copy()
        self.pd_target_change_clip_mask[:] = False
        self.pd_tracking_error_clip_mask[:] = False

        if self.pd_target_change_limit is not None:
            max_change = float(self.pd_target_change_limit)
            limited = np.clip(
                target,
                self.previous_cycle_target_pos - max_change,
                self.previous_cycle_target_pos + max_change,
            )
            self.pd_target_change_clip_mask[:] = np.abs(limited - target) > 1e-7
            target = limited

        if self.pd_tracking_error_limits is not None:
            q = self._get_srl_qpos()
            limited = np.clip(
                target,
                q - self.pd_tracking_error_limits,
                q + self.pd_tracking_error_limits,
            )
            self.pd_tracking_error_clip_mask[:] = np.abs(limited - target) > 1e-7
            target = limited

        return np.clip(target, self.pd_low, self.pd_high).astype(np.float32)

    def _maybe_apply_velocity_perturbation(self):
        if (
            not self.cfg.domain_randomization_enable
            or not self.cfg.velocity_perturbation_enable
        ):
            return
        perturb_prob = self.cfg.velocity_perturbation_prob * self.episode_dr_progress
        if self.rng.random() >= perturb_prob:
            return
        min_v, max_v = self.cfg.velocity_perturbation_range
        mag = float(self.rng.uniform(min_v, max_v))
        angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.data.qvel[0] += mag * np.cos(angle)
        self.data.qvel[1] += mag * np.sin(angle)

    def _validate_horizontal_force_perturbation_config(self):
        probability = float(self.cfg.horizontal_force_perturbation_prob)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "horizontal_force_perturbation_prob must be in [0, 1]."
            )
        force_lo, force_hi = map(
            float, self.cfg.horizontal_force_perturbation_range
        )
        if force_lo < 0.0 or force_hi < force_lo:
            raise ValueError(
                "horizontal_force_perturbation_range must be nonnegative and ordered."
            )
        duration_lo, duration_hi = map(
            float, self.cfg.horizontal_force_perturbation_duration_range
        )
        if duration_lo <= 0.0 or duration_hi < duration_lo:
            raise ValueError(
                "horizontal_force_perturbation_duration_range must be positive and ordered."
            )

    def _maybe_start_horizontal_force_pulse(self):
        if (
            not self.cfg.domain_randomization_enable
            or not self.cfg.horizontal_force_perturbation_enable
            or self.horizontal_force_pulse_remaining_steps > 0
        ):
            return
        trigger_prob = (
            self.cfg.horizontal_force_perturbation_prob
            * self.episode_dr_progress
        )
        if self.rng.random() >= trigger_prob:
            return

        force_lo, force_hi = self.cfg.horizontal_force_perturbation_range
        magnitude = float(self.rng.uniform(force_lo, force_hi))
        magnitude *= self.episode_dr_progress
        axis = int(self.rng.integers(0, 2))
        sign = -1.0 if self.rng.random() < 0.5 else 1.0
        self.horizontal_force_pulse[:] = 0.0
        self.horizontal_force_pulse[axis] = sign * magnitude

        duration_lo, duration_hi = (
            self.cfg.horizontal_force_perturbation_duration_range
        )
        duration = float(self.rng.uniform(duration_lo, duration_hi))
        duration_steps = max(1, int(round(duration / self.cfg.dt)))
        self.horizontal_force_pulse_remaining_steps = duration_steps
        self.horizontal_force_pulse_last_duration_steps = duration_steps
        self.horizontal_force_pulse_event_count += 1

    def _before_physics_step(self, physics_step):
        del physics_step
        if self.horizontal_force_pulse_remaining_steps > 0:
            self.data.xfrc_applied[self.base_id, :2] = self.horizontal_force_pulse
            self.horizontal_force_pulse_remaining_steps -= 1
            if self.horizontal_force_pulse_remaining_steps == 0:
                self.horizontal_force_pulse[:] = 0.0
        else:
            self.data.xfrc_applied[self.base_id, :2] = 0.0

    def _uniform(self, value_range):
        lo, hi = self._scaled_range(value_range)
        return float(self.rng.uniform(lo, hi))

    def _uniform_array(self, value_range, size):
        lo, hi = self._scaled_range(value_range)
        return self.rng.uniform(lo, hi, size=size).astype(np.float64)

    def _scaled_range(self, value_range):
        lo, hi = float(value_range[0]), float(value_range[1])
        lo = 1.0 + self.episode_dr_progress * (lo - 1.0)
        hi = 1.0 + self.episode_dr_progress * (hi - 1.0)
        return lo, hi

    def _uniform_range_half(self, value_range, *, upper_half):
        lo, hi = self._scaled_range(value_range)
        midpoint = 0.5 * (lo + hi)
        if upper_half:
            lo = midpoint
        else:
            hi = midpoint
        return float(self.rng.uniform(lo, hi))

    def _reset_dr_scenario(self):
        scenario_names = (
            "normal",
            "fixed_delay",
            "longitudinal_gravity",
            "lateral_gravity",
            "high_kp_low_kd",
            "combined_hard",
        )
        if (
            not self.cfg.domain_randomization_enable
            or not self.cfg.dr_stratified_sampling_enable
        ):
            probabilities = np.array(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
        else:
            final_probabilities = np.asarray(
                self.cfg.dr_scenario_probabilities, dtype=np.float64
            )
            if final_probabilities.shape != (len(scenario_names),):
                raise ValueError(
                    "dr_scenario_probabilities must contain six values for "
                    "normal/fixed_delay/longitudinal_gravity/lateral_gravity/"
                    "high_kp_low_kd/combined_hard"
                )
            if np.any(final_probabilities < 0.0) or final_probabilities.sum() <= 0.0:
                raise ValueError("dr_scenario_probabilities must be non-negative and non-zero")
            final_probabilities = final_probabilities / final_probabilities.sum()
            progress = self.episode_dr_progress
            probabilities = progress * final_probabilities
            probabilities[0] += 1.0 - progress
            probabilities = probabilities / probabilities.sum()
        self.dr_scenario_probabilities = probabilities
        self.dr_scenario = str(self.rng.choice(scenario_names, p=probabilities))

    def set_dr_progress(self, progress):
        self.dr_progress = float(np.clip(progress, 0.0, 1.0))

    def _store_nominal_model_params(self):
        self.nominal_gravity = self.model.opt.gravity.copy()
        self.nominal_geom_friction = self.model.geom_friction.copy()
        self.nominal_geom_solref = self.model.geom_solref.copy()
        self.nominal_geom_solimp = self.model.geom_solimp.copy()
        self.nominal_geom_size = self.model.geom_size.copy()
        self.nominal_body_mass = self.model.body_mass.copy()
        self.nominal_body_inertia = self.model.body_inertia.copy()
        self.nominal_body_ipos = self.model.body_ipos.copy()
        self.nominal_jnt_range = self.model.jnt_range.copy()
        self.nominal_dof_damping = self.model.dof_damping.copy()
        self.nominal_dof_frictionloss = self.model.dof_frictionloss.copy()
        self.nominal_dof_armature = self.model.dof_armature.copy()

    def _restore_nominal_model_params(self):
        self.model.opt.gravity[:] = self.nominal_gravity
        self.model.geom_friction[:] = self.nominal_geom_friction
        self.model.geom_solref[:] = self.nominal_geom_solref
        self.model.geom_solimp[:] = self.nominal_geom_solimp
        self.model.geom_size[:] = self.nominal_geom_size
        self.model.body_mass[:] = self.nominal_body_mass
        self.model.body_inertia[:] = self.nominal_body_inertia
        self.model.body_ipos[:] = self.nominal_body_ipos
        self.model.jnt_range[:] = self.nominal_jnt_range
        self.model.dof_damping[:] = self.nominal_dof_damping
        self.model.dof_frictionloss[:] = self.nominal_dof_frictionloss
        self.model.dof_armature[:] = self.nominal_dof_armature
        self.joint_low, self.joint_high = self._get_joint_ranges()
        self.pd_low, self.pd_high = self._make_pd_action_bounds()
        self.kp[:] = self.nominal_kp
        self.kd[:] = self.nominal_kd
        self.effort_limits[:] = self.nominal_effort_limits
        self.gear_ratio = self.nominal_gear_ratio

    def _reset_randomized_params(self):
        self._restore_nominal_model_params()
        self.dr_sampled_params = {
            "scenario": self.dr_scenario,
            "scenario_probability": float(
                self.dr_scenario_probabilities[
                    (
                        "normal",
                        "fixed_delay",
                        "longitudinal_gravity",
                        "lateral_gravity",
                        "high_kp_low_kd",
                        "combined_hard",
                    ).index(self.dr_scenario)
                ]
            ),
            "gravity_delta": np.zeros(3, dtype=np.float64),
            "friction_scale": 1.0,
            "mass_scale": 1.0,
            "mass_link_scale_min": 1.0,
            "mass_link_scale_max": 1.0,
            "inertia_scale": 1.0,
            "inertia_link_scale_min": 1.0,
            "inertia_link_scale_max": 1.0,
            "base_com_shift": np.zeros(3, dtype=np.float64),
            "link_com_max_abs": 0.0,
            "damping_scale": 1.0,
            "damping_joint_scale_min": 1.0,
            "damping_joint_scale_max": 1.0,
            "frictionloss_scale": 1.0,
            "frictionloss_joint_scale_min": 1.0,
            "frictionloss_joint_scale_max": 1.0,
            "armature_scale": 1.0,
            "armature_joint_scale_min": 1.0,
            "armature_joint_scale_max": 1.0,
            "joint_limit_max_abs": 0.0,
            "kp_scale": 1.0,
            "kp_joint_scale_min": 1.0,
            "kp_joint_scale_max": 1.0,
            "kd_scale": 1.0,
            "kd_joint_scale_min": 1.0,
            "kd_joint_scale_max": 1.0,
            "effort_scale": 1.0,
            "effort_joint_scale_min": 1.0,
            "effort_joint_scale_max": 1.0,
            "gear_scale": 1.0,
            "foot_friction_scale_min": 1.0,
            "foot_friction_scale_max": 1.0,
            "foot_solref_timeconst_scale_min": 1.0,
            "foot_solref_timeconst_scale_max": 1.0,
            "foot_solref_dampratio_scale_min": 1.0,
            "foot_solref_dampratio_scale_max": 1.0,
            "foot_solimp_width_scale_min": 1.0,
            "foot_solimp_width_scale_max": 1.0,
            "foot_radius_scale_min": 1.0,
            "foot_radius_scale_max": 1.0,
        }
        if not self.cfg.domain_randomization_enable:
            self.body_weight = float(
                np.sum(self.model.body_mass) * np.linalg.norm(self.model.opt.gravity)
            )
            return

        if self.cfg.dr_gravity_std > 0.0:
            gravity_delta = self.rng.normal(
                0.0, self.cfg.dr_gravity_std * self.episode_dr_progress, 3
            )
            if self.dr_scenario == "longitudinal_gravity":
                sigma_lo, sigma_hi = (
                    self.cfg.dr_hard_longitudinal_gravity_sigma_range
                )
                sigma_lo = max(float(sigma_lo), 0.0)
                sigma_hi = max(float(sigma_hi), sigma_lo)
                longitudinal_magnitude = self.rng.uniform(sigma_lo, sigma_hi)
                longitudinal_magnitude *= (
                    self.cfg.dr_gravity_std * self.episode_dr_progress
                )
                longitudinal_sign = -1.0 if self.rng.random() < 0.5 else 1.0
                gravity_delta[0] = longitudinal_sign * longitudinal_magnitude
            elif self.dr_scenario == "lateral_gravity":
                sigma_lo, sigma_hi = self.cfg.dr_hard_lateral_gravity_sigma_range
                sigma_lo = max(float(sigma_lo), 0.0)
                sigma_hi = max(float(sigma_hi), sigma_lo)
                lateral_magnitude = self.rng.uniform(sigma_lo, sigma_hi)
                lateral_magnitude *= (
                    self.cfg.dr_gravity_std * self.episode_dr_progress
                )
                lateral_sign = -1.0 if self.rng.random() < 0.5 else 1.0
                gravity_delta[1] = lateral_sign * lateral_magnitude
            elif self.dr_scenario == "combined_hard":
                sigma_lo, sigma_hi = (
                    self.cfg.dr_combined_longitudinal_gravity_sigma_range
                )
                sigma_lo = max(float(sigma_lo), 0.0)
                sigma_hi = max(float(sigma_hi), sigma_lo)
                longitudinal_magnitude = self.rng.uniform(sigma_lo, sigma_hi)
                longitudinal_magnitude *= (
                    self.cfg.dr_gravity_std * self.episode_dr_progress
                )
                longitudinal_sign = -1.0 if self.rng.random() < 0.5 else 1.0
                gravity_delta[0] = longitudinal_sign * longitudinal_magnitude
            self.model.opt.gravity[:] = self.nominal_gravity + gravity_delta
            self.dr_sampled_params["gravity_delta"] = gravity_delta.copy()
        friction_scale = self._uniform(self.cfg.dr_friction_range)
        mass_scale = self._uniform(self.cfg.dr_mass_range)
        inertia_scale = self._uniform(self.cfg.dr_inertia_range)
        mass_link_scales = self._uniform_array(self.cfg.dr_mass_link_range, self.model.nbody - 1)
        inertia_link_scales = self._uniform_array(
            self.cfg.dr_inertia_link_range, self.model.nbody - 1
        )
        self.model.geom_friction[:] = self.nominal_geom_friction * friction_scale
        self.model.body_mass[1:] = self.nominal_body_mass[1:] * mass_scale * mass_link_scales
        self.model.body_inertia[1:] = (
            self.nominal_body_inertia[1:] * inertia_scale * inertia_link_scales[:, None]
        )
        self.dr_sampled_params["friction_scale"] = friction_scale
        self.dr_sampled_params["mass_scale"] = mass_scale
        self.dr_sampled_params["mass_link_scale_min"] = float(np.min(mass_link_scales))
        self.dr_sampled_params["mass_link_scale_max"] = float(np.max(mass_link_scales))
        self.dr_sampled_params["inertia_scale"] = inertia_scale
        self.dr_sampled_params["inertia_link_scale_min"] = float(np.min(inertia_link_scales))
        self.dr_sampled_params["inertia_link_scale_max"] = float(np.max(inertia_link_scales))

        foot_geom_ids = np.asarray([self.left_foot_geom_id, self.right_foot_geom_id], dtype=np.int32)
        foot_friction_scales = self._uniform_array(self.cfg.dr_foot_friction_range, 2)
        foot_timeconst_scales = self._uniform_array(self.cfg.dr_foot_solref_timeconst_range, 2)
        foot_dampratio_scales = self._uniform_array(
            self.cfg.dr_foot_solref_dampratio_range, 2
        )
        foot_width_scales = self._uniform_array(self.cfg.dr_foot_solimp_width_range, 2)
        foot_radius_scales = self._uniform_array(self.cfg.dr_foot_radius_range, 2)
        self.model.geom_friction[foot_geom_ids, 0] *= foot_friction_scales
        self.model.geom_solref[foot_geom_ids, 0] = np.maximum(
            2.0 * self.cfg.dt,
            self.nominal_geom_solref[foot_geom_ids, 0] * foot_timeconst_scales,
        )
        self.model.geom_solref[foot_geom_ids, 1] = (
            self.nominal_geom_solref[foot_geom_ids, 1] * foot_dampratio_scales
        )
        self.model.geom_solimp[foot_geom_ids, 2] = np.maximum(
            1.0e-6,
            self.nominal_geom_solimp[foot_geom_ids, 2] * foot_width_scales,
        )
        self.model.geom_size[foot_geom_ids, 0] = (
            self.nominal_geom_size[foot_geom_ids, 0] * foot_radius_scales
        )
        for name, values in (
            ("foot_friction_scale", foot_friction_scales),
            ("foot_solref_timeconst_scale", foot_timeconst_scales),
            ("foot_solref_dampratio_scale", foot_dampratio_scales),
            ("foot_solimp_width_scale", foot_width_scales),
            ("foot_radius_scale", foot_radius_scales),
        ):
            self.dr_sampled_params[f"{name}_min"] = float(np.min(values))
            self.dr_sampled_params[f"{name}_max"] = float(np.max(values))

        if self.cfg.dr_base_com_shift > 0.0:
            shift_limit = self.cfg.dr_base_com_shift * self.episode_dr_progress
            shift = self.rng.uniform(
                -shift_limit,
                shift_limit,
                3,
            ).astype(np.float64)
            self.model.body_ipos[self.base_id] = self.nominal_body_ipos[self.base_id] + shift
            self.dr_sampled_params["base_com_shift"] = shift.copy()
        if self.cfg.dr_link_com_shift > 0.0:
            shift_limit = self.cfg.dr_link_com_shift * self.episode_dr_progress
            shifts = self.rng.uniform(
                -shift_limit,
                shift_limit,
                self.model.nbody * 3,
            ).reshape(self.model.nbody, 3)
            shifts[0] = 0.0
            shifts[self.base_id] = 0.0
            self.model.body_ipos[:] = self.model.body_ipos + shifts
            self.dr_sampled_params["link_com_max_abs"] = float(np.max(np.abs(shifts)))

        damping_scale = self._uniform(self.cfg.dr_damping_range)
        frictionloss_scale = self._uniform(self.cfg.dr_frictionloss_range)
        armature_scale = self._uniform(self.cfg.dr_armature_range)
        dof_count = self.act_dim
        damping_joint_scales = self._uniform_array(self.cfg.dr_damping_joint_range, dof_count)
        frictionloss_joint_scales = self._uniform_array(
            self.cfg.dr_frictionloss_joint_range, dof_count
        )
        armature_joint_scales = self._uniform_array(self.cfg.dr_armature_joint_range, dof_count)
        self.model.dof_damping[self.srl_dof_indices] = (
            self.nominal_dof_damping[self.srl_dof_indices]
            * damping_scale
            * damping_joint_scales
        )
        self.model.dof_frictionloss[self.srl_dof_indices] = (
            self.nominal_dof_frictionloss[self.srl_dof_indices]
            * frictionloss_scale
            * frictionloss_joint_scales
        )
        self.model.dof_armature[self.srl_dof_indices] = (
            self.nominal_dof_armature[self.srl_dof_indices]
            * armature_scale
            * armature_joint_scales
        )
        self.dr_sampled_params["damping_scale"] = damping_scale
        self.dr_sampled_params["damping_joint_scale_min"] = float(
            np.min(damping_joint_scales)
        )
        self.dr_sampled_params["damping_joint_scale_max"] = float(
            np.max(damping_joint_scales)
        )
        self.dr_sampled_params["frictionloss_scale"] = frictionloss_scale
        self.dr_sampled_params["frictionloss_joint_scale_min"] = float(
            np.min(frictionloss_joint_scales)
        )
        self.dr_sampled_params["frictionloss_joint_scale_max"] = float(
            np.max(frictionloss_joint_scales)
        )
        self.dr_sampled_params["armature_scale"] = armature_scale
        self.dr_sampled_params["armature_joint_scale_min"] = float(
            np.min(armature_joint_scales)
        )
        self.dr_sampled_params["armature_joint_scale_max"] = float(
            np.max(armature_joint_scales)
        )
        if self.cfg.dr_joint_limit_std > 0.0:
            limit_noise = self.rng.normal(
                0.0,
                self.cfg.dr_joint_limit_std * self.episode_dr_progress,
                (len(self.srl_joint_ids), 2),
            )
            self.dr_sampled_params["joint_limit_max_abs"] = float(np.max(np.abs(limit_noise)))
            for local_idx, joint_id in enumerate(self.srl_joint_ids):
                nominal_low, nominal_high = self.nominal_jnt_range[joint_id]
                low = nominal_low + limit_noise[local_idx, 0]
                high = nominal_high + limit_noise[local_idx, 1]
                if low > high:
                    low, high = high, low
                self.model.jnt_range[joint_id] = (low, high)
            self.joint_low, self.joint_high = self._get_joint_ranges()
            self.pd_low, self.pd_high = self._make_pd_action_bounds()
        if self.dr_scenario in ("high_kp_low_kd", "combined_hard"):
            kp_scale = self._uniform_range_half(
                self.cfg.dr_kp_range, upper_half=True
            )
            kd_scale = self._uniform_range_half(
                self.cfg.dr_kd_range, upper_half=False
            )
        else:
            kp_scale = self._uniform(self.cfg.dr_kp_range)
            kd_scale = self._uniform(self.cfg.dr_kd_range)
        effort_scale = self._uniform(self.cfg.dr_effort_range)
        kp_joint_scales = self._uniform_array(self.cfg.dr_kp_joint_range, self.act_dim)
        kd_joint_scales = self._uniform_array(self.cfg.dr_kd_joint_range, self.act_dim)
        effort_joint_scales = self._uniform_array(self.cfg.dr_effort_joint_range, self.act_dim)
        self.kp[:] = self.nominal_kp * kp_scale * kp_joint_scales
        self.kd[:] = self.nominal_kd * kd_scale * kd_joint_scales
        self.effort_limits[:] = self.nominal_effort_limits * effort_scale * effort_joint_scales
        self.gear_ratio = self.nominal_gear_ratio
        self.dr_sampled_params["kp_scale"] = kp_scale
        self.dr_sampled_params["kp_joint_scale_min"] = float(np.min(kp_joint_scales))
        self.dr_sampled_params["kp_joint_scale_max"] = float(np.max(kp_joint_scales))
        self.dr_sampled_params["kd_scale"] = kd_scale
        self.dr_sampled_params["kd_joint_scale_min"] = float(np.min(kd_joint_scales))
        self.dr_sampled_params["kd_joint_scale_max"] = float(np.max(kd_joint_scales))
        self.dr_sampled_params["effort_scale"] = effort_scale
        self.dr_sampled_params["effort_joint_scale_min"] = float(np.min(effort_joint_scales))
        self.dr_sampled_params["effort_joint_scale_max"] = float(np.max(effort_joint_scales))
        self.dr_sampled_params["gear_scale"] = 1.0
        mujoco.mj_setConst(self.model, self.data)
        self.body_weight = float(
            np.sum(self.model.body_mass) * np.linalg.norm(self.model.opt.gravity)
        )

    def get_domain_randomization_summary(self):
        summary = dict(self.dr_sampled_params)
        summary.update(
            {
                "action_delay_steps": float(self.action_delay_steps),
                "action_delay_physics_steps": int(self.action_delay_physics_steps),
                "action_delay_ms": 1000.0 * self.cfg.dt * self.action_delay_physics_steps,
                "obs_noise_std": float(
                    self.cfg.obs_noise_std * self.episode_dr_progress
                ),
                "action_noise_std": float(
                    self.cfg.action_noise_std * self.episode_dr_progress
                ),
                "velocity_perturbation_prob": float(
                    self.cfg.velocity_perturbation_prob * self.episode_dr_progress
                )
                if self.cfg.velocity_perturbation_enable
                else 0.0,
                "horizontal_force_perturbation_prob": float(
                    self.cfg.horizontal_force_perturbation_prob
                    * self.episode_dr_progress
                )
                if self.cfg.horizontal_force_perturbation_enable
                else 0.0,
                "horizontal_force_perturbation_range": tuple(
                    float(value)
                    for value in self.cfg.horizontal_force_perturbation_range
                ),
                "horizontal_force_perturbation_duration_range": tuple(
                    float(value)
                    for value in self.cfg.horizontal_force_perturbation_duration_range
                ),
                "dr_progress": float(self.episode_dr_progress),
                "scheduled_dr_progress": float(self.dr_progress),
            }
        )
        return summary

    def _reset_action_delay(self):
        if not self.cfg.domain_randomization_enable:
            self.action_delay_physics_steps = 0
            self.action_delay_steps = 0.0
            return
        if self.cfg.action_delay_max_physics_steps is not None:
            configured_max = max(int(self.cfg.action_delay_max_physics_steps), 0)
        else:
            configured_max = (
                max(int(self.cfg.action_delay_max_steps), 0) * self.cfg.decimation
            )

        # Stochastically open the next 5 ms level as curriculum progresses.
        # At full progress this samples uniformly from 0..configured_max.
        scaled_max = configured_max * self.episode_dr_progress
        if self.dr_scenario in ("fixed_delay", "combined_hard") and scaled_max > 0.0:
            # Targeted episodes always use the current curriculum frontier:
            # 5 ms early, 10 ms midway, and 15 ms at full DR.
            self.action_delay_physics_steps = min(
                configured_max, max(1, int(np.floor(scaled_max + 0.5)))
            )
        else:
            active_max = int(np.floor(scaled_max))
            fractional_level = scaled_max - active_max
            if active_max < configured_max and self.rng.random() < fractional_level:
                active_max += 1
            if active_max > 0:
                self.action_delay_physics_steps = int(
                    self.rng.integers(0, active_max + 1)
                )
            else:
                self.action_delay_physics_steps = 0
        self.action_delay_steps = (
            float(self.action_delay_physics_steps) / max(int(self.cfg.decimation), 1)
        )

    def _reset_action_delay_queues(self, target_pos=None, control_action=None):
        if target_pos is None:
            target_pos = self.default_dof_pos
        if control_action is None:
            control_action = np.zeros(self.act_dim, dtype=np.float32)
        target_pos = np.asarray(target_pos, dtype=np.float32)
        control_action = np.asarray(control_action, dtype=np.float32)
        self.delayed_target_queue.clear()
        self.delayed_control_action_queue.clear()
        for _ in range(self.action_delay_physics_steps):
            self.delayed_target_queue.append(target_pos.copy())
            self.delayed_control_action_queue.append(control_action.copy())

    def _get_srl_joint_ids(self):
        names = (
            "left_hip_x_joint",
            "left_hip_y_joint",
            "left_knee_joint",
            "right_hip_x_joint",
            "right_hip_y_joint",
            "right_knee_joint",
        )
        ids = []
        for name in names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Missing joint in XML: {name}")
            ids.append(jid)
        return np.asarray(ids, dtype=np.int32)

    def _get_joint_ranges(self):
        ranges = []
        for jid in self.srl_joint_ids:
            low, high = self.model.jnt_range[jid]
            ranges.append((float(low), float(high)))
        ranges = np.asarray(ranges, dtype=np.float32)
        return np.minimum(ranges[:, 0], ranges[:, 1]), np.maximum(ranges[:, 0], ranges[:, 1])

    def _get_srl_qpos(self):
        return self.data.qpos[self.srl_qpos_indices].astype(np.float32)

    def _get_srl_qvel(self):
        return self.data.qvel[self.srl_dof_indices].astype(np.float32)

    def _make_pd_action_bounds(self):
        pd_low = np.minimum(self.joint_low, self.default_dof_pos).astype(np.float32)
        pd_high = np.maximum(self.joint_high, self.default_dof_pos).astype(np.float32)
        return pd_low, pd_high

    def _action_to_pd_targets(self, action):
        action = np.clip(action * self.cfg.action_scale, -1.0, 1.0).astype(np.float32)
        neg_scale = self.default_dof_pos - self.pd_low
        pos_scale = self.pd_high - self.default_dof_pos
        target = np.where(
            action >= 0.0,
            self.default_dof_pos + action * pos_scale,
            self.default_dof_pos + action * neg_scale,
        )
        return np.clip(target, self.pd_low, self.pd_high).astype(np.float32)

    def _pd_targets_to_action(self, pd_target):
        delta = pd_target - self.default_dof_pos
        neg_scale = self.default_dof_pos - self.pd_low
        pos_scale = self.pd_high - self.default_dof_pos
        scale = np.where(delta >= 0.0, pos_scale, neg_scale)
        return np.clip(delta / (scale * self.cfg.action_scale + 1e-8), -1.0, 1.0).astype(np.float32)

    def _estimate_pd_torques(self, target_pos):
        curr_pos = self._get_srl_qpos()
        curr_vel = self._get_srl_qvel()
        torques = self.kp * (target_pos - curr_pos) - self.kd * curr_vel
        return np.clip(torques, -self.effort_limits, self.effort_limits).astype(np.float32)

    def _get_clean_single_frame_obs(self):
        root_quat = self.data.qpos[3:7]
        root_rot_mat = self.data.xmat[self.base_id].reshape(3, 3)
        euler = quat_to_euler_xyz(root_quat)
        yaw_err = self.cfg.target_yaw - euler[0]
        yaw_err = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))
        euler_err = np.array([yaw_err, -euler[1], -euler[2]], dtype=np.float32)

        local_lin_vel = root_rot_mat.T @ self.data.qvel[0:3]
        local_ang_vel = self.data.qvel[3:6].copy()
        srl_dof_obs = self._get_srl_qpos() - self.default_dof_pos
        srl_dof_vel = self._get_srl_qvel()

        phase_t = (2.0 * np.pi / self.cfg.gait_period) * float(self.phase_counter)
        obs = np.concatenate(
            [
                np.array([self.data.qpos[2]], dtype=np.float32),
                local_lin_vel.astype(np.float32),
                local_ang_vel.astype(np.float32),
                euler_err,
                srl_dof_obs.astype(np.float32),
                (self.cfg.dof_vel_obs_scale * srl_dof_vel).astype(np.float32),
                self.raw_action.astype(np.float32),
                np.array([np.sin(phase_t), np.cos(phase_t)], dtype=np.float32),
            ]
        )
        return np.clip(obs, -self.cfg.clip_obs, self.cfg.clip_obs).astype(np.float32)

    def _add_actor_obs_noise(self, clean_obs):
        obs = np.asarray(clean_obs, dtype=np.float32).copy()
        if not self.cfg.domain_randomization_enable:
            return obs
        progress = self.episode_dr_progress
        if self.cfg.obs_noise_dof_pos_std > 0.0:
            obs[10:16] += self.rng.normal(
                0.0, self.cfg.obs_noise_dof_pos_std * progress, self.act_dim
            ).astype(np.float32)
        if self.cfg.obs_noise_dof_vel_std > 0.0:
            obs[16:22] += self.cfg.dof_vel_obs_scale * self.rng.normal(
                0.0, self.cfg.obs_noise_dof_vel_std * progress, self.act_dim
            ).astype(np.float32)
        if self.cfg.obs_noise_std > 0.0:
            obs += self.rng.normal(
                0.0, self.cfg.obs_noise_std * progress, obs.shape
            ).astype(np.float32)
        return np.clip(obs, -self.cfg.clip_obs, self.cfg.clip_obs).astype(np.float32)

    def _get_stacked_obs(self, history):
        flat_history = np.concatenate(list(history)[::-1])
        task_cmd = np.array(
            [self.cfg.target_vel_x, self.cfg.target_ang_vel_z, self.cfg.target_height],
            dtype=np.float32,
        )
        return np.concatenate([flat_history, task_cmd]).astype(np.float32)

    def _get_srl_end_body_pos(self):
        return np.vstack(
            [
                self.data.xpos[self.left_foot_id].copy(),
                self.data.xpos[self.right_foot_id].copy(),
            ]
        )

    def _get_foot_contact_forces(self):
        left_force = 0.0
        right_force = 0.0
        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self.floor_geom_id:
                other_geom = geom2
            elif geom2 == self.floor_geom_id:
                other_geom = geom1
            else:
                continue
            if other_geom != self.left_foot_geom_id and other_geom != self.right_foot_geom_id:
                continue
            force6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_idx, force6)
            normal_force = abs(float(force6[0]))
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

    def _foot_clearance_penalty(self):
        srl_end_body_pos = self._get_srl_end_body_pos()
        delta = srl_end_body_pos - self.prev_srl_end_body_pos
        dx, dy, dz = delta[:, 0], delta[:, 1], delta[:, 2]
        pz = srl_end_body_pos[:, 2]
        v_xy = np.sqrt(dx * dx + dy * dy) / self.control_dt
        v_xy = np.where(v_xy < 0.8, 0.0, v_xy)
        clearance_penalty = float(np.sum((self.cfg.foot_clearance - pz) ** 2 * v_xy))
        self.prev_srl_end_body_pos[:] = srl_end_body_pos
        return clearance_penalty

    def _motor_cost_terms(self):
        tau = self.last_applied_torques.astype(np.float32)
        qd = self._get_srl_qvel()
        abs_tau = np.abs(tau)
        peak_ratio = float(np.max(abs_tau / max(self.cfg.srl_peak_nm, 1e-6)))
        self.srl_peak_ratio_window = max(
            self._srl_peak_decay * self.srl_peak_ratio_window,
            peak_ratio,
        )
        tau2 = float(np.mean((abs_tau / max(self.cfg.srl_rated_nm, 1e-6)) ** 2))
        self.srl_tau2_ema = self._srl_thermal_gamma * self.srl_tau2_ema + (
            1.0 - self._srl_thermal_gamma
        ) * tau2
        power = np.abs(tau * qd)
        power_ratio = float(np.mean(power / max(self.cfg.srl_rated_w, 1e-6)))
        peak_cost = max(self.srl_peak_ratio_window - self.cfg.srl_peak_start_ratio, 0.0) ** 2
        thermal_cost = max(self.srl_tau2_ema - self.cfg.srl_thermal_start, 0.0) ** 2
        power_cost = max(power_ratio - self.cfg.srl_power_start_ratio, 0.0) ** 2
        total = (
            self.cfg.srl_peak_cost_scale * peak_cost
            + self.cfg.srl_thermal_cost_scale * thermal_cost
            + self.cfg.srl_power_cost_scale * power_cost
        )
        return total, peak_cost, thermal_cost, power_cost

    def _compute_reward(self, obs, action):
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

        action_cost = float(np.sum(action ** 2))
        srl_dof_pos[0] *= 3.0
        srl_dof_pos[3] *= 3.0
        dof_pos_cost = float(np.sum(srl_dof_pos ** 2))
        dof_vel_cost = float(np.sum(srl_dof_vel_scaled ** 2))

        frame = 30
        act_dim = self.act_dim
        dof_vel_prev_raw = obs[16 + frame:16 + frame + act_dim]
        dof_acc = srl_dof_vel_scaled - dof_vel_prev_raw
        dof_acc_reward_raw = float(np.exp(-2.0 * np.sum(dof_acc ** 2)))
        dof_acc_reward = warmup * dof_acc_reward_raw + (1.0 - warmup)

        actions_prev_raw = obs[22 + frame:22 + frame + act_dim]
        actions_prev_prev_raw = obs[22 + 2 * frame:22 + 2 * frame + act_dim]
        actions_prev = warmup * actions_prev_raw + (1.0 - warmup) * action
        actions_prev_prev = warmup * actions_prev_prev_raw + (1.0 - warmup) * actions_prev
        actions_rate = warmup * float(np.sum((action - actions_prev) ** 2))
        actions_smoothness = warmup * float(
            np.sum((action - 2.0 * actions_prev + actions_prev_prev) ** 2)
        )

        yaw_err, pitch_err, roll_err = euler_err
        ori_cost = 0.2 * yaw_err * yaw_err + pitch_err * pitch_err + roll_err * roll_err
        orientation_reward = float(np.exp(-8.0 * ori_cost))

        pelvis_height_error = root_h - target_pelvis_height
        pelvis_height_reward = float(np.exp(-6.0 * (3.0 * pelvis_height_error) ** 2))

        wx, wy, wz = local_ang_vel
        wz_err = wz - target_ang_vel_z
        ang_vel_cost = 3.0 * (wx * wx + wy * wy) + 0.5 * wz_err * wz_err
        ang_vel_tracking_reward = float(np.exp(-2.0 * ang_vel_cost))

        srl_end_body_pos = self._get_srl_end_body_pos()
        left_foot_height = float(srl_end_body_pos[0, 2])
        right_foot_height = float(srl_end_body_pos[1, 2])
        no_feet_on_ground = (
            left_foot_height > self.cfg.foot_contact_height
            and right_foot_height > self.cfg.foot_contact_height
        )
        no_fly_coef = 5.0 if target_vel_x < 0.1 else 1.0
        no_fly_penalty = no_fly_coef * float(no_feet_on_ground)

        clearance_penalty = self._foot_clearance_penalty()
        root_rot_mat = self.data.xmat[self.base_id].reshape(3, 3)
        root_pos = self.data.xpos[self.base_id]
        local_foot_pos = (srl_end_body_pos - root_pos) @ root_rot_mat
        left_foot_y = float(local_foot_pos[0, 1])
        right_foot_y = float(local_foot_pos[1, 1])
        lateral_distance = abs(left_foot_y - right_foot_y)
        lateral_below = max(self.cfg.lateral_min_distance - lateral_distance, 0.0)
        lateral_above = max(lateral_distance - self.cfg.lateral_max_distance, 0.0)
        lateral_symmetry = left_foot_y + right_foot_y
        lateral_target_error = (
            lateral_distance - self.cfg.lateral_target_distance
            if self.cfg.lateral_target_distance > 0.0
            else 0.0
        )
        lateral_target_penalty = (
            self.cfg.lateral_target_weight * lateral_target_error * lateral_target_error
        )
        lateral_penalty = float(
            lateral_below
            + lateral_above
            + self.cfg.lateral_symmetry_weight * lateral_symmetry * lateral_symmetry
            + lateral_target_penalty
        )

        phase_t = (2.0 * np.pi / self.cfg.gait_period) * float(self.phase_counter)
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
        gait_similarity_penalty = float(
            stance_miss_left + stance_miss_right + flying_miss_left + flying_miss_right
        )

        current_local_ang_vel = local_ang_vel.astype(np.float32)
        base_ang_acc = (current_local_ang_vel - self.prev_local_ang_vel) / self.control_dt
        self.prev_local_ang_vel[:] = current_local_ang_vel
        base_wobble_penalty = pitch_err * pitch_err + roll_err * roll_err + 0.5 * (wx * wx + wy * wy)
        pitch_wobble_penalty = pitch_err * pitch_err
        roll_wobble_penalty = roll_err * roll_err + self.cfg.roll_rate_weight * wx * wx
        left_foot_force_bw = self.last_left_foot_force_max / max(self.body_weight, 1e-6)
        right_foot_force_bw = self.last_right_foot_force_max / max(self.body_weight, 1e-6)
        left_foot_force_excess = max(left_foot_force_bw - self.cfg.foot_force_threshold_bw, 0.0)
        right_foot_force_excess = max(right_foot_force_bw - self.cfg.foot_force_threshold_bw, 0.0)
        foot_impact_penalty = warmup * float(
            left_foot_force_excess ** self.cfg.foot_force_penalty_power
            + right_foot_force_excess ** self.cfg.foot_force_penalty_power
        )
        motor_cost, peak_cost, thermal_cost, power_cost = self._motor_cost_terms()

        reward = (
            self.cfg.alive_reward_scale * alive_reward
            + self.cfg.progress_reward_scale * progress_reward
            + self.cfg.vel_tracking_reward_scale * vel_tracking_reward
            + self.cfg.tracking_ang_vel_reward_scale * ang_vel_tracking_reward
            + self.cfg.orientation_reward_scale * orientation_reward
            + self.cfg.pelvis_height_reward_scale * pelvis_height_reward
            - self.cfg.torques_cost_scale * action_cost
            - self.cfg.dof_vel_cost_scale * dof_vel_cost
            - self.cfg.dof_pos_cost_scale * dof_pos_cost
            + self.cfg.dof_acc_cost_scale * dof_acc_reward
            - self.cfg.actions_rate_scale * actions_rate
            - self.cfg.actions_smoothness_scale * actions_smoothness
            - self.cfg.no_fly_penalty_scale * no_fly_penalty
            - self.cfg.gait_similarity_penalty_scale * gait_similarity_penalty
            - self.cfg.clearance_penalty_scale * clearance_penalty
            - self.cfg.lateral_distance_penalty_scale * lateral_penalty
            - self.cfg.srl_motor_cost_scale * motor_cost
            - self.cfg.base_wobble_penalty_scale * base_wobble_penalty
            - self.cfg.pitch_wobble_penalty_scale * pitch_wobble_penalty
            - self.cfg.roll_wobble_penalty_scale * roll_wobble_penalty
            - self.cfg.base_ang_acc_penalty_scale * float(np.sum(base_ang_acc ** 2))
            - self.cfg.yaw_drift_penalty_scale * float(yaw_err * yaw_err)
            - self.cfg.foot_impact_penalty_scale * foot_impact_penalty
        )

        terminated = bool(root_h < self.cfg.termination_height)
        truncated = bool(self.rl_step_counter >= self.cfg.max_episode_steps)
        applied_termination_penalty = 0.0
        if terminated:
            applied_termination_penalty = float(self.cfg.termination_penalty)
            reward += applied_termination_penalty

        info = {
            "reward_total": float(reward),
            "penalty_termination": applied_termination_penalty,
            "reward_alive": float(alive_reward),
            "reward_progress": float(progress_reward),
            "reward_vel_tracking": vel_tracking_reward,
            "reward_ang_vel_tracking": ang_vel_tracking_reward,
            "reward_orientation": orientation_reward,
            "reward_pelvis_height": pelvis_height_reward,
            "reward_height": pelvis_height_reward,
            "reward_dof_acc": dof_acc_reward,
            "penalty_torques": action_cost,
            "penalty_dof_vel": dof_vel_cost,
            "penalty_dof_pos": dof_pos_cost,
            "penalty_no_fly": no_fly_penalty,
            "penalty_gait_similarity": gait_similarity_penalty,
            "penalty_clearance": clearance_penalty,
            "penalty_lateral": lateral_penalty,
            "foot_lateral_distance": lateral_distance,
            "left_foot_local_y": left_foot_y,
            "right_foot_local_y": right_foot_y,
            "foot_lateral_target_error": lateral_target_error,
            "penalty_lateral_target": lateral_target_penalty,
            "penalty_actions_rate": actions_rate,
            "penalty_actions_smoothness": actions_smoothness,
            "penalty_motor": motor_cost,
            "penalty_motor_peak": peak_cost,
            "penalty_motor_thermal": thermal_cost,
            "penalty_motor_power": power_cost,
            "penalty_base_wobble": base_wobble_penalty,
            "penalty_pitch_wobble": pitch_wobble_penalty,
            "penalty_roll_wobble": roll_wobble_penalty,
            "penalty_base_ang_acc": float(np.sum(base_ang_acc ** 2)),
            "penalty_yaw_drift": float(yaw_err * yaw_err),
            "penalty_foot_impact": foot_impact_penalty,
            "left_foot_force_max": float(self.last_left_foot_force_max),
            "right_foot_force_max": float(self.last_right_foot_force_max),
            "left_foot_force_bw": float(left_foot_force_bw),
            "right_foot_force_bw": float(right_foot_force_bw),
            "root_height": root_h,
            "vel_x": float(local_vel[0]),
            "wx": float(wx),
            "wy": float(wy),
            "wz": float(wz),
            "yaw_err": float(yaw_err),
            "pitch_err": float(pitch_err),
            "roll_err": float(roll_err),
            "pitch": float(-pitch_err),
            "roll": float(-roll_err),
            "target_filter_cutoff_hz": float(self.cfg.action_filter_cutoff_hz),
            "target_filter_enabled": float(self.cfg.action_filter_enable),
            "domain_randomization_enabled": float(self.cfg.domain_randomization_enable),
            "dr_scenario": self.dr_scenario,
            "action_delay_steps": float(self.action_delay_steps),
            "action_delay_physics_steps": float(self.action_delay_physics_steps),
            "action_delay_ms": float(
                1000.0 * self.cfg.dt * self.action_delay_physics_steps
            ),
            "horizontal_force_pulse_x": float(self.horizontal_force_pulse[0]),
            "horizontal_force_pulse_y": float(self.horizontal_force_pulse[1]),
            "horizontal_force_pulse_remaining_steps": float(
                self.horizontal_force_pulse_remaining_steps
            ),
            "horizontal_force_pulse_event_count": float(
                self.horizontal_force_pulse_event_count
            ),
            "gear_ratio": float(self.gear_ratio),
        }
        return float(reward), terminated, truncated, info

    def _get_info(self):
        root_rot_mat = self.data.xmat[self.base_id].reshape(3, 3)
        local_vel = root_rot_mat.T @ self.data.qvel[0:3]
        local_ang_vel = self.data.qvel[3:6]
        return {
            "step": self.rl_step_counter,
            "root_height": float(self.data.qpos[2]),
            "vel_x": float(local_vel[0]),
            "wx": float(local_ang_vel[0]),
            "wy": float(local_ang_vel[1]),
            "wz": float(local_ang_vel[2]),
            "target_vel_x": float(self.cfg.target_vel_x),
            "target_ang_vel_z": float(self.cfg.target_ang_vel_z),
            "target_height": float(self.cfg.target_height),
            "root_height_init": float(self.cfg.root_height),
            "control_backend": self.cfg.control_backend,
            "action_filter_enabled": bool(self.cfg.action_filter_enable),
            "action_filter_cutoff_hz": float(self.cfg.action_filter_cutoff_hz),
            "domain_randomization_enabled": float(self.cfg.domain_randomization_enable),
            "dr_scenario": self.dr_scenario,
            "action_delay_steps": float(self.action_delay_steps),
            "action_delay_physics_steps": float(self.action_delay_physics_steps),
            "action_delay_ms": float(
                1000.0 * self.cfg.dt * self.action_delay_physics_steps
            ),
            "gear_ratio": float(self.gear_ratio),
        }
