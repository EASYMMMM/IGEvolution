# 改代码是用于增加了柔性关节的xml训练的

import numpy as np
import torch

from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.SRLEvo.srl_real_bot import (
    SRL_Real_Bot,
    compute_srl_bot_observations,
    compute_srl_bot_observations_mirrored,
)


class SRL_Real_Bot_Compliant(SRL_Real_Bot):
    """Eight-DOF compliant model with the original six-DOF policy interface."""

    DEFAULT_DOF_ORDER = (
        "left_hip_x_joint",
        "left_hip_y_joint",
        "left_hip_connector_y_slide",
        "left_knee_joint",
        "right_hip_x_joint",
        "right_hip_y_joint",
        "right_hip_connector_y_slide",
        "right_knee_joint",
    )
    DEFAULT_ACTUATED_DOF_NAMES = (
        "left_hip_x_joint",
        "left_hip_y_joint",
        "left_knee_joint",
        "right_hip_x_joint",
        "right_hip_y_joint",
        "right_knee_joint",
    )
    DEFAULT_PASSIVE_DOF_NAMES = (
        "left_hip_connector_y_slide",
        "right_hip_connector_y_slide",
    )

    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        env_cfg = cfg["env"]
        if not bool(env_cfg.get("pdControl", True)) or bool(
            env_cfg.get("forceControl", False)
        ):
            raise ValueError(
                "SRL_Real_Bot_Compliant currently requires pdControl=True and "
                "forceControl=False"
            )

        self.expected_dof_names = tuple(
            env_cfg.get("compliant_dof_order", self.DEFAULT_DOF_ORDER)
        )
        self.actuated_dof_names = tuple(
            env_cfg.get(
                "actuated_dof_names", self.DEFAULT_ACTUATED_DOF_NAMES
            )
        )
        self.passive_dof_names = tuple(
            env_cfg.get("passive_dof_names", self.DEFAULT_PASSIVE_DOF_NAMES)
        )
        if len(self.expected_dof_names) != 8:
            raise ValueError("compliant_dof_order must contain exactly 8 names")
        if len(self.actuated_dof_names) != 6:
            raise ValueError("actuated_dof_names must contain exactly 6 names")
        if len(self.passive_dof_names) != 2:
            raise ValueError("passive_dof_names must contain exactly 2 names")
        if set(self.actuated_dof_names) & set(self.passive_dof_names):
            raise ValueError("Actuated and passive DOF names must be disjoint")
        if set(self.expected_dof_names) != (
            set(self.actuated_dof_names) | set(self.passive_dof_names)
        ):
            raise ValueError(
                "compliant_dof_order must contain all actuated and passive DOFs"
            )

        self._actuated_dof_ids_list = [
            self.expected_dof_names.index(name)
            for name in self.actuated_dof_names
        ]
        self._passive_dof_ids_list = [
            self.expected_dof_names.index(name) for name in self.passive_dof_names
        ]

        active_defaults = list(env_cfg["default_joint_angles"])
        if len(active_defaults) == len(self.expected_dof_names):
            active_defaults = [
                active_defaults[index] for index in self._actuated_dof_ids_list
            ]
        if len(active_defaults) != len(self.actuated_dof_names):
            raise ValueError(
                "default_joint_angles must contain 6 actuated-joint values"
            )
        self.actuated_default_joint_angles = [float(v) for v in active_defaults]

        active_efforts = list(
            env_cfg.get(
                "srl_effort_limits",
                [env_cfg.get("srl_max_effort", 400.0)] * 6,
            )
        )
        if len(active_efforts) == len(self.expected_dof_names):
            active_efforts = [
                active_efforts[index] for index in self._actuated_dof_ids_list
            ]
        if len(active_efforts) != len(self.actuated_dof_names):
            raise ValueError("srl_effort_limits must contain 6 actuator limits")
        self.actuated_effort_limits = [float(v) for v in active_efforts]

        self.passive_rest_positions = self._pair_from_config(
            env_cfg.get("passive_dof_rest_positions", [0.0, 0.0]),
            "passive_dof_rest_positions",
        )
        self.passive_stiffness = self._pair_from_config(
            env_cfg.get("passive_dof_stiffness", 94000.0),
            "passive_dof_stiffness",
        )
        self.passive_damping = self._pair_from_config(
            env_cfg.get("passive_dof_damping", 200.0),
            "passive_dof_damping",
        )
        self.passive_effort_limits = self._pair_from_config(
            env_cfg.get("passive_dof_effort_limits", 1000.0),
            "passive_dof_effort_limits",
        )
        self.passive_max_velocity = self._pair_from_config(
            env_cfg.get("passive_dof_max_velocity", 2.0),
            "passive_dof_max_velocity",
        )

        full_defaults = self._expand_to_full(
            self.actuated_default_joint_angles, self.passive_rest_positions
        )
        full_efforts = self._expand_to_full(
            self.actuated_effort_limits, self.passive_effort_limits
        )

        # The parent owns the simulator tensors and therefore receives full
        # eight-DOF initialization data. The policy-facing values remain six-D.
        env_cfg["default_joint_angles"] = full_defaults
        env_cfg["srl_effort_limits"] = full_efforts

        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )

        # Motor thermal statistics apply only to the six physical actuators.
        self.srl_tau2_ema = torch.zeros(
            (self.num_envs, len(self.actuated_dof_names)), device=self.device
        )

    @staticmethod
    def _pair_from_config(value, name):
        if isinstance(value, (int, float)):
            return [float(value), float(value)]
        values = [float(v) for v in value]
        if len(values) != 2:
            raise ValueError("{} must be a scalar or contain 2 values".format(name))
        return values

    def _expand_to_full(self, active_values, passive_values):
        by_name = dict(zip(self.actuated_dof_names, active_values))
        by_name.update(dict(zip(self.passive_dof_names, passive_values)))
        return [float(by_name[name]) for name in self.expected_dof_names]

    def create_sim(self):
        self.up_axis_idx = 2
        num_physical_dofs = len(self.expected_dof_names)
        self.torques = torch.zeros(
            self.num_envs,
            num_physical_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            num_physical_dofs, dtype=torch.float, device=self.device
        )
        self.d_gains = torch.zeros_like(self.p_gains)
        self.torque_limits = torch.zeros_like(self.p_gains)
        self.actuated_dof_ids = torch.tensor(
            self._actuated_dof_ids_list,
            dtype=torch.long,
            device=self.device,
        )
        self.passive_dof_ids = torch.tensor(
            self._passive_dof_ids_list,
            dtype=torch.long,
            device=self.device,
        )

        self.sim = VecTask.create_sim(
            self,
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        SRL_Real_Bot._create_envs(
            self,
            self.num_envs,
            self.cfg["env"]["envSpacing"],
            int(np.sqrt(self.num_envs)),
        )
        self._validate_loaded_dofs()
        self._configure_passive_dofs()

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _validate_loaded_dofs(self):
        loaded = tuple(self._dof_names)
        if self.num_dof != len(self.expected_dof_names) or loaded != tuple(
            self.expected_dof_names
        ):
            raise RuntimeError(
                "Compliant asset DOF layout mismatch. Expected {}, loaded {}".format(
                    self.expected_dof_names, loaded
                )
            )
        print("Compliant DOF layout verified (physical=8, actuated=6, passive=2)")
        for index, name in enumerate(loaded):
            role = "actuated" if index in self._actuated_dof_ids_list else "passive"
            print("  DOF {}: {} ({})".format(index, name, role))

    def _configure_passive_dofs(self):
        for env_ptr, handle in zip(self.envs, self.humanoid_handles):
            props = self.gym.get_actor_dof_properties(env_ptr, handle)
            for pair_index, dof_index in enumerate(self._passive_dof_ids_list):
                props["driveMode"][dof_index] = gymapi.DOF_MODE_POS
                props["stiffness"][dof_index] = self.passive_stiffness[pair_index]
                props["damping"][dof_index] = self.passive_damping[pair_index]
                props["effort"][dof_index] = self.passive_effort_limits[pair_index]
                props["velocity"][dof_index] = self.passive_max_velocity[pair_index]
            self.gym.set_actor_dof_properties(env_ptr, handle, props)

        reference = self.gym.get_actor_dof_properties(
            self.envs[0], self.humanoid_handles[0]
        )
        self.p_gains[:] = torch.as_tensor(
            reference["stiffness"], dtype=torch.float, device=self.device
        )
        self.d_gains[:] = torch.as_tensor(
            reference["damping"], dtype=torch.float, device=self.device
        )
        self.torque_limits[:] = torch.as_tensor(
            reference["effort"], dtype=torch.float, device=self.device
        )
        for pair_index, dof_index in enumerate(self._passive_dof_ids_list):
            print(
                "  passive {}: rest={:.6f} m K={:.1f} N/m D={:.1f} Ns/m "
                "limit={:.1f} N".format(
                    self._dof_names[dof_index],
                    self.passive_rest_positions[pair_index],
                    float(reference["stiffness"][dof_index]),
                    float(reference["damping"][dof_index]),
                    float(reference["effort"][dof_index]),
                )
            )

    def _build_pd_action_offset_scale(self):
        self._pd_action_offset = torch.tensor(
            self.default_joint_angles, device=self.device, dtype=torch.float
        ).unsqueeze(0).repeat(self.num_envs, 1)
        margin = float(self.cfg["env"].get("soft_joint_limit_margin", 0.0))
        action_low = (self.dof_limits_lower + margin).unsqueeze(0)
        action_high = (self.dof_limits_upper - margin).unsqueeze(0)
        self._pd_action_low = torch.min(action_low, self._pd_action_offset)
        self._pd_action_high = torch.max(action_high, self._pd_action_offset)

        passive_rest = torch.tensor(
            self.passive_rest_positions, device=self.device, dtype=torch.float
        )
        self._pd_action_offset[:, self.passive_dof_ids] = passive_rest
        self._pd_action_low[:, self.passive_dof_ids] = passive_rest
        self._pd_action_high[:, self.passive_dof_ids] = passive_rest

    def _action_to_pd_targets(self, action):
        if action.ndim != 2 or action.shape[1] != len(self.actuated_dof_names):
            raise RuntimeError(
                "Compliant policy action must have shape [N, 6], got {}".format(
                    tuple(action.shape)
                )
            )
        action = torch.clamp(action * self.action_scale, -1.0, 1.0)
        offset = self._pd_action_offset[:, self.actuated_dof_ids]
        low = self._pd_action_low[:, self.actuated_dof_ids]
        high = self._pd_action_high[:, self.actuated_dof_ids]
        negative_scale = offset - low
        positive_scale = high - offset
        active_targets = torch.where(
            action >= 0.0,
            offset + action * positive_scale,
            offset + action * negative_scale,
        )

        targets = self._pd_action_offset.clone()
        targets[:, self.actuated_dof_ids] = torch.max(
            torch.min(active_targets, high), low
        )
        return targets

    def _srl_pd_targets_to_action(self, pd_target):
        active_target = pd_target[:, self.actuated_dof_ids]
        offset = self._pd_action_offset[:, self.actuated_dof_ids]
        low = self._pd_action_low[:, self.actuated_dof_ids]
        high = self._pd_action_high[:, self.actuated_dof_ids]
        delta = active_target - offset
        negative_scale = offset - low
        positive_scale = high - offset
        scale = torch.where(delta >= 0.0, positive_scale, negative_scale)
        return torch.clamp(
            delta / (scale * self.action_scale + 1e-8), -1.0, 1.0
        )

    def _reset_srl_action_filter(self, env_ids):
        q0 = self.dof_pos[env_ids].clone()
        passive_rest = torch.tensor(
            self.passive_rest_positions, device=self.device, dtype=q0.dtype
        )
        q0[:, self.passive_dof_ids] = passive_rest
        self.srl_lpf_x1[env_ids] = q0
        self.srl_lpf_x2[env_ids] = q0
        self.srl_lpf_y1[env_ids] = q0
        self.srl_lpf_y2[env_ids] = q0

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        env_ids = env_ids.reshape(-1).to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        env_grid = env_ids.unsqueeze(1).expand(-1, self.passive_dof_ids.numel())
        passive_grid = self.passive_dof_ids.unsqueeze(0).expand(
            env_ids.numel(), -1
        )
        passive_rest = torch.tensor(
            self.passive_rest_positions,
            device=self.device,
            dtype=self.dof_pos.dtype,
        ).unsqueeze(0).expand(env_ids.numel(), -1)
        self.dof_pos[env_grid, passive_grid] = passive_rest
        self.dof_vel[env_grid, passive_grid] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self._refresh_sim_tensors()

    def _compute_srl_motor_costs(self):
        tau = torch.index_select(
            self.dof_force_tensor, dim=1, index=self.actuated_dof_ids
        )
        tau_abs = torch.abs(tau)

        tau_ratio_peak = tau_abs / self.srl_peak_nm
        peak_ratio_inst = torch.max(tau_ratio_peak, dim=1).values
        self.srl_peak_ratio_window = torch.maximum(
            peak_ratio_inst,
            self.srl_peak_ratio_window * self._srl_peak_decay,
        )
        peak_cost = torch.clamp(
            (self.srl_peak_ratio_window - self.srl_peak_start_ratio)
            / (1.0 - self.srl_peak_start_ratio),
            min=0.0,
        ) ** 2

        tau_ratio_rated = tau_abs / self.srl_rated_nm
        gamma = self._srl_thermal_gamma
        self.srl_tau2_ema = gamma * self.srl_tau2_ema + (
            1.0 - gamma
        ) * (tau_ratio_rated ** 2)
        thermal_cost_per_dof = torch.clamp(
            (self.srl_tau2_ema - self.srl_thermal_start)
            / (1.0 - self.srl_thermal_start),
            min=0.0,
        ) ** 2
        thermal_cost = torch.mean(thermal_cost_per_dof, dim=1)

        qd = torch.index_select(
            self.dof_vel, dim=1, index=self.actuated_dof_ids
        )
        power_ratio = torch.abs(tau * qd) / self.srl_rated_w
        power_cost_per_dof = torch.clamp(
            (power_ratio - self.srl_power_start_ratio)
            / (1.0 - self.srl_power_start_ratio),
            min=0.0,
        ) ** 2
        power_cost = torch.mean(power_cost_per_dof, dim=1)
        return peak_cost, thermal_cost, power_cost

    def _compute_srl_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self.srl_root_states
            root_states[:, 3:7] = self.root_states[:, 3:7]
            root_states[:, 10:13] = self.root_states[:, 10:13]
            dof_pos = torch.index_select(
                self.dof_pos, dim=1, index=self.actuated_dof_ids
            )
            dof_vel = torch.index_select(
                self.dof_vel, dim=1, index=self.actuated_dof_ids
            )
            dof_force = torch.index_select(
                self.dof_force_tensor, dim=1, index=self.actuated_dof_ids
            )
            progress_buf = self.progress_buf
            phase_buf = self.phase_buf
            initial_dof_pos = torch.index_select(
                self.initial_dof_pos, dim=1, index=self.actuated_dof_ids
            )
            gravity_vec = self.gravity_vec
            actions = self.actions
            targets = self.targets
            potentials = self.potentials
            target_vel_x = self.target_vel_x
            target_yaw = self.target_yaw
        else:
            root_states = self.srl_root_states[env_ids]
            root_states[:, 3:7] = self.root_states[env_ids, 3:7]
            root_states[:, 10:13] = self.root_states[env_ids, 10:13]
            dof_pos = torch.index_select(
                self.dof_pos[env_ids], dim=1, index=self.actuated_dof_ids
            )
            dof_vel = torch.index_select(
                self.dof_vel[env_ids], dim=1, index=self.actuated_dof_ids
            )
            dof_force = torch.index_select(
                self.dof_force_tensor[env_ids],
                dim=1,
                index=self.actuated_dof_ids,
            )
            progress_buf = self.progress_buf[env_ids]
            phase_buf = self.phase_buf[env_ids]
            initial_dof_pos = torch.index_select(
                self.initial_dof_pos[env_ids],
                dim=1,
                index=self.actuated_dof_ids,
            )
            gravity_vec = self.gravity_vec[env_ids]
            actions = self.actions[env_ids]
            targets = self.targets[env_ids]
            potentials = self.potentials[env_ids]
            target_vel_x = self.target_vel_x[env_ids]
            target_yaw = self.target_yaw[env_ids]

        obs, potentials, prev_potentials = compute_srl_bot_observations(
            progress_buf,
            phase_buf,
            initial_dof_pos,
            root_states,
            dof_pos,
            dof_vel,
            target_yaw,
            dof_force,
            gravity_vec,
            actions,
            self.obs_scales_tensor,
            targets,
            potentials,
            self.control_dt,
            target_vel_x,
            self.gait_period,
        )
        obs_mirrored = compute_srl_bot_observations_mirrored(
            progress_buf,
            phase_buf,
            self.mirror_mat_srl_dof,
            initial_dof_pos,
            root_states,
            dof_pos,
            dof_vel,
            target_yaw,
            dof_force,
            gravity_vec,
            actions,
            self.obs_scales_tensor,
            targets,
            potentials,
            self.control_dt,
            target_vel_x,
            self.gait_period,
        )
        return obs, obs_mirrored, potentials, prev_potentials

    def post_physics_step(self):
        super().post_physics_step()
        passive_pos = torch.index_select(
            self.dof_pos, dim=1, index=self.passive_dof_ids
        )
        passive_vel = torch.index_select(
            self.dof_vel, dim=1, index=self.passive_dof_ids
        )
        self.extras["passive_dof_abs_pos_max"] = passive_pos.abs().max()
        self.extras["passive_dof_abs_vel_max"] = passive_vel.abs().max()
        self.extras["srl_torques"] = torch.index_select(
            self.torques[0], dim=0, index=self.actuated_dof_ids
        )
