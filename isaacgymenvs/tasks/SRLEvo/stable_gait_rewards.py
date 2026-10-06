import torch

class StableGaitRewardMixin:
    """Shared, opt-in gait penalties; leaves observation layout to each task."""

    def __init__(self, cfg, *args, **kwargs):
        gait_cfg = cfg["env"].get("stable_gait_reward", {})
        self.stable_gait_reward_enable = bool(gait_cfg.get("enable", False))
        self.stable_gait_roll_scale = float(gait_cfg.get("roll_scale", 6.0))
        self.stable_gait_roll_rate_weight = float(
            gait_cfg.get("roll_rate_weight", 0.5)
        )
        self.stable_gait_hip_x_velocity_scale = float(
            gait_cfg.get("hip_x_velocity_scale", 0.001)
        )
        self.stable_gait_hip_x_velocity_deadband = float(
            gait_cfg.get("hip_x_velocity_deadband", 0.2)
        )
        self.stable_gait_swing_lateral_velocity_scale = float(
            gait_cfg.get("swing_lateral_velocity_scale", 0.5)
        )
        self.stable_gait_swing_lateral_velocity_deadband = float(
            gait_cfg.get("swing_lateral_velocity_deadband", 0.2)
        )
        self.stable_gait_foot_channel_scale = float(
            gait_cfg.get("foot_channel_scale", 20.0)
        )
        self.stable_gait_foot_channel_center = float(
            gait_cfg.get("foot_channel_center", 0.17)
        )
        self.stable_gait_foot_channel_half_width = float(
            gait_cfg.get("foot_channel_half_width", 0.07)
        )
        self.stable_gait_recovery_channel_half_width = float(
            gait_cfg.get("recovery_channel_half_width", 0.12)
        )
        self.stable_gait_foot_separation_target = float(
            gait_cfg.get("foot_separation_target", 0.34)
        )
        self.stable_gait_foot_separation_scale = float(
            gait_cfg.get("foot_separation_scale", 0.25)
        )
        self.stable_gait_landing_width_enable = bool(
            gait_cfg.get("landing_width_enable", False)
        )
        self.stable_gait_landing_min_separation = float(
            gait_cfg.get("landing_min_separation", 0.34)
        )
        self.stable_gait_landing_max_separation = float(
            gait_cfg.get("landing_max_separation", 0.45)
        )
        self.stable_gait_landing_narrow_scale = float(
            gait_cfg.get("landing_narrow_scale", 10.0)
        )
        self.stable_gait_landing_wide_scale = float(
            gait_cfg.get("landing_wide_scale", 0.5)
        )
        self.stable_gait_landing_contact_height = float(
            gait_cfg.get("landing_contact_height", 0.055)
        )
        self.stable_gait_landing_approach_height = float(
            gait_cfg.get("landing_approach_height", 0.12)
        )
        self.stable_gait_landing_min_downward_speed = float(
            gait_cfg.get("landing_min_downward_speed", 0.03)
        )
        self.stable_gait_stance_slip_scale = float(
            gait_cfg.get("stance_slip_scale", 0.0)
        )
        self.stable_gait_stance_slip_deadband = float(
            gait_cfg.get("stance_slip_deadband", 0.05)
        )
        self.stable_gait_stance_slip_grace_steps = int(
            gait_cfg.get("stance_slip_grace_steps", 2)
        )
        if self.stable_gait_landing_max_separation <= self.stable_gait_landing_min_separation:
            raise ValueError("landing_max_separation must exceed landing_min_separation")
        if self.stable_gait_landing_approach_height <= self.stable_gait_landing_contact_height:
            raise ValueError("landing_approach_height must exceed landing_contact_height")
        if self.stable_gait_stance_slip_grace_steps < 0:
            raise ValueError("stance_slip_grace_steps must be nonnegative")
        self._stable_gait_contact_age = None
        self.stable_gait_recovery_roll_threshold = float(
            gait_cfg.get("recovery_roll_threshold", 0.08)
        )
        self.stable_gait_recovery_roll_rate_threshold = float(
            gait_cfg.get("recovery_roll_rate_threshold", 0.4)
        )
        self.stable_gait_recovery_multiplier = float(
            gait_cfg.get("recovery_multiplier", 0.25)
        )
        self.stable_gait_turn_rate_threshold = float(
            gait_cfg.get("turn_rate_threshold", 0.15)
        )

        super().__init__(cfg, *args, **kwargs)

    def compute_reward(self, actions):
        super().compute_reward(actions)
        if not self.stable_gait_reward_enable:
            return

        penalties = self._compute_stable_gait_penalties()
        total_penalty = (
            self.stable_gait_roll_scale * penalties["roll"]
            + self.stable_gait_hip_x_velocity_scale * penalties["hip_x_velocity"]
            + self.stable_gait_swing_lateral_velocity_scale
            * penalties["swing_lateral_velocity"]
            + self.stable_gait_foot_channel_scale * penalties["foot_channel"]
            + self.stable_gait_foot_separation_scale
            * penalties["foot_separation"]
            + penalties["landing_width"]
            + self.stable_gait_stance_slip_scale * penalties["stance_slip"]
        )

        # Preserve the base task's exact terminal reward on fall frames.
        active = (self._terminate_buf == 0).to(total_penalty.dtype)
        self.rew_buf -= active * total_penalty

        self.extras["stable_gait/total_penalty"] = total_penalty.mean()
        for name in (
            "roll", "hip_x_velocity", "swing_lateral_velocity",
            "foot_channel", "foot_separation", "foot_separation_m",
        ):
            value = penalties[name]
            self.extras["stable_gait/{}".format(name)] = value.mean()

    def _compute_stable_gait_penalties(self):
        roll = self.full_obs_buf[:, 9]
        roll_rate = self.full_obs_buf[:, 4]
        recovery = (torch.abs(roll) >= self.stable_gait_recovery_roll_threshold) | (
            torch.abs(roll_rate) >= self.stable_gait_recovery_roll_rate_threshold
        )
        turning = (
            torch.abs(self.target_ang_vel_z) >= self.stable_gait_turn_rate_threshold
        )
        relaxed = recovery | turning
        relaxation = torch.where(
            relaxed,
            torch.full_like(roll, self.stable_gait_recovery_multiplier),
            torch.ones_like(roll),
        )

        roll_penalty = roll.square() + self.stable_gait_roll_rate_weight * (
            roll_rate.square()
        )

        hip_x_velocity = self.dof_vel[:, [0, 3]]
        hip_x_excess = torch.clamp(
            torch.abs(hip_x_velocity) - self.stable_gait_hip_x_velocity_deadband,
            min=0.0,
        )
        hip_x_velocity_penalty = relaxation * torch.sum(
            hip_x_excess.square(), dim=1
        )

        feet_pos = self._rigid_body_pos[:, self._srl_end_ids, :]
        feet_vel = self._rigid_body_vel[:, self._srl_end_ids, :]
        root_pos = self.srl_root_states[:, 0:3]
        root_vel = self.srl_root_states[:, 7:10]
        root_ang_vel = self.srl_root_states[:, 10:13]
        root_quat = self.srl_root_states[:, 3:7]

        qx, qy, qz, qw = root_quat.unbind(dim=1)
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy.square() + qz.square()),
        )
        sin_yaw = torch.sin(yaw).unsqueeze(1)
        cos_yaw = torch.cos(yaw).unsqueeze(1)

        feet_rel_pos = feet_pos - root_pos.unsqueeze(1)
        feet_rel_vel = (
            feet_vel
            - root_vel.unsqueeze(1)
            - torch.cross(
                root_ang_vel.unsqueeze(1).expand_as(feet_rel_pos),
                feet_rel_pos,
                dim=2,
            )
        )
        feet_local_y = (
            -sin_yaw * feet_rel_pos[:, :, 0]
            + cos_yaw * feet_rel_pos[:, :, 1]
        )
        feet_local_vy = (
            -sin_yaw * feet_rel_vel[:, :, 0]
            + cos_yaw * feet_rel_vel[:, :, 1]
        )

        swing = (feet_pos[:, :, 2] > 0.055).to(feet_local_vy.dtype)
        lateral_speed_excess = torch.clamp(
            torch.abs(feet_local_vy)
            - self.stable_gait_swing_lateral_velocity_deadband,
            min=0.0,
        )
        swing_lateral_velocity_penalty = relaxation * torch.sum(
            swing * lateral_speed_excess.square(), dim=1
        )

        centers = feet_local_y.new_tensor(
            [
                self.stable_gait_foot_channel_center,
                -self.stable_gait_foot_channel_center,
            ]
        )
        nominal_width = torch.full_like(
            feet_local_y, self.stable_gait_foot_channel_half_width
        )
        recovery_width = torch.full_like(
            feet_local_y, self.stable_gait_recovery_channel_half_width
        )
        channel_width = torch.where(relaxed.unsqueeze(1), recovery_width, nominal_width)
        channel_excess = torch.clamp(
            torch.abs(feet_local_y - centers.unsqueeze(0)) - channel_width,
            min=0.0,
        )
        foot_channel_penalty = relaxation * torch.sum(
            swing * channel_excess.square(), dim=1
        )

        foot_separation = torch.abs(feet_local_y[:, 0] - feet_local_y[:, 1])
        foot_separation_penalty = relaxation * (
            foot_separation - self.stable_gait_foot_separation_target
        ).square()

        landing_width_penalty = torch.zeros_like(foot_separation)
        contact = feet_pos[:, :, 2] <= self.stable_gait_landing_contact_height
        if self.stable_gait_landing_width_enable:
            approaching = (
                (feet_pos[:, :, 2] > self.stable_gait_landing_contact_height)
                & (feet_pos[:, :, 2] <= self.stable_gait_landing_approach_height)
                & (feet_vel[:, :, 2] < -self.stable_gait_landing_min_downward_speed)
            )
            landing_or_double_support = approaching.any(dim=1) | contact.all(dim=1)
            narrow = torch.clamp(
                self.stable_gait_landing_min_separation - foot_separation, min=0.0
            )
            wide = torch.clamp(
                foot_separation - self.stable_gait_landing_max_separation, min=0.0
            )
            landing_width_penalty = (
                relaxation * landing_or_double_support.to(foot_separation.dtype)
                * (self.stable_gait_landing_narrow_scale * narrow.square()
                   + self.stable_gait_landing_wide_scale * wide.square())
            )

        stance_slip_penalty = torch.zeros_like(foot_separation)
        if self.stable_gait_stance_slip_scale > 0.0:
            if self._stable_gait_contact_age is None:
                self._stable_gait_contact_age = torch.zeros_like(contact, dtype=torch.long)
            previous_age = torch.where(
                (self.progress_buf <= 1).unsqueeze(1),
                torch.zeros_like(self._stable_gait_contact_age),
                self._stable_gait_contact_age,
            )
            self._stable_gait_contact_age = torch.where(contact, previous_age + 1, 0)
            established_contact = (
                self._stable_gait_contact_age > self.stable_gait_stance_slip_grace_steps
            )
            foot_world_yaw_vy = (
                -sin_yaw * feet_vel[:, :, 0] + cos_yaw * feet_vel[:, :, 1]
            )
            slip_excess = torch.clamp(
                torch.abs(foot_world_yaw_vy) - self.stable_gait_stance_slip_deadband,
                min=0.0,
            )
            stance_slip_penalty = relaxation * torch.sum(
                established_contact * slip_excess.square(), dim=1
            )

        return {
            "roll": roll_penalty,
            "hip_x_velocity": hip_x_velocity_penalty,
            "swing_lateral_velocity": swing_lateral_velocity_penalty,
            "foot_channel": foot_channel_penalty,
            "foot_separation": foot_separation_penalty,
            "foot_separation_m": foot_separation,
            "landing_width": landing_width_penalty,
            "stance_slip": stance_slip_penalty,
        }
