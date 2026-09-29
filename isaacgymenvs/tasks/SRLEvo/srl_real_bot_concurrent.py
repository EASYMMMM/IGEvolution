import torch

from isaacgymenvs.tasks.SRLEvo.srl_real_bot import SRL_Real_Bot


class SRL_Real_Bot_Concurrent(SRL_Real_Bot):
    """SRL task variant for concurrent policy and state-estimator training."""

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

        cfg["env"]["srl_policy_obs_remove_ids"] = [0, 1, 2, 3]
        cfg["env"]["append_current_privileged_obs"] = True
        super().__init__(cfg, *args, **kwargs)

        if self.num_obs != 137 or self.srl_full_obs_size != 153:
            raise RuntimeError(
                "Concurrent SRL task requires 137-D actor observations and "
                "153-D critic states; got {} and {}".format(
                    self.num_obs, self.srl_full_obs_size
                )
            )

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
        )

        # Preserve the base task's exact terminal reward on fall frames.
        active = (self._terminate_buf == 0).to(total_penalty.dtype)
        self.rew_buf -= active * total_penalty

        self.extras["stable_gait/total_penalty"] = total_penalty.mean()
        for name, value in penalties.items():
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

        return {
            "roll": roll_penalty,
            "hip_x_velocity": hip_x_velocity_penalty,
            "swing_lateral_velocity": swing_lateral_velocity_penalty,
            "foot_channel": foot_channel_penalty,
            "foot_separation": foot_separation_penalty,
            "foot_separation_m": foot_separation,
        }
