from __future__ import annotations

from typing import Optional

import numpy as np

from mujoco_ppo.hri_isaac_dataset import split_policy_srl_obs
from mujoco_ppo.srl_mujoco_hri import (
    SRLMujocoHRIEnv,
    WalkEnvConfig,
)
from mujoco_ppo.srl_mujoco_hri_force_env import SRLMujocoHRIForceEnv as _BaseHRIEnv


class SRLMujocoHRI218Env(SRLMujocoHRIEnv):
    """MuJoCo SRL HRI environment with 218D full-state HRI observation.

    Layout:
        5 * 43D frames + 3D task command = 218D

    Each frame keeps the original MuJoCo 30D SRL observation used by the stable
    single-SRL policy, then appends 13D human/HRI fields:
        [base 30D, humanoid_euler_err 3D, load_cell 6D, human_leg_pitch 4D]
    """

    obs_dim = 218
    act_dim = 6
    policy_frame_dim = 43
    policy_base_frame_dim = 30
    human_frame_dim = 13
    task_dim = 3

    def __init__(self, config: Optional[WalkEnvConfig] = None):
        super().__init__(config or WalkEnvConfig())
        # The replayed Isaac policy obs is still the deployable 198D vector.
        # Keep this buffer sized to what the dataset stores.
        self.current_isaac_srl_obs = np.zeros(198, dtype=np.float32)

    def _sync_isaac_sample_to_env(self):
        if self.current_isaac_sample is None:
            self.current_isaac_srl_obs[:] = 0.0
            self.current_teacher_action[:] = 0.0
            self.current_replay_human_frame[:] = 0.0
            return

        sample = self.current_isaac_sample
        replay_target_vel_x = None
        replay_target_ang_vel_z = None
        replay_target_height = None
        if "srl_obs" in sample:
            srl_obs = sample["srl_obs"].detach().cpu().numpy().astype(np.float32)
            copy_dim = min(self.current_isaac_srl_obs.shape[0], srl_obs.shape[0])
            self.current_isaac_srl_obs[:copy_dim] = srl_obs[:copy_dim]
            if copy_dim < self.current_isaac_srl_obs.shape[0]:
                self.current_isaac_srl_obs[copy_dim:] = 0.0

            parts = split_policy_srl_obs(sample["srl_obs"].detach().cpu(), self.isaac_obs_spec)
            current_frame = parts["frames"][0].numpy().astype(np.float32)
            if self.cfg.use_isaac_human_obs:
                human_frame = current_frame[26:39].copy()
                human_frame[3:9] *= self._hri_ramp_alpha(self.cfg.hri_wrench_ramp_time)
                self.current_replay_human_frame[:] = human_frame
            else:
                self.current_replay_human_frame[:] = 0.0
            task_cmd = parts["task_cmd"].numpy().astype(np.float32)
            replay_target_vel_x = float(task_cmd[0])
            replay_target_ang_vel_z = float(task_cmd[1])
            replay_target_height = float(task_cmd[2])

        if "target_vel_x" in sample:
            replay_target_vel_x = float(sample["target_vel_x"].item())
        if "target_ang_vel_z" in sample:
            replay_target_ang_vel_z = float(sample["target_ang_vel_z"].item())
        if "target_pelvis_height" in sample:
            replay_target_height = float(sample["target_pelvis_height"].item())
        if (
            replay_target_vel_x is not None
            or replay_target_ang_vel_z is not None
            or replay_target_height is not None
        ):
            self._set_isaac_task_command(
                replay_target_vel_x if replay_target_vel_x is not None else self.cfg.target_vel_x,
                replay_target_ang_vel_z if replay_target_ang_vel_z is not None else self.cfg.target_ang_vel_z,
                replay_target_height if replay_target_height is not None else self.cfg.target_height,
            )

        if "raw_mu_srl" in sample:
            self.current_teacher_action[:] = sample["raw_mu_srl"].detach().cpu().numpy().astype(np.float32)

        if self.cfg.apply_isaac_load_cell_wrench and "virtual_load_cell" in sample:
            wrench = sample["virtual_load_cell"].detach().cpu().numpy().astype(np.float32)
            self._set_isaac_load_cell_wrench(wrench)

    def _get_policy_single_frame_obs(self):
        base_frame = _BaseHRIEnv._get_single_frame_obs(self)
        base_policy = base_frame[:30].astype(np.float32)

        if self.current_isaac_sample is not None:
            human_frame = self.current_replay_human_frame.astype(np.float32)
        else:
            human_frame = np.concatenate(
                [
                    np.zeros(3, dtype=np.float32),
                    (self.hri_wrench_local * self.cfg.hri_wrench_obs_scale).astype(np.float32),
                    np.zeros(4, dtype=np.float32),
                ]
            )

        frame = np.concatenate([base_policy, human_frame]).astype(np.float32)
        return np.clip(frame, -self.cfg.clip_obs, self.cfg.clip_obs)


EnvConfig = WalkEnvConfig
SRLMujocoWalkEnv = SRLMujocoHRI218Env
