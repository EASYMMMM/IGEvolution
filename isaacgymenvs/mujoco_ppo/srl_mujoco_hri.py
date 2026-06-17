import collections
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from mujoco_ppo.hri_isaac_dataset import (
    IsaacHRIChunkDataset,
    IsaacHRIObsSpec,
    IsaacHRISequenceReplay,
    split_policy_srl_obs,
)
from mujoco_ppo.srl_mujoco_hri_force_env import (
    SRLMujocoHRIForceEnv as _BaseHRIEnv,
    WalkEnvConfig as _BaseWalkEnvConfig,
)


@dataclass
class WalkEnvConfig(_BaseWalkEnvConfig):
    # IsaacGym SRL-HRI replay dataset collected by collect_srl_hri_trajectories.py.
    # If None, the environment still returns 198D obs, but human-related fields
    # are zero except for the current MuJoCo virtual load-cell wrench.
    isaac_dataset_dir: Optional[str] = None
    isaac_dataset_pattern: str = "srl_hri_traj_chunk_*.pt"
    isaac_replay_seq_len: int = 1000
    isaac_replay_fields: Tuple[str, ...] = (
        "srl_obs",
        "raw_mu_srl",
        "virtual_load_cell",
        "target_vel_x",
        "target_ang_vel_z",
        "target_pelvis_height",
        "done",
    )
    isaac_replay_cache_size: int = 2
    isaac_replay_avoid_done: bool = True
    apply_isaac_load_cell_wrench: bool = True
    use_isaac_human_obs: bool = True
    isaac_load_cell_force_scale: float = 1.0
    isaac_load_cell_torque_scale: float = 1.0
    isaac_load_cell_force_clip_x: float = 100.0
    isaac_load_cell_force_clip_y: float = 50.0
    isaac_load_cell_force_clip_z: float = 100.0
    hri_wrench_ramp_time: float = 0.0
    hri_command_ramp_time: float = 0.0


class SRLMujocoHRIEnv(_BaseHRIEnv):
    """MuJoCo SRL environment with IsaacGym-compatible 198D HRI policy obs.

    Policy observation layout matches the latest deployable IsaacGym SRL policy:
        5 * 39D frames + 3D task command = 198D

    Each 39D frame removes root height and local root linear velocity, then adds
    replayed human/HRI fields:
        [local_ang_vel(3), euler_err(3), dof_pos(6), dof_vel(6), prev_action(6),
         sin/cos(2), humanoid_euler_err(3), load_cell(6), human_leg_pitch(4)]
    """

    obs_dim = 198
    act_dim = 6
    policy_frame_dim = 39
    policy_base_frame_dim = 26
    human_frame_dim = 13
    task_dim = 3

    def __init__(self, config: Optional[WalkEnvConfig] = None):
        cfg = config or WalkEnvConfig()
        self.policy_obs_history = collections.deque(maxlen=cfg.frame_stack)
        self.isaac_obs_spec = IsaacHRIObsSpec()
        self.isaac_dataset = None
        self.isaac_replay = None
        self.current_isaac_sample = None
        self.current_isaac_srl_obs = np.zeros(self.obs_dim, dtype=np.float32)
        self.current_teacher_action = np.zeros(self.act_dim, dtype=np.float32)
        self.current_replay_human_frame = np.zeros(self.human_frame_dim, dtype=np.float32)
        self.hri_steps_since_reset = 0
        self.hri_wrench_ramp_alpha = 1.0
        self.hri_command_ramp_alpha = 1.0
        self.hri_default_target_vel_x = float(cfg.target_vel_x)
        self.hri_default_target_ang_vel_z = float(cfg.target_ang_vel_z)
        self.hri_default_target_height = float(cfg.target_height)
        self._base_reset_in_progress = False
        super().__init__(cfg)
        self._init_isaac_replay()

    def _init_isaac_replay(self):
        if not self.cfg.isaac_dataset_dir:
            return
        self.isaac_dataset = IsaacHRIChunkDataset(
            self.cfg.isaac_dataset_dir,
            pattern=self.cfg.isaac_dataset_pattern,
            fields=self.cfg.isaac_replay_fields,
            cache_size=self.cfg.isaac_replay_cache_size,
        )
        self.isaac_replay = IsaacHRISequenceReplay(
            self.isaac_dataset,
            seq_len=self.cfg.isaac_replay_seq_len,
            fields=self.cfg.isaac_replay_fields,
            avoid_done=self.cfg.isaac_replay_avoid_done,
        )

    def reset(self, seed=None):
        self._base_reset_in_progress = True
        try:
            _BaseHRIEnv.reset(self, seed=seed)
        finally:
            self._base_reset_in_progress = False

        if self.isaac_replay is not None:
            self.current_isaac_sample = self.isaac_replay.reset()
        else:
            self.current_isaac_sample = None
        self.hri_steps_since_reset = 0
        self.hri_wrench_ramp_alpha = self._hri_ramp_alpha(self.cfg.hri_wrench_ramp_time)
        self.hri_command_ramp_alpha = self._hri_ramp_alpha(self.cfg.hri_command_ramp_time)
        self._sync_isaac_sample_to_env()

        self.policy_obs_history.clear()
        initial_policy_frame = self._get_policy_single_frame_obs()
        for _ in range(self.cfg.frame_stack):
            self.policy_obs_history.append(initial_policy_frame.copy())

        return self._get_stacked_obs(append_current=False), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.cfg.clip_actions, self.cfg.clip_actions)

        policy_action = (
            (1.0 - self.cfg.action_delay_alpha) * action
            + self.cfg.action_delay_alpha * self.last_action
        ).astype(np.float32)

        self.prev_prev_action[:] = self.prev_action
        self.prev_action[:] = self.last_action

        # In replay mode, the current Isaac load-cell wrench is already synced
        # before the action is chosen. In fallback mode, keep the old generators.
        if self.isaac_replay is None:
            self._update_human_trajectory_command()
            self._update_hri_proxy_wrench()

        raw_target_pos = self.default_dof_pos + self.action_scale * policy_action
        target_pos = self._apply_srl_action_filter(raw_target_pos)
        filtered_action = self._pd_targets_to_action(target_pos)
        self.last_action[:] = filtered_action
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
                self._apply_hri_wrench()
                import mujoco

                mujoco.mj_step(self.model, self.data)
                left_force, right_force = self._get_foot_contact_forces()
                left_foot_force_max = max(left_foot_force_max, left_force)
                right_foot_force_max = max(right_foot_force_max, right_force)
        elif self.cfg.torque_update_mode == "physics":
            import mujoco

            for _ in range(self.cfg.decimation):
                torques = self._compute_torques(target_pos)
                torque_rate_sum += float(np.sum((torques - self.prev_torques) ** 2))
                self.prev_torques[:] = torques
                self.last_applied_torques[:] = torques
                self.data.ctrl[:] = torques / self.cfg.gear_ratio
                self._apply_hri_wrench()
                mujoco.mj_step(self.model, self.data)
                left_force, right_force = self._get_foot_contact_forces()
                left_foot_force_max = max(left_foot_force_max, left_force)
                right_foot_force_max = max(right_foot_force_max, right_force)
        else:
            raise ValueError(f"Unsupported torque_update_mode: {self.cfg.torque_update_mode}")

        self.last_left_foot_force_max = left_foot_force_max
        self.last_right_foot_force_max = right_foot_force_max
        self.rl_step_counter += 1
        self.hri_steps_since_reset += 1

        if self.isaac_replay is not None:
            self.current_isaac_sample = self.isaac_replay.step()
            self._sync_isaac_sample_to_env()

        reward_obs = _BaseHRIEnv._get_stacked_obs(self, append_current=True)
        obs = self._get_stacked_obs(append_current=True)
        reward, terminated, truncated, reward_info = self._compute_reward(
            reward_obs,
            filtered_action,
            torque_rate_sum / max(self.cfg.decimation, 1),
        )

        info = self._get_info()
        info["raw_action"] = action.copy()
        info["filtered_action"] = filtered_action.copy()
        info["last_torques"] = self.last_applied_torques.copy()
        info["teacher_action"] = self.current_teacher_action.copy()
        info["isaac_srl_obs"] = self.current_isaac_srl_obs.copy()
        info["hri_wrench_ramp_alpha"] = float(self.hri_wrench_ramp_alpha)
        info["hri_command_ramp_alpha"] = float(self.hri_command_ramp_alpha)
        if self.isaac_replay is not None:
            info["isaac_sequence_info"] = dict(self.isaac_replay.sequence_info)
        info.update(reward_info)
        return obs, reward, terminated, truncated, info

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
            self.current_isaac_srl_obs[:] = srl_obs
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

    def _hri_ramp_alpha(self, ramp_time: float) -> float:
        ramp_time = float(ramp_time)
        if ramp_time <= 0.0:
            return 1.0
        ramp_steps = max(1.0, ramp_time / max(float(self.control_dt), 1e-6))
        return float(np.clip(float(self.hri_steps_since_reset) / ramp_steps, 0.0, 1.0))

    def _set_isaac_task_command(self, target_vel_x: float, target_ang_vel_z: float, target_height: float):
        alpha = self._hri_ramp_alpha(self.cfg.hri_command_ramp_time)
        self.hri_command_ramp_alpha = alpha
        self.cfg.target_vel_x = (1.0 - alpha) * self.hri_default_target_vel_x + alpha * float(target_vel_x)
        self.cfg.target_ang_vel_z = (
            (1.0 - alpha) * self.hri_default_target_ang_vel_z
            + alpha * float(target_ang_vel_z)
        )
        self.cfg.target_height = (1.0 - alpha) * self.hri_default_target_height + alpha * float(target_height)

    def _set_isaac_load_cell_wrench(self, wrench: np.ndarray):
        alpha = self._hri_ramp_alpha(self.cfg.hri_wrench_ramp_time)
        self.hri_wrench_ramp_alpha = alpha
        force = alpha * float(self.cfg.isaac_load_cell_force_scale) * wrench[:3]
        clips = np.array(
            [
                self.cfg.isaac_load_cell_force_clip_x,
                self.cfg.isaac_load_cell_force_clip_y,
                self.cfg.isaac_load_cell_force_clip_z,
            ],
            dtype=np.float32,
        )
        for idx, limit in enumerate(clips):
            if limit > 0.0:
                force[idx] = np.clip(force[idx], -limit, limit)

        self.hri_wrench_local[:3] = force
        self.hri_wrench_local[3:] = alpha * float(self.cfg.isaac_load_cell_torque_scale) * wrench[3:]
        self._sync_local_wrench_to_global()

    def _sync_local_wrench_to_global(self):
        root_rot_mat = self.data.xmat[self.base_id].reshape(3, 3)
        self.hri_wrench_global[:3] = root_rot_mat @ self.hri_wrench_local[:3]
        self.hri_wrench_global[3:] = root_rot_mat @ self.hri_wrench_local[3:]

    def _get_policy_single_frame_obs(self):
        base_frame = _BaseHRIEnv._get_single_frame_obs(self)
        base_policy = np.concatenate(
            [
                base_frame[4:7],    # local root angular velocity
                base_frame[7:10],   # euler error
                base_frame[10:16],  # SRL joint position offset
                base_frame[16:22],  # SRL joint velocity, scaled
                base_frame[22:28],  # previous action
                base_frame[28:30],  # sin/cos phase
            ]
        ).astype(np.float32)

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

    def _get_stacked_obs(self, append_current=True):
        if self._base_reset_in_progress:
            return _BaseHRIEnv._get_stacked_obs(self, append_current=append_current)

        if append_current:
            self.policy_obs_history.append(self._get_policy_single_frame_obs())
        history = list(self.policy_obs_history)[::-1]
        frame_history = np.concatenate(history)
        task_cmd = np.array(
            [self.cfg.target_vel_x, self.cfg.target_ang_vel_z, self.cfg.target_height],
            dtype=np.float32,
        )
        obs = np.concatenate([frame_history, task_cmd]).astype(np.float32)
        return np.clip(obs, -self.cfg.clip_obs, self.cfg.clip_obs)

    def get_teacher_action(self):
        return self.current_teacher_action.copy()


EnvConfig = WalkEnvConfig
SRLMujocoWalkEnv = SRLMujocoHRIEnv
