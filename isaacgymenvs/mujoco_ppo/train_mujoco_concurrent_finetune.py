from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, fields
from typing import Tuple

import numpy as np
import torch
import yaml

import mujoco_ppo.train_mujoco_v2_finetune as v2_train
import mujoco_ppo.train_mujoco_v3_finetune as v3_train
from mujoco_ppo.models import numpy_to_torch_obs, safe_torch_load
from mujoco_ppo.models_concurrent import ConcurrentModelConfig, load_concurrent_checkpoint
from mujoco_ppo.srl_mujoco_concurrent_env import (
    ConcurrentWalkEnvConfig,
    SRLMujocoConcurrentEnv,
)
from mujoco_ppo.srl_mujoco_v2_env import quat_to_euler_xyz


PRIVILEGED_MIRROR_SIGNS = np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float32)


@dataclass
class ConcurrentPPOConfig(v2_train.V2PPOConfig):
    target_vel_x: float = 1.0
    target_ang_vel_z: float = 0.0
    initial_pose_randomization_enable: bool = False
    initial_pose_root_height_range: Tuple[float, float] = (0.85, 1.14)
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
    estimator_learning_rate: float = 1e-4
    estimator_update_epochs: int = 4
    estimator_minibatch_size: int = 1024
    actor_lr_schedule: str = "constant"
    actor_lr_kl_low: float = 0.002
    actor_lr_kl_high: float = 0.006
    actor_min_learning_rate: float = 1e-6
    actor_max_learning_rate: float = 2e-5
    actor_lr_multiplier: float = 1.5
    estimator_history_len: int = 10
    estimator_hidden_sizes: Tuple[int, ...] = (256, 128, 64)
    estimator_only_updates: int = 50
    actor_input_mode: str = "bootstrap"
    bootstrap_levels: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)
    bootstrap_initial_level: int = 0
    bootstrap_auto_advance: bool = True
    bootstrap_warmup_updates: int = 50
    bootstrap_min_updates_per_level: int = 25
    bootstrap_metric_window: int = 100
    bootstrap_min_completed_episodes: int = 10
    bootstrap_ground_truth_min_length: float = 500.0
    bootstrap_promote_length_ratio: float = 0.9
    bootstrap_promote_reward_ratio: float = 0.85
    bootstrap_demote_length_ratio: float = 0.7
    bootstrap_max_normalized_mse: float = 0.2
    bootstrap_fixed_env_quota: bool = True
    bootstrap_min_ground_truth_envs: int = 1
    bootstrap_pre_full_dr_max_probability: float = 0.5
    wobble_penalty_warmup_updates: int = 50
    wobble_penalty_ramp_updates: int = 150
    lateral_penalty_warmup_updates: int = 0
    lateral_penalty_ramp_updates: int = 150
    lateral_initial_min_distance: float = 0.25
    lateral_initial_max_distance: float = 0.85
    lateral_initial_target_weight: float = 0.0
    lateral_initial_symmetry_weight: float = 0.0
    lateral_initial_velocity_penalty_scale: float = 0.0
    lateral_initial_channel_penalty_scale: float = 0.0
    lateral_initial_hip_x_velocity_penalty_scale: float = 0.0

    startup_support_curriculum_enable: bool = False
    startup_support_probability_start: float = 0.2
    startup_support_probability_end: float = 0.5
    startup_support_curriculum_warmup_updates: int = 0
    startup_support_curriculum_ramp_updates: int = 500
    startup_support_hold_until_unloaded: bool = False
    startup_support_post_unload_hold_time: float = 0.0
    horizontal_force_curriculum_warmup_updates: int = 0
    horizontal_force_curriculum_ramp_updates: int = 200
    foot_impact_penalty_warmup_updates: int = 0
    foot_impact_penalty_ramp_updates: int = 200

    dr_curriculum_enable: bool = True
    dr_curriculum_levels: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    dr_curriculum_initial_level: int = 0
    dr_curriculum_start_update: int = 0
    dr_curriculum_min_updates_per_level: int = 150
    dr_curriculum_required_passes: int = 2
    dr_curriculum_gt_min_success: float = 0.6
    dr_curriculum_est_min_success: float = 0.5
    dr_curriculum_gt_min_length: float = 4000.0
    dr_curriculum_est_min_length: float = 3500.0

    evaluation_suite_enable: bool = True
    evaluation_narrow_distance: float = 0.30
    best_gait_min_success: float = 0.8
    best_gait_min_length: float = 4000.0
    evaluation_history_filename: str = "evaluation_history.jsonl"
    training_history_filename: str = "training_history.jsonl"

    # Isaac Gym task reward defaults, with Stage 4 CLI overrides represented
    # explicitly so the MuJoCo run is reproducible from this config alone.
    alive_reward_scale: float = 0.0
    progress_reward_scale: float = 0.0
    torques_cost_scale: float = 5e-4
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
    foot_lateral_velocity_penalty_scale: float = 0.0
    foot_lateral_velocity_deadband: float = 0.25
    foot_lateral_channel_penalty_scale: float = 0.0
    foot_lateral_channel_center: float = 0.17
    foot_lateral_channel_half_width: float = 0.07
    foot_lateral_channel_turn_relaxation: float = 0.08
    foot_lateral_channel_recovery_relaxation: float = 0.08
    foot_lateral_channel_max_relaxation: float = 0.12
    action_filter_jointwise_enable: bool = False
    action_filter_hip_x_cutoff_hz_range: Tuple[float, float] = (8.0, 10.0)
    action_filter_hip_y_cutoff_hz_range: Tuple[float, float] = (5.0, 7.0)
    action_filter_knee_cutoff_hz_range: Tuple[float, float] = (6.0, 9.0)
    hip_x_velocity_penalty_scale: float = 0.0
    hip_x_velocity_deadband: float = 0.2
    hip_x_velocity_recovery_roll_threshold: float = 0.08
    hip_x_velocity_recovery_rate_threshold: float = 0.4
    hip_x_velocity_recovery_scale: float = 0.25
    srl_motor_cost_scale: float = 0.0


class ConcurrentRolloutBuffer(v3_train.AsymmetricRolloutBuffer):
    def __init__(self, steps, num_envs, actor_obs_dim, critic_obs_dim, act_dim,
                 history_len, device):
        super().__init__(steps, num_envs, actor_obs_dim, critic_obs_dim, act_dim, device)
        self.estimator_history = torch.zeros(
            (steps, num_envs, history_len, 26), device=device
        )
        self.estimator_target = torch.zeros((steps, num_envs, 4), device=device)

    def add_concurrent(self, estimator_history, estimator_target, *args):
        index = self.ptr
        self.estimator_history[index] = estimator_history
        self.estimator_target[index] = estimator_target
        super().add(*args)


def make_env_config(cfg, *, evaluation=False):
    base = v3_train.make_env_config(cfg, evaluation=evaluation)
    values = asdict(base)
    values["target_vel_x"] = float(cfg.target_vel_x)
    values["target_ang_vel_z"] = float(cfg.target_ang_vel_z)
    for name in (
        "initial_pose_randomization_enable",
        "initial_pose_root_height_range",
        "initial_pose_ground_clearance",
        "initial_pose_low_hip_y_offset",
        "initial_pose_low_knee_offset",
        "initial_pose_high_hip_y_offset",
        "initial_pose_high_knee_offset",
        "initial_pose_hip_y_asymmetry_max",
        "initial_pose_knee_asymmetry_max",
        "initial_pose_foot_height_tolerance",
        "initial_pose_joint_limit_margin",
        "initial_pose_max_attempts",
    ):
        values[name] = getattr(cfg, name)
    values["startup_support_hold_until_unloaded"] = bool(
        cfg.startup_support_hold_until_unloaded
    )
    values["startup_support_post_unload_hold_time"] = float(
        cfg.startup_support_post_unload_hold_time
    )
    for name in (
        "alive_reward_scale", "progress_reward_scale", "torques_cost_scale",
        "dof_acc_cost_scale", "dof_vel_cost_scale", "dof_pos_cost_scale",
        "no_fly_penalty_scale", "vel_tracking_reward_scale",
        "tracking_ang_vel_reward_scale", "gait_similarity_penalty_scale",
        "pelvis_height_reward_scale", "orientation_reward_scale",
        "clearance_penalty_scale", "lateral_distance_penalty_scale",
        "foot_lateral_velocity_penalty_scale",
        "foot_lateral_velocity_deadband",
        "foot_lateral_channel_penalty_scale",
        "foot_lateral_channel_center",
        "foot_lateral_channel_half_width",
        "foot_lateral_channel_turn_relaxation",
        "foot_lateral_channel_recovery_relaxation",
        "foot_lateral_channel_max_relaxation",
        "action_filter_jointwise_enable",
        "action_filter_hip_x_cutoff_hz_range",
        "action_filter_hip_y_cutoff_hz_range",
        "action_filter_knee_cutoff_hz_range",
        "hip_x_velocity_penalty_scale",
        "hip_x_velocity_deadband",
        "hip_x_velocity_recovery_roll_threshold",
        "hip_x_velocity_recovery_rate_threshold",
        "hip_x_velocity_recovery_scale",
        "srl_motor_cost_scale",
    ):
        values[name] = getattr(cfg, name)
    return ConcurrentWalkEnvConfig(**values)


def make_env(cfg, *, evaluation=False):
    return SRLMujocoConcurrentEnv(
        make_env_config(cfg, evaluation=evaluation),
        estimator_history_len=cfg.estimator_history_len,
    )


def make_policy(cfg, env):
    inferred = v2_train.infer_model_config(
        cfg.checkpoint_path, cfg.checkpoint_key, obs_dim=137, act_dim=env.act_dim
    )
    model_cfg = ConcurrentModelConfig(
        hidden_sizes=inferred.hidden_sizes,
        estimator_history_len=cfg.estimator_history_len,
        estimator_hidden_sizes=tuple(cfg.estimator_hidden_sizes),
    )
    return load_concurrent_checkpoint(
        cfg.checkpoint_path, model_cfg, device=cfg.device,
        checkpoint_key=cfg.checkpoint_key,
    )


def _arrays_from_infos(infos):
    deployable = np.stack([info["deployable_obs"] for info in infos])
    mirrored = np.stack([info["actor_obs_mirrored"] for info in infos])
    history = np.stack([info["estimator_history"] for info in infos])
    target = np.stack([info["estimator_target"] for info in infos])
    critic = np.stack([info["critic_obs"] for info in infos])
    return deployable, mirrored, history, target, critic


@torch.no_grad()
def _compose(policy, deployable_np, mirrored_np, history_np, target_np, modes_np, device):
    deployable = numpy_to_torch_obs(deployable_np, device=device)
    history = numpy_to_torch_obs(history_np, device=device)
    target = numpy_to_torch_obs(target_np, device=device)
    modes = torch.as_tensor(modes_np, dtype=torch.bool, device=device)
    actor_obs, estimate = policy.compose_actor_obs(
        deployable, history, target, modes
    )
    selected = actor_obs[:, -4:]
    signs = torch.as_tensor(PRIVILEGED_MIRROR_SIGNS, device=device)
    mirrored = torch.cat(
        (numpy_to_torch_obs(mirrored_np, device=device), selected * signs), dim=-1
    )
    return actor_obs, mirrored, history, target, estimate


def _critic_value(policy, critic_np, device, normalize_value):
    return v3_train.critic_value(policy, critic_np, device, normalize_value)


def _save(policy, optimizers, cfg, update, save_dir, bootstrap_index, name=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(
        save_dir, name or f"mujoco_concurrent_update_{update:05d}.pt"
    )
    torch.save({
        "update": int(update),
        "model_state_dict": policy.state_dict(),
        "concurrent_estimator": policy.estimator.state_dict(),
        "concurrent_estimator_config": policy.estimator.model_config(),
        "optimizer_state_dict": {key: value.state_dict() for key, value in optimizers.items()},
        "concurrent_bootstrap_state": {"level_index": int(bootstrap_index)},
        "config": asdict(cfg),
        "actor_obs_dim": 137,
        "critic_obs_dim": 153,
    }, path)
    return path


_REWARD_COMPONENTS = (
    ("alive", "reward_alive", "alive_reward_scale", 1.0),
    ("progress", "reward_progress", "progress_reward_scale", 1.0),
    ("velocity", "reward_vel_tracking", "vel_tracking_reward_scale", 1.0),
    ("angular_velocity", "reward_ang_vel_tracking", "tracking_ang_vel_reward_scale", 1.0),
    ("orientation", "reward_orientation", "orientation_reward_scale", 1.0),
    ("pelvis_height", "reward_pelvis_height", "pelvis_height_reward_scale", 1.0),
    ("dof_acc", "reward_dof_acc", "dof_acc_cost_scale", 1.0),
    ("torques", "penalty_torques", "torques_cost_scale", -1.0),
    ("dof_velocity", "penalty_dof_vel", "dof_vel_cost_scale", -1.0),
    ("dof_position", "penalty_dof_pos", "dof_pos_cost_scale", -1.0),
    ("action_rate", "penalty_actions_rate", "actions_rate_scale", -1.0),
    ("action_smoothness", "penalty_actions_smoothness", "actions_smoothness_scale", -1.0),
    ("no_fly", "penalty_no_fly", "no_fly_penalty_scale", -1.0),
    ("gait_similarity", "penalty_gait_similarity", "gait_similarity_penalty_scale", -1.0),
    ("clearance", "penalty_clearance", "clearance_penalty_scale", -1.0),
    ("foot_distance", "penalty_lateral", "lateral_distance_penalty_scale", -1.0),
    ("motor", "penalty_motor", "srl_motor_cost_scale", -1.0),
    ("base_wobble", "penalty_base_wobble", "base_wobble_penalty_scale", -1.0),
    ("pitch_wobble", "penalty_pitch_wobble", "pitch_wobble_penalty_scale", -1.0),
    ("roll_wobble", "penalty_roll_wobble", "roll_wobble_penalty_scale", -1.0),
    ("base_ang_acc", "penalty_base_ang_acc", "base_ang_acc_penalty_scale", -1.0),
    ("yaw_drift", "penalty_yaw_drift", "yaw_drift_penalty_scale", -1.0),
    ("foot_impact", "penalty_foot_impact", "foot_impact_penalty_scale", -1.0),
    ("foot_lateral_velocity", "penalty_foot_lateral_velocity", "foot_lateral_velocity_penalty_scale", -1.0),
    ("foot_lateral_channel", "penalty_foot_lateral_channel", "foot_lateral_channel_penalty_scale", -1.0),
    ("hip_x_velocity", "penalty_hip_x_velocity", "hip_x_velocity_penalty_scale", -1.0),
)


def _weighted_reward_components(info, cfg):
    result = {
        name: sign * float(getattr(cfg, scale_name)) * float(info.get(info_name, 0.0))
        for name, info_name, scale_name, sign in _REWARD_COMPONENTS
    }
    result["termination"] = float(info.get("penalty_termination", 0.0))
    return result


def _append_jsonl(path, value):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


@torch.no_grad()
def evaluate_policy_condition(policy, cfg, dr_progress, use_estimate, seeds=None,
                              support_mode="none"):
    seeds = list(seeds or range(
        cfg.eval_during_training_start_seed,
        cfg.eval_during_training_start_seed + cfg.eval_during_training_seeds,
    ))
    lengths, returns, errors, absolute_errors = [], [], [], []
    separations, lateral_speeds, channel_violations, rolls = [], [], [], []
    hip_x_velocities, hip_x_recovery = [], []
    support_fractions, foot_force_ratios = [], []
    foot_force_exceeded, horizontal_force_active = [], []
    reward_components = defaultdict(list)
    for seed in seeds:
        env = make_env(cfg, evaluation=True)
        if support_mode == "none":
            env.cfg.startup_support_enable = False
        elif support_mode == "residual":
            env.cfg.startup_support_enable = True
            env.cfg.startup_support_mode_probabilities = (0.0, 0.0, 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported evaluation support mode: {support_mode}")
        env.set_dr_progress(float(dr_progress) if cfg.domain_randomization_enable else 0.0)
        _, info = env.reset(seed=int(seed))
        total = 0.0
        for step in range(int(cfg.eval_episode_steps)):
            deployable, mirrored, history, target, _ = _arrays_from_infos([info])
            actor_obs, _, _, _, estimate = _compose(
                policy, deployable, mirrored, history, target,
                np.asarray([use_estimate], dtype=bool), cfg.device,
            )
            action = policy.act_deterministic(actor_obs, cfg.action_clip)[0].cpu().numpy()
            _, reward, terminated, truncated, info = env.step(action)
            total += reward
            error = estimate[0].cpu().numpy() - target[0]
            errors.append(np.square(error))
            absolute_errors.append(np.abs(error))
            separations.append(float(info.get("foot_lateral_distance", 0.0)))
            lateral_speeds.append(0.5 * (
                float(info.get("left_foot_lateral_speed", 0.0))
                + float(info.get("right_foot_lateral_speed", 0.0))
            ))
            channel_violations.append(
                float(info.get("foot_lateral_channel_violation", 0.0))
            )
            rolls.append(float(info.get("roll", 0.0)))
            hip_x_velocities.append(0.5 * (
                abs(float(info.get("left_hip_x_velocity", 0.0)))
                + abs(float(info.get("right_hip_x_velocity", 0.0)))
            ))
            hip_x_recovery.append(
                float(info.get("hip_x_velocity_recovery_active", 0.0))
            )
            support_fractions.append(float(info.get("startup_support_fraction", 0.0)))
            max_force_ratio = max(
                float(info.get("left_foot_force_bw", 0.0)),
                float(info.get("right_foot_force_bw", 0.0)),
            )
            foot_force_ratios.append(max_force_ratio)
            foot_force_exceeded.append(max_force_ratio > cfg.foot_force_threshold_bw)
            horizontal_force_active.append(
                abs(float(info.get("horizontal_force_pulse_x", 0.0))) > 0.0
                or abs(float(info.get("horizontal_force_pulse_y", 0.0))) > 0.0
            )
            for name, value in _weighted_reward_components(info, env.cfg).items():
                reward_components[name].append(value)
            if terminated or truncated:
                break
        lengths.append(step + 1)
        returns.append(total)

    mae = np.mean(np.asarray(absolute_errors), axis=0) if absolute_errors else np.zeros(4)
    result = {
        "dr_progress": float(dr_progress),
        "input_mode": "estimated" if use_estimate else "ground_truth",
        "support_mode": support_mode,
        "episodes": len(seeds),
        "length": float(np.mean(lengths)),
        "return": float(np.mean(returns)),
        "success": float(np.mean(np.asarray(lengths) >= cfg.eval_episode_steps)),
        "estimator_mse": float(np.mean(errors)) if errors else float("nan"),
        "estimator_mae_height": float(mae[0]),
        "estimator_mae_vx": float(mae[1]),
        "estimator_mae_vy": float(mae[2]),
        "estimator_mae_vz": float(mae[3]),
        "foot_separation_mean": float(np.mean(separations)),
        "foot_separation_min": float(np.min(separations)),
        "narrow_step_fraction": float(
            np.mean(np.asarray(separations) < cfg.evaluation_narrow_distance)
        ),
        "foot_lateral_speed": float(np.mean(lateral_speeds)),
        "foot_channel_violation_fraction": float(np.mean(channel_violations)),
        "base_roll_rms": float(np.sqrt(np.mean(np.square(rolls)))),
        "hip_x_velocity_abs_mean": float(np.mean(hip_x_velocities)),
        "hip_x_velocity_recovery_fraction": float(np.mean(hip_x_recovery)),
        "support_fraction_mean": float(np.mean(support_fractions)),
        "foot_force_bw_mean": float(np.mean(foot_force_ratios)),
        "foot_force_bw_max": float(np.max(foot_force_ratios)),
        "foot_force_exceed_fraction": float(np.mean(foot_force_exceeded)),
        "horizontal_force_active_fraction": float(np.mean(horizontal_force_active)),
        "weighted_reward_components": {
            name: float(np.mean(values)) for name, values in reward_components.items()
        },
    }
    return result


def run_evaluation_suite(policy, cfg, current_dr_progress, seeds=None):
    suite = {}
    cache = {}
    for condition, progress in (
        ("no_dr", 0.0), ("current_dr", float(current_dr_progress)), ("full_dr", 1.0)
    ):
        for mode, use_estimate in (("gt", False), ("est", True)):
            cache_key = (float(progress), bool(use_estimate), "none")
            if cache_key not in cache:
                cache[cache_key] = evaluate_policy_condition(
                    policy, cfg, progress, use_estimate, seeds=seeds,
                    support_mode="none",
                )
            suite[f"{condition}/{mode}"] = copy.deepcopy(cache[cache_key])
            result = suite[f"{condition}/{mode}"]
            print(
                f"[eval {condition}/{mode}] len={result['length']:.0f} "
                f"success={result['success']:.2f} return={result['return']:.0f} "
                f"sep={result['foot_separation_mean']:.3f} "
                f"narrow={result['narrow_step_fraction']:.2f} "
                f"lat_v={result['foot_lateral_speed']:.3f} "
                f"roll={result['base_roll_rms']:.3f} "
                f"impact={result['foot_force_exceed_fraction']:.3f}"
            )
    if cfg.startup_support_enable:
        for mode, use_estimate in (("gt", False), ("est", True)):
            key = f"full_dr_support/{mode}"
            suite[key] = evaluate_policy_condition(
                policy, cfg, 1.0, use_estimate, seeds=seeds,
                support_mode="residual",
            )
            result = suite[key]
            print(
                f"[eval {key}] len={result['length']:.0f} "
                f"success={result['success']:.2f} return={result['return']:.0f} "
                f"support={result['support_fraction_mean']:.3f} "
                f"sep={result['foot_separation_mean']:.3f} "
                f"roll={result['base_roll_rms']:.3f} "
                f"impact={result['foot_force_exceed_fraction']:.3f}"
            )
    return suite


def evaluate_estimator_policy(policy, cfg, seeds=None):
    return evaluate_policy_condition(policy, cfg, 1.0, True, seeds=seeds)


def _update_bootstrap(cfg, update, level_index, level_start, mse, gt_lengths,
                      est_lengths, gt_returns, est_returns,
                      max_probability=1.0):
    if not cfg.bootstrap_auto_advance or update - level_start < cfg.bootstrap_min_updates_per_level:
        return level_index, level_start
    if level_index == 0:
        ready = (update >= cfg.bootstrap_warmup_updates and
                 len(gt_lengths) >= cfg.bootstrap_min_completed_episodes and
                 np.mean(gt_lengths) >= cfg.bootstrap_ground_truth_min_length and
                 mse <= cfg.bootstrap_max_normalized_mse)
        return (level_index + 1, update) if ready else (level_index, level_start)
    minimum = cfg.bootstrap_min_completed_episodes
    if min(len(gt_lengths), len(est_lengths)) < minimum:
        return level_index, level_start
    length_ratio = np.mean(est_lengths) / max(np.mean(gt_lengths), 1e-6)
    reward_ratio = np.mean(est_returns) / max(np.mean(gt_returns), 1e-6)
    if length_ratio < cfg.bootstrap_demote_length_ratio:
        return max(level_index - 1, 0), update
    next_allowed = (
        level_index < len(cfg.bootstrap_levels) - 1
        and cfg.bootstrap_levels[level_index + 1] <= max_probability + 1e-9
    )
    if (next_allowed and
            length_ratio >= cfg.bootstrap_promote_length_ratio and
            reward_ratio >= cfg.bootstrap_promote_reward_ratio and
            mse <= cfg.bootstrap_max_normalized_mse):
        return level_index + 1, update
    return level_index, level_start


def _target_estimator_count(num_envs, fraction, minimum_ground_truth=1):
    max_estimated = max(int(num_envs) - int(minimum_ground_truth), 0)
    return int(np.clip(round(float(fraction) * int(num_envs)), 0, max_estimated))


def _fixed_estimator_actor_input(cfg):
    mode = str(cfg.actor_input_mode).strip().lower()
    if mode not in ("bootstrap", "estimated"):
        raise ValueError("actor_input_mode must be 'bootstrap' or 'estimated'")
    return mode == "estimated"


def _actor_estimator_fraction(cfg, levels, level_index, max_probability):
    if _fixed_estimator_actor_input(cfg):
        return 1.0
    return min(levels[level_index], max_probability)


def _actor_minimum_ground_truth_envs(cfg):
    return 0 if _fixed_estimator_actor_input(cfg) else int(
        cfg.bootstrap_min_ground_truth_envs
    )


def _initial_episode_modes(num_envs, fraction, rng, minimum_ground_truth=1):
    count = _target_estimator_count(num_envs, fraction, minimum_ground_truth)
    modes = np.zeros(int(num_envs), dtype=bool)
    if count:
        modes[rng.permutation(int(num_envs))[:count]] = True
    return modes


def _rebalance_episode_modes(modes, reset_indices, fraction, rng,
                             minimum_ground_truth=1):
    """Reassign only environments that reached an episode boundary."""
    reset_indices = np.asarray(list(reset_indices), dtype=np.int64)
    if reset_indices.size == 0:
        return modes
    target = _target_estimator_count(
        len(modes), fraction, minimum_ground_truth
    )
    reset_set = set(reset_indices.tolist())
    fixed_estimated = sum(
        bool(value) for index, value in enumerate(modes) if index not in reset_set
    )
    estimated_resets = int(np.clip(target - fixed_estimated, 0, reset_indices.size))
    shuffled = rng.permutation(reset_indices)
    modes[reset_indices] = False
    modes[shuffled[:estimated_resets]] = True
    return modes


def _curriculum_progress(update, warmup_updates, ramp_updates):
    if update <= warmup_updates:
        return 0.0
    return float(np.clip(
        (update - warmup_updates) / max(int(ramp_updates), 1), 0.0, 1.0
    ))


def _startup_support_probability(cfg, update):
    if not cfg.startup_support_enable:
        return 0.0
    if not cfg.startup_support_curriculum_enable:
        return float(cfg.startup_support_mode_probabilities[3])
    progress = _curriculum_progress(
        update,
        cfg.startup_support_curriculum_warmup_updates,
        cfg.startup_support_curriculum_ramp_updates,
    )
    return float(
        cfg.startup_support_probability_start
        + progress * (
            cfg.startup_support_probability_end
            - cfg.startup_support_probability_start
        )
    )


def _set_residual_support_probability(env, probability):
    probability = float(np.clip(probability, 0.0, 1.0))
    env.cfg.startup_support_mode_probabilities = (
        1.0 - probability, 0.0, 0.0, probability
    )


def _dr_promotion_passes(cfg, suite):
    gt = suite["current_dr/gt"]
    est = suite["current_dr/est"]
    return (
        gt["success"] >= cfg.dr_curriculum_gt_min_success
        and est["success"] >= cfg.dr_curriculum_est_min_success
        and gt["length"] >= cfg.dr_curriculum_gt_min_length
        and est["length"] >= cfg.dr_curriculum_est_min_length
    )


def _gait_score(result):
    return -(
        4.0 * result["narrow_step_fraction"]
        + 2.0 * result["foot_channel_violation_fraction"]
        + result["foot_lateral_speed"]
        + 0.25 * result.get("hip_x_velocity_abs_mean", 0.0)
        + 4.0 * result["base_roll_rms"]
    )


def _flatten_evaluation_suite(suite):
    flattened = {}
    for condition, result in suite.items():
        prefix = f"evaluation/{condition}"
        for name, value in result.items():
            if name == "weighted_reward_components":
                for component, component_value in value.items():
                    flattened[f"{prefix}/reward_weighted_{component}"] = component_value
            elif isinstance(value, (int, float)):
                flattened[f"{prefix}/{name}"] = value
    return flattened


def _adaptive_actor_learning_rate(cfg, current_lr, kl):
    schedule = str(cfg.actor_lr_schedule).lower()
    if schedule == "constant":
        return float(current_lr)
    if schedule != "adaptive":
        raise ValueError(
            "actor_lr_schedule must be 'constant' or 'adaptive'"
        )

    kl_low = float(cfg.actor_lr_kl_low)
    kl_high = float(cfg.actor_lr_kl_high)
    multiplier = float(cfg.actor_lr_multiplier)
    min_lr = float(cfg.actor_min_learning_rate)
    max_lr = float(cfg.actor_max_learning_rate)
    if not 0.0 < kl_low < kl_high:
        raise ValueError(
            "actor LR KL bounds must satisfy 0 < low < high"
        )
    if multiplier <= 1.0:
        raise ValueError("actor_lr_multiplier must be > 1")
    if not 0.0 < min_lr <= max_lr:
        raise ValueError(
            "actor learning-rate bounds must satisfy 0 < min <= max"
        )

    learning_rate = float(current_lr)
    if kl > kl_high:
        learning_rate /= multiplier
    elif kl < kl_low:
        learning_rate *= multiplier
    return float(np.clip(learning_rate, min_lr, max_lr))


def train(cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    envs = [make_env(cfg) for _ in range(max(int(cfg.num_envs), 1))]
    policy, metadata = make_policy(cfg, envs[0])
    print("Loaded checkpoint:", metadata)
    print("Concurrent layout: estimator 10x26 -> 256 -> 128 -> 64 -> 4; actor 137 -> 512 -> 256 -> 128 -> 6; critic 153 -> 512 -> 256 -> 128 -> 1")
    if cfg.init_log_std is not None:
        policy.log_std.data.fill_(float(cfg.init_log_std))

    actor_params = list(policy.actor_mlp.parameters()) + list(policy.mu.parameters()) + [policy.log_std]
    critic_params = list(policy.critic_mlp.parameters()) + list(policy.value.parameters())
    optimizers = {
        "actor": torch.optim.Adam(actor_params, lr=cfg.actor_learning_rate),
        "critic": torch.optim.Adam(critic_params, lr=cfg.critic_learning_rate),
        "estimator": torch.optim.Adam(policy.estimator.parameters(), lr=cfg.estimator_learning_rate),
    }
    # Validate the schedule before the first expensive rollout.
    _adaptive_actor_learning_rate(
        cfg, optimizers["actor"].param_groups[0]["lr"],
        0.5 * (float(cfg.actor_lr_kl_low) + float(cfg.actor_lr_kl_high)),
    )
    print(
        f"Actor LR schedule: {cfg.actor_lr_schedule} "
        f"initial={cfg.actor_learning_rate:.2e} "
        f"range=[{cfg.actor_min_learning_rate:.2e}, "
        f"{cfg.actor_max_learning_rate:.2e}] "
        f"KL band=[{cfg.actor_lr_kl_low:.5f}, "
        f"{cfg.actor_lr_kl_high:.5f}] per update"
    )
    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    levels = tuple(float(value) for value in cfg.bootstrap_levels)
    if not levels or levels[0] != 0.0 or any(a >= b for a, b in zip(levels, levels[1:])):
        raise ValueError("bootstrap_levels must start at 0 and be strictly increasing")
    fixed_estimator_input = _fixed_estimator_actor_input(cfg)
    print(
        "Actor rollout input: "
        + ("100% estimated" if fixed_estimator_input else "bootstrap GT/estimated")
    )
    if not fixed_estimator_input and levels[-1] > 0.9:
        raise ValueError("MuJoCo fine-tuning keeps at least 10% ground-truth episodes")
    level_index = int(cfg.bootstrap_initial_level)
    level_start = 0
    dr_levels = tuple(float(value) for value in cfg.dr_curriculum_levels)
    if (not dr_levels or dr_levels[0] != 0.0 or dr_levels[-1] != 1.0
            or any(a >= b for a, b in zip(dr_levels, dr_levels[1:]))):
        raise ValueError("dr_curriculum_levels must start at 0, end at 1, and increase")
    dr_level_index = int(cfg.dr_curriculum_initial_level)
    if not 0 <= dr_level_index < len(dr_levels):
        raise ValueError("dr_curriculum_initial_level is out of range")
    progress = dr_levels[dr_level_index] if cfg.dr_curriculum_enable else 0.0
    dr_level_start = 0
    dr_consecutive_passes = 0

    for name in (
        "startup_support_probability_start",
        "startup_support_probability_end",
    ):
        value = float(getattr(cfg, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if cfg.startup_support_curriculum_ramp_updates < 0:
        raise ValueError("startup_support_curriculum_ramp_updates must be >= 0")

    infos = []
    support_probability = _startup_support_probability(cfg, 0)
    for index, env in enumerate(envs):
        env.set_dr_progress(progress)
        _set_residual_support_probability(env, support_probability)
        _, info = env.reset(seed=cfg.seed + index * 100003)
        infos.append(info)
    episode_returns = np.zeros(len(envs))
    episode_lengths = np.zeros(len(envs), dtype=np.int64)
    max_bootstrap_probability = (
        1.0 if progress >= 1.0
        else cfg.bootstrap_pre_full_dr_max_probability
    )
    target_estimator_fraction = _actor_estimator_fraction(
        cfg, levels, level_index, max_bootstrap_probability
    )
    minimum_ground_truth_envs = _actor_minimum_ground_truth_envs(cfg)
    if cfg.bootstrap_fixed_env_quota:
        modes = _initial_episode_modes(
            len(envs), target_estimator_fraction, envs[0].rng,
            minimum_ground_truth_envs,
        )
    else:
        modes = np.asarray(
            [env.rng.random() < target_estimator_fraction for env in envs],
            dtype=bool,
        )
    window = int(cfg.bootstrap_metric_window)
    gt_lengths, est_lengths = deque(maxlen=window), deque(maxlen=window)
    gt_returns, est_returns = deque(maxlen=window), deque(maxlen=window)
    run_name = f"mujoco_concurrent_{time.strftime('%m%d_%H%M%S')}"
    save_dir = os.path.join(cfg.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving checkpoints to: {save_dir}")
    wandb_run = v2_train.base_train.init_wandb(cfg, run_name)
    best_stability = -float("inf")
    best_gait = -float("inf")
    training_history_path = os.path.join(save_dir, cfg.training_history_filename)
    evaluation_history_path = os.path.join(save_dir, cfg.evaluation_history_filename)
    wobble_targets = {
        name: float(getattr(cfg, name))
        for name in (
            "base_wobble_penalty_scale", "pitch_wobble_penalty_scale",
            "roll_wobble_penalty_scale", "base_ang_acc_penalty_scale",
            "yaw_drift_penalty_scale",
        )
    }
    horizontal_force_probability_target = float(
        cfg.horizontal_force_perturbation_prob
    )
    foot_impact_penalty_target = float(cfg.foot_impact_penalty_scale)

    for update in range(1, cfg.total_updates + 1):
        if cfg.domain_randomization_enable and cfg.dr_curriculum_enable:
            progress = dr_levels[dr_level_index]
        elif cfg.domain_randomization_enable:
            progress = np.clip((update - cfg.dr_warmup_updates) / max(cfg.dr_ramp_updates, 1), 0, 1)
        else:
            progress = 0.0
        support_probability = _startup_support_probability(cfg, update)
        horizontal_force_progress = _curriculum_progress(
            update,
            cfg.horizontal_force_curriculum_warmup_updates,
            cfg.horizontal_force_curriculum_ramp_updates,
        )
        foot_impact_progress = _curriculum_progress(
            update,
            cfg.foot_impact_penalty_warmup_updates,
            cfg.foot_impact_penalty_ramp_updates,
        )
        if update <= cfg.wobble_penalty_warmup_updates:
            wobble_progress = 0.0
        else:
            wobble_progress = float(np.clip(
                (update - cfg.wobble_penalty_warmup_updates)
                / max(cfg.wobble_penalty_ramp_updates, 1), 0.0, 1.0
            ))
        if update <= cfg.lateral_penalty_warmup_updates:
            lateral_progress = 0.0
        else:
            lateral_progress = float(np.clip(
                (update - cfg.lateral_penalty_warmup_updates)
                / max(cfg.lateral_penalty_ramp_updates, 1), 0.0, 1.0
            ))
        for env in envs:
            env.set_dr_progress(float(progress))
            _set_residual_support_probability(env, support_probability)
            env.cfg.horizontal_force_perturbation_prob = (
                horizontal_force_probability_target * horizontal_force_progress
            )
            env.cfg.foot_impact_penalty_scale = (
                foot_impact_penalty_target * foot_impact_progress
            )
            for name, target_scale in wobble_targets.items():
                setattr(env.cfg, name, wobble_progress * target_scale)
            env.cfg.lateral_min_distance = float(
                cfg.lateral_initial_min_distance
                + lateral_progress
                * (cfg.lateral_min_distance - cfg.lateral_initial_min_distance)
            )
            env.cfg.lateral_max_distance = float(
                cfg.lateral_initial_max_distance
                + lateral_progress
                * (cfg.lateral_max_distance - cfg.lateral_initial_max_distance)
            )
            env.cfg.lateral_target_weight = float(
                cfg.lateral_initial_target_weight
                + lateral_progress
                * (cfg.lateral_target_weight - cfg.lateral_initial_target_weight)
            )
            env.cfg.lateral_symmetry_weight = float(
                cfg.lateral_initial_symmetry_weight
                + lateral_progress
                * (cfg.lateral_symmetry_weight - cfg.lateral_initial_symmetry_weight)
            )
            env.cfg.foot_lateral_velocity_penalty_scale = float(
                cfg.lateral_initial_velocity_penalty_scale
                + lateral_progress * (
                    cfg.foot_lateral_velocity_penalty_scale
                    - cfg.lateral_initial_velocity_penalty_scale
                )
            )
            env.cfg.foot_lateral_channel_penalty_scale = float(
                cfg.lateral_initial_channel_penalty_scale
                + lateral_progress * (
                    cfg.foot_lateral_channel_penalty_scale
                    - cfg.lateral_initial_channel_penalty_scale
                )
            )
            env.cfg.hip_x_velocity_penalty_scale = float(
                cfg.lateral_initial_hip_x_velocity_penalty_scale
                + lateral_progress * (
                    cfg.hip_x_velocity_penalty_scale
                    - cfg.lateral_initial_hip_x_velocity_penalty_scale
                )
            )
        deployable, mirrored_np, history_np, target_np, critic_np = _arrays_from_infos(infos)
        buffer = ConcurrentRolloutBuffer(
            cfg.rollout_steps, len(envs), 137, 153, 6,
            cfg.estimator_history_len, device,
        )
        rollout_rewards = []
        estimated_fraction = []
        foot_separations = []
        left_foot_local_y = []
        right_foot_local_y = []
        foot_lateral_speeds = []
        foot_channel_violations = []
        base_roll = []
        hip_x_action_saturated = []
        hip_x_velocity_magnitudes = []
        hip_x_velocity_recovery_active = []
        support_fractions = []
        support_mode_active = []
        foot_force_ratios = []
        foot_force_exceeded = []
        horizontal_force_active = []
        rollout_reward_components = defaultdict(list)
        for _step in range(cfg.rollout_steps):
            actor_obs, mirrored_obs, history, target, _ = _compose(
                policy, deployable, mirrored_np, history_np, target_np, modes, device
            )
            critic = numpy_to_torch_obs(critic_np, device=device)
            with torch.no_grad():
                actions, logprobs, values = policy.act(actor_obs, critic, cfg.action_clip)
                if cfg.normalize_value:
                    values = policy.value_norm.denormalize(values)
            results = [env.step(actions[i].cpu().numpy()) for i, env in enumerate(envs)]
            rewards = np.asarray([item[1] for item in results], dtype=np.float32)
            terminated = np.asarray([item[2] for item in results], dtype=np.float32)
            truncated = np.asarray([item[3] for item in results], dtype=np.float32)
            next_infos = [item[4] for item in results]
            foot_separations.extend(
                float(info.get("foot_lateral_distance", 0.0))
                for info in next_infos
            )
            left_foot_local_y.extend(
                float(info.get("left_foot_local_y", 0.0)) for info in next_infos
            )
            right_foot_local_y.extend(
                float(info.get("right_foot_local_y", 0.0)) for info in next_infos
            )
            foot_lateral_speeds.extend(
                0.5 * (
                    float(info.get("left_foot_lateral_speed", 0.0))
                    + float(info.get("right_foot_lateral_speed", 0.0))
                ) for info in next_infos
            )
            foot_channel_violations.extend(
                float(info.get("foot_lateral_channel_violation", 0.0))
                for info in next_infos
            )
            for env, info in zip(envs, next_infos):
                for name, value in _weighted_reward_components(info, env.cfg).items():
                    rollout_reward_components[name].append(value)
            base_roll.extend(
                float(quat_to_euler_xyz(env.data.qpos[3:7])[2]) for env in envs
            )
            action_values = actions.detach().cpu().numpy()
            hip_x_action_saturated.extend(
                (np.abs(action_values[:, (0, 3)]) > 0.95).reshape(-1).tolist()
            )
            for info in next_infos:
                hip_x_velocity_magnitudes.append(0.5 * (
                    abs(float(info.get("left_hip_x_velocity", 0.0)))
                    + abs(float(info.get("right_hip_x_velocity", 0.0)))
                ))
                hip_x_velocity_recovery_active.append(
                    float(info.get("hip_x_velocity_recovery_active", 0.0))
                )
            support_fractions.extend(
                float(info.get("startup_support_fraction", 0.0))
                for info in next_infos
            )
            support_mode_active.extend(
                info.get("startup_support_mode", "none") != "none"
                for info in next_infos
            )
            for info in next_infos:
                force_ratio = max(
                    float(info.get("left_foot_force_bw", 0.0)),
                    float(info.get("right_foot_force_bw", 0.0)),
                )
                foot_force_ratios.append(force_ratio)
                foot_force_exceeded.append(
                    force_ratio > cfg.foot_force_threshold_bw
                )
                horizontal_force_active.append(
                    abs(float(info.get("horizontal_force_pulse_x", 0.0))) > 0.0
                    or abs(float(info.get("horizontal_force_pulse_y", 0.0))) > 0.0
                )
            next_arrays = _arrays_from_infos(next_infos)
            next_values = _critic_value(policy, next_arrays[4], device, cfg.normalize_value)
            buffer.add_concurrent(
                history, target, actor_obs, mirrored_obs, critic, actions, logprobs,
                torch.as_tensor(rewards * cfg.reward_scale, device=device),
                torch.as_tensor(terminated, device=device),
                torch.as_tensor(truncated, device=device), values, next_values,
            )
            episode_returns += rewards
            episode_lengths += 1
            rollout_rewards.extend(rewards.tolist())
            estimated_fraction.extend(modes.astype(float).tolist())
            reset_indices = []
            for index, env in enumerate(envs):
                if not (terminated[index] or truncated[index]):
                    continue
                lengths = est_lengths if modes[index] else gt_lengths
                returns = est_returns if modes[index] else gt_returns
                lengths.append(int(episode_lengths[index]))
                returns.append(float(episode_returns[index]))
                _, next_infos[index] = env.reset()
                reset_indices.append(index)
                episode_returns[index] = 0.0
                episode_lengths[index] = 0
            if cfg.bootstrap_fixed_env_quota:
                _rebalance_episode_modes(
                    modes, reset_indices, target_estimator_fraction,
                    envs[0].rng, minimum_ground_truth_envs,
                )
            else:
                for index in reset_indices:
                    modes[index] = (
                        envs[index].rng.random() < target_estimator_fraction
                    )
            infos = next_infos
            deployable, mirrored_np, history_np, target_np, critic_np = _arrays_from_infos(infos)

        flat_history = buffer.estimator_history.reshape(-1, cfg.estimator_history_len, 26)
        flat_target = buffer.estimator_target.reshape(-1, 4)
        policy.estimator.update_normalization(flat_history, flat_target)
        indices = torch.arange(flat_history.shape[0], device=device)
        estimator_losses = []
        for _ in range(cfg.estimator_update_epochs):
            indices = indices[torch.randperm(indices.numel(), device=device)]
            for start in range(0, indices.numel(), cfg.estimator_minibatch_size):
                selected = indices[start:start + cfg.estimator_minibatch_size]
                loss = policy.estimator.normalized_mse(flat_history[selected], flat_target[selected])
                optimizers["estimator"].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.estimator.parameters(), cfg.max_grad_norm)
                optimizers["estimator"].step()
                estimator_losses.append(float(loss.detach()))
        estimator_mse = float(np.mean(estimator_losses))

        buffer.compute_returns(cfg.gamma, cfg.gae_lambda)
        if cfg.normalize_value:
            policy.value_norm.update(buffer.returns)
        buffer.advantages = (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)
        actor_enabled = update > max(cfg.critic_warmup_updates, cfg.estimator_only_updates)
        losses = {
            "pg": [], "vf": [], "ref": [], "sym": [], "kl": [],
            "clip_fraction": [], "entropy": [], "bounds": [], "std": [],
        }
        actor_lr_used = float(optimizers["actor"].param_groups[0]["lr"])
        completed_update_epochs = 0
        for _ in range(cfg.update_epochs):
            epoch_kls = []
            for batch in buffer.batches(cfg.minibatch_size):
                b_actor, b_mirror, b_critic, b_actions, old_logp, adv, returns, old_values = batch
                dist = policy.dist(policy.normalize_actor_obs(b_actor), cfg.action_clip)
                new_logp = dist.log_prob(b_actions).sum(-1)
                ratio = (new_logp - old_logp).exp()
                clip_fraction = (
                    torch.abs(ratio - 1.0) > cfg.clip_coef
                ).float().mean()
                pg = torch.max(-adv * ratio, -adv * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                mean = dist.raw_mean
                bounds = (torch.clamp(mean - cfg.action_clip, min=0).square() +
                          torch.clamp(-cfg.action_clip - mean, min=0).square()).sum(-1).mean()
                with torch.no_grad():
                    ref_mean = reference.actor(reference.normalize_actor_obs(b_actor))
                ref_loss = (mean - ref_mean).square().mean()
                mirror_mean = policy.actor(policy.normalize_actor_obs(b_mirror))
                mirror_mean = SRLMujocoConcurrentEnv.mirror_actions(mirror_mean)
                sym_loss = (mean - mirror_mean).square().mean()
                entropy = dist.entropy().sum(-1).mean()
                if actor_enabled:
                    actor_loss = (pg - cfg.ent_coef * entropy +
                                  cfg.reference_policy_coef * ref_loss +
                                  cfg.actor_sym_loss_coef * sym_loss +
                                  cfg.bounds_loss_coef * bounds)
                    optimizers["actor"].zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
                    optimizers["actor"].step()
                predicted = policy.critic(policy.normalize_critic_obs(b_critic))
                targets = policy.value_norm.normalize(returns) if cfg.normalize_value else returns
                old_v = policy.value_norm.normalize(old_values) if cfg.normalize_value else old_values
                if cfg.clip_value:
                    clipped = old_v + torch.clamp(predicted - old_v, -cfg.clip_coef, cfg.clip_coef)
                    vf = 0.5 * torch.max((predicted-targets).square(), (clipped-targets).square()).mean()
                else:
                    vf = 0.5 * (predicted-targets).square().mean()
                optimizers["critic"].zero_grad(set_to_none=True)
                (cfg.vf_coef * vf).backward()
                torch.nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
                optimizers["critic"].step()
                kl = ((ratio - 1) - (new_logp - old_logp)).mean().detach()
                losses["pg"].append(float(pg)); losses["vf"].append(float(vf))
                losses["ref"].append(float(ref_loss)); losses["sym"].append(float(sym_loss))
                losses["kl"].append(float(kl))
                losses["clip_fraction"].append(float(clip_fraction))
                losses["entropy"].append(float(entropy))
                losses["bounds"].append(float(bounds))
                losses["std"].append(float(dist.stddev.mean()))
                epoch_kls.append(float(kl))
            completed_update_epochs += 1
            epoch_kl = float(np.mean(epoch_kls))
            if cfg.target_kl is not None and epoch_kl > cfg.target_kl:
                break

        update_kl = float(np.mean(losses["kl"]))
        if actor_enabled:
            next_actor_lr = _adaptive_actor_learning_rate(
                cfg, actor_lr_used, update_kl
            )
            for param_group in optimizers["actor"].param_groups:
                param_group["lr"] = next_actor_lr

        old_level = level_index
        if not fixed_estimator_input:
            level_index, level_start = _update_bootstrap(
                cfg, update, level_index, level_start, estimator_mse,
                gt_lengths, est_lengths, gt_returns, est_returns,
                max_probability=max_bootstrap_probability,
            )
        if old_level != level_index:
            print(f"Bootstrap changed: {levels[old_level]:.2f} -> {levels[level_index]:.2f}")
        target_estimator_fraction = _actor_estimator_fraction(
            cfg, levels, level_index, max_bootstrap_probability
        )
        metrics = {
            "train/reward_mean": float(np.mean(rollout_rewards)),
            "train/dr_progress": float(progress),
            "train/wobble_penalty_progress": wobble_progress,
            "train/lateral_penalty_progress": lateral_progress,
            "train/horizontal_force_curriculum_progress": horizontal_force_progress,
            "train/foot_impact_penalty_progress": foot_impact_progress,
            "gait/foot_separation_mean": float(np.mean(foot_separations)),
            "gait/foot_separation_min": float(np.min(foot_separations)),
            "gait/narrow_step_fraction": float(
                np.mean(np.asarray(foot_separations) < cfg.lateral_min_distance)
            ),
            "gait/left_foot_local_y_range": float(np.ptp(left_foot_local_y)),
            "gait/right_foot_local_y_range": float(np.ptp(right_foot_local_y)),
            "gait/foot_lateral_speed_mean": float(np.mean(foot_lateral_speeds)),
            "gait/foot_channel_violation_fraction": float(
                np.mean(foot_channel_violations)
            ),
            "gait/hip_x_action_saturation": float(np.mean(hip_x_action_saturated)),
            "gait/hip_x_velocity_abs_mean": float(
                np.mean(hip_x_velocity_magnitudes)
            ),
            "gait/hip_x_velocity_recovery_fraction": float(
                np.mean(hip_x_velocity_recovery_active)
            ),
            "gait/base_roll_rms": float(
                np.sqrt(np.mean(np.square(base_roll)))
            ),
            "startup_support/target_episode_probability": support_probability,
            "startup_support/active_env_step_fraction": float(
                np.mean(support_mode_active)
            ),
            "startup_support/fraction_mean": float(np.mean(support_fractions)),
            "robustness/foot_force_bw_mean": float(np.mean(foot_force_ratios)),
            "robustness/foot_force_bw_max": float(np.max(foot_force_ratios)),
            "robustness/foot_force_exceed_fraction": float(
                np.mean(foot_force_exceeded)
            ),
            "robustness/horizontal_force_active_fraction": float(
                np.mean(horizontal_force_active)
            ),
            "concurrent/estimator_normalized_mse": estimator_mse,
            "concurrent/bootstrap_probability": target_estimator_fraction,
            "concurrent/bootstrap_requested_probability": (
                1.0 if fixed_estimator_input else levels[level_index]
            ),
            "concurrent/target_estimated_env_count": _target_estimator_count(
                len(envs), target_estimator_fraction,
                minimum_ground_truth_envs,
            ),
            "concurrent/estimated_env_fraction": float(np.mean(estimated_fraction)),
            "concurrent/ground_truth_episode_length": float(np.mean(gt_lengths)) if gt_lengths else 0.0,
            "concurrent/estimated_episode_length": float(np.mean(est_lengths)) if est_lengths else 0.0,
            "loss/policy": float(np.mean(losses["pg"])),
            "loss/value": float(np.mean(losses["vf"])),
            "loss/reference": float(np.mean(losses["ref"])),
            "loss/symmetry": float(np.mean(losses["sym"])),
            "loss/bounds": float(np.mean(losses["bounds"])),
            "ppo/kl": update_kl,
            "ppo/clip_fraction": float(np.mean(losses["clip_fraction"])),
            "ppo/entropy": float(np.mean(losses["entropy"])),
            "ppo/policy_std": float(np.mean(losses["std"])),
            "ppo/update_epochs_completed": int(completed_update_epochs),
            "ppo/actor_learning_rate": float(
                optimizers["actor"].param_groups[0]["lr"]
            ),
            "ppo/actor_learning_rate_used": actor_lr_used,
            "ppo/critic_learning_rate": float(
                optimizers["critic"].param_groups[0]["lr"]
            ),
            "concurrent/estimator_learning_rate": float(
                optimizers["estimator"].param_groups[0]["lr"]
            ),
        }
        metrics.update({
            f"reward_weighted/{name}": float(np.mean(values))
            for name, values in rollout_reward_components.items()
        })
        length_text = (
            f"len(est)={metrics['concurrent/estimated_episode_length']:.0f}"
            if fixed_estimator_input else
            "len(gt/est)="
            f"{metrics['concurrent/ground_truth_episode_length']:.0f}/"
            f"{metrics['concurrent/estimated_episode_length']:.0f}"
        )
        print(f"[update {update:04d}] reward={metrics['train/reward_mean']:.3f} mse={estimator_mse:.5f} p={target_estimator_fraction:.2f} est_frac={metrics['concurrent/estimated_env_fraction']:.2f} {length_text} actor_on={int(actor_enabled)} lr={metrics['ppo/actor_learning_rate_used']:.2e}->{metrics['ppo/actor_learning_rate']:.2e} kl={metrics['ppo/kl']:.5f} epochs={completed_update_epochs} dr={progress:.2f} support={support_probability:.2f}/{metrics['startup_support/active_env_step_fraction']:.2f} push={horizontal_force_progress:.2f} impact={metrics['robustness/foot_force_exceed_fraction']:.3f} wobble={wobble_progress:.2f} lateral={lateral_progress:.2f} sep={metrics['gait/foot_separation_mean']:.3f}/{metrics['gait/foot_separation_min']:.3f}")
        v2_train.base_train.log_wandb(wandb_run, metrics, step=update)
        _append_jsonl(training_history_path, {"update": update, **metrics})

        if cfg.eval_during_training and update % cfg.eval_every_updates == 0:
            if cfg.evaluation_suite_enable or cfg.dr_curriculum_enable:
                suite = run_evaluation_suite(policy, cfg, progress)
            else:
                result = evaluate_estimator_policy(policy, cfg)
                suite = {"full_dr/est": result}
            _append_jsonl(
                evaluation_history_path,
                {"update": update, "dr_progress": progress, "results": suite},
            )
            v2_train.base_train.log_wandb(
                wandb_run, _flatten_evaluation_suite(suite), step=update
            )

            full_est = suite["full_dr/est"]
            supported_est = suite.get("full_dr_support/est", full_est)
            stability_score = (
                min(full_est["success"], supported_est["success"]) * 1e6
                + min(full_est["length"], supported_est["length"]) * 100
                + 0.5 * (full_est["return"] + supported_est["return"])
            )
            if stability_score > best_stability:
                best_stability = stability_score
                print("Saved best full-DR estimated stability checkpoint:", _save(
                    policy, optimizers, cfg, update, save_dir, level_index,
                    "mujoco_concurrent_best_stability.pt"))
                _save(
                    policy, optimizers, cfg, update, save_dir, level_index,
                    "mujoco_concurrent_best_eval.pt",
                )

            gait_eligible = (
                full_est["success"] >= cfg.best_gait_min_success
                and full_est["length"] >= cfg.best_gait_min_length
                and supported_est["success"] >= cfg.best_gait_min_success
                and supported_est["length"] >= cfg.best_gait_min_length
            )
            gait_score = _gait_score(full_est)
            if gait_eligible and gait_score > best_gait:
                best_gait = gait_score
                print("Saved best full-DR estimated gait checkpoint:", _save(
                    policy, optimizers, cfg, update, save_dir, level_index,
                    "mujoco_concurrent_best_gait.pt"))

            if (cfg.domain_randomization_enable and cfg.dr_curriculum_enable
                    and dr_level_index < len(dr_levels) - 1):
                enough_time = (
                    update >= cfg.dr_curriculum_start_update
                    and update - dr_level_start
                    >= cfg.dr_curriculum_min_updates_per_level
                )
                passed = enough_time and _dr_promotion_passes(cfg, suite)
                dr_consecutive_passes = (
                    dr_consecutive_passes + 1 if passed else 0
                )
                print(
                    f"[DR gate] level={progress:.2f} pass={int(passed)} "
                    f"consecutive={dr_consecutive_passes}/"
                    f"{cfg.dr_curriculum_required_passes}"
                )
                if dr_consecutive_passes >= cfg.dr_curriculum_required_passes:
                    old_progress = progress
                    dr_level_index += 1
                    progress = dr_levels[dr_level_index]
                    dr_level_start = update
                    dr_consecutive_passes = 0
                    max_bootstrap_probability = (
                        1.0 if progress >= 1.0
                        else cfg.bootstrap_pre_full_dr_max_probability
                    )
                    target_estimator_fraction = _actor_estimator_fraction(
                        cfg, levels, level_index, max_bootstrap_probability
                    )
                    print(f"DR promoted: {old_progress:.2f} -> {progress:.2f}")
                    for index, env in enumerate(envs):
                        env.set_dr_progress(progress)
                        _, infos[index] = env.reset()
                    episode_returns.fill(0.0)
                    episode_lengths.fill(0)
                    gt_lengths.clear(); est_lengths.clear()
                    gt_returns.clear(); est_returns.clear()
                    if cfg.bootstrap_fixed_env_quota:
                        modes = _initial_episode_modes(
                            len(envs), target_estimator_fraction,
                            envs[0].rng, minimum_ground_truth_envs,
                        )
                    else:
                        modes = np.asarray([
                            env.rng.random() < target_estimator_fraction
                            for env in envs
                        ], dtype=bool)
        if update % cfg.save_every == 0:
            print("Saved:", _save(policy, optimizers, cfg, update, save_dir, level_index))

    print("Saved final:", _save(policy, optimizers, cfg, cfg.total_updates,
                                save_dir, level_index, "mujoco_concurrent_final.pt"))
    if wandb_run is not None:
        wandb_run.finish()


def _load_config(path):
    values = v2_train._load_yaml_defaults(path)
    known = {field.name for field in fields(ConcurrentPPOConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown YAML config keys: {unknown}")
    for key in ("default_dof_pos", "kp", "kd", "effort_limits",
                "estimator_hidden_sizes", "bootstrap_levels",
                "dr_curriculum_levels", "initial_pose_root_height_range"):
        if key in values and values[key] is not None:
            values[key] = tuple(values[key])
    return ConcurrentPPOConfig(**values)


def _apply_cli_overrides(cfg, entries):
    known = {field.name for field in fields(ConcurrentPPOConfig)}
    for entry in entries or ():
        if "=" not in entry:
            raise ValueError(f"Invalid --set value {entry!r}; expected NAME=VALUE")
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        if key not in known:
            raise ValueError(f"Unknown config key in --set: {key}")

        current = getattr(cfg, key)
        value = yaml.safe_load(raw_value)
        if isinstance(current, tuple):
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"--set {key} expects a YAML list")
            value = tuple(value)
        elif isinstance(current, bool):
            if not isinstance(value, bool):
                raise ValueError(f"--set {key} expects true or false")
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, str):
            value = str(value)
        setattr(cfg, key, value)


def parse_args():
    parser = argparse.ArgumentParser(description="Concurrent Actor/Estimator PPO fine-tuning in MuJoCo")
    parser.add_argument("--config", default="cfg/mujoco_concurrent_finetune.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--xml")
    parser.add_argument("--device")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--set", dest="config_overrides", action="append", default=[],
        metavar="NAME=VALUE", help="override one flattened YAML config value",
    )
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = _load_config(args.config)
    overrides = {
        "checkpoint_path": args.checkpoint, "xml_path": args.xml,
        "device": args.device, "total_updates": args.updates,
        "num_envs": args.num_envs, "seed": args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    _apply_cli_overrides(cfg, args.config_overrides)
    if args.eval_only:
        cfg.eval_only = True
    if cfg.eval_only:
        policy, metadata = make_policy(cfg, make_env(cfg, evaluation=True))
        print("Loaded checkpoint:", metadata)
        run_evaluation_suite(
            policy, cfg,
            cfg.dr_curriculum_levels[cfg.dr_curriculum_initial_level],
        )
    else:
        train(cfg)


if __name__ == "__main__":
    main()
