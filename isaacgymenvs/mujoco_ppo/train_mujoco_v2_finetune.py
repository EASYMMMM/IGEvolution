from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from typing import Optional, Tuple

import numpy as np
import torch
import yaml

import mujoco_ppo.train_mujoco_finetune as base_train
from mujoco_ppo.models import ModelConfig, load_isaac_checkpoint, safe_torch_load
from mujoco_ppo.srl_mujoco_v2_env import V2WalkEnvConfig, SRLMujocoV2Env


@dataclass
class V2PPOConfig:
    checkpoint_path: str = "checkpoints/SRL_Real_Bot_v2_s1.pth"
    checkpoint_key: str = "model"
    xml_path: str = "mjcf/srl_real_v1/srl_real_bot_v2_pos.xml"
    total_updates: int = 100
    rollout_steps: int = 2048
    num_envs: int = 1
    training_episode_steps: int = 5000
    reset_envs_each_rollout: bool = False
    learning_rate: float = 1e-5
    actor_learning_rate: float = 5e-6
    critic_learning_rate: float = 1e-4
    critic_warmup_updates: int = 5
    reference_policy_coef: float = 0.0
    actor_sym_loss_coef: float = 0.0
    reward_scale: float = 0.01
    normalize_value: bool = True
    clip_value: bool = True
    bounds_loss_coef: float = 5e-4
    cvar_fraction: float = 0.0
    cvar_weight: float = 1.0
    failure_weight: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    update_epochs: int = 10
    minibatch_size: int = 256
    target_kl: Optional[float] = 0.005
    seed: int = 1
    device: str = "cpu"
    save_dir: str = "mujoco_ppo/runs"
    save_every: int = 10
    action_clip: float = 1.0
    eval_only: bool = False
    eval_steps: int = 2000
    eval_episode_steps: int = 5000
    eval_num_seeds: int = 1
    eval_during_training: bool = False
    eval_every_updates: int = 20
    eval_during_training_seeds: int = 5
    eval_during_training_start_seed: int = 101
    eval_during_training_secondary_start_seed: Optional[int] = None
    eval_during_training_steps: int = 2000
    eval_impact_p99_weight: float = 0.02
    eval_report: Optional[str] = None
    save_failure_traces: Optional[str] = None
    failure_trace_steps: int = 200
    debug_rollout: bool = False
    init_log_std: Optional[float] = None
    resume_optimizer: bool = False

    control_backend: str = "position_target"
    action_filter_enable: bool = True
    action_filter_cutoff_hz: float = 8.0
    pd_tracking_error_limits: Optional[Tuple[float, ...]] = None
    pd_target_change_limit: Optional[float] = None
    root_height: float = 1.1
    default_dof_pos: Tuple[float, ...] = (0.0, -0.55, -0.3, 0.0, -0.55, -0.3)
    kp: Tuple[float, ...] = (250.0, 300.0, 300.0, 250.0, 300.0, 300.0)
    kd: Tuple[float, ...] = (18.0, 22.0, 14.0, 18.0, 22.0, 14.0)
    effort_limits: Tuple[float, ...] = (120.0, 120.0, 400.0, 120.0, 120.0, 400.0)
    termination_penalty: float = -25.0
    actions_rate_scale: float = 0.3
    actions_smoothness_scale: float = 0.6

    base_wobble_penalty_scale: float = 0.0
    pitch_wobble_penalty_scale: float = 0.0
    roll_wobble_penalty_scale: float = 0.0
    roll_rate_weight: float = 0.3
    base_ang_acc_penalty_scale: float = 0.0
    yaw_drift_penalty_scale: float = 0.0
    foot_impact_penalty_scale: float = 0.0
    foot_force_threshold_bw: float = 1.8
    foot_force_penalty_power: float = 2.0
    lateral_min_distance: float = 0.25
    lateral_max_distance: float = 0.85
    lateral_symmetry_weight: float = 0.0
    lateral_target_distance: float = 0.0
    lateral_target_weight: float = 0.0
    randomize_initial_phase: bool = True
    initial_phase: Optional[int] = None
    task_training_stage: int = 0
    startup_inplace_time: float = 0.8
    startup_ramp_time: float = 0.8

    # Optional v3 walking startup support. The v2 environment ignores these
    # fields; they live here so the shared YAML/CLI loader can be reused.
    startup_support_enable: bool = False
    startup_support_in_evaluation: bool = False
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

    # Optional v3 suspended-swing task settings. Walking environments ignore
    # these fields; they live here so the shared YAML/CLI loader can be reused.
    swing_reference_path: Optional[str] = None
    swing_inplace_reference_path: Optional[str] = None
    swing_forward_reference_path: Optional[str] = None
    swing_inplace_probability: float = 0.5
    swing_forward_command_vx: float = 1.0
    swing_continuous_command: bool = False
    swing_fixed_command_vx: Optional[float] = None
    swing_reset_critic: bool = True
    swing_reset_actor_output: bool = True
    swing_suspension_pos_kp: Tuple[float, ...] = (120.0, 120.0, 300.0)
    swing_suspension_pos_kd: Tuple[float, ...] = (45.0, 45.0, 80.0)
    swing_suspension_rot_kp: Tuple[float, ...] = (30.0, 30.0, 8.0)
    swing_suspension_rot_kd: Tuple[float, ...] = (8.0, 8.0, 3.0)
    swing_q_tracking_scale: float = 8.0
    swing_action_tracking_scale: float = 2.0
    swing_orientation_scale: float = 1.0
    swing_position_scale: float = 0.5
    swing_dof_vel_penalty_scale: float = 0.01
    swing_torque_penalty_scale: float = 2.0e-5
    swing_action_rate_penalty_scale: float = 0.05
    swing_action_smoothness_penalty_scale: float = 0.10
    swing_termination_position_error: float = 0.60
    swing_termination_tilt: float = 0.80
    swing_fixed_base_timeconst: float = 0.001
    swing_hip_y_amplitude: float = 0.06
    swing_knee_amplitude: float = 0.08
    swing_target_tracking_scale: float = 4.0
    swing_qd_tracking_scale: float = 2.0
    swing_hip_x_penalty_scale: float = 6.0
    swing_hip_x_velocity_penalty_scale: float = 1.0
    swing_foot_x_amplitude: float = 0.10
    swing_foot_z_amplitude: float = 0.08
    swing_foot_position_tracking_scale: float = 8.0
    swing_foot_velocity_tracking_scale: float = 1.5
    swing_posture_tracking_scale: float = 0.25
    swing_torque_rate_penalty_scale: float = 0.02
    swing_joint_limit_penalty_scale: float = 0.50
    swing_joint_limit_margin_fraction: float = 0.10
    swing_normalized_qd_scale: float = 8.0
    swing_termination_joint_tolerance: float = 0.03

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
    dr_stratified_evaluation_enable: bool = False
    dr_scenario_probabilities: Tuple[float, ...] = (0.45, 0.15, 0.15, 0.05, 0.10, 0.10)
    dr_low_friction_fixed_delay_probability: float = 0.0
    dr_low_friction_fixed_delay_range: Tuple[float, float] = (0.7, 1.0)
    dr_hard_longitudinal_gravity_sigma_range: Tuple[float, float] = (1.5, 2.5)
    dr_hard_lateral_gravity_sigma_range: Tuple[float, float] = (1.0, 2.0)
    dr_combined_longitudinal_gravity_sigma_range: Tuple[float, float] = (2.0, 3.0)
    obs_noise_std: float = 0.0
    obs_noise_dof_pos_std: float = 0.0
    obs_noise_dof_vel_std: float = 0.0
    action_noise_std: float = 0.0
    action_delay_max_steps: int = 0
    action_delay_max_physics_steps: Optional[int] = None
    velocity_perturbation_enable: bool = False
    velocity_perturbation_prob: float = 1.0 / 500.0
    velocity_perturbation_range: Tuple[float, float] = (0.2, 0.8)
    horizontal_force_perturbation_enable: bool = False
    horizontal_force_perturbation_prob: float = 1.0 / 1000.0
    horizontal_force_perturbation_range: Tuple[float, float] = (20.0, 80.0)
    horizontal_force_perturbation_duration_range: Tuple[float, float] = (0.05, 0.20)
    dr_warmup_updates: int = 10
    dr_ramp_updates: int = 80

    wandb_enabled: bool = False
    wandb_project: str = "srl-mujoco-v2-finetune"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"


def _select_checkpoint_state(checkpoint, checkpoint_key: str):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and checkpoint_key in checkpoint:
        return checkpoint[checkpoint_key]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")


def _infer_hidden_sizes_from_state(state) -> Tuple[int, ...]:
    layers = []
    pattern = re.compile(r"(?:^|\.)(actor_mlp)\.(\d+)\.weight$")
    for key, value in state.items():
        match = pattern.search(str(key))
        if match is None:
            continue
        tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.ndim != 2:
            continue
        layers.append((int(match.group(2)), int(tensor.shape[0])))
    layers.sort(key=lambda item: item[0])
    if not layers:
        raise RuntimeError("Could not infer actor hidden sizes from checkpoint actor_mlp weights.")
    return tuple(width for _idx, width in layers)


def infer_model_config(checkpoint_path: str, checkpoint_key: str, obs_dim: int, act_dim: int) -> ModelConfig:
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    state = _select_checkpoint_state(checkpoint, checkpoint_key)
    hidden_sizes = _infer_hidden_sizes_from_state(state)
    return ModelConfig(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes)


def make_env_config(cfg: V2PPOConfig, *, evaluation: bool = False):
    stratified_sampling = cfg.dr_stratified_sampling_enable and (
        not evaluation or cfg.dr_stratified_evaluation_enable
    )
    return V2WalkEnvConfig(
        xml_path=cfg.xml_path,
        control_backend=cfg.control_backend,
        action_filter_enable=cfg.action_filter_enable,
        action_filter_cutoff_hz=cfg.action_filter_cutoff_hz,
        pd_tracking_error_limits=cfg.pd_tracking_error_limits,
        pd_target_change_limit=cfg.pd_target_change_limit,
        root_height=cfg.root_height,
        default_dof_pos=cfg.default_dof_pos,
        kp=cfg.kp,
        kd=cfg.kd,
        effort_limits=cfg.effort_limits,
        max_episode_steps=(
            int(cfg.eval_episode_steps) if evaluation else int(cfg.training_episode_steps)
        ),
        termination_penalty=cfg.termination_penalty,
        actions_rate_scale=cfg.actions_rate_scale,
        actions_smoothness_scale=cfg.actions_smoothness_scale,
        base_wobble_penalty_scale=cfg.base_wobble_penalty_scale,
        pitch_wobble_penalty_scale=cfg.pitch_wobble_penalty_scale,
        roll_wobble_penalty_scale=cfg.roll_wobble_penalty_scale,
        roll_rate_weight=cfg.roll_rate_weight,
        base_ang_acc_penalty_scale=cfg.base_ang_acc_penalty_scale,
        yaw_drift_penalty_scale=cfg.yaw_drift_penalty_scale,
        foot_impact_penalty_scale=cfg.foot_impact_penalty_scale,
        foot_force_threshold_bw=cfg.foot_force_threshold_bw,
        foot_force_penalty_power=cfg.foot_force_penalty_power,
        lateral_min_distance=cfg.lateral_min_distance,
        lateral_max_distance=cfg.lateral_max_distance,
        lateral_symmetry_weight=cfg.lateral_symmetry_weight,
        lateral_target_distance=cfg.lateral_target_distance,
        lateral_target_weight=cfg.lateral_target_weight,
        randomize_initial_phase=cfg.randomize_initial_phase,
        initial_phase=cfg.initial_phase,
        domain_randomization_enable=cfg.domain_randomization_enable,
        dr_friction_range=cfg.dr_friction_range,
        dr_kp_range=cfg.dr_kp_range,
        dr_kp_joint_range=cfg.dr_kp_joint_range,
        dr_kd_range=cfg.dr_kd_range,
        dr_kd_joint_range=cfg.dr_kd_joint_range,
        dr_effort_range=cfg.dr_effort_range,
        dr_effort_joint_range=cfg.dr_effort_joint_range,
        dr_gear_range=cfg.dr_gear_range,
        dr_mass_range=cfg.dr_mass_range,
        dr_mass_link_range=cfg.dr_mass_link_range,
        dr_inertia_range=cfg.dr_inertia_range,
        dr_inertia_link_range=cfg.dr_inertia_link_range,
        dr_damping_range=cfg.dr_damping_range,
        dr_damping_joint_range=cfg.dr_damping_joint_range,
        dr_frictionloss_range=cfg.dr_frictionloss_range,
        dr_frictionloss_joint_range=cfg.dr_frictionloss_joint_range,
        dr_armature_range=cfg.dr_armature_range,
        dr_armature_joint_range=cfg.dr_armature_joint_range,
        dr_foot_friction_range=cfg.dr_foot_friction_range,
        dr_foot_solref_timeconst_range=cfg.dr_foot_solref_timeconst_range,
        dr_foot_solref_dampratio_range=cfg.dr_foot_solref_dampratio_range,
        dr_foot_solimp_width_range=cfg.dr_foot_solimp_width_range,
        dr_foot_radius_range=cfg.dr_foot_radius_range,
        dr_base_com_shift=cfg.dr_base_com_shift,
        dr_link_com_shift=cfg.dr_link_com_shift,
        dr_gravity_std=cfg.dr_gravity_std,
        dr_joint_limit_std=cfg.dr_joint_limit_std,
        dr_stratified_sampling_enable=stratified_sampling,
        dr_scenario_probabilities=cfg.dr_scenario_probabilities,
        dr_low_friction_fixed_delay_probability=(
            cfg.dr_low_friction_fixed_delay_probability
        ),
        dr_low_friction_fixed_delay_range=(
            cfg.dr_low_friction_fixed_delay_range
        ),
        dr_hard_longitudinal_gravity_sigma_range=(
            cfg.dr_hard_longitudinal_gravity_sigma_range
        ),
        dr_hard_lateral_gravity_sigma_range=(
            cfg.dr_hard_lateral_gravity_sigma_range
        ),
        dr_combined_longitudinal_gravity_sigma_range=(
            cfg.dr_combined_longitudinal_gravity_sigma_range
        ),
        obs_noise_std=cfg.obs_noise_std,
        obs_noise_dof_pos_std=cfg.obs_noise_dof_pos_std,
        obs_noise_dof_vel_std=cfg.obs_noise_dof_vel_std,
        action_noise_std=cfg.action_noise_std,
        action_delay_max_steps=cfg.action_delay_max_steps,
        action_delay_max_physics_steps=cfg.action_delay_max_physics_steps,
        velocity_perturbation_enable=cfg.velocity_perturbation_enable,
        velocity_perturbation_prob=cfg.velocity_perturbation_prob,
        velocity_perturbation_range=cfg.velocity_perturbation_range,
        horizontal_force_perturbation_enable=(
            cfg.horizontal_force_perturbation_enable
        ),
        horizontal_force_perturbation_prob=(
            cfg.horizontal_force_perturbation_prob
        ),
        horizontal_force_perturbation_range=(
            cfg.horizontal_force_perturbation_range
        ),
        horizontal_force_perturbation_duration_range=(
            cfg.horizontal_force_perturbation_duration_range
        ),
    )


def make_env_and_model(cfg: V2PPOConfig):
    env = SRLMujocoV2Env(make_env_config(cfg, evaluation=cfg.eval_only))

    model_cfg = infer_model_config(
        cfg.checkpoint_path,
        checkpoint_key=cfg.checkpoint_key,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
    )
    policy, metadata = load_isaac_checkpoint(
        cfg.checkpoint_path,
        model_cfg=model_cfg,
        device=cfg.device,
        strict_critic=False,
    )
    metadata = dict(metadata)
    metadata.update(
        {
            "v2_env": True,
            "model_hidden_sizes": model_cfg.hidden_sizes,
            "control_backend": cfg.control_backend,
            "action_filter_enable": cfg.action_filter_enable,
            "action_filter_cutoff_hz": cfg.action_filter_cutoff_hz,
            "root_height": cfg.root_height,
            "default_dof_pos": cfg.default_dof_pos,
            "effort_limits": cfg.effort_limits,
            "domain_randomization_enable": cfg.domain_randomization_enable,
            "velocity_perturbation_enable": cfg.velocity_perturbation_enable,
        }
    )
    return env, policy, metadata


def make_additional_env(cfg: V2PPOConfig):
    return SRLMujocoV2Env(make_env_config(cfg, evaluation=False))


def save_checkpoint(policy, optimizer, cfg: V2PPOConfig, update_idx: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"mujoco_v2_finetune_update_{update_idx:05d}.pt")
    torch.save(
        {
            "update": update_idx,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(cfg),
        },
        save_path,
    )
    return save_path


def _rms(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def _wobble_metrics(window):
    if not window:
        return {
            "score": 0.0,
            "pitch_rms": 0.0,
            "roll_rms": 0.0,
            "roll_mean": 0.0,
            "wx_rms": 0.0,
            "wy_rms": 0.0,
            "wz_rms": 0.0,
        }
    arr = np.asarray(window, dtype=np.float32)
    pitch_rms = _rms(arr[:, 0])
    roll_rms = _rms(arr[:, 1])
    roll_mean = float(np.mean(arr[:, 1]))
    wx_rms = _rms(arr[:, 2])
    wy_rms = _rms(arr[:, 3])
    wz_rms = _rms(arr[:, 4])
    score = pitch_rms + roll_rms + 0.5 * (wx_rms + wy_rms) + 0.2 * wz_rms
    return {
        "score": float(score),
        "pitch_rms": pitch_rms,
        "roll_rms": roll_rms,
        "roll_mean": roll_mean,
        "wx_rms": wx_rms,
        "wy_rms": wy_rms,
        "wz_rms": wz_rms,
    }


def _print_eval_dr(env, episode_idx):
    if not env.cfg.domain_randomization_enable:
        return
    params = env.get_domain_randomization_summary()
    gravity = np.asarray(params["gravity_delta"], dtype=np.float64)
    base_com = np.asarray(params["base_com_shift"], dtype=np.float64)
    print(
        f"[eval dr episode={episode_idx:03d}] "
        f"progress={params['dr_progress']:.3f} "
        f"scenario={params['scenario']} "
        f"friction={params['friction_scale']:.3f} "
        f"mass={params['mass_scale']:.3f}*"
        f"[{params['mass_link_scale_min']:.3f},{params['mass_link_scale_max']:.3f}] "
        f"inertia={params['inertia_scale']:.3f}*"
        f"[{params['inertia_link_scale_min']:.3f},{params['inertia_link_scale_max']:.3f}] "
        f"armature={params['armature_scale']:.3f}*"
        f"[{params['armature_joint_scale_min']:.3f},{params['armature_joint_scale_max']:.3f}] "
        f"damping={params['damping_scale']:.3f}*"
        f"[{params['damping_joint_scale_min']:.3f},{params['damping_joint_scale_max']:.3f}] "
        f"frictionloss={params['frictionloss_scale']:.3f}*"
        f"[{params['frictionloss_joint_scale_min']:.3f},"
        f"{params['frictionloss_joint_scale_max']:.3f}] "
        f"kp={params['kp_scale']:.3f}*"
        f"[{params['kp_joint_scale_min']:.3f},{params['kp_joint_scale_max']:.3f}] "
        f"kd={params['kd_scale']:.3f}*"
        f"[{params['kd_joint_scale_min']:.3f},{params['kd_joint_scale_max']:.3f}] "
        f"effort={params['effort_scale']:.3f}*"
        f"[{params['effort_joint_scale_min']:.3f},{params['effort_joint_scale_max']:.3f}] "
        f"foot_mu=[{params['foot_friction_scale_min']:.3f},"
        f"{params['foot_friction_scale_max']:.3f}] "
        f"foot_tc=[{params['foot_solref_timeconst_scale_min']:.3f},"
        f"{params['foot_solref_timeconst_scale_max']:.3f}] "
        f"foot_zeta=[{params['foot_solref_dampratio_scale_min']:.3f},"
        f"{params['foot_solref_dampratio_scale_max']:.3f}] "
        f"foot_width=[{params['foot_solimp_width_scale_min']:.3f},"
        f"{params['foot_solimp_width_scale_max']:.3f}] "
        f"foot_radius=[{params['foot_radius_scale_min']:.3f},"
        f"{params['foot_radius_scale_max']:.3f}] "
        f"gravity_delta=({gravity[0]:+.3f},{gravity[1]:+.3f},{gravity[2]:+.3f}) "
        f"base_com=({base_com[0]:+.4f},{base_com[1]:+.4f},{base_com[2]:+.4f}) "
        f"link_com_max={params['link_com_max_abs']:.4f} "
        f"joint_limit_max={params['joint_limit_max_abs']:.4f} "
        f"delay={params['action_delay_physics_steps']}physics_steps/"
        f"{params['action_delay_ms']:.1f}ms "
        f"obs_noise={params['obs_noise_std']:.4f} "
        f"action_noise={params['action_noise_std']:.4f} "
        f"vel_perturb_prob={params['velocity_perturbation_prob']:.4f} "
        f"force_pulse_prob={params['horizontal_force_perturbation_prob']:.4f} "
        f"force_pulse_N={params['horizontal_force_perturbation_range']} "
        f"force_pulse_s={params['horizontal_force_perturbation_duration_range']}"
    )


def _run_eval_single(
    env,
    policy,
    cfg: V2PPOConfig,
    seed: int,
    wandb_run=None,
    print_windows: bool = True,
):
    policy.eval()
    obs_np, _ = env.reset(seed=seed)
    episode_idx = 1
    failure_count = 0
    episode_return = 0.0
    episode_length = 0
    eval_print_window = 100
    foot_bw_window = []
    impact_window = []
    wobble_window = []
    roll_wobble_window = []
    lateral_window = []
    lateral_target_error_window = []
    foot_y_window = []
    foot_bw_all = []
    impact_all = []
    wobble_all = []
    roll_wobble_all = []
    lateral_all = []
    lateral_target_error_all = []
    foot_y_all = []
    startup_retreat_all = []
    threshold_bw = float(env.cfg.foot_force_threshold_bw)
    episode_records = []
    trace_steps = max(int(cfg.failure_trace_steps), 1)
    failure_trace = deque(maxlen=trace_steps)
    episode_dr = env.get_domain_randomization_summary()

    def finish_episode(outcome, final_info):
        record = {
            "seed": int(seed),
            "episode": int(episode_idx),
            "outcome": outcome,
            "failed": int(outcome == "terminated"),
            "length": int(episode_length),
            "return": float(episode_return),
            "final_root_height": float(final_info.get("root_height", 0.0)),
            "final_pitch": float(final_info.get("pitch", 0.0)),
            "final_roll": float(final_info.get("roll", 0.0)),
            "final_wx": float(final_info.get("wx", 0.0)),
            "final_wy": float(final_info.get("wy", 0.0)),
            "final_wz": float(final_info.get("wz", 0.0)),
            "startup_support_mode": str(
                final_info.get("startup_support_mode", "none")
            ),
            "startup_support_initial_fraction": float(
                final_info.get("startup_support_initial_fraction", 0.0)
            ),
            "startup_support_unload_time": float(
                final_info.get("startup_support_unload_time", 0.0)
            ),
            "startup_max_retreat_distance": float(
                final_info.get("startup_max_retreat_distance", 0.0)
            ),
            "startup_retreat_terminated": int(
                bool(final_info.get("startup_retreat_terminated", False))
            ),
        }
        for key, value in episode_dr.items():
            if isinstance(value, (tuple, list, np.ndarray)):
                record[f"dr_{key}"] = json.dumps(np.asarray(value).tolist())
            else:
                record[f"dr_{key}"] = value
        episode_records.append(record)

        if outcome == "terminated" and cfg.save_failure_traces:
            os.makedirs(cfg.save_failure_traces, exist_ok=True)
            samples = list(failure_trace)
            if samples:
                payload = {
                    key: np.stack([sample[key] for sample in samples])
                    for key in samples[0]
                }
                for key, value in episode_dr.items():
                    payload[f"dr_{key}"] = np.asarray(value)
                trace_path = os.path.join(
                    cfg.save_failure_traces,
                    f"failure_seed_{seed:05d}_episode_{episode_idx:03d}.npz",
                )
                np.savez_compressed(trace_path, **payload)
                record["failure_trace"] = trace_path
                print(f"[eval failure trace] saved {trace_path}")

    print(f"Running v2 eval-only for {cfg.eval_steps} steps with seed={seed}...")
    _print_eval_dr(env, episode_idx)
    for step_idx in range(cfg.eval_steps):
        obs_t = base_train.numpy_to_torch_obs(obs_np, device=cfg.device)
        with torch.no_grad():
            action_t = policy.act_deterministic(obs_t)

        action_np = action_t.squeeze(0).cpu().numpy()
        action_np = np.clip(action_np, -cfg.action_clip, cfg.action_clip)
        obs_np, reward, terminated, truncated, info = env.step(action_np)

        episode_return += reward
        episode_length += 1
        left_foot_bw = float(info.get("left_foot_force_bw", 0.0))
        right_foot_bw = float(info.get("right_foot_force_bw", 0.0))
        impact_penalty = float(info.get("penalty_foot_impact", 0.0))
        pitch = float(info.get("pitch", info.get("pitch_err", 0.0)))
        roll = float(info.get("roll", info.get("roll_err", 0.0)))
        wx = float(info.get("wx", 0.0))
        wy = float(info.get("wy", 0.0))
        wz = float(info.get("wz", 0.0))
        roll_wobble_penalty = float(info.get("penalty_roll_wobble", 0.0))
        foot_lateral_distance = float(info.get("foot_lateral_distance", 0.0))
        foot_lateral_target_error = float(info.get("foot_lateral_target_error", 0.0))
        left_foot_local_y = float(info.get("left_foot_local_y", 0.0))
        right_foot_local_y = float(info.get("right_foot_local_y", 0.0))
        if cfg.save_failure_traces:
            failure_trace.append(
                {
                "step": np.asarray(step_idx + 1, dtype=np.int64),
                "episode_step": np.asarray(episode_length, dtype=np.int64),
                "obs": np.asarray(obs_np, dtype=np.float32).copy(),
                "action": np.asarray(action_np, dtype=np.float32).copy(),
                "qpos": np.asarray(env.data.qpos, dtype=np.float32).copy(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float32).copy(),
                "root_height": np.asarray(info.get("root_height", 0.0), dtype=np.float32),
                "pitch": np.asarray(info.get("pitch", 0.0), dtype=np.float32),
                "roll": np.asarray(info.get("roll", 0.0), dtype=np.float32),
                "wx": np.asarray(info.get("wx", 0.0), dtype=np.float32),
                "wy": np.asarray(info.get("wy", 0.0), dtype=np.float32),
                "wz": np.asarray(info.get("wz", 0.0), dtype=np.float32),
                "startup_support_fraction": np.asarray(
                    info.get("startup_support_fraction", 0.0), dtype=np.float32
                ),
                "startup_support_force_n": np.asarray(
                    info.get("startup_support_force_n", 0.0), dtype=np.float32
                ),
                "startup_support_command_scale": np.asarray(
                    info.get("startup_support_command_scale", 1.0), dtype=np.float32
                ),
                "startup_retreat_distance": np.asarray(
                    info.get("startup_retreat_distance", 0.0), dtype=np.float32
                ),
                "startup_max_retreat_distance": np.asarray(
                    info.get("startup_max_retreat_distance", 0.0), dtype=np.float32
                ),
                "startup_no_retreat_active": np.asarray(
                    info.get("startup_no_retreat_active", False), dtype=np.bool_
                ),
                "left_foot_force_bw": np.asarray(left_foot_bw, dtype=np.float32),
                "right_foot_force_bw": np.asarray(right_foot_bw, dtype=np.float32),
                "control_action": np.asarray(info.get("control_action", action_np), dtype=np.float32).copy(),
                "filtered_action": np.asarray(info.get("filtered_action", action_np), dtype=np.float32).copy(),
                "target_pos": np.asarray(info.get("target_pos", np.zeros(env.act_dim)), dtype=np.float32).copy(),
                "torques": np.asarray(info.get("last_torques", np.zeros(env.act_dim)), dtype=np.float32).copy(),
                }
            )

        foot_bw_pair = (left_foot_bw, right_foot_bw)
        wobble_sample = (pitch, roll, wx, wy, wz)
        foot_y_pair = (left_foot_local_y, right_foot_local_y)
        foot_bw_window.append(foot_bw_pair)
        impact_window.append(impact_penalty)
        wobble_window.append(wobble_sample)
        roll_wobble_window.append(roll_wobble_penalty)
        lateral_window.append(foot_lateral_distance)
        lateral_target_error_window.append(foot_lateral_target_error)
        foot_y_window.append(foot_y_pair)
        foot_bw_all.append(foot_bw_pair)
        impact_all.append(impact_penalty)
        wobble_all.append(wobble_sample)
        roll_wobble_all.append(roll_wobble_penalty)
        lateral_all.append(foot_lateral_distance)
        lateral_target_error_all.append(foot_lateral_target_error)
        foot_y_all.append(foot_y_pair)
        startup_retreat_all.append(
            float(info.get("startup_max_retreat_distance", 0.0))
        )

        if print_windows and (step_idx + 1) % eval_print_window == 0:
            window_arr = np.asarray(foot_bw_window, dtype=np.float32)
            impact_arr = np.asarray(impact_window, dtype=np.float32)
            roll_wobble_arr = np.asarray(roll_wobble_window, dtype=np.float32)
            lateral_arr = np.asarray(lateral_window, dtype=np.float32)
            lateral_target_error_arr = np.asarray(lateral_target_error_window, dtype=np.float32)
            foot_y_arr = np.asarray(foot_y_window, dtype=np.float32)
            left_mean, right_mean = np.mean(window_arr, axis=0)
            left_max, right_max = np.max(window_arr, axis=0)
            left_y_mean, right_y_mean = np.mean(foot_y_arr, axis=0)
            exceed_ratio = float(np.mean(np.any(window_arr > threshold_bw, axis=1)))
            impact_p95 = float(np.percentile(impact_arr, 95))
            impact_p99 = float(np.percentile(impact_arr, 99))
            wobble = _wobble_metrics(wobble_window)
            print(
                f"[eval {step_idx + 1:05d}] "
                f"window={eval_print_window} "
                f"reward_last={reward:8.4f} "
                f"root_h={info.get('root_height', 0.0):.3f} "
                f"vel_x={info.get('vel_x', 0.0):+.3f} "
                f"vel_track={info.get('reward_vel_tracking', 0.0):.4f} "
                f"ori={info.get('reward_orientation', 0.0):.4f} "
                f"height_reward={info.get('reward_pelvis_height', 0.0):.4f} "
                f"wobble_score={wobble['score']:.4f} "
                f"wx_rms={wobble['wx_rms']:.4f} "
                f"wy_rms={wobble['wy_rms']:.4f} "
                f"wz_rms={wobble['wz_rms']:.4f} "
                f"roll_penalty_mean={float(np.mean(roll_wobble_arr)):.4f} "
                f"foot_width_mean={float(np.mean(lateral_arr)):.3f} "
                f"foot_width_min={float(np.min(lateral_arr)):.3f} "
                f"foot_width_max={float(np.max(lateral_arr)):.3f} "
                f"target_err_mean={float(np.mean(lateral_target_error_arr)):+.3f} "
                f"foot_y_mean=({left_y_mean:+.3f},{right_y_mean:+.3f}) "
                f"foot_bw_mean=({left_mean:.2f},{right_mean:.2f}) "
                f"foot_bw_max=({left_max:.2f},{right_max:.2f}) "
                f"exceed>{threshold_bw:.2f}={100.0 * exceed_ratio:.1f}% "
                f"impact_mean={float(np.mean(impact_arr)):.4f} "
                f"impact_p95={impact_p95:.4f} "
                f"impact_p99={impact_p99:.4f} "
                f"impact_max={float(np.max(impact_arr)):.4f}"
            )
            base_train.log_wandb(
                wandb_run,
                {
                    "eval_window/reward_last": float(reward),
                    "eval_window/root_h": float(info.get("root_height", 0.0)),
                    "eval_window/vel_x": float(info.get("vel_x", 0.0)),
                    "eval_window/vel_track": float(info.get("reward_vel_tracking", 0.0)),
                    "eval_window/orientation": float(info.get("reward_orientation", 0.0)),
                    "eval_window/height_reward": float(info.get("reward_pelvis_height", 0.0)),
                    "eval_window/wobble_score": wobble["score"],
                    "eval_window/wx_rms": wobble["wx_rms"],
                    "eval_window/wy_rms": wobble["wy_rms"],
                    "eval_window/wz_rms": wobble["wz_rms"],
                    "eval_window/roll_wobble_penalty_mean": float(np.mean(roll_wobble_arr)),
                    "eval_window/foot_width_mean": float(np.mean(lateral_arr)),
                    "eval_window/foot_width_min": float(np.min(lateral_arr)),
                    "eval_window/foot_width_max": float(np.max(lateral_arr)),
                    "eval_window/foot_width_target_error_mean": float(
                        np.mean(lateral_target_error_arr)
                    ),
                    "eval_window/left_foot_local_y_mean": float(left_y_mean),
                    "eval_window/right_foot_local_y_mean": float(right_y_mean),
                    "eval_window/foot_bw_left_mean": float(left_mean),
                    "eval_window/foot_bw_right_mean": float(right_mean),
                    "eval_window/foot_bw_left_max": float(left_max),
                    "eval_window/foot_bw_right_max": float(right_max),
                    "eval_window/foot_exceed_ratio": exceed_ratio,
                    "eval_window/impact_penalty_mean": float(np.mean(impact_arr)),
                    "eval_window/impact_penalty_p95": impact_p95,
                    "eval_window/impact_penalty_p99": impact_p99,
                    "eval_window/impact_penalty_max": float(np.max(impact_arr)),
                },
                step=step_idx + 1,
            )
            foot_bw_window.clear()
            impact_window.clear()
            wobble_window.clear()
            roll_wobble_window.clear()
            lateral_window.clear()
            lateral_target_error_window.clear()
            foot_y_window.clear()

        if terminated or truncated:
            if terminated:
                failure_count += 1
            print(
                f"[eval done] len={episode_length:4d} "
                f"return={episode_return:9.3f} "
                f"root_h={info.get('root_height', 0.0):.3f}"
            )
            finish_episode("terminated" if terminated else "truncated", info)
            if step_idx + 1 < cfg.eval_steps:
                obs_np, _ = env.reset()
                episode_idx += 1
                episode_dr = env.get_domain_randomization_summary()
                failure_trace.clear()
                _print_eval_dr(env, episode_idx)
            episode_return = 0.0
            episode_length = 0

    if episode_length > 0:
        finish_episode("eval_limit", info)

    if foot_bw_all:
        all_arr = np.asarray(foot_bw_all, dtype=np.float32)
        impact_arr = np.asarray(impact_all, dtype=np.float32)
        roll_wobble_arr = np.asarray(roll_wobble_all, dtype=np.float32)
        lateral_arr = np.asarray(lateral_all, dtype=np.float32)
        lateral_target_error_arr = np.asarray(lateral_target_error_all, dtype=np.float32)
        foot_y_arr = np.asarray(foot_y_all, dtype=np.float32)
        left_mean, right_mean = np.mean(all_arr, axis=0)
        left_max, right_max = np.max(all_arr, axis=0)
        left_y_mean, right_y_mean = np.mean(foot_y_arr, axis=0)
        exceed_ratio = float(np.mean(np.any(all_arr > threshold_bw, axis=1)))
        impact_p95 = float(np.percentile(impact_arr, 95))
        impact_p99 = float(np.percentile(impact_arr, 99))
        wobble = _wobble_metrics(wobble_all)
        summary = {
            "seed": int(seed),
            "failure_count": int(failure_count),
            "success": float(failure_count == 0),
            "wobble_score": wobble["score"],
            "pitch_rms": wobble["pitch_rms"],
            "roll_rms": wobble["roll_rms"],
            "signed_roll_mean": wobble["roll_mean"],
            "wx_rms": wobble["wx_rms"],
            "wy_rms": wobble["wy_rms"],
            "wz_rms": wobble["wz_rms"],
            "foot_width_mean": float(np.mean(lateral_arr)),
            "foot_width_min": float(np.min(lateral_arr)),
            "foot_width_max": float(np.max(lateral_arr)),
            "foot_exceed_ratio": exceed_ratio,
            "impact_mean": float(np.mean(impact_arr)),
            "impact_p95": impact_p95,
            "impact_p99": impact_p99,
            "impact_max": float(np.max(impact_arr)),
            "startup_retreat_max": float(
                np.max(startup_retreat_all) if startup_retreat_all else 0.0
            ),
            "startup_retreat_failure_count": int(
                sum(
                    int(record.get("startup_retreat_terminated", 0))
                    for record in episode_records
                )
            ),
            "episode_records": episode_records,
        }
        print(
            f"[eval summary] seed={seed} steps={len(foot_bw_all)} "
            f"wobble_score={wobble['score']:.4f} "
            f"pitch_rms={wobble['pitch_rms']:.4f} "
            f"roll_rms={wobble['roll_rms']:.4f} "
            f"roll_mean={wobble['roll_mean']:+.4f} "
            f"wx_rms={wobble['wx_rms']:.4f} "
            f"wy_rms={wobble['wy_rms']:.4f} "
            f"wz_rms={wobble['wz_rms']:.4f} "
            f"roll_penalty_mean={float(np.mean(roll_wobble_arr)):.4f} "
            f"foot_width_mean={float(np.mean(lateral_arr)):.3f} "
            f"foot_width_min={float(np.min(lateral_arr)):.3f} "
            f"foot_width_max={float(np.max(lateral_arr)):.3f} "
            f"target_err_mean={float(np.mean(lateral_target_error_arr)):+.3f} "
            f"foot_y_mean=({left_y_mean:+.3f},{right_y_mean:+.3f}) "
            f"foot_bw_mean=({left_mean:.2f},{right_mean:.2f}) "
            f"foot_bw_max=({left_max:.2f},{right_max:.2f}) "
            f"exceed>{threshold_bw:.2f}={100.0 * exceed_ratio:.1f}% "
            f"impact_mean={float(np.mean(impact_arr)):.4f} "
            f"impact_p95={impact_p95:.4f} "
            f"impact_p99={impact_p99:.4f} "
            f"impact_max={float(np.max(impact_arr)):.4f} "
            f"startup_retreat_max="
            f"{max(startup_retreat_all, default=0.0):.4f}m "
            f"startup_retreat_failures="
            f"{sum(int(record.get('startup_retreat_terminated', 0)) for record in episode_records)}"
        )
        base_train.log_wandb(
            wandb_run,
            {
                "eval/steps": len(foot_bw_all),
                "eval/wobble_score": wobble["score"],
                "eval/pitch_rms": wobble["pitch_rms"],
                "eval/roll_rms": wobble["roll_rms"],
                "eval/signed_roll_mean": wobble["roll_mean"],
                "eval/wx_rms": wobble["wx_rms"],
                "eval/wy_rms": wobble["wy_rms"],
                "eval/wz_rms": wobble["wz_rms"],
                "eval/roll_wobble_penalty_mean": float(np.mean(roll_wobble_arr)),
                "eval/foot_width_mean": float(np.mean(lateral_arr)),
                "eval/foot_width_min": float(np.min(lateral_arr)),
                "eval/foot_width_max": float(np.max(lateral_arr)),
                "eval/foot_width_target_error_mean": float(np.mean(lateral_target_error_arr)),
                "eval/left_foot_local_y_mean": float(left_y_mean),
                "eval/right_foot_local_y_mean": float(right_y_mean),
                "eval/foot_bw_left_mean": float(left_mean),
                "eval/foot_bw_right_mean": float(right_mean),
                "eval/foot_bw_left_max": float(left_max),
                "eval/foot_bw_right_max": float(right_max),
                "eval/foot_exceed_ratio": exceed_ratio,
                "eval/impact_penalty_mean": float(np.mean(impact_arr)),
                "eval/impact_penalty_p95": impact_p95,
                "eval/impact_penalty_p99": impact_p99,
                "eval/impact_penalty_max": float(np.max(impact_arr)),
            },
            step=len(foot_bw_all),
        )
        return summary
    return None


def run_eval(env, policy, cfg: V2PPOConfig, wandb_run=None):
    num_seeds = max(int(cfg.eval_num_seeds), 1)
    summaries = []
    for offset in range(num_seeds):
        seed = int(cfg.seed) + offset
        if num_seeds > 1:
            print(f"\n===== Multi-seed eval {offset + 1}/{num_seeds}: seed={seed} =====")
        summary = _run_eval_single(env, policy, cfg, seed=seed, wandb_run=wandb_run)
        if summary is not None:
            summaries.append(summary)

    if cfg.eval_report:
        report_dir = os.path.dirname(os.path.abspath(cfg.eval_report))
        os.makedirs(report_dir, exist_ok=True)
        records = [record for summary in summaries for record in summary.get("episode_records", [])]
        if records:
            fieldnames = sorted({key for record in records for key in record})
            with open(cfg.eval_report, "w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
            print(f"[eval report] saved {cfg.eval_report} episodes={len(records)}")

    if len(summaries) <= 1:
        return summaries[0] if summaries else None

    def mean_of(key):
        return float(np.mean([item[key] for item in summaries]))

    def max_of(key):
        return float(np.max([item[key] for item in summaries]))

    success_rate = mean_of("success")
    print(
        f"\n[eval multi-seed summary] seeds={len(summaries)} "
        f"range={summaries[0]['seed']}..{summaries[-1]['seed']} "
        f"success={100.0 * success_rate:.1f}% "
        f"failures={sum(item['failure_count'] for item in summaries)} "
        f"wobble_mean={mean_of('wobble_score'):.4f} "
        f"wobble_worst={max_of('wobble_score'):.4f} "
        f"pitch_mean={mean_of('pitch_rms'):.4f} "
        f"pitch_worst={max_of('pitch_rms'):.4f} "
        f"roll_rms_mean={mean_of('roll_rms'):.4f} "
        f"roll_worst={max_of('roll_rms'):.4f} "
        f"roll_bias_mean={mean_of('signed_roll_mean'):+.4f} "
        f"roll_bias_abs_worst={max(abs(item['signed_roll_mean']) for item in summaries):.4f} "
        f"wx_mean={mean_of('wx_rms'):.4f} "
        f"wx_worst={max_of('wx_rms'):.4f} "
        f"wy_mean={mean_of('wy_rms'):.4f} "
        f"wy_worst={max_of('wy_rms'):.4f} "
        f"wz_mean={mean_of('wz_rms'):.4f} "
        f"wz_worst={max_of('wz_rms'):.4f} "
        f"foot_width_mean={mean_of('foot_width_mean'):.3f} "
        f"foot_width_min_worst={min(item['foot_width_min'] for item in summaries):.3f} "
        f"exceed_mean={100.0 * mean_of('foot_exceed_ratio'):.1f}% "
        f"exceed_worst={100.0 * max_of('foot_exceed_ratio'):.1f}% "
        f"impact_p99_mean={mean_of('impact_p99'):.4f} "
        f"impact_p99_worst={max_of('impact_p99'):.4f} "
        f"impact_max_worst={max_of('impact_max'):.4f} "
        f"startup_retreat_mean={mean_of('startup_retreat_max'):.4f}m "
        f"startup_retreat_worst={max_of('startup_retreat_max'):.4f}m "
        f"startup_retreat_failures="
        f"{sum(item['startup_retreat_failure_count'] for item in summaries)}"
    )
    return summaries


_PERIODIC_EVAL_ENV = None
_BEST_EVAL_STATE = None


def run_periodic_eval(policy, optimizer, cfg, update_idx, save_dir, wandb_run=None):
    global _PERIODIC_EVAL_ENV, _BEST_EVAL_STATE
    if _PERIODIC_EVAL_ENV is None:
        _PERIODIC_EVAL_ENV = SRLMujocoV2Env(make_env_config(cfg, evaluation=True))

    eval_cfg = replace(
        cfg,
        eval_only=True,
        eval_steps=int(cfg.eval_during_training_steps),
        eval_num_seeds=1,
    )
    group_starts = [("primary", int(cfg.eval_during_training_start_seed))]
    secondary_start = getattr(
        cfg, "eval_during_training_secondary_start_seed", None
    )
    if secondary_start is not None and int(secondary_start) != group_starts[0][1]:
        group_starts.append(("secondary", int(secondary_start)))

    group_metrics = []
    all_summaries = []
    seed_count = max(int(cfg.eval_during_training_seeds), 1)
    for group_name, start_seed in group_starts:
        print(
            f"\n===== Periodic held-out DR eval at update {update_idx} "
            f"group={group_name}: seeds={start_seed}..{start_seed + seed_count - 1} ====="
        )
        summaries = []
        for offset in range(seed_count):
            seed = start_seed + offset
            summary = _run_eval_single(
                _PERIODIC_EVAL_ENV,
                policy,
                eval_cfg,
                seed=seed,
                wandb_run=None,
                print_windows=False,
            )
            if summary is not None:
                summaries.append(summary)
                all_summaries.append(summary)
        if summaries:
            metrics = {
                "name": group_name,
                "start_seed": start_seed,
                "success_rate": float(np.mean([item["success"] for item in summaries])),
                "wobble_mean": float(
                    np.mean([item["wobble_score"] for item in summaries])
                ),
                "impact_p99_mean": float(
                    np.mean([item["impact_p99"] for item in summaries])
                ),
            }
            group_metrics.append(metrics)
            print(
                f"[periodic eval group] name={group_name} "
                f"success={100.0 * metrics['success_rate']:.1f}% "
                f"wobble_mean={metrics['wobble_mean']:.4f} "
                f"impact_p99_mean={metrics['impact_p99_mean']:.4f}"
            )

    if not group_metrics:
        policy.train()
        return None

    # Conservative checkpoint selection: both seed groups must improve. Metrics
    # are taken from the worse group, not averaged across easy and hard groups.
    success_rate = min(item["success_rate"] for item in group_metrics)
    wobble_mean = max(item["wobble_mean"] for item in group_metrics)
    impact_p99_mean = max(item["impact_p99_mean"] for item in group_metrics)
    eval_score = (
        100.0 * success_rate
        - wobble_mean
        - float(cfg.eval_impact_p99_weight) * impact_p99_mean
    )
    print(
        f"[periodic eval conservative summary] update={update_idx} "
        f"worst_group_success={100.0 * success_rate:.1f}% "
        f"worst_group_wobble={wobble_mean:.4f} "
        f"worst_group_impact_p99={impact_p99_mean:.4f} "
        f"score={eval_score:.4f}"
    )

    if _BEST_EVAL_STATE is None or eval_score > _BEST_EVAL_STATE["score"]:
        _BEST_EVAL_STATE = {
            "score": eval_score,
            "success_rate": success_rate,
            "wobble_mean": wobble_mean,
            "impact_p99_mean": impact_p99_mean,
            "group_metrics": group_metrics,
            "update": int(update_idx),
        }
        best_path = os.path.join(save_dir, "mujoco_v2_finetune_best_eval.pt")
        torch.save(
            {
                "update": int(update_idx),
                "best_eval_score": eval_score,
                "best_eval_success_rate": success_rate,
                "best_eval_wobble_mean": wobble_mean,
                "best_eval_impact_p99_mean": impact_p99_mean,
                "best_eval_seeds": [item["seed"] for item in all_summaries],
                "best_eval_group_metrics": group_metrics,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
            },
            best_path,
        )
        print(f"Saved best held-out eval checkpoint: {best_path}")

    base_train.log_wandb(
        wandb_run,
        {
            "heldout_eval/success_rate": success_rate,
            "heldout_eval/wobble_mean": wobble_mean,
            "heldout_eval/impact_p99_mean": impact_p99_mean,
            "heldout_eval/score": eval_score,
            "heldout_eval/best_score": _BEST_EVAL_STATE["score"],
            **{
                f"heldout_eval/{item['name']}_success_rate": item["success_rate"]
                for item in group_metrics
            },
            **{
                f"heldout_eval/{item['name']}_wobble_mean": item["wobble_mean"]
                for item in group_metrics
            },
        },
        step=update_idx,
    )
    policy.train()
    return dict(_BEST_EVAL_STATE)


def _parse_six(values, name: str) -> Tuple[float, ...]:
    if len(values) != 6:
        raise ValueError(f"{name} expects exactly 6 values, got {len(values)}")
    return tuple(float(v) for v in values)


def _parse_pair(values, name: str) -> Tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} expects exactly 2 values, got {len(values)}")
    lo, hi = float(values[0]), float(values[1])
    if lo > hi:
        raise ValueError(f"{name} lower bound must be <= upper bound, got {lo} > {hi}")
    return lo, hi


def _load_yaml_defaults(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config root must be a mapping: {path}")
    defaults = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if nested_key in defaults:
                    raise ValueError(f"Duplicate YAML config key: {nested_key}")
                defaults[nested_key] = nested_value
        else:
            defaults[key] = value
    return defaults


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()
    yaml_defaults = _load_yaml_defaults(config_args.config)

    parser = argparse.ArgumentParser(
        description="PPO finetuning for the MuJoCo v2 SRL environment."
    )
    parser.add_argument("--config", type=str, default=config_args.config)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/SRL_Real_Bot_v2_s1.pth")
    parser.add_argument("--checkpoint-key", type=str, default="model")
    parser.add_argument("--xml", type=str, default="mjcf/srl_real_v1/srl_real_bot_v2_pos.xml")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--training-episode-steps", type=int, default=5000)
    parser.add_argument("--reset-envs-each-rollout", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--actor-lr", type=float, default=5e-6)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--critic-warmup-updates", type=int, default=5)
    parser.add_argument("--reference-policy-coef", type=float, default=0.0)
    parser.add_argument("--actor-sym-loss-coef", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument(
        "--normalize-value", dest="normalize_value", action="store_true", default=True
    )
    parser.add_argument(
        "--no-value-normalization", dest="normalize_value", action="store_false"
    )
    parser.add_argument(
        "--clip-value", dest="clip_value", action="store_true", default=True
    )
    parser.add_argument(
        "--no-value-clipping", dest="clip_value", action="store_false"
    )
    parser.add_argument("--bounds-loss-coef", type=float, default=5e-4)
    parser.add_argument("--cvar-fraction", type=float, default=0.0)
    parser.add_argument("--cvar-weight", type=float, default=1.0)
    parser.add_argument("--failure-weight", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=0.005)
    parser.add_argument("--init-log-std", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--eval-episode-steps", type=int, default=5000)
    parser.add_argument("--eval-num-seeds", type=int, default=1)
    parser.add_argument("--eval-during-training", action="store_true")
    parser.add_argument("--eval-every-updates", type=int, default=20)
    parser.add_argument("--eval-during-training-seeds", type=int, default=5)
    parser.add_argument("--eval-during-training-start-seed", type=int, default=101)
    parser.add_argument(
        "--eval-during-training-secondary-start-seed", type=int, default=None
    )
    parser.add_argument("--eval-during-training-steps", type=int, default=2000)
    parser.add_argument("--eval-impact-p99-weight", type=float, default=0.02)
    parser.add_argument("--eval-report", type=str, default=None)
    parser.add_argument("--save-failure-traces", type=str, default=None)
    parser.add_argument("--failure-trace-steps", type=int, default=200)
    parser.add_argument("--debug-rollout", action="store_true")
    parser.add_argument("--resume-optimizer", action="store_true")

    parser.add_argument(
        "--control-backend",
        type=str,
        default="position_target",
        choices=["position_target", "torque_pd"],
    )
    parser.add_argument("--filter-cutoff-hz", type=float, default=8.0)
    parser.add_argument("--no-action-filter", action="store_true")
    parser.add_argument(
        "--pd-tracking-error-limits",
        type=float,
        nargs=6,
        default=None,
        metavar=("J0", "J1", "J2", "J3", "J4", "J5"),
        help=(
            "Optional per-joint |q_target-q| limits in radians, applied after "
            "the action filter."
        ),
    )
    parser.add_argument(
        "--pd-target-change-limit",
        type=float,
        default=None,
        help=(
            "Optional maximum PD-target change per policy control period in radians."
        ),
    )
    parser.add_argument("--root-height", type=float, default=1.1)
    parser.add_argument("--default-dof-pos", type=float, nargs=6, default=None)
    parser.add_argument("--kp", type=float, nargs=6, default=None)
    parser.add_argument("--kd", type=float, nargs=6, default=None)
    parser.add_argument("--effort-limits", type=float, nargs=6, default=None)
    parser.add_argument("--termination-penalty", type=float, default=-25.0)
    parser.add_argument(
        "--actions-rate-scale",
        type=float,
        default=0.3,
        help="Penalty coefficient for the squared first action difference.",
    )
    parser.add_argument(
        "--actions-smoothness-scale",
        type=float,
        default=0.6,
        help="Penalty coefficient for the squared second action difference.",
    )

    parser.add_argument("--base-wobble-penalty-scale", type=float, default=0.0)
    parser.add_argument("--pitch-wobble-penalty-scale", type=float, default=0.0)
    parser.add_argument("--roll-wobble-penalty-scale", type=float, default=0.0)
    parser.add_argument("--roll-rate-weight", type=float, default=0.3)
    parser.add_argument("--base-ang-acc-penalty-scale", type=float, default=0.0)
    parser.add_argument("--yaw-drift-penalty-scale", type=float, default=0.0)
    parser.add_argument("--foot-impact-penalty-scale", type=float, default=0.0)
    parser.add_argument("--foot-force-threshold-bw", type=float, default=1.8)
    parser.add_argument("--foot-force-penalty-power", type=float, default=2.0)
    parser.add_argument("--lateral-min-distance", type=float, default=0.25)
    parser.add_argument("--lateral-max-distance", type=float, default=0.85)
    parser.add_argument("--lateral-symmetry-weight", type=float, default=0.0)
    parser.add_argument("--lateral-target-distance", type=float, default=0.0)
    parser.add_argument("--lateral-target-weight", type=float, default=0.0)
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument("--no-randomize-initial-phase", action="store_true")
    phase_group.add_argument(
        "--initial-phase",
        type=int,
        default=None,
        help="Fix every episode reset to this gait phase (wrapped by gait period).",
    )
    parser.add_argument("--task-training-stage", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--startup-inplace-time", type=float, default=0.8)
    parser.add_argument("--startup-ramp-time", type=float, default=0.8)
    parser.add_argument("--startup-support", dest="startup_support_enable", action="store_true")
    parser.add_argument(
        "--startup-support-in-evaluation",
        action="store_true",
        help="Apply sampled startup support during eval; periodic eval is unassisted by default.",
    )
    parser.add_argument(
        "--startup-support-mode-probabilities",
        type=float,
        nargs=4,
        default=(0.4, 0.4, 0.1, 0.1),
        metavar=("NONE", "SMOOTH", "FAST", "RESIDUAL"),
    )
    parser.add_argument("--startup-support-fraction-range", type=float, nargs=2, default=(0.2, 0.8))
    parser.add_argument("--startup-support-unload-time-range", type=float, nargs=2, default=(0.8, 3.0))
    parser.add_argument(
        "--startup-support-fast-unload-time-range", type=float, nargs=2, default=(0.25, 0.8)
    )
    parser.add_argument(
        "--startup-support-residual-fraction-range", type=float, nargs=2, default=(0.02, 0.10)
    )
    parser.add_argument(
        "--startup-support-residual-hold-time-range", type=float, nargs=2, default=(0.3, 1.0)
    )
    parser.add_argument(
        "--startup-support-final-unload-time-range", type=float, nargs=2, default=(0.3, 1.0)
    )
    parser.add_argument(
        "--startup-support-command-gate",
        dest="startup_support_command_gate_enable",
        action="store_true",
        help=(
            "Hold vx/wz at zero in supported episodes until the support fraction "
            "falls below the configured threshold."
        ),
    )
    parser.add_argument(
        "--startup-support-command-gate-threshold", type=float, default=0.15
    )
    parser.add_argument(
        "--startup-support-command-ramp-time", type=float, default=0.8
    )
    parser.add_argument(
        "--startup-support-fluctuation",
        dest="startup_support_fluctuation_enable",
        action="store_true",
        help="Add low-frequency correlated noise during the residual-support plateau.",
    )
    parser.add_argument(
        "--startup-support-noise-std-range",
        type=float,
        nargs=2,
        default=(0.02, 0.04),
    )
    parser.add_argument(
        "--startup-support-noise-tau-range",
        type=float,
        nargs=2,
        default=(0.5, 1.5),
    )
    parser.add_argument("--startup-support-noise-limit", type=float, default=0.06)
    parser.add_argument(
        "--startup-support-actual-max-fraction", type=float, default=0.5
    )
    parser.add_argument(
        "--startup-tether",
        dest="startup_tether_enable",
        action="store_true",
        help="Apply an optional horizontal spring-damper tether to the base.",
    )
    parser.add_argument(
        "--startup-tether-stiffness-1mg-range",
        type=float,
        nargs=2,
        default=(100.0, 250.0),
    )
    parser.add_argument(
        "--startup-tether-damping-range",
        type=float,
        nargs=2,
        default=(5.0, 25.0),
    )
    parser.add_argument("--startup-tether-deadzone", type=float, default=0.01)
    parser.add_argument(
        "--startup-tether-force-limit-fraction", type=float, default=0.10
    )
    parser.add_argument(
        "--startup-tether-scale-with-support",
        dest="startup_tether_scale_with_support",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-startup-tether-scale-with-support",
        dest="startup_tether_scale_with_support",
        action="store_false",
    )
    parser.add_argument(
        "--startup-no-retreat",
        dest="startup_no_retreat_enable",
        action="store_true",
        help=(
            "Enable v3 startup-only retreat/action penalties and optional "
            "rearward-distance termination. Disabled by default."
        ),
    )
    parser.add_argument("--startup-no-retreat-duration", type=float, default=0.0)
    parser.add_argument("--startup-retreat-penalty-scale", type=float, default=0.0)
    parser.add_argument(
        "--startup-negative-vx-penalty-scale", type=float, default=0.0
    )
    parser.add_argument(
        "--startup-action-magnitude-penalty-scale", type=float, default=0.0
    )
    parser.add_argument(
        "--startup-action-rate-penalty-scale", type=float, default=0.0
    )
    parser.add_argument("--startup-max-retreat-distance", type=float, default=0.0)
    parser.add_argument("--swing-reference", type=str, default=None)
    parser.add_argument("--swing-inplace-reference", type=str, default=None)
    parser.add_argument("--swing-forward-reference", type=str, default=None)
    parser.add_argument("--swing-inplace-probability", type=float, default=0.5)
    parser.add_argument("--swing-forward-command-vx", type=float, default=1.0)
    parser.add_argument("--swing-continuous-command", action="store_true")
    parser.add_argument("--swing-fixed-command-vx", type=float, default=None)
    parser.add_argument(
        "--swing-reset-critic",
        dest="swing_reset_critic",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-swing-reset-critic",
        dest="swing_reset_critic",
        action="store_false",
    )
    parser.add_argument(
        "--swing-reset-actor-output",
        dest="swing_reset_actor_output",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-swing-reset-actor-output",
        dest="swing_reset_actor_output",
        action="store_false",
    )
    parser.add_argument("--swing-suspension-pos-kp", type=float, nargs=3, default=(120.0, 120.0, 300.0))
    parser.add_argument("--swing-suspension-pos-kd", type=float, nargs=3, default=(45.0, 45.0, 80.0))
    parser.add_argument("--swing-suspension-rot-kp", type=float, nargs=3, default=(30.0, 30.0, 8.0))
    parser.add_argument("--swing-suspension-rot-kd", type=float, nargs=3, default=(8.0, 8.0, 3.0))
    parser.add_argument("--swing-q-tracking-scale", type=float, default=8.0)
    parser.add_argument("--swing-action-tracking-scale", type=float, default=2.0)
    parser.add_argument("--swing-orientation-scale", type=float, default=1.0)
    parser.add_argument("--swing-position-scale", type=float, default=0.5)
    parser.add_argument("--swing-dof-vel-penalty-scale", type=float, default=0.01)
    parser.add_argument("--swing-torque-penalty-scale", type=float, default=2.0e-5)
    parser.add_argument("--swing-action-rate-penalty-scale", type=float, default=0.05)
    parser.add_argument("--swing-action-smoothness-penalty-scale", type=float, default=0.10)
    parser.add_argument("--swing-termination-position-error", type=float, default=0.60)
    parser.add_argument("--swing-termination-tilt", type=float, default=0.80)
    parser.add_argument("--swing-fixed-base-timeconst", type=float, default=0.001)
    parser.add_argument("--swing-hip-y-amplitude", type=float, default=0.06)
    parser.add_argument("--swing-knee-amplitude", type=float, default=0.08)
    parser.add_argument("--swing-target-tracking-scale", type=float, default=4.0)
    parser.add_argument("--swing-qd-tracking-scale", type=float, default=2.0)
    parser.add_argument("--swing-hip-x-penalty-scale", type=float, default=6.0)
    parser.add_argument(
        "--swing-hip-x-velocity-penalty-scale", type=float, default=1.0
    )
    parser.add_argument("--swing-foot-x-amplitude", type=float, default=0.10)
    parser.add_argument("--swing-foot-z-amplitude", type=float, default=0.08)
    parser.add_argument("--swing-foot-position-tracking-scale", type=float, default=8.0)
    parser.add_argument("--swing-foot-velocity-tracking-scale", type=float, default=1.5)
    parser.add_argument("--swing-posture-tracking-scale", type=float, default=0.25)
    parser.add_argument("--swing-torque-rate-penalty-scale", type=float, default=0.02)
    parser.add_argument("--swing-joint-limit-penalty-scale", type=float, default=0.50)
    parser.add_argument("--swing-joint-limit-margin-fraction", type=float, default=0.10)
    parser.add_argument("--swing-normalized-qd-scale", type=float, default=8.0)
    parser.add_argument("--swing-termination-joint-tolerance", type=float, default=0.03)

    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--dr-friction-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-kp-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-kp-joint-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-kd-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-kd-joint-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-effort-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-effort-joint-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-gear-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-mass-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-mass-link-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-inertia-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-inertia-link-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-damping-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-damping-joint-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-frictionloss-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-frictionloss-joint-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-armature-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-armature-joint-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-foot-friction-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument(
        "--dr-foot-solref-timeconst-range", type=float, nargs=2, default=(1.0, 1.0)
    )
    parser.add_argument(
        "--dr-foot-solref-dampratio-range", type=float, nargs=2, default=(1.0, 1.0)
    )
    parser.add_argument(
        "--dr-foot-solimp-width-range", type=float, nargs=2, default=(1.0, 1.0)
    )
    parser.add_argument("--dr-foot-radius-range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dr-base-com-shift", type=float, default=0.0)
    parser.add_argument("--dr-link-com-shift", type=float, default=0.0)
    parser.add_argument("--dr-gravity-std", type=float, default=0.0)
    parser.add_argument("--dr-joint-limit-std", type=float, default=0.0)
    parser.add_argument("--dr-stratified-sampling", action="store_true")
    parser.add_argument("--dr-stratified-evaluation", action="store_true")
    parser.add_argument(
        "--dr-scenario-probabilities",
        type=float,
        nargs=6,
        default=(0.45, 0.15, 0.15, 0.05, 0.10, 0.10),
    )
    parser.add_argument(
        "--dr-hard-longitudinal-gravity-sigma-range",
        type=float,
        nargs=2,
        default=(1.5, 2.5),
    )
    parser.add_argument(
        "--dr-hard-lateral-gravity-sigma-range",
        type=float,
        nargs=2,
        default=(1.0, 2.0),
    )
    parser.add_argument(
        "--dr-combined-longitudinal-gravity-sigma-range",
        type=float,
        nargs=2,
        default=(2.0, 3.0),
    )
    parser.add_argument("--obs-noise-std", type=float, default=0.0)
    parser.add_argument("--obs-noise-dof-pos-std", type=float, default=0.0)
    parser.add_argument("--obs-noise-dof-vel-std", type=float, default=0.0)
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument("--action-delay-max-steps", type=int, default=0)
    parser.add_argument("--action-delay-max-physics-steps", type=int, default=None)
    parser.add_argument("--velocity-perturbation", action="store_true")
    parser.add_argument("--velocity-perturbation-prob", type=float, default=1.0 / 500.0)
    parser.add_argument("--velocity-perturbation-range", type=float, nargs=2, default=(0.2, 0.8))
    parser.add_argument("--horizontal-force-perturbation", action="store_true")
    parser.add_argument(
        "--horizontal-force-perturbation-prob", type=float, default=1.0 / 1000.0
    )
    parser.add_argument(
        "--horizontal-force-perturbation-range",
        type=float,
        nargs=2,
        default=(20.0, 80.0),
    )
    parser.add_argument(
        "--horizontal-force-perturbation-duration-range",
        type=float,
        nargs=2,
        default=(0.05, 0.20),
    )
    parser.add_argument("--dr-warmup-updates", type=int, default=10)
    parser.add_argument("--dr-ramp-updates", type=int, default=80)
    parser.add_argument("--ent-coef", type=float, default=0.0)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="srl-mujoco-v2-finetune")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(yaml_defaults) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown YAML config keys: {unknown_keys}")
    parser.set_defaults(**yaml_defaults)
    return parser.parse_args()


def config_from_args(args):
    cfg = V2PPOConfig(
        checkpoint_path=args.checkpoint,
        checkpoint_key=args.checkpoint_key,
        xml_path=args.xml,
        total_updates=args.updates,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        training_episode_steps=args.training_episode_steps,
        reset_envs_each_rollout=args.reset_envs_each_rollout,
        learning_rate=args.lr,
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        critic_warmup_updates=args.critic_warmup_updates,
        reference_policy_coef=args.reference_policy_coef,
        actor_sym_loss_coef=args.actor_sym_loss_coef,
        reward_scale=args.reward_scale,
        normalize_value=args.normalize_value,
        clip_value=args.clip_value,
        bounds_loss_coef=args.bounds_loss_coef,
        cvar_fraction=args.cvar_fraction,
        cvar_weight=args.cvar_weight,
        failure_weight=args.failure_weight,
        target_kl=args.target_kl,
        device=args.device,
        seed=args.seed,
        save_every=args.save_every,
        eval_only=args.eval_only,
        eval_steps=args.eval_steps,
        eval_episode_steps=args.eval_episode_steps,
        eval_num_seeds=args.eval_num_seeds,
        eval_during_training=args.eval_during_training,
        eval_every_updates=args.eval_every_updates,
        eval_during_training_seeds=args.eval_during_training_seeds,
        eval_during_training_start_seed=args.eval_during_training_start_seed,
        eval_during_training_secondary_start_seed=(
            args.eval_during_training_secondary_start_seed
        ),
        eval_during_training_steps=args.eval_during_training_steps,
        eval_impact_p99_weight=args.eval_impact_p99_weight,
        eval_report=args.eval_report,
        save_failure_traces=args.save_failure_traces,
        failure_trace_steps=args.failure_trace_steps,
        debug_rollout=args.debug_rollout,
        init_log_std=args.init_log_std,
        resume_optimizer=args.resume_optimizer,
        control_backend=args.control_backend,
        action_filter_enable=not args.no_action_filter,
        action_filter_cutoff_hz=args.filter_cutoff_hz,
        pd_tracking_error_limits=(
            _parse_six(args.pd_tracking_error_limits, "--pd-tracking-error-limits")
            if args.pd_tracking_error_limits is not None
            else None
        ),
        pd_target_change_limit=args.pd_target_change_limit,
        root_height=args.root_height,
        default_dof_pos=_parse_six(args.default_dof_pos, "--default-dof-pos")
        if args.default_dof_pos is not None
        else V2PPOConfig.default_dof_pos,
        kp=_parse_six(args.kp, "--kp") if args.kp is not None else V2PPOConfig.kp,
        kd=_parse_six(args.kd, "--kd") if args.kd is not None else V2PPOConfig.kd,
        effort_limits=_parse_six(args.effort_limits, "--effort-limits")
        if args.effort_limits is not None
        else V2PPOConfig.effort_limits,
        termination_penalty=args.termination_penalty,
        actions_rate_scale=args.actions_rate_scale,
        actions_smoothness_scale=args.actions_smoothness_scale,
        base_wobble_penalty_scale=args.base_wobble_penalty_scale,
        pitch_wobble_penalty_scale=args.pitch_wobble_penalty_scale,
        roll_wobble_penalty_scale=args.roll_wobble_penalty_scale,
        roll_rate_weight=args.roll_rate_weight,
        base_ang_acc_penalty_scale=args.base_ang_acc_penalty_scale,
        yaw_drift_penalty_scale=args.yaw_drift_penalty_scale,
        foot_impact_penalty_scale=args.foot_impact_penalty_scale,
        foot_force_threshold_bw=args.foot_force_threshold_bw,
        foot_force_penalty_power=args.foot_force_penalty_power,
        lateral_min_distance=args.lateral_min_distance,
        lateral_max_distance=args.lateral_max_distance,
        lateral_symmetry_weight=args.lateral_symmetry_weight,
        lateral_target_distance=args.lateral_target_distance,
        lateral_target_weight=args.lateral_target_weight,
        randomize_initial_phase=(
            args.initial_phase is None and not args.no_randomize_initial_phase
        ),
        initial_phase=args.initial_phase,
        task_training_stage=args.task_training_stage,
        startup_inplace_time=args.startup_inplace_time,
        startup_ramp_time=args.startup_ramp_time,
        startup_support_enable=args.startup_support_enable,
        startup_support_in_evaluation=args.startup_support_in_evaluation,
        startup_support_mode_probabilities=tuple(args.startup_support_mode_probabilities),
        startup_support_fraction_range=_parse_pair(
            args.startup_support_fraction_range, "--startup-support-fraction-range"
        ),
        startup_support_unload_time_range=_parse_pair(
            args.startup_support_unload_time_range,
            "--startup-support-unload-time-range",
        ),
        startup_support_fast_unload_time_range=_parse_pair(
            args.startup_support_fast_unload_time_range,
            "--startup-support-fast-unload-time-range",
        ),
        startup_support_residual_fraction_range=_parse_pair(
            args.startup_support_residual_fraction_range,
            "--startup-support-residual-fraction-range",
        ),
        startup_support_residual_hold_time_range=_parse_pair(
            args.startup_support_residual_hold_time_range,
            "--startup-support-residual-hold-time-range",
        ),
        startup_support_final_unload_time_range=_parse_pair(
            args.startup_support_final_unload_time_range,
            "--startup-support-final-unload-time-range",
        ),
        startup_support_command_gate_enable=(
            args.startup_support_command_gate_enable
        ),
        startup_support_command_gate_threshold=(
            args.startup_support_command_gate_threshold
        ),
        startup_support_command_ramp_time=(
            args.startup_support_command_ramp_time
        ),
        startup_support_fluctuation_enable=(
            args.startup_support_fluctuation_enable
        ),
        startup_support_noise_std_range=_parse_pair(
            args.startup_support_noise_std_range,
            "--startup-support-noise-std-range",
        ),
        startup_support_noise_tau_range=_parse_pair(
            args.startup_support_noise_tau_range,
            "--startup-support-noise-tau-range",
        ),
        startup_support_noise_limit=args.startup_support_noise_limit,
        startup_support_actual_max_fraction=(
            args.startup_support_actual_max_fraction
        ),
        startup_tether_enable=args.startup_tether_enable,
        startup_tether_stiffness_1mg_range=_parse_pair(
            args.startup_tether_stiffness_1mg_range,
            "--startup-tether-stiffness-1mg-range",
        ),
        startup_tether_damping_range=_parse_pair(
            args.startup_tether_damping_range,
            "--startup-tether-damping-range",
        ),
        startup_tether_deadzone=args.startup_tether_deadzone,
        startup_tether_force_limit_fraction=(
            args.startup_tether_force_limit_fraction
        ),
        startup_tether_scale_with_support=(
            args.startup_tether_scale_with_support
        ),
        startup_no_retreat_enable=args.startup_no_retreat_enable,
        startup_no_retreat_duration=args.startup_no_retreat_duration,
        startup_retreat_penalty_scale=args.startup_retreat_penalty_scale,
        startup_negative_vx_penalty_scale=(
            args.startup_negative_vx_penalty_scale
        ),
        startup_action_magnitude_penalty_scale=(
            args.startup_action_magnitude_penalty_scale
        ),
        startup_action_rate_penalty_scale=(
            args.startup_action_rate_penalty_scale
        ),
        startup_max_retreat_distance=args.startup_max_retreat_distance,
        swing_reference_path=args.swing_reference,
        swing_inplace_reference_path=args.swing_inplace_reference,
        swing_forward_reference_path=args.swing_forward_reference,
        swing_inplace_probability=args.swing_inplace_probability,
        swing_forward_command_vx=args.swing_forward_command_vx,
        swing_continuous_command=args.swing_continuous_command,
        swing_fixed_command_vx=args.swing_fixed_command_vx,
        swing_reset_critic=args.swing_reset_critic,
        swing_reset_actor_output=args.swing_reset_actor_output,
        swing_suspension_pos_kp=tuple(args.swing_suspension_pos_kp),
        swing_suspension_pos_kd=tuple(args.swing_suspension_pos_kd),
        swing_suspension_rot_kp=tuple(args.swing_suspension_rot_kp),
        swing_suspension_rot_kd=tuple(args.swing_suspension_rot_kd),
        swing_q_tracking_scale=args.swing_q_tracking_scale,
        swing_action_tracking_scale=args.swing_action_tracking_scale,
        swing_orientation_scale=args.swing_orientation_scale,
        swing_position_scale=args.swing_position_scale,
        swing_dof_vel_penalty_scale=args.swing_dof_vel_penalty_scale,
        swing_torque_penalty_scale=args.swing_torque_penalty_scale,
        swing_action_rate_penalty_scale=args.swing_action_rate_penalty_scale,
        swing_action_smoothness_penalty_scale=(
            args.swing_action_smoothness_penalty_scale
        ),
        swing_termination_position_error=args.swing_termination_position_error,
        swing_termination_tilt=args.swing_termination_tilt,
        swing_fixed_base_timeconst=args.swing_fixed_base_timeconst,
        swing_hip_y_amplitude=args.swing_hip_y_amplitude,
        swing_knee_amplitude=args.swing_knee_amplitude,
        swing_target_tracking_scale=args.swing_target_tracking_scale,
        swing_qd_tracking_scale=args.swing_qd_tracking_scale,
        swing_hip_x_penalty_scale=args.swing_hip_x_penalty_scale,
        swing_hip_x_velocity_penalty_scale=(
            args.swing_hip_x_velocity_penalty_scale
        ),
        swing_foot_x_amplitude=args.swing_foot_x_amplitude,
        swing_foot_z_amplitude=args.swing_foot_z_amplitude,
        swing_foot_position_tracking_scale=(
            args.swing_foot_position_tracking_scale
        ),
        swing_foot_velocity_tracking_scale=(
            args.swing_foot_velocity_tracking_scale
        ),
        swing_posture_tracking_scale=args.swing_posture_tracking_scale,
        swing_torque_rate_penalty_scale=args.swing_torque_rate_penalty_scale,
        swing_joint_limit_penalty_scale=args.swing_joint_limit_penalty_scale,
        swing_joint_limit_margin_fraction=args.swing_joint_limit_margin_fraction,
        swing_normalized_qd_scale=args.swing_normalized_qd_scale,
        swing_termination_joint_tolerance=(
            args.swing_termination_joint_tolerance
        ),
        domain_randomization_enable=args.domain_randomization,
        dr_friction_range=_parse_pair(args.dr_friction_range, "--dr-friction-range"),
        dr_kp_range=_parse_pair(args.dr_kp_range, "--dr-kp-range"),
        dr_kp_joint_range=_parse_pair(args.dr_kp_joint_range, "--dr-kp-joint-range"),
        dr_kd_range=_parse_pair(args.dr_kd_range, "--dr-kd-range"),
        dr_kd_joint_range=_parse_pair(args.dr_kd_joint_range, "--dr-kd-joint-range"),
        dr_effort_range=_parse_pair(args.dr_effort_range, "--dr-effort-range"),
        dr_effort_joint_range=_parse_pair(
            args.dr_effort_joint_range, "--dr-effort-joint-range"
        ),
        dr_gear_range=_parse_pair(args.dr_gear_range, "--dr-gear-range"),
        dr_mass_range=_parse_pair(args.dr_mass_range, "--dr-mass-range"),
        dr_mass_link_range=_parse_pair(args.dr_mass_link_range, "--dr-mass-link-range"),
        dr_inertia_range=_parse_pair(args.dr_inertia_range, "--dr-inertia-range"),
        dr_inertia_link_range=_parse_pair(
            args.dr_inertia_link_range, "--dr-inertia-link-range"
        ),
        dr_damping_range=_parse_pair(args.dr_damping_range, "--dr-damping-range"),
        dr_damping_joint_range=_parse_pair(
            args.dr_damping_joint_range, "--dr-damping-joint-range"
        ),
        dr_frictionloss_range=_parse_pair(args.dr_frictionloss_range, "--dr-frictionloss-range"),
        dr_frictionloss_joint_range=_parse_pair(
            args.dr_frictionloss_joint_range, "--dr-frictionloss-joint-range"
        ),
        dr_armature_range=_parse_pair(args.dr_armature_range, "--dr-armature-range"),
        dr_armature_joint_range=_parse_pair(
            args.dr_armature_joint_range, "--dr-armature-joint-range"
        ),
        dr_foot_friction_range=_parse_pair(
            args.dr_foot_friction_range, "--dr-foot-friction-range"
        ),
        dr_foot_solref_timeconst_range=_parse_pair(
            args.dr_foot_solref_timeconst_range, "--dr-foot-solref-timeconst-range"
        ),
        dr_foot_solref_dampratio_range=_parse_pair(
            args.dr_foot_solref_dampratio_range, "--dr-foot-solref-dampratio-range"
        ),
        dr_foot_solimp_width_range=_parse_pair(
            args.dr_foot_solimp_width_range, "--dr-foot-solimp-width-range"
        ),
        dr_foot_radius_range=_parse_pair(args.dr_foot_radius_range, "--dr-foot-radius-range"),
        dr_base_com_shift=args.dr_base_com_shift,
        dr_link_com_shift=args.dr_link_com_shift,
        dr_gravity_std=args.dr_gravity_std,
        dr_joint_limit_std=args.dr_joint_limit_std,
        dr_stratified_sampling_enable=args.dr_stratified_sampling,
        dr_stratified_evaluation_enable=args.dr_stratified_evaluation,
        dr_scenario_probabilities=tuple(
            float(value) for value in args.dr_scenario_probabilities
        ),
        dr_hard_longitudinal_gravity_sigma_range=_parse_pair(
            args.dr_hard_longitudinal_gravity_sigma_range,
            "--dr-hard-longitudinal-gravity-sigma-range",
        ),
        dr_hard_lateral_gravity_sigma_range=_parse_pair(
            args.dr_hard_lateral_gravity_sigma_range,
            "--dr-hard-lateral-gravity-sigma-range",
        ),
        dr_combined_longitudinal_gravity_sigma_range=_parse_pair(
            args.dr_combined_longitudinal_gravity_sigma_range,
            "--dr-combined-longitudinal-gravity-sigma-range",
        ),
        obs_noise_std=args.obs_noise_std,
        obs_noise_dof_pos_std=args.obs_noise_dof_pos_std,
        obs_noise_dof_vel_std=args.obs_noise_dof_vel_std,
        action_noise_std=args.action_noise_std,
        action_delay_max_steps=args.action_delay_max_steps,
        action_delay_max_physics_steps=args.action_delay_max_physics_steps,
        velocity_perturbation_enable=args.velocity_perturbation,
        velocity_perturbation_prob=args.velocity_perturbation_prob,
        velocity_perturbation_range=_parse_pair(
            args.velocity_perturbation_range, "--velocity-perturbation-range"
        ),
        horizontal_force_perturbation_enable=(
            args.horizontal_force_perturbation
        ),
        horizontal_force_perturbation_prob=(
            args.horizontal_force_perturbation_prob
        ),
        horizontal_force_perturbation_range=_parse_pair(
            args.horizontal_force_perturbation_range,
            "--horizontal-force-perturbation-range",
        ),
        horizontal_force_perturbation_duration_range=_parse_pair(
            args.horizontal_force_perturbation_duration_range,
            "--horizontal-force-perturbation-duration-range",
        ),
        dr_warmup_updates=args.dr_warmup_updates,
        dr_ramp_updates=args.dr_ramp_updates,
        ent_coef=args.ent_coef,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )

    if cfg.wandb_run_name is None:
        prefix = "mujoco_v2_eval" if cfg.eval_only else "mujoco_v2_finetune"
        cfg.wandb_run_name = f"{prefix}_{time.strftime('%m%d_%H%M%S')}"

    return cfg


def main():
    args = parse_args()
    cfg = config_from_args(args)

    global _PERIODIC_EVAL_ENV, _BEST_EVAL_STATE
    _PERIODIC_EVAL_ENV = None
    _BEST_EVAL_STATE = None
    base_train.make_env_and_model = make_env_and_model
    base_train.make_additional_env = make_additional_env
    base_train.save_checkpoint = save_checkpoint
    base_train.run_eval = run_eval
    base_train.run_periodic_eval = run_periodic_eval
    base_train.train(cfg)


if __name__ == "__main__":
    main()
