from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import torch

import mujoco_ppo.train_mujoco_v2_finetune as v2_train
from mujoco_ppo.models import numpy_to_torch_obs
from mujoco_ppo.models_v3 import AsymmetricModelConfig, load_asymmetric_checkpoint
from mujoco_ppo.srl_mujoco_v4_env import SRLMujocoV4Env, V4WalkEnvConfig


V4_EFFORT_LIMITS = (90.0, 90.0, 350.0, 90.0, 90.0, 350.0)
V4_DEFAULT_XML = "mjcf/srl_real_v1/srl_real_bot_nobattery.xml"
V4_CONTROL_DT = 0.015


def resolve_physics_timing(physics_hz: float) -> Tuple[float, int]:
    physics_hz = float(physics_hz)
    if physics_hz <= 0.0:
        raise ValueError("--physics-hz must be positive.")
    decimation = int(round(V4_CONTROL_DT * physics_hz))
    if decimation < 1 or not np.isclose(
        decimation / physics_hz, V4_CONTROL_DT, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError(
            "--physics-hz must preserve the 0.015 s policy period with an "
            "integer decimation (for example 200 or 400 Hz)."
        )
    return 1.0 / physics_hz, decimation


@dataclass
class V4PPOConfig(v2_train.V2PPOConfig):
    physics_hz: float = 200.0
    gait_period: int = 54
    gait_period_start: Optional[int] = None
    gait_period_ramp_updates: int = 0
    dr_connector_stiffness_range: Tuple[float, float] = (1.0, 1.0)
    dr_connector_damping_range: Tuple[float, float] = (1.0, 1.0)


def parse_v4_args():
    """Parse v4-only options, then delegate the established options to v2."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--physics-hz", type=float, default=200.0)
    parser.add_argument("--gait-period", type=int, default=54)
    parser.add_argument("--gait-period-start", type=int, default=None)
    parser.add_argument("--gait-period-ramp-updates", type=int, default=0)
    parser.add_argument(
        "--dr-connector-stiffness-range",
        type=float,
        nargs=2,
        default=(1.0, 1.0),
    )
    parser.add_argument(
        "--dr-connector-damping-range",
        type=float,
        nargs=2,
        default=(1.0, 1.0),
    )
    v4_args, remaining = parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        shared_args = v2_train.parse_args()
    finally:
        sys.argv = original_argv
    if v4_args.gait_period < 2:
        raise ValueError("--gait-period must be at least 2 control steps.")
    if v4_args.gait_period_start is not None and v4_args.gait_period_start < 2:
        raise ValueError("--gait-period-start must be at least 2 control steps.")
    if v4_args.gait_period_ramp_updates < 0:
        raise ValueError("--gait-period-ramp-updates cannot be negative.")
    resolve_physics_timing(v4_args.physics_hz)
    for name, value_range in (
        (
            "--dr-connector-stiffness-range",
            v4_args.dr_connector_stiffness_range,
        ),
        (
            "--dr-connector-damping-range",
            v4_args.dr_connector_damping_range,
        ),
    ):
        lo, hi = map(float, value_range)
        if lo <= 0.0 or hi < lo:
            raise ValueError(f"{name} must be positive and ordered.")
    return shared_args, v4_args


class AsymmetricRolloutBuffer:
    def __init__(self, steps, num_envs, actor_obs_dim, critic_obs_dim, act_dim, device):
        shape = (steps, num_envs)
        self.steps = steps
        self.num_envs = num_envs
        self.actor_obs = torch.zeros((*shape, actor_obs_dim), device=device)
        self.mirrored_actor_obs = torch.zeros((*shape, actor_obs_dim), device=device)
        self.critic_obs = torch.zeros((*shape, critic_obs_dim), device=device)
        self.actions = torch.zeros((*shape, act_dim), device=device)
        self.logprobs = torch.zeros(shape, device=device)
        self.rewards = torch.zeros(shape, device=device)
        self.terminated = torch.zeros(shape, device=device)
        self.truncated = torch.zeros(shape, device=device)
        self.values = torch.zeros(shape, device=device)
        self.next_values = torch.zeros(shape, device=device)
        self.advantages = torch.zeros(shape, device=device)
        self.returns = torch.zeros(shape, device=device)
        self.ptr = 0

    def add(self, actor_obs, mirrored_actor_obs, critic_obs, actions, logprobs, rewards,
            terminated, truncated, values, next_values):
        i = self.ptr
        self.actor_obs[i] = actor_obs
        self.mirrored_actor_obs[i] = mirrored_actor_obs
        self.critic_obs[i] = critic_obs
        self.actions[i] = actions
        self.logprobs[i] = logprobs
        self.rewards[i] = rewards
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.values[i] = values
        self.next_values[i] = next_values
        self.ptr += 1

    def compute_returns(self, gamma, gae_lambda):
        gae = torch.zeros(self.num_envs, device=self.rewards.device)
        for t in reversed(range(self.steps)):
            bootstrap = 1.0 - self.terminated[t]
            continuation = 1.0 - torch.clamp(
                self.terminated[t] + self.truncated[t], 0.0, 1.0
            )
            delta = (
                self.rewards[t]
                + gamma * bootstrap * self.next_values[t]
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * continuation * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def batches(self, minibatch_size):
        size = self.steps * self.num_envs
        arrays = (
            self.actor_obs.reshape(size, -1),
            self.mirrored_actor_obs.reshape(size, -1),
            self.critic_obs.reshape(size, -1),
            self.actions.reshape(size, -1),
            self.logprobs.reshape(size),
            self.advantages.reshape(size),
            self.returns.reshape(size),
            self.values.reshape(size),
        )
        indices = torch.randperm(size, device=self.rewards.device)
        for start in range(0, size, minibatch_size):
            selected = indices[start : start + minibatch_size]
            yield tuple(array[selected] for array in arrays)


def make_env_config(cfg, *, evaluation=False):
    base = v2_train.make_env_config(cfg, evaluation=evaluation)
    values = asdict(base)
    physics_dt, decimation = resolve_physics_timing(cfg.physics_hz)
    values["dt"] = physics_dt
    values["decimation"] = decimation
    values["dof_vel_obs_scale"] = 0.1
    values["action_filter_cutoff_hz"] = float(cfg.action_filter_cutoff_hz)
    values["effort_limits"] = tuple(cfg.effort_limits)
    initial_gait_period = cfg.gait_period
    if not evaluation and cfg.gait_period_start is not None:
        initial_gait_period = cfg.gait_period_start
    values["gait_period"] = int(initial_gait_period)
    values["dr_connector_stiffness_range"] = tuple(
        cfg.dr_connector_stiffness_range
    )
    values["dr_connector_damping_range"] = tuple(
        cfg.dr_connector_damping_range
    )
    values["task_training_stage"] = int(cfg.task_training_stage)
    values["startup_inplace_time"] = float(cfg.startup_inplace_time)
    values["startup_ramp_time"] = float(cfg.startup_ramp_time)
    values["startup_support_enable"] = bool(
        cfg.startup_support_enable
        and (not evaluation or cfg.startup_support_in_evaluation)
    )
    values["startup_support_mode_probabilities"] = tuple(
        cfg.startup_support_mode_probabilities
    )
    values["startup_support_fraction_range"] = tuple(
        cfg.startup_support_fraction_range
    )
    values["startup_support_unload_time_range"] = tuple(
        cfg.startup_support_unload_time_range
    )
    values["startup_support_fast_unload_time_range"] = tuple(
        cfg.startup_support_fast_unload_time_range
    )
    values["startup_support_residual_fraction_range"] = tuple(
        cfg.startup_support_residual_fraction_range
    )
    values["startup_support_residual_hold_time_range"] = tuple(
        cfg.startup_support_residual_hold_time_range
    )
    values["startup_support_final_unload_time_range"] = tuple(
        cfg.startup_support_final_unload_time_range
    )
    values["startup_support_command_gate_enable"] = bool(
        cfg.startup_support_command_gate_enable
    )
    values["startup_support_command_gate_threshold"] = float(
        cfg.startup_support_command_gate_threshold
    )
    values["startup_support_command_ramp_time"] = float(
        cfg.startup_support_command_ramp_time
    )
    values["startup_support_fluctuation_enable"] = bool(
        cfg.startup_support_fluctuation_enable
    )
    values["startup_support_noise_std_range"] = tuple(
        cfg.startup_support_noise_std_range
    )
    values["startup_support_noise_tau_range"] = tuple(
        cfg.startup_support_noise_tau_range
    )
    values["startup_support_noise_limit"] = float(
        cfg.startup_support_noise_limit
    )
    values["startup_support_actual_max_fraction"] = float(
        cfg.startup_support_actual_max_fraction
    )
    values["startup_tether_enable"] = bool(cfg.startup_tether_enable)
    values["startup_tether_stiffness_1mg_range"] = tuple(
        cfg.startup_tether_stiffness_1mg_range
    )
    values["startup_tether_damping_range"] = tuple(
        cfg.startup_tether_damping_range
    )
    values["startup_tether_deadzone"] = float(cfg.startup_tether_deadzone)
    values["startup_tether_force_limit_fraction"] = float(
        cfg.startup_tether_force_limit_fraction
    )
    values["startup_tether_scale_with_support"] = bool(
        cfg.startup_tether_scale_with_support
    )
    values["startup_no_retreat_enable"] = bool(
        cfg.startup_no_retreat_enable
    )
    values["startup_no_retreat_duration"] = float(
        cfg.startup_no_retreat_duration
    )
    values["startup_retreat_penalty_scale"] = float(
        cfg.startup_retreat_penalty_scale
    )
    values["startup_negative_vx_penalty_scale"] = float(
        cfg.startup_negative_vx_penalty_scale
    )
    values["startup_action_magnitude_penalty_scale"] = float(
        cfg.startup_action_magnitude_penalty_scale
    )
    values["startup_action_rate_penalty_scale"] = float(
        cfg.startup_action_rate_penalty_scale
    )
    values["startup_max_retreat_distance"] = float(
        cfg.startup_max_retreat_distance
    )
    return V4WalkEnvConfig(**values)


def make_env(cfg, *, evaluation=False):
    return SRLMujocoV4Env(make_env_config(cfg, evaluation=evaluation))


def make_policy(cfg, env):
    inferred = v2_train.infer_model_config(
        cfg.checkpoint_path,
        cfg.checkpoint_key,
        obs_dim=env.actor_obs_dim,
        act_dim=env.act_dim,
    )
    model_cfg = AsymmetricModelConfig(
        actor_obs_dim=env.actor_obs_dim,
        critic_obs_dim=env.critic_obs_dim,
        act_dim=env.act_dim,
        hidden_sizes=inferred.hidden_sizes,
        init_log_std=0.0,
    )
    policy, metadata = load_asymmetric_checkpoint(
        cfg.checkpoint_path,
        model_cfg=model_cfg,
        device=cfg.device,
        checkpoint_key=cfg.checkpoint_key,
    )
    metadata.update(
        {
            "v4_env": True,
            "model_hidden_sizes": model_cfg.hidden_sizes,
            "dof_vel_obs_scale": 0.1,
            "effort_limits": tuple(cfg.effort_limits),
            "gait_period": int(cfg.gait_period),
            "gait_period_start": cfg.gait_period_start,
            "gait_period_ramp_updates": int(cfg.gait_period_ramp_updates),
            "dr_connector_stiffness_range": tuple(
                cfg.dr_connector_stiffness_range
            ),
            "dr_connector_damping_range": tuple(
                cfg.dr_connector_damping_range
            ),
        }
    )
    return policy, metadata


def critic_value(policy, critic_obs_np, device, normalize_value):
    critic_obs = numpy_to_torch_obs(critic_obs_np, device=device)
    with torch.no_grad():
        value = policy.critic(policy.normalize_critic_obs(critic_obs))
        if normalize_value:
            value = policy.value_norm.denormalize(value)
    return value


def optimizer_state(actor_optimizer, critic_optimizer):
    return {
        "actor": actor_optimizer.state_dict(),
        "critic": critic_optimizer.state_dict(),
    }


def save_checkpoint(policy, actor_optimizer, critic_optimizer, cfg, update, save_dir, name=None):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_prefix = (
        "mujoco_v4_swing"
        if getattr(cfg, "swing_reference_path", None)
        else "mujoco_v4_finetune"
    )
    filename = name or f"{checkpoint_prefix}_update_{update:05d}.pt"
    path = os.path.join(save_dir, filename)
    torch.save(
        {
            "update": int(update),
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer_state(actor_optimizer, critic_optimizer),
            "config": asdict(cfg),
            "actor_obs_dim": 133,
            "critic_obs_dim": 153,
        },
        path,
    )
    return path


def run_heldout_eval(
    policy,
    actor_optimizer,
    critic_optimizer,
    cfg,
    update,
    save_dir,
    state,
    wandb_run=None,
):
    env = state.get("env")
    if env is None:
        env = make_env(cfg, evaluation=True)
        env.set_dr_progress(1.0)
        state["env"] = env
    eval_cfg = copy.copy(cfg)
    eval_cfg.eval_only = True
    eval_cfg.eval_steps = int(cfg.eval_during_training_steps)
    eval_cfg.eval_num_seeds = 1
    summaries = []
    for offset in range(max(int(cfg.eval_during_training_seeds), 1)):
        summary = v2_train._run_eval_single(
            env,
            policy,
            eval_cfg,
            seed=int(cfg.eval_during_training_start_seed) + offset,
            print_windows=False,
        )
        if summary:
            summaries.append(summary)
    if not summaries:
        return
    success = float(np.mean([item["success"] for item in summaries]))
    wobble = float(np.mean([item["wobble_score"] for item in summaries]))
    impact = float(np.mean([item["impact_p99"] for item in summaries]))
    retreat = float(
        np.mean([item.get("startup_retreat_max", 0.0) for item in summaries])
    )
    score = 100.0 * success - wobble - float(cfg.eval_impact_p99_weight) * impact
    if cfg.startup_no_retreat_enable:
        # Crossing the configured distance already counts as failure. This
        # small tie-breaker prefers less retreat among equally successful runs.
        score -= retreat
    print(
        f"[v4 heldout] update={update} success={100.0 * success:.1f}% "
        f"wobble={wobble:.4f} impact_p99={impact:.4f} "
        f"startup_retreat={retreat:.4f}m score={score:.4f}"
    )
    v2_train.base_train.log_wandb(
        wandb_run,
        {
            "eval/success_rate": success,
            "eval/wobble_score": wobble,
            "eval/impact_p99": impact,
            "eval/startup_retreat_max_mean": retreat,
            "eval/selection_score": score,
        },
        step=update,
    )
    if score > state.get("best_score", -float("inf")):
        state["best_score"] = score
        path = save_checkpoint(
            policy,
            actor_optimizer,
            critic_optimizer,
            cfg,
            update,
            save_dir,
            "mujoco_v4_finetune_best_eval.pt",
        )
        print(f"Saved best held-out v4 checkpoint: {path}")


def train(
    cfg,
    *,
    env_factory=None,
    policy_factory=None,
    eval_fn=None,
    heldout_eval_fn=None,
    run_name_prefix="mujoco_v4_finetune",
):
    env_factory = env_factory or make_env
    policy_factory = policy_factory or make_policy
    eval_fn = eval_fn or v2_train.run_eval
    heldout_eval_fn = heldout_eval_fn or run_heldout_eval

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    envs = [env_factory(cfg, evaluation=cfg.eval_only)]
    if not cfg.eval_only:
        envs.extend(
            env_factory(cfg) for _ in range(max(int(cfg.num_envs), 1) - 1)
        )
    env = envs[0]
    policy, metadata = policy_factory(cfg, env)
    if cfg.init_log_std is not None:
        policy.log_std.data.fill_(float(cfg.init_log_std))
    print("Loaded checkpoint metadata:")
    print(metadata)
    print(
        f"v4 actor_obs_dim={env.actor_obs_dim} critic_obs_dim={env.critic_obs_dim} "
        f"act_dim={env.act_dim} num_envs={len(envs)}"
    )
    print(
        f"physics={cfg.physics_hz:g}Hz dt={env.cfg.dt:.6f}s "
        f"policy={1.0 / env.control_dt:.3f}Hz "
        f"control_dt={env.control_dt:.6f}s decimation={env.cfg.decimation}"
    )
    print(
        f"gait_period={cfg.gait_period:g} control steps/"
        f"{cfg.gait_period * env.control_dt:.3f}s "
        f"({1.0 / (cfg.gait_period * env.control_dt):.3f} cycles/s)"
    )
    if cfg.gait_period_start is not None and cfg.gait_period_ramp_updates > 0:
        print(
            "gait_period_curriculum="
            f"{cfg.gait_period_start}->{cfg.gait_period} over "
            f"{cfg.gait_period_ramp_updates} updates"
        )
    print(
        "connector_dr="
        f"stiffness_scale={tuple(cfg.dr_connector_stiffness_range)}, "
        f"damping_scale={tuple(cfg.dr_connector_damping_range)}"
    )

    if cfg.eval_only:
        env.set_dr_progress(1.0)
        eval_fn(env, policy, cfg)
        return

    actor_params = list(policy.actor_mlp.parameters()) + list(policy.mu.parameters()) + [policy.log_std]
    critic_params = list(policy.critic_mlp.parameters()) + list(policy.value.parameters())
    actor_optimizer = torch.optim.Adam(actor_params, lr=float(cfg.actor_learning_rate))
    critic_optimizer = torch.optim.Adam(critic_params, lr=float(cfg.critic_learning_rate))
    if cfg.resume_optimizer:
        checkpoint = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
        optimizers = checkpoint.get("optimizer_state_dict", {}) if isinstance(checkpoint, dict) else {}
        if isinstance(optimizers, dict) and "actor" in optimizers and "critic" in optimizers:
            actor_optimizer.load_state_dict(optimizers["actor"])
            critic_optimizer.load_state_dict(optimizers["critic"])
            print("Loaded v4 actor and critic optimizer states.")

    reference_policy = None
    if cfg.reference_policy_coef > 0.0:
        reference_policy = copy.deepcopy(policy).eval()
        for parameter in reference_policy.parameters():
            parameter.requires_grad_(False)

    num_envs = len(envs)
    reset_results = []
    for i, worker in enumerate(envs):
        worker.set_dr_progress(0.0 if cfg.domain_randomization_enable else 1.0)
        reset_results.append(worker.reset(seed=cfg.seed + i * 100003))
    actor_obs_np = np.stack([result[0] for result in reset_results])
    mirrored_actor_obs_np = np.stack(
        [result[1]["actor_obs_mirrored"] for result in reset_results]
    )
    critic_obs_np = np.stack([result[1]["critic_obs"] for result in reset_results])
    episode_returns = np.zeros(num_envs, dtype=np.float64)
    episode_lengths = np.zeros(num_envs, dtype=np.int64)

    run_name = f"{run_name_prefix}_{time.strftime('%m%d_%H%M%S')}"
    save_dir = os.path.join(cfg.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving v4 checkpoints to: {save_dir}")
    wandb_run = v2_train.base_train.init_wandb(cfg, run_name)
    best_returns = deque(maxlen=5)
    recent_episode_returns = deque(maxlen=100)
    recent_episode_lengths = deque(maxlen=100)
    recent_episode_successes = deque(maxlen=100)
    best_return = -float("inf")
    heldout_state = {}
    cumulative_completed = 0
    cumulative_terminated = 0
    cumulative_truncated = 0

    def dr_progress(update):
        if not cfg.domain_randomization_enable:
            return 1.0
        if update <= cfg.dr_warmup_updates:
            return 0.0
        if cfg.dr_ramp_updates <= 0:
            return 1.0
        return float(np.clip(
            (update - cfg.dr_warmup_updates) / cfg.dr_ramp_updates, 0.0, 1.0
        ))

    def gait_period_for_update(update):
        if cfg.gait_period_start is None or cfg.gait_period_ramp_updates <= 0:
            return int(cfg.gait_period)
        if cfg.gait_period_ramp_updates == 1:
            return int(cfg.gait_period)
        curriculum_progress = np.clip(
            (update - 1) / (cfg.gait_period_ramp_updates - 1), 0.0, 1.0
        )
        period = (
            float(cfg.gait_period_start)
            + curriculum_progress
            * (float(cfg.gait_period) - float(cfg.gait_period_start))
        )
        return int(round(period))

    for update in range(1, cfg.total_updates + 1):
        progress = dr_progress(update)
        current_gait_period = gait_period_for_update(update)
        for worker in envs:
            worker.set_dr_progress(progress)
            worker.set_gait_period(current_gait_period)
        buffer = AsymmetricRolloutBuffer(
            cfg.rollout_steps,
            num_envs,
            env.actor_obs_dim,
            env.critic_obs_dim,
            env.act_dim,
            device,
        )
        reward_values = []
        root_heights = []
        startup_support_fractions = []
        startup_support_nominal_fractions = []
        startup_support_noises = []
        startup_support_command_scales = []
        startup_tether_displacements = []
        startup_tether_forces = []
        startup_tether_limited_count = 0
        startup_retreat_distances = []
        startup_retreat_terminations = 0
        pd_target_change_clip_fractions = []
        pd_tracking_error_clip_fractions = []
        startup_support_mode_counts = {
            "none": 0,
            "smooth": 0,
            "fast": 0,
            "residual": 0,
        }
        completed = False
        completed_returns = []
        completed_lengths = []
        terminated_count = 0
        truncated_count = 0

        for step in range(cfg.rollout_steps):
            actor_obs = numpy_to_torch_obs(actor_obs_np, device=device)
            mirrored_actor_obs = numpy_to_torch_obs(
                mirrored_actor_obs_np, device=device
            )
            critic_obs = numpy_to_torch_obs(critic_obs_np, device=device)
            with torch.no_grad():
                actions, logprobs, values = policy.act(
                    actor_obs, critic_obs, action_clip=cfg.action_clip
                )
                if cfg.normalize_value:
                    values = policy.value_norm.denormalize(values)
            # CensoredNormal already returns the exact bounded action whose
            # censored log probability is stored in the rollout buffer.
            action_np = actions.cpu().numpy()
            results = [worker.step(action_np[i]) for i, worker in enumerate(envs)]
            next_actor_obs = np.stack([result[0] for result in results])
            next_mirrored_actor_obs = np.stack(
                [result[4]["actor_obs_mirrored"] for result in results]
            )
            next_critic_obs = np.stack([result[4]["critic_obs"] for result in results])
            rewards_np = np.asarray([result[1] for result in results], dtype=np.float32)
            terminated_np = np.asarray([result[2] for result in results], dtype=np.float32)
            truncated_np = np.asarray([result[3] for result in results], dtype=np.float32)
            infos = [result[4] for result in results]
            next_values = critic_value(policy, next_critic_obs, device, cfg.normalize_value)

            buffer.add(
                actor_obs,
                mirrored_actor_obs,
                critic_obs,
                actions,
                logprobs,
                torch.as_tensor(rewards_np * cfg.reward_scale, device=device),
                torch.as_tensor(terminated_np, device=device),
                torch.as_tensor(truncated_np, device=device),
                values,
                next_values,
            )
            episode_returns += rewards_np
            episode_lengths += 1
            reward_values.extend(rewards_np.tolist())
            root_heights.extend(float(info.get("root_height", 0.0)) for info in infos)
            startup_support_fractions.extend(
                float(info.get("startup_support_fraction", 0.0)) for info in infos
            )
            startup_support_nominal_fractions.extend(
                float(info.get("startup_support_nominal_fraction", 0.0))
                for info in infos
            )
            startup_support_noises.extend(
                float(info.get("startup_support_noise", 0.0)) for info in infos
            )
            startup_support_command_scales.extend(
                float(info.get("startup_support_command_scale", 1.0))
                for info in infos
            )
            startup_tether_displacements.extend(
                float(info.get("startup_tether_displacement_norm", 0.0))
                for info in infos
            )
            startup_tether_forces.extend(
                float(info.get("startup_tether_force_norm", 0.0))
                for info in infos
            )
            startup_tether_limited_count += sum(
                int(bool(info.get("startup_tether_force_limited", False)))
                for info in infos
            )
            startup_retreat_distances.extend(
                float(info.get("startup_max_retreat_distance", 0.0))
                for info in infos
            )
            startup_retreat_terminations += sum(
                int(bool(info.get("startup_retreat_terminated", False)))
                for info in infos
            )
            pd_target_change_clip_fractions.extend(
                float(info.get("pd_target_change_clip_fraction", 0.0))
                for info in infos
            )
            pd_tracking_error_clip_fractions.extend(
                float(info.get("pd_tracking_error_clip_fraction", 0.0))
                for info in infos
            )
            for info in infos:
                mode = str(info.get("startup_support_mode", "none"))
                startup_support_mode_counts[mode] = (
                    startup_support_mode_counts.get(mode, 0) + 1
                )

            collection_actor_obs = next_actor_obs.copy()
            collection_mirrored_actor_obs = next_mirrored_actor_obs.copy()
            collection_critic_obs = next_critic_obs.copy()
            for i, worker in enumerate(envs):
                if not (terminated_np[i] or truncated_np[i]):
                    continue
                completed = True
                best_returns.append(float(episode_returns[i]))
                completed_returns.append(float(episode_returns[i]))
                completed_lengths.append(int(episode_lengths[i]))
                recent_episode_returns.append(float(episode_returns[i]))
                recent_episode_lengths.append(int(episode_lengths[i]))
                recent_episode_successes.append(float(bool(truncated_np[i])))
                terminated_count += int(bool(terminated_np[i]))
                truncated_count += int(bool(truncated_np[i]))
                print(
                    f"[update {update:04d}] env={i:02d} episode done | "
                    f"len={episode_lengths[i]:4d} return={episode_returns[i]:9.3f} "
                    f"root_h={infos[i].get('root_height', 0.0):.3f} "
                    f"support={infos[i].get('startup_support_mode', 'none')} "
                    f"retreat={infos[i].get('startup_max_retreat_distance', 0.0):.3f}m"
                )
                reset_obs, reset_info = worker.reset()
                collection_actor_obs[i] = reset_obs
                collection_mirrored_actor_obs[i] = reset_info["actor_obs_mirrored"]
                collection_critic_obs[i] = reset_info["critic_obs"]
                episode_returns[i] = 0.0
                episode_lengths[i] = 0
            actor_obs_np = collection_actor_obs
            mirrored_actor_obs_np = collection_mirrored_actor_obs
            critic_obs_np = collection_critic_obs

            if cfg.debug_rollout and update == 1 and step < 5:
                print(
                    f"[debug {step:03d}] reward={rewards_np[0]:8.4f} "
                    f"root_h={infos[0].get('root_height', 0.0):.3f} "
                    f"actor_obs={actor_obs_np.shape[-1]} critic_obs={critic_obs_np.shape[-1]}"
                )

        buffer.compute_returns(cfg.gamma, cfg.gae_lambda)
        if cfg.normalize_value:
            policy.value_norm.update(buffer.returns)
        buffer.advantages = (
            buffer.advantages - buffer.advantages.mean()
        ) / (buffer.advantages.std() + 1e-8)

        actor_enabled = update > max(int(cfg.critic_warmup_updates), 0)
        totals = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "ref": 0.0, "sym": 0.0,
                  "bound": 0.0,
                  "actor_grad": 0.0, "critic_grad": 0.0, "batches": 0}
        approx_kl = 0.0
        for _epoch in range(cfg.update_epochs):
            epoch_kl_sum = 0.0
            epoch_kl_samples = 0
            for batch in buffer.batches(cfg.minibatch_size):
                (b_actor, b_mirrored_actor, b_critic, b_actions, b_old_logprob,
                 b_adv, b_returns, b_values) = batch
                actor_n = policy.normalize_actor_obs(b_actor)
                dist = policy.dist(actor_n, action_clip=cfg.action_clip)
                new_logprob = dist.log_prob(b_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                logratio = new_logprob - b_old_logprob
                ratio = logratio.exp()
                with torch.no_grad():
                    batch_kl = ((ratio - 1.0) - logratio).mean()
                    batch_samples = int(logratio.numel())
                    epoch_kl_sum += float(batch_kl) * batch_samples
                    epoch_kl_samples += batch_samples
                pg = torch.max(
                    -b_adv * ratio,
                    -b_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef),
                ).mean()
                # Keep the legacy raw-mean regularizers. The environment-facing
                # mean is bounded, while bounds/reference/symmetry still shape
                # the unconstrained network head as before.
                mean = dist.raw_mean
                bounds = (
                    torch.clamp(mean - cfg.action_clip, min=0.0).square()
                    + torch.clamp(-cfg.action_clip - mean, min=0.0).square()
                ).sum(-1).mean()
                ref = torch.zeros((), device=device)
                if actor_enabled and reference_policy is not None:
                    with torch.no_grad():
                        ref_mean = reference_policy.actor(
                            reference_policy.normalize_actor_obs(b_actor)
                        )
                    ref = (mean - ref_mean).square().mean()
                mirrored_mean = policy.actor(
                    policy.normalize_actor_obs(b_mirrored_actor)
                )
                mirrored_mean = SRLMujocoV4Env.mirror_actions(mirrored_mean)
                symmetry = (mean - mirrored_mean).square().mean()
                if actor_enabled:
                    actor_loss = (
                        pg - cfg.ent_coef * entropy
                        + cfg.reference_policy_coef * ref
                        + cfg.actor_sym_loss_coef * symmetry
                        + cfg.bounds_loss_coef * bounds
                    )
                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_grad = torch.nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
                    actor_optimizer.step()
                    totals["actor_grad"] += float(actor_grad)

                predicted = policy.critic(policy.normalize_critic_obs(b_critic))
                if cfg.normalize_value:
                    targets = policy.value_norm.normalize(b_returns)
                    old_values = policy.value_norm.normalize(b_values)
                else:
                    targets, old_values = b_returns, b_values
                if cfg.clip_value:
                    clipped = old_values + torch.clamp(
                        predicted - old_values, -cfg.clip_coef, cfg.clip_coef
                    )
                    vf = 0.5 * torch.max(
                        (predicted - targets).square(), (clipped - targets).square()
                    ).mean()
                else:
                    vf = 0.5 * (predicted - targets).square().mean()
                critic_optimizer.zero_grad(set_to_none=True)
                (cfg.vf_coef * vf).backward()
                critic_grad = torch.nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
                critic_optimizer.step()

                totals["pg"] += float(pg)
                totals["vf"] += float(vf)
                totals["ent"] += float(entropy)
                totals["ref"] += float(ref)
                totals["sym"] += float(symmetry)
                totals["bound"] += float(bounds)
                totals["critic_grad"] += float(critic_grad)
                totals["batches"] += 1
            approx_kl = epoch_kl_sum / max(epoch_kl_samples, 1)
            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        count = max(totals["batches"], 1)
        print(
            f"[update {update:04d}] pg={totals['pg']/count:9.5f} "
            f"vf={totals['vf']/count:9.5f} entropy={totals['ent']/count:8.5f} "
            f"ref={totals['ref']/count:8.5f} sym={totals['sym']/count:8.5f} "
            f"actor_on={int(actor_enabled)} "
            f"grad=({totals['actor_grad']/count:.3f},{totals['critic_grad']/count:.3f}) "
            f"kl={approx_kl:.5f} dr={progress:.3f} "
            f"gait={current_gait_period:d} "
            f"reward={np.mean(reward_values):.3f} root_h={np.mean(root_heights):.3f} "
            f"support={np.mean(startup_support_fractions):.3f}/"
            f"{np.mean(startup_support_nominal_fractions):.3f} "
            f"tether={np.mean(startup_tether_forces):.1f}N "
            f"target_clip=({np.mean(pd_target_change_clip_fractions):.2%},"
            f"{np.mean(pd_tracking_error_clip_fractions):.2%})"
        )
        completed_count = len(completed_lengths)
        cumulative_completed += completed_count
        cumulative_terminated += terminated_count
        cumulative_truncated += truncated_count
        wandb_metrics = {
            "train/reward_mean": float(np.mean(reward_values)),
            "train/root_height_mean": float(np.mean(root_heights)),
            "train/dr_progress": progress,
            "train/gait_period": current_gait_period,
            "startup_support/fraction_mean": float(
                np.mean(startup_support_fractions)
                if startup_support_fractions else 0.0
            ),
            "startup_support/nominal_fraction_mean": float(
                np.mean(startup_support_nominal_fractions)
                if startup_support_nominal_fractions else 0.0
            ),
            "startup_support/noise_rms": float(
                np.sqrt(np.mean(np.square(startup_support_noises)))
                if startup_support_noises else 0.0
            ),
            "startup_support/command_scale_mean": float(
                np.mean(startup_support_command_scales)
                if startup_support_command_scales else 1.0
            ),
            "startup_tether/displacement_mean": float(
                np.mean(startup_tether_displacements)
                if startup_tether_displacements else 0.0
            ),
            "startup_tether/displacement_max": float(
                np.max(startup_tether_displacements)
                if startup_tether_displacements else 0.0
            ),
            "startup_tether/force_mean": float(
                np.mean(startup_tether_forces)
                if startup_tether_forces else 0.0
            ),
            "startup_tether/force_max": float(
                np.max(startup_tether_forces)
                if startup_tether_forces else 0.0
            ),
            "startup_tether/force_limit_fraction": float(
                startup_tether_limited_count / max(len(startup_tether_forces), 1)
            ),
            "startup_no_retreat/max_distance_mean": float(
                np.mean(startup_retreat_distances)
                if startup_retreat_distances else 0.0
            ),
            "startup_no_retreat/max_distance_max": float(
                np.max(startup_retreat_distances)
                if startup_retreat_distances else 0.0
            ),
            "startup_no_retreat/terminations": int(
                startup_retreat_terminations
            ),
            "control/pd_target_change_clip_fraction": float(
                np.mean(pd_target_change_clip_fractions)
            ),
            "control/pd_tracking_error_clip_fraction": float(
                np.mean(pd_tracking_error_clip_fractions)
            ),
            "episode/completed_count": completed_count,
            "episode/terminated_count": terminated_count,
            "episode/truncated_count": truncated_count,
            "episode/completed_total": cumulative_completed,
            "episode/terminated_total": cumulative_terminated,
            "episode/survived_total": cumulative_truncated,
            "episode/success_rate_total": (
                float(cumulative_truncated) / cumulative_completed
                if cumulative_completed else 0.0
            ),
            "episode/active_length_mean": float(np.mean(episode_lengths)),
            "episode/termination_rate_per_1000_steps": (
                1000.0 * terminated_count / (cfg.rollout_steps * num_envs)
            ),
            "ppo/policy_loss": totals["pg"] / count,
            "ppo/value_loss": totals["vf"] / count,
            "ppo/entropy": totals["ent"] / count,
            "ppo/reference_loss": totals["ref"] / count,
            "ppo/symmetry_loss": totals["sym"] / count,
            "ppo/bounds_loss": totals["bound"] / count,
            "ppo/actor_grad_norm": totals["actor_grad"] / count,
            "ppo/critic_grad_norm": totals["critic_grad"] / count,
            "ppo/approx_kl": approx_kl,
            "policy/log_std_mean": float(policy.log_std.detach().mean().cpu()),
        }
        support_samples = max(sum(startup_support_mode_counts.values()), 1)
        for mode, mode_count in startup_support_mode_counts.items():
            wandb_metrics[f"startup_support/{mode}_sample_fraction"] = (
                float(mode_count) / support_samples
            )
        if completed_count:
            wandb_metrics.update({
                "episode/length_mean": float(np.mean(completed_lengths)),
                "episode/return_mean": float(np.mean(completed_returns)),
                "episode/success_rate": float(truncated_count) / completed_count,
            })
        if recent_episode_returns:
            wandb_metrics.update({
                "episode/return_mean_100": float(np.mean(recent_episode_returns)),
                "episode/length_mean_100": float(np.mean(recent_episode_lengths)),
                "episode/success_rate_100": float(np.mean(recent_episode_successes)),
                "episode/failure_rate_100": 1.0 - float(np.mean(recent_episode_successes)),
                "episode/window_size": len(recent_episode_returns),
            })
        v2_train.base_train.log_wandb(
            wandb_run,
            wandb_metrics,
            step=update,
        )

        if update % cfg.save_every == 0 or update == cfg.total_updates:
            print("Saved checkpoint:", save_checkpoint(
                policy, actor_optimizer, critic_optimizer, cfg, update, save_dir
            ))
        if completed and len(best_returns) >= 3:
            mean_return = float(np.mean(best_returns))
            if mean_return > best_return:
                best_return = mean_return
                print("Saved best return checkpoint:", save_checkpoint(
                    policy, actor_optimizer, critic_optimizer, cfg, update, save_dir,
                    f"{run_name_prefix}_best.pt",
                ))
        if (
            cfg.eval_during_training
            and cfg.eval_every_updates > 0
            and (update % cfg.eval_every_updates == 0 or update == cfg.total_updates)
        ):
            heldout_eval_fn(
                policy,
                actor_optimizer,
                critic_optimizer,
                cfg,
                update,
                save_dir,
                heldout_state,
                wandb_run,
            )
            policy.train()

    if wandb_run is not None:
        wandb_run.finish()


def main():
    args, v4_args = parse_v4_args()
    if args.xml == "mjcf/srl_real_v1/srl_real_bot_v2_pos.xml":
        args.xml = V4_DEFAULT_XML
    base_cfg = v2_train.config_from_args(args)
    cfg = V4PPOConfig(
        **asdict(base_cfg),
        physics_hz=float(v4_args.physics_hz),
        gait_period=int(v4_args.gait_period),
        gait_period_start=(
            None
            if v4_args.gait_period_start is None
            else int(v4_args.gait_period_start)
        ),
        gait_period_ramp_updates=int(v4_args.gait_period_ramp_updates),
        dr_connector_stiffness_range=tuple(
            v4_args.dr_connector_stiffness_range
        ),
        dr_connector_damping_range=tuple(v4_args.dr_connector_damping_range),
    )
    if args.effort_limits is None:
        cfg.effort_limits = V4_EFFORT_LIMITS
    cfg.wandb_project = cfg.wandb_project.replace("v2", "v4")
    if cfg.wandb_run_name and cfg.wandb_run_name.startswith("mujoco_v2_"):
        cfg.wandb_run_name = cfg.wandb_run_name.replace("mujoco_v2_", "mujoco_v4_", 1)
    train(cfg)


if __name__ == "__main__":
    main()
