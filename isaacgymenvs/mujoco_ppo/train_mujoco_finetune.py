from __future__ import annotations

import argparse
import copy
import os
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch

from mujoco_ppo.models import ModelConfig, load_isaac_checkpoint, numpy_to_torch_obs
from mujoco_ppo.srl_mujoco_v1_env import EnvConfig, SRLMujocoWalkEnv


KP_PRESETS = {
    "env_default": (
        (114.0, 199.5, 266.0, 114.0, 199.5, 266.0),
        (22.0, 27.5, 44.0, 22.0, 27.5, 44.0),
    ),
    "run_v1_current": (
        (120.0, 210.0, 280.0, 120.0, 210.0, 280.0),
        (20.0, 25.0, 40.0, 20.0, 25.0, 40.0),
    ),
}


@dataclass
class PPOConfig:
    checkpoint_path: str = "checkpoints/SRL_Real_s5_v1-7.pth"
    xml_path: str = "mjcf/srl_real_v1/srl_real_bot_v1.xml"
    total_updates: int = 200
    rollout_steps: int = 1024
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    update_epochs: int = 10
    minibatch_size: int = 256
    target_kl: Optional[float] = 0.02
    seed: int = 1
    device: str = "cpu"
    save_dir: str = "mujoco_ppo/runs"
    save_every: int = 20
    action_clip: float = 1.0
    eval_only: bool = False
    eval_steps: int = 2000
    debug_rollout: bool = False
    init_log_std: Optional[float] = None
    resume_optimizer: bool = False
    base_wobble_penalty_scale: float = 8.0
    base_ang_acc_penalty_scale: float = 0.001
    yaw_drift_penalty_scale: float = 0.3
    foot_impact_penalty_scale: float = 0.0
    foot_force_threshold_bw: float = 1.8
    foot_force_penalty_power: float = 2.0
    torque_update_mode: str = "physics"
    kp_preset: str = "env_default"
    max_torque_step: float = 0.0
    wandb_enabled: bool = False
    wandb_project: str = "srl-mujoco-finetune"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"


class RolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        shape = (rollout_steps, num_envs)
        self.obs = torch.zeros((*shape, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((*shape, act_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros(shape, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(shape, dtype=torch.float32, device=device)
        self.terminated = torch.zeros(shape, dtype=torch.float32, device=device)
        self.truncated = torch.zeros(shape, dtype=torch.float32, device=device)
        self.values = torch.zeros(shape, dtype=torch.float32, device=device)
        self.next_values = torch.zeros(shape, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(shape, dtype=torch.float32, device=device)
        self.returns = torch.zeros(shape, dtype=torch.float32, device=device)
        self.sample_weights = torch.ones(shape, dtype=torch.float32, device=device)
        self.ptr = 0

    def add(self, obs, action, logprob, reward, terminated, truncated, value, next_value):
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.logprobs[idx] = logprob
        self.rewards[idx] = reward
        self.terminated[idx] = terminated
        self.truncated[idx] = truncated
        self.values[idx] = value
        self.next_values[idx] = next_value
        self.ptr += 1

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float):
        last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for t in reversed(range(self.rollout_steps)):
            bootstrap_mask = 1.0 - self.terminated[t]
            continuation_mask = 1.0 - torch.clamp(
                self.terminated[t] + self.truncated[t], 0.0, 1.0
            )
            delta = (
                self.rewards[t]
                + gamma * bootstrap_mask * self.next_values[t]
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * continuation_mask * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def multiply_env_weights(self, env_weights):
        env_weights = torch.as_tensor(
            env_weights, dtype=torch.float32, device=self.device
        ).reshape(self.num_envs)
        self.sample_weights *= env_weights.unsqueeze(0)

    def mark_segment_weight(self, env_id: int, start: int, end: int, weight: float):
        start = max(int(start), 0)
        end = min(int(end), self.rollout_steps)
        if end <= start:
            return
        current = self.sample_weights[start:end, int(env_id)]
        self.sample_weights[start:end, int(env_id)] = torch.maximum(
            current, torch.full_like(current, max(float(weight), 1.0))
        )

    def normalize_sample_weights(self):
        mean_weight = torch.clamp(self.sample_weights.mean(), min=1e-6)
        self.sample_weights /= mean_weight

    def batches(self, minibatch_size: int):
        batch_size = self.rollout_steps * self.num_envs
        obs = self.obs.reshape(batch_size, self.obs_dim)
        actions = self.actions.reshape(batch_size, self.act_dim)
        logprobs = self.logprobs.reshape(batch_size)
        advantages = self.advantages.reshape(batch_size)
        returns = self.returns.reshape(batch_size)
        values = self.values.reshape(batch_size)
        sample_weights = self.sample_weights.reshape(batch_size)
        indices = torch.randperm(batch_size, device=self.device)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = indices[start:end]
            yield (
                obs[mb_inds],
                actions[mb_inds],
                logprobs[mb_inds],
                advantages[mb_inds],
                returns[mb_inds],
                values[mb_inds],
                sample_weights[mb_inds],
            )


def make_env_and_model(cfg: PPOConfig):
    if cfg.kp_preset not in KP_PRESETS:
        raise ValueError(f"Unsupported kp_preset: {cfg.kp_preset}. Choices: {sorted(KP_PRESETS)}")
    kp, kd = KP_PRESETS[cfg.kp_preset]
    env_cfg = EnvConfig(
        xml_path=cfg.xml_path,
        base_wobble_penalty_scale=cfg.base_wobble_penalty_scale,
        base_ang_acc_penalty_scale=cfg.base_ang_acc_penalty_scale,
        yaw_drift_penalty_scale=cfg.yaw_drift_penalty_scale,
        foot_impact_penalty_scale=cfg.foot_impact_penalty_scale,
        foot_force_threshold_bw=cfg.foot_force_threshold_bw,
        foot_force_penalty_power=cfg.foot_force_penalty_power,
        torque_update_mode=cfg.torque_update_mode,
        kp=kp,
        kd=kd,
        max_torque_step=cfg.max_torque_step,
    )
    env = SRLMujocoWalkEnv(env_cfg)

    model_cfg = ModelConfig(obs_dim=env.obs_dim, act_dim=env.act_dim)
    policy, metadata = load_isaac_checkpoint(
        cfg.checkpoint_path,
        model_cfg=model_cfg,
        device=cfg.device,
        strict_critic=False,
    )
    return env, policy, metadata


def evaluate_value(policy, obs_np: np.ndarray, device: torch.device, normalize_value: bool = False):
    obs_t = numpy_to_torch_obs(obs_np, device=device)
    with torch.no_grad():
        obs_n = policy.normalize_obs(obs_t)
        value = policy.critic(obs_n)
        if normalize_value:
            value = policy.value_norm.denormalize(value)
    return value


def save_checkpoint(policy, optimizer, cfg: PPOConfig, update_idx: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"mujoco_finetune_update_{update_idx:05d}.pt")
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


def init_wandb(cfg: PPOConfig, run_name: str):
    if not cfg.wandb_enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is not installed. Install it or run without --wandb."
        ) from exc

    return wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name or run_name,
        mode=cfg.wandb_mode,
        config=asdict(cfg),
    )


def log_wandb(wandb_run, metrics, step=None):
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def rms(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def train(cfg: PPOConfig):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    env, policy, metadata = make_env_and_model(cfg)
    num_envs = max(int(getattr(cfg, "num_envs", 1)), 1)
    envs = [env]
    make_additional_env_fn = globals().get("make_additional_env")
    if num_envs > 1:
        if not callable(make_additional_env_fn):
            raise RuntimeError("num_envs > 1 requires a make_additional_env(cfg) callback.")
        envs.extend(make_additional_env_fn(cfg) for _ in range(num_envs - 1))

    if cfg.init_log_std is not None:
        policy.log_std.data.fill_(cfg.init_log_std)
        print("Set policy log_std to:", policy.log_std.data.cpu().numpy())
        print("Policy std:", torch.exp(policy.log_std).data.cpu().numpy())

    policy.train()
    actor_lr = float(getattr(cfg, "actor_learning_rate", cfg.learning_rate))
    critic_lr = float(getattr(cfg, "critic_learning_rate", cfg.learning_rate))
    actor_params = (
        list(policy.actor_mlp.parameters())
        + list(policy.mu.parameters())
        + [policy.log_std]
    )
    critic_params = list(policy.critic_mlp.parameters()) + list(policy.value.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": actor_params, "lr": actor_lr, "name": "actor"},
            {"params": critic_params, "lr": critic_lr, "name": "critic"},
        ]
    )
    reference_policy_coef = float(getattr(cfg, "reference_policy_coef", 0.0))
    reward_scale = float(getattr(cfg, "reward_scale", 1.0))
    normalize_value = bool(getattr(cfg, "normalize_value", False))
    clip_value = bool(getattr(cfg, "clip_value", True))
    bounds_loss_coef = float(getattr(cfg, "bounds_loss_coef", 0.0))
    reference_policy = None
    if reference_policy_coef > 0.0:
        reference_policy = copy.deepcopy(policy).eval()
        for parameter in reference_policy.parameters():
            parameter.requires_grad_(False)
    if cfg.resume_optimizer:
        checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device, weights_only=False)
        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for param_group in optimizer.param_groups:
                group_name = param_group.get("name")
                param_group["lr"] = actor_lr if group_name == "actor" else critic_lr
            print("Loaded optimizer state from checkpoint.")
        else:
            print("No optimizer state found in checkpoint; using a fresh optimizer.")

    dr_warmup = max(int(getattr(cfg, "dr_warmup_updates", 0)), 0)
    dr_ramp = max(int(getattr(cfg, "dr_ramp_updates", 0)), 0)

    def dr_progress_for_update(update_idx):
        if not bool(getattr(cfg, "domain_randomization_enable", False)):
            return 1.0
        if update_idx <= dr_warmup:
            return 0.0
        if dr_ramp <= 0:
            return 1.0
        return float(np.clip((update_idx - dr_warmup) / dr_ramp, 0.0, 1.0))

    initial_dr_progress = 1.0 if cfg.eval_only else dr_progress_for_update(0)
    for worker_env in envs:
        if hasattr(worker_env, "set_dr_progress"):
            worker_env.set_dr_progress(initial_dr_progress)
    reset_results = [
        worker_env.reset(seed=cfg.seed + worker_id * 100003)
        for worker_id, worker_env in enumerate(envs)
    ]
    obs_np = np.stack([result[0] for result in reset_results])
    episode_return = np.zeros(num_envs, dtype=np.float64)
    episode_length = np.zeros(num_envs, dtype=np.int64)

    print("Loaded Isaac checkpoint metadata:")
    print(metadata)

    run_name = cfg.wandb_run_name or (
        f"mujoco_eval_{time.strftime('%m%d_%H%M%S')}"
        if cfg.eval_only
        else f"mujoco_finetune_{time.strftime('%m%d_%H%M%S')}"
    )
    wandb_run = init_wandb(cfg, run_name)

    if cfg.eval_only:
        run_eval(env, policy, cfg, wandb_run=wandb_run)
        if wandb_run is not None:
            wandb_run.finish()
        return

    save_dir = os.path.join(cfg.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving finetune checkpoints to: {save_dir}")
    recent_episode_returns = deque(maxlen=5)
    best_mean_episode_return = -float("inf")
    best_min_episodes = 3
    best_filename = (
        "mujoco_v2_finetune_best.pt"
        if bool(metadata.get("v2_env", False))
        else "mujoco_finetune_best.pt"
    )
    periodic_eval_fn = globals().get("run_periodic_eval")
    if bool(getattr(cfg, "eval_during_training", False)) and callable(periodic_eval_fn):
        periodic_eval_fn(policy, optimizer, cfg, 0, save_dir, wandb_run)
        policy.train()

    for update in range(1, cfg.total_updates + 1):
        episode_completed_this_update = False
        current_dr_progress = dr_progress_for_update(update)
        for worker_env in envs:
            if hasattr(worker_env, "set_dr_progress"):
                worker_env.set_dr_progress(current_dr_progress)
        buffer = RolloutBuffer(
            cfg.rollout_steps, num_envs, env.obs_dim, env.act_dim, device
        )
        segment_start_steps = np.zeros(num_envs, dtype=np.int64)
        failure_segment_count = 0
        rollout_rewards = []
        rollout_root_h = []
        rollout_vel_x = []
        rollout_wx = []
        rollout_wy = []
        rollout_wz = []
        rollout_left_foot_bw = []
        rollout_right_foot_bw = []
        rollout_impact_penalty = []
        rollout_base_wobble_penalty = []
        rollout_yaw_drift_penalty = []
        clipped_action_count = 0
        sampled_action_count = 0
        clipped_action_count_per_joint = np.zeros(env.act_dim, dtype=np.int64)
        sampled_action_rows = 0
        rollout_scenario_steps = {}

        for step_idx in range(cfg.rollout_steps):
            obs_t = numpy_to_torch_obs(obs_np, device=device)
            with torch.no_grad():
                action_t, logprob_t, value_t = policy.act(obs_t)
                if normalize_value:
                    value_t = policy.value_norm.denormalize(value_t)

            raw_action_t = action_t
            raw_action_np = raw_action_t.cpu().numpy()
            action_np = np.clip(raw_action_np, -cfg.action_clip, cfg.action_clip)
            clipped_action_count += int(
                np.count_nonzero(np.abs(raw_action_np) > cfg.action_clip)
            )
            sampled_action_count += int(raw_action_np.size)
            clipped_action_count_per_joint += np.count_nonzero(
                np.abs(raw_action_np) > cfg.action_clip, axis=0
            )
            sampled_action_rows += int(raw_action_np.shape[0])

            step_results = [
                worker_env.step(action_np[env_id])
                for env_id, worker_env in enumerate(envs)
            ]
            transition_next_obs = np.stack([result[0] for result in step_results])
            rewards_np = np.asarray([result[1] for result in step_results], dtype=np.float32)
            terminated_np = np.asarray([result[2] for result in step_results], dtype=np.float32)
            truncated_np = np.asarray([result[3] for result in step_results], dtype=np.float32)
            if (
                bool(getattr(cfg, "reset_envs_each_rollout", False))
                and step_idx == cfg.rollout_steps - 1
            ):
                # Cut every unfinished segment at the rollout boundary. GAE
                # still bootstraps a truncation, but no episode can leak into
                # the next buffer where its earlier transitions are unavailable.
                truncated_np = np.maximum(truncated_np, 1.0 - terminated_np)
            infos = [result[4] for result in step_results]
            for info in infos:
                scenario = str(info.get("dr_scenario", "normal"))
                rollout_scenario_steps[scenario] = (
                    rollout_scenario_steps.get(scenario, 0) + 1
                )
            next_value_t = evaluate_value(
                policy, transition_next_obs, device, normalize_value=normalize_value
            )

            buffer.add(
                obs=obs_t,
                action=raw_action_t,
                logprob=logprob_t,
                reward=torch.as_tensor(
                    rewards_np * reward_scale, dtype=torch.float32, device=device
                ),
                terminated=torch.as_tensor(terminated_np, dtype=torch.float32, device=device),
                truncated=torch.as_tensor(truncated_np, dtype=torch.float32, device=device),
                value=value_t,
                next_value=next_value_t,
            )

            episode_return += rewards_np
            episode_length += 1
            next_collection_obs = transition_next_obs.copy()
            rollout_rewards.extend(float(value) for value in rewards_np)
            rollout_root_h.extend(float(info.get("root_height", 0.0)) for info in infos)
            rollout_vel_x.extend(float(info.get("vel_x", 0.0)) for info in infos)
            rollout_wx.extend(float(info.get("wx", 0.0)) for info in infos)
            rollout_wy.extend(float(info.get("wy", 0.0)) for info in infos)
            rollout_wz.extend(float(info.get("wz", 0.0)) for info in infos)
            rollout_left_foot_bw.extend(float(info.get("left_foot_force_bw", 0.0)) for info in infos)
            rollout_right_foot_bw.extend(float(info.get("right_foot_force_bw", 0.0)) for info in infos)
            rollout_impact_penalty.extend(float(info.get("penalty_foot_impact", 0.0)) for info in infos)
            rollout_base_wobble_penalty.extend(float(info.get("penalty_base_wobble", 0.0)) for info in infos)
            rollout_yaw_drift_penalty.extend(float(info.get("penalty_yaw_drift", 0.0)) for info in infos)

            if cfg.debug_rollout and update == 1 and step_idx < 5:
                info = infos[0]
                print(
                    f"[debug step {step_idx:03d}] "
                    f"reward={rewards_np[0]:8.4f} "
                    f"root_h={info.get('root_height', 0.0):.3f} "
                    f"vel_track={info.get('reward_vel_tracking', 0.0):.4f} "
                    f"ori={info.get('reward_orientation', 0.0):.4f} "
                    f"height={info.get('reward_pelvis_height', 0.0):.4f} "
                    f"no_fly={info.get('penalty_no_fly', 0.0):.4f} "
                    f"clear={info.get('penalty_clearance', 0.0):.4f} "
                    f"lat={info.get('penalty_lateral', 0.0):.4f} "
                    f"foot_bw=({info.get('left_foot_force_bw', 0.0):.2f},"
                    f"{info.get('right_foot_force_bw', 0.0):.2f}) "
                    f"impact={info.get('penalty_foot_impact', 0.0):.4f}"
                )
                print(
                    "  sampled action:",
                    np.array2string(raw_action_np[0], precision=3, suppress_small=True),
                    "min={:.3f} max={:.3f} mean={:.3f} std={:.3f}".format(
                    float(np.min(raw_action_np[0])),
                    float(np.max(raw_action_np[0])),
                    float(np.mean(raw_action_np[0])),
                    float(np.std(raw_action_np[0])),
                    ),
                )

            for env_id, worker_env in enumerate(envs):
                done = bool(terminated_np[env_id] or truncated_np[env_id])
                if not done:
                    continue
                if bool(terminated_np[env_id]):
                    buffer.mark_segment_weight(
                        env_id,
                        segment_start_steps[env_id],
                        step_idx + 1,
                        getattr(cfg, "failure_weight", 1.0),
                    )
                    failure_segment_count += 1
                segment_start_steps[env_id] = step_idx + 1
                recent_episode_returns.append(float(episode_return[env_id]))
                episode_completed_this_update = True
                info = infos[env_id]
                print(
                    f"[update {update:04d}] env={env_id:02d} episode done | "
                    f"len={episode_length[env_id]:4d} "
                    f"return={episode_return[env_id]:9.3f} "
                    f"root_h={info.get('root_height', 0.0):.3f}"
                )
                log_wandb(
                    wandb_run,
                    {
                        "episode/return": float(episode_return[env_id]),
                        "episode/length": int(episode_length[env_id]),
                        "episode/root_h_done": float(info.get("root_height", 0.0)),
                    },
                    step=update,
                )
                reset_obs, _ = worker_env.reset()
                next_collection_obs[env_id] = reset_obs
                episode_return[env_id] = 0.0
                episode_length[env_id] = 0
            obs_np = next_collection_obs

        buffer.compute_returns_and_advantages(cfg.gamma, cfg.gae_lambda)
        cvar_fraction = float(np.clip(getattr(cfg, "cvar_fraction", 0.0), 0.0, 1.0))
        cvar_weight = max(float(getattr(cfg, "cvar_weight", 1.0)), 1.0)
        rollout_return_per_env = buffer.rewards.sum(dim=0)
        failed_env_mask = buffer.terminated.bool().any(dim=0)
        env_sample_weights = torch.ones(num_envs, dtype=torch.float32, device=device)
        tail_env_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        if cvar_fraction > 0.0:
            tail_count = max(1, int(np.ceil(cvar_fraction * num_envs)))
            # A termination must rank ahead of a merely low-return rollout.
            ranking_score = rollout_return_per_env.detach().clone()
            if torch.any(failed_env_mask):
                score_span = torch.max(torch.abs(ranking_score)).clamp(min=1.0)
                ranking_score[failed_env_mask] -= 10.0 * score_span
            tail_indices = torch.argsort(ranking_score)[:tail_count]
            tail_env_mask[tail_indices] = True
            env_sample_weights[tail_env_mask] = cvar_weight
            buffer.multiply_env_weights(env_sample_weights)
        buffer.normalize_sample_weights()
        if normalize_value:
            policy.value_norm.update(buffer.returns)
        advantages = buffer.advantages
        buffer.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        approx_kl = None
        pg_loss_epoch = 0.0
        vf_loss_epoch = 0.0
        entropy_epoch = 0.0
        reference_loss_epoch = 0.0
        bounds_loss_epoch = 0.0
        actor_grad_norm_epoch = 0.0
        critic_grad_norm_epoch = 0.0
        num_batches = 0
        critic_warmup_updates = max(int(getattr(cfg, "critic_warmup_updates", 0)), 0)
        actor_enabled = update > critic_warmup_updates

        for _ in range(cfg.update_epochs):
            for batch in buffer.batches(cfg.minibatch_size):
                (
                    b_obs,
                    b_actions,
                    b_logprobs,
                    b_advantages,
                    b_returns,
                    b_values,
                    b_sample_weights,
                ) = batch

                actor_obs_n = policy.normalize_obs(b_obs)
                action_dist = policy.dist(actor_obs_n)
                new_logprob = action_dist.log_prob(b_actions).sum(dim=-1)
                entropy = action_dist.entropy().sum(dim=-1)
                logratio = new_logprob - b_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()

                pg_loss_1 = -b_advantages * ratio
                pg_loss_2 = -b_advantages * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss_samples = torch.max(pg_loss_1, pg_loss_2)
                pg_loss = torch.sum(pg_loss_samples * b_sample_weights) / torch.clamp(
                    b_sample_weights.sum(), min=1e-6
                )

                entropy_loss = entropy.mean()
                current_mean = action_dist.mean
                bounds_loss = (
                    torch.clamp(current_mean - cfg.action_clip, min=0.0).square()
                    + torch.clamp(-cfg.action_clip - current_mean, min=0.0).square()
                ).sum(dim=-1).mean()
                reference_loss = torch.zeros((), dtype=torch.float32, device=device)
                if actor_enabled and reference_policy is not None:
                    with torch.no_grad():
                        reference_obs_n = reference_policy.normalize_obs(b_obs)
                        reference_mean = reference_policy.actor(reference_obs_n)
                    reference_loss = torch.mean((current_mean - reference_mean) ** 2)

                if actor_enabled:
                    actor_loss = (
                        pg_loss
                        - cfg.ent_coef * entropy_loss
                        + reference_policy_coef * reference_loss
                        + bounds_loss_coef * bounds_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                        actor_params, cfg.max_grad_norm
                    )
                    optimizer.step()
                    actor_grad_norm_epoch += float(actor_grad_norm)

                critic_obs_n = policy.normalize_obs(b_obs)
                new_value = policy.critic(critic_obs_n)
                if normalize_value:
                    value_targets = policy.value_norm.normalize(b_returns)
                    old_value_targets = policy.value_norm.normalize(b_values)
                else:
                    value_targets = b_returns
                    old_value_targets = b_values
                if clip_value:
                    value_pred_clipped = old_value_targets + torch.clamp(
                        new_value - old_value_targets, -cfg.clip_coef, cfg.clip_coef
                    )
                    value_loss_unclipped = (new_value - value_targets) ** 2
                    value_loss_clipped = (value_pred_clipped - value_targets) ** 2
                    value_loss_samples = torch.max(
                        value_loss_unclipped, value_loss_clipped
                    )
                else:
                    value_loss_samples = (new_value - value_targets) ** 2
                value_loss = 0.5 * torch.sum(
                    value_loss_samples * b_sample_weights
                ) / torch.clamp(b_sample_weights.sum(), min=1e-6)

                optimizer.zero_grad(set_to_none=True)
                (cfg.vf_coef * value_loss).backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    critic_params, cfg.max_grad_norm
                )
                optimizer.step()
                critic_grad_norm_epoch += float(critic_grad_norm)

                pg_loss_epoch += pg_loss.item()
                vf_loss_epoch += value_loss.item()
                entropy_epoch += entropy_loss.item()
                reference_loss_epoch += reference_loss.item()
                bounds_loss_epoch += bounds_loss.item()
                num_batches += 1

            if cfg.target_kl is not None and approx_kl is not None and approx_kl > cfg.target_kl:
                break

        if num_batches > 0:
            pg_loss_epoch /= num_batches
            vf_loss_epoch /= num_batches
            entropy_epoch /= num_batches
            reference_loss_epoch /= num_batches
            bounds_loss_epoch /= num_batches
            actor_grad_norm_epoch /= num_batches
            critic_grad_norm_epoch /= num_batches

        action_clip_fraction = (
            clipped_action_count / sampled_action_count if sampled_action_count > 0 else 0.0
        )
        action_clip_fraction_per_joint = (
            clipped_action_count_per_joint / sampled_action_rows
            if sampled_action_rows > 0
            else np.zeros(env.act_dim, dtype=np.float64)
        )
        action_clip_joint_text = ",".join(
            f"{100.0 * value:.2f}" for value in action_clip_fraction_per_joint
        )
        scenario_step_total = max(sum(rollout_scenario_steps.values()), 1)
        scenario_fraction = {
            name: count / scenario_step_total
            for name, count in sorted(rollout_scenario_steps.items())
        }
        scenario_text = ",".join(
            f"{name}:{100.0 * fraction:.1f}%"
            for name, fraction in scenario_fraction.items()
        )

        print(
            f"[update {update:04d}] "
            f"pg_loss={pg_loss_epoch:9.5f} "
            f"vf_loss={vf_loss_epoch:9.5f} "
            f"entropy={entropy_epoch:8.5f} "
            f"ref_loss={reference_loss_epoch:8.5f} "
            f"bound_loss={bounds_loss_epoch:8.5f} "
            f"actor_on={int(actor_enabled)} "
            f"grad=({actor_grad_norm_epoch:.3f},{critic_grad_norm_epoch:.3f}) "
            f"action_clip={100.0 * action_clip_fraction:.2f}% "
            f"action_clip_joints=({action_clip_joint_text})% "
            f"cvar_tail={int(tail_env_mask.sum().item())}/{num_envs} "
            f"failure_segments={failure_segment_count} "
            f"weighted_steps={100.0 * float((buffer.sample_weights > 1.0).float().mean()):.1f}% "
            f"risk_w=[{float(buffer.sample_weights.min()):.2f},"
            f"{float(buffer.sample_weights.max()):.2f}] "
            f"scenarios=({scenario_text}) "
            f"approx_kl={(approx_kl if approx_kl is not None else 0.0):8.5f}"
        )
        foot_arr = np.column_stack([rollout_left_foot_bw, rollout_right_foot_bw])
        foot_exceed_ratio = float(
            np.mean(np.any(foot_arr > cfg.foot_force_threshold_bw, axis=1))
        ) if foot_arr.size else 0.0
        log_wandb(
            wandb_run,
            {
                "loss/pg_loss": pg_loss_epoch,
                "loss/vf_loss": vf_loss_epoch,
                "loss/entropy": entropy_epoch,
                "loss/reference_policy": reference_loss_epoch,
                "loss/bounds_loss": bounds_loss_epoch,
                "train/actor_enabled": float(actor_enabled),
                "train/actor_grad_norm": actor_grad_norm_epoch,
                "train/critic_grad_norm": critic_grad_norm_epoch,
                "train/action_clip_fraction": action_clip_fraction,
                "train/cvar_tail_envs": int(tail_env_mask.sum().item()),
                "train/failed_envs": int(failed_env_mask.sum().item()),
                "train/failure_segments": int(failure_segment_count),
                "train/weighted_step_fraction": float(
                    (buffer.sample_weights > 1.0).float().mean()
                ),
                "train/risk_weight_max": float(buffer.sample_weights.max()),
                "train/rollout_return_env_min": float(rollout_return_per_env.min()),
                "train/rollout_return_env_mean": float(rollout_return_per_env.mean()),
                **{
                    f"train/action_clip_fraction_joint_{joint_idx}": float(value)
                    for joint_idx, value in enumerate(action_clip_fraction_per_joint)
                },
                **{
                    f"train/dr_scenario_fraction_{name}": float(fraction)
                    for name, fraction in scenario_fraction.items()
                },
                "train/value_norm_mean": float(policy.value_norm.running_mean),
                "train/value_norm_std": float(torch.sqrt(policy.value_norm.running_var)),
                "train/dr_progress": current_dr_progress,
                "loss/approx_kl": approx_kl if approx_kl is not None else 0.0,
                "train/reward_mean": float(np.mean(rollout_rewards)) if rollout_rewards else 0.0,
                "train/reward_sum": float(np.sum(rollout_rewards)) if rollout_rewards else 0.0,
                "train/root_h_mean": float(np.mean(rollout_root_h)) if rollout_root_h else 0.0,
                "train/vel_x_mean": float(np.mean(rollout_vel_x)) if rollout_vel_x else 0.0,
                "train/wx_rms": rms(rollout_wx),
                "train/wy_rms": rms(rollout_wy),
                "train/wz_rms": rms(rollout_wz),
                "train/foot_bw_left_mean": float(np.mean(rollout_left_foot_bw)) if rollout_left_foot_bw else 0.0,
                "train/foot_bw_right_mean": float(np.mean(rollout_right_foot_bw)) if rollout_right_foot_bw else 0.0,
                "train/foot_bw_left_max": float(np.max(rollout_left_foot_bw)) if rollout_left_foot_bw else 0.0,
                "train/foot_bw_right_max": float(np.max(rollout_right_foot_bw)) if rollout_right_foot_bw else 0.0,
                "train/foot_exceed_ratio": foot_exceed_ratio,
                "train/impact_penalty_mean": float(np.mean(rollout_impact_penalty)) if rollout_impact_penalty else 0.0,
                "train/impact_penalty_max": float(np.max(rollout_impact_penalty)) if rollout_impact_penalty else 0.0,
                "train/base_wobble_penalty_mean": float(np.mean(rollout_base_wobble_penalty)) if rollout_base_wobble_penalty else 0.0,
                "train/yaw_drift_penalty_mean": float(np.mean(rollout_yaw_drift_penalty)) if rollout_yaw_drift_penalty else 0.0,
            },
            step=update,
        )

        if update % cfg.save_every == 0 or update == cfg.total_updates:
            save_path = save_checkpoint(policy, optimizer, cfg, update, save_dir)
            print(f"Saved checkpoint: {save_path}")

        if episode_completed_this_update and len(recent_episode_returns) >= best_min_episodes:
            mean_episode_return = float(np.mean(recent_episode_returns))
            if mean_episode_return > best_mean_episode_return:
                best_mean_episode_return = mean_episode_return
                best_path = os.path.join(save_dir, best_filename)
                torch.save(
                    {
                        "update": update,
                        "best_mean_episode_return": best_mean_episode_return,
                        "best_return_window": len(recent_episode_returns),
                        "model_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": asdict(cfg),
                    },
                    best_path,
                )
                print(
                    f"Saved best checkpoint: {best_path} "
                    f"mean_return={best_mean_episode_return:.3f} "
                    f"window={len(recent_episode_returns)}"
                )
                log_wandb(
                    wandb_run,
                    {
                        "best/mean_episode_return": best_mean_episode_return,
                        "best/update": update,
                    },
                    step=update,
                )

        eval_every = max(int(getattr(cfg, "eval_every_updates", 0)), 0)
        if (
            bool(getattr(cfg, "eval_during_training", False))
            and callable(periodic_eval_fn)
            and eval_every > 0
            and (update % eval_every == 0 or update == cfg.total_updates)
        ):
            periodic_eval_fn(policy, optimizer, cfg, update, save_dir, wandb_run)
            policy.train()

    if wandb_run is not None:
        wandb_run.finish()


def run_eval(env: SRLMujocoWalkEnv, policy, cfg: PPOConfig, wandb_run=None):
    policy.eval()
    obs_np, _ = env.reset(seed=cfg.seed)
    episode_return = 0.0
    episode_length = 0
    eval_print_window = 100
    foot_bw_window = []
    impact_window = []
    foot_bw_all = []
    impact_all = []
    threshold_bw = float(env.cfg.foot_force_threshold_bw)

    print(f"Running eval-only for {cfg.eval_steps} steps...")
    for step_idx in range(cfg.eval_steps):
        obs_t = numpy_to_torch_obs(obs_np, device=cfg.device)
        with torch.no_grad():
            action_t = policy.act_deterministic(obs_t)

        action_np = action_t.squeeze(0).cpu().numpy()
        action_np = np.clip(action_np, -cfg.action_clip, cfg.action_clip)
        obs_np, reward, terminated, truncated, info = env.step(action_np)

        episode_return += reward
        episode_length += 1
        left_foot_bw = float(info.get("left_foot_force_bw", 0.0))
        right_foot_bw = float(info.get("right_foot_force_bw", 0.0))
        foot_bw_pair = (left_foot_bw, right_foot_bw)
        impact_penalty = float(info.get("penalty_foot_impact", 0.0))
        foot_bw_window.append(foot_bw_pair)
        impact_window.append(impact_penalty)
        foot_bw_all.append(foot_bw_pair)
        impact_all.append(impact_penalty)

        if (step_idx + 1) % eval_print_window == 0:
            window_arr = np.asarray(foot_bw_window, dtype=np.float32)
            impact_arr = np.asarray(impact_window, dtype=np.float32)
            left_mean, right_mean = np.mean(window_arr, axis=0)
            left_max, right_max = np.max(window_arr, axis=0)
            exceed_ratio = float(np.mean(np.any(window_arr > threshold_bw, axis=1)))
            print(
                f"[eval {step_idx + 1:05d}] "
                f"window={eval_print_window} "
                f"reward_last={reward:8.4f} "
                f"root_h={info.get('root_height', 0.0):.3f} "
                f"vel_track={info.get('reward_vel_tracking', 0.0):.4f} "
                f"ori={info.get('reward_orientation', 0.0):.4f} "
                f"height={info.get('reward_pelvis_height', 0.0):.4f} "
                f"foot_bw_mean=({left_mean:.2f},{right_mean:.2f}) "
                f"foot_bw_max=({left_max:.2f},{right_max:.2f}) "
                f"exceed>{threshold_bw:.2f}={100.0 * exceed_ratio:.1f}% "
                f"impact_mean={float(np.mean(impact_arr)):.4f} "
                f"impact_max={float(np.max(impact_arr)):.4f}"
            )
            log_wandb(
                wandb_run,
                {
                    "eval_window/reward_last": float(reward),
                    "eval_window/root_h": float(info.get("root_height", 0.0)),
                    "eval_window/vel_track": float(info.get("reward_vel_tracking", 0.0)),
                    "eval_window/orientation": float(info.get("reward_orientation", 0.0)),
                    "eval_window/height_reward": float(info.get("reward_pelvis_height", 0.0)),
                    "eval_window/foot_bw_left_mean": float(left_mean),
                    "eval_window/foot_bw_right_mean": float(right_mean),
                    "eval_window/foot_bw_left_max": float(left_max),
                    "eval_window/foot_bw_right_max": float(right_max),
                    "eval_window/foot_exceed_ratio": exceed_ratio,
                    "eval_window/impact_penalty_mean": float(np.mean(impact_arr)),
                    "eval_window/impact_penalty_max": float(np.max(impact_arr)),
                },
                step=step_idx + 1,
            )
            foot_bw_window.clear()
            impact_window.clear()

        if terminated or truncated:
            print(
                f"[eval done] len={episode_length:4d} "
                f"return={episode_return:9.3f} "
                f"root_h={info.get('root_height', 0.0):.3f}"
            )
            obs_np, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

    if foot_bw_all:
        all_arr = np.asarray(foot_bw_all, dtype=np.float32)
        impact_arr = np.asarray(impact_all, dtype=np.float32)
        left_mean, right_mean = np.mean(all_arr, axis=0)
        left_max, right_max = np.max(all_arr, axis=0)
        exceed_ratio = float(np.mean(np.any(all_arr > threshold_bw, axis=1)))
        print(
            f"[eval summary] steps={len(foot_bw_all)} "
            f"foot_bw_mean=({left_mean:.2f},{right_mean:.2f}) "
            f"foot_bw_max=({left_max:.2f},{right_max:.2f}) "
            f"exceed>{threshold_bw:.2f}={100.0 * exceed_ratio:.1f}% "
            f"impact_mean={float(np.mean(impact_arr)):.4f} "
            f"impact_max={float(np.max(impact_arr)):.4f}"
        )
        log_wandb(
            wandb_run,
            {
                "eval/steps": len(foot_bw_all),
                "eval/foot_bw_left_mean": float(left_mean),
                "eval/foot_bw_right_mean": float(right_mean),
                "eval/foot_bw_left_max": float(left_max),
                "eval/foot_bw_right_max": float(right_max),
                "eval/foot_exceed_ratio": exceed_ratio,
                "eval/impact_penalty_mean": float(np.mean(impact_arr)),
                "eval/impact_penalty_max": float(np.max(impact_arr)),
            },
            step=len(foot_bw_all),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal PPO finetuning on MuJoCo using Isaac checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/SRL_Real_s5_v1-7.pth")
    parser.add_argument("--xml", type=str, default="mjcf/srl_real_v1/srl_real_bot_v1.xml")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--init-log-std", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--debug-rollout", action="store_true")
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--base-wobble-penalty-scale", type=float, default=8.0)
    parser.add_argument("--base-ang-acc-penalty-scale", type=float, default=0.001)
    parser.add_argument("--yaw-drift-penalty-scale", type=float, default=0.3)
    parser.add_argument("--foot-impact-penalty-scale", type=float, default=0.0)
    parser.add_argument("--foot-force-threshold-bw", type=float, default=1.8)
    parser.add_argument("--foot-force-penalty-power", type=float, default=2.0)
    parser.add_argument("--torque-update-mode", type=str, default="physics", choices=["physics", "control"])
    parser.add_argument("--kp-preset", type=str, default="env_default", choices=sorted(KP_PRESETS.keys()))
    parser.add_argument("--max-torque-step", type=float, default=0.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="srl-mujoco-finetune")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PPOConfig(
        checkpoint_path=args.checkpoint,
        xml_path=args.xml,
        total_updates=args.updates,
        rollout_steps=args.rollout_steps,
        learning_rate=args.lr,
        target_kl=args.target_kl,
        device=args.device,
        seed=args.seed,
        save_every=args.save_every,
        eval_only=args.eval_only,
        eval_steps=args.eval_steps,
        debug_rollout=args.debug_rollout,
        init_log_std=args.init_log_std,
        resume_optimizer=args.resume_optimizer,
        base_wobble_penalty_scale=args.base_wobble_penalty_scale,
        base_ang_acc_penalty_scale=args.base_ang_acc_penalty_scale,
        yaw_drift_penalty_scale=args.yaw_drift_penalty_scale,
        foot_impact_penalty_scale=args.foot_impact_penalty_scale,
        foot_force_threshold_bw=args.foot_force_threshold_bw,
        foot_force_penalty_power=args.foot_force_penalty_power,
        torque_update_mode=args.torque_update_mode,
        kp_preset=args.kp_preset,
        max_torque_step=args.max_torque_step,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )
    train(cfg)
