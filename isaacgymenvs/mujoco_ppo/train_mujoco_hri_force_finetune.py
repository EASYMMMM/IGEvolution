from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mujoco_ppo.models import ActorCritic, ModelConfig, load_isaac_checkpoint, numpy_to_torch_obs
from mujoco_ppo.srl_mujoco_hri_force_env import EnvConfig, SRLMujocoHRIForceEnv


@dataclass
class PPOConfig:
    checkpoint_path: str = "mujoco_ppo/runs/foot_impact_update40_lr1e-5/mujoco_finetune_update_00004.pt"
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
    base_wobble_penalty_scale: float = 2.0
    base_ang_acc_penalty_scale: float = 0.001
    yaw_drift_penalty_scale: float = 0.3
    foot_impact_penalty_scale: float = 0.0
    foot_force_threshold_bw: float = 1.8
    foot_force_penalty_power: float = 2.0
    hri_wrench_penalty_scale: float = 0.0
    hri_wrench_mode: str = "external_sine"
    human_track_pos_penalty_scale: float = 3.0
    human_track_vel_penalty_scale: float = 0.4
    wandb_enabled: bool = False
    wandb_project: str = "srl-mujoco-hri-force-finetune"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"


class RolloutBuffer:
    def __init__(self, rollout_steps: int, obs_dim: int, act_dim: int, device: torch.device):
        self.rollout_steps = rollout_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.obs = torch.zeros((rollout_steps, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((rollout_steps, act_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.dones = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.values = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.returns = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, value):
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.logprobs[idx] = logprob
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor, last_done: torch.Tensor, gamma: float, gae_lambda: float):
        last_gae = 0.0
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def batches(self, minibatch_size: int):
        batch_size = self.rollout_steps
        indices = torch.randperm(batch_size, device=self.device)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = indices[start:end]
            yield (
                self.obs[mb_inds],
                self.actions[mb_inds],
                self.logprobs[mb_inds],
                self.advantages[mb_inds],
                self.returns[mb_inds],
                self.values[mb_inds],
            )


def _infer_checkpoint_obs_dim(checkpoint_path: str, device: str):
    def _numel(value):
        return int(value.numel()) if hasattr(value, "numel") else int(np.asarray(value).size)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if state is None:
        model_state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        for key in ("running_mean_std.running_mean", "obs_norm.running_mean"):
            if key in model_state:
                return _numel(model_state[key]), checkpoint
        for key, value in model_state.items():
            if "actor_mlp" in key and str(key).endswith("weight") and getattr(value, "ndim", 0) == 2:
                return int(value.shape[1]), checkpoint
        return None, checkpoint

    for key in ("obs_norm.running_mean", "running_mean_std.running_mean"):
        if key in state:
            return _numel(state[key]), checkpoint
    first_weight = state.get("actor_mlp.0.weight")
    if first_weight is not None:
        return int(first_weight.shape[1]), checkpoint
    return None, checkpoint


def _copy_expanded_linear(dst: nn.Linear, src: nn.Linear):
    with torch.no_grad():
        dst.weight.zero_()
        dst.bias.copy_(src.bias)
        rows = min(dst.weight.shape[0], src.weight.shape[0])
        cols = min(dst.weight.shape[1], src.weight.shape[1])
        dst.weight[:rows, :cols].copy_(src.weight[:rows, :cols])


def _copy_same_shape_linears(dst_modules, src_modules):
    for dst, src in zip(dst_modules, src_modules):
        if isinstance(dst, nn.Linear) and isinstance(src, nn.Linear):
            if dst.weight.shape == src.weight.shape:
                dst.weight.data.copy_(src.weight.data)
                dst.bias.data.copy_(src.bias.data)


def load_hri_force_checkpoint(checkpoint_path: str, model_cfg: ModelConfig, device: str):
    src_obs_dim, checkpoint = _infer_checkpoint_obs_dim(checkpoint_path, device)
    if src_obs_dim is None or src_obs_dim == model_cfg.obs_dim:
        policy, metadata = load_isaac_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            device=device,
            strict_critic=False,
        )
        metadata["warm_start_expanded_obs"] = False
        return policy, metadata

    src_cfg = ModelConfig(
        obs_dim=src_obs_dim,
        act_dim=model_cfg.act_dim,
        hidden_sizes=model_cfg.hidden_sizes,
        activation=model_cfg.activation,
        init_log_std=model_cfg.init_log_std,
    )
    src_policy, metadata = load_isaac_checkpoint(
        checkpoint_path,
        model_cfg=src_cfg,
        device=device,
        strict_critic=False,
    )
    dst_policy = ActorCritic(model_cfg).to(device)

    src_actor_layers = [m for m in src_policy.actor_mlp if isinstance(m, nn.Linear)]
    dst_actor_layers = [m for m in dst_policy.actor_mlp if isinstance(m, nn.Linear)]
    src_critic_layers = [m for m in src_policy.critic_mlp if isinstance(m, nn.Linear)]
    dst_critic_layers = [m for m in dst_policy.critic_mlp if isinstance(m, nn.Linear)]

    _copy_expanded_linear(dst_actor_layers[0], src_actor_layers[0])
    _copy_expanded_linear(dst_critic_layers[0], src_critic_layers[0])
    _copy_same_shape_linears(dst_actor_layers[1:], src_actor_layers[1:])
    _copy_same_shape_linears(dst_critic_layers[1:], src_critic_layers[1:])

    dst_policy.mu.weight.data.copy_(src_policy.mu.weight.data)
    dst_policy.mu.bias.data.copy_(src_policy.mu.bias.data)
    dst_policy.value.weight.data.copy_(src_policy.value.weight.data)
    dst_policy.value.bias.data.copy_(src_policy.value.bias.data)
    dst_policy.log_std.data.copy_(src_policy.log_std.data)

    with torch.no_grad():
        dst_policy.obs_norm.running_mean.zero_()
        dst_policy.obs_norm.running_var.fill_(1.0)
        copy_dim = min(src_obs_dim, model_cfg.obs_dim)
        dst_policy.obs_norm.running_mean[:copy_dim].copy_(
            src_policy.obs_norm.running_mean[:copy_dim]
        )
        dst_policy.obs_norm.running_var[:copy_dim].copy_(
            src_policy.obs_norm.running_var[:copy_dim]
        )

    metadata = dict(metadata)
    metadata.update(
        {
            "warm_start_expanded_obs": True,
            "source_obs_dim": src_obs_dim,
            "target_obs_dim": model_cfg.obs_dim,
            "new_obs_init": "first-layer weights zero, obs_norm mean=0 var=1",
        }
    )
    return dst_policy, metadata


def make_env_and_model(cfg: PPOConfig):
    env_cfg = EnvConfig(
        xml_path=cfg.xml_path,
        base_wobble_penalty_scale=cfg.base_wobble_penalty_scale,
        base_ang_acc_penalty_scale=cfg.base_ang_acc_penalty_scale,
        yaw_drift_penalty_scale=cfg.yaw_drift_penalty_scale,
        foot_impact_penalty_scale=cfg.foot_impact_penalty_scale,
        foot_force_threshold_bw=cfg.foot_force_threshold_bw,
        foot_force_penalty_power=cfg.foot_force_penalty_power,
        hri_wrench_penalty_scale=cfg.hri_wrench_penalty_scale,
        hri_wrench_mode=cfg.hri_wrench_mode,
        human_track_pos_penalty_scale=cfg.human_track_pos_penalty_scale,
        human_track_vel_penalty_scale=cfg.human_track_vel_penalty_scale,
    )
    env = SRLMujocoHRIForceEnv(env_cfg)

    model_cfg = ModelConfig(obs_dim=env.obs_dim, act_dim=env.act_dim)
    policy, metadata = load_hri_force_checkpoint(
        cfg.checkpoint_path,
        model_cfg=model_cfg,
        device=cfg.device,
    )
    return env, policy, metadata


def evaluate_value(policy, obs_np: np.ndarray, device: torch.device):
    obs_t = numpy_to_torch_obs(obs_np, device=device)
    with torch.no_grad():
        obs_n = policy.normalize_obs(obs_t)
        value = policy.critic(obs_n)
    return value.squeeze(0)


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

    if cfg.init_log_std is not None:
        policy.log_std.data.fill_(cfg.init_log_std)
        print("Set policy log_std to:", policy.log_std.data.cpu().numpy())
        print("Policy std:", torch.exp(policy.log_std).data.cpu().numpy())

    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    if cfg.resume_optimizer:
        checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device, weights_only=False)
        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                for param_group in optimizer.param_groups:
                    param_group["lr"] = cfg.learning_rate
                print("Loaded optimizer state from checkpoint.")
            except Exception as exc:
                print("Could not resume optimizer state; using fresh optimizer.")
                print(f"Optimizer resume error: {exc}")
        else:
            print("No optimizer state found in checkpoint; using a fresh optimizer.")

    obs_np, _ = env.reset(seed=cfg.seed)
    episode_return = 0.0
    episode_length = 0

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

    for update in range(1, cfg.total_updates + 1):
        buffer = RolloutBuffer(cfg.rollout_steps, env.obs_dim, env.act_dim, device)
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
        rollout_hri_penalty = []
        rollout_hri_force_norm = []
        rollout_hri_shear = []
        rollout_hri_torque_norm = []
        rollout_hri_fx = []
        rollout_hri_fz = []
        rollout_hri_ty = []
        rollout_human_track_pos = []
        rollout_human_track_vel = []

        for step_idx in range(cfg.rollout_steps):
            obs_t = numpy_to_torch_obs(obs_np, device=device)
            with torch.no_grad():
                action_t, logprob_t, value_t = policy.act(obs_t)

            raw_action_t = action_t.squeeze(0)
            raw_action_np = raw_action_t.cpu().numpy()
            action_np = np.clip(raw_action_np, -cfg.action_clip, cfg.action_clip)

            next_obs_np, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            buffer.add(
                obs=obs_t.squeeze(0),
                action=raw_action_t,
                logprob=logprob_t.squeeze(0),
                reward=torch.tensor(reward, dtype=torch.float32, device=device),
                done=torch.tensor(float(done), dtype=torch.float32, device=device),
                value=value_t.squeeze(0),
            )

            episode_return += reward
            episode_length += 1
            obs_np = next_obs_np
            rollout_rewards.append(float(reward))
            rollout_root_h.append(float(info.get("root_height", 0.0)))
            rollout_vel_x.append(float(info.get("vel_x", 0.0)))
            rollout_wx.append(float(info.get("wx", 0.0)))
            rollout_wy.append(float(info.get("wy", 0.0)))
            rollout_wz.append(float(info.get("wz", 0.0)))
            rollout_left_foot_bw.append(float(info.get("left_foot_force_bw", 0.0)))
            rollout_right_foot_bw.append(float(info.get("right_foot_force_bw", 0.0)))
            rollout_impact_penalty.append(float(info.get("penalty_foot_impact", 0.0)))
            rollout_base_wobble_penalty.append(float(info.get("penalty_base_wobble", 0.0)))
            rollout_yaw_drift_penalty.append(float(info.get("penalty_yaw_drift", 0.0)))
            rollout_hri_penalty.append(float(info.get("penalty_hri_wrench", 0.0)))
            rollout_hri_force_norm.append(float(info.get("hri_force_norm", 0.0)))
            rollout_hri_shear.append(float(info.get("hri_shear", 0.0)))
            rollout_hri_torque_norm.append(float(info.get("hri_torque_norm", 0.0)))
            rollout_hri_fx.append(float(info.get("hri_fx", 0.0)))
            rollout_hri_fz.append(float(info.get("hri_fz", 0.0)))
            rollout_hri_ty.append(float(info.get("hri_ty", 0.0)))
            rollout_human_track_pos.append(float(info.get("human_track_pos_error", 0.0)))
            rollout_human_track_vel.append(float(info.get("human_track_vel_error", 0.0)))

            if cfg.debug_rollout and update == 1 and step_idx < 5:
                print(
                    f"[debug step {step_idx:03d}] "
                    f"reward={reward:8.4f} "
                    f"root_h={info.get('root_height', 0.0):.3f} "
                    f"vel_track={info.get('reward_vel_tracking', 0.0):.4f} "
                    f"ori={info.get('reward_orientation', 0.0):.4f} "
                    f"height={info.get('reward_pelvis_height', 0.0):.4f} "
                    f"no_fly={info.get('penalty_no_fly', 0.0):.4f} "
                    f"clear={info.get('penalty_clearance', 0.0):.4f} "
                    f"lat={info.get('penalty_lateral', 0.0):.4f} "
                    f"foot_bw=({info.get('left_foot_force_bw', 0.0):.2f},"
                    f"{info.get('right_foot_force_bw', 0.0):.2f}) "
                    f"impact={info.get('penalty_foot_impact', 0.0):.4f} "
                    f"hri=({info.get('hri_fx', 0.0):.1f},"
                    f"{info.get('hri_fz', 0.0):.1f},"
                    f"{info.get('hri_ty', 0.0):.1f}) "
                    f"human_err=({info.get('human_track_pos_error', 0.0):.3f}m,"
                    f"{info.get('human_track_vel_error', 0.0):.3f}m/s)"
                )
                print(
                    "  sampled action:",
                    np.array2string(raw_action_np, precision=3, suppress_small=True),
                    "min={:.3f} max={:.3f} mean={:.3f} std={:.3f}".format(
                    float(np.min(raw_action_np)),
                    float(np.max(raw_action_np)),
                    float(np.mean(raw_action_np)),
                    float(np.std(raw_action_np)),
                    ),
                )


            if done:
                print(
                    f"[update {update:04d}] episode done | len={episode_length:4d} "
                    f"return={episode_return:9.3f} root_h={info.get('root_height', 0.0):.3f}"
                )
                log_wandb(
                    wandb_run,
                    {
                        "episode/return": episode_return,
                        "episode/length": episode_length,
                        "episode/root_h_done": float(info.get("root_height", 0.0)),
                    },
                    step=update,
                )
                obs_np, _ = env.reset()
                episode_return = 0.0
                episode_length = 0

        with torch.no_grad():
            last_value = evaluate_value(policy, obs_np, device)
            last_done = torch.tensor(0.0, dtype=torch.float32, device=device)

        buffer.compute_returns_and_advantages(last_value, last_done, cfg.gamma, cfg.gae_lambda)
        advantages = buffer.advantages
        buffer.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        approx_kl = None
        pg_loss_epoch = 0.0
        vf_loss_epoch = 0.0
        entropy_epoch = 0.0
        num_batches = 0

        for _ in range(cfg.update_epochs):
            for batch in buffer.batches(cfg.minibatch_size):
                b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values = batch

                new_logprob, entropy, new_value = policy.evaluate_actions(b_obs, b_actions)
                logratio = new_logprob - b_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()

                pg_loss_1 = -b_advantages * ratio
                pg_loss_2 = -b_advantages * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                value_pred_clipped = b_values + torch.clamp(new_value - b_values, -cfg.clip_coef, cfg.clip_coef)
                value_loss_unclipped = (new_value - b_returns) ** 2
                value_loss_clipped = (value_pred_clipped - b_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()

                pg_loss_epoch += pg_loss.item()
                vf_loss_epoch += value_loss.item()
                entropy_epoch += entropy_loss.item()
                num_batches += 1

            if cfg.target_kl is not None and approx_kl is not None and approx_kl > cfg.target_kl:
                break

        if num_batches > 0:
            pg_loss_epoch /= num_batches
            vf_loss_epoch /= num_batches
            entropy_epoch /= num_batches

        print(
            f"[update {update:04d}] "
            f"pg_loss={pg_loss_epoch:9.5f} "
            f"vf_loss={vf_loss_epoch:9.5f} "
            f"entropy={entropy_epoch:8.5f} "
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
                "train/hri_penalty_mean": float(np.mean(rollout_hri_penalty)) if rollout_hri_penalty else 0.0,
                "train/hri_force_norm_mean": float(np.mean(rollout_hri_force_norm)) if rollout_hri_force_norm else 0.0,
                "train/hri_force_norm_max": float(np.max(rollout_hri_force_norm)) if rollout_hri_force_norm else 0.0,
                "train/hri_shear_mean": float(np.mean(rollout_hri_shear)) if rollout_hri_shear else 0.0,
                "train/hri_torque_norm_mean": float(np.mean(rollout_hri_torque_norm)) if rollout_hri_torque_norm else 0.0,
                "train/hri_fx_rms": rms(rollout_hri_fx),
                "train/hri_fz_mean": float(np.mean(rollout_hri_fz)) if rollout_hri_fz else 0.0,
                "train/hri_ty_rms": rms(rollout_hri_ty),
                "train/human_track_pos_error_mean": float(np.mean(rollout_human_track_pos)) if rollout_human_track_pos else 0.0,
                "train/human_track_vel_error_mean": float(np.mean(rollout_human_track_vel)) if rollout_human_track_vel else 0.0,
            },
            step=update,
        )

        if update % cfg.save_every == 0 or update == cfg.total_updates:
            save_path = save_checkpoint(policy, optimizer, cfg, update, save_dir)
            print(f"Saved checkpoint: {save_path}")

    if wandb_run is not None:
        wandb_run.finish()


def run_eval(env: SRLMujocoHRIForceEnv, policy, cfg: PPOConfig, wandb_run=None):
    policy.eval()
    obs_np, _ = env.reset(seed=cfg.seed)
    episode_return = 0.0
    episode_length = 0
    eval_print_window = 100
    foot_bw_window = []
    impact_window = []
    hri_force_window = []
    foot_bw_all = []
    impact_all = []
    hri_force_all = []
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
        hri_force_norm = float(info.get("hri_force_norm", 0.0))
        foot_bw_window.append(foot_bw_pair)
        impact_window.append(impact_penalty)
        hri_force_window.append(hri_force_norm)
        foot_bw_all.append(foot_bw_pair)
        impact_all.append(impact_penalty)
        hri_force_all.append(hri_force_norm)

        if (step_idx + 1) % eval_print_window == 0:
            window_arr = np.asarray(foot_bw_window, dtype=np.float32)
            impact_arr = np.asarray(impact_window, dtype=np.float32)
            hri_force_arr = np.asarray(hri_force_window, dtype=np.float32)
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
                f"impact_max={float(np.max(impact_arr)):.4f} "
                f"hri_force_mean={float(np.mean(hri_force_arr)):.1f}N"
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
                    "eval_window/hri_force_norm_mean": float(np.mean(hri_force_arr)),
                    "eval_window/hri_force_norm_max": float(np.max(hri_force_arr)),
                },
                step=step_idx + 1,
            )
            foot_bw_window.clear()
            impact_window.clear()
            hri_force_window.clear()

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
        hri_force_arr = np.asarray(hri_force_all, dtype=np.float32)
        left_mean, right_mean = np.mean(all_arr, axis=0)
        left_max, right_max = np.max(all_arr, axis=0)
        exceed_ratio = float(np.mean(np.any(all_arr > threshold_bw, axis=1)))
        print(
            f"[eval summary] steps={len(foot_bw_all)} "
            f"foot_bw_mean=({left_mean:.2f},{right_mean:.2f}) "
            f"foot_bw_max=({left_max:.2f},{right_max:.2f}) "
            f"exceed>{threshold_bw:.2f}={100.0 * exceed_ratio:.1f}% "
            f"impact_mean={float(np.mean(impact_arr)):.4f} "
            f"impact_max={float(np.max(impact_arr)):.4f} "
            f"hri_force_mean={float(np.mean(hri_force_arr)):.1f}N "
            f"hri_force_max={float(np.max(hri_force_arr)):.1f}N"
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
                "eval/hri_force_norm_mean": float(np.mean(hri_force_arr)),
                "eval/hri_force_norm_max": float(np.max(hri_force_arr)),
            },
            step=len(foot_bw_all),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="PPO finetuning on MuJoCo with virtual HRI force observation.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="mujoco_ppo/runs/foot_impact_update40_lr1e-5/mujoco_finetune_update_00004.pt",
    )
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
    parser.add_argument("--base-wobble-penalty-scale", type=float, default=2.0)
    parser.add_argument("--base-ang-acc-penalty-scale", type=float, default=0.001)
    parser.add_argument("--yaw-drift-penalty-scale", type=float, default=0.3)
    parser.add_argument("--foot-impact-penalty-scale", type=float, default=0.0)
    parser.add_argument("--foot-force-threshold-bw", type=float, default=1.8)
    parser.add_argument("--foot-force-penalty-power", type=float, default=2.0)
    parser.add_argument("--hri-wrench-penalty-scale", type=float, default=0.0)
    parser.add_argument("--human-track-pos-penalty-scale", type=float, default=3.0)
    parser.add_argument("--human-track-vel-penalty-scale", type=float, default=0.4)
    parser.add_argument(
        "--hri-wrench-mode",
        type=str,
        default="external_sine",
        choices=["external_sine", "proxy_accel", "spring_damper_ref", "human_traj_track"],
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="srl-mujoco-hri-force-finetune")
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
        hri_wrench_penalty_scale=args.hri_wrench_penalty_scale,
        hri_wrench_mode=args.hri_wrench_mode,
        human_track_pos_penalty_scale=args.human_track_pos_penalty_scale,
        human_track_vel_penalty_scale=args.human_track_vel_penalty_scale,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )
    train(cfg)
