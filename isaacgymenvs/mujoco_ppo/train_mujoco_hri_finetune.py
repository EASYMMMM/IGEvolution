from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mujoco_ppo.models import (
    ActorCritic,
    ModelConfig,
    _collect_mlp_layers,
    _copy_linear,
    _unwrap_tensor,
    numpy_to_torch_obs,
)
from mujoco_ppo.srl_mujoco_hri import EnvConfig, SRLMujocoHRIEnv


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class PPOConfig:
    checkpoint_path: str
    checkpoint_key: str = "model_srl"
    xml_path: str = "mjcf/srl_real_v1/srl_real_bot_v1.xml"
    isaac_dataset_dir: Optional[str] = None
    isaac_replay_seq_len: int = 1000
    total_updates: int = 200
    rollout_steps: int = 1024
    learning_rate: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    bc_coef: float = 0.05
    bc_warmup_updates: int = 0
    max_grad_norm: float = 1.0
    update_epochs: int = 5
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
    torque_update_mode: str = "physics"
    srl_action_filter: bool = False
    srl_action_filter_cutoff_hz: float = 4.0
    apply_isaac_load_cell_wrench: bool = True
    use_isaac_human_obs: bool = True
    isaac_load_cell_force_scale: float = 1.0
    isaac_load_cell_torque_scale: float = 1.0
    isaac_load_cell_force_clip_x: float = 100.0
    isaac_load_cell_force_clip_y: float = 50.0
    isaac_load_cell_force_clip_z: float = 100.0
    hri_wrench_ramp_time: float = 0.0
    hri_command_ramp_time: float = 0.0
    base_wobble_penalty_scale: float = 2.0
    base_ang_acc_penalty_scale: float = 0.001
    yaw_drift_penalty_scale: float = 0.3
    foot_impact_penalty_scale: float = 0.0
    foot_force_threshold_bw: float = 1.8
    foot_force_penalty_power: float = 2.0
    hri_wrench_penalty_scale: float = 0.0
    wandb_enabled: bool = False
    wandb_project: str = "srl-mujoco-hri-finetune"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"


class RolloutBuffer:
    def __init__(self, rollout_steps: int, obs_dim: int, act_dim: int, device: torch.device):
        self.rollout_steps = rollout_steps
        self.obs = torch.zeros((rollout_steps, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((rollout_steps, act_dim), dtype=torch.float32, device=device)
        self.teacher_actions = torch.zeros((rollout_steps, act_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.dones = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.values = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.returns = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.device = device
        self.ptr = 0

    def add(self, obs, action, teacher_action, logprob, reward, done, value):
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.teacher_actions[idx] = teacher_action
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
        indices = torch.randperm(self.rollout_steps, device=self.device)
        for start in range(0, self.rollout_steps, minibatch_size):
            mb_inds = indices[start:start + minibatch_size]
            yield (
                self.obs[mb_inds],
                self.actions[mb_inds],
                self.teacher_actions[mb_inds],
                self.logprobs[mb_inds],
                self.advantages[mb_inds],
                self.returns[mb_inds],
                self.values[mb_inds],
            )


def _numel(value):
    return int(value.numel()) if hasattr(value, "numel") else int(np.asarray(value).size)


def _selected_model_state(checkpoint, checkpoint_key: str):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], "mujoco_finetune"
    if isinstance(checkpoint, dict) and checkpoint_key in checkpoint:
        return checkpoint[checkpoint_key], checkpoint_key
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"], "model"
    return checkpoint, "raw"


def _infer_checkpoint_obs_dim(checkpoint_path: str, device: str, checkpoint_key: str):
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    model_state, selected_key = _selected_model_state(checkpoint, checkpoint_key)
    for key, value in model_state.items():
        if "actor_mlp" in key and str(key).endswith("weight") and getattr(value, "ndim", 0) == 2:
            return int(value.shape[1]), checkpoint, selected_key
    for key in ("obs_norm.running_mean", "running_mean_std.running_mean"):
        if key in model_state:
            return _numel(model_state[key]), checkpoint, selected_key
    return None, checkpoint, selected_key


def _infer_checkpoint_hidden_sizes(checkpoint_path: str, device: str, checkpoint_key: str):
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    model_state, _selected_key = _selected_model_state(checkpoint, checkpoint_key)
    actor_layers = _collect_mlp_layers(model_state, "actor_mlp")
    hidden_sizes = []
    for weight, _bias in actor_layers:
        if getattr(weight, "ndim", 0) == 2:
            hidden_sizes.append(int(weight.shape[0]))
    return tuple(hidden_sizes) if hidden_sizes else None


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


def _load_selected_checkpoint(checkpoint_path: str, model_cfg: ModelConfig, device: str, checkpoint_key: str, strict_critic: bool = False):
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    model_state, selected_key = _selected_model_state(checkpoint, checkpoint_key)

    if selected_key == "mujoco_finetune":
        policy = ActorCritic(model_cfg).to(device)
        policy.load_state_dict(model_state, strict=True)
        return policy, {
            "checkpoint_type": "mujoco_finetune",
            "selected_key": selected_key,
            "checkpoint_keys": sorted(list(checkpoint.keys())) if isinstance(checkpoint, dict) else [],
            "critic_loaded": True,
            "obs_norm_loaded": True,
            "update": checkpoint.get("update") if isinstance(checkpoint, dict) else None,
            "optimizer_state_available": isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint,
        }

    policy = ActorCritic(model_cfg).to(device)

    actor_layers = _collect_mlp_layers(model_state, "actor_mlp")
    actor_linears = [m for m in policy.actor_mlp if isinstance(m, nn.Linear)]
    if len(actor_layers) != len(actor_linears):
        raise RuntimeError(f"Actor layer count mismatch: checkpoint={len(actor_layers)}, model={len(actor_linears)}")
    for linear, (weight, bias) in zip(actor_linears, actor_layers):
        _copy_linear(linear, weight, bias)

    mu_weight = model_state.get("a2c_network.mu.weight")
    mu_bias = model_state.get("a2c_network.mu.bias")
    if mu_weight is None:
        raise KeyError("Missing actor output layer: a2c_network.mu.weight")
    _copy_linear(policy.mu, mu_weight, mu_bias)

    critic_loaded = False
    critic_layers = _collect_mlp_layers(model_state, "critic_mlp")
    critic_linears = [m for m in policy.critic_mlp if isinstance(m, nn.Linear)]
    if critic_layers and len(critic_layers) == len(critic_linears):
        critic_shapes_match = True
        for linear, (weight, bias) in zip(critic_linears, critic_layers):
            weight_tensor = _unwrap_tensor(weight)
            bias_tensor = _unwrap_tensor(bias) if bias is not None else None
            if linear.weight.shape != weight_tensor.shape:
                critic_shapes_match = False
                break
            if bias_tensor is not None and linear.bias.shape != bias_tensor.shape:
                critic_shapes_match = False
                break

        if critic_shapes_match:
            for linear, (weight, bias) in zip(critic_linears, critic_layers):
                _copy_linear(linear, weight, bias)
            value_weight = model_state.get("a2c_network.value.weight")
            value_bias = model_state.get("a2c_network.value.bias")
            if value_weight is None:
                value_weight = model_state.get("value.weight")
                value_bias = model_state.get("value.bias")
            if value_weight is not None:
                _copy_linear(policy.value, value_weight, value_bias)
                critic_loaded = True
        elif strict_critic:
            src_shapes = [tuple(_unwrap_tensor(weight).shape) for weight, _bias in critic_layers]
            dst_shapes = [tuple(linear.weight.shape) for linear in critic_linears]
            raise RuntimeError(f"Critic shape mismatch: checkpoint={src_shapes}, model={dst_shapes}")
    elif strict_critic:
        raise RuntimeError("Could not find critic_mlp weights in checkpoint.")

    mean_key = "running_mean_std.running_mean"
    var_key = "running_mean_std.running_var"
    if mean_key in model_state and var_key in model_state:
        policy.obs_norm.running_mean.copy_(_unwrap_tensor(model_state[mean_key]).to(policy.obs_norm.running_mean))
        policy.obs_norm.running_var.copy_(_unwrap_tensor(model_state[var_key]).to(policy.obs_norm.running_var))

    for key in ("a2c_network.sigma", "a2c_network.log_std", "sigma", "log_std"):
        if key in model_state:
            sigma_tensor = _unwrap_tensor(model_state[key]).to(device=policy.log_std.device, dtype=policy.log_std.dtype)
            if sigma_tensor.shape == policy.log_std.shape:
                policy.log_std.data.copy_(sigma_tensor)
                break

    return policy, {
        "checkpoint_type": "isaacgym",
        "selected_key": selected_key,
        "checkpoint_keys": sorted(list(checkpoint.keys())) if isinstance(checkpoint, dict) else [],
        "critic_loaded": critic_loaded,
        "obs_norm_loaded": mean_key in model_state and var_key in model_state,
    }


def load_hri_checkpoint(checkpoint_path: str, model_cfg: ModelConfig, device: str, checkpoint_key: str):
    src_obs_dim, _checkpoint, selected_key = _infer_checkpoint_obs_dim(checkpoint_path, device, checkpoint_key)
    if src_obs_dim is None or src_obs_dim == model_cfg.obs_dim:
        policy, metadata = _load_selected_checkpoint(checkpoint_path, model_cfg, device, checkpoint_key)
        metadata["warm_start_expanded_obs"] = False
        metadata["source_obs_dim"] = src_obs_dim
        metadata["target_obs_dim"] = model_cfg.obs_dim
        return policy, metadata

    src_cfg = ModelConfig(
        obs_dim=src_obs_dim,
        act_dim=model_cfg.act_dim,
        hidden_sizes=model_cfg.hidden_sizes,
        activation=model_cfg.activation,
        init_log_std=model_cfg.init_log_std,
    )
    src_policy, metadata = _load_selected_checkpoint(checkpoint_path, src_cfg, device, checkpoint_key)
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
        dst_policy.obs_norm.running_mean[:copy_dim].copy_(src_policy.obs_norm.running_mean[:copy_dim])
        dst_policy.obs_norm.running_var[:copy_dim].copy_(src_policy.obs_norm.running_var[:copy_dim])

    metadata = dict(metadata)
    metadata.update(
        {
            "warm_start_expanded_obs": True,
            "selected_key": selected_key,
            "source_obs_dim": src_obs_dim,
            "target_obs_dim": model_cfg.obs_dim,
            "new_obs_init": "first-layer weights zero, obs_norm mean=0 var=1",
        }
    )
    return dst_policy, metadata


def make_env_and_model(cfg: PPOConfig):
    env_cfg = EnvConfig(
        xml_path=cfg.xml_path,
        isaac_dataset_dir=cfg.isaac_dataset_dir,
        isaac_replay_seq_len=cfg.isaac_replay_seq_len,
        torque_update_mode=cfg.torque_update_mode,
        srl_action_filter=cfg.srl_action_filter,
        srl_action_filter_cutoff_hz=cfg.srl_action_filter_cutoff_hz,
        apply_isaac_load_cell_wrench=cfg.apply_isaac_load_cell_wrench,
        use_isaac_human_obs=cfg.use_isaac_human_obs,
        isaac_load_cell_force_scale=cfg.isaac_load_cell_force_scale,
        isaac_load_cell_torque_scale=cfg.isaac_load_cell_torque_scale,
        isaac_load_cell_force_clip_x=cfg.isaac_load_cell_force_clip_x,
        isaac_load_cell_force_clip_y=cfg.isaac_load_cell_force_clip_y,
        isaac_load_cell_force_clip_z=cfg.isaac_load_cell_force_clip_z,
        hri_wrench_ramp_time=cfg.hri_wrench_ramp_time,
        hri_command_ramp_time=cfg.hri_command_ramp_time,
        base_wobble_penalty_scale=cfg.base_wobble_penalty_scale,
        base_ang_acc_penalty_scale=cfg.base_ang_acc_penalty_scale,
        yaw_drift_penalty_scale=cfg.yaw_drift_penalty_scale,
        foot_impact_penalty_scale=cfg.foot_impact_penalty_scale,
        foot_force_threshold_bw=cfg.foot_force_threshold_bw,
        foot_force_penalty_power=cfg.foot_force_penalty_power,
        hri_wrench_penalty_scale=cfg.hri_wrench_penalty_scale,
    )
    env = SRLMujocoHRIEnv(env_cfg)
    hidden_sizes = _infer_checkpoint_hidden_sizes(
        cfg.checkpoint_path,
        cfg.device,
        cfg.checkpoint_key,
    )
    if hidden_sizes is None:
        hidden_sizes = ModelConfig().hidden_sizes
    model_cfg = ModelConfig(obs_dim=env.obs_dim, act_dim=env.act_dim, hidden_sizes=hidden_sizes)
    policy, metadata = load_hri_checkpoint(cfg.checkpoint_path, model_cfg, cfg.device, cfg.checkpoint_key)
    metadata["model_hidden_sizes"] = hidden_sizes
    return env, policy, metadata


def evaluate_value(policy, obs_np: np.ndarray, device: torch.device):
    obs_t = numpy_to_torch_obs(obs_np, device=device)
    with torch.no_grad():
        obs_n = policy.normalize_obs(obs_t)
        value = policy.critic(obs_n)
    return value.squeeze(0)


def save_checkpoint(policy, optimizer, cfg: PPOConfig, update_idx: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"mujoco_hri_finetune_update_{update_idx:05d}.pt")
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
    import wandb

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
    if cfg.bc_coef > 0.0 and not cfg.isaac_dataset_dir:
        raise ValueError("bc_coef > 0 requires --dataset so teacher_action is available.")

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
        checkpoint = safe_torch_load(cfg.checkpoint_path, map_location=cfg.device)
        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                for param_group in optimizer.param_groups:
                    param_group["lr"] = cfg.learning_rate
                print("Loaded optimizer state from checkpoint.")
            except Exception as exc:
                print("Could not resume optimizer state; using fresh optimizer.")
                print(f"Optimizer resume error: {exc}")

    obs_np, _ = env.reset(seed=cfg.seed)
    episode_return = 0.0
    episode_length = 0

    print("Loaded checkpoint metadata:")
    print(metadata)
    print(f"env obs_dim={env.obs_dim} act_dim={env.act_dim}")
    print(f"torque_update_mode={env.cfg.torque_update_mode}")
    print(f"srl_action_filter={env.cfg.srl_action_filter} cutoff_hz={env.cfg.srl_action_filter_cutoff_hz}")
    if cfg.isaac_dataset_dir:
        print(f"Using Isaac HRI dataset: {cfg.isaac_dataset_dir}")

    run_name = cfg.wandb_run_name or (
        f"mujoco_hri_eval_{time.strftime('%m%d_%H%M%S')}"
        if cfg.eval_only
        else f"mujoco_hri_finetune_{time.strftime('%m%d_%H%M%S')}"
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
        rollout_hri_fx = []
        rollout_hri_fz = []
        rollout_hri_ty = []
        rollout_teacher_mse = []
        rollout_teacher_action_norm = []

        for step_idx in range(cfg.rollout_steps):
            obs_t = numpy_to_torch_obs(obs_np, device=device)
            with torch.no_grad():
                action_t, logprob_t, value_t = policy.act(obs_t)

            raw_action_t = action_t.squeeze(0)
            raw_action_np = raw_action_t.cpu().numpy()
            action_np = np.clip(raw_action_np, -cfg.action_clip, cfg.action_clip)

            next_obs_np, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            teacher_action_np = np.asarray(info.get("teacher_action", np.zeros(env.act_dim)), dtype=np.float32)
            teacher_action_t = torch.from_numpy(teacher_action_np).to(device=device)

            buffer.add(
                obs=obs_t.squeeze(0),
                action=raw_action_t,
                teacher_action=teacher_action_t,
                logprob=logprob_t.squeeze(0),
                reward=torch.tensor(reward, dtype=torch.float32, device=device),
                done=torch.tensor(float(done), dtype=torch.float32, device=device),
                value=value_t.squeeze(0),
            )

            teacher_mse = float(np.mean((raw_action_np - teacher_action_np) ** 2))
            rollout_teacher_mse.append(teacher_mse)
            rollout_teacher_action_norm.append(float(np.linalg.norm(teacher_action_np)))

            episode_return += reward
            episode_length += 1
            obs_np = next_obs_np
            rollout_rewards.append(float(reward))
            rollout_root_h.append(float(info.get("root_height", 0.0)))
            rollout_vel_x.append(float(info.get("vel_x", 0.0)))
            rollout_wx.append(float(info.get("wx", 0.0)))
            rollout_wy.append(float(info.get("wy", 0.0)))
            rollout_hri_fx.append(float(info.get("hri_fx", 0.0)))
            rollout_hri_fz.append(float(info.get("hri_fz", 0.0)))
            rollout_hri_ty.append(float(info.get("hri_ty", 0.0)))

            if cfg.debug_rollout and update == 1 and step_idx < 5:
                print(
                    f"[debug step {step_idx:03d}] reward={reward:8.4f} "
                    f"root_h={info.get('root_height', 0.0):.3f} "
                    f"vel_x={info.get('vel_x', 0.0):.3f} "
                    f"hri=({info.get('hri_fx', 0.0):.1f},{info.get('hri_fz', 0.0):.1f},{info.get('hri_ty', 0.0):.1f}) "
                    f"teacher_mse={teacher_mse:.5f}"
                )
                print("  action :", np.array2string(raw_action_np, precision=3, suppress_small=True))
                print("  teacher:", np.array2string(teacher_action_np, precision=3, suppress_small=True))

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
        bc_loss_epoch = 0.0
        num_batches = 0

        bc_coef = cfg.bc_coef
        if cfg.bc_warmup_updates > 0:
            bc_coef *= min(1.0, float(update) / float(cfg.bc_warmup_updates))

        for _ in range(cfg.update_epochs):
            for batch in buffer.batches(cfg.minibatch_size):
                b_obs, b_actions, b_teacher_actions, b_logprobs, b_advantages, b_returns, b_values = batch

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

                obs_n = policy.normalize_obs(b_obs)
                student_mean = policy.actor(obs_n)
                bc_loss = torch.mean((student_mean - b_teacher_actions) ** 2)

                entropy_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss + bc_coef * bc_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()

                pg_loss_epoch += pg_loss.item()
                vf_loss_epoch += value_loss.item()
                entropy_epoch += entropy_loss.item()
                bc_loss_epoch += bc_loss.item()
                num_batches += 1

            if cfg.target_kl is not None and approx_kl is not None and approx_kl > cfg.target_kl:
                break

        if num_batches > 0:
            pg_loss_epoch /= num_batches
            vf_loss_epoch /= num_batches
            entropy_epoch /= num_batches
            bc_loss_epoch /= num_batches

        print(
            f"[update {update:04d}] "
            f"pg_loss={pg_loss_epoch:9.5f} "
            f"vf_loss={vf_loss_epoch:9.5f} "
            f"bc_loss={bc_loss_epoch:9.5f} "
            f"bc_coef={bc_coef:7.4f} "
            f"entropy={entropy_epoch:8.5f} "
            f"approx_kl={(approx_kl if approx_kl is not None else 0.0):8.5f} "
            f"teacher_mse_rollout={float(np.mean(rollout_teacher_mse)) if rollout_teacher_mse else 0.0:.5f}"
        )

        log_wandb(
            wandb_run,
            {
                "loss/pg_loss": pg_loss_epoch,
                "loss/vf_loss": vf_loss_epoch,
                "loss/bc_loss": bc_loss_epoch,
                "loss/bc_coef": bc_coef,
                "loss/entropy": entropy_epoch,
                "loss/approx_kl": approx_kl if approx_kl is not None else 0.0,
                "train/reward_mean": float(np.mean(rollout_rewards)) if rollout_rewards else 0.0,
                "train/reward_sum": float(np.sum(rollout_rewards)) if rollout_rewards else 0.0,
                "train/root_h_mean": float(np.mean(rollout_root_h)) if rollout_root_h else 0.0,
                "train/vel_x_mean": float(np.mean(rollout_vel_x)) if rollout_vel_x else 0.0,
                "train/wx_rms": rms(rollout_wx),
                "train/wy_rms": rms(rollout_wy),
                "train/hri_fx_rms": rms(rollout_hri_fx),
                "train/hri_fz_mean": float(np.mean(rollout_hri_fz)) if rollout_hri_fz else 0.0,
                "train/hri_ty_rms": rms(rollout_hri_ty),
                "train/teacher_mse_rollout": float(np.mean(rollout_teacher_mse)) if rollout_teacher_mse else 0.0,
                "train/teacher_action_norm": float(np.mean(rollout_teacher_action_norm)) if rollout_teacher_action_norm else 0.0,
            },
            step=update,
        )

        if update % cfg.save_every == 0 or update == cfg.total_updates:
            save_path = save_checkpoint(policy, optimizer, cfg, update, save_dir)
            print(f"Saved checkpoint: {save_path}")

    if wandb_run is not None:
        wandb_run.finish()


def run_eval(env: SRLMujocoHRIEnv, policy, cfg: PPOConfig, wandb_run=None):
    policy.eval()
    obs_np, _ = env.reset(seed=cfg.seed)
    episode_return = 0.0
    episode_length = 0
    teacher_mse_all = []

    print(f"Running eval-only for {cfg.eval_steps} steps...")
    for step_idx in range(cfg.eval_steps):
        obs_t = numpy_to_torch_obs(obs_np, device=cfg.device)
        with torch.no_grad():
            action_t = policy.act_deterministic(obs_t)

        action_np = np.clip(action_t.squeeze(0).cpu().numpy(), -cfg.action_clip, cfg.action_clip)
        obs_np, reward, terminated, truncated, info = env.step(action_np)
        teacher = np.asarray(info.get("teacher_action", np.zeros(env.act_dim)), dtype=np.float32)
        teacher_mse = float(np.mean((action_np - teacher) ** 2))
        teacher_mse_all.append(teacher_mse)

        episode_return += reward
        episode_length += 1

        if (step_idx + 1) % 100 == 0:
            print(
                f"[eval {step_idx + 1:05d}] reward_last={reward:8.4f} "
                f"root_h={info.get('root_height', 0.0):.3f} "
                f"vel_x={info.get('vel_x', 0.0):.3f} "
                f"hri_fz={info.get('hri_fz', 0.0):.1f} "
                f"teacher_mse={float(np.mean(teacher_mse_all[-100:])):.5f}"
            )

        if terminated or truncated:
            print(
                f"[eval done] len={episode_length:4d} "
                f"return={episode_return:9.3f} root_h={info.get('root_height', 0.0):.3f}"
            )
            obs_np, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

    if teacher_mse_all:
        print(f"[eval summary] teacher_mse={float(np.mean(teacher_mse_all)):.5f}")
        log_wandb(wandb_run, {"eval/teacher_mse": float(np.mean(teacher_mse_all))})


def parse_args():
    parser = argparse.ArgumentParser(description="PPO finetuning on MuJoCo HRI 198D obs with teacher imitation loss.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--checkpoint-key", type=str, default="model_srl")
    parser.add_argument("--dataset", type=str, default=None, help="IsaacGym HRI dataset directory.")
    parser.add_argument("--isaac-replay-seq-len", type=int, default=1000)
    parser.add_argument("--xml", type=str, default="mjcf/srl_real_v1/srl_real_bot_v1.xml")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--bc-coef", type=float, default=0.05)
    parser.add_argument("--bc-warmup-updates", type=int, default=0)
    parser.add_argument("--init-log-std", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--debug-rollout", action="store_true")
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--torque-update-mode", type=str, default="physics", choices=["physics", "control"])
    parser.add_argument("--srl-action-filter", action="store_true")
    parser.add_argument("--srl-action-filter-cutoff-hz", type=float, default=4.0)
    parser.add_argument("--no-isaac-load-cell-wrench", action="store_true")
    parser.add_argument("--zero-isaac-human-obs", action="store_true")
    parser.add_argument("--isaac-load-cell-force-scale", type=float, default=1.0)
    parser.add_argument("--isaac-load-cell-torque-scale", type=float, default=1.0)
    parser.add_argument("--isaac-load-cell-force-clip-x", type=float, default=100.0)
    parser.add_argument("--isaac-load-cell-force-clip-y", type=float, default=50.0)
    parser.add_argument("--isaac-load-cell-force-clip-z", type=float, default=100.0)
    parser.add_argument("--hri-wrench-ramp-time", type=float, default=0.0)
    parser.add_argument("--hri-command-ramp-time", type=float, default=0.0)
    parser.add_argument("--base-wobble-penalty-scale", type=float, default=2.0)
    parser.add_argument("--base-ang-acc-penalty-scale", type=float, default=0.001)
    parser.add_argument("--yaw-drift-penalty-scale", type=float, default=0.3)
    parser.add_argument("--foot-impact-penalty-scale", type=float, default=0.0)
    parser.add_argument("--foot-force-threshold-bw", type=float, default=1.8)
    parser.add_argument("--foot-force-penalty-power", type=float, default=2.0)
    parser.add_argument("--hri-wrench-penalty-scale", type=float, default=0.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="srl-mujoco-hri-finetune")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PPOConfig(
        checkpoint_path=args.checkpoint,
        checkpoint_key=args.checkpoint_key,
        isaac_dataset_dir=args.dataset,
        isaac_replay_seq_len=args.isaac_replay_seq_len,
        xml_path=args.xml,
        total_updates=args.updates,
        rollout_steps=args.rollout_steps,
        learning_rate=args.lr,
        target_kl=args.target_kl,
        bc_coef=args.bc_coef,
        bc_warmup_updates=args.bc_warmup_updates,
        init_log_std=args.init_log_std,
        device=args.device,
        seed=args.seed,
        save_every=args.save_every,
        eval_only=args.eval_only,
        eval_steps=args.eval_steps,
        debug_rollout=args.debug_rollout,
        resume_optimizer=args.resume_optimizer,
        torque_update_mode=args.torque_update_mode,
        srl_action_filter=args.srl_action_filter,
        srl_action_filter_cutoff_hz=args.srl_action_filter_cutoff_hz,
        apply_isaac_load_cell_wrench=not args.no_isaac_load_cell_wrench,
        use_isaac_human_obs=not args.zero_isaac_human_obs,
        isaac_load_cell_force_scale=args.isaac_load_cell_force_scale,
        isaac_load_cell_torque_scale=args.isaac_load_cell_torque_scale,
        isaac_load_cell_force_clip_x=args.isaac_load_cell_force_clip_x,
        isaac_load_cell_force_clip_y=args.isaac_load_cell_force_clip_y,
        isaac_load_cell_force_clip_z=args.isaac_load_cell_force_clip_z,
        hri_wrench_ramp_time=args.hri_wrench_ramp_time,
        hri_command_ramp_time=args.hri_command_ramp_time,
        base_wobble_penalty_scale=args.base_wobble_penalty_scale,
        base_ang_acc_penalty_scale=args.base_ang_acc_penalty_scale,
        yaw_drift_penalty_scale=args.yaw_drift_penalty_scale,
        foot_impact_penalty_scale=args.foot_impact_penalty_scale,
        foot_force_threshold_bw=args.foot_force_threshold_bw,
        foot_force_penalty_power=args.foot_force_penalty_power,
        hri_wrench_penalty_scale=args.hri_wrench_penalty_scale,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )
    train(cfg)
