from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mujoco_ppo.models import safe_torch_load
from mujoco_ppo.models_v3 import AsymmetricModelConfig, AsymmetricActorCritic, load_asymmetric_checkpoint


class ConcurrentPrivilegedEstimator(nn.Module):
    """Estimate current root height and body-frame linear velocity."""

    def __init__(self, input_dim=26, history_len=10, output_dim=4,
                 hidden_dims=(256, 128, 64), normalization_epsilon=1e-4):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.output_dim = int(output_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.normalization_epsilon = float(normalization_epsilon)
        layers = []
        previous = self.input_dim * self.history_len
        for width in self.hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ELU()))
            previous = width
        layers.append(nn.Linear(previous, self.output_dim))
        self.network = nn.Sequential(*layers)
        self.register_buffer("input_mean", torch.zeros(self.input_dim))
        self.register_buffer("input_var", torch.ones(self.input_dim))
        self.register_buffer("input_count", torch.tensor(self.normalization_epsilon))
        self.register_buffer("target_mean", torch.zeros(self.output_dim))
        self.register_buffer("target_var", torch.ones(self.output_dim))
        self.register_buffer("target_count", torch.tensor(self.normalization_epsilon))

    @staticmethod
    def _moments(mean, var, count, batch):
        batch = batch.float()
        n = torch.as_tensor(float(batch.shape[0]), device=batch.device)
        batch_mean = batch.mean(0)
        batch_var = batch.var(0, unbiased=False)
        delta = batch_mean - mean
        total = count + n
        new_mean = mean + delta * n / total
        new_var = (var * count + batch_var * n + delta.square() * count * n / total) / total
        return new_mean, new_var.clamp_min(1e-8), total

    def _validate(self, history, target=None):
        expected = (self.history_len, self.input_dim)
        if history.ndim != 3 or tuple(history.shape[1:]) != expected:
            raise ValueError(f"expected history [batch, {expected[0]}, {expected[1]}], got {tuple(history.shape)}")
        if target is not None and (target.ndim != 2 or target.shape[1] != self.output_dim):
            raise ValueError(f"expected target [batch, {self.output_dim}], got {tuple(target.shape)}")

    @torch.no_grad()
    def update_normalization(self, history, target):
        self._validate(history, target)
        values = self._moments(self.input_mean, self.input_var, self.input_count,
                               history.reshape(-1, self.input_dim))
        targets = self._moments(self.target_mean, self.target_var, self.target_count, target)
        self.input_mean.copy_(values[0]); self.input_var.copy_(values[1]); self.input_count.copy_(values[2])
        self.target_mean.copy_(targets[0]); self.target_var.copy_(targets[1]); self.target_count.copy_(targets[2])

    def input_std(self):
        return self.input_var.clamp_min(1e-8).sqrt()

    def target_std(self):
        return self.target_var.clamp_min(1e-8).sqrt()

    def forward_normalized(self, history):
        self._validate(history)
        normalized = (history - self.input_mean) / self.input_std()
        return self.network(normalized.reshape(history.shape[0], -1))

    def forward(self, history):
        return self.forward_normalized(history) * self.target_std() + self.target_mean

    def normalized_mse(self, history, target):
        self._validate(history, target)
        normalized_target = (target - self.target_mean) / self.target_std()
        return F.mse_loss(self.forward_normalized(history), normalized_target)

    def model_config(self):
        return {"input_dim": self.input_dim, "history_len": self.history_len,
                "output_dim": self.output_dim, "hidden_dims": list(self.hidden_dims),
                "normalization_epsilon": self.normalization_epsilon}


@dataclass
class ConcurrentModelConfig:
    actor_obs_dim: int = 137
    critic_obs_dim: int = 153
    act_dim: int = 6
    hidden_sizes: Tuple[int, ...] = (512, 256, 128)
    estimator_history_len: int = 10
    estimator_hidden_sizes: Tuple[int, ...] = (256, 128, 64)


class ConcurrentActorCritic(AsymmetricActorCritic):
    def __init__(self, cfg: Optional[ConcurrentModelConfig] = None):
        self.concurrent_cfg = cfg or ConcurrentModelConfig()
        super().__init__(AsymmetricModelConfig(
            actor_obs_dim=self.concurrent_cfg.actor_obs_dim,
            critic_obs_dim=self.concurrent_cfg.critic_obs_dim,
            act_dim=self.concurrent_cfg.act_dim,
            hidden_sizes=self.concurrent_cfg.hidden_sizes,
        ))
        self.estimator = ConcurrentPrivilegedEstimator(
            history_len=self.concurrent_cfg.estimator_history_len,
            hidden_dims=self.concurrent_cfg.estimator_hidden_sizes,
        )

    def compose_actor_obs(self, deployable_obs, history, truth, use_estimate):
        estimate = self.estimator(history).detach()
        mask = use_estimate.to(dtype=torch.bool).reshape(-1, 1)
        selected = torch.where(mask, estimate, truth)
        return torch.cat((deployable_obs, selected), dim=-1), estimate


def load_concurrent_checkpoint(path: str, cfg: ConcurrentModelConfig, device="cpu",
                               checkpoint_key="model"):
    checkpoint = safe_torch_load(path, map_location=device)
    policy = ConcurrentActorCritic(cfg).to(device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        incompatible = policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if any(key.startswith(("actor_mlp.", "mu.")) for key in incompatible.missing_keys):
            raise RuntimeError("MuJoCo checkpoint is missing actor weights")
        if "concurrent_estimator" in checkpoint:
            policy.estimator.load_state_dict(checkpoint["concurrent_estimator"], strict=True)
        return policy, {"checkpoint_type": "mujoco_concurrent", "update": checkpoint.get("update")}

    base, metadata = load_asymmetric_checkpoint(
        path,
        AsymmetricModelConfig(137, 153, 6, cfg.hidden_sizes),
        device=device,
        checkpoint_key=checkpoint_key,
    )
    policy.actor_obs_norm.load_state_dict(base.actor_obs_norm.state_dict())
    policy.critic_obs_norm.load_state_dict(base.critic_obs_norm.state_dict())
    policy.value_norm.load_state_dict(base.value_norm.state_dict())
    policy.actor_mlp.load_state_dict(base.actor_mlp.state_dict())
    policy.critic_mlp.load_state_dict(base.critic_mlp.state_dict())
    policy.mu.load_state_dict(base.mu.state_dict())
    policy.value.load_state_dict(base.value.state_dict())
    policy.log_std.data.copy_(base.log_std.data)
    if not isinstance(checkpoint, dict) or "concurrent_estimator" not in checkpoint:
        raise ValueError("Expected concurrent_estimator in the Isaac Gym checkpoint")
    policy.estimator.load_state_dict(checkpoint["concurrent_estimator"], strict=True)
    metadata = dict(metadata)
    metadata["checkpoint_type"] = "isaacgym_concurrent"
    return policy, metadata
