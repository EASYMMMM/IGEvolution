from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from mujoco_ppo.models import (
    RunningNorm,
    RunningValueNorm,
    _build_mlp,
    _copy_linear,
    _unwrap_tensor,
    safe_torch_load,
)


class CensoredNormal:
    """Normal distribution whose samples are censored to symmetric bounds.

    Values outside the action interval are mapped to the corresponding bound.
    Consequently, each bound has a probability mass equal to the Gaussian tail
    beyond it.  Accounting for those masses keeps rollout actions, executed
    actions, and PPO log probabilities consistent while preserving the legacy
    hard-clipping behavior.
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, action_clip: float):
        if action_clip <= 0.0:
            raise ValueError(f"action_clip must be positive, got {action_clip}")
        self.base_dist = Normal(loc, scale)
        self.action_clip = float(action_clip)

    @property
    def raw_mean(self) -> torch.Tensor:
        return self.base_dist.mean

    @property
    def mean(self) -> torch.Tensor:
        return torch.clamp(
            self.raw_mean, -self.action_clip, self.action_clip
        )

    @property
    def stddev(self) -> torch.Tensor:
        return self.base_dist.stddev

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        return torch.clamp(
            self.base_dist.sample(sample_shape),
            -self.action_clip,
            self.action_clip,
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        low = torch.as_tensor(
            -self.action_clip, device=value.device, dtype=value.dtype
        )
        high = torch.as_tensor(
            self.action_clip, device=value.device, dtype=value.dtype
        )
        loc = self.base_dist.loc
        scale = self.base_dist.scale

        interior_log_prob = self.base_dist.log_prob(value)
        # P(X <= low) and P(X >= high), evaluated stably in log space.
        low_log_mass = torch.special.log_ndtr((low - loc) / scale)
        high_log_mass = torch.special.log_ndtr((loc - high) / scale)
        return torch.where(
            value <= low,
            low_log_mass,
            torch.where(value >= high, high_log_mass, interior_log_prob),
        )

    def entropy(self) -> torch.Tensor:
        # The mixed continuous/discrete censored entropy has no simple closed
        # form. Preserve the previous Gaussian entropy regularizer; PPO ratios
        # still use the exact censored log probability above.
        return self.base_dist.entropy()


@dataclass
class AsymmetricModelConfig:
    actor_obs_dim: int = 133
    critic_obs_dim: int = 153
    act_dim: int = 6
    hidden_sizes: Tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    init_log_std: float = 0.0


class AsymmetricActorCritic(nn.Module):
    def __init__(self, cfg: Optional[AsymmetricModelConfig] = None):
        super().__init__()
        self.cfg = cfg or AsymmetricModelConfig()
        self.actor_obs_norm = RunningNorm(self.cfg.actor_obs_dim)
        self.critic_obs_norm = RunningNorm(self.cfg.critic_obs_dim)
        self.value_norm = RunningValueNorm()

        self.actor_mlp = _build_mlp(
            self.cfg.actor_obs_dim, self.cfg.hidden_sizes, self.cfg.activation
        )
        self.critic_mlp = _build_mlp(
            self.cfg.critic_obs_dim, self.cfg.hidden_sizes, self.cfg.activation
        )
        self.mu = nn.Linear(self.cfg.hidden_sizes[-1], self.cfg.act_dim)
        self.value = nn.Linear(self.cfg.hidden_sizes[-1], 1)
        self.log_std = nn.Parameter(
            torch.full((self.cfg.act_dim,), self.cfg.init_log_std)
        )

    def normalize_actor_obs(self, obs: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
        return self.actor_obs_norm(obs, clip=clip)

    def normalize_critic_obs(self, obs: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
        return self.critic_obs_norm(obs, clip=clip)

    def actor(self, actor_obs_n: torch.Tensor) -> torch.Tensor:
        return self.mu(self.actor_mlp(actor_obs_n))

    def critic(self, critic_obs_n: torch.Tensor) -> torch.Tensor:
        return self.value(self.critic_mlp(critic_obs_n)).squeeze(-1)

    def dist(
        self, actor_obs_n: torch.Tensor, action_clip: float = 1.0
    ) -> CensoredNormal:
        mean = self.actor(actor_obs_n)
        std = torch.exp(self.log_std).expand_as(mean)
        return CensoredNormal(mean, std, action_clip=action_clip)

    def act(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        action_clip: float = 1.0,
    ):
        actor_obs_n = self.normalize_actor_obs(actor_obs)
        critic_obs_n = self.normalize_critic_obs(critic_obs)
        dist = self.dist(actor_obs_n, action_clip=action_clip)
        action = dist.sample()
        return (
            action,
            dist.log_prob(action).sum(dim=-1),
            self.critic(critic_obs_n),
        )

    def act_deterministic(
        self, actor_obs: torch.Tensor, action_clip: float = 1.0
    ) -> torch.Tensor:
        actor_obs_n = self.normalize_actor_obs(actor_obs)
        return self.dist(actor_obs_n, action_clip=action_clip).mean

    def evaluate_actions(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        action_clip: float = 1.0,
    ):
        dist = self.dist(
            self.normalize_actor_obs(actor_obs), action_clip=action_clip
        )
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(self.normalize_critic_obs(critic_obs))
        return log_prob, entropy, value


def _copy_running_norm(norm: RunningNorm, state: Dict[str, Any], prefixes) -> bool:
    for prefix in prefixes:
        mean_key = f"{prefix}.running_mean"
        var_key = f"{prefix}.running_var"
        if mean_key not in state or var_key not in state:
            continue
        mean = _unwrap_tensor(state[mean_key]).reshape(-1)
        var = _unwrap_tensor(state[var_key]).reshape(-1)
        if mean.numel() != norm.running_mean.numel():
            continue
        norm.running_mean.copy_(mean.to(norm.running_mean))
        norm.running_var.copy_(var.to(norm.running_var))
        return True
    return False


def _copy_value_norm(norm: RunningValueNorm, state: Dict[str, Any], prefixes) -> bool:
    for prefix in prefixes:
        mean_key = f"{prefix}.running_mean"
        var_key = f"{prefix}.running_var"
        if mean_key not in state or var_key not in state:
            continue
        mean = _unwrap_tensor(state[mean_key]).reshape(-1)
        var = _unwrap_tensor(state[var_key]).reshape(-1)
        if mean.numel() != 1 or var.numel() != 1:
            continue
        norm.running_mean.copy_(mean[0].to(norm.running_mean))
        norm.running_var.copy_(var[0].to(norm.running_var))
        count_key = f"{prefix}.count"
        if count_key in state:
            count = _unwrap_tensor(state[count_key]).reshape(-1)
            if count.numel() == 1:
                norm.count.copy_(count[0].to(norm.count))
        return True
    return False


def _find_tensor(state: Dict[str, Any], suffixes):
    for suffix in suffixes:
        if suffix in state:
            return state[suffix]
    for key, value in state.items():
        if any(str(key).endswith(suffix) for suffix in suffixes):
            return value
    return None


def _collect_named_mlp_layers(state: Dict[str, Any], token: str):
    weights: Dict[int, Any] = {}
    biases: Dict[int, Any] = {}
    for key, value in state.items():
        parts = str(key).split(".")
        try:
            token_idx = parts.index(token)
        except ValueError:
            continue
        if token_idx + 2 >= len(parts) or not parts[token_idx + 1].isdigit():
            continue
        layer_idx = int(parts[token_idx + 1])
        parameter_name = parts[token_idx + 2]
        if parameter_name == "weight":
            weights[layer_idx] = value
        elif parameter_name == "bias":
            biases[layer_idx] = value
    return [(weights[idx], biases.get(idx)) for idx in sorted(weights)]


def _load_mlp_and_head(
    mlp: nn.Sequential,
    head: nn.Linear,
    state: Dict[str, Any],
    mlp_token: str,
    head_weight_suffixes,
    head_bias_suffixes,
) -> bool:
    source_layers = _collect_named_mlp_layers(state, mlp_token)
    target_layers = [module for module in mlp if isinstance(module, nn.Linear)]
    if len(source_layers) != len(target_layers):
        return False
    if any(tuple(_unwrap_tensor(weight).shape) != tuple(layer.weight.shape)
           for layer, (weight, _bias) in zip(target_layers, source_layers)):
        return False
    for layer, (weight, bias) in zip(target_layers, source_layers):
        _copy_linear(layer, weight, bias)
    head_weight = _find_tensor(state, head_weight_suffixes)
    head_bias = _find_tensor(state, head_bias_suffixes)
    if head_weight is None or tuple(_unwrap_tensor(head_weight).shape) != tuple(head.weight.shape):
        return False
    _copy_linear(head, head_weight, head_bias)
    return True


def _unwrap_central_value_state(checkpoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in (
        "assymetric_vf_nets",  # Historical rl-games checkpoint spelling.
        "asymmetric_vf_nets",
        "central_value",
        "central_critic",
        "central_value_net",
    ):
        candidate = checkpoint.get(key)
        if not isinstance(candidate, dict):
            continue
        for nested_key in ("model", "model_state_dict", "state_dict"):
            nested = candidate.get(nested_key)
            if isinstance(nested, dict):
                return nested
        return candidate
    return None


def load_asymmetric_checkpoint(
    checkpoint_path: str,
    model_cfg: Optional[AsymmetricModelConfig] = None,
    device: str = "cpu",
    checkpoint_key: str = "model",
):
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    policy = AsymmetricActorCritic(model_cfg).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        incompatible = policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
        metadata = {
            "checkpoint_type": "mujoco_v3_finetune",
            "checkpoint_keys": sorted(checkpoint.keys()),
            "actor_obs_dim": policy.cfg.actor_obs_dim,
            "critic_obs_dim": policy.cfg.critic_obs_dim,
            "critic_loaded": not any(key.startswith("critic_") or key.startswith("value.")
                                      for key in incompatible.missing_keys),
            "actor_obs_norm_loaded": not any(key.startswith("actor_obs_norm.")
                                               for key in incompatible.missing_keys),
            "critic_obs_norm_loaded": not any(key.startswith("critic_obs_norm.")
                                                for key in incompatible.missing_keys),
            "update": checkpoint.get("update"),
        }
        return policy, metadata

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")
    actor_state = checkpoint.get(checkpoint_key, checkpoint.get("model", checkpoint))
    if not isinstance(actor_state, dict):
        raise TypeError(f"Checkpoint key {checkpoint_key!r} does not contain a state dict")

    actor_loaded = _load_mlp_and_head(
        policy.actor_mlp,
        policy.mu,
        actor_state,
        "actor_mlp",
        ("a2c_network.mu.weight", "mu.weight"),
        ("a2c_network.mu.bias", "mu.bias"),
    )
    if not actor_loaded:
        raise RuntimeError(
            "Could not load the 133D Isaac actor. Check that this is the new "
            "non-privileged policy checkpoint and that hidden sizes match."
        )

    actor_norm_loaded = _copy_running_norm(
        policy.actor_obs_norm,
        actor_state,
        ("running_mean_std", "actor_obs_norm"),
    )
    sigma = _find_tensor(
        actor_state,
        ("a2c_network.sigma", "a2c_network.log_std", "sigma", "log_std"),
    )
    if sigma is not None and tuple(_unwrap_tensor(sigma).shape) == tuple(policy.log_std.shape):
        policy.log_std.data.copy_(_unwrap_tensor(sigma).to(policy.log_std))

    critic_loaded = False
    critic_norm_loaded = False
    value_norm_loaded = False
    central_state = _unwrap_central_value_state(checkpoint)
    if central_state is not None:
        critic_loaded = _load_mlp_and_head(
            policy.critic_mlp,
            policy.value,
            central_state,
            # rl-games names the MLP inside its central-value model actor_mlp.
            "actor_mlp",
            ("a2c_network.value.weight", "value.weight"),
            ("a2c_network.value.bias", "value.bias"),
        )
        critic_norm_loaded = _copy_running_norm(
            policy.critic_obs_norm,
            central_state,
            (
                "model.running_mean_std",
                "running_mean_std",
                "critic_obs_norm",
            ),
        )
        value_norm_loaded = _copy_value_norm(
            policy.value_norm,
            central_state,
            (
                "model.value_mean_std",
                "value_mean_std",
                "value_norm",
            ),
        )

    metadata = {
        "checkpoint_type": "isaacgym_asymmetric",
        "checkpoint_keys": sorted(checkpoint.keys()),
        "selected_key": checkpoint_key,
        "actor_obs_dim": policy.cfg.actor_obs_dim,
        "critic_obs_dim": policy.cfg.critic_obs_dim,
        "actor_loaded": actor_loaded,
        "critic_loaded": critic_loaded,
        "actor_obs_norm_loaded": actor_norm_loaded,
        "critic_obs_norm_loaded": critic_norm_loaded,
        "value_norm_loaded": value_norm_loaded,
    }
    return policy, metadata
