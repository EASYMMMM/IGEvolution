from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class ModelConfig:
    obs_dim: int = 153
    act_dim: int = 6
    hidden_sizes: Tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    init_log_std: float = 0.0


class RunningNorm(nn.Module):
    """Checkpoint-compatible observation normalization holder."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(obs_dim))
        self.register_buffer("running_var", torch.ones(obs_dim))

    def forward(self, obs: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
        std = torch.sqrt(self.running_var + 1e-8)
        obs = (obs - self.running_mean) / std
        return torch.clamp(obs, -clip, clip)


def _build_mlp(input_dim: int, hidden_sizes: Tuple[int, ...], activation: str) -> nn.Sequential:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    if activation not in activations:
        raise ValueError(f"Unsupported activation: {activation}")

    layers: List[nn.Module] = []
    last_dim = input_dim
    act_cls = activations[activation]
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(act_cls())
        last_dim = hidden_dim
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.obs_norm = RunningNorm(self.cfg.obs_dim)

        self.actor_mlp = _build_mlp(self.cfg.obs_dim, self.cfg.hidden_sizes, self.cfg.activation)
        self.critic_mlp = _build_mlp(self.cfg.obs_dim, self.cfg.hidden_sizes, self.cfg.activation)

        self.mu = nn.Linear(self.cfg.hidden_sizes[-1], self.cfg.act_dim)
        self.value = nn.Linear(self.cfg.hidden_sizes[-1], 1)
        self.log_std = nn.Parameter(torch.full((self.cfg.act_dim,), self.cfg.init_log_std))

    def normalize_obs(self, obs: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
        return self.obs_norm(obs, clip=clip)

    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.actor_mlp(obs)
        return self.mu(x)

    def critic(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.critic_mlp(obs)
        return self.value(x).squeeze(-1)

    def dist(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor):
        obs_n = self.normalize_obs(obs)
        dist = self.dist(obs_n)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs_n)
        return action, log_prob, value

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        obs_n = self.normalize_obs(obs)
        return self.actor(obs_n)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        obs_n = self.normalize_obs(obs)
        dist = self.dist(obs_n)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs_n)
        return log_prob, entropy, value


def _unwrap_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _copy_linear(dst: nn.Linear, weight: Any, bias: Optional[Any]):
    dst.weight.data.copy_(_unwrap_tensor(weight).to(dtype=dst.weight.dtype))
    if bias is not None:
        dst.bias.data.copy_(_unwrap_tensor(bias).to(dtype=dst.bias.dtype))


def _collect_mlp_layers(model_state: Dict[str, Any], token: str):
    weights: Dict[int, Any] = {}
    biases: Dict[int, Any] = {}
    for key, value in model_state.items():
        if token not in key:
            continue
        parts = key.split(".")
        if len(parts) < 4:
            continue
        idx = int(parts[2])
        if parts[-1] == "weight":
            weights[idx] = value
        elif parts[-1] == "bias":
            biases[idx] = value

    layers = []
    for idx in sorted(weights.keys()):
        layers.append((weights[idx], biases.get(idx)))
    return layers


def load_isaac_checkpoint(
    checkpoint_path: str,
    model_cfg: Optional[ModelConfig] = None,
    device="cpu",
    strict_critic: bool = False,
):
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        policy = ActorCritic(model_cfg).to(device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        metadata = {
            "checkpoint_type": "mujoco_finetune",
            "checkpoint_keys": sorted(list(checkpoint.keys())),
            "critic_loaded": True,
            "obs_norm_loaded": True,
            "update": checkpoint.get("update"),
            "optimizer_state_available": "optimizer_state_dict" in checkpoint,
        }
        return policy, metadata

    model_state = checkpoint["model"] if "model" in checkpoint else checkpoint

    policy = ActorCritic(model_cfg).to(device)

    actor_layers = _collect_mlp_layers(model_state, "actor_mlp")
    actor_linears = [m for m in policy.actor_mlp if isinstance(m, nn.Linear)]
    if len(actor_layers) != len(actor_linears):
        raise RuntimeError(
            f"Actor layer count mismatch: checkpoint={len(actor_layers)}, model={len(actor_linears)}"
        )
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
        for linear, (weight, bias) in zip(critic_linears, critic_layers):
            _copy_linear(linear, weight, bias)

        value_weight = (
            model_state.get("a2c_network.value.weight")
            or model_state.get("value.weight")
        )
        value_bias = (
            model_state.get("a2c_network.value.bias")
            or model_state.get("value.bias")
        )
        if value_weight is not None:
            _copy_linear(policy.value, value_weight, value_bias)
            critic_loaded = True
    elif strict_critic:
        raise RuntimeError("Could not find critic_mlp weights in Isaac checkpoint.")

    mean_key = "running_mean_std.running_mean"
    var_key = "running_mean_std.running_var"
    if mean_key in model_state and var_key in model_state:
        policy.obs_norm.running_mean.copy_(_unwrap_tensor(model_state[mean_key]).to(policy.obs_norm.running_mean))
        policy.obs_norm.running_var.copy_(_unwrap_tensor(model_state[var_key]).to(policy.obs_norm.running_var))

    sigma_candidates = [
        "a2c_network.sigma",
        "a2c_network.log_std",
        "sigma",
        "log_std",
    ]
    for key in sigma_candidates:
        if key in model_state:
            sigma_tensor = _unwrap_tensor(model_state[key]).to(device=policy.log_std.device, dtype=policy.log_std.dtype)
            if sigma_tensor.shape == policy.log_std.shape:
                policy.log_std.data.copy_(sigma_tensor)
                break

    metadata = {
        "checkpoint_type": "isaacgym",
        "checkpoint_keys": sorted(list(checkpoint.keys())),
        "critic_loaded": critic_loaded,
        "obs_norm_loaded": mean_key in model_state and var_key in model_state,
    }
    return policy, metadata


def numpy_to_torch_obs(obs: np.ndarray, device="cpu") -> torch.Tensor:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    return torch.from_numpy(obs).to(device)
