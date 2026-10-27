"""153-D latent actor with the same privileged critic as concurrent PPO."""

from dataclasses import dataclass

import torch

from mujoco_ppo.latent_encoder import ConcurrentLatentEncoder
from mujoco_ppo.models import safe_torch_load
from mujoco_ppo.models_v3 import AsymmetricActorCritic, AsymmetricModelConfig, load_asymmetric_checkpoint


def mirror_frames(frames):
    """Mirror 26-D frames without reversing the oldest-to-newest time axis."""
    result = frames.clone()
    result[..., (0, 2, 3, 5, 24, 25)] *= -1
    for start in (6, 12, 18):
        result[..., start:start + 6] = frames[..., start:start + 6][..., [3, 4, 5, 0, 1, 2]]
    return result


def mirror_explicit(target):
    return target * target.new_tensor([1, 1, -1, 1])


@dataclass
class LatentModelConfig(AsymmetricModelConfig):
    actor_obs_dim: int = 153
    critic_obs_dim: int = 153


class ConcurrentLatentActorCritic(AsymmetricActorCritic):
    checkpoint_prefix = "mujoco_latent"

    def __init__(self, cfg=None, encoder=None):
        super().__init__(cfg or LatentModelConfig())
        self.encoder = encoder if encoder is not None else ConcurrentLatentEncoder()
        self.latent_mode = "normal"

    @property
    def estimator(self):
        return self.encoder

    def compose_actor_obs(self, deployable, history, truth, use_estimate):
        output = self.encoder(history)
        selected = torch.where(use_estimate.bool().reshape(-1, 1), output["explicit"].detach(), truth)
        latent = self.encoder.ablate_latent(output["latent"], self.latent_mode)
        return torch.cat((deployable, selected, latent), -1), output["explicit"]

    def compose_mirrored_actor_obs(self, deployable, mirrored, history, truth, modes):
        actor, estimate = self.compose_actor_obs(deployable, history, truth, modes)
        mirror, _ = self.compose_actor_obs(mirrored, mirror_frames(history), mirror_explicit(truth), modes)
        return actor, mirror, estimate

    def latent_checkpoint_payload(self):
        return {"checkpoint_type": "mujoco_concurrent_latent",
                "concurrent_latent_model": self.encoder.state_dict(),
                "concurrent_latent_config": self.encoder.model_config()}


def load_latent_checkpoint(path, device="cpu", checkpoint_key="model"):
    saved = safe_torch_load(path, map_location="cpu")
    if not isinstance(saved, dict) or "concurrent_latent_model" not in saved:
        raise ValueError("Expected a latent checkpoint containing concurrent_latent_model (not a 137-D concurrent policy)")
    encoder = ConcurrentLatentEncoder.from_checkpoint(
        saved["concurrent_latent_model"], saved.get("concurrent_latent_config", {}))
    if (encoder.input_dim, encoder.explicit_dim, encoder.latent_dim) != (26, 4, 16):
        raise ValueError("This trainer requires 26-D frames, explicit=4 and latent=16")
    state = saved.get("model_state_dict", saved.get(checkpoint_key, saved.get("model")))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint has no actor state")
    layers = [(key, value) for key, value in state.items()
              if "actor_mlp." in key and key.endswith(".weight")]
    layers.sort(key=lambda item: int(item[0].split("actor_mlp.")[1].split(".")[0]))
    if not layers or layers[0][1].shape[1] != 153:
        raise ValueError("Expected a 153-D latent actor")
    cfg = LatentModelConfig(hidden_sizes=tuple(value.shape[0] for _, value in layers))
    if "model_state_dict" in saved:
        policy = ConcurrentLatentActorCritic(cfg, encoder).to(device)
        policy.load_state_dict(state, strict=True)
        return policy, {"checkpoint_type": "mujoco_concurrent_latent", "update": saved.get("update")}
    base, metadata = load_asymmetric_checkpoint(path, cfg, device, checkpoint_key)
    policy = ConcurrentLatentActorCritic(cfg, encoder).to(device)
    policy.load_state_dict(base.state_dict(), strict=False)
    if "running_mean_std" in saved:
        norm = saved["running_mean_std"]
        policy.actor_obs_norm.running_mean.copy_(norm["running_mean"].reshape(153).to(device))
        policy.actor_obs_norm.running_var.copy_(norm["running_var"].reshape(153).to(device))
    metadata["checkpoint_type"] = "isaacgym_concurrent_latent"
    return policy, metadata
