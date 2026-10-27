"""Standalone PyTorch port of the Isaac ConcurrentLatentEncoder.

Keep parameter/buffer names and decoder versions compatible with
learning/SRLEvo/concurrent_latent_models.py; no Isaac Gym import is needed.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConcurrentLatentEncoder(nn.Module):
    def __init__(self, input_dim=26, history_len=10, explicit_dim=4, latent_dim=16,
                 hidden_dims=(256, 128, 64), decoder_hidden_dims=(128, 128),
                 com_cop_dim=3, contact_dim=2, normalization_epsilon=1e-4,
                 model_version=2):
        super().__init__()
        self.input_dim, self.history_len = int(input_dim), int(history_len)
        self.explicit_dim, self.latent_dim = int(explicit_dim), int(latent_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.decoder_hidden_dims = tuple(decoder_hidden_dims)
        self.com_cop_dim, self.contact_dim = int(com_cop_dim), int(contact_dim)
        self.normalization_epsilon = float(normalization_epsilon)
        self.model_version = int(model_version)
        if self.model_version not in (1, 2):
            raise ValueError("Unsupported latent decoder version")
        self.encoder = self._mlp(self.input_dim * self.history_len,
                                 self.hidden_dims[:-1], self.hidden_dims[-1])
        self.explicit_head = nn.Linear(self.hidden_dims[-1], self.explicit_dim)
        self.latent_head = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.com_cop_head = nn.Linear(self.hidden_dims[-1], self.com_cop_dim)
        self.contact_head = nn.Linear(self.hidden_dims[-1], self.contact_dim)
        decoder_width = self.explicit_dim + self.latent_dim
        if self.model_version == 1:
            decoder_width += self.input_dim
        self.next_observation_decoder = self._mlp(
            decoder_width, self.decoder_hidden_dims, self.input_dim)
        for name, dim in (("input", self.input_dim), ("explicit", self.explicit_dim),
                          ("next", self.input_dim), ("com_cop", self.com_cop_dim)):
            self.register_buffer(name + "_mean", torch.zeros(dim))
            self.register_buffer(name + "_var", torch.ones(dim))
            self.register_buffer(name + "_count", torch.tensor(self.normalization_epsilon))

    @staticmethod
    def _mlp(input_dim, hidden_dims, output_dim):
        layers = []
        for width in hidden_dims:
            layers.extend((nn.Linear(input_dim, width), nn.ELU()))
            input_dim = width
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _std(var):
        return var.clamp_min(1e-8).sqrt()

    def _validate_history(self, history):
        if history.ndim != 3 or tuple(history.shape[1:]) != (self.history_len, self.input_dim):
            raise ValueError("Expected history [batch, {}, {}], got {}".format(
                self.history_len, self.input_dim, tuple(history.shape)))

    @torch.no_grad()
    def update_normalization(self, history, explicit_target, next_target=None, com_cop_target=None):
        self._validate_history(history)
        for name, batch in (("input", history.reshape(-1, self.input_dim)),
                            ("explicit", explicit_target), ("next", next_target),
                            ("com_cop", com_cop_target)):
            if batch is None or batch.shape[0] == 0:
                continue
            mean, var, count = (getattr(self, name + suffix) for suffix in ("_mean", "_var", "_count"))
            batch = batch.float()
            n = torch.as_tensor(float(batch.shape[0]), device=batch.device)
            delta = batch.mean(0) - mean
            total = count + n
            new_var = (var * count + batch.var(0, unbiased=False) * n
                       + delta.square() * count * n / total) / total
            mean.add_(delta * n / total)
            var.copy_(new_var.clamp_min(1e-8))
            count.copy_(total)

    def encode_normalized(self, history):
        self._validate_history(history)
        normalized = (history - self.input_mean) / self._std(self.input_var)
        feature = self.encoder(normalized.reshape(history.shape[0], -1))
        return normalized, feature, self.explicit_head(feature), self.latent_head(feature)

    def forward(self, history):
        normalized, feature, explicit, latent = self.encode_normalized(history)
        return {"explicit": explicit * self._std(self.explicit_var) + self.explicit_mean,
                "explicit_normalized": explicit, "latent": latent,
                "com_cop_normalized": self.com_cop_head(feature),
                "contact_logits": self.contact_head(feature), "normalized_history": normalized}

    def decode(self, output, latent=None):
        parts = [output["explicit_normalized"], output["latent"] if latent is None else latent]
        if self.model_version == 1:
            parts.insert(0, output["normalized_history"][:, -1])
        return self.next_observation_decoder(torch.cat(parts, -1))

    def compute_losses(self, history, explicit_target, next_target=None,
                       com_cop_target=None, contact_target=None):
        output = self(history)
        losses = {"explicit": F.mse_loss(output["explicit_normalized"],
                  (explicit_target - self.explicit_mean) / self._std(self.explicit_var)),
                  "latent_l2": output["latent"].square().mean()}
        if next_target is not None:
            losses["next_observation"] = F.mse_loss(self.decode(output),
                (next_target - self.next_mean) / self._std(self.next_var))
        if com_cop_target is not None:
            losses["com_cop"] = F.mse_loss(output["com_cop_normalized"],
                (com_cop_target - self.com_cop_mean) / self._std(self.com_cop_var))
        if contact_target is not None:
            losses["contact"] = F.binary_cross_entropy_with_logits(output["contact_logits"], contact_target)
        return losses

    @staticmethod
    def ablate_latent(latent, mode):
        if mode == "normal":
            return latent
        if mode == "zero":
            return torch.zeros_like(latent)
        if mode == "shuffle" and latent.shape[0] >= 2:
            return latent.roll(1, 0)
        raise ValueError("Expected normal/zero, or shuffle with at least two samples")

    @classmethod
    def from_checkpoint(cls, state, config):
        config = dict(config)
        width = state["next_observation_decoder.0.weight"].shape[1]
        encoded = int(config.get("explicit_dim", 4)) + int(config.get("latent_dim", 16))
        version = 2 if width == encoded else 1 if width == encoded + int(config.get("input_dim", 26)) else None
        if version is None or int(config.get("model_version", version)) != version:
            raise ValueError("Latent decoder weights disagree with model configuration")
        config["model_version"] = version
        model = cls(**config)
        model.load_state_dict(state, strict=True)
        return model

    def model_config(self):
        return {key: getattr(self, key) for key in (
            "input_dim", "history_len", "explicit_dim", "latent_dim", "hidden_dims",
            "decoder_hidden_dims", "com_cop_dim", "contact_dim", "normalization_epsilon", "model_version")}
