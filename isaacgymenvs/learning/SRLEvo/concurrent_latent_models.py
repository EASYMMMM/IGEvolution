import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcurrentLatentEncoder(nn.Module):
    """Causal history encoder with explicit, latent, and auxiliary heads."""

    def __init__(
        self,
        input_dim=26,
        history_len=10,
        explicit_dim=4,
        latent_dim=16,
        hidden_dims=(256, 128, 64),
        decoder_hidden_dims=(128, 128),
        com_cop_dim=3,
        contact_dim=2,
        normalization_epsilon=1e-4,
        model_version=2,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.explicit_dim = int(explicit_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(v) for v in hidden_dims)
        self.decoder_hidden_dims = tuple(int(v) for v in decoder_hidden_dims)
        self.com_cop_dim = int(com_cop_dim)
        self.contact_dim = int(contact_dim)
        self.normalization_epsilon = float(normalization_epsilon)
        self.model_version = int(model_version)
        if self.model_version not in (1, 2):
            raise ValueError("Unsupported latent model_version: {}".format(model_version))

        self.encoder = self._mlp(
            self.input_dim * self.history_len,
            self.hidden_dims[:-1],
            self.hidden_dims[-1],
        )
        feature_dim = self.hidden_dims[-1]
        self.explicit_head = nn.Linear(feature_dim, self.explicit_dim)
        self.latent_head = nn.Linear(feature_dim, self.latent_dim)
        self.com_cop_head = nn.Linear(feature_dim, self.com_cop_dim)
        self.contact_head = nn.Linear(feature_dim, self.contact_dim)
        decoder_input_dim = self.explicit_dim + self.latent_dim
        if self.model_version == 1:
            decoder_input_dim += self.input_dim
        self.next_observation_decoder = self._mlp(
            decoder_input_dim, self.decoder_hidden_dims, self.input_dim
        )

        self.register_buffer("input_mean", torch.zeros(self.input_dim))
        self.register_buffer("input_var", torch.ones(self.input_dim))
        self.register_buffer("input_count", torch.tensor(self.normalization_epsilon))
        self.register_buffer("explicit_mean", torch.zeros(self.explicit_dim))
        self.register_buffer("explicit_var", torch.ones(self.explicit_dim))
        self.register_buffer("explicit_count", torch.tensor(self.normalization_epsilon))
        self.register_buffer("next_mean", torch.zeros(self.input_dim))
        self.register_buffer("next_var", torch.ones(self.input_dim))
        self.register_buffer("next_count", torch.tensor(self.normalization_epsilon))
        self.register_buffer("com_cop_mean", torch.zeros(self.com_cop_dim))
        self.register_buffer("com_cop_var", torch.ones(self.com_cop_dim))
        self.register_buffer("com_cop_count", torch.tensor(self.normalization_epsilon))

    @staticmethod
    def _mlp(input_dim, hidden_dims, output_dim):
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ELU()))
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _updated_moments(mean, var, count, batch):
        batch = batch.float()
        if batch.shape[0] == 0:
            return mean, var, count
        batch_count = torch.as_tensor(float(batch.shape[0]), device=batch.device)
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        delta = batch_mean - mean
        total = count + batch_count
        new_mean = mean + delta * batch_count / total
        new_var = (
            var * count
            + batch_var * batch_count
            + delta.square() * count * batch_count / total
        ) / total
        return new_mean, new_var.clamp_min(1e-8), total

    @staticmethod
    def _copy_moments(mean, var, count, values):
        updated = ConcurrentLatentEncoder._updated_moments(mean, var, count, values)
        mean.copy_(updated[0])
        var.copy_(updated[1])
        count.copy_(updated[2])

    def _validate_history(self, history):
        expected = (self.history_len, self.input_dim)
        if history.ndim != 3 or tuple(history.shape[1:]) != expected:
            raise ValueError(
                "expected history [batch, {}, {}], got {}".format(
                    self.history_len, self.input_dim, tuple(history.shape)
                )
            )

    @staticmethod
    def _std(var):
        return torch.sqrt(var.clamp_min(1e-8))

    @torch.no_grad()
    def update_normalization(
        self, history, explicit_target, next_target=None, com_cop_target=None
    ):
        self._validate_history(history)
        self._copy_moments(
            self.input_mean,
            self.input_var,
            self.input_count,
            history.reshape(-1, self.input_dim),
        )
        self._copy_moments(
            self.explicit_mean,
            self.explicit_var,
            self.explicit_count,
            explicit_target,
        )
        if next_target is not None:
            self._copy_moments(
                self.next_mean, self.next_var, self.next_count, next_target
            )
        if com_cop_target is not None:
            self._copy_moments(
                self.com_cop_mean,
                self.com_cop_var,
                self.com_cop_count,
                com_cop_target,
            )

    def encode_normalized(self, history):
        self._validate_history(history)
        normalized = (history - self.input_mean) / self._std(self.input_var)
        feature = self.encoder(normalized.reshape(history.shape[0], -1))
        explicit_normalized = self.explicit_head(feature)
        latent = self.latent_head(feature)
        return normalized, feature, explicit_normalized, latent

    def forward(self, history):
        normalized, feature, explicit_normalized, latent = self.encode_normalized(history)
        explicit = (
            explicit_normalized * self._std(self.explicit_var) + self.explicit_mean
        )
        return {
            "explicit": explicit,
            "explicit_normalized": explicit_normalized,
            "latent": latent,
            "com_cop_normalized": self.com_cop_head(feature),
            "contact_logits": self.contact_head(feature),
            "normalized_history": normalized,
        }

    def compute_losses(
        self,
        history,
        explicit_target,
        next_target=None,
        com_cop_target=None,
        contact_target=None,
    ):
        output = self(history)
        explicit_target_normalized = (
            explicit_target - self.explicit_mean
        ) / self._std(self.explicit_var)
        losses = {
            "explicit": F.mse_loss(
                output["explicit_normalized"], explicit_target_normalized
            ),
            "latent_l2": output["latent"].square().mean(),
        }

        if next_target is not None:
            next_prediction = self.decode(output)
            next_target_normalized = (
                next_target - self.next_mean
            ) / self._std(self.next_var)
            losses["next_observation"] = F.mse_loss(
                next_prediction, next_target_normalized
            )

        if com_cop_target is not None:
            normalized_target = (
                com_cop_target - self.com_cop_mean
            ) / self._std(self.com_cop_var)
            losses["com_cop"] = F.mse_loss(
                output["com_cop_normalized"], normalized_target
            )
        if contact_target is not None:
            losses["contact"] = F.binary_cross_entropy_with_logits(
                output["contact_logits"], contact_target
            )
        return losses

    def decode(self, output, latent=None):
        parts = [output["explicit_normalized"],
                 output["latent"] if latent is None else latent]
        if self.model_version == 1:
            parts.insert(0, output["normalized_history"][:, -1])
        return self.next_observation_decoder(torch.cat(parts, dim=-1))

    @staticmethod
    def ablate_latent(latent, mode):
        if mode == "normal":
            return latent
        if mode == "zero":
            return torch.zeros_like(latent)
        if mode == "shuffle":
            if latent.shape[0] < 2:
                raise ValueError("shuffle requires at least two parallel environments/samples")
            # A fixed derangement avoids extra RNG draws and self-assignment.
            return latent.roll(1, dims=0)
        raise ValueError("latent mode must be normal, zero, or shuffle")

    @torch.no_grad()
    def diagnostic_metrics(self, history, explicit_target, next_target=None):
        output = self(history)
        latent = output["latent"]
        metrics = {"latent_std_{:02d}".format(i): value.item()
                   for i, value in enumerate(latent.std(dim=0, unbiased=False))}
        error = output["explicit"] - explicit_target
        for i, name in enumerate(("height", "vx", "vy", "vz")):
            metrics["explicit_{}_mae".format(name)] = error[:, i].abs().mean().item()
        if next_target is not None:
            target = (next_target - self.next_mean) / self._std(self.next_var)
            for mode in ("normal", "zero", "shuffle"):
                if mode == "shuffle" and latent.shape[0] < 2:
                    continue
                prediction = self.decode(output, self.ablate_latent(latent, mode))
                metrics["next_mse_" + mode] = F.mse_loss(prediction, target).item()
        return metrics

    @classmethod
    def from_checkpoint(cls, state, config):
        config = dict(config)
        input_dim = int(config.get("input_dim", 26))
        encoded_dim = int(config.get("explicit_dim", 4)) + int(config.get("latent_dim", 16))
        width = state["next_observation_decoder.0.weight"].shape[1]
        if width == encoded_dim:
            inferred = 2
        elif width == input_dim + encoded_dim:
            inferred = 1
        else:
            raise ValueError("Unrecognized latent decoder input width: {}".format(width))
        if int(config.get("model_version", inferred)) != inferred:
            raise ValueError("Latent checkpoint metadata disagrees with decoder weights")
        config["model_version"] = inferred
        model = cls(**config)
        model.load_state_dict(state, strict=True)
        return model

    def model_config(self):
        return {
            "model_version": self.model_version,
            "input_dim": self.input_dim,
            "history_len": self.history_len,
            "explicit_dim": self.explicit_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": list(self.hidden_dims),
            "decoder_hidden_dims": list(self.decoder_hidden_dims),
            "com_cop_dim": self.com_cop_dim,
            "contact_dim": self.contact_dim,
            "normalization_epsilon": self.normalization_epsilon,
        }

