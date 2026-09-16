import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcurrentPrivilegedEstimator(nn.Module):
    """MLP estimator for current height and body-frame linear velocity."""

    def __init__(
        self,
        input_dim=26,
        history_len=10,
        output_dim=4,
        hidden_dims=(256, 128, 64),
        normalization_epsilon=1e-4,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.output_dim = int(output_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.normalization_epsilon = float(normalization_epsilon)

        layers = []
        previous_dim = self.input_dim * self.history_len
        for hidden_dim in self.hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, self.output_dim))
        self.network = nn.Sequential(*layers)

        self.register_buffer("input_mean", torch.zeros(self.input_dim))
        self.register_buffer("input_var", torch.ones(self.input_dim))
        self.register_buffer("input_count", torch.tensor(self.normalization_epsilon))
        self.register_buffer("target_mean", torch.zeros(self.output_dim))
        self.register_buffer("target_var", torch.ones(self.output_dim))
        self.register_buffer("target_count", torch.tensor(self.normalization_epsilon))

    @staticmethod
    def _updated_moments(mean, var, count, batch):
        batch = batch.float()
        batch_count = torch.tensor(
            float(batch.shape[0]), device=batch.device, dtype=batch.dtype
        )
        if batch_count.item() == 0:
            return mean, var, count

        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        delta = batch_mean - mean
        total_count = count + batch_count
        new_mean = mean + delta * batch_count / total_count
        first_moment = var * count
        second_moment = batch_var * batch_count
        correction = delta.square() * count * batch_count / total_count
        new_var = (first_moment + second_moment + correction) / total_count
        return new_mean, new_var.clamp_min(1e-8), total_count

    @torch.no_grad()
    def update_normalization(self, history, target):
        self._validate(history, target)
        flat_input = history.reshape(-1, self.input_dim)
        input_stats = self._updated_moments(
            self.input_mean, self.input_var, self.input_count, flat_input
        )
        target_stats = self._updated_moments(
            self.target_mean, self.target_var, self.target_count, target
        )
        self.input_mean.copy_(input_stats[0])
        self.input_var.copy_(input_stats[1])
        self.input_count.copy_(input_stats[2])
        self.target_mean.copy_(target_stats[0])
        self.target_var.copy_(target_stats[1])
        self.target_count.copy_(target_stats[2])

    def _validate(self, history, target=None):
        expected = (self.history_len, self.input_dim)
        if history.ndim != 3 or tuple(history.shape[1:]) != expected:
            raise ValueError(
                "expected history [batch, {}, {}], got {}".format(
                    self.history_len, self.input_dim, tuple(history.shape)
                )
            )
        if target is not None:
            if target.ndim != 2 or target.shape[1] != self.output_dim:
                raise ValueError(
                    "expected target [batch, {}], got {}".format(
                        self.output_dim, tuple(target.shape)
                    )
                )

    def input_std(self):
        return torch.sqrt(self.input_var.clamp_min(1e-8))

    def target_std(self):
        return torch.sqrt(self.target_var.clamp_min(1e-8))

    def forward_normalized(self, history):
        self._validate(history)
        normalized = (history - self.input_mean) / self.input_std()
        return self.network(normalized.reshape(history.shape[0], -1))

    def forward(self, history):
        normalized_output = self.forward_normalized(history)
        return normalized_output * self.target_std() + self.target_mean

    def normalized_mse(self, history, target):
        self._validate(history, target)
        normalized_target = (target - self.target_mean) / self.target_std()
        return F.mse_loss(self.forward_normalized(history), normalized_target)

    def model_config(self):
        return {
            "input_dim": self.input_dim,
            "history_len": self.history_len,
            "output_dim": self.output_dim,
            "hidden_dims": list(self.hidden_dims),
            "normalization_epsilon": self.normalization_epsilon,
        }
