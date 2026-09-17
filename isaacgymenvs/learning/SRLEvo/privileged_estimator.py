import torch
import torch.nn as nn


class PrivilegedStateEstimator(nn.Module):
    """Estimate current root height and local root velocity from history."""

    def __init__(self, input_dim=26, history_len=64, output_dim=4):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.output_dim = int(output_dim)

        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 64, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.input_dim, self.history_len)
            encoded_dim = self.encoder(dummy).shape[-1]

        self.head = nn.Sequential(
            nn.Linear(encoded_dim + self.input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, self.output_dim),
        )

        self.register_buffer("input_mean", torch.zeros(self.input_dim))
        self.register_buffer("input_std", torch.ones(self.input_dim))
        self.register_buffer("target_mean", torch.zeros(self.output_dim))
        self.register_buffer("target_std", torch.ones(self.output_dim))

    def set_normalization(self, input_mean, input_std, target_mean, target_std):
        self.input_mean.copy_(input_mean)
        self.input_std.copy_(input_std.clamp_min(1e-6))
        self.target_mean.copy_(target_mean)
        self.target_std.copy_(target_std.clamp_min(1e-6))

    def forward_normalized(self, history):
        if history.ndim != 3:
            raise ValueError("history must have shape [batch, history_len, input_dim]")
        if history.shape[1:] != (self.history_len, self.input_dim):
            raise ValueError(
                "expected history shape [batch, {}, {}], got {}".format(
                    self.history_len, self.input_dim, tuple(history.shape)
                )
            )
        normalized = (history - self.input_mean) / self.input_std
        current = normalized[:, -1]
        encoded = self.encoder(normalized.transpose(1, 2))
        return self.head(torch.cat((encoded, current), dim=-1))

    def forward(self, history):
        """Return [root height, body-frame vx, vy, vz] in raw observation units."""
        normalized_output = self.forward_normalized(history)
        return normalized_output * self.target_std + self.target_mean


def load_privileged_estimator(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PrivilegedStateEstimator(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
