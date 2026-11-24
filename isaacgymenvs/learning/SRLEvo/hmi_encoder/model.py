"""Small causal TCN and explicit, checkpointed input calibration."""
import torch
from torch import nn
from torch.nn import functional as F


class FeatureNormalizer(nn.Module):
    def __init__(self, width, clip):
        super().__init__()
        self.register_buffer('mean', torch.zeros(width - 1))
        self.register_buffer('variance', torch.ones(width - 1))
        self.register_buffer('count', torch.tensor(0., dtype=torch.float64))
        self.clip = clip

    @torch.no_grad()
    def calibrate(self, features):
        data = features.reshape(-1, features.shape[-1])
        data = data[data[:, -1] > 0, :-1]
        if not len(data):
            return
        mean = data.mean(0)
        variance = data.var(0, unbiased=False)
        n = len(data)
        total = self.count + n
        delta = mean - self.mean
        self.variance.copy_((self.variance * self.count + variance * n
                             + delta.square() * self.count * n / total) / total)
        self.mean.add_(delta * (n / total))
        self.count.copy_(total)

    def forward(self, x):
        valid = x[..., -1:]
        data = ((x[..., :-1] - self.mean) / self.variance.clamp_min(1e-4).sqrt()).clamp(-self.clip, self.clip)
        return torch.cat((data * valid, valid), dim=-1)


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=2, dilation=dilation)
        self.norm = nn.LayerNorm(channels)  # channels only, independently at each time

    def forward(self, x):
        # Keep conv as the parameter owner: old checkpoint keys/shapes and
        # optimizer parameter ordering remain unchanged. For this two-tap
        # causal cross-correlation, y[t] = W[0] x[t-d] + W[1] x[t] + b.
        # Linear uses GPU matrix multiplication without needing cuDNN.
        current = x.transpose(1, 2)  # [batch, time, channels]
        delayed = F.pad(current, (0, 0, self.dilation, 0))[:, :current.shape[1], :]
        y = (F.linear(delayed, self.conv.weight[:, :, 0])
             + F.linear(current, self.conv.weight[:, :, 1], self.conv.bias))
        return x + F.elu(self.norm(y)).transpose(1, 2)


class HMIEncoder(nn.Module):
    def __init__(self, config, feature_dim=41):
        super().__init__()
        self.normalizer = FeatureNormalizer(feature_dim, config.input_clip)
        self.project = nn.Linear(feature_dim, config.channels)
        self.temporal = nn.Sequential(*(CausalBlock(config.channels, d) for d in config.dilations))
        self.bottleneck = nn.Sequential(nn.Linear(config.channels, config.channels), nn.ELU(),
                                        nn.Linear(config.channels, config.latent_dim), nn.Tanh())
        self.predictor = nn.Sequential(nn.Linear(config.latent_dim, 64), nn.ELU(),
                                       nn.Linear(64, 3 * len(config.horizons)))

    def forward(self, history):
        x = self.project(self.normalizer(history)).transpose(1, 2)
        z = self.bottleneck(self.temporal(x)[..., -1])
        return z, self.predictor(z).reshape(-1, self.predictor[-1].out_features // 3, 3)
