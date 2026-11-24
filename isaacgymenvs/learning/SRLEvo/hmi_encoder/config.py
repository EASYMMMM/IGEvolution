"""Versioned, optional encoder configuration. No Isaac Gym dependency."""
from dataclasses import asdict, dataclass, fields


@dataclass
class EncoderConfig:
    enabled: bool = False
    layout: str = 'srl39_zero_moments_v1'
    history_length: int = 64
    channels: int = 32
    dilations: tuple = (1, 2, 4, 8, 16, 32)
    latent_dim: int = 15
    horizons: tuple = (0, 8, 16)
    horizon_weights: tuple = (1.0, 0.5, 0.25)
    target_scale: tuple = (1.0, 1.0, 1.0)
    auxiliary_coef: float = 1.0
    learning_rate: float = 3e-4
    humanoid_lr_scale: float = 0.3
    microbatch_size: int = 2048
    warmup_rollouts: int = 100
    ramp_rollouts: int = 300
    calibration_rollouts: int = 10
    input_clip: float = 5.0

    @classmethod
    def from_dict(cls, config):
        values = dict(config or {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown HMI encoder settings: {sorted(unknown)}')
        for key in ('dilations', 'horizons', 'horizon_weights', 'target_scale'):
            if key in values:
                values[key] = tuple(values[key])
        result = cls(**values)
        result.validate()
        return result

    def validate(self):
        if self.layout != 'srl39_zero_moments_v1' or self.latent_dim != 15:
            raise ValueError('This layout reserves exactly 15 slots. Implement a new layout adapter/version to change it.')
        if self.history_length < 2 or self.channels < 1 or not self.dilations or min(self.dilations) < 1:
            raise ValueError('Invalid encoder architecture')
        if 1 + sum(self.dilations) < self.history_length:
            raise ValueError('TCN receptive field does not cover history_length')
        if not self.horizons or self.horizons[0] != 0 or any(h < 0 for h in self.horizons):
            raise ValueError('horizons must start at zero and be nonnegative')
        if len(set(self.horizons)) != len(self.horizons) or len(self.horizons) != len(self.horizon_weights):
            raise ValueError('Invalid prediction horizons/weights')
        if len(self.target_scale) != 3 or min(self.target_scale) <= 0 or min(self.horizon_weights) < 0:
            raise ValueError('Invalid target scales/weights')
        if self.microbatch_size < 1 or self.learning_rate <= 0 or self.humanoid_lr_scale <= 0:
            raise ValueError('Invalid optimizer settings')
        if not 1 <= self.calibration_rollouts <= self.warmup_rollouts or self.ramp_rollouts < 1:
            raise ValueError('Require 1 <= calibration_rollouts <= warmup_rollouts and ramp_rollouts >= 1')
        if self.input_clip <= 0 or self.auxiliary_coef < 0:
            raise ValueError('Invalid normalization/loss settings')

    def to_dict(self):
        return asdict(self)
