"""Optional history, supervised labels and checkpoint lifecycle.

The legacy environment is read through SRL39Layout; no environment mutation.
Histories are ephemeral and reset on process restore (simulator state is not
replayed). Learned weights, normalization and curriculum ARE checkpointed.
"""
import torch
from .config import EncoderConfig
from .layout import SRL39Layout
from .model import HMIEncoder


class EncoderRuntime:
    checkpoint_key = 'hmi_encoder'
    version = 1

    def __init__(self, config, device, num_envs, control_dt, training=True):
        self.config = config
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.control_dt = float(control_dt)
        self.model = HMIEncoder(config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate) if training else None
        self.completed_rollouts = 0
        self.alpha = 0.0
        self.slots = torch.tensor(SRL39Layout.slots, device=device)
        self.history = torch.zeros(num_envs, config.history_length, SRL39Layout.feature_dim, device=device)
        self.mirrored_history = torch.zeros_like(self.history)
        self.history_episode = torch.full(self.history.shape[:2], -1, dtype=torch.long, device=device)
        self.episode = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.previous_yaw = torch.zeros(num_envs, device=device)
        self.has_label = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.sequence = None

    def reset(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        ids = ids.to(self.device).flatten().long()
        self.history[ids] = 0
        self.mirrored_history[ids] = 0
        self.history_episode[ids] = -1
        self.has_label[ids] = False
        self.episode[ids] += 1

    def begin_rollout(self, horizon):
        c = self.config
        ramp_step = self.completed_rollouts - c.warmup_rollouts + 1
        self.alpha = min(1.0, max(0.0, ramp_step / c.ramp_rollouts))
        if max(c.horizons) >= horizon:
            raise ValueError('PPO horizon_length must exceed the largest encoder prediction horizon')
        self.horizon = horizon
        k = c.history_length
        shape = (k - 1 + horizon, self.num_envs, SRL39Layout.feature_dim)
        self.sequence = torch.empty(shape, device=self.device)
        self.mirrored_sequence = torch.empty_like(self.sequence)
        self.sequence_episode = torch.empty(shape[:2], dtype=torch.long, device=self.device)
        self.sequence[:k-1] = self.history[:, 1:].transpose(0, 1)
        self.mirrored_sequence[:k-1] = self.mirrored_history[:, 1:].transpose(0, 1)
        self.sequence_episode[:k-1] = self.history_episode[:, 1:].transpose(0, 1)
        # world vx, vy, human yaw rate, current robot yaw, valid human yaw rate
        self.label_states = torch.empty(horizon, self.num_envs, 5, device=self.device)

    @torch.no_grad()
    def observe(self, obs, mirrored_obs=None, reset_ids=None, step=None, env=None):
        if reset_ids is not None:
            self.reset(reset_ids)
        if step == 0 and torch.any(obs[:, self.slots] != 0):
            raise ValueError('Reserved moment slots are no longer zero; select a new encoder layout')
        frame = SRL39Layout.features(obs)
        mirrored = SRL39Layout.features(mirrored_obs) if mirrored_obs is not None else frame
        self.history = torch.cat((self.history[:, 1:], frame[:, None]), 1)
        self.mirrored_history = torch.cat((self.mirrored_history[:, 1:], mirrored[:, None]), 1)
        self.history_episode = torch.cat((self.history_episode[:, 1:], self.episode[:, None]), 1)
        if step is not None:
            index = self.config.history_length - 1 + step
            self.sequence[index] = frame
            self.mirrored_sequence[index] = mirrored
            self.sequence_episode[index] = self.episode
            vel, yaw, robot_yaw = SRL39Layout.label_state(env)
            vel, yaw, robot_yaw = vel.to(self.device), yaw.to(self.device), robot_yaw.to(self.device)
            delta = yaw - self.previous_yaw
            yaw_rate = torch.atan2(delta.sin(), delta.cos()) / self.control_dt
            self.label_states[step] = torch.cat((vel, yaw_rate[:, None], robot_yaw[:, None],
                                                 self.has_label[:, None].float()), -1)
            self.previous_yaw.copy_(yaw)
            self.has_label.fill_(True)

    def payload(self, history=None):
        history = self.history if history is None else history
        z, predictions = self.model(history)
        return (z, self.alpha, self.slots), predictions

    def finish_rollout(self):
        """Future labels use the anchor's heading, not the future robot heading."""
        t = torch.arange(self.horizon, device=self.device)
        current_episode = self.sequence_episode[self.config.history_length-1:]
        target_list, mask_list = [], []
        yaw = self.label_states[..., 3]
        for horizon in self.config.horizons:
            future = (t + horizon).clamp(max=self.horizon - 1)
            state = self.label_states[future]
            vx, vy = state[..., 0], state[..., 1]
            targets = torch.stack((yaw.cos() * vx + yaw.sin() * vy,
                                   -yaw.sin() * vx + yaw.cos() * vy, state[..., 2]), -1)
            valid = ((t + horizon < self.horizon)[:, None]
                     & (current_episode == current_episode[future]) & state[..., 4].bool())
            target_list.append(targets)
            mask_list.append(valid)
        self.targets = torch.stack(target_list, 2)
        self.target_mask = torch.stack(mask_list, 2)

    def sample(self, sample_ids, mirrored=False):
        # rl_games swap_and_flatten01 is ENV-major, then time.
        ids = sample_ids.flatten().long()
        env_ids, times = ids // self.horizon, ids % self.horizon
        indices = times[:, None] + torch.arange(self.config.history_length, device=self.device)
        source = self.mirrored_sequence if mirrored else self.sequence
        history = source[indices, env_ids[:, None]].clone()
        episodes = self.sequence_episode[indices, env_ids[:, None]]
        history *= (episodes == episodes[:, -1:])[..., None]
        targets = self.targets[times, env_ids].clone()
        if mirrored:
            targets[..., 1:] *= -1
        return history, targets, self.target_mask[times, env_ids]

    @torch.no_grad()
    def finish_update(self):
        # Calibration occurs only between PPO iterations, while alpha == 0.
        if self.completed_rollouts < self.config.calibration_rollouts:
            start = self.config.history_length - 1
            self.model.normalizer.calibrate(self.sequence[start:])
            self.model.normalizer.calibrate(self.mirrored_sequence[start:])
        self.completed_rollouts += 1

    def checkpoint(self):
        return {'version': self.version, 'config': self.config.to_dict(),
                'control_dt': self.control_dt, 'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
                'completed_rollouts': self.completed_rollouts, 'alpha': self.alpha}

    def load(self, state, resume_optimizer=True):
        saved_config = EncoderConfig.from_dict(state['config'])
        # These knobs do not change inference semantics and may be edited on resume.
        editable = {'microbatch_size', 'learning_rate', 'humanoid_lr_scale', 'auxiliary_coef',
                    'horizon_weights', 'target_scale'}
        changed = [key for key, value in saved_config.to_dict().items()
                   if key not in editable and value != self.config.to_dict()[key]]
        if state['version'] != self.version or changed:
            raise ValueError(f'Encoder checkpoint/config mismatch ({changed}). Restore saved settings or explicitly migrate the encoder.')
        if abs(state['control_dt'] - self.control_dt) > 1e-8:
            raise ValueError('Encoder checkpoint control period mismatch')
        self.model.load_state_dict(state['model'], strict=True)
        if resume_optimizer and self.optimizer is not None:
            if state.get('optimizer') is None:
                raise ValueError('Encoder training checkpoint lacks optimizer state')
            self.optimizer.load_state_dict(state['optimizer'])
            for group in self.optimizer.param_groups:
                group['lr'] = self.config.learning_rate
        self.completed_rollouts = int(state['completed_rollouts'])
        self.alpha = float(state['alpha'])
        if not 0 <= self.alpha <= 1 or self.completed_rollouts < 0:
            raise ValueError('Invalid encoder curriculum state')
        self.reset()


def create_runtime(config, env, device, training=True):
    config = EncoderConfig.from_dict(config)
    if not config.enabled:
        return None
    SRL39Layout.validate_env(env)
    return EncoderRuntime(config, device, env.num_envs, env.control_dt, training)


def check_checkpoint_mode(runtime, checkpoint):
    if EncoderRuntime.checkpoint_key in checkpoint and runtime is None:
        raise ValueError('Checkpoint contains an HMI encoder. Enable train.params.config.hmi_encoder.enabled; refusing to silently discard it.')
