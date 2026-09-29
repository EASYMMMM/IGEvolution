import numpy as np
import torch

from isaacgymenvs.learning import common_player
from isaacgymenvs.learning.SRLEvo.concurrent_v2_estimator import (
    ConcurrentV2Estimator,
)
from isaacgymenvs.learning.SRLEvo.srl_bot_concurrent import (
    SRL_Bot_Concurrent_Agent,
    TARGET_NAMES,
)
from rl_games.algos_torch import torch_ext


class SRL_Bot_Concurrent_v2_Agent(SRL_Bot_Concurrent_Agent):
    """PPO with five frames of per-timestep estimated explicit state."""

    architecture = "concurrent_v2"
    architecture_version = 1
    frame_stack = 5
    full_frame_dim = 30
    deployable_frame_dim = 26
    deployable_obs_dim = 133
    actor_obs_dim = 153
    critic_obs_dim = 153
    estimator_input_dim = 26
    estimator_output_dim = 4

    _CONFIG_ALIASES = {
        "estimator_history_len": "concurrent_estimator_history_len",
        "estimator_hidden_dims": "concurrent_estimator_hidden_dims",
        "estimator_learning_rate": "concurrent_estimator_learning_rate",
        "estimator_mini_epochs": "concurrent_estimator_mini_epochs",
        "estimator_minibatch_size": "concurrent_estimator_minibatch_size",
        "estimator_samples_per_step": "concurrent_estimator_samples_per_step",
        "estimator_grad_norm": "concurrent_estimator_grad_norm",
        "estimator_update_normalization": "concurrent_estimator_update_normalization",
        "bootstrap_levels": "concurrent_bootstrap_levels",
        "bootstrap_initial_level": "concurrent_bootstrap_initial_level",
        "bootstrap_auto_advance": "concurrent_bootstrap_auto_advance",
        "bootstrap_warmup_epochs": "concurrent_bootstrap_warmup_epochs",
        "bootstrap_min_epochs_per_level": "concurrent_bootstrap_min_epochs_per_level",
        "bootstrap_min_completed_episodes": "concurrent_bootstrap_min_completed_episodes",
        "bootstrap_metric_window": "concurrent_bootstrap_metric_window",
        "bootstrap_ground_truth_min_length": "concurrent_bootstrap_ground_truth_min_length",
        "bootstrap_promote_length_ratio": "concurrent_bootstrap_promote_length_ratio",
        "bootstrap_promote_reward_ratio": "concurrent_bootstrap_promote_reward_ratio",
        "bootstrap_demote_length_ratio": "concurrent_bootstrap_demote_length_ratio",
        "bootstrap_max_normalized_mse": "concurrent_bootstrap_max_normalized_mse",
        "task_training_stage": "concurrent_task_training_stage",
        "bootstrap_reset_on_stage_change": "concurrent_bootstrap_reset_on_stage_change",
        "bootstrap_reset_on_restore": "concurrent_bootstrap_reset_on_restore",
        "bootstrap_reset_probability": "concurrent_bootstrap_reset_probability",
        "bootstrap_clear_metrics_on_reset": "concurrent_bootstrap_clear_metrics_on_reset",
        "eval_use_estimator": "concurrent_eval_use_estimator",
    }

    def __init__(self, base_name, params):
        config = params["config"]
        for source, destination in self._CONFIG_ALIASES.items():
            source = "concurrent_v2_" + source
            if source in config:
                config[destination] = config[source]

        super().__init__(base_name, params)

        estimator_config = self.concurrent_estimator.model_config()
        self.concurrent_estimator = ConcurrentV2Estimator(
            input_dim=estimator_config["input_dim"],
            history_len=estimator_config["history_len"],
            output_dim=estimator_config["output_dim"],
            hidden_dims=tuple(estimator_config["hidden_dims"]),
            normalization_epsilon=estimator_config["normalization_epsilon"],
        ).to(self.ppo_device)
        self.concurrent_estimator_optimizer = torch.optim.Adam(
            self.concurrent_estimator.parameters(),
            lr=float(config.get("concurrent_v2_estimator_learning_rate", 1e-3)),
            eps=1e-8,
        )
        self.temporal_loss_coef = float(
            config.get("concurrent_v2_estimator_temporal_loss_coef", 0.05)
        )
        if self.temporal_loss_coef < 0.0:
            raise ValueError("concurrent_v2 temporal loss coefficient must be >= 0")

        self.actor_frame_history = None
        self.mirrored_actor_frame_history = None
        self.previous_estimate = None
        self.previous_truth = None
        self.temporal_valid = None
        self.estimator_training_previous_estimates = []
        self.estimator_training_previous_targets = []
        self.estimator_training_temporal_valid = []
        print(
            "Concurrent_v2 enabled: actor=5x30+3, critic=5x30+3, "
            "estimator=10x26->4, temporal_coef={:.3g}".format(
                self.temporal_loss_coef
            )
        )

    def _sample_episode_modes(self, env_ids):
        """Assign an exact EST quota to each newly sampled environment batch."""
        if env_ids.numel() == 0:
            return
        count = int(round(self.bootstrap_probability * env_ids.numel()))
        sampled = torch.zeros(env_ids.numel(), device=self.ppo_device, dtype=torch.bool)
        if count > 0:
            permutation = torch.randperm(env_ids.numel(), device=self.ppo_device)
            sampled[permutation[:count]] = True
        self.episode_uses_estimator[env_ids] = sampled

    def _update_actor_frame_history(
        self, current_frame, mirrored_current_frame, reset_ids
    ):
        if self.actor_frame_history is None:
            self.actor_frame_history = current_frame.unsqueeze(1).expand(
                -1, self.frame_stack, -1
            ).clone()
            self.mirrored_actor_frame_history = (
                mirrored_current_frame.unsqueeze(1)
                .expand(-1, self.frame_stack, -1)
                .clone()
            )
            return

        self.actor_frame_history[:, 1:] = self.actor_frame_history[:, :-1].clone()
        self.actor_frame_history[:, 0] = current_frame
        self.mirrored_actor_frame_history[:, 1:] = (
            self.mirrored_actor_frame_history[:, :-1].clone()
        )
        self.mirrored_actor_frame_history[:, 0] = mirrored_current_frame
        if reset_ids.numel() > 0:
            self.actor_frame_history[reset_ids] = current_frame[
                reset_ids
            ].unsqueeze(1).expand(-1, self.frame_stack, -1)
            self.mirrored_actor_frame_history[reset_ids] = mirrored_current_frame[
                reset_ids
            ].unsqueeze(1).expand(-1, self.frame_stack, -1)

    def _validate_layout(self, observation, mirrored_observation, states):
        if self._observation_layout_validated:
            return
        observation_error = (observation - states).abs().max().item()
        command_error = (observation[:, -3:] - states[:, -3:]).abs().max().item()
        truth = states[:, : self.estimator_output_dim]
        mirrored_truth = mirrored_observation[:, : self.estimator_output_dim]
        mirror_error = (mirrored_truth - truth * self._mirror_sign).abs().max().item()
        if max(observation_error, command_error, mirror_error) > 1e-5:
            raise RuntimeError(
                "Concurrent_v2 observation layout mismatch: obs={:.3e}, "
                "command={:.3e}, mirror={:.3e}".format(
                    observation_error, command_error, mirror_error
                )
            )
        self._observation_layout_validated = True
        print("Concurrent_v2 layout verified: five 30-D frames + three commands")

    def _build_concurrent_actor_observation(self, raw_obs, done_env_ids):
        observation = raw_obs["obs"]
        mirrored_observation = raw_obs["obs_mirrored"]
        states = raw_obs["states"]
        if observation.shape[1] != self.actor_obs_dim:
            raise RuntimeError("Concurrent_v2 expected a 153-D raw observation")
        if states.shape[1] != self.critic_obs_dim:
            raise RuntimeError("Concurrent_v2 expected a 153-D critic state")
        self._validate_layout(observation, mirrored_observation, states)

        reset_ids = torch.as_tensor(
            done_env_ids, device=self.ppo_device, dtype=torch.long
        ).reshape(-1)
        current_proprio = observation[
            :, self.estimator_output_dim : self.full_frame_dim
        ]
        mirrored_current_proprio = mirrored_observation[
            :, self.estimator_output_dim : self.full_frame_dim
        ]
        self._update_history(current_proprio, reset_ids)

        with torch.no_grad():
            estimate = self.concurrent_estimator(self.estimator_history)
        truth = states[:, : self.estimator_output_dim]
        selected = torch.where(
            self.episode_uses_estimator.unsqueeze(1), estimate, truth
        )
        mirrored_selected = torch.where(
            self.episode_uses_estimator.unsqueeze(1),
            estimate * self._mirror_sign,
            mirrored_observation[:, : self.estimator_output_dim],
        )
        current_frame = torch.cat((selected, current_proprio), dim=-1)
        mirrored_current_frame = torch.cat(
            (mirrored_selected, mirrored_current_proprio), dim=-1
        )
        self._update_actor_frame_history(
            current_frame, mirrored_current_frame, reset_ids
        )

        actor_observation = torch.cat(
            (self.actor_frame_history.reshape(observation.shape[0], -1), observation[:, -3:]),
            dim=-1,
        )
        mirrored_actor_observation = torch.cat(
            (
                self.mirrored_actor_frame_history.reshape(observation.shape[0], -1),
                mirrored_observation[:, -3:],
            ),
            dim=-1,
        )

        if self.previous_estimate is None:
            self.previous_estimate = estimate.clone()
            self.previous_truth = truth.clone()
            self.temporal_valid = torch.zeros(
                observation.shape[0], device=self.ppo_device, dtype=torch.bool
            )
        if reset_ids.numel() > 0:
            self.temporal_valid[reset_ids] = False

        sample_count = min(self.estimator_samples_per_step, observation.shape[0])
        sample_ids = torch.randint(
            0, observation.shape[0], (sample_count,), device=self.ppo_device
        )
        self.estimator_training_history.append(
            self.estimator_history[sample_ids].detach().clone()
        )
        self.estimator_training_targets.append(truth[sample_ids].detach().clone())
        self.estimator_training_previous_estimates.append(
            self.previous_estimate[sample_ids].detach().clone()
        )
        self.estimator_training_previous_targets.append(
            self.previous_truth[sample_ids].detach().clone()
        )
        self.estimator_training_temporal_valid.append(
            self.temporal_valid[sample_ids].detach().clone()
        )
        self.previous_estimate.copy_(estimate)
        self.previous_truth.copy_(truth)
        self.temporal_valid.fill_(True)
        if reset_ids.numel() > 0:
            self.temporal_valid[reset_ids] = False
        return actor_observation, mirrored_actor_observation, estimate, truth

    def play_steps(self):
        self.estimator_training_previous_estimates = []
        self.estimator_training_previous_targets = []
        self.estimator_training_temporal_valid = []
        return super().play_steps()

    def _train_concurrent_estimator(self):
        history = torch.cat(self.estimator_training_history, dim=0)
        targets = torch.cat(self.estimator_training_targets, dim=0)
        previous_estimates = torch.cat(
            self.estimator_training_previous_estimates, dim=0
        )
        previous_targets = torch.cat(
            self.estimator_training_previous_targets, dim=0
        )
        temporal_valid = torch.cat(
            self.estimator_training_temporal_valid, dim=0
        )
        if self.update_estimator_normalization:
            self.concurrent_estimator.update_normalization(history, targets)

        self.concurrent_estimator.train()
        explicit_losses = []
        temporal_losses = []
        total = history.shape[0]
        for _ in range(self.estimator_mini_epochs):
            permutation = torch.randperm(total, device=self.ppo_device)
            for start in range(0, total, self.estimator_minibatch_size):
                ids = permutation[start : start + self.estimator_minibatch_size]
                losses = self.concurrent_estimator.normalized_losses(
                    history[ids],
                    targets[ids],
                    previous_estimate=previous_estimates[ids],
                    previous_target=previous_targets[ids],
                    temporal_valid=temporal_valid[ids],
                )
                total_loss = (
                    losses["explicit"]
                    + self.temporal_loss_coef * losses["temporal"]
                )
                self.concurrent_estimator_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.concurrent_estimator.parameters(), self.estimator_grad_norm
                )
                self.concurrent_estimator_optimizer.step()
                explicit_losses.append(float(losses["explicit"].detach()))
                temporal_losses.append(float(losses["temporal"].detach()))
        self.concurrent_estimator.eval()
        self.concurrent_metrics["temporal_loss"] = (
            float(np.mean(temporal_losses)) if temporal_losses else 0.0
        )
        return float(np.mean(explicit_losses)) if explicit_losses else 0.0

    def _write_concurrent_metrics(self):
        frame = self.frame // self.num_agents
        self.writer.add_scalar(
            "concurrent_v2/bootstrap_probability",
            self.concurrent_metrics["bootstrap_probability"],
            frame,
        )
        self.writer.add_scalar(
            "concurrent_v2/estimated_env_fraction",
            self.concurrent_metrics["estimated_env_fraction"],
            frame,
        )
        self.writer.add_scalar(
            "concurrent_v2/estimator_normalized_mse",
            self.concurrent_metrics["normalized_mse"],
            frame,
        )
        for index, name in enumerate(TARGET_NAMES):
            self.writer.add_scalar(
                "concurrent_v2/{}_mae".format(name),
                self.concurrent_metrics["mae"][index].item(),
                frame,
            )
            self.writer.add_scalar(
                "concurrent_v2/{}_rmse".format(name),
                self.concurrent_metrics["rmse"][index].item(),
                frame,
            )
        for prefix, lengths, rewards in (
            ("ground_truth", self.gt_episode_lengths, self.gt_episode_rewards),
            (
                "estimated",
                self.estimated_episode_lengths,
                self.estimated_episode_rewards,
            ),
        ):
            if lengths:
                self.writer.add_scalar(
                    "concurrent_v2/{}_episode_length".format(prefix),
                    self._mean_or_zero(lengths),
                    frame,
                )
                self.writer.add_scalar(
                    "concurrent_v2/{}_episode_reward".format(prefix),
                    self._mean_or_zero(rewards),
                    frame,
                )
        self.writer.add_scalar(
            "concurrent_v2/temporal_loss",
            self.concurrent_metrics.get("temporal_loss", 0.0),
            frame,
        )

    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        state["concurrent_v2_estimator"] = state.pop("concurrent_estimator")
        state["concurrent_v2_estimator_optimizer"] = state.pop(
            "concurrent_estimator_optimizer"
        )
        state["concurrent_v2_estimator_config"] = state.pop(
            "concurrent_estimator_config"
        )
        state["concurrent_v2_bootstrap_state"] = state.pop(
            "concurrent_bootstrap_state"
        )
        state["concurrent_v2_metadata"] = {
            "architecture": self.architecture,
            "version": self.architecture_version,
            "actor_obs_dim": self.actor_obs_dim,
            "critic_obs_dim": self.critic_obs_dim,
            "frame_stack": self.frame_stack,
            "full_frame_dim": self.full_frame_dim,
            "explicit_dim": self.estimator_output_dim,
            "estimator_history_len": self.concurrent_estimator.history_len,
            "task_training_stage": self.concurrent_task_training_stage,
        }
        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        metadata = weights.get("concurrent_v2_metadata")
        if metadata is None or metadata.get("architecture") != self.architecture:
            raise RuntimeError(
                "Concurrent_v2 only accepts checkpoints created by Concurrent_v2"
            )
        translated = dict(weights)
        translated["concurrent_estimator"] = translated[
            "concurrent_v2_estimator"
        ]
        if "concurrent_v2_estimator_optimizer" in translated:
            translated["concurrent_estimator_optimizer"] = translated[
                "concurrent_v2_estimator_optimizer"
            ]
        translated["concurrent_estimator_config"] = translated.get(
            "concurrent_v2_estimator_config", {}
        )
        translated["concurrent_bootstrap_state"] = translated.get(
            "concurrent_v2_bootstrap_state", {}
        )
        return super().set_full_state_weights(translated, set_epoch=set_epoch)


class SRL_Bot_Concurrent_v2_Player(common_player.CommonPlayer):
    architecture = "concurrent_v2"
    frame_stack = 5
    full_frame_dim = 30
    explicit_dim = 4
    input_dim = 26
    raw_obs_dim = 153

    def __init__(self, params):
        super().__init__(params)
        config = params["config"]
        self.eval_use_estimator = bool(
            config.get("concurrent_v2_eval_use_estimator", True)
        )
        self.estimator = ConcurrentV2Estimator(
            input_dim=self.input_dim,
            history_len=int(config.get("concurrent_v2_estimator_history_len", 10)),
            output_dim=self.explicit_dim,
            hidden_dims=tuple(
                int(value)
                for value in config.get(
                    "concurrent_v2_estimator_hidden_dims", [256, 128, 64]
                )
            ),
        ).to(self.device)
        self.estimator.eval()
        self.estimator_history = None
        self.actor_frame_history = None
        self.pending_reset_ids = None

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        metadata = checkpoint.get("concurrent_v2_metadata", {})
        if metadata.get("architecture") != self.architecture:
            raise RuntimeError("checkpoint is not a Concurrent_v2 checkpoint")
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.model.running_mean_std.load_state_dict(
                checkpoint["running_mean_std"]
            )
        self.estimator.load_state_dict(checkpoint["concurrent_v2_estimator"])
        env_state = checkpoint.get("env_state")
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)
        print("Loaded Concurrent_v2 checkpoint '{}'".format(fn))

    def _env_reset_done(self):
        obs, done_env_ids = super()._env_reset_done()
        self.pending_reset_ids = torch.as_tensor(
            done_env_ids, device=self.device, dtype=torch.long
        ).reshape(-1)
        return obs, done_env_ids

    @staticmethod
    def _shift_and_append(history, current, reset_ids, length):
        if history is None:
            return current.unsqueeze(1).expand(-1, length, -1).clone()
        history[:, 1:] = history[:, :-1].clone()
        history[:, 0] = current
        if reset_ids is not None and reset_ids.numel() > 0:
            history[reset_ids] = current[reset_ids].unsqueeze(1).expand(
                -1, length, -1
            )
        return history

    def _actor_observation(self, observation):
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.shape[1] != self.raw_obs_dim:
            raise RuntimeError("Concurrent_v2 player expected 153-D observations")
        reset_ids = self.pending_reset_ids
        current_proprio = observation[:, self.explicit_dim : self.full_frame_dim]

        if self.estimator_history is None:
            self.estimator_history = current_proprio.unsqueeze(1).expand(
                -1, self.estimator.history_len, -1
            ).clone()
        else:
            self.estimator_history[:, :-1] = self.estimator_history[:, 1:].clone()
            self.estimator_history[:, -1] = current_proprio
            if reset_ids is not None and reset_ids.numel() > 0:
                self.estimator_history[reset_ids] = current_proprio[
                    reset_ids
                ].unsqueeze(1).expand(-1, self.estimator.history_len, -1)

        estimate = self.estimator(self.estimator_history)
        explicit = estimate if self.eval_use_estimator else observation[:, :4]
        current_frame = torch.cat((explicit, current_proprio), dim=-1)
        self.actor_frame_history = self._shift_and_append(
            self.actor_frame_history,
            current_frame,
            reset_ids,
            self.frame_stack,
        )
        self.pending_reset_ids = None
        return torch.cat(
            (
                self.actor_frame_history.reshape(observation.shape[0], -1),
                observation[:, -3:],
            ),
            dim=-1,
        )

    def get_action(self, obs_dict, is_determenistic=False):
        with torch.no_grad():
            observation = self._actor_observation(obs_dict["obs"])
        actor_obs = dict(obs_dict)
        actor_obs["obs"] = observation
        return common_player.CommonPlayer.get_action(
            self, actor_obs, is_determenistic
        )
