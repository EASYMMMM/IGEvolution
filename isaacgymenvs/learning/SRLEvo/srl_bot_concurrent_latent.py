import json
from pathlib import Path

import numpy as np
import torch

from isaacgymenvs.learning import common_player
from isaacgymenvs.learning.SRLEvo.concurrent_latent_models import (
    ConcurrentLatentEncoder,
)
from isaacgymenvs.learning.SRLEvo.srl_bot_concurrent import (
    SRL_Bot_Concurrent_Agent,
)
from rl_games.algos_torch import torch_ext


LATENT_TARGET_NAMES = ("root_height", "local_vx", "local_vy", "local_vz")


class SRL_Bot_ConcurrentLatent_Agent(SRL_Bot_Concurrent_Agent):
    """PPO agent using explicit state estimates plus a learned causal latent."""

    deployable_obs_dim = 133
    actor_obs_dim = 153
    critic_obs_dim = 153
    estimator_input_dim = 26
    estimator_output_dim = 4
    latent_dim = 16
    auxiliary_slot_dim = 16
    com_cop_dim = 3
    contact_dim = 2

    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        config = params["config"]
        history_len = int(config.get("latent_history_len", 10))
        hidden_dims = tuple(
            int(v) for v in config.get("latent_encoder_hidden_dims", [256, 128, 64])
        )
        decoder_hidden_dims = tuple(
            int(v) for v in config.get("latent_decoder_hidden_dims", [128, 128])
        )
        configured_latent_dim = int(config.get("latent_dim", self.latent_dim))
        if configured_latent_dim != self.latent_dim:
            raise ValueError(
                "task/network layout currently requires latent_dim={}, got {}".format(
                    self.latent_dim, configured_latent_dim
                )
            )

        self.concurrent_estimator = ConcurrentLatentEncoder(
            input_dim=self.estimator_input_dim,
            history_len=history_len,
            explicit_dim=self.estimator_output_dim,
            latent_dim=self.latent_dim,
            hidden_dims=hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            com_cop_dim=self.com_cop_dim,
            contact_dim=self.contact_dim,
        ).to(self.ppo_device)
        self.concurrent_estimator_optimizer = torch.optim.Adam(
            self.concurrent_estimator.parameters(),
            lr=float(config.get("latent_learning_rate", 1e-3)),
            eps=1e-8,
        )
        self.estimator_mini_epochs = int(config.get("latent_mini_epochs", 2))
        self.estimator_minibatch_size = int(
            config.get("latent_minibatch_size", 8192)
        )
        if self.estimator_minibatch_size < 2:
            raise ValueError("latent_minibatch_size must fit an original/mirror pair")
        self.estimator_samples_per_step = int(
            config.get("latent_samples_per_step", 256)
        )
        self.estimator_grad_norm = float(config.get("latent_grad_norm", 1.0))
        self.update_estimator_normalization = bool(
            config.get("latent_update_normalization", True)
        )

        self.use_next_observation_loss = bool(
            config.get("latent_use_next_observation_loss", True)
        )
        self.use_com_cop_auxiliary = bool(
            config.get("latent_use_com_cop_auxiliary", False)
        )
        self.use_contact_auxiliary = bool(
            config.get("latent_use_contact_auxiliary", False)
        )
        if bool(config.get("latent_use_mirror_consistency", False)):
            raise ValueError("Latent v2 removes mirror-latent equality; set latent_use_mirror_consistency=False")
        self.est_only_after_epochs = int(config.get("latent_est_only_after_epochs", 600))
        self.est_only_min_task_stage = int(config.get("latent_est_only_min_task_stage", 3))
        if self.est_only_after_epochs < -1:
            raise ValueError("latent_est_only_after_epochs must be >= 0, or -1 to disable")
        self.latent_stage_start_epoch = int(self.epoch_num)
        self.latent_diagnostics_every = int(config.get("latent_diagnostics_every", 25))
        self.latent_diagnostics_samples = int(config.get("latent_diagnostics_samples", 1024))
        if self.latent_diagnostics_every < 1 or self.latent_diagnostics_samples < 2:
            raise ValueError("Latent diagnostics require interval >= 1 and samples >= 2")
        self.latent_diagnostics = {}
        self._est_only_announced = False
        self.loss_weights = {
            "explicit": float(config.get("latent_explicit_loss_coef", 1.0)),
            "next_observation": float(
                config.get("latent_next_observation_loss_coef", 1.0)
            ),
            "com_cop": float(config.get("latent_com_cop_loss_coef", 0.25)),
            "contact": float(config.get("latent_contact_loss_coef", 0.1)),
            "latent_l2": float(config.get("latent_l2_loss_coef", 1e-4)),
        }

        self.mirrored_estimator_history = None
        self.pending_decoder_history = None
        self.pending_decoder_explicit_target = None
        self.pending_decoder_mirrored_history = None
        self.latent_training_mirrored_history = []
        self.latent_training_com_cop = []
        self.latent_training_contacts = []
        self.latent_decoder_history = []
        self.latent_decoder_explicit_targets = []
        self.latent_decoder_next_targets = []
        self.latent_decoder_mirrored_history = []
        self.latent_decoder_mirrored_next_targets = []
        self.latent_metrics = {}
        print(
            "Concurrent latent encoder enabled: history={} input={} latent={} "
            "next_obs={} com_cop={} contact={}".format(
                history_len,
                self.estimator_input_dim,
                self.latent_dim,
                self.use_next_observation_loss,
                self.use_com_cop_auxiliary,
                self.use_contact_auxiliary,
            )
        )

    @property
    def est_only_active(self):
        after = getattr(self, "est_only_after_epochs", -1)
        return (after >= 0
                and self.concurrent_task_training_stage >= self.est_only_min_task_stage
                and int(self.epoch_num) - self.latent_stage_start_epoch >= after)

    @property
    def bootstrap_probability(self):
        return 1.0 if self.est_only_active else super().bootstrap_probability

    def _maybe_update_bootstrap_level(self, normalized_mse):
        if self.est_only_active:
            return
        super()._maybe_update_bootstrap_level(normalized_mse)

    def _update_mirrored_history(self, mirrored_deployable, reset_env_ids):
        current = mirrored_deployable[:, : self.estimator_input_dim]
        if self.mirrored_estimator_history is None:
            self.mirrored_estimator_history = current.unsqueeze(1).expand(
                -1, self.concurrent_estimator.history_len, -1
            ).clone()
            return
        self.mirrored_estimator_history[:, :-1] = self.mirrored_estimator_history[
            :, 1:
        ].clone()
        self.mirrored_estimator_history[:, -1] = current
        if reset_env_ids.numel() > 0:
            self.mirrored_estimator_history[reset_env_ids] = current[
                reset_env_ids
            ].unsqueeze(1).expand(-1, self.concurrent_estimator.history_len, -1)

    def _collect_decoder_transitions(self, current_frame, mirrored_frame, reset_ids):
        if self.pending_decoder_history is not None:
            valid = torch.ones(
                current_frame.shape[0], device=self.ppo_device, dtype=torch.bool
            )
            valid[reset_ids] = False
            valid_ids = valid.nonzero(as_tuple=False).reshape(-1)
            count = min(self.estimator_samples_per_step, valid_ids.numel())
            if count > 0:
                selected = valid_ids[
                    torch.randint(0, valid_ids.numel(), (count,), device=self.ppo_device)
                ]
                self.latent_decoder_history.append(
                    self.pending_decoder_history[selected].detach().clone()
                )
                self.latent_decoder_explicit_targets.append(
                    self.pending_decoder_explicit_target[selected].detach().clone()
                )
                self.latent_decoder_next_targets.append(
                    current_frame[selected].detach().clone()
                )
                self.latent_decoder_mirrored_history.append(
                    self.pending_decoder_mirrored_history[selected].detach().clone()
                )
                self.latent_decoder_mirrored_next_targets.append(
                    mirrored_frame[selected].detach().clone()
                )

    def _build_concurrent_actor_observation(self, raw_obs, done_env_ids):
        observation = raw_obs["obs"]
        mirrored_observation = raw_obs["obs_mirrored"]
        states = raw_obs["states"]
        if observation.shape[1] != self.actor_obs_dim:
            raise RuntimeError("expected 153-D ConcurrentLatent raw observation")
        if states.shape[1] != self.critic_obs_dim:
            raise RuntimeError("expected 153-D critic state")

        reset_ids = torch.as_tensor(
            done_env_ids, device=self.ppo_device, dtype=torch.long
        ).reshape(-1)
        deployable = observation[:, : self.deployable_obs_dim]
        mirrored_deployable = mirrored_observation[:, : self.deployable_obs_dim]
        raw_truth = observation[
            :, self.deployable_obs_dim : self.deployable_obs_dim + self.estimator_output_dim
        ]
        truth = states[:, : self.estimator_output_dim]
        labels = observation[:, -self.auxiliary_slot_dim :]
        com_cop_target = labels[:, : self.com_cop_dim]
        contact_target = labels[
            :, self.com_cop_dim : self.com_cop_dim + self.contact_dim
        ]

        if not self._observation_layout_validated:
            state_error = (raw_truth - truth).abs().max()
            mirror_error = (mirrored_observation[:, 133:137] - truth * self._mirror_sign).abs().max()
            if max(float(state_error.item()), float(mirror_error.item())) > 1e-5:
                raise RuntimeError(
                    "ConcurrentLatent truth/state layout mismatch: {:.3e}".format(
                        state_error.item()
                    )
                )
            self._observation_layout_validated = True
            print(
                "ConcurrentLatent observation layout verified: "
                "133 deployable + 4 truth + 16 supervision slots"
            )

        current_frame = deployable[:, : self.estimator_input_dim]
        self._collect_decoder_transitions(
            current_frame, mirrored_deployable[:, :self.estimator_input_dim], reset_ids
        )
        self._update_history(deployable, reset_ids)
        self._update_mirrored_history(mirrored_deployable, reset_ids)

        # Encoder updates are performed after the PPO epoch from sampled
        # histories, so rollout inference must not retain a horizon-long graph.
        with torch.no_grad():
            output = self.concurrent_estimator(self.estimator_history)
            mirrored_output = self.concurrent_estimator(self.mirrored_estimator_history)
        estimate = output["explicit"]
        latent = output["latent"]
        selected_explicit = torch.where(
            self.episode_uses_estimator.unsqueeze(1), estimate.detach(), truth
        )
        mirrored_explicit = torch.where(
            self.episode_uses_estimator.unsqueeze(1),
            mirrored_output["explicit"],
            mirrored_observation[
                :,
                self.deployable_obs_dim : self.deployable_obs_dim
                + self.estimator_output_dim,
            ],
        )
        actor_observation = torch.cat(
            (deployable, selected_explicit, latent.detach()), dim=-1
        )
        mirrored_actor_observation = torch.cat(
            (mirrored_deployable, mirrored_explicit, mirrored_output["latent"]), dim=-1
        )

        count = min(self.estimator_samples_per_step, observation.shape[0])
        sample_ids = torch.randint(
            0, observation.shape[0], (count,), device=self.ppo_device
        )
        self.estimator_training_history.append(
            self.estimator_history[sample_ids].detach().clone()
        )
        self.estimator_training_targets.append(truth[sample_ids].detach().clone())
        self.latent_training_mirrored_history.append(
            self.mirrored_estimator_history[sample_ids].detach().clone()
        )
        self.latent_training_com_cop.append(
            com_cop_target[sample_ids].detach().clone()
        )
        self.latent_training_contacts.append(
            contact_target[sample_ids].detach().clone()
        )
        self.pending_decoder_history = self.estimator_history.detach().clone()
        self.pending_decoder_explicit_target = truth.detach().clone()
        self.pending_decoder_mirrored_history = self.mirrored_estimator_history.detach().clone()
        return actor_observation, mirrored_actor_observation, estimate, truth

    def play_steps(self):
        if self.est_only_active and not self._est_only_announced:
            print("Latent late EST-only enabled: new episodes use 100% estimates; "
                  "existing GT episodes finish before switching.")
            self._est_only_announced = True
        self.latent_training_mirrored_history = []
        self.latent_training_com_cop = []
        self.latent_training_contacts = []
        self.latent_decoder_history = []
        self.latent_decoder_explicit_targets = []
        self.latent_decoder_next_targets = []
        self.latent_decoder_mirrored_history = []
        self.latent_decoder_mirrored_next_targets = []
        return super().play_steps()

    def _weighted_sum(self, losses):
        total = torch.zeros((), device=self.ppo_device)
        for name, value in losses.items():
            total = total + self.loss_weights.get(name, 0.0) * value
        return total

    def _train_concurrent_estimator(self):
        history = torch.cat(self.estimator_training_history, dim=0)
        explicit_targets = torch.cat(self.estimator_training_targets, dim=0)
        mirrored_history = torch.cat(self.latent_training_mirrored_history, dim=0)
        com_cop_targets = torch.cat(self.latent_training_com_cop, dim=0)
        contact_targets = torch.cat(self.latent_training_contacts, dim=0)
        decoder_history = (
            torch.cat(self.latent_decoder_history, dim=0)
            if self.latent_decoder_history
            else None
        )
        decoder_explicit = (
            torch.cat(self.latent_decoder_explicit_targets, dim=0)
            if self.latent_decoder_explicit_targets
            else None
        )
        decoder_next = (
            torch.cat(self.latent_decoder_next_targets, dim=0)
            if self.latent_decoder_next_targets
            else None
        )
        decoder_mirrored_history = (
            torch.cat(self.latent_decoder_mirrored_history, dim=0)
            if self.latent_decoder_mirrored_history else None
        )
        decoder_mirrored_next = (
            torch.cat(self.latent_decoder_mirrored_next_targets, dim=0)
            if self.latent_decoder_mirrored_next_targets else None
        )

        # Measure fresh rollout samples BEFORE fitting/updating their normalizers.
        self.latent_diagnostics = {}
        if int(self.epoch_num) % self.latent_diagnostics_every == 0:
            probe_history = decoder_history if decoder_history is not None else history
            probe_truth = decoder_explicit if decoder_history is not None else explicit_targets
            count = min(self.latent_diagnostics_samples, probe_history.shape[0])
            ids = torch.linspace(0, probe_history.shape[0] - 1, count,
                                 device=self.ppo_device).long()
            self.latent_diagnostics = self.concurrent_estimator.diagnostic_metrics(
                probe_history[ids], probe_truth[ids],
                next_target=decoder_next[ids] if decoder_next is not None else None,
            )

        if self.update_estimator_normalization:
            self.concurrent_estimator.update_normalization(
                torch.cat((history, mirrored_history)),
                torch.cat((explicit_targets, explicit_targets * self._mirror_sign)),
                next_target=(torch.cat((decoder_next, decoder_mirrored_next))
                             if self.use_next_observation_loss and decoder_next is not None else None),
                com_cop_target=(torch.cat((com_cop_targets, com_cop_targets *
                                          com_cop_targets.new_tensor([1., -1., 1.])))
                                if self.use_com_cop_auxiliary else None),
            )

        self.concurrent_estimator.train()
        metrics = {}
        total_samples = history.shape[0]
        # The configured batch budget includes both original and mirrored samples.
        original_batch_size = self.estimator_minibatch_size // 2
        for _ in range(self.estimator_mini_epochs):
            permutation = torch.randperm(total_samples, device=self.ppo_device)
            for start in range(0, total_samples, original_batch_size):
                ids = permutation[start : start + original_batch_size]
                losses = self.concurrent_estimator.compute_losses(
                    torch.cat((history[ids], mirrored_history[ids])),
                    torch.cat((explicit_targets[ids], explicit_targets[ids] * self._mirror_sign)),
                    com_cop_target=(
                        torch.cat((com_cop_targets[ids], com_cop_targets[ids] *
                                   com_cop_targets.new_tensor([1., -1., 1.])))
                        if self.use_com_cop_auxiliary else None
                    ),
                    contact_target=(
                        torch.cat((contact_targets[ids], contact_targets[ids][:, [1, 0]]))
                        if self.use_contact_auxiliary
                        else None
                    ),
                )
                total_loss = self._weighted_sum(losses)
                if self.use_next_observation_loss and decoder_history is not None:
                    decoder_ids = torch.randint(
                        0,
                        decoder_history.shape[0],
                        (ids.shape[0],),
                        device=self.ppo_device,
                    )
                    decoder_losses = self.concurrent_estimator.compute_losses(
                        torch.cat((decoder_history[decoder_ids], decoder_mirrored_history[decoder_ids])),
                        torch.cat((decoder_explicit[decoder_ids],
                                   decoder_explicit[decoder_ids] * self._mirror_sign)),
                        next_target=torch.cat((decoder_next[decoder_ids], decoder_mirrored_next[decoder_ids])),
                    )
                    next_loss = decoder_losses["next_observation"]
                    total_loss = total_loss + self.loss_weights["next_observation"] * next_loss
                    losses["next_observation"] = next_loss

                self.concurrent_estimator_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.concurrent_estimator.parameters(), self.estimator_grad_norm
                )
                self.concurrent_estimator_optimizer.step()
                metrics.setdefault("total", []).append(float(total_loss.detach()))
                for name, value in losses.items():
                    metrics.setdefault(name, []).append(float(value.detach()))

        self.concurrent_estimator.eval()
        self.latent_metrics = {
            name: float(np.mean(values)) for name, values in metrics.items()
        }
        return self.latent_metrics.get("explicit", 0.0)

    def _write_concurrent_metrics(self):
        super()._write_concurrent_metrics()
        frame = self.frame // self.num_agents
        for name, value in self.latent_metrics.items():
            self.writer.add_scalar("concurrent_latent/{}_loss".format(name), value, frame)
        self.writer.add_scalar("concurrent_latent/est_only_active", float(self.est_only_active), frame)
        for name, value in self.latent_diagnostics.items():
            self.writer.add_scalar("latent_diagnostics/pre_update/" + name, value, frame)

    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        state["concurrent_latent_model"] = state.pop("concurrent_estimator")
        state["concurrent_latent_optimizer"] = state.pop(
            "concurrent_estimator_optimizer"
        )
        state["concurrent_latent_config"] = state.pop(
            "concurrent_estimator_config"
        )
        state["concurrent_latent_schedule"] = {
            "stage_start_epoch": self.latent_stage_start_epoch,
            "task_training_stage": self.concurrent_task_training_stage,
        }
        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        translated = dict(weights)
        if "concurrent_latent_model" not in translated:
            raise KeyError("checkpoint does not contain concurrent_latent_model")
        restored = ConcurrentLatentEncoder.from_checkpoint(
            translated["concurrent_latent_model"], translated.get("concurrent_latent_config", {})
        )
        if restored.model_config() != self.concurrent_estimator.model_config():
            raise ValueError("Latent training checkpoint architecture differs from this v2 encoder. "
                             "Start this design from scratch; legacy checkpoints remain inference-only.")
        translated["concurrent_estimator"] = translated["concurrent_latent_model"]
        translated["concurrent_estimator_optimizer"] = translated.get(
            "concurrent_latent_optimizer"
        )
        translated["concurrent_estimator_config"] = translated.get(
            "concurrent_latent_config", {}
        )
        if translated["concurrent_estimator_optimizer"] is None:
            translated.pop("concurrent_estimator_optimizer")
        schedule = weights.get("concurrent_latent_schedule", {})
        same_stage = schedule.get("task_training_stage") == self.concurrent_task_training_stage
        self.latent_stage_start_epoch = (
            int(schedule["stage_start_epoch"]) if same_stage and set_epoch
            else int(weights.get("epoch", self.epoch_num) if set_epoch else self.epoch_num)
        )
        result = super().set_full_state_weights(translated, set_epoch=set_epoch)
        self._est_only_announced = False
        return result


class SRL_Bot_ConcurrentLatent_Player(common_player.CommonPlayer):
    """Inference player for the actor and causal concurrent latent encoder."""

    deployable_obs_dim = 133
    raw_obs_dim = 153
    explicit_dim = 4
    latent_dim = 16
    input_dim = 26

    def __init__(self, params):
        super().__init__(params)
        config = params["config"]
        self.eval_use_estimator = bool(config.get("latent_eval_use_estimator", True))
        self.eval_seed = params.get("seed", config.get("seed"))
        self.eval_latent_mode = str(config.get("latent_eval_mode", "normal"))
        if self.eval_latent_mode not in ("normal", "zero", "shuffle"):
            raise ValueError("latent_eval_mode must be normal, zero, or shuffle")
        self.latent_evaluation_enable = bool(config.get("latent_evaluation_enable", False))
        self.latent_evaluation_steps = int(config.get("latent_evaluation_steps", 5000))
        self.latent_evaluation_output = str(config.get("latent_evaluation_output", ""))
        if self.latent_evaluation_steps < 1:
            raise ValueError("latent_evaluation_steps must be positive")
        self.encoder = ConcurrentLatentEncoder(
            input_dim=self.input_dim,
            history_len=int(config.get("latent_history_len", 10)),
            explicit_dim=self.explicit_dim,
            latent_dim=self.latent_dim,
            hidden_dims=tuple(
                int(v)
                for v in config.get("latent_encoder_hidden_dims", [256, 128, 64])
            ),
            decoder_hidden_dims=tuple(
                int(v)
                for v in config.get("latent_decoder_hidden_dims", [128, 128])
            ),
        ).to(self.device)
        self.encoder.eval()
        self.history = None
        self.pending_reset_ids = None

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if "concurrent_latent_model" not in checkpoint:
            raise KeyError("checkpoint does not contain concurrent_latent_model")
        self.encoder = ConcurrentLatentEncoder.from_checkpoint(
            checkpoint["concurrent_latent_model"], checkpoint.get("concurrent_latent_config", {})
        ).to(self.device).eval()
        if (self.encoder.input_dim, self.encoder.explicit_dim, self.encoder.latent_dim) != (26, 4, 16):
            raise ValueError("Player requires encoder dimensions 26 -> explicit(4) + latent(16)")
        self.history = None
        self.checkpoint_path = str(fn)
        env_state = checkpoint.get("env_state")
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)
        print("Loaded ConcurrentLatent checkpoint '{}'".format(fn))

    def _env_reset_done(self):
        obs, done_env_ids = super()._env_reset_done()
        self.pending_reset_ids = torch.as_tensor(
            done_env_ids, device=self.device, dtype=torch.long
        ).reshape(-1)
        return obs, done_env_ids

    def _actor_observation(self, observation):
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.shape[1] != self.raw_obs_dim:
            raise RuntimeError(
                "ConcurrentLatent player expected 153-D raw observations, got {}".format(
                    tuple(observation.shape)
                )
            )
        deployable = observation[:, : self.deployable_obs_dim]
        current = deployable[:, : self.input_dim]
        if self.history is None:
            self.history = current.unsqueeze(1).expand(
                -1, self.encoder.history_len, -1
            ).clone()
        else:
            self.history[:, :-1] = self.history[:, 1:].clone()
            self.history[:, -1] = current
            if self.pending_reset_ids is not None and self.pending_reset_ids.numel() > 0:
                reset_ids = self.pending_reset_ids
                self.history[reset_ids] = current[reset_ids].unsqueeze(1).expand(
                    -1, self.encoder.history_len, -1
                )
        self.pending_reset_ids = None
        output = self.encoder(self.history)
        self.last_explicit_error = output["explicit"] - observation[:, 133:137]
        if self.eval_use_estimator:
            explicit = output["explicit"]
        else:
            explicit = observation[
                :, self.deployable_obs_dim : self.deployable_obs_dim + self.explicit_dim
            ]
        latent = self.encoder.ablate_latent(output["latent"], self.eval_latent_mode)
        return torch.cat((deployable, explicit, latent), dim=-1)

    def run(self):
        if not self.latent_evaluation_enable:
            return super().run()
        if self.is_rnn or self.num_agents != 1:
            raise ValueError("Latent diagnostics require the feed-forward single-agent SRL task")
        # SRL ends at progress >= episodeLength - 1. Reserve the extra tick so
        # the requested number of control steps is actually measurable.
        self.env.max_episode_length = max(
            self.env.max_episode_length, self.latent_evaluation_steps + 1
        )
        if self.eval_latent_mode == "shuffle" and self.env.num_envs < 2:
            raise ValueError("shuffle evaluation requires num_envs >= 2")
        self.env_reset(self.env)
        self.history = None
        self.pending_reset_ids = None
        env = self.env
        n = env.num_envs
        device = env.device
        active = torch.ones(n, device=device, dtype=torch.bool)
        failed = torch.zeros_like(active)
        timed_out = torch.zeros_like(active)
        lengths = torch.zeros(n, device=device)
        returns = torch.zeros_like(lengths)
        roll2 = torch.zeros_like(lengths)
        hip2 = torch.zeros_like(lengths)
        separation = torch.zeros_like(lengths)
        vx_error = torch.zeros_like(lengths)
        explicit_abs = torch.zeros(n, 4, device=device)
        explicit2 = torch.zeros_like(explicit_abs)
        # Each environment contributes its first episode only; no short-episode
        # overrepresentation. Use the same seed/config in separate mode runs.
        with torch.no_grad():
            for _ in range(self.latent_evaluation_steps):
                obs, _ = self._env_reset_done()
                action = self.get_action(obs, is_determenistic=True)
                ids = active.nonzero(as_tuple=False).reshape(-1)
                roll2[ids] += env.full_obs_buf[ids, 9].square()
                hip2[ids] += env.dof_vel[ids][:, [0, 3]].square().mean(dim=1)
                vx_error[ids] += (env.full_obs_buf[ids, 1] - env.target_vel_x[ids]).abs()
                root = env.srl_root_states[ids]
                qx, qy, qz, qw = root[:, 3:7].unbind(dim=1)
                yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy.square() + qz.square()))
                feet = env._rigid_body_pos[ids][:, env._srl_end_ids]
                delta = feet[:, 0] - feet[:, 1]
                separation[ids] += (-yaw.sin() * delta[:, 0] + yaw.cos() * delta[:, 1]).abs()
                error = self.last_explicit_error.to(device)[ids]
                explicit_abs[ids] += error.abs()
                explicit2[ids] += error.square()
                _, reward, done, info = self.env_step(env, action)
                lengths[ids] += 1
                returns[ids] += reward.to(device).reshape(n, -1).mean(dim=1)[ids]
                done = done.to(device).reshape(-1).bool()
                terminated = env._terminate_buf.reshape(-1).bool()
                failed |= active & terminated
                timed_out |= active & done & ~terminated
                active &= ~(done | terminated)
                if self.render_env:
                    env.render(mode="human")
                if not active.any():
                    break
        denominator = lengths.clamp_min(1)
        success = ~failed & (lengths >= self.latent_evaluation_steps)
        rows = []
        for i in range(n):
            rows.append({
                "env_id": i, "steps": int(lengths[i].item()),
                "success": bool(success[i].item()), "failed": bool(failed[i].item()),
                "environment_timeout": bool(timed_out[i].item()),
                "return": returns[i].item(),
                "roll_rms": (roll2[i] / denominator[i]).sqrt().item(),
                "hip_x_velocity_rms": (hip2[i] / denominator[i]).sqrt().item(),
                "foot_separation": (separation[i] / denominator[i]).item(),
                "vx_tracking_mae": (vx_error[i] / denominator[i]).item(),
                "explicit_mae": (explicit_abs[i] / denominator[i]).cpu().tolist(),
                "explicit_rmse": (explicit2[i] / denominator[i]).sqrt().cpu().tolist(),
            })
        report = {
            "checkpoint": getattr(self, "checkpoint_path", ""),
            "model_version": self.encoder.model_version,
            "mode": self.eval_latent_mode, "explicit_mode": "est" if self.eval_use_estimator else "gt",
            "seed": self.eval_seed, "horizon": self.latent_evaluation_steps,
            "num_envs": n, "success_rate": success.float().mean().item(),
            "mean_length": lengths.mean().item(), "mean_return": returns.mean().item(),
            "episodes": rows,
        }
        print("[latent eval {}/{}] len={:.0f} success={:.3f} return={:.0f}".format(
            report["explicit_mode"], report["mode"], report["mean_length"],
            report["success_rate"], report["mean_return"]))
        if timed_out.any() and (lengths[timed_out] < self.latent_evaluation_steps).any():
            print("WARNING: environment episodeLength is shorter than requested evaluation horizon; "
                  "these episodes are censored, not counted as full-horizon successes.")
        if self.latent_evaluation_output:
            path = Path(self.latent_evaluation_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("Latent evaluation saved to {}".format(path))
        return report

    def get_action(self, obs_dict, is_determenistic=False):
        with torch.no_grad():
            observation = self._actor_observation(obs_dict["obs"])
        actor_obs = dict(obs_dict)
        actor_obs["obs"] = observation
        return common_player.CommonPlayer.get_action(
            self, actor_obs, is_determenistic
        )
