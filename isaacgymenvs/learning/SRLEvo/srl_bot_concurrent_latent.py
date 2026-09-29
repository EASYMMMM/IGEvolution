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
        self.use_mirror_consistency = bool(
            config.get("latent_use_mirror_consistency", True)
        )
        self.loss_weights = {
            "explicit": float(config.get("latent_explicit_loss_coef", 1.0)),
            "next_observation": float(
                config.get("latent_next_observation_loss_coef", 1.0)
            ),
            "com_cop": float(config.get("latent_com_cop_loss_coef", 0.25)),
            "contact": float(config.get("latent_contact_loss_coef", 0.1)),
            "mirror_latent": float(
                config.get("latent_mirror_consistency_loss_coef", 0.05)
            ),
            "latent_l2": float(config.get("latent_l2_loss_coef", 1e-4)),
        }

        self.mirrored_estimator_history = None
        self.pending_decoder_history = None
        self.pending_decoder_explicit_target = None
        self.latent_training_mirrored_history = []
        self.latent_training_com_cop = []
        self.latent_training_contacts = []
        self.latent_decoder_history = []
        self.latent_decoder_explicit_targets = []
        self.latent_decoder_next_targets = []
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

    def _collect_decoder_transitions(self, current_frame, reset_ids):
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
            if float(state_error.item()) > 1e-5:
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
        self._collect_decoder_transitions(current_frame, reset_ids)
        self._update_history(deployable, reset_ids)
        self._update_mirrored_history(mirrored_deployable, reset_ids)

        # Encoder updates are performed after the PPO epoch from sampled
        # histories, so rollout inference must not retain a horizon-long graph.
        with torch.no_grad():
            output = self.concurrent_estimator(self.estimator_history)
        estimate = output["explicit"]
        latent = output["latent"]
        selected_explicit = torch.where(
            self.episode_uses_estimator.unsqueeze(1), estimate.detach(), truth
        )
        mirrored_explicit = torch.where(
            self.episode_uses_estimator.unsqueeze(1),
            estimate.detach() * self._mirror_sign,
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
            (mirrored_deployable, mirrored_explicit, latent.detach()), dim=-1
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
        return actor_observation, mirrored_actor_observation, estimate, truth

    def play_steps(self):
        self.latent_training_mirrored_history = []
        self.latent_training_com_cop = []
        self.latent_training_contacts = []
        self.latent_decoder_history = []
        self.latent_decoder_explicit_targets = []
        self.latent_decoder_next_targets = []
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

        if self.update_estimator_normalization:
            self.concurrent_estimator.update_normalization(
                history,
                explicit_targets,
                next_target=decoder_next if self.use_next_observation_loss else None,
                com_cop_target=com_cop_targets if self.use_com_cop_auxiliary else None,
            )

        self.concurrent_estimator.train()
        metrics = {}
        total_samples = history.shape[0]
        for _ in range(self.estimator_mini_epochs):
            permutation = torch.randperm(total_samples, device=self.ppo_device)
            for start in range(0, total_samples, self.estimator_minibatch_size):
                ids = permutation[start : start + self.estimator_minibatch_size]
                losses = self.concurrent_estimator.compute_losses(
                    history[ids],
                    explicit_targets[ids],
                    com_cop_target=(
                        com_cop_targets[ids] if self.use_com_cop_auxiliary else None
                    ),
                    contact_target=(
                        contact_targets[ids]
                        if self.use_contact_auxiliary
                        else None
                    ),
                    mirrored_history=(
                        mirrored_history[ids]
                        if self.use_mirror_consistency
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
                        decoder_history[decoder_ids],
                        decoder_explicit[decoder_ids],
                        next_target=decoder_next[decoder_ids],
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

    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        state["concurrent_latent_model"] = state.pop("concurrent_estimator")
        state["concurrent_latent_optimizer"] = state.pop(
            "concurrent_estimator_optimizer"
        )
        state["concurrent_latent_config"] = state.pop(
            "concurrent_estimator_config"
        )
        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        translated = dict(weights)
        if "concurrent_latent_model" not in translated:
            raise KeyError("checkpoint does not contain concurrent_latent_model")
        translated["concurrent_estimator"] = translated["concurrent_latent_model"]
        translated["concurrent_estimator_optimizer"] = translated.get(
            "concurrent_latent_optimizer"
        )
        translated["concurrent_estimator_config"] = translated.get(
            "concurrent_latent_config", {}
        )
        if translated["concurrent_estimator_optimizer"] is None:
            translated.pop("concurrent_estimator_optimizer")
        return super().set_full_state_weights(translated, set_epoch=set_epoch)


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
        checkpoint_config = checkpoint.get("concurrent_latent_config", {})
        expected = self.encoder.model_config()
        for key in (
            "input_dim",
            "history_len",
            "explicit_dim",
            "latent_dim",
            "hidden_dims",
            "decoder_hidden_dims",
        ):
            if key in checkpoint_config and checkpoint_config[key] != expected[key]:
                raise RuntimeError(
                    "latent checkpoint {}={} does not match config {}".format(
                        key, checkpoint_config[key], expected[key]
                    )
                )
        self.encoder.load_state_dict(checkpoint["concurrent_latent_model"])
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
        if self.eval_use_estimator:
            explicit = output["explicit"]
        else:
            explicit = observation[
                :, self.deployable_obs_dim : self.deployable_obs_dim + self.explicit_dim
            ]
        return torch.cat((deployable, explicit, output["latent"]), dim=-1)

    def get_action(self, obs_dict, is_determenistic=False):
        with torch.no_grad():
            observation = self._actor_observation(obs_dict["obs"])
        actor_obs = dict(obs_dict)
        actor_obs["obs"] = observation
        return common_player.CommonPlayer.get_action(
            self, actor_obs, is_determenistic
        )
