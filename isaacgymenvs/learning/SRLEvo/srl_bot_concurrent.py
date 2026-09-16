from collections import deque
import time

import numpy as np
import torch

from isaacgymenvs.learning import common_player
from isaacgymenvs.learning.SRLEvo.concurrent_privileged_estimator import (
    ConcurrentPrivilegedEstimator,
)
from isaacgymenvs.learning.SRLEvo.srl_bot_continuous import (
    SRL_Bot_Agent,
    swap_and_flatten01,
)
from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext


TARGET_NAMES = ("root_height", "local_vx", "local_vy", "local_vz")


class SRL_Bot_Concurrent_Agent(SRL_Bot_Agent):
    """PPO agent that trains a causal state estimator on the same rollouts."""

    deployable_obs_dim = 133
    actor_obs_dim = 137
    critic_obs_dim = 153
    estimator_input_dim = 26
    estimator_output_dim = 4
    mirror_privileged_sign = (1.0, 1.0, -1.0, 1.0)

    def __init__(self, base_name, params):
        config = params["config"]
        if bool(config.get("train_with_privileged_estimator", False)):
            raise ValueError(
                "The frozen-estimator fine-tuning path must be disabled for "
                "concurrent training"
            )

        super().__init__(base_name, params)
        if tuple(self.obs_shape) != (self.actor_obs_dim,):
            raise RuntimeError(
                "Concurrent actor observation must be 137-D, got {}".format(
                    self.obs_shape
                )
            )
        if not self.has_central_value or tuple(self.state_shape) != (
            self.critic_obs_dim,
        ):
            raise RuntimeError(
                "Concurrent training requires a 153-D asymmetric central critic"
            )

        history_len = int(config.get("concurrent_estimator_history_len", 10))
        hidden_dims = tuple(
            int(value)
            for value in config.get(
                "concurrent_estimator_hidden_dims", [256, 128, 64]
            )
        )
        self.concurrent_estimator = ConcurrentPrivilegedEstimator(
            input_dim=self.estimator_input_dim,
            history_len=history_len,
            output_dim=self.estimator_output_dim,
            hidden_dims=hidden_dims,
        ).to(self.ppo_device)
        self.concurrent_estimator_optimizer = torch.optim.Adam(
            self.concurrent_estimator.parameters(),
            lr=float(config.get("concurrent_estimator_learning_rate", 1e-3)),
            eps=1e-8,
        )
        self.estimator_mini_epochs = int(
            config.get("concurrent_estimator_mini_epochs", 2)
        )
        self.estimator_minibatch_size = int(
            config.get("concurrent_estimator_minibatch_size", 8192)
        )
        self.estimator_samples_per_step = int(
            config.get("concurrent_estimator_samples_per_step", 1024)
        )
        self.estimator_grad_norm = float(
            config.get("concurrent_estimator_grad_norm", 1.0)
        )
        self.update_estimator_normalization = bool(
            config.get("concurrent_estimator_update_normalization", True)
        )

        levels = [
            float(value)
            for value in config.get(
                "concurrent_bootstrap_levels", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
            )
        ]
        if not levels or levels[0] != 0.0 or levels[-1] != 1.0:
            raise ValueError("bootstrap levels must start at 0.0 and end at 1.0")
        if levels != sorted(set(levels)):
            raise ValueError("bootstrap levels must be unique and increasing")
        self.bootstrap_levels = levels
        self.bootstrap_level_index = int(
            config.get("concurrent_bootstrap_initial_level", 0)
        )
        if not 0 <= self.bootstrap_level_index < len(self.bootstrap_levels):
            raise ValueError("invalid concurrent_bootstrap_initial_level")
        self.bootstrap_auto_advance = bool(
            config.get("concurrent_bootstrap_auto_advance", True)
        )
        self.bootstrap_warmup_epochs = int(
            config.get("concurrent_bootstrap_warmup_epochs", 300)
        )
        self.bootstrap_min_epochs_per_level = int(
            config.get("concurrent_bootstrap_min_epochs_per_level", 100)
        )
        self.bootstrap_min_completed_episodes = int(
            config.get("concurrent_bootstrap_min_completed_episodes", 50)
        )
        self.bootstrap_ground_truth_min_length = float(
            config.get("concurrent_bootstrap_ground_truth_min_length", 4000.0)
        )
        self.bootstrap_promote_length_ratio = float(
            config.get("concurrent_bootstrap_promote_length_ratio", 0.90)
        )
        self.bootstrap_promote_reward_ratio = float(
            config.get("concurrent_bootstrap_promote_reward_ratio", 0.85)
        )
        self.bootstrap_demote_length_ratio = float(
            config.get("concurrent_bootstrap_demote_length_ratio", 0.70)
        )
        self.bootstrap_max_normalized_mse = float(
            config.get("concurrent_bootstrap_max_normalized_mse", 0.20)
        )
        self.concurrent_task_training_stage = int(
            config.get("concurrent_task_training_stage", -1)
        )
        self.bootstrap_reset_on_stage_change = bool(
            config.get("concurrent_bootstrap_reset_on_stage_change", True)
        )
        self.bootstrap_reset_on_restore = bool(
            config.get("concurrent_bootstrap_reset_on_restore", False)
        )
        self.bootstrap_reset_probability = float(
            config.get("concurrent_bootstrap_reset_probability", 0.5)
        )
        self.bootstrap_clear_metrics_on_reset = bool(
            config.get("concurrent_bootstrap_clear_metrics_on_reset", True)
        )
        if not 0.0 <= self.bootstrap_reset_probability <= 1.0:
            raise ValueError(
                "concurrent_bootstrap_reset_probability must be in [0, 1]"
            )
        metric_window = int(config.get("concurrent_bootstrap_metric_window", 200))
        self.gt_episode_lengths = deque(maxlen=metric_window)
        self.gt_episode_rewards = deque(maxlen=metric_window)
        self.estimated_episode_lengths = deque(maxlen=metric_window)
        self.estimated_episode_rewards = deque(maxlen=metric_window)
        self.bootstrap_level_start_epoch = int(self.epoch_num)

        self.estimator_history = None
        self.episode_uses_estimator = None
        self.estimator_training_history = []
        self.estimator_training_targets = []
        self.concurrent_metrics = {}
        self._observation_layout_validated = False
        self._mirror_sign = torch.tensor(
            self.mirror_privileged_sign,
            device=self.ppo_device,
            dtype=torch.float32,
        )

        print(
            "Concurrent estimator enabled: history={} input={} hidden={} "
            "bootstrap_levels={}".format(
                history_len,
                self.estimator_input_dim,
                hidden_dims,
                self.bootstrap_levels,
            )
        )

    @property
    def bootstrap_probability(self):
        return self.bootstrap_levels[self.bootstrap_level_index]

    def _sample_episode_modes(self, env_ids):
        if env_ids.numel() == 0:
            return
        probability = self.bootstrap_probability
        if probability <= 0.0:
            sampled = torch.zeros_like(env_ids, dtype=torch.bool)
        elif probability >= 1.0:
            sampled = torch.ones_like(env_ids, dtype=torch.bool)
        else:
            sampled = torch.rand(env_ids.shape[0], device=self.ppo_device) < probability
        self.episode_uses_estimator[env_ids] = sampled

    def _update_history(self, deployable_obs, reset_env_ids):
        current = deployable_obs[:, : self.estimator_input_dim]
        num_envs = current.shape[0]
        if self.estimator_history is None:
            self.estimator_history = current.unsqueeze(1).expand(
                -1, self.concurrent_estimator.history_len, -1
            ).clone()
            self.episode_uses_estimator = torch.zeros(
                num_envs, device=self.ppo_device, dtype=torch.bool
            )
            all_ids = torch.arange(num_envs, device=self.ppo_device)
            self._sample_episode_modes(all_ids)
            return

        self.estimator_history[:, :-1] = self.estimator_history[:, 1:].clone()
        self.estimator_history[:, -1] = current
        if reset_env_ids.numel() > 0:
            self.estimator_history[reset_env_ids] = current[
                reset_env_ids
            ].unsqueeze(1).expand(-1, self.concurrent_estimator.history_len, -1)
            self._sample_episode_modes(reset_env_ids)

    def _build_concurrent_actor_observation(self, raw_obs, done_env_ids):
        observation = raw_obs["obs"]
        mirrored_observation = raw_obs["obs_mirrored"]
        states = raw_obs["states"]
        if observation.shape[1] != self.actor_obs_dim:
            raise RuntimeError("expected 137-D actor observation")
        if states.shape[1] != self.critic_obs_dim:
            raise RuntimeError("expected 153-D critic state")

        if not self._observation_layout_validated:
            actor_truth = observation[:, -self.estimator_output_dim :]
            critic_truth = states[:, : self.estimator_output_dim]
            actor_commands = observation[
                :, self.deployable_obs_dim - 3 : self.deployable_obs_dim
            ]
            critic_commands = states[:, -3:]
            mirrored_truth = mirrored_observation[
                :, -self.estimator_output_dim :
            ]
            truth_error = (actor_truth - critic_truth).abs().max().item()
            command_error = (actor_commands - critic_commands).abs().max().item()
            mirror_error = (
                mirrored_truth - critic_truth * self._mirror_sign
            ).abs().max().item()
            if max(truth_error, command_error, mirror_error) > 1e-5:
                raise RuntimeError(
                    "Concurrent observation layout mismatch: truth={:.3e}, "
                    "commands={:.3e}, mirror={:.3e}".format(
                        truth_error, command_error, mirror_error
                    )
                )
            self._observation_layout_validated = True
            print(
                "Concurrent observation layout verified: "
                "133 deployable + 4 privileged, critic=153"
            )

        reset_ids = torch.as_tensor(
            done_env_ids, device=self.ppo_device, dtype=torch.long
        ).reshape(-1)
        deployable = observation[:, : self.deployable_obs_dim]
        mirrored_deployable = mirrored_observation[:, : self.deployable_obs_dim]
        self._update_history(deployable, reset_ids)

        estimate = self.concurrent_estimator(self.estimator_history)
        target = states[:, : self.estimator_output_dim]
        selected = torch.where(
            self.episode_uses_estimator.unsqueeze(1), estimate.detach(), target
        )
        mirrored_selected = torch.where(
            self.episode_uses_estimator.unsqueeze(1),
            estimate.detach() * self._mirror_sign,
            mirrored_observation[:, -self.estimator_output_dim :],
        )
        actor_observation = torch.cat((deployable, selected), dim=-1)
        mirrored_actor_observation = torch.cat(
            (mirrored_deployable, mirrored_selected), dim=-1
        )

        sample_count = min(self.estimator_samples_per_step, observation.shape[0])
        sample_ids = torch.randint(
            0, observation.shape[0], (sample_count,), device=self.ppo_device
        )
        self.estimator_training_history.append(
            self.estimator_history[sample_ids].detach().clone()
        )
        self.estimator_training_targets.append(target[sample_ids].detach().clone())
        return actor_observation, mirrored_actor_observation, estimate, target

    def _record_completed_modes(self, env_done_indices):
        if env_done_indices.numel() == 0:
            return
        modes = self.episode_uses_estimator[env_done_indices]
        lengths = self.current_lengths[env_done_indices].detach().cpu().tolist()
        rewards = (
            self.current_rewards[env_done_indices, 0].detach().cpu().tolist()
        )
        modes = modes.detach().cpu().tolist()
        for mode, length, reward in zip(modes, lengths, rewards):
            if mode:
                self.estimated_episode_lengths.append(float(length))
                self.estimated_episode_rewards.append(float(reward))
            else:
                self.gt_episode_lengths.append(float(length))
                self.gt_episode_rewards.append(float(reward))

    def play_steps(self):
        update_list = self.update_list
        step_time = 0.0
        absolute_error_sum = torch.zeros(4, device=self.ppo_device)
        squared_error_sum = torch.zeros(4, device=self.ppo_device)
        error_count = 0
        self.estimator_training_history = []
        self.estimator_training_targets = []
        estimated_env_fraction_sum = 0.0

        for n in range(self.horizon_length):
            self.obs, done_env_ids = self._env_reset_done()
            raw_obs = self.obs
            actor_observation, mirrored_actor_observation, estimate, target = (
                self._build_concurrent_actor_observation(raw_obs, done_env_ids)
            )
            actor_obs = dict(raw_obs)
            actor_obs["obs"] = actor_observation
            actor_obs["obs_mirrored"] = mirrored_actor_observation

            error = estimate - target
            absolute_error_sum += error.abs().sum(dim=0)
            squared_error_sum += error.square().sum(dim=0)
            error_count += error.shape[0]
            estimated_env_fraction_sum += float(
                self.episode_uses_estimator.float().mean().item()
            )

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(actor_obs, masks)
            else:
                res_dict = self.get_action_values(actor_obs)
            if self.has_central_value:
                res_dict["values"] = self.get_central_value(
                    {"states": raw_obs["states"]}
                )

            self.experience_buffer.update_data("obses", n, actor_observation)
            self.experience_buffer.update_data("dones", n, self.dones)
            self.experience_buffer.update_data(
                "obs_mirrored", n, mirrored_actor_observation
            )
            for key in update_list:
                self.experience_buffer.update_data(key, n, res_dict[key])
            if self.has_central_value:
                self.experience_buffer.update_data("states", n, raw_obs["states"])

            step_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            step_time += time.time() - step_start

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and "time_outs" in infos:
                shaped_rewards += self.gamma * res_dict["values"] * self.cast_obs(
                    infos["time_outs"]
                ).unsqueeze(1).float()
            self.experience_buffer.update_data("rewards", n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[:: self.num_agents]
            env_done_ids = env_done_indices.reshape(-1)
            self._record_completed_modes(env_done_ids)
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(
                self.current_shaped_rewards[env_done_indices]
            )
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()
            self.current_rewards *= not_dones.unsqueeze(1)
            self.current_shaped_rewards *= not_dones.unsqueeze(1)
            self.current_lengths *= not_dones

        denominator = max(error_count, 1)
        self.concurrent_metrics = {
            "mae": (absolute_error_sum / denominator).detach().cpu(),
            "rmse": torch.sqrt(squared_error_sum / denominator).detach().cpu(),
            "bootstrap_probability": float(self.bootstrap_probability),
            "estimated_env_fraction": estimated_env_fraction_sum
            / float(self.horizon_length),
        }

        if self.has_central_value:
            last_values = self.get_central_value({"states": self.obs["states"]})
        else:
            last_values = self.get_values(self.obs)
        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_advs = self.discount_values(
            fdones, last_values, mb_fdones, mb_values, mb_rewards
        )
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(
            swap_and_flatten01, self.tensor_list
        )
        batch_dict["returns"] = swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size
        batch_dict["step_time"] = step_time
        batch_dict["obs_mirrored"] = a2c_common.swap_and_flatten01(
            self.experience_buffer.tensor_dict["obs_mirrored"]
        )
        return batch_dict

    def _train_concurrent_estimator(self):
        history = torch.cat(self.estimator_training_history, dim=0)
        targets = torch.cat(self.estimator_training_targets, dim=0)
        if self.update_estimator_normalization:
            self.concurrent_estimator.update_normalization(history, targets)

        self.concurrent_estimator.train()
        losses = []
        total = history.shape[0]
        for _ in range(self.estimator_mini_epochs):
            permutation = torch.randperm(total, device=self.ppo_device)
            for start in range(0, total, self.estimator_minibatch_size):
                indices = permutation[start : start + self.estimator_minibatch_size]
                loss = self.concurrent_estimator.normalized_mse(
                    history[indices], targets[indices]
                )
                self.concurrent_estimator_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.concurrent_estimator.parameters(), self.estimator_grad_norm
                )
                self.concurrent_estimator_optimizer.step()
                losses.append(float(loss.detach().item()))
        self.concurrent_estimator.eval()
        return float(np.mean(losses)) if losses else 0.0

    @staticmethod
    def _mean_or_zero(values):
        return float(np.mean(values)) if values else 0.0

    def _maybe_update_bootstrap_level(self, normalized_mse):
        if not self.bootstrap_auto_advance:
            return
        epoch = int(self.epoch_num)
        if epoch - self.bootstrap_level_start_epoch < self.bootstrap_min_epochs_per_level:
            return

        minimum = self.bootstrap_min_completed_episodes
        if self.bootstrap_level_index == 0:
            ready = (
                epoch >= self.bootstrap_warmup_epochs
                and len(self.gt_episode_lengths) >= minimum
                and self._mean_or_zero(self.gt_episode_lengths)
                >= self.bootstrap_ground_truth_min_length
                and normalized_mse <= self.bootstrap_max_normalized_mse
            )
            if ready:
                self.bootstrap_level_index += 1
                self.bootstrap_level_start_epoch = epoch
                print(
                    "Bootstrap promoted to {:.2f} at epoch {}".format(
                        self.bootstrap_probability, epoch
                    )
                )
            return

        if len(self.estimated_episode_lengths) < minimum or len(
            self.gt_episode_lengths
        ) < minimum:
            return
        gt_length = max(self._mean_or_zero(self.gt_episode_lengths), 1e-6)
        gt_reward = self._mean_or_zero(self.gt_episode_rewards)
        estimated_length = self._mean_or_zero(self.estimated_episode_lengths)
        estimated_reward = self._mean_or_zero(self.estimated_episode_rewards)
        length_ratio = estimated_length / gt_length
        if gt_reward > 0.0:
            reward_ratio = estimated_reward / gt_reward
        else:
            reward_ratio = 1.0 if estimated_reward >= gt_reward else 0.0

        if (
            length_ratio < self.bootstrap_demote_length_ratio
            and self.bootstrap_level_index > 0
        ):
            self.bootstrap_level_index -= 1
            self.bootstrap_level_start_epoch = epoch
            print(
                "Bootstrap demoted to {:.2f} at epoch {} (length ratio {:.3f})".format(
                    self.bootstrap_probability, epoch, length_ratio
                )
            )
        elif (
            self.bootstrap_level_index < len(self.bootstrap_levels) - 1
            and length_ratio >= self.bootstrap_promote_length_ratio
            and reward_ratio >= self.bootstrap_promote_reward_ratio
            and normalized_mse <= self.bootstrap_max_normalized_mse
        ):
            self.bootstrap_level_index += 1
            self.bootstrap_level_start_epoch = epoch
            print(
                "Bootstrap promoted to {:.2f} at epoch {} "
                "(length ratio {:.3f}, reward ratio {:.3f})".format(
                    self.bootstrap_probability, epoch, length_ratio, reward_ratio
                )
            )

    def _write_concurrent_metrics(self):
        frame = self.frame // self.num_agents
        self.writer.add_scalar(
            "concurrent/bootstrap_probability",
            self.concurrent_metrics["bootstrap_probability"],
            frame,
        )
        self.writer.add_scalar(
            "concurrent/estimated_env_fraction",
            self.concurrent_metrics["estimated_env_fraction"],
            frame,
        )
        self.writer.add_scalar(
            "concurrent/estimator_normalized_mse",
            self.concurrent_metrics["normalized_mse"],
            frame,
        )
        for index, name in enumerate(TARGET_NAMES):
            self.writer.add_scalar(
                "concurrent/{}_mae".format(name),
                self.concurrent_metrics["mae"][index].item(),
                frame,
            )
            self.writer.add_scalar(
                "concurrent/{}_rmse".format(name),
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
                    "concurrent/{}_episode_length".format(prefix),
                    self._mean_or_zero(lengths),
                    frame,
                )
                self.writer.add_scalar(
                    "concurrent/{}_episode_reward".format(prefix),
                    self._mean_or_zero(rewards),
                    frame,
                )

    def train_epoch(self):
        result = super().train_epoch()
        normalized_mse = self._train_concurrent_estimator()
        self.concurrent_metrics["normalized_mse"] = normalized_mse
        self._maybe_update_bootstrap_level(normalized_mse)
        self._write_concurrent_metrics()
        return result

    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        state["concurrent_estimator"] = self.concurrent_estimator.state_dict()
        state["concurrent_estimator_optimizer"] = (
            self.concurrent_estimator_optimizer.state_dict()
        )
        state["concurrent_estimator_config"] = self.concurrent_estimator.model_config()
        state["concurrent_bootstrap_state"] = {
            "level_index": self.bootstrap_level_index,
            "level_start_epoch": self.bootstrap_level_start_epoch,
            "task_training_stage": self.concurrent_task_training_stage,
            "gt_episode_lengths": list(self.gt_episode_lengths),
            "gt_episode_rewards": list(self.gt_episode_rewards),
            "estimated_episode_lengths": list(self.estimated_episode_lengths),
            "estimated_episode_rewards": list(self.estimated_episode_rewards),
        }
        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        result = super().set_full_state_weights(weights, set_epoch=set_epoch)
        if "concurrent_estimator" not in weights:
            raise KeyError(
                "Checkpoint does not contain a concurrently trained estimator"
            )
        self.concurrent_estimator.load_state_dict(weights["concurrent_estimator"])
        if "concurrent_estimator_optimizer" in weights:
            self.concurrent_estimator_optimizer.load_state_dict(
                weights["concurrent_estimator_optimizer"]
            )
        bootstrap = weights.get("concurrent_bootstrap_state", {})
        saved_stage = int(bootstrap.get("task_training_stage", -1))
        stage_changed = (
            self.bootstrap_reset_on_stage_change
            and saved_stage >= 0
            and self.concurrent_task_training_stage >= 0
            and saved_stage != self.concurrent_task_training_stage
        )
        reset_bootstrap = self.bootstrap_reset_on_restore or stage_changed

        if reset_bootstrap:
            self.bootstrap_level_index = min(
                range(len(self.bootstrap_levels)),
                key=lambda index: abs(
                    self.bootstrap_levels[index]
                    - self.bootstrap_reset_probability
                ),
            )
            self.bootstrap_level_start_epoch = int(self.epoch_num)
        else:
            self.bootstrap_level_index = int(
                bootstrap.get("level_index", self.bootstrap_level_index)
            )
            self.bootstrap_level_start_epoch = int(
                bootstrap.get("level_start_epoch", self.epoch_num)
            )

        for destination, key in (
            (self.gt_episode_lengths, "gt_episode_lengths"),
            (self.gt_episode_rewards, "gt_episode_rewards"),
            (self.estimated_episode_lengths, "estimated_episode_lengths"),
            (self.estimated_episode_rewards, "estimated_episode_rewards"),
        ):
            destination.clear()
            if not (reset_bootstrap and self.bootstrap_clear_metrics_on_reset):
                destination.extend(bootstrap.get(key, []))

        if reset_bootstrap:
            reason = (
                "explicit restore reset"
                if self.bootstrap_reset_on_restore
                else "task stage {} -> {}".format(
                    saved_stage, self.concurrent_task_training_stage
                )
            )
            print(
                "Restored concurrent estimator; reset bootstrap to {:.2f} at "
                "epoch {} ({}, metrics_cleared={})".format(
                    self.bootstrap_probability,
                    self.epoch_num,
                    reason,
                    self.bootstrap_clear_metrics_on_reset,
                )
            )
        else:
            print(
                "Restored concurrent estimator and bootstrap probability "
                "{:.2f} (task stage {})".format(
                    self.bootstrap_probability,
                    self.concurrent_task_training_stage,
                )
            )
        return result


class SRL_Bot_Concurrent_Player(common_player.CommonPlayer):
    """Inference player for a jointly saved actor and MLP estimator."""

    deployable_obs_dim = 133
    actor_obs_dim = 137
    estimator_input_dim = 26
    mirror_privileged_sign = (1.0, 1.0, -1.0, 1.0)

    def __init__(self, params):
        super().__init__(params)
        config = params["config"]
        self.concurrent_eval_use_estimator = bool(
            config.get("concurrent_eval_use_estimator", True)
        )
        self.concurrent_estimator = ConcurrentPrivilegedEstimator(
            input_dim=self.estimator_input_dim,
            history_len=int(config.get("concurrent_estimator_history_len", 10)),
            output_dim=4,
            hidden_dims=tuple(
                int(value)
                for value in config.get(
                    "concurrent_estimator_hidden_dims", [256, 128, 64]
                )
            ),
        ).to(self.device)
        self.concurrent_estimator.eval()
        self.estimator_history = None
        self.pending_reset_ids = None

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.model.running_mean_std.load_state_dict(
                checkpoint["running_mean_std"]
            )
        if "concurrent_estimator" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain a concurrently trained estimator"
            )
        checkpoint_config = checkpoint.get("concurrent_estimator_config", {})
        expected_config = self.concurrent_estimator.model_config()
        for key in ("input_dim", "history_len", "output_dim", "hidden_dims"):
            if key in checkpoint_config and checkpoint_config[key] != expected_config[key]:
                raise RuntimeError(
                    "Estimator checkpoint {}={} does not match config {}".format(
                        key, checkpoint_config[key], expected_config[key]
                    )
                )
        self.concurrent_estimator.load_state_dict(checkpoint["concurrent_estimator"])
        env_state = checkpoint.get("env_state")
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)
        print(
            "Loaded concurrent actor/estimator checkpoint '{}' (epoch {})".format(
                fn, checkpoint.get("epoch", checkpoint.get("iter", 0))
            )
        )

    def _env_reset_done(self):
        obs, done_env_ids = super()._env_reset_done()
        self.pending_reset_ids = torch.as_tensor(
            done_env_ids, device=self.device, dtype=torch.long
        ).reshape(-1)
        return obs, done_env_ids

    def _estimated_actor_observation(self, observation):
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.shape[1] != self.actor_obs_dim:
            raise RuntimeError(
                "Concurrent player expected 137-D observations, got {}".format(
                    tuple(observation.shape)
                )
            )
        deployable = observation[:, : self.deployable_obs_dim]
        current = deployable[:, : self.estimator_input_dim]
        if self.estimator_history is None:
            self.estimator_history = current.unsqueeze(1).expand(
                -1, self.concurrent_estimator.history_len, -1
            ).clone()
        else:
            self.estimator_history[:, :-1] = self.estimator_history[:, 1:].clone()
            self.estimator_history[:, -1] = current
            if self.pending_reset_ids is not None and self.pending_reset_ids.numel() > 0:
                reset_ids = self.pending_reset_ids
                self.estimator_history[reset_ids] = current[reset_ids].unsqueeze(
                    1
                ).expand(-1, self.concurrent_estimator.history_len, -1)
        self.pending_reset_ids = None
        estimate = self.concurrent_estimator(self.estimator_history)
        return torch.cat((deployable, estimate), dim=-1)

    def get_action(self, obs_dict, is_determenistic=False):
        observation = obs_dict["obs"]
        if self.concurrent_eval_use_estimator:
            with torch.no_grad():
                observation = self._estimated_actor_observation(observation)
        actor_obs = dict(obs_dict)
        actor_obs["obs"] = observation
        return common_player.CommonPlayer.get_action(
            self, actor_obs, is_determenistic
        )
