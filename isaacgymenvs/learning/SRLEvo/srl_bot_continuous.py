'''
srl_bot_continuous.py
用于训练单独SRL Bot
'''
from rl_games.algos_torch import a2c_continuous
from rl_games.common import a2c_common, common_losses
from rl_games.algos_torch import torch_ext
from rl_games.common.a2c_common import print_statistics
import time
import torch
import numpy as np
import time
import copy
import os
import torch.distributed as dist

from isaacgymenvs.learning.SRLEvo.privileged_estimator import (
    load_privileged_estimator,
)
from isaacgymenvs.learning.SRLEvo.privileged_estimator_adapter import (
    EstimatedObservationAdapter,
)

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

class SRL_Bot_Agent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        a2c_continuous.A2CAgent.__init__(self, base_name, params)
        config = params['config']
        self.a_sym_loss_coef = config.get('a_sym_loss_coef', None)
        self.train_with_privileged_estimator = bool(
            config.get('train_with_privileged_estimator', False)
        )
        if self.has_central_value:
            # The actor sees hardware-realistic observations, while the value
            # baseline is trained from privileged simulator states.
            self.critic_coef = 0.0

        self.init_central_from_legacy = bool(
            config.get(
                'train_privileged_estimator_init_central_from_legacy', False
            )
        )
        self.reset_optimizer_on_legacy_init = bool(
            config.get(
                'train_privileged_estimator_reset_optimizer_on_legacy_init',
                False,
            )
        )
        self.estimator_train_alpha_start = float(
            config.get('estimator_train_alpha_start', 1.0)
        )
        self.estimator_train_alpha_end = float(
            config.get('estimator_train_alpha_end', 1.0)
        )
        self.estimator_train_alpha_ramp_start_epoch = int(
            config.get('estimator_train_alpha_ramp_start_epoch', -1)
        )
        self.estimator_train_alpha_ramp_end_epoch = int(
            config.get('estimator_train_alpha_ramp_end_epoch', -1)
        )
        self.privileged_estimator_model = None
        self.privileged_estimator_state = {}
        self.privileged_estimator_adapter = None
        self.estimator_rollout_metrics = {}

        if self.train_with_privileged_estimator:
            if not self.has_central_value:
                raise ValueError(
                    "train_with_privileged_estimator requires central_value_config"
                )
            if self.is_rnn:
                raise NotImplementedError(
                    "Estimator PPO fine-tuning currently supports feed-forward actors only"
                )
            for name, value in (
                ('estimator_train_alpha_start', self.estimator_train_alpha_start),
                ('estimator_train_alpha_end', self.estimator_train_alpha_end),
            ):
                if not 0.0 <= value <= 1.0:
                    raise ValueError("{} must be in [0, 1]".format(name))

            estimator_checkpoint = config.get(
                'train_privileged_estimator_checkpoint', ''
            )
            if not estimator_checkpoint:
                raise ValueError(
                    "train_privileged_estimator_checkpoint must be provided when "
                    "train_with_privileged_estimator=True"
                )
            self.privileged_estimator_model, self.privileged_estimator_state = (
                load_privileged_estimator(
                    estimator_checkpoint, device=self.ppo_device
                )
            )
            self.privileged_estimator_model.eval()
            for parameter in self.privileged_estimator_model.parameters():
                parameter.requires_grad_(False)
            print(
                "Frozen privileged estimator enabled: checkpoint={} epoch={} "
                "alpha={:.3f}->{:.3f} ramp_epochs={}->{}".format(
                    estimator_checkpoint,
                    int(self.privileged_estimator_state.get('epoch', -1)),
                    self.estimator_train_alpha_start,
                    self.estimator_train_alpha_end,
                    self.estimator_train_alpha_ramp_start_epoch,
                    self.estimator_train_alpha_ramp_end_epoch,
                )
            )

    def _initialize_central_value_from_legacy(self, weights):
        source_state = dict(weights['model'])

        # Some rl-games releases stored normalization statistics both inside
        # model and as top-level checkpoint entries. Accept either layout.
        for checkpoint_key, model_prefix in (
            ('running_mean_std', 'running_mean_std.'),
            ('reward_mean_std', 'value_mean_std.'),
        ):
            stats = weights.get(checkpoint_key)
            if stats is not None:
                for key, value in stats.items():
                    source_state.setdefault(model_prefix + key, value)

        target_state = self.central_value_net.model.state_dict()
        transferable_prefixes = (
            'a2c_network.actor_cnn.',
            'a2c_network.actor_mlp.',
            'a2c_network.value.',
            'running_mean_std.',
            'value_mean_std.',
        )
        required_keys = [
            key for key in target_state
            if key.startswith(transferable_prefixes)
        ]
        mlp_keys = [
            key for key in required_keys
            if key.startswith('a2c_network.actor_mlp.')
        ]
        value_keys = [
            key for key in required_keys
            if key.startswith('a2c_network.value.')
        ]
        if not mlp_keys or not value_keys:
            raise RuntimeError(
                "Central critic does not expose the expected actor_mlp and "
                "value parameters; legacy value migration is unsupported"
            )

        missing = [key for key in required_keys if key not in source_state]
        mismatched = [
            key for key in required_keys
            if key in source_state
            and tuple(source_state[key].shape) != tuple(target_state[key].shape)
        ]
        if missing or mismatched:
            raise RuntimeError(
                "Cannot initialize central critic from legacy value network. "
                "Missing keys: {}. Shape mismatches: {}".format(
                    missing, mismatched
                )
            )

        for key in required_keys:
            target_state[key] = source_state[key].detach().clone()
        self.central_value_net.model.load_state_dict(target_state, strict=True)

        group_counts = {}
        for name, prefix in (
            ('mlp', 'a2c_network.actor_mlp.'),
            ('value', 'a2c_network.value.'),
            ('input_rms', 'running_mean_std.'),
            ('value_rms', 'value_mean_std.'),
        ):
            group_counts[name] = sum(
                key.startswith(prefix) for key in required_keys
            )
        print(
            "Initialized central critic from legacy value network: "
            "mlp={} value={} input_rms={} value_rms={}".format(
                group_counts['mlp'],
                group_counts['value'],
                group_counts['input_rms'],
                group_counts['value_rms'],
            )
        )
        return self.central_value_net.state_dict()

    def set_full_state_weights(self, weights, set_epoch=True):
        should_migrate_legacy_value = (
            self.train_with_privileged_estimator
            and self.init_central_from_legacy
            and self.has_central_value
            and 'assymetric_vf_nets' not in weights
        )
        if should_migrate_legacy_value:
            weights = copy.copy(weights)
            weights['assymetric_vf_nets'] = (
                self._initialize_central_value_from_legacy(weights)
            )
            if self.reset_optimizer_on_legacy_init:
                weights['optimizer'] = self.optimizer.state_dict()
                print(
                    "Reset actor optimizer while restoring legacy policy "
                    "weights"
                )
        return super().set_full_state_weights(weights, set_epoch=set_epoch)

    def _get_estimator_train_alpha(self):
        start = self.estimator_train_alpha_ramp_start_epoch
        end = self.estimator_train_alpha_ramp_end_epoch
        if start < 0 or end <= start:
            return self.estimator_train_alpha_start
        epoch = int(self.epoch_num)
        if epoch <= start:
            return self.estimator_train_alpha_start
        if epoch >= end:
            return self.estimator_train_alpha_end
        fraction = float(epoch - start) / float(end - start)
        return (
            self.estimator_train_alpha_start
            + fraction
            * (self.estimator_train_alpha_end - self.estimator_train_alpha_start)
        )

    def _build_estimator_actor_observation(self, raw_obs, done_env_ids):
        observation = raw_obs['obs']
        mirrored_observation = raw_obs['obs_mirrored']
        if observation.ndim != 2 or observation.shape[1] != 153:
            raise RuntimeError(
                "Estimator PPO fine-tuning requires a 153D actor. Set "
                "task.env.srl_policy_obs_remove_ids=[]; got {}".format(
                    tuple(observation.shape)
                )
            )
        if raw_obs['states'].ndim != 2 or raw_obs['states'].shape[1] != 153:
            raise RuntimeError(
                "Estimator PPO fine-tuning requires 153D critic states; got {}".format(
                    tuple(raw_obs['states'].shape)
                )
            )

        num_envs = observation.shape[0]
        if self.privileged_estimator_adapter is None:
            self.privileged_estimator_adapter = EstimatedObservationAdapter(
                self.privileged_estimator_model,
                num_envs=num_envs,
                device=observation.device,
            )
        elif self.privileged_estimator_adapter.num_envs != num_envs:
            raise RuntimeError(
                "PPO environment count changed from {} to {}".format(
                    self.privileged_estimator_adapter.num_envs, num_envs
                )
            )

        first = torch.zeros(num_envs, dtype=torch.bool, device=observation.device)
        if len(done_env_ids) > 0:
            reset_ids = torch.as_tensor(
                done_env_ids, device=observation.device, dtype=torch.long
            ).reshape(-1)
            first[reset_ids] = True

        alpha = self._get_estimator_train_alpha()
        actor_observation, mirrored_actor_observation, estimate = (
            self.privileged_estimator_adapter.transform_pair(
                observation,
                mirrored_observation,
                first,
                alpha=alpha,
            )
        )
        actor_obs = dict(raw_obs)
        actor_obs['obs'] = actor_observation
        actor_obs['obs_mirrored'] = mirrored_actor_observation
        return actor_obs, estimate, alpha

    def init_tensors(self):
        super().init_tensors()
        self.experience_buffer.tensor_dict['obs_mirrored'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0
        estimator_abs_error_sum = torch.zeros(4, device=self.ppo_device)
        estimator_error_count = 0
        estimator_alpha = 0.0

        for n in range(self.horizon_length):
            self.obs, done_env_ids = self._env_reset_done() # 重置环境
            raw_obs = self.obs
            if self.train_with_privileged_estimator:
                actor_obs, estimate, estimator_alpha = (
                    self._build_estimator_actor_observation(raw_obs, done_env_ids)
                )
                clean_target = raw_obs['states'][:, :150].reshape(
                    estimate.shape[0], 5, 30
                )[:, 0, :4]
                estimator_abs_error_sum += (estimate - clean_target).abs().sum(dim=0)
                estimator_error_count += estimate.shape[0]
            else:
                actor_obs = raw_obs

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(actor_obs, masks)
            else:
                res_dict = self.get_action_values(actor_obs)
            if self.has_central_value:
                res_dict['values'] = self.get_central_value({'states': raw_obs['states']})

            self.experience_buffer.update_data('obses', n, actor_obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            # mirrored_obs
            self.experience_buffer.update_data(
                'obs_mirrored', n, actor_obs['obs_mirrored']
            )
            mirrored_obs = {}
            mirrored_obs['obs']  =  actor_obs['obs_mirrored']

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, raw_obs['states'])

            # simulation step
            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]
     
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        if self.train_with_privileged_estimator:
            self.estimator_rollout_metrics = {
                'alpha': float(estimator_alpha),
                'mae': (
                    estimator_abs_error_sum / max(estimator_error_count, 1)
                ).detach().cpu(),
            }

        if self.has_central_value:
            last_values = self.get_central_value({'states': self.obs['states']})
        else:
            last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time
        batch_dict['obs_mirrored'] = a2c_common.swap_and_flatten01(self.experience_buffer.tensor_dict['obs_mirrored'] ) # 设置返回值

        return batch_dict

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []
        a_sym_losses = []
        c_sym_losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss, a_sym_loss = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                a_sym_losses.append(a_sym_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)
                if self.schedule_type == 'legacy':
                    av_kls = kl
                    if self.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.world_size
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.world_size
            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, a_sym_losses, entropies, kls, last_lr, lr_mul



    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, sym_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            if self.global_rank == 0:
                self.diagnostics.epoch(self, current_epoch = epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames

                print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
                                epoch_num, self.max_epochs, frame, self.max_frames)

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame,
                                scaled_time, scaled_play_time, curr_frames)

                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)
                
                self.writer.add_scalar('losses/sym_loss', torch_ext.mean_list(sym_losses).item(), frame)
                if self.train_with_privileged_estimator:
                    self.writer.add_scalar(
                        'estimator_train/alpha',
                        self.estimator_rollout_metrics['alpha'],
                        frame,
                    )
                    for index, name in enumerate(
                        ('root_height', 'local_vx', 'local_vy', 'local_vz')
                    ):
                        self.writer.add_scalar(
                            'estimator_train/{}_mae'.format(name),
                            self.estimator_rollout_metrics['mae'][index].item(),
                            frame,
                        )
                # if self.has_soft_aug:
                #     self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)
                        self.writer.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i], epoch_num)
                        self.writer.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if epoch_num % self.save_freq == 0:
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

            if should_exit:
                return self.last_mean_rewards, epoch_num


    def get_mirrored_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)

        return   res_dict

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        # mirror loss
        obs_mirrored_batch = input_dict['obs_mirrored']
        obs_mirrored_batch = self._preproc_obs(obs_mirrored_batch) # 预处理观测
        batch_mirrored_dict = {}
        batch_mirrored_dict['obs'] = obs_mirrored_batch
        res_dict_srl_mirrored =  self.get_mirrored_action_values(batch_mirrored_dict)
        mu_mirrored = res_dict_srl_mirrored['mus']

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']            

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model,value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)

            # Actor symmetric loss
            actor_sym_info = self.sym_loss(mu,mu_mirrored)
            actor_sym_loss = actor_sym_info['sym_loss']

            # Critic symmetric loss
            # res_mirrored_train = self.model(batch_mirrored_train)
            # values_sym = res_mirrored_train['values']        # V(g·s), shape [B,1]
            # critic_sym_loss = (values_sym - return_batch.detach()) ** 2

            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1), actor_sym_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss, actor_sym_loss,  = losses[0], losses[1], losses[2], losses[3], losses[4]

            loss = a_loss \
                   + 0.5 * c_loss * self.critic_coef \
                   - entropy * self.entropy_coef \
                   + b_loss * self.bounds_loss_coef \
                   + self.a_sym_loss_coef * actor_sym_loss
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss, actor_sym_loss)
        
    def sym_loss(self, mus, mus_mirrored):
        # 计算mus和mus_mirrored之间的平方误差
        mus_perm = torch.matmul(mus_mirrored, self.vec_env.env.mirror_mat_srl_dof)
        loss = torch.mean((mus - mus_perm) ** 2, dim=1)
        sym_info = {}
        sym_info['sym_loss'] = loss
        return sym_info
    
    def _env_reset_done(self):
        obs, done_env_ids = self.vec_env.reset_done()
        return self.obs_to_tensors(obs), done_env_ids
    
    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = returns - values

        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas
        dataset_dict['obs_mirrored'] = batch_dict['obs_mirrored']

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['dones'] = dones
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)
