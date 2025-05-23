'''
SRL训练Agent
Humanoid-SRL训练框架
'''

from isaacgymenvs.utils.torch_jit_utils import to_torch
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv
from rl_games.common.experience import ExperienceBuffer
from rl_games.common.interval_summary_writer import IntervalSummaryWriter
from rl_games.common import common_losses
from rl_games.algos_torch.moving_mean_std import GeneralizedMovingStats
import time
from datetime import datetime
import numpy as np
from torch import optim
import torch 
from torch import nn
import os
from rl_games.algos_torch import central_value
import isaacgymenvs.learning.replay_buffer as replay_buffer
import isaacgymenvs.learning.common_agent as common_agent 
from .. import amp_datasets as amp_datasets
from tensorboardX import SummaryWriter
from rl_games.common import datasets
from gym.spaces.box import Box
from rl_games.algos_torch.self_play_manager import SelfPlayManager
import torch.distributed as dist
from rl_games.common.diagnostics import DefaultDiagnostics, PpoDiagnostics
import gym
from .mp_util import WandbWriter
from collections import deque


def my_safe_load(filename, **kwargs):
    return torch_ext.safe_filesystem_op(torch.load, filename, **kwargs)

def my_load_checkpoint(filename,**kwargs):
    print("=> my loading checkpoint '{}'".format(filename))
    state = my_safe_load(filename, **kwargs)
    return state

class SRLAgent(common_agent.CommonAgent):
    '''
    SRL-Gym v1
    SRL Agent 和 Humanoid Agent使用相同的观测和奖励
    '''
    def __init__(self, base_name, params, cfg = None):
        # super().__init__(base_name, params)

        # CommonAgent.__init__
        a2c_common.A2CBase.__init__(self, base_name, params)

        config = params['config']
        self._load_config_params(config)

        self.is_discrete = False
        self._setup_action_space()
        self.bounds_loss_coef = config.get('bounds_loss_coef', None)
        self.sym_loss_coef = config.get('sym_loss_coef',None)
        self.clip_actions = config.get('clip_actions', True)
        self.network_path = self.nn_dir 
        
        net_config = self._build_net_config(params)  # 构建网络配置
        self.model = self.network.build(net_config,role='humanoid')
        self.model.to(self.ppo_device)
        self.model_srl = self.network.build(net_config,role='srl')
        self.model_srl.to(self.ppo_device)
        self.states = None

        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.last_lr_srl = float(self.last_lr)

        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        self.optimizer_srl = optim.Adam(self.model_srl.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        # if self.has_central_value:
        #     cv_config = {
        #         'state_shape' : torch_ext.shape_whc_to_cwh(self.state_shape), 
        #         'value_size' : self.value_size,
        #         'ppo_device' : self.ppo_device, 
        #         'num_agents' : self.num_agents, 
        #         'num_steps' : self.horizon_length, 
        #         'num_actors' : self.num_actors, 
        #         'num_actions' : self.actions_num, 
        #         'seq_len' : self.seq_len, 
        #         'model' : self.central_value_config['network'],
        #         'config' : self.central_value_config, 
        #         'writter' : self.writer,
        #         'multi_gpu' : self.multi_gpu
        #     }
        #     self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = amp_datasets.AMPDataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        self.dataset_srl = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        self.algo_observer.after_init(self)
        self.scaler_srl = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.game_rewards_srl  = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_rewards_amp  = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_rewards_task = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_rewards_v_p  = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)  # velocity penalty
        self.game_rewards_t_c  = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)  # torque cost
        self.game_rewards_u_r  = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)  # upper reward
        self.game_rewards_srl_t_c  = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)

        # 观测值标准化
        if self.normalize_value:
            # self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std
            self.value_mean_std =  self.model.value_mean_std
            self.value_mean_std_srl =  self.model_srl.value_mean_std

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)

        self._srl_dof = 8

        self.evaluate_rewards = deque(maxlen=6)  # Reward used to evaluate the design
        self.evaluate_amp_rewards = deque(maxlen=6) 
        return

    def init_tensors(self):
        # 初始化张量
        super().init_tensors()

        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }
        env_info = self.env_info
        env_info['action_space'] = Box(-1,1,(self.actions_num_humanoid,1))
        self.experience_buffer = ExperienceBuffer(env_info, algo_info, self.ppo_device)
        env_info['action_space'] = Box(-1,1,(self.actions_num_srl,1))
        
        self.current_rewards_srl = torch.zeros_like(self.current_rewards,dtype=torch.float32, device=self.ppo_device)
        self.current_rewards_amp = torch.zeros_like(self.current_rewards,dtype=torch.float32, device=self.ppo_device)
        self.current_rewards_task = torch.zeros_like(self.current_rewards,dtype=torch.float32, device=self.ppo_device)
        self.current_rewards_v_p = torch.zeros_like(self.current_rewards,dtype=torch.float32, device=self.ppo_device)
        self.current_rewards_t_c = torch.zeros_like(self.current_rewards,dtype=torch.float32, device=self.ppo_device)
        self.current_rewards_u_r = torch.zeros_like(self.current_rewards,dtype=torch.float32, device=self.ppo_device)
        self.current_rewards_srl_t_c = torch.zeros_like(self.current_rewards,dtype=torch.float32, device=self.ppo_device)


        self.experience_buffer_srl = ExperienceBuffer(env_info, algo_info, self.ppo_device)
        self.experience_buffer.tensor_dict['next_obses'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        self.experience_buffer.tensor_dict['next_values'] = torch.zeros_like(self.experience_buffer.tensor_dict['values'])
        self.experience_buffer_srl.tensor_dict['next_obses'] = torch.zeros_like(self.experience_buffer_srl.tensor_dict['obses'])
        self.experience_buffer_srl.tensor_dict['next_values'] = torch.zeros_like(self.experience_buffer_srl.tensor_dict['values'])
        self.experience_buffer_srl.tensor_dict['obs_mirrored'] = torch.zeros_like(self.experience_buffer_srl.tensor_dict['obses'])

        self._build_amp_buffers() # 构建 AMP 缓冲区  

        if self._humanoid_checkpoint :
            self._load_humanoid_network(self._humanoid_checkpoint)    
        if self._hsrl_checkpoint :
            self._load_hsrl_checkpoint(self._hsrl_checkpoint)
            
        if self.mirror_loss:
            self.tensor_list += ['obses_mirrored']
        return
    
    def _load_humanoid_network(self, fn):
        # fn: save path
        # fn = self._humanoid_checkpoint
        checkpoint = my_load_checkpoint(fn,map_location=self.device)
        self.set_weights(checkpoint)
        return

    def set_eval(self):
        super().set_eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()
        return

    def set_train(self):
        super().set_train()
        self.model_srl.train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()  # 设置 AMP 输入标准化为训练模式
        return

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_amp_input:
            state['amp_input_mean_std'] = self._amp_input_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self._normalize_amp_input:
            self._amp_input_mean_std.load_state_dict(weights['amp_input_mean_std'])
        return

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        self.model_srl.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }
        if self._humanoid_obs_masked : # MLY:humanoid观测掩码
            masked_input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : self.mask_humanoid_obs(processed_obs), 
            'rnn_states' : self.rnn_states                
            }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self._humanoid_obs_masked : # humanoid 部分观测  
                # Asymmetric A-C
                # self.model.forward (eval): {"actions", "neglogpacs", "values", "rnn_states", "mus", "sigmas"}
                res_dict_masked = self.model(masked_input_dict)
                res_dict['actions'] = res_dict_masked['actions'] 
                res_dict['neglogpacs'] = res_dict_masked['neglogpacs'] 
                res_dict['mus'] = res_dict_masked['mus'] 
                res_dict['sigmas'] = res_dict_masked['sigmas'] 
            
            res_dict_srl = self.model_srl(input_dict)

        return res_dict, res_dict_srl

    def mask_humanoid_obs(self, obs):
        # root_h 1; root_rot_obs 6; local_root_vel 3 ; local_root_ang_vel 3 ; dof_obs 60; dof_vel 36 ; flat_local_key_pos 12
        mask = torch.ones_like(obs,device=self.ppo_device)
        mask[: , 125: ]  = 0  # SRL dof position
        masked_obs = obs * mask
        return masked_obs
    
    def play_steps(self):
        # 执行一轮完整的经验收集过程 
        self.set_eval()

        epinfos = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs, done_env_ids = self._env_reset_done() # 重置环境
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer_srl.update_data('obses', n, self.obs['obs'])
            

            res_dict, res_dict_srl = self.get_action_values(self.obs) # 获取动作值

            if self.mirror_loss: # 镜像损失
                self.experience_buffer_srl.update_data('obs_mirrored', n, self.obs['obs_mirrored']) # 镜像观测
                mirrored_obs = {}
                mirrored_obs['obs']  =  self.obs['obs_mirrored']
                
            for k in update_list: # 更新经验缓冲区: action
                self.experience_buffer.update_data(k, n, res_dict[k]) 
                self.experience_buffer_srl.update_data(k, n, res_dict_srl[k]) 

            # if self.has_central_value:
            #     self.experience_buffer.update_data('states', n, self.obs['states'])
            
            # action拼接
            conbined_action = torch.cat((res_dict['actions'], res_dict_srl['actions']), dim=-1)
            
            #self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            self.obs, rewards, self.dones, infos = self.env_step(conbined_action)
            
            # # srl reward
            # rewards_srl = infos["srl_rewards"]
            # rewards_srl = rewards_srl.unsqueeze(1)

            ''' humanoid buffer '''
            shaped_rewards = self.rewards_shaper(rewards)  # DefaultRewardsShaper
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])
            
            ''' srl buffer '''
            # shaped_rewards_srl = self.rewards_shaper(rewards_srl)
            self.experience_buffer_srl.update_data('rewards', n, shaped_rewards)
            self.experience_buffer_srl.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer_srl.update_data('dones', n, self.dones)
            if self.mirror_loss:
                self.experience_buffer_srl.update_data('obs_mirrored', n, mirrored_obs['obs'])


            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)

            _velocity_penalty = infos['v_penalty']
            _torque_cost = infos['torque_cost']
            _upper_reward = infos['upper_reward'] 
            _srl_torque_cost = infos['srl_torque_cost']
            
            ''' humanoid value of next state '''
            next_vals = self._eval_critic(self.obs)  
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)
            
            ''' Srl value of next state '''
            # next_vals_srl = self._eval_critic_srl(self.obs)  
            # next_vals_srl *= (1.0 - terminated)
            # self.experience_buffer_srl.update_data('next_values', n, next_vals_srl)
            self.experience_buffer_srl.update_data('next_values', n, next_vals)
            
            # self.current_rewards_srl += rewards_srl

            # calculate AMP reward 
            _amp_rewards = self._calc_amp_rewards(infos['amp_obs']) 
            # calculate total reward
            _total_rewards = self._combine_rewards(rewards,_amp_rewards)
            # store reward
            self.current_rewards_task += rewards     # task reward
            self.current_rewards      += _total_rewards  # task reward + amp reward
            self.current_rewards_amp  += _amp_rewards['disc_rewards']
            self.current_rewards_t_c  += _torque_cost.unsqueeze(1)       # 分量 
            self.current_rewards_v_p  += _velocity_penalty.unsqueeze(1)  # 分量
            self.current_rewards_u_r  += _upper_reward.unsqueeze(1)      # upper reward
            self.current_rewards_srl_t_c += _srl_torque_cost.unsqueeze(1) 

            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[done_indices])
            # self.game_rewards_srl.update(self.current_rewards_srl[done_indices])
            self.game_rewards_amp.update(self.current_rewards_amp[done_indices])
            self.game_rewards_task.update(self.current_rewards_task[done_indices])
            self.game_rewards_v_p.update(self.current_rewards_v_p[done_indices])
            self.game_rewards_t_c.update(self.current_rewards_t_c[done_indices])
            self.game_rewards_u_r.update(self.current_rewards_u_r[done_indices])
            self.game_rewards_srl_t_c.update(self.current_rewards_srl_t_c[done_indices])

            self.current_lengths += 1
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()
            # TODO：可以在这里添加评估指标
            for idx in done_indices:
                ep_lenth = self.current_lengths[idx].item()
                srl_t_c = - self.current_rewards_srl_t_c[idx].item() / ep_lenth
                evaluate_t_c = self.current_rewards_t_c[idx].item() / ep_lenth # torque cose
                evaluate_u_r = self.current_rewards_u_r[idx].item() / ep_lenth # upper reward
                amp_reward = self.current_rewards_amp[idx].item()
                # if amp_reward < 200:
                #     amp_reward = -200 + amp_reward
                # else:
                #     amp_reward = 0
                reward = 2*evaluate_t_c + 2*evaluate_u_r + srl_t_c 
                # 添加新的 reward 到队列中
                self.evaluate_rewards.append(reward)
                self.evaluate_amp_rewards.append(amp_reward)
            ''' If env is done, reset current_reward '''
            
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            # self.current_rewards_srl = self.current_rewards_srl * not_dones.unsqueeze(1)
            self.current_rewards_amp     = self.current_rewards_amp     * not_dones.unsqueeze(1)
            self.current_rewards_task    = self.current_rewards_task    * not_dones.unsqueeze(1)
            self.current_rewards_v_p     = self.current_rewards_v_p     * not_dones.unsqueeze(1)
            self.current_rewards_t_c     = self.current_rewards_t_c     * not_dones.unsqueeze(1)
            self.current_rewards_u_r     = self.current_rewards_u_r     * not_dones.unsqueeze(1)
            self.current_rewards_srl_t_c = self.current_rewards_srl_t_c * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            if (self.vec_env.env.viewer and (n == (self.horizon_length - 1))):
                self._amp_debug(infos)

        # calculate Humanoid minibatch returns
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()  # 获取完成标志
        mb_values = self.experience_buffer.tensor_dict['values'] # 获取价值
        mb_next_values = self.experience_buffer.tensor_dict['next_values'] # 获取下一个价值

        mb_rewards = self.experience_buffer.tensor_dict['rewards'] # 获取奖励
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs'] # 获取 AMP 观测
        amp_rewards = self._calc_amp_rewards(mb_amp_obs) # 计算 AMP 奖励
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards) # 合并奖励

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values) # 折扣价值
        mb_returns = mb_advs + mb_values # 价值 = 当前价值+未来折扣价值

        # calculate SRL minibatch returns
        # mb_fdones_srl = self.experience_buffer_srl.tensor_dict['dones'].float()  # 获取完成标志
        # mb_values_srl = self.experience_buffer_srl.tensor_dict['values'] # 获取价值
        # mb_next_values_srl = self.experience_buffer_srl.tensor_dict['next_values'] # 获取下一个价值

        # mb_rewards_srl = self.experience_buffer_srl.tensor_dict['rewards'] # 获取奖励
        # mb_rewards_srl = self._combine_rewards(mb_rewards_srl, amp_rewards) # 合并奖励

        # mb_advs_srl = self.discount_values(mb_fdones_srl, mb_values_srl, mb_rewards_srl, mb_next_values_srl) # 折扣价值
        # mb_returns_srl = mb_advs_srl + mb_values_srl # 价值 = 当前价值+未来折扣价值

        # humanoid batch
        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list) # 获取转换后的批次字典
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns) # 设置返回值
        batch_dict['played_frames'] = self.batch_size

        # srl batch
        batch_dict_srl = self.experience_buffer_srl.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list) # 获取转换后的批次字典
        batch_dict_srl['returns'] = a2c_common.swap_and_flatten01(mb_returns) # 设置返回值
        batch_dict_srl['obs_mirrored'] = a2c_common.swap_and_flatten01(self.experience_buffer_srl.tensor_dict['obs_mirrored'] ) # 设置返回值
        batch_dict_srl['played_frames'] = self.batch_size

        # humanoid AMP batch
        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        return batch_dict, batch_dict_srl


    def prepare_dataset_srl(self, batch_dict):
        # srl dataset
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
        if self.mirror_loss:
            dataset_dict['obs_mirrored'] = batch_dict['obs_mirrored']

        self.dataset_srl.update_values_dict(dataset_dict)

    def prepare_dataset(self, batch_dict, batch_dict_srl):
        super().prepare_dataset(batch_dict)
        self.prepare_dataset_srl(batch_dict_srl)
        self.dataset.values_dict['amp_obs'] = batch_dict['amp_obs'] # 设置 AMP 观测
        self.dataset.values_dict['amp_obs_demo'] = batch_dict['amp_obs_demo']
        self.dataset.values_dict['amp_obs_replay'] = batch_dict['amp_obs_replay']
        return

    def train_epoch(self):
        play_time_start = time.time()
        with torch.no_grad():
            batch_dict, batch_dict_srl = self.play_steps() 

        play_time_end = time.time()
        prepare_dataset_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)
        
        self._update_amp_demos() # 更新 AMP 示范
        num_obs_samples = batch_dict['amp_obs'].shape[0]
        amp_obs_demo = self._amp_obs_demo_buffer.sample(num_obs_samples)['amp_obs']
        batch_dict['amp_obs_demo'] = amp_obs_demo

        if (self._amp_replay_buffer.get_total_count() == 0):
            batch_dict['amp_obs_replay'] = batch_dict['amp_obs'] # 设置 AMP 重放观测
        else:
            batch_dict['amp_obs_replay'] = self._amp_replay_buffer.sample(num_obs_samples)['amp_obs']

        self.set_train()  # 设置为训练模式

        self.curr_frames = batch_dict.pop('played_frames') #当前帧数
        self.prepare_dataset(batch_dict, batch_dict_srl) # 准备数据集
        self.algo_observer.after_steps()

        # if self.has_central_value:
        #     self.train_central_value()

        train_info = None

        prepare_dataset_time_end = time.time()
        update_time_start = time.time()

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):  
                # 调用 train_actor_critic 方法执行一次训练步骤

                curr_train_info = self.train_actor_critic(self.dataset[i],self.dataset_srl[i])
                if self.schedule_type == 'legacy':
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if (train_info is None):
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)
            
            if 'kl' in train_info:
                av_kls = torch_ext.mean_list(train_info['kl'])
            elif 'kl_srl' in train_info:
                av_kls = torch_ext.mean_list(train_info['kl_srl'])

            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        prepare_dataset_time = prepare_dataset_time_end - prepare_dataset_time_start
        total_time = update_time_end - play_time_start

        self._store_replay_amp_obs(batch_dict['amp_obs']) # 存储 AMP 重放观测

        train_info['play_time'] = play_time
        train_info['prepare_dataset_time'] = prepare_dataset_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        self._record_train_batch_info(batch_dict, train_info) # 记录训练批次信息

        return train_info

    def train_actor_critic(self, input_dict, input_dict_srl):
        self.calc_gradients(input_dict)
        self.calc_gradients_srl(input_dict_srl)
        
        return self.train_result

    def calc_gradients_srl(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values'] # 获取旧的值函数预测
        old_action_log_probs_batch = input_dict['old_logp_actions'] # 获取旧的动作对数概率
        advantage = input_dict['advantages'] # 获取优势
        old_mu_batch = input_dict['mu'] # 获取旧的均值
        old_sigma_batch = input_dict['sigma'] # 获取旧的标准差
        return_batch = input_dict['returns']  # 获取返回值
        actions_batch = input_dict['actions']  # 获取动作
        obs_batch = input_dict['obs'] # 获取观测
        obs_batch = self._preproc_obs(obs_batch) # 预处理观测


        lr = self.last_lr 
        kl = 1.0  # 初始化 KL 散度
        lr_mul = 1.0 # 初始化学习率倍乘因子
        curr_e_clip = lr_mul * self.e_clip # 计算当前的裁剪阈值

        batch_dict = {
            'is_train' : True,
            'prev_actions' : actions_batch, 
            'obs' : obs_batch,
        }

        if self.mirror_loss:
            obs_mirrored_batch = input_dict['obs_mirrored']
            obs_mirrored_batch = self._preproc_obs(obs_mirrored_batch) # 预处理观测
            batch_mirrored_dict = {}
            batch_mirrored_dict['obs'] = obs_mirrored_batch
            _, res_dict_srl_mirrored =  self.get_action_values(batch_mirrored_dict)
            mu_mirrored = res_dict_srl_mirrored['mus']
        rnn_masks = None
        
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model_srl(batch_dict) # 通过模型计算结果
            action_log_probs = res_dict['prev_neglogp']  # 获取动作的对数概率
            values = res_dict['values'] # 获取值函数输出
            entropy = res_dict['entropy'] # 获取熵
            mu = res_dict['mus'] # 获取动作均值
            sigma = res_dict['sigmas'] # 获取动作标准差
            
            # 计算 actor 损失
            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_info['actor_loss_srl'] = a_info['actor_loss']
            del a_info['actor_loss']
            a_loss = a_info['actor_loss_srl']

            # 计算 critic 损失
            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_info['critic_loss_srl'] = c_info['critic_loss']
            del c_info['critic_loss']
            c_loss = c_info['critic_loss_srl']

            # 计算边界损失
            b_loss = self.bound_loss(mu)

            if self.mirror_loss:
                # 计算对称损失
                sym_info = self.sym_loss(mu,mu_mirrored)
                sym_loss = sym_info['sym_loss']
                losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1), sym_loss.unsqueeze(1)], rnn_masks)
                a_loss, c_loss, entropy, b_loss, sym_loss = losses[0], losses[1], losses[2], losses[3], losses[4]
                # 计算总损失
                loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss + self.sym_loss_coef * sym_loss

            else:
                # 应用掩码并计算损失
                losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
                a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]
                
                # 计算总损失
                loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss 
                 
            
            # 梯度清零
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model_srl.parameters():
                    param.grad = None

        # 反向传播和梯度裁剪
        # 缩放损失值，以提高计算精度。然后调用 .backward() 方法进行反向传播，计算每个参数的梯度。
        # self.scaler.scale(loss).backward()  # 1. 缩放损失并进行反向传播，计算梯度
        # self.scaler.step(self.optimizer)    # 2. 调用优化器的 step 方法，更新模型参数，并处理梯度的取消缩放
        # self.scaler.update()                # 3. 更新缩放因子，根据训练情况调整缩放因子以适应下一次迭代

        self.scaler_srl.scale(loss).backward()
         
        if self.truncate_grads:
            # multiGPU 相关代码已删除
            self.scaler_srl.unscale_(self.optimizer_srl)
            nn.utils.clip_grad_norm_(self.model_srl.parameters(), self.grad_norm)
            self.scaler_srl.step(self.optimizer_srl)
            self.scaler_srl.update()    
        else:
            self.scaler_srl.step(self.optimizer_srl)
            self.scaler_srl.update()
        
        # 计算 KL 散度
        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
                    
        srl_train_result = {
            'entropy_srl': entropy,
            'kl_srl': kl_dist,
            'last_lr_srl': self.last_lr, 
            'lr_mul_srl': lr_mul, 
            'b_loss_srl': b_loss
        }
        self.train_result.update(srl_train_result)
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        if self.mirror_loss:
            self.train_result.update(sym_info)

        return        

    def sym_loss(self, mus, mus_mirrored):
         
        # 计算mus和mus_mirrored之间的平方误差
 
        mus_perm = torch.matmul(mus_mirrored, self.vec_env.env.mirror_act_srl_mat)
        loss = torch.mean((mus - mus_perm) ** 2, dim=1)
        sym_info = {}
        sym_info['sym_loss'] = loss
        return sym_info

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values'] # 获取旧的值函数预测
        old_action_log_probs_batch = input_dict['old_logp_actions'] # 获取旧的动作对数概率
        advantage = input_dict['advantages'] # 获取优势
        old_mu_batch = input_dict['mu'] # 获取旧的均值
        old_sigma_batch = input_dict['sigma'] # 获取旧的标准差
        return_batch = input_dict['returns']  # 获取返回值
        actions_batch = input_dict['actions']  # 获取动作
        obs_batch = input_dict['obs'] # 获取观测
        obs_batch = self._preproc_obs(obs_batch) # 预处理观测

        amp_obs = input_dict['amp_obs'][0:self._amp_minibatch_size] # 获取 AMP 观测
        amp_obs = self._preproc_amp_obs(amp_obs) # 预处理 AMP 观测
        amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size] # 获取 AMP 重放观测
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay) # 预处理 AMP 重放观测

        amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]  # 获取 AMP 示范观测
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo) # 预处理 AMP 示范观测
        amp_obs_demo.requires_grad_(True)  # 设置 AMP 示范观测需要梯度

        lr = self.last_lr 
        kl = 1.0  # 初始化 KL 散度
        lr_mul = 1.0 # 初始化学习率倍乘因子
        curr_e_clip = lr_mul * self.e_clip # 计算当前的裁剪阈值

        if self._humanoid_obs_masked:
            masked_obs_batch = self.mask_humanoid_obs(obs_batch)
            masked_batch_dict = {
                'is_train': True,
                'prev_actions': actions_batch, 
                'obs' : masked_obs_batch,
                'amp_obs' : amp_obs,
                'amp_obs_replay' : amp_obs_replay,
                'amp_obs_demo' : amp_obs_demo
            }
        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
            'amp_obs' : amp_obs,
            'amp_obs_replay' : amp_obs_replay,
            'amp_obs_demo' : amp_obs_demo
        }


        rnn_masks = None

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict) # 通过模型计算结果  srl_model forward
            values = res_dict['values'] # 获取值函数输出

            if self._humanoid_obs_masked:  # Asymmetric A-C
                res_dict = self.model(masked_batch_dict) # 通过模型计算结果  srl_model forward
            
            action_log_probs = res_dict['prev_neglogp']  # 获取动作的对数概率
            entropy = res_dict['entropy'] # 获取熵
            mu = res_dict['mus'] # 获取动作均值
            sigma = res_dict['sigmas'] # 获取动作标准差
            disc_agent_logit = res_dict['disc_agent_logit'] # 获取对抗网络输出（代理）
            disc_agent_replay_logit = res_dict['disc_agent_replay_logit'] # 获取对抗网络输出（重放）
            disc_demo_logit = res_dict['disc_demo_logit'] # 获取对抗网络输出（示范）
            
            # 计算 actor 损失
            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']

            # 计算 critic 损失
            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            # 计算边界损失
            b_loss = self.bound_loss(mu)

            # 应用掩码并计算损失
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]
            
            # 计算对抗网络损失
            disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
            disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo)
            disc_loss = disc_info['disc_loss']

            # 计算总损失
            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                 + self._disc_coef * disc_loss
            
            # 梯度清零
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        # 反向传播和梯度裁剪
        # 缩放损失值，以提高计算精度。然后调用 .backward() 方法进行反向传播，计算每个参数的梯度。
        # self.scaler.scale(loss).backward()  # 1. 缩放损失并进行反向传播，计算梯度
        # self.scaler.step(self.optimizer)    # 2. 调用优化器的 step 方法，更新模型参数，并处理梯度的取消缩放
        # self.scaler.update()                # 3. 更新缩放因子，根据训练情况调整缩放因子以适应下一次迭代

        if not self._train_srl_only:
            self.scaler.scale(loss).backward()
             
            if self.truncate_grads:
                # multiGPU相关代码已删除
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
        
        # 计算 KL 散度
        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
                    
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)

        return

    def _actor_loss(self, old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip):
        clip_frac = None
        if (self.ppo):
            # 计算新旧策略概率比 r_t(theta)
            ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
            # 未裁剪的目标函数 surr1
            surr1 = advantage * ratio
            # 裁剪后的目标函数 surr2
            surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip,
                                    1.0 + curr_e_clip)
            # 选择 surr1 和 surr2 的最小值
            a_loss = torch.max(-surr1, -surr2)

            clipped = torch.abs(ratio - 1.0) > curr_e_clip
            clip_frac = torch.mean(clipped.float())
            clip_frac = clip_frac.detach()
        else:
            a_loss = (action_log_probs * advantage)
    
        info = {
            'actor_loss': a_loss,
            'actor_clip_frac': clip_frac
        }
        return info

    def _critic_loss(self, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
        if clip_value:
            # 裁剪后的值函数预测
            value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
            # 未裁剪的值函数损失
            value_losses = (values - return_batch)**2 
            # 选择未裁剪和裁剪后的值函数损失的最大值
            value_losses_clipped = (value_pred_clipped - return_batch)**2
            c_loss = torch.max(value_losses, value_losses_clipped)
        else:
            c_loss = (return_batch - values)**2

        info = {
            'critic_loss': c_loss
        }
        return info

    def _load_config_params(self, config):
        super()._load_config_params(config)

        self.actions_num_humanoid = config['actions_num_humanoid']
        self.actions_num_srl = config['actions_num_srl']

        self._start_frame = config.get('start_frame',0)
        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']

        self._amp_observation_space = self.env_info['amp_observation_space']
        self._amp_batch_size = int(config['amp_batch_size'])
        self._amp_minibatch_size = int(config['amp_minibatch_size'])
        assert(self._amp_minibatch_size <= self.minibatch_size)

        self._disc_coef = config['disc_coef']
        self._disc_logit_reg = config['disc_logit_reg']
        self._disc_grad_penalty = config['disc_grad_penalty']
        self._disc_weight_decay = config['disc_weight_decay']
        self._disc_reward_scale = config['disc_reward_scale']
        self._normalize_amp_input = config.get('normalize_amp_input', True)

        self._train_srl_only = config['train_srl_only']
        self._humanoid_checkpoint = config.get('humanoid_checkpoint',False)
        self._hsrl_checkpoint = config.get('hsrl_checkpoint',False)

        self.mirror_loss = config.get('mirror_loss', False)
        self._humanoid_obs_masked = config.get('humanoid_obs_masked', False)

        return

    def _build_net_config(self, params):
        # 在Common_Agent中定义所有网络：self.model = self.network.build(net_config)
        config = super()._build_net_config()
        config['actions_num_humanoid'] = self.actions_num_humanoid
        config['actions_num_srl'] = self.actions_num_srl
        config['amp_input_shape'] = self._amp_observation_space.shape
        return config

    def _init_train(self):
        super()._init_train()
        self._init_amp_demo_buf()
        return

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if (self._disc_weight_decay != 0):
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty,
            'disc_logit_loss': disc_logit_loss,
            'disc_agent_acc': disc_agent_acc,
            'disc_demo_acc': disc_demo_acc,
            'disc_agent_logit': disc_agent_logit,
            'disc_demo_logit': disc_demo_logit
        }
        return disc_info

    def train(self):
        
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.frame = self._start_frame
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs
 

        self.model_output_file = os.path.join(self.network_path, 
            self.config['name'] + '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
        if self.config.get('model_output_file', False):
            self.model_output_file = self.config.get('model_output_file')

        self._init_train()

        # global rank of the GPU
        # multi-gpu training is not currently supported for AMP
        self.global_rank = int(os.getenv("RANK", "0"))

        while True:
            epoch_num = self.update_epoch()
            train_info = self.train_epoch()

            sum_time = train_info['total_time']
            total_time += sum_time
            frame = self.frame

            if self.global_rank == 0:
                scaled_time = sum_time
                scaled_play_time = train_info['play_time']
                curr_frames = self.curr_frames # 获取当前帧数
                self.frame += curr_frames # 更新总帧数
                if self.print_stats: 
                    fps_step = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(f'fps step: {fps_step:.1f} fps total: {fps_total:.1f}')
                print('logging to tensorboard ... ')
                self.writer.add_scalar('performance/total_fps', curr_frames / scaled_time, frame)
                self.writer.add_scalar('performance/step_fps', curr_frames / scaled_play_time, frame)
                self.writer.add_scalar('info/epochs', epoch_num, frame)
                self._log_train_info(train_info, frame)
                print('logging done ')

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)
                
                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()

                    # mean_rewards_srl = self.game_rewards_srl.get_mean()
                    mean_rewards_amp = self.game_rewards_amp.get_mean()
                    mean_rewards_task = self.game_rewards_task.get_mean()
                    mean_rewards_t_c = self.game_rewards_t_c.get_mean()
                    mean_rewards_v_p = self.game_rewards_v_p.get_mean()
                    mean_rewards_u_r = self.game_rewards_u_r.get_mean()
                    mean_rewards_srl_t_c = self.game_rewards_srl_t_c.get_mean()

                    for i in range(self.value_size):
                        self.writer.add_scalar('rewards/total_frame'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar('rewards/total_iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar('rewards/total_time'.format(i), mean_rewards[i], total_time)
                        # task
                        self.writer.add_scalar('rewards/srl_torque_cost'.format(i), mean_rewards_srl_t_c[i], frame)
                        self.writer.add_scalar('rewards/task'.format(i), mean_rewards_task[i], frame)
                        self.writer.add_scalar('rewards/v_penalty'.format(i), mean_rewards_v_p[i], frame)
                        self.writer.add_scalar('rewards/torque_cost'.format(i), mean_rewards_t_c[i], frame)
                        self.writer.add_scalar('rewards/u_reward'.format(i), mean_rewards_u_r[i], frame)
                        
                        # amp
                        self.writer.add_scalar('rewards/AMP'.format(i), mean_rewards_amp[i], frame)
                    self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    
                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                if self.save_freq > 0:
                    if (epoch_num % self.save_freq == 0):
                        self.save(self.model_output_file + "_" + str(epoch_num))

                if epoch_num > self.max_epochs:
                    self.save(self.model_output_file)
                    print('MAX EPOCHS NUM!')
                    self.writer.close()
                    avg_evaluate_rewards = sum(self.evaluate_rewards) / len(self.evaluate_rewards)
                    avg_evaluate_amp_rewards = sum(self.evaluate_amp_rewards) / len(self.evaluate_amp_rewards)
                    return avg_evaluate_rewards, avg_evaluate_amp_rewards, epoch_num, self.frame, 

                update_time = 0
         
    def _load_hsrl_checkpoint(self, fn,):
        # restore nn of humanoid & srl
        checkpoint = my_load_checkpoint(fn,map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model_srl.load_state_dict(checkpoint['model_srl'])

    def save(self, fn):
        state = self.get_full_state_weights()
        state['model_srl'] = self.model_srl.state_dict()
        state['optimizer_srl'] = self.optimizer_srl.state_dict()
        torch_ext.save_checkpoint(fn, state)

    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss
    
    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _fetch_amp_obs_demo(self, num_samples):
        amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo(num_samples)
        return amp_obs_demo

    def _build_amp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['amp_obs'] = torch.zeros(batch_shape + self._amp_observation_space.shape,
                                                                    device=self.ppo_device)
        
        amp_obs_demo_buffer_size = int(self.config['amp_obs_demo_buffer_size'])
        self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)

        self._amp_replay_keep_prob = self.config['amp_replay_keep_prob']
        replay_buffer_size = int(self.config['amp_replay_buffer_size'])
        self._amp_replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)

        self.tensor_list += ['amp_obs']
        return

    def _init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._amp_batch_size))

        for i in range(num_batches):
            curr_samples = self._fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({'amp_obs': curr_samples})

        return
    
    def _update_amp_demos(self):
        new_amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
        self._amp_obs_demo_buffer.store({'amp_obs': new_amp_obs_demo})
        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards['disc_rewards']
        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_r
        return combined_rewards

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
            disc_r *= self._disc_reward_scale
        return disc_r

    def _store_replay_amp_obs(self, amp_obs):
        buf_size = self._amp_replay_buffer.get_buffer_size()
        buf_total_count = self._amp_replay_buffer.get_total_count()
        if (buf_total_count > buf_size):
            keep_probs = to_torch(np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]), device=self.ppo_device)
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]

        self._amp_replay_buffer.store({'amp_obs': amp_obs})
        return

    def _record_train_batch_info(self, batch_dict, train_info):
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        self.writer.add_scalar('performance/total_time',train_info['total_time'], frame)
        self.writer.add_scalar('performance/prepare_dataset_time', train_info['prepare_dataset_time'], frame)

        self.writer.add_scalar('losses/disc_loss', torch_ext.mean_list(train_info['disc_loss']).item(), frame)

        self.writer.add_scalar('info/disc_agent_acc', torch_ext.mean_list(train_info['disc_agent_acc']).item(), frame)
        self.writer.add_scalar('info/disc_demo_acc', torch_ext.mean_list(train_info['disc_demo_acc']).item(), frame)
        self.writer.add_scalar('info/disc_agent_logit', torch_ext.mean_list(train_info['disc_agent_logit']).item(), frame)
        self.writer.add_scalar('info/disc_demo_logit', torch_ext.mean_list(train_info['disc_demo_logit']).item(), frame)
        self.writer.add_scalar('info/disc_grad_penalty', torch_ext.mean_list(train_info['disc_grad_penalty']).item(), frame)
        self.writer.add_scalar('info/disc_logit_loss', torch_ext.mean_list(train_info['disc_logit_loss']).item(), frame)

        disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
        self.writer.add_scalar('info/disc_reward_mean', disc_reward_mean.item(), frame)
        self.writer.add_scalar('info/disc_reward_std', disc_reward_std.item(), frame)

        self.writer.add_scalar('losses/a_loss_srl', torch_ext.mean_list(train_info['actor_loss_srl']).item(), frame)
        self.writer.add_scalar('losses/c_loss_srl', torch_ext.mean_list(train_info['critic_loss_srl']).item(), frame)
        
        self.writer.add_scalar('losses/bounds_loss_srl', torch_ext.mean_list(train_info['b_loss_srl']).item(), frame)
        self.writer.add_scalar('losses/entropy_srl', torch_ext.mean_list(train_info['entropy_srl']).item(), frame)
        # self.writer.add_scalar('info/last_lr', train_info['last_lr'][-1] * train_info['lr_mul'][-1], frame)
        # self.writer.add_scalar('info/lr_mul', train_info['lr_mul'][-1], frame)
        # self.writer.add_scalar('info/e_clip', self.e_clip * train_info['lr_mul'][-1], frame)
        # self.writer.add_scalar('info/clip_frac', torch_ext.mean_list(train_info['actor_clip_frac']).item(), frame)
        # self.writer.add_scalar('info/kl', torch_ext.mean_list(train_info['kl']).item(), frame)
        if self.mirror_loss:
            self.writer.add_scalar('losses/sym_loss_srl', torch_ext.mean_list(train_info['sym_loss']).item(), frame)
        return

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward)
        return
    
    def _eval_critic_srl(self, obs_dict):
        self.model_srl.eval()
        obs = obs_dict['obs']

        processed_obs = self._preproc_obs(obs)
        if self.normalize_input:
            processed_obs = self.model_srl.norm_obs(processed_obs)
        value = self.model_srl.a2c_network.eval_critic(processed_obs)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value