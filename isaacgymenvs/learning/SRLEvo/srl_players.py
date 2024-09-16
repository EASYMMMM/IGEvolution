# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch 

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.players import rescale_actions
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer
from rl_games.common.tr_helpers import unsqueeze_obs
import isaacgymenvs.learning.common_player as common_player
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def my_safe_load(filename, **kwargs):
    return torch_ext.safe_filesystem_op(torch.load, filename, **kwargs)

def my_load_checkpoint(filename,**kwargs):
    print("=> my loading checkpoint '{}'".format(filename))
    state = my_safe_load(filename, **kwargs)
    return state

class SRLPlayerContinuous(common_player.CommonPlayer):

    def __init__(self, params):
        config = params['config']

        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._disc_reward_scale = config['disc_reward_scale']
        self._print_disc_prediction = config.get('print_disc_prediction', False)
        self.actions_num_humanoid = config.get('actions_num_humanoid')
        super().__init__(params)

        
        return

    def restore(self, fn):
        
        checkpoint = my_load_checkpoint(fn,map_location='cuda:0')
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

        self.model_srl.load_state_dict(checkpoint['model_srl'])
        if self._normalize_amp_input:
            self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        return
    
    def _build_net(self, config):
        #super()._build_net(config)
        self.model = self.network.build(config,role='humanoid')
        self.model.to(self.device)
        self.model_srl = self.network.build(config,role='srl')
        self.model_srl.to(self.device)

        self.model.eval()
        self.model_srl.eval()
        self.is_rnn = self.model.is_rnn()

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()
        return

    def get_action(self, obs_dict, is_deterministic = False):
        obs = obs_dict['obs']
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
            res_dict_srl = self.model_srl(input_dict)
        mu_humanoid = res_dict['mus']
        action_humanoid = res_dict['actions']
        mu_srl = res_dict_srl['mus']
        action_srl = res_dict_srl['actions']
        mu = torch.cat((mu_humanoid,mu_srl),1)
        action = torch.cat((action_humanoid, action_srl),1)
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action
        
    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        # 存储第一个环境的动作数据
        actions_env0 = []
        episode_count_env0 = 0
        # 新增：为每个 episode 创建列表来存储数据
        episode_data = {
            'root_pos': [],
            'srl_end_pos': [],
            'key_body_pos': [],
            'dof_forces':[],
            'obs':[],
            'done':[],
        }

        need_init_rnn = self.is_rnn
        print('Start Playing')
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obs_dict = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            episode_actions = []
            episode_velocity = []

            for n in range(self.max_steps):
                obs_dict, done_env_ids = self._env_reset_done()

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    action = self.get_action(obs_dict, is_determenistic)
                obs_dict, r, done, info =  self.env_step(self.env, action)
                cr += r
                steps += 1
  
                self._post_step(info)
                
                # 只记录第一个环境的动作
                dof_forces = info["dof_forces"]
                episode_actions.append(dof_forces[0].cpu().numpy())  # 假设动作输出是Tensor
                episode_velocity.append(info["x_velocity"][0].cpu().numpy()) # 

                # 记录第一个智能体的肢体位置数据
                root_pos = info["root_pos"].cpu().numpy()
                srl_end_pos = info["srl_end_pos"].cpu().numpy()
                key_body_pos = info["key_body_pos"].cpu().numpy()
                dof_pos = info["dof_pos"].cpu().numpy()
                # 将这些数据分别存储在当前 episode 的对应列表中
                episode_data['root_pos'].append(root_pos)
                episode_data['srl_end_pos'].append(srl_end_pos)
                episode_data['key_body_pos'].append(key_body_pos)
                episode_data['dof_forces'].append(dof_forces[0].cpu().numpy())
                episode_data['obs'].append( dof_pos)
                
                if render:
                    self.env.render(mode = 'human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count
                if 0 in done_indices:
                    episode_data['done'].append(1)
                else:
                    episode_data['done'].append(0)

                if done_count > 0:
                    if 0 in done_indices: # 第一个环境结束
                        print('Env-0 end ')
                        print('Env-0 average velocity (x):',np.mean(episode_velocity))
                        episode_velocity = []
                        actions_env0.append(episode_actions)
                        episode_count_env0 += 1
                        games_played += 1

                        # 只保存当前 episode 的数据
                        data_to_save = {
                            f'episode_root_pos': episode_data['root_pos'],
                            f'episode_srl_end_pos': episode_data['srl_end_pos'],
                            f'episode_key_body_pos': episode_data['key_body_pos'],
                            f'episode_dof_forces': episode_data['dof_forces'],
                            f'episode_obs': episode_data['obs'],
                            f'episode_dones': episode_data['done'],
                        }

                        print(f"Episode {episode_count_env0} Data saved.")
                        if episode_count_env0 == 3:
                            sio.savemat('run_data/GA314_best_env0_episode_data.mat', data_to_save)
                            print("已保存env0的前三个episode的数据到env0_episode_data.mat")

                        # 当第一个环境完成两个episode时，绘制动作曲线
                        # if episode_count_env0 == 1:
                        #     self.plot_actions(actions_env0)
                        # if episode_count_env0 == 3:
                        #     self.action0_ave(actions_env0)

                    if self.is_rnn:
                        for s in self.states:
                            s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if self.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count)

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)

        return

    def action0_ave(self, actions_env0):
        # 计算输出动作的均值和方差
        # 转换成Numpy数组以方便计算
        actions_env0 = np.array(actions_env0)  # shape: (5, num_steps, num_joints)

        # 计算平均值和方差
        mean_actions = np.mean(actions_env0, axis=(0, 1))  # 平均值
        var_actions = np.var(actions_env0, axis=(0, 1))  # 方差

        joint_names = self.env.dof_names

        # 输出结果
        for index, (mean, var) in enumerate(zip(mean_actions, var_actions)):
            print(f"Joint {joint_names[index]}: Mean = {mean:.5f}, Variance = {var:.5f}")
            
    def plot_actions(self, actions_env0, ):
        joint_names = self.env.dof_names
        draw_list = [ ['right_hip_x', 'right_hip_y', 'right_hip_z',
                     'right_knee', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z'],
                     [ 'left_hip_x', 'left_hip_y', 'left_hip_z', 
                     'left_knee', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z',]] 
        srl_draw_list = [name for name in joint_names if 'SRL' in name]
        draw_list.append(srl_draw_list)
        # 获取关节名称，假设它们可以通过某个函数获得
        
        # 创建一个字典，键是关节名称，值是关节的索引
        joint_indices = {name: i for i, name in enumerate(joint_names)}
        
        for d_l in draw_list:
            # 确定需要绘制的关节的索引
            indices_to_draw = [joint_indices[name] for name in d_l if name in joint_indices]

            # plt.figure(figsize=(12, 8))
            plt.figure()
            for episode_index, actions in enumerate(actions_env0):
                plt.subplot(len(actions_env0), 1, episode_index + 1)
                for joint_index in indices_to_draw:
                    t = actions[joint_index]
                    action_ave = t.sum().item() / len(actions[joint_index])
                    plt.plot([action[joint_index] for action in actions], label=f'{joint_names[joint_index]} ave={action_ave:.5f}')
                    plt.title(f'Episode {episode_index + 1} Actions')
                    plt.xlabel('Time Step')
                    plt.ylabel('Action Value')
                    plt.legend()
            plt.tight_layout()
            plt.show()

        

    def _post_step(self, info):
        super()._post_step(info)
        if self._print_disc_prediction:
            self._amp_debug(info)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
        config['actions_num_humanoid'] = self.actions_num_humanoid
        self.actions_num_srl = - self.actions_num_humanoid + self.actions_num
        config['actions_num_srl'] = self.actions_num_srl
        return config

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs.to(self.device))
            amp_rewards = self._calc_amp_rewards(amp_obs.to(self.device))
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward)
        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

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
            prob = 1.0 / (1.0 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r
