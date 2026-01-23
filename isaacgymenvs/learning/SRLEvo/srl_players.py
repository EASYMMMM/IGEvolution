'''
SRL-Gym
训练结果演示
'''
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
from scipy.signal import butter, filtfilt

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
        # ---- SRL-Gym Defined ----
        self.seperate_obs = config.get('seperate_obs',False)
        self.obs_num_humanoid = config.get('obs_num_humanoid',False)
        self.obs_num_srl = config.get('obs_num_srl',False)
        self._save_data = config.get('save_data', False)
        self._save_load_cell_data = config.get('save_load_cell_data', False)
        self._humanoid_obs_masked = config.get('humanoid_obs_masked', False)
        self.normalize_srl_input = config.get('normalize_srl_input', False)
        self.srl_units = params['network']['mlp']['srl_units']
        
        super().__init__(params)
        self.obs_log = []
        self.target_yaw_log = []
        self.load_cell_fd_log = []
        self.srl_torques_log = []
        self.srl_virtual_passive_pos_log = []
        self.srl_virtual_load_cell_log = []
        self.load_cell_log = []
        self.obs_num_humanoid = self.env.get_humanoid_obs_size()
        self.obs_num_srl = self.env.get_srl_obs_size()
        self.priv_obs_num_srl = self.env.get_srl_priv_obs_size()

        self.play_deterministic = config.get('play_deterministic', False)
        
        return

    def restore(self, fn):
        
        checkpoint = my_load_checkpoint(fn,map_location='cuda:0')
        self.model.load_state_dict(checkpoint['model'])
        print("=> loaded humanoid checkpoint '{}' (iter {})".format(fn, checkpoint.get('iter', 0)))
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

        self.model_srl.load_state_dict(checkpoint['model_srl'])
        print("=> loaded srl checkpoint '{}' (iter {})".format(fn, checkpoint.get('iter', 0)))

        if self._normalize_amp_input:
            self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        return
    
    def _build_net(self, config):
        #super()._build_net(config)

        config['input_shape'] = self.env.get_humanoid_obs_size()
        self.model = self.network.build(config,role='humanoid')
        self.model.to(self.device)
        
        self.network.network_builder.params['mlp']['units'] = self.srl_units
        config['input_shape'] = self.env.get_srl_obs_size()
        config['normalize_input'] = self.normalize_srl_input 
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
        self.model.eval() # humanoid policy
        self.model_srl.eval() # srl policy
        processed_obs = self._preproc_obs(obs)
        if not self.obs_num_humanoid+self.obs_num_srl == processed_obs.shape[1]:
            raise ValueError
        humanoid_obs = processed_obs[:, :self.obs_num_humanoid]
        srl_obs = processed_obs[:, -self.obs_num_srl:]
        priv_srl_obs = processed_obs[:, -self.priv_obs_num_srl:]
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : humanoid_obs,
            'rnn_states' : self.states
        }
        srl_input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : srl_obs,
            'rnn_states' : self.states,
            'priv_obs': priv_srl_obs 
        }

 
        with torch.no_grad():
            res_dict = self.model(input_dict)
            res_dict_srl = self.model_srl(srl_input_dict)
        mu_humanoid = res_dict['mus']
        action_humanoid = res_dict['actions']
        mu_srl = res_dict_srl['mus']
        action_srl = res_dict_srl['actions']
        mu = torch.cat((mu_humanoid,mu_srl),1)
        action = torch.cat((action_humanoid, action_srl),dim=-1)
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
        with torch.no_grad():
            n_games = self.games_num
            render = self.render_env
            n_game_life = self.n_game_life
            is_determenistic = self.play_deterministic
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
                'srl_end_vel': [],
                'key_body_pos': [],
                'action':[],
                'obs':[],
                'done':[],
            }

            load_cell_data = []

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
                cr_srl = torch.zeros(batch_size, dtype=torch.float32)
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
                    rewards_srl = info["srl_rewards"].cpu()
                    cr_srl += rewards_srl

                    obs = obs_dict['obs']  # shape: [num_envs, obs_dim]
                    if isinstance(obs, torch.Tensor):
                        obs_np = obs.detach().cpu().numpy()[0, -self.obs_num_srl:]  # 取第一个环境
                    else:
                        obs_np = np.array(obs[0, :])
                    self.obs_log.append(obs_np)
                    self.target_yaw_log.append(info['target_yaw'].cpu().numpy())
                    self.load_cell_fd_log.append(info['load_cell_fd'].cpu().numpy())
                    self.load_cell_log.append(info['load_cell'].cpu().numpy())
                    self.srl_virtual_passive_pos_log.append(info['srl_virtual_passive_pos'].cpu().numpy())
                    self.srl_virtual_load_cell_log.append(info['srl_virtual_load_cell'].cpu().numpy())
                    self.srl_torques_log.append(info['srl_torques'].cpu().numpy())
                    self._post_step(info)
                    
                    # 只记录第一个环境的动作
                    episode_actions.append(action[0].cpu().numpy())  # 假设动作输出是Tensor
                    episode_velocity.append(info["x_velocity"][0].cpu().numpy()) # 

                    # 记录第一个智能体的肢体位置数据
                    root_pos = info["root_pos"].cpu().numpy()
                    srl_end_pos = info["srl_end_pos"].cpu().numpy()
                    srl_end_vel = info["srl_end_vel"].cpu().numpy()
                    key_body_pos = info["key_body_pos"].cpu().numpy()
                    dof_pos = info["dof_pos"].cpu().numpy()
                    # 将这些数据分别存储在当前 episode 的对应列表中
                    episode_data['root_pos'].append(root_pos)
                    episode_data['srl_end_pos'].append(srl_end_pos)
                    episode_data['srl_end_vel'].append(srl_end_vel)
                    episode_data['key_body_pos'].append(key_body_pos)
                    episode_data['action'].append(action[0].cpu().numpy())
                    episode_data['obs'].append( dof_pos)

                    if self._save_load_cell_data:
                        # MLY: 绘图时选择交互位置六轴力传感器或者足部数据
                        # load_cell_val = info["load_cell"].cpu().numpy()
                        load_cell_val = info["right_srl_end_sensor"].cpu().numpy()
                        load_cell_data.append(load_cell_val)
                    
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

                    # FIXME: 是否画图
                    if done_count > 0: # if done_count > 0:
                        if 0 in done_indices:
                            # 转为 numpy 数组，shape: [T, D]
                            target_yaw = []
                            obs_array = np.stack(self.obs_log, axis=0)
                            target_yaw = np.stack([t if isinstance(t, np.ndarray) else t.cpu().numpy() for t in self.target_yaw_log], axis=0)
                            load_cell_fd_array = np.stack([t if isinstance(t, np.ndarray) else t.cpu().numpy() for t in self.load_cell_fd_log], axis=0)
                            load_cell_array = np.stack([t if isinstance(t, np.ndarray) else t.cpu().numpy() for t in self.load_cell_log], axis=0)
                            srl_virtual_passive_pos_array = np.stack([t if isinstance(t, np.ndarray) else t.cpu().numpy() for t in self.srl_virtual_passive_pos_log], axis=0)
                            srl_virtual_load_cell_array = np.stack([t if isinstance(t, np.ndarray) else t.cpu().numpy() for t in self.srl_virtual_load_cell_log], axis=0)
                            srl_torques_array = np.stack([t if isinstance(t, np.ndarray) else t.cpu().numpy() for t in self.srl_torques_log], axis=0)
                            self.obs_log.clear()
                            self.target_yaw_log.clear()
                            self.load_cell_fd_log.clear()
                            self.load_cell_log.clear()
                            self.srl_virtual_passive_pos_log.clear()
                            self.srl_virtual_load_cell_log.clear()
                            self.srl_torques_log.clear()
                            num_dims = 51 
                            split1 = 17  # Part 1 包含 0 到 16 (共 17 维)
                            split2 = 34  # Part 2 包含 17 到 33 (共 17 维)
                            dims_per_plot = split1 # 每张图的维度数量，这里是 17

                            # --- Part 1: 维度 0 到 split1 - 1 (共 17 维) ---
                            start_dim1 = 0
                            end_dim1 = split1
                            num_rows1 = end_dim1 - start_dim1
                            fig1, axs1 = plt.subplots(num_rows1, 1, figsize=(10, num_rows1 * 1.5), sharex=True)
                            if num_rows1 == 1:
                                axs1 = [axs1]
                            for i in range(start_dim1, end_dim1):
                                axs1[i - start_dim1].plot(obs_array[:, i])
                                axs1[i - start_dim1].set_ylabel(f"D{i}")
                                axs1[i - start_dim1].grid(True)          
                            axs1[-1].set_xlabel("Step")
                            plt.suptitle(f"Episode {games_played} Observation (Part 1: D{start_dim1}-D{end_dim1-1})")
                            plt.tight_layout()
                            plt.show()

                            start_dim2 = split1
                            end_dim2 = split2
                            num_rows2 = end_dim2 - start_dim2
                            fig2, axs2 = plt.subplots(num_rows2, 1, figsize=(10, num_rows2 * 1.5), sharex=True)
                            if num_rows2 == 1:
                                axs2 = [axs2]
                            for i in range(start_dim2, end_dim2):
                                axs2[i - start_dim2].plot(obs_array[:, i])
                                axs2[i - start_dim2].set_ylabel(f"D{i}")
                                axs2[i - start_dim2].grid(True)
                            axs2[-1].set_xlabel("Step")
                            plt.suptitle(f"Episode {games_played} Observation (Part 2: D{start_dim2}-D{end_dim2-1})")
                            plt.tight_layout()
                            plt.show()

                            start_dim3 = split2
                            end_dim3 = num_dims
                            num_rows3 = end_dim3 - start_dim3
                            fig3, axs3 = plt.subplots(num_rows3, 1, figsize=(10, num_rows3 * 1.5), sharex=True)
                            if num_rows3 == 1:
                                axs3 = [axs3]
                            for i in range(start_dim3, end_dim3):
                                axs3[i - start_dim3].plot(obs_array[:, i])
                                axs3[i - start_dim3].set_ylabel(f"D{i}")
                                axs3[i - start_dim3].grid(True)
                            axs3[-1].set_xlabel("Step")
                            plt.suptitle(f"Episode {games_played} Observation (Part 3: D{start_dim3}-D{end_dim3-1})")
                            plt.tight_layout()
                            plt.show()
                            self.obs_log = []

                            # Target Tracking
                            fig3, axs3 = plt.subplots(5, 1, figsize=(10, 4 * 2.5), sharex=True)
                            axs3[0].plot(obs_array[:, 0], label='Actual Value')
                            axs3[0].plot(obs_array[:, -1], label='Target Value', linestyle='--')
                            axs3[0].set_ylabel('Pel H')
                            axs3[0].legend()
                            axs3[0].grid(True)
                            axs3[1].plot(obs_array[:, 1], label='Actual Value')
                            axs3[1].plot(obs_array[:, -3], label='Target Value', linestyle='--')
                            axs3[1].set_ylabel('Vel X')
                            axs3[1].legend()
                            axs3[1].grid(True)
                            angvel_z_smooth = lowpass_filter(obs_array[:, 6])
                            prev_yaw = obs_array[:,7+30]
                            yaw = obs_array[:,7]
                            real_dt = 0.0166 * 2
                            yaw_vel = (prev_yaw - yaw)/(real_dt)
                            axs3[2].plot(obs_array[:, 6], label='Actual Value')
                            axs3[2].plot(angvel_z_smooth, label='Smoothed Value')
                            axs3[2].plot(obs_array[:, -2], label='Target Value', linestyle='--')
                            axs3[2].set_ylabel('AngVel Z')
                            axs3[2].legend()
                            axs3[2].grid(True)
                            axs3[3].plot(target_yaw[:, 0]-obs_array[:, 7], label='Actual Value')
                            axs3[3].plot(target_yaw[:, 0], label='Target Value', linestyle='--')
                            axs3[3].set_ylabel('Ang Z')
                            axs3[3].legend()
                            axs3[3].grid(True)
                            axs3[4].plot(obs_array[:, 8], label='Actual Value')
                            axs3[4].set_ylabel('Ang Y')
                            axs3[4].legend()
                            axs3[4].grid(True)
                            plt.suptitle("Target Tracking")
                            plt.tight_layout()
                            plt.show()

                            # Load Cell Forces
                            fig4, axs4 = plt.subplots(4, 1, figsize=(10, 4 * 2.5), sharex=True)
                            axs4[0].plot(srl_virtual_passive_pos_array[:, 0], label='Pos X')
                            axs4[0].plot(srl_virtual_passive_pos_array[:, 1], label='Pos Y')
                            axs4[0].plot(srl_virtual_passive_pos_array[:, 2], label='Pos Z')                          
                            axs4[0].set_ylabel('srl virtual passive pos')
                            axs4[0].legend()
                            axs4[0].grid(True)
                            axs4[0].set_ylim([-0.020, 0.020])
                            axs4[1].plot(srl_virtual_load_cell_array[:, 0], label='Pos X')
                            axs4[1].plot(srl_virtual_load_cell_array[:, 1], label='Pos Y')
                            axs4[1].plot(srl_virtual_load_cell_array[:, 2], label='Pos Z')
                            axs4[1].set_ylabel('Virtual Load Cell Torques')
                            axs4[1].legend()
                            axs4[1].grid(True)
                            axs4[2].plot(load_cell_array[:, 0], label='Force X')
                            axs4[2].plot(load_cell_array[:, 1], label='Force Y')
                            axs4[2].plot(load_cell_array[:, 2], label='Force Z')
                            axs4[2].set_ylabel('Load Cell Forces')
                            axs4[2].legend()
                            axs4[2].grid(True)
                            axs4[2].set_ylim([-400, 400])
                            axs4[3].plot(srl_virtual_load_cell_array[:, 0], label='Torque X')
                            axs4[3].plot(srl_virtual_load_cell_array[:, 1], label='Torque Y')
                            axs4[3].plot(srl_virtual_load_cell_array[:, 2], label='Torque Z')
                            axs4[3].set_ylabel('Virtual Load Cell Torques')
                            axs4[3].legend()
                            axs4[3].grid(True)
                            axs4[3].set_ylim([-500, 500])
                            plt.suptitle("Load Cell Data")
                            plt.tight_layout()
                            plt.show()

                            # srl torques
                            # --- compute metrics ---
                            dt = getattr(self.env, "control_dt", 0.015)  # 如果 env 里有 control_dt 就用它；没有就用你默认的 0.015
                            metrics = torque_episode_metrics(
                                srl_torques_array,         # (T,6)
                                dt=dt,
                                peak_nm=320.0,
                                rated_nm=85.0,
                                rated_cont_s=240.0,
                                rms_window_s=1.0
                            )

                            J = srl_torques_array.shape[1]
                            wj = metrics["worst_joint_by_equiv"]

                            # --- pretty print to console ---
                            print("\n========== SRL Motor Torque Episode Metrics ==========")
                            print(f"episode_len = {metrics['episode_len_s']:.2f}s, dt = {metrics['dt']:.4f}s, win_steps = {metrics['win_steps']}")
                            for j in range(J):
                                print(
                                    f"[J{j}] peak={metrics['peak_nm'][j]:.1f}  "
                                    f"rms={metrics['rms_nm'][j]:.1f}  "
                                    f"roll_rms_max(1s)={metrics['rolling_rms_max_nm'][j]:.1f}  "
                                    f"t>rated={metrics['t_above_rated_s'][j]:.2f}s  "
                                    f"max_cont>rated={metrics['max_cont_above_rated_s'][j]:.2f}s  "
                                    f"t>peak={metrics['t_above_peak_s'][j]:.2f}s  "
                                    f"equiv_rated_time={metrics['equiv_rated_time_s'][j]:.1f}s  "
                                    f"margin_to_240s={metrics['thermal_margin_s'][j]:.1f}s"
                                )
                            print(f"worst_joint_by_equiv = J{wj}")
                            print("======================================================\n")

                            srl_dof_vel_array = obs_array[:, 16:22]   
                            pmet = power_episode_metrics(srl_torques_array, srl_dof_vel_array, dt=dt, window_s=1.0)

                            print("\n========== SRL Power Metrics (Mechanical) ==========")
                            print(f"win_steps={pmet['win_steps']}")
                            for j in range(srl_torques_array.shape[1]):
                                print(
                                    f"[J{j}] meanP={pmet['mean_P'][j]:.1f}W  "
                                    f"mean|P|={pmet['mean_abs_P'][j]:.1f}W  "
                                    f"rmsP={pmet['rms_P'][j]:.1f}W  "
                                    f"roll_rms_max(1s)={pmet['rolling_rms_max_P'][j]:.1f}W  "
                                    f"peak|P|={pmet['peak_abs_P'][j]:.1f}W  "
                                    f"E_abs={pmet['E_abs'][j]:.1f}J  "
                                    f"E_pos={pmet['E_pos'][j]:.1f}J  "
                                    f"E_neg={pmet['E_neg'][j]:.1f}J"
                                )
                            print("TOTAL:",
                                f"meanP={pmet['total']['mean_total_P']:.1f}W,",
                                f"mean|P|={pmet['total']['mean_total_abs_P']:.1f}W,",
                                f"rmsP={pmet['total']['rms_total_P']:.1f}W,",
                                f"peak|P|={pmet['total']['peak_total_abs_P']:.1f}W,",
                                f"E_abs={pmet['total']['E_total_abs']:.1f}J")
                            print("====================================================\n")
                            
                            # --- plot ---
                            fig5, axs5 = plt.subplots(2, 1, figsize=(10, 4 * 2.5), sharex=True)
                            axs5[0].plot(srl_torques_array[:, 0], label='Joint 0')
                            axs5[0].plot(srl_torques_array[:, 1], label='Joint 1')
                            axs5[0].plot(srl_torques_array[:, 2], label='Joint 2')
                            axs5[0].set_ylabel('srl torques')
                            axs5[0].legend()
                            axs5[0].grid(True)
                            axs5[0].set_ylim([-300, 300])

                            axs5[1].plot(srl_torques_array[:, 3], label='Joint 3')
                            axs5[1].plot(srl_torques_array[:, 4], label='Joint 4')
                            axs5[1].plot(srl_torques_array[:, 5], label='Joint 5')
                            axs5[1].set_ylabel('srl torques')
                            axs5[1].legend()
                            axs5[1].grid(True)
                            axs5[1].set_ylim([-300, 300])

                            # --- add a compact summary box on the plot ---
                            summary = (
                                f"len={metrics['episode_len_s']:.1f}s  dt={metrics['dt']:.4f}s\n"
                                f"Worst(J{wj}): peak={metrics['peak_nm'][wj]:.1f}  rms={metrics['rms_nm'][wj]:.1f}\n"
                                f"t>rated={metrics['t_above_rated_s'][wj]:.2f}s  max_cont>rated={metrics['max_cont_above_rated_s'][wj]:.2f}s\n"
                                f"equiv_rated_time={metrics['equiv_rated_time_s'][wj]:.1f}s  margin_to_240s={metrics['thermal_margin_s'][wj]:.1f}s"
                            )
                            axs5[0].text(
                                0.01, 0.98, summary,
                                transform=axs5[0].transAxes,
                                va='top', ha='left',
                                fontsize=9,
                                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none')
                            )

                            plt.suptitle("Srl Torques + Episode Metrics")
                            plt.tight_layout()  
                            plt.show()                           


                        if self._save_data:
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
                                    f'episode_action': episode_data['action'],
                                    f'episode_obs': episode_data['obs'],
                                    f'episode_dones': episode_data['done'],
                                }

                                print(f"Episode {episode_count_env0} Data saved.")
                                if episode_count_env0 == 3:
                                    sio.savemat('run_data/GA314_best_env0_episode_data.mat', data_to_save)
                                    print("已保存env0的前三个episode的数据到env0_episode_data.mat")



                        if self._save_load_cell_data:
                            if 0 in done_indices:
                                # 转换 load cell 数据为 NumPy 数组
                                load_cell_data_np = np.array(load_cell_data)

                                # 提取前三个维度 (Fx, Fy, Fz)
                                forces = load_cell_data_np[:, :3]
                                # 提取后三个维度 (Mx, My, Mz)
                                torques = load_cell_data_np[:, 3:]

                                # 生成时间轴
                                time_steps = np.arange(len(load_cell_data_np))

                                period_z_data = forces[:, 2]  # 取前 estimated_period 长度的数据

                                # 计算一个周期内的均值、方差、积分
                                mean_fz = np.mean(period_z_data)  # 均值
                                var_fz = np.var(period_z_data)    # 方差
                                integral_fz = np.trapz(period_z_data, dx=1) / len(load_cell_data_np) # 梯形积分，dx=1 代表等间距采样

                                # **打印结果**
                                print(f"Fz 交互力的单周期均值: {mean_fz:.4f}")
                                print(f"Fz 交互力的单周期方差: {var_fz:.4f}")
                                print(f"Fz 交互力的单周期积分: {integral_fz:.4f}")

                                # 绘制 Fx, Fy, Fz 曲线
                                plt.figure(figsize=(10, 5))
                                plt.plot(time_steps, forces[:, 0], label='Fx', linestyle='-',  )
                                plt.plot(time_steps, forces[:, 1], label='Fy', linestyle='-',  )
                                plt.plot(time_steps, forces[:, 2], label='Fz', linestyle='-',  )
                                plt.xlabel('Time Step')
                                plt.ylabel('Force (N)')
                                plt.title('Load Cell Forces (Fx, Fy, Fz)')
                                plt.legend()
                                plt.grid()
                                plt.show()

                                # 绘制 Mx, My, Mz 曲线
                                plt.figure(figsize=(10, 5))
                                plt.plot(time_steps, torques[:, 0], label='Mx', linestyle='-', )
                                plt.plot(time_steps, torques[:, 1], label='My', linestyle='-', )
                                plt.plot(time_steps, torques[:, 2], label='Mz', linestyle='-', )
                                plt.xlabel('Time Step')
                                plt.ylabel('Torque (Nm)')
                                plt.title('Load Cell Torques (Mx, My, Mz)')
                                plt.legend()
                                plt.grid()
                                plt.show()

                                # 绘制 外肢体末端 pos 曲线
                                srl_right_end_pos_np = np.array(episode_data['srl_end_pos'])
                                srl_right_end_pos = srl_right_end_pos_np[:,0,:]
                                plt.figure(figsize=(10, 5))
                                # plt.plot(time_steps, srl_right_end_pos[:,0], label='x', linestyle='-',  )
                                # plt.plot(time_steps, srl_right_end_pos[:,1], label='y', linestyle='-',  )
                                plt.plot(time_steps, srl_right_end_pos[:,2], label='z', linestyle='-',  )
                                plt.xlabel('Time Step')
                                plt.ylabel('Displace')
                                plt.title('SRL right end postion (x, y, z)')
                                plt.legend()
                                plt.grid()
                                plt.show()

                                # 差分计算速度，使用 np.diff 并补一个零使长度一致
                                vel_xyz = np.diff(srl_right_end_pos, axis=0)
                                vel_xyz = np.vstack([vel_xyz, np.zeros((1, 3))])  # 末尾补零保持维度一致
                                vx, vy, vz = vel_xyz[:, 0], vel_xyz[:, 1], vel_xyz[:, 2]

                                # 绘图：右末端 x/y/z 方向速度随时间变化
                                plt.figure(figsize=(10, 4))
                                plt.plot(time_steps, vx, label='Vx')
                                plt.plot(time_steps, vy, label='Vy')
                                plt.plot(time_steps, vz, label='Vz')
                                plt.xlabel('Time Step')
                                plt.ylabel('Diff Velocity (m/s)')
                                plt.title('Right SRL End Velocity Over Time (via finite difference)')
                                plt.legend()
                                plt.grid(True)
                                plt.tight_layout()
                                plt.show()

                                # 绘制 外肢体末端 vel 曲线
                                srl_right_end_vel_np = np.array(episode_data['srl_end_vel'])
                                srl_right_end_vel = srl_right_end_vel_np[:,0,:]
                                # srl_right_end_vel = lowpass_filter(srl_right_end_vel, cutoff=12, fs=60.0, order=4)
                                plt.figure(figsize=(10, 5))
                                plt.plot(time_steps, srl_right_end_vel[:,0], label='x', linestyle='-',  )
                                plt.plot(time_steps, srl_right_end_vel[:,1], label='y', linestyle='-',  )
                                plt.plot(time_steps, srl_right_end_vel[:,2], label='z', linestyle='-',  )
                                plt.xlabel('Time Step')
                                plt.ylabel('Velocity')
                                plt.title('SRL right end velocity (x, y, z)')
                                plt.legend()
                                plt.grid()
                                plt.show()

                                load_cell_data = []

                        if self.is_rnn:
                            for s in self.states:
                                s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()
                        cur_srl_rewards = cr_srl[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        cr_srl = cr_srl * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if self.print_stats:
                            if print_game_res:
                                print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)
                            else:
                                print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count)
                                print('SRL reward:', cur_srl_rewards/done_count, 'steps:', cur_steps/done_count)

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
        config['actions_num_humanoid'] = self.env.get_humanoid_action_size()
        config['actions_num_srl'] = self.env.get_srl_action_size()
        config['obs_num_humanoid'] =[self.env.get_humanoid_obs_size(),]
        config['obs_num_srl'] =  [self.env.get_srl_obs_size(),]
        config['priv_obs_num_srl'] = (self.env.get_srl_priv_obs_size(),)
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


 
class SRL_Bot_PlayerContinuous(common_player.CommonPlayer):
    def __init__(self,params):
        super().__init__(params)
        self.obs_log = []
        self.target_yaw_log = []
        self.srl_torques_log = []

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

        need_init_rnn = self.is_rnn
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

            for n in range(self.max_steps):
                obs_dict, done_env_ids = self._env_reset_done()

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    action = self.get_action(obs_dict, is_determenistic)
                obs_dict, r, done, info =  self.env_step(self.env, action)

                obs = obs_dict['obs']  # shape: [num_envs, obs_dim]
                if isinstance(obs, torch.Tensor):
                    obs_np = obs.detach().cpu().numpy()[0, :]  # 取第一个环境
                else:
                    obs_np = np.array(obs[0, :])
                self.obs_log.append(obs_np)
                self.target_yaw_log.append(info['target_yaw'].cpu().numpy())
                self.srl_torques_log.append(info['srl_torques'].cpu().numpy())

                cr += r
                steps += 1
  
                self._post_step(info)

                if render:
                    self.env.render(mode = 'human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if 0 in done_indices:
                        # 转为 numpy 数组，shape: [T, D]
                        target_yaw = []
                        obs_array = np.stack(self.obs_log, axis=0)
                        target_yaw = np.stack([t if isinstance(t, np.ndarray) else t.cpu().numpy() for t in self.target_yaw_log], axis=0)
                        srl_torques_array = np.stack([t if isinstance(t, np.ndarray) else t.cpu().numpy() for t in self.srl_torques_log], axis=0)
                        self.obs_log.clear()
                        self.target_yaw_log.clear()
                        self.srl_torques_log.clear()

                        num_dims = 30
                        mid = num_dims // 2

                        # 前半维度
                        fig1, axs1 = plt.subplots(mid, 1, figsize=(10, mid * 1.5), sharex=True)
                        if mid == 1:
                            axs1 = [axs1]

                        for i in range(mid):
                            axs1[i].plot(obs_array[:, i])
                            axs1[i].set_ylabel(f"D{i}")
                            axs1[i].grid(True)
                        axs1[-1].set_xlabel("Step")
                        plt.suptitle(f"Episode {games_played} Observation (Part 1)")
                        plt.tight_layout()
                        plt.show()

                        # 后半维度
                        fig2, axs2 = plt.subplots(num_dims - mid, 1, figsize=(10, (num_dims - mid) * 1.5), sharex=True)
                        if (num_dims - mid) == 1:
                            axs2 = [axs2]

                        for i in range(mid, num_dims):
                            axs2[i - mid].plot(obs_array[:, i])
                            axs2[i - mid].set_ylabel(f"D{i}")
                            axs2[i - mid].grid(True)
                        axs2[-1].set_xlabel("Step")
                        plt.suptitle(f"Episode {games_played} Observation (Part 2)")
                        plt.tight_layout()
                        plt.show()
                        self.obs_log = []

                        # Target Tracking
                        fig3, axs3 = plt.subplots(4, 1, figsize=(10, 4 * 2.5), sharex=True)
                        axs3[0].plot(obs_array[:, 0], label='Actual Value')
                        axs3[0].plot(obs_array[:, -1], label='Target Value', linestyle='--')
                        axs3[0].set_ylabel('Pel H')
                        axs3[0].legend()
                        axs3[0].grid(True)
                        axs3[1].plot(obs_array[:, 1], label='Actual Value')
                        axs3[1].plot(obs_array[:, -3], label='Target Value', linestyle='--')
                        axs3[1].set_ylabel('Vel X')
                        axs3[1].legend()
                        axs3[1].grid(True)
                        angvel_z_smooth = lowpass_filter(obs_array[:, 6])
                        prev_yaw = obs_array[:,7+30]
                        yaw = obs_array[:,7]
                        real_dt = 0.0166 * 2
                        yaw_vel = (prev_yaw - yaw)/(real_dt)
                        axs3[2].plot(obs_array[:, 6], label='Actual Value')
                        axs3[2].plot(angvel_z_smooth, label='Smoothed Value')
                        axs3[2].plot(obs_array[:, -2], label='Target Value', linestyle='--')
                        axs3[2].set_ylabel('AngVel Z')
                        axs3[2].legend()
                        axs3[2].grid(True)
                        axs3[3].plot(target_yaw[:, 0]-obs_array[:, 7], label='Actual Value')
                        axs3[3].plot(target_yaw[:, 0], label='Target Value', linestyle='--')
                        axs3[3].set_ylabel('AngVel Z')
                        axs3[3].legend()
                        axs3[3].grid(True)
                        plt.suptitle("Target Tracking")
                        plt.tight_layout()
                        plt.show()

                        # --- plot ---
                        # srl torques
                        # --- compute metrics ---
                        dt = getattr(self.env, "control_dt", 0.015)  # 如果 env 里有 control_dt 就用它；没有就用你默认的 0.015
                        metrics = torque_episode_metrics(
                            srl_torques_array,         # (T,6)
                            dt=dt,
                            peak_nm=160.0,
                            rated_nm=60.0,
                            rated_cont_s=240.0,
                            rms_window_s=1.0
                        )

                        J = srl_torques_array.shape[1]
                        wj = metrics["worst_joint_by_equiv"]

                        # --- pretty print to console ---
                        print("\n========== SRL Motor Torque Episode Metrics ==========")
                        print(f"episode_len = {metrics['episode_len_s']:.2f}s, dt = {metrics['dt']:.4f}s, win_steps = {metrics['win_steps']}")
                        for j in range(J):
                            print(
                                f"[J{j}] peak={metrics['peak_nm'][j]:.1f}  "
                                f"rms={metrics['rms_nm'][j]:.1f}  "
                                f"roll_rms_max(1s)={metrics['rolling_rms_max_nm'][j]:.1f}  "
                                f"t>rated={metrics['t_above_rated_s'][j]:.2f}s  "
                                f"max_cont>rated={metrics['max_cont_above_rated_s'][j]:.2f}s  "
                                f"t>peak={metrics['t_above_peak_s'][j]:.2f}s  "
                                f"equiv_rated_time={metrics['equiv_rated_time_s'][j]:.1f}s  "
                                f"margin_to_240s={metrics['thermal_margin_s'][j]:.1f}s"
                            )
                        print(f"worst_joint_by_equiv = J{wj}")
                        print("======================================================\n")
                        fig5, axs5 = plt.subplots(2, 1, figsize=(10, 4 * 2.5), sharex=True)
                        axs5[0].plot(srl_torques_array[:, 0], label='Joint 0')
                        axs5[0].plot(srl_torques_array[:, 1], label='Joint 1')
                        axs5[0].plot(srl_torques_array[:, 2], label='Joint 2')
                        axs5[0].set_ylabel('srl torques')
                        axs5[0].legend()
                        axs5[0].grid(True)
                        axs5[0].set_ylim([-300, 300])

                        axs5[1].plot(srl_torques_array[:, 3], label='Joint 3')
                        axs5[1].plot(srl_torques_array[:, 4], label='Joint 4')
                        axs5[1].plot(srl_torques_array[:, 5], label='Joint 5')
                        axs5[1].set_ylabel('srl torques')
                        axs5[1].legend()
                        axs5[1].grid(True)
                        axs5[1].set_ylim([-300, 300])
                        plt.suptitle("Srl Torques + Episode Metrics")
                        plt.tight_layout()  
                        plt.show()   

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
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)
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


def lowpass_filter(data, cutoff=2.5, fs=30.0, order=4):
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data)
    return filtered


def torque_episode_metrics(tau: np.ndarray,
                           dt: float,
                           peak_nm: float = 320.0,
                           rated_nm: float = 85.0,
                           rated_cont_s: float = 240.0,
                           rms_window_s: float = 1.0):
    """
    tau: (T, J) torque in N*m
    dt : seconds per step
    returns: dict of per-joint metrics + some episode-level helpers
    """
    tau = np.asarray(tau)
    assert tau.ndim == 2, f"tau should be (T,J), got {tau.shape}"
    T, J = tau.shape
    abs_tau = np.abs(tau)

    # basic
    peak = abs_tau.max(axis=0)                      # (J,)
    mean_abs = abs_tau.mean(axis=0)
    rms_all = np.sqrt(np.mean(tau**2, axis=0))

    # threshold times
    above_rated = abs_tau > rated_nm
    above_peak  = abs_tau > peak_nm
    t_above_rated = above_rated.sum(axis=0) * dt
    t_above_peak  = above_peak.sum(axis=0) * dt

    # max continuous time above rated
    max_cont_above_rated = np.zeros(J, dtype=np.float64)
    for j in range(J):
        run = 0
        best = 0
        for m in above_rated[:, j]:
            run = run + 1 if m else 0
            best = max(best, run)
        max_cont_above_rated[j] = best * dt

    # rolling RMS over window (e.g. 1 second)
    win = max(1, int(round(rms_window_s / max(dt, 1e-9))))
    if win <= 1:
        rolling_rms_max = rms_all.copy()
    else:
        kernel = np.ones(win, dtype=np.float64) / win
        # valid conv on tau^2, then sqrt
        rolling_rms = np.sqrt(np.vstack([
            np.convolve(tau[:, j]**2, kernel, mode="valid")
            for j in range(J)
        ]).T)  # (T-win+1, J)
        rolling_rms_max = rolling_rms.max(axis=0)

    # I^2 t style "rated-equivalent time" (heuristic thermal accumulation)
    # equiv_time = ∫ (|tau|/rated)^2 dt
    equiv_rated_time = np.sum((abs_tau / rated_nm)**2, axis=0) * dt
    margin_s = rated_cont_s - equiv_rated_time  # >0 means "below 4-min rated thermal budget" under this heuristic

    # helpers
    episode_len_s = T * dt
    worst_joint_by_equiv = int(np.argmax(equiv_rated_time))

    return {
        "dt": dt,
        "T": T,
        "episode_len_s": episode_len_s,
        "peak_nm": peak,
        "mean_abs_nm": mean_abs,
        "rms_nm": rms_all,
        "rolling_rms_max_nm": rolling_rms_max,
        "t_above_rated_s": t_above_rated,
        "max_cont_above_rated_s": max_cont_above_rated,
        "t_above_peak_s": t_above_peak,
        "equiv_rated_time_s": equiv_rated_time,
        "thermal_margin_s": margin_s,
        "worst_joint_by_equiv": worst_joint_by_equiv,
        "win_steps": win,
    }


def power_episode_metrics(tau: np.ndarray,
                          omega: np.ndarray,
                          dt: float,
                          window_s: float = 1.0):
    """
    tau:   (T,J) N*m
    omega: (T,J) rad/s
    dt: seconds per step
    """
    tau = np.asarray(tau)
    omega = np.asarray(omega)
    assert tau.shape == omega.shape and tau.ndim == 2, f"shape mismatch: {tau.shape} vs {omega.shape}"

    P = tau * omega                       # mechanical power, W
    P_abs = np.abs(P)

    mean_P = P.mean(axis=0)               # signed average power (regen can cancel)
    mean_abs_P = P_abs.mean(axis=0)       # average absolute power (better for “effort”)
    rms_P = np.sqrt(np.mean(P**2, axis=0))
    peak_abs_P = P_abs.max(axis=0)

    # positive/negative power split (useful: drive vs braking/regen)
    P_pos = np.clip(P, 0.0, None)
    P_neg = np.clip(P, None, 0.0)
    mean_P_pos = P_pos.mean(axis=0)
    mean_P_neg = P_neg.mean(axis=0)       # negative number

    # energies (J)
    E_pos = np.sum(P_pos, axis=0) * dt
    E_neg = np.sum(P_neg, axis=0) * dt    # negative
    E_abs = np.sum(P_abs, axis=0) * dt

    # rolling RMS power
    win = max(1, int(round(window_s / max(dt, 1e-9))))
    if win <= 1:
        rolling_rms_max = rms_P.copy()
    else:
        kernel = np.ones(win, dtype=np.float64) / win
        rolling_rms = np.sqrt(np.vstack([
            np.convolve(P[:, j]**2, kernel, mode="valid")
            for j in range(P.shape[1])
        ]).T)
        rolling_rms_max = rolling_rms.max(axis=0)

    # episode-level total power (sum joints)
    P_total = P.sum(axis=1)               # (T,)
    metrics_total = {
        "mean_total_P": float(P_total.mean()),
        "mean_total_abs_P": float(np.abs(P_total).mean()),
        "rms_total_P": float(np.sqrt(np.mean(P_total**2))),
        "peak_total_abs_P": float(np.max(np.abs(P_total))),
        "E_total_pos": float(np.sum(np.clip(P_total, 0.0, None)) * dt),
        "E_total_neg": float(np.sum(np.clip(P_total, None, 0.0)) * dt),
        "E_total_abs": float(np.sum(np.abs(P_total)) * dt),
    }

    return {
        "P": P,
        "mean_P": mean_P,
        "mean_abs_P": mean_abs_P,
        "rms_P": rms_P,
        "peak_abs_P": peak_abs_P,
        "mean_P_pos": mean_P_pos,
        "mean_P_neg": mean_P_neg,
        "E_pos": E_pos,
        "E_neg": E_neg,
        "E_abs": E_abs,
        "rolling_rms_max_P": rolling_rms_max,
        "win_steps": win,
        "total": metrics_total,
    }