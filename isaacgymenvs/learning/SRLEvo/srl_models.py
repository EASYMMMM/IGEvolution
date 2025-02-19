'''
Neural Network Model
Humanoid-SRL控制策略网络的包装函数
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

import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from rl_games.common.extensions.distributions import CategoricalMasked
from torch.distributions import Categorical
import torch.nn as nn
import torch
import torch.nn.functional as F

class ModelSRLContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config, role:str=None):
        if role not in ['humanoid', 'srl']:
            raise ValueError("Invalid role. Must be 'humanoid' or 'srl'")
         # 分别为humanoid和srl构建网络
        if role == 'humanoid':
            net = self.network_builder.build('humanoid',  **config) # 使用网络构建器构建网络
            # net = srl_network_builder.HumanoidBuilder(**config)
            print('====== Humanoid Netwrok ======')
            for name, _ in net.named_parameters():
                print(name)
            print('==============================')
        elif role == 'srl':
            net = self.network_builder.build('srl',  **config) # 使用网络构建器构建网络     
            # net = srl_network_builder.SRLBuilder(**config)
            print('======== SRL Netwrok =========')
            for name, _ in net.named_parameters():
                print(name)
            print('==============================')
        obs_shape = config['input_shape']
        normalize_value = config.get('normalize_value', False) # True
        normalize_input = config.get('normalize_input', False) # True
        value_size = config.get('value_size', 1)

        return self.Network(net, role=role,obs_shape=obs_shape,
            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size)


    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, role:str, **kwargs):
            super().__init__(a2c_network, **kwargs)
            self.role = role
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            # result = super().forward(input_dict)
            
            prev_actions = input_dict.get('prev_actions', None) # 获取之前的动作
            input_dict['obs'] = self.norm_obs(input_dict['obs']) # 标准化观测输入
            
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd).clamp(min=1e-6) # 计算sigma
            distr = torch.distributions.Normal(mu, sigma, validate_args=False) # 创建正态分布
                  
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma,
                }                               
            else: # eval
                selected_action = distr.sample()
                # 计算选定动作的负对数概率
                neglogp = self.neglogp(selected_action, mu, sigma, logstd) 
                result = {
                    'actions' : selected_action,
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value), 
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma,
                }

            # AMP
            if (is_train) and self.role=='humanoid':
                amp_obs = input_dict['amp_obs'] # 获取AMP观测输入
                disc_agent_logit = self.a2c_network.eval_disc(amp_obs) # 评估AMP对抗性网络
                result["disc_agent_logit"] = disc_agent_logit

                amp_obs_replay = input_dict['amp_obs_replay'] # 获取AMP重放观测输入
                disc_agent_replay_logit = self.a2c_network.eval_disc(amp_obs_replay) # 评估AMP对抗性网络
                result["disc_agent_replay_logit"] = disc_agent_replay_logit

                amp_demo_obs = input_dict['amp_obs_demo']  # 获取AMP示范观测输入
                disc_demo_logit = self.a2c_network.eval_disc(amp_demo_obs) # 评估AMP对抗性网络
                result["disc_demo_logit"] = disc_demo_logit

            return result