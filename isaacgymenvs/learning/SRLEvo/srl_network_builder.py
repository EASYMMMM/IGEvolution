'''
Neural Network
创建控制策略的神经网络
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

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder
from rl_games.algos_torch.network_builder import NetworkBuilder

import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0


class HumanoidBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num_humanoid') # 获取动作数量
            input_shape = kwargs.pop('input_shape') # 获取输入形状
            self.value_size = kwargs.pop('value_size', 1) # 获取价值输出的大小，默认为1
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params) # 载入参数设置
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()
            

            # mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)
            # 直接使用输入形状初始化MLP，不经过CNN层的处理
            mlp_input_shape = input_shape[0]

            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            # 设置MLP层参数
            mlp_args = {
                'input_size' : in_mlp_shape, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)
            
            # 创建值输出层
            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            # 根据动作空间类型，创建适当的输出层
            # mu和sigma用于表示策略网络的输出，特别是用于连续动作空间中的动作均值和
            # 标准差。这些参数定义了一个高斯分布，策略网络通过从该分布中采样来生成动
            # 作。
            self.mu = torch.nn.Linear(out_size, actions_num)
            self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])  # None 无激活函数 线性输出
            mu_init = self.init_factory.create(**self.space_config['mu_init'])
            self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])  # None 无激活函数 线性输出
            sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

            if self.fixed_sigma: # True 表示 sigma 不通过网络学习
                self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
            else:
                self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)

            for m in self.modules():         
                # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                #     cnn_init(m.weight)
                #     if getattr(m, "bias", None) is not None:
                #         torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)  

            # 构建用于AMP的对抗性网络部分      
            amp_input_shape = kwargs.get('amp_input_shape')
            self._build_disc(amp_input_shape)

            return

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)

            a_out = self.actor_mlp(obs)
            c_out = self.critic_mlp(obs)
                        
            value = self.value_act(self.value(c_out))
            
            # 在前向传播过程中，mu和sigma用于计算动作的均值和标准差：
            mu = self.mu_act(self.mu(a_out))
            if self.fixed_sigma: # sigma 为常量值
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(a_out))

            return mu, sigma, value, states
    
        def load(self, params):
            super().load(params)
            # 加载对抗性网络部分的配置参数
            self._disc_units = params['disc']['units']
            self._disc_activation = params['disc']['activation']
            self._disc_initializer = params['disc']['initializer']
            return

        def eval_critic(self, obs):
            #c_out = self.critic_cnn(obs)
            #c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(obs)              
            value = self.value_act(self.value(c_out))
            return value

        def eval_disc(self, amp_obs):
            disc_mlp_out = self._disc_mlp(amp_obs)
            disc_logits = self._disc_logits(disc_mlp_out)
            return disc_logits

        def get_disc_logit_weights(self):
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            weights = []
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._disc_logits.weight))
            return weights

        def _build_disc(self, input_shape):
            self._disc_mlp = nn.Sequential()

            mlp_args = {
                'input_size' : input_shape[0], 
                'units' : self._disc_units, 
                'activation' : self._disc_activation, 
                'dense_func' : torch.nn.Linear
            }
            self._disc_mlp = self._build_mlp(**mlp_args)
            
            mlp_out_size = self._disc_units[-1]
            self._disc_logits = torch.nn.Linear(mlp_out_size, 1)

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 

            torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias) 

            return


        def _build_srl(self, input_shape):
            # srl控制器
            self._disc_mlp = nn.Sequential()

            mlp_args = {
                'input_size' : input_shape[0], 
                'units' : self._disc_units, 
                'activation' : self._disc_activation, 
                'dense_func' : torch.nn.Linear
            }
            self._disc_mlp = self._build_mlp(**mlp_args)
            
            mlp_out_size = self._disc_units[-1]
            self._disc_logits = torch.nn.Linear(mlp_out_size, 1)

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 

            torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias) 

            return       

    def build(self, name, **kwargs):
        net = HumanoidBuilder.Network(self.params, **kwargs)
        return net


class SRLBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num_srl') # 获取动作数量
            input_shape = kwargs.pop('input_shape') # 获取输入形状
            self.value_size = kwargs.pop('value_size', 1) # 获取价值输出的大小，默认为1
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params) # 载入参数设置
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()
            

            # mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)
            # 直接使用输入形状初始化MLP，不经过CNN层的处理
            mlp_input_shape = input_shape[0]

            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            # 设置MLP层参数
            mlp_args = {
                'input_size' : in_mlp_shape, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)
            
            # 创建值输出层
            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            # 根据动作空间类型，创建适当的输出层
            # 连续动作：mu+sigma
            self.mu = torch.nn.Linear(out_size, actions_num)
            self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])  # None 无激活函数 线性输出
            mu_init = self.init_factory.create(**self.space_config['mu_init'])
            self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])  # None 无激活函数 线性输出
            sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

            if self.fixed_sigma: # True 表示 sigma 不通过网络学习
                self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
            else:
                self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)

            for m in self.modules():         
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)  

            return

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)

            a_out = self.actor_mlp(obs)
            c_out = self.critic_mlp(obs)
                        
            value = self.value_act(self.value(c_out))
            
            # 在前向传播过程中，mu和sigma用于计算动作的均值和标准差：
            mu = self.mu_act(self.mu(a_out))
            if self.fixed_sigma: # sigma 为常量值
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(a_out))

            return mu, sigma, value, states

        def eval_critic(self, obs):
            #c_out = self.critic_cnn(obs)
            #c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(obs)              
            value = self.value_act(self.value(c_out))
            return value
                
        def load(self, params):
            super().load(params)
            return

    def build(self, name, **kwargs):
        if name == 'srl':
            net = SRLBuilder.Network(self.params, **kwargs)
        if name == 'humanoid':
            net = HumanoidBuilder.Network(self.params, **kwargs)
        return net