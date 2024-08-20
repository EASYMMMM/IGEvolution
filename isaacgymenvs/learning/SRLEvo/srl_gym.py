from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv
from rl_games.common.experience import ExperienceBuffer

# from isaacgymenvs.utils.torch_jit_utils import to_torch


import time
from datetime import datetime
import numpy as np
from torch import optim
import torch 
from torch import nn
import os
from rl_games.algos_torch import central_value
# import isaacgymenvs.learning.replay_buffer as replay_buffer
# import isaacgymenvs.learning.common_agent as common_agent 
# from .. import amp_datasets as amp_datasets
from tensorboardX import SummaryWriter
from rl_games.common import datasets
from gym.spaces.box import Box
from srl_continuous import SRLAgent
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model_grammar')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from model_grammar import SRL_mode1,ModelGenerator
 

class SRLGym(SRLAgent):
    def __init__(self, base_name, params):
        super.__init__(self, base_name, params)

    def train(self):
        
        self.gen_SRL_mjcf('hsrl_test_pretrain','mode1',self.SRL_designer(), pretrain=True)
        


    def gen_SRL_mjcf(self, name, srl_mode, srl_params, pretrain = False):
        srl_generator = { "mode1": SRL_mode1 }[srl_mode]
        srl_R = srl_generator( name=name, pretrain=pretrain, **srl_params)
        mjcf_generator = ModelGenerator(srl_R,save_path='..../assets/mjcf/humanoid_srl')
        back_load = not pretrain
        mjcf_generator.gen_basic_humanoid_xml()
        mjcf_generator.get_SRL_dfs(back_load=back_load)
        mjcf_generator.generate()
        
    
    def SRL_designer(self,):
        # 外肢体形态参数生成函数

        srl_params = {
                    "first_leg_lenth" : 0.40,
                    "first_leg_size"  : 0.03,
                    "second_leg_lenth": 0.80,
                    "second_leg_size" : 0.03,
                    "third_leg_size"  : 0.03,
                }
        return srl_params


if __name__ == '__main__':
    srl_mode = 'mode1'
    name = 'humanoid_srl_mode1'
    pretrain = False
    srl_params = {
                    "first_leg_lenth" : 0.40,
                    "first_leg_size"  : 0.03,
                    "second_leg_lenth": 0.80,
                    "second_leg_size" : 0.03,
                    "third_leg_size"  : 0.03,
                }    
    srl_generator = { "mode1": SRL_mode1 }[srl_mode]
    srl_R = srl_generator( name=name, pretrain=pretrain, **srl_params)

    # 使用绝对路径来确定 save_path
    base_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_path, '../../../assets/mjcf/humanoid_srl/')

    mjcf_generator = ModelGenerator(srl_R,save_path=save_path)
    back_load = not pretrain
    mjcf_generator.gen_basic_humanoid_xml()
    mjcf_generator.get_SRL_dfs(back_load=back_load)
    mjcf_generator.generate()