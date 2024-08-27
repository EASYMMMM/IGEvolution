
import os
from .srl_continuous import SRLAgent
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model_grammar')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from model_grammar import SRL_mode1,ModelGenerator
from isaacgymenvs.learning.SRLEvo.srlgym_mp import SRLGym_process

class SRLGym( ):
    def __init__(self, cfg):
        self.cfg = cfg 
        self.mjcf_folder = 'mjcf/humanoid_srl'
        self.process_cls = SRLGym_process
        

    def train(self):
        self.create_sim_model('hsrl_test_pretrain','mode1',self.SRL_designer(), pretrain=True)
        super().train()

    def generate_SRL_mjcf(self, name, srl_mode, srl_params, pretrain = False):
        # generate SRL mjcf 'xml' file
        srl_generator = { "mode1": SRL_mode1 }[srl_mode]
        srl_R = srl_generator( name=name, pretrain=pretrain, **srl_params)
        abs_path =  os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../assets/'+self.mjcf_folder))  
        mjcf_generator = ModelGenerator(srl_R,save_path=abs_path)
        back_load = not pretrain
        mjcf_generator.gen_basic_humanoid_xml()
        mjcf_generator.get_SRL_dfs(back_load=back_load)
        mjcf_generator.generate()
        
    def create_sim_model(self, name, srl_mode, srl_params, pretrain=False):
        self.generate_SRL_mjcf(name, srl_mode, srl_params, pretrain)
        self.vec_env.env.cfg["env"]["asset"]["assetFileName"] = self.mjcf_folder+'/'+ name + '.xml'
         
        self.vec_env.env.restart_sim()
    

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