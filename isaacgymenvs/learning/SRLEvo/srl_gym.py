import os
from .srl_continuous import SRLAgent
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model_grammar')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from model_grammar import SRL_mode1,ModelGenerator
from isaacgymenvs.learning.SRLEvo.srlgym_mp import SRLGym_process
from isaacgymenvs.learning.SRLEvo.designer_opt import GeneticAlgorithmOptimizer
from datetime import datetime
from isaacgymenvs.learning.SRLEvo.mp_util import subproc_worker 
from omegaconf import OmegaConf
from copy import deepcopy
import wandb
from isaacgymenvs.utils.reformat import omegaconf_to_dict
import time
import random
import shutil
import csv

class SRLGym( ):
    def __init__(self, cfg):
        self.cfg = cfg 
        self.mjcf_folder = 'mjcf/humanoid_srl'
        self.process_cls = SRLGym_process
        self.wandb_group_name = cfg['experiment'] + 'Group' + datetime.now().strftime("_%d-%H-%M-%S")
        self.wandb_exp_name = cfg['experiment'] + datetime.now().strftime("_%d-%H-%M-%S")
        self.init_cfg()
        self.curr_frame = 0
        self.best_evaluate_reward = -100500

    def log_design_param(self,design_param,step):
        info_dict = {
                "design/leg1_lenth" : design_param["first_leg_lenth"],
                "design/leg1_size"  : design_param["first_leg_size"],
                "design/leg2_lenth": design_param["second_leg_lenth"],
                "design/leg2_size" : design_param["second_leg_size"],
                "design/end_size"  : design_param["third_leg_size"],
        }
        wandb.log(info_dict,step=step)
         

    def init_cfg(self):
        self.cfg['wandb_project'] = 'SRLGym'
        self.experiment_dir = os.path.join('runs', self.cfg.train.params.config.name + '_Evo' + 
            '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
        OmegaConf.update(self.cfg, "experiment_dir", self.experiment_dir)
        

    def design_train(self,):
        pass

    def train_test(self):
        cfg = self.cfg
        wandb_exp_name = self.wandb_exp_name
        # self.init_wandb(cfg,wandb_exp_name )
        curr_frame = 1

        model_output_path =  os.path.join(self.experiment_dir,  'nn')
        os.makedirs(model_output_path, exist_ok=True)  # 创建输出文件夹

        design_opt = {"random":self.random_SRL_designer,
                      "GA":self.GA_SRL_design_opt, }[cfg['train']['gym']['design_opt']]

        subproc_cls_runner = subproc_worker(SRLGym_process, ctx="spawn", daemon=False)
        


        # 在预先训练模型上进行第二步训练
        for i in range(2):

            cfg['train']['params']['config']['start_frame'] = curr_frame+1
            xml_name = 'hsrl_mode1'
            train_cfg = deepcopy(cfg)
            srl_params = design_opt()
            # 生成xml模型
            self.generate_SRL_xml(xml_name,'mode1',srl_params,pretrain=False)
            # 设置xml路径
            train_cfg['task']['env']['asset']['assetFileName'] = self.mjcf_folder + '/' + xml_name + '.xml'  # XML模型路径
            # 设置hsrl预训练
            train_cfg['train']['params']['config']['hsrl_checkpoint'] = 'runs/SRL_walk_v1.8.3_4090_03-17-37-52/nn/SRL_walk_v1.8.3_4090_03-17-37-58.pth'   # 预训练加载点

            # 设置模型输出路径
            model_name = 'mode1_id'+ str(i)
            model_output_file = os.path.join(model_output_path, model_name)
            train_cfg['train']['params']['config']['model_output_file'] = model_output_file  # 模型输出路径
            train_cfg['train']['params']['config']['train_dir'] =  os.path.join(self.experiment_dir, 'logs')
        
            runner = subproc_cls_runner(train_cfg)
            try:
                evaluate_reward, _, frame = runner.rlgpu(wandb_exp_name,design_params=srl_params).results
                print('frame=',frame)
            except Exception as e:
                print(f"Error during execution: {e}")
            finally:
                runner.close()
                print('close runner')
            curr_frame = curr_frame + frame

    def train_GA_test(self):
        design_opt = GeneticAlgorithmOptimizer(self.default_SRL_designer(),
                                            self.design_evaluate,
                                            population_size=20,
                                            num_iterations=10)
        best_individuals = design_opt.optimize()

        # 记录每一代的最优设计及其评估值到CSV文件
        best_params = best_individuals[-1][0]  # 获取最后一代的最优参数
        csv_file = os.path.join(self.experiment_dir, "GA_optimize_result.csv")
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Generation', 'Best_Score'] + list(best_params.keys())
            writer.writerow(header)
            
            for i, (params, score) in enumerate(best_individuals):
                row = [i + 1, score] + list(params.values())
                writer.writerow(row)

        print(f"Optimization results saved to {csv_file}")

    def generate_SRL_xml(self, name, srl_mode, srl_params, pretrain = False):
        # generate SRL mjcf xml file
        srl_generator = { "mode1": SRL_mode1 }[srl_mode]
        srl_R = srl_generator( name=name, pretrain=pretrain, **srl_params)
        abs_path =  os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../assets/'+self.mjcf_folder))  
        mjcf_generator = ModelGenerator(srl_R,save_path=abs_path)
        back_load = not pretrain
        mjcf_generator.gen_basic_humanoid_xml()
        mjcf_generator.get_SRL_dfs(back_load=back_load)
        mjcf_generator.generate()
        

    def design_evaluate(self, design_params):
        cfg = self.cfg
        xml_name = 'hsrl_mode1'
        train_cfg = deepcopy(cfg)
        train_cfg['train']['params']['config']['start_frame'] = self.curr_frame + 1
        srl_params = design_params
        # 生成xml模型
        self.generate_SRL_xml(xml_name,'mode1',srl_params,pretrain=False)
        # 设置xml路径
        train_cfg['task']['env']['asset']['assetFileName'] = self.mjcf_folder + '/' + xml_name + '.xml'  # XML模型路径
        # 设置hsrl预训练
        train_cfg['train']['params']['config']['hsrl_checkpoint'] = 'runs/SRL_walk_v1.8.3_4090_03-17-37-52/nn/SRL_walk_v1.8.3_4090_03-17-37-58.pth'   # 预训练加载点

        # 设置模型输出路径
        model_name = 'mode1_id'
        model_output_path =  os.path.join(self.experiment_dir,  'nn')
        os.makedirs(model_output_path, exist_ok=True)  # 创建输出文件夹
        model_output_file = os.path.join(model_output_path, model_name)
        train_cfg['train']['params']['config']['model_output_file'] = model_output_file  # 模型输出路径
        train_cfg['train']['params']['config']['train_dir'] =  os.path.join(self.experiment_dir, 'logs')
        subproc_cls_runner = subproc_worker(SRLGym_process, ctx="spawn", daemon=False)
        runner = subproc_cls_runner(train_cfg)
        try:
            evaluate_reward, _, frame = runner.rlgpu(self.wandb_exp_name,design_params=srl_params).results
            print('frame=',frame)
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            runner.close()
            print('close runner')
        self.curr_frame = self.curr_frame + frame
        if evaluate_reward > self.best_evaluate_reward:
            self.best_evaluate_reward = evaluate_reward  # 更新最佳评估分数
            best_model_path = os.path.join(model_output_path, 'best_model.pth')
            shutil.copy(model_output_file+'.pth', best_model_path)  # 复制当前最优模型为 best_model.pth
            print(f"Best model saved with reward {evaluate_reward} at {best_model_path}")
        return evaluate_reward

    
    def random_SRL_designer(self):
        # 基础尺寸
        base_params = {
            "first_leg_lenth" : 0.40,
            "first_leg_size"  : 0.03,
            "second_leg_lenth": 0.80,
            "second_leg_size" : 0.03,
            "third_leg_size"  : 0.03,
        }
        
        # 生成随机浮动的尺寸，并将结果保留到四位小数
        srl_params = {
            key: round(random.uniform(value * 0.7, value * 1.3), 4)  # 在上下浮动30%范围内生成随机值并保留四位小数
            for key, value in base_params.items()
        }
        
        return srl_params


    def default_SRL_designer(self,):
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