import os
from .srl_continuous import SRLAgent
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model_grammar')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from model_grammar import SRL_mode1,ModelGenerator
from isaacgymenvs.learning.SRLEvo.srlgym_mp import SRLGym_process
from isaacgymenvs.learning.SRLEvo.designer_opt import GeneticAlgorithmOptimizer,BayesianOptimizer,GeneticAlgorithmOptimizer_v2,RandomOptimizer
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
import glob

def sync_tensorboard_logs(main_log_dir):
    """手动同步子进程生成的 TensorBoard 日志到 WandB"""
    # 查找所有的 tfevent 文件
    log_files = glob.glob(os.path.join(main_log_dir, '**','summaries', 'events.out.tfevents.*'), recursive=True)
    for log_file in log_files:
        print(f"Syncing file: {log_file}")
        # 将文件保存到 WandB
        wandb.save(log_file)

class SRLGym( ):
    def __init__(self, cfg):
        self.cfg = cfg 
        self.mjcf_folder = 'mjcf/humanoid_srl'
        self.hsrl_checkpoint = cfg['train']['params']['config']['hsrl_checkpoint']  # human-srl model
        self.process_cls = SRLGym_process
        self.wandb_group_name = cfg['experiment'] + 'Group' + datetime.now().strftime("_%d-%H-%M-%S")
        self.wandb_exp_name = cfg['experiment'] + datetime.now().strftime("_%d-%H-%M-%S")
        self.init_cfg()
        self.curr_frame = 0
        self.best_evaluate_reward = -100500
        self.best_design_param = {}

    def init_cfg(self):
        self.cfg['wandb_project'] = 'SRLGym'
        self.experiment_dir = os.path.join('runs', self.cfg.train.params.config.name + '_Evo' + 
            '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
        OmegaConf.update(self.cfg, "experiment_dir", self.experiment_dir)        
    
    def init_wandb(self,cfg, wandb_experiment_name):
        
        wandb_unique_id = f"uid_{wandb_experiment_name}"
        print(f"Wandb using unique id {wandb_unique_id}")
        # this can fail occasionally, so we try a couple more times
        def my_init_wandb():
            wandb.init(
                project=cfg.wandb_project,
                #entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                tags=cfg.wandb_tags,
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=wandb_experiment_name,
                resume=True,
                settings=wandb.Settings(start_method='spawn'),
            )

        print('Initializing WandB...')
        try:
            my_init_wandb()
        except Exception as exc:
            print(f'Could not initialize WandB! {exc}')

        if isinstance(cfg, dict):
            wandb.config.update(cfg, allow_val_change=True)
        else:
            wandb.config.update(omegaconf_to_dict(cfg), allow_val_change=True)

    def train(self):        
        if self.cfg["train"]["gym"]["design_opt"] == 'GA':
            self.train_GA()
        elif self.cfg["train"]["gym"]["design_opt"]=='BO':
            self.train_BO()
        elif self.cfg["train"]["gym"]["design_opt"]=='GA_v2':
            self.train_GA_v2()
        elif self.cfg["train"]["gym"]["design_opt"]=='random':
            self.train_random()


    def train_wandb_test(self):
        cfg = self.cfg
        wandb_exp_name = self.wandb_exp_name
        self.init_wandb(cfg,wandb_exp_name )
        curr_frame = 1
        iteration = 1
        cfg['wandb_activate'] = False
        model_output_path =  os.path.join(self.experiment_dir,  'nn')
        logs_output_path = os.path.join(self.experiment_dir, 'logs')
        os.makedirs(model_output_path, exist_ok=True)  # 创建输出文件夹

        design_opt = {"random":self.random_SRL_designer,
                      }['random']

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
            train_cfg['train']['params']['config']['train_dir'] =  logs_output_path
        
            runner = subproc_cls_runner(train_cfg)
            try:
                evaluate_reward, _, frame, summary_dir = runner.rlgpu(wandb_exp_name,design_params=srl_params).results
                print('frame=',frame)
            except Exception as e:
                print(f"Error during execution: {e}")
            finally:
                runner.close()
                print('close runner')
            curr_frame = curr_frame + frame
            self._log_design_param(srl_params,iteration)
            wandb.log({'Evolution/reward':evaluate_reward, 'iteration': iteration} )
            iteration = iteration+1
        sync_tensorboard_logs(logs_output_path)
        wandb.finish()

    def train_test(self):
        cfg = self.cfg
        wandb_exp_name = self.wandb_exp_name
        # self.init_wandb(cfg,wandb_exp_name )
        curr_frame = 1
        iteration = 1
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
                evaluate_reward, _, frame, summary_dir = runner.rlgpu(wandb_exp_name,design_params=srl_params).results
                print('frame=',frame)
            except Exception as e:
                print(f"Error during execution: {e}")
            finally:
                runner.close()
                print('close runner')
            curr_frame = curr_frame + frame
            self._log_design_param(srl_params,iteration)
            wandb.log({'Evolution/reward':evaluate_reward, 'iteration': iteration} )
            iteration = iteration+1

    def train_GA(self):
        cfg = self.cfg
        wandb_exp_name = self.wandb_exp_name
        self.init_wandb(cfg,wandb_exp_name )
        self.iteration = 1
        curr_frame = 1
        kwargs = {"population_size":self.cfg['train']['gym']['GA_population_size'],
                  "num_iterations":self.cfg['train']['gym']['GA_num_iterations'],
                  "mutation_rate":self.cfg['train']['gym']['GA_mutation_rate'],
                  "crossover_rate":self.cfg['train']['gym']['GA_crossover_rate'],
                  "bounds_scale":self.cfg['train']['gym']['GA_bounds_scale']}
        if self.cfg['task']['env']['design_param_obs']:
            design_opt = GeneticAlgorithmOptimizer(self.default_SRL_designer(),
                                                self.design_evaluate_with_general_model,
                                                **kwargs)
        else:
            design_opt = GeneticAlgorithmOptimizer(self.default_SRL_designer(),
                                                self.design_evaluate,
                                                **kwargs)
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
        
        logs_output_path =  os.path.join(self.experiment_dir, 'logs')
        self.final_design_train(max_epoch=600)
        sync_tensorboard_logs(logs_output_path)
        print(f"Optimization results saved to {csv_file}")

    def train_GA_v2(self):
        cfg = self.cfg
        wandb_exp_name = self.wandb_exp_name
        self.init_wandb( cfg , wandb_exp_name)
        self.iteration = 1
        curr_frame = 1
        kwargs = {"population_size":self.cfg['train']['gym']['GA_population_size'],
                  "num_iterations":self.cfg['train']['gym']['GA_num_iterations'],
                  "mutation_rate":self.cfg['train']['gym']['GA_mutation_rate'],
                  "crossover_rate":self.cfg['train']['gym']['GA_crossover_rate'],
                  "bounds_scale":self.cfg['train']['gym']['GA_bounds_scale']}
        if self.cfg['task']['env']['design_param_obs']:
            design_opt = GeneticAlgorithmOptimizer_v2(self.default_SRL_designer(),
                                                self.design_evaluate_with_general_model,
                                                **kwargs)
        else:
            design_opt = GeneticAlgorithmOptimizer_v2(self.default_SRL_designer(),
                                                self.design_evaluate,
                                                **kwargs)
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
        
        logs_output_path =  os.path.join(self.experiment_dir, 'logs')
        self.final_design_train(max_epoch=600)
        sync_tensorboard_logs(logs_output_path)
        print(f"Optimization results saved to {csv_file}")

    def train_random(self):
        cfg = self.cfg
        wandb_exp_name = self.wandb_exp_name
        self.init_wandb( cfg , wandb_exp_name)
        self.iteration = 1
        kwargs = {"num_iterations":self.cfg['train']['gym']['RA_num_iterations'],}
        if self.cfg['task']['env']['design_param_obs']:
            design_opt = RandomOptimizer(self.default_SRL_designer(),
                                                self.design_evaluate_with_general_model,
                                                **kwargs)
        else:
            design_opt = RandomOptimizer(self.default_SRL_designer(),
                                                self.design_evaluate,
                                                **kwargs)
        best_individuals = design_opt.optimize()

        # 记录每一代的最优设计及其评估值到CSV文件
        best_params = best_individuals[-1][0]  # 获取最后一代的最优参数
        csv_file = os.path.join(self.experiment_dir, "R_optimize_result.csv")
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Generation', 'Best_Score'] + list(best_params.keys())
            writer.writerow(header)
            
            for i, (params, score) in enumerate(best_individuals):
                row = [i + 1, score] + list(params.values())
                writer.writerow(row)
        
        logs_output_path =  os.path.join(self.experiment_dir, 'logs')
        self.final_design_train(max_epoch=600)
        sync_tensorboard_logs(logs_output_path)
        print(f"Optimization results saved to {csv_file}")


    def train_BO(self):
        cfg = self.cfg
        wandb_exp_name = self.wandb_exp_name
        self.init_wandb(cfg,wandb_exp_name )
        self.iteration = 1
        curr_frame = 1
        base_param = self.default_SRL_designer()
        kwargs = {"n_initial_points":self.cfg['train']['gym']['BO_n_initial_points'],
                  "num_iterations":self.cfg['train']['gym']['BO_num_iterations'], }
        if self.cfg['task']['env']['design_param_obs']:
            design_opt = BayesianOptimizer( base_param,
                                            self.design_evaluate_with_general_model,
                                            **kwargs)
        else:
            design_opt = BayesianOptimizer( base_param,
                                            self.design_evaluate,
                                            **kwargs)
        

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
        
        logs_output_path =  os.path.join(self.experiment_dir, 'logs')
        self.final_design_train(max_epoch=600)
        sync_tensorboard_logs(logs_output_path)
        print(f"Optimization results saved to {csv_file}")

    def generate_SRL_xml(self, name, srl_mode, srl_params, pretrain = False, save_path = False):
        # generate SRL mjcf xml file
        srl_generator = { "mode1": SRL_mode1 }[srl_mode]
        srl_R = srl_generator( name=name, pretrain=pretrain, **srl_params)
        if save_path:
            abs_path = save_path
        else:
            abs_path =  os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../assets/'+self.mjcf_folder))  
        mjcf_generator = ModelGenerator(srl_R,save_path=abs_path)
        back_load = not pretrain
        mjcf_generator.gen_basic_humanoid_xml()
        mjcf_generator.get_SRL_dfs(back_load=back_load)
        mjcf_generator.generate()
        
    def _log_design_param(self,design_param,step):
        info_dict = {
                "design/leg1_lenth" : design_param["first_leg_lenth"],
                "design/leg1_size"  : design_param["first_leg_size"],
                "design/leg2_lenth": design_param["second_leg_lenth"],
                "design/leg2_size" : design_param["second_leg_size"],
                "design/end_size"  : design_param["third_leg_size"],
                "iteration" :  step
        }
        wandb.log(info_dict )

    def design_evaluate(self, design_params, max_epoch=False):
        cfg = self.cfg
        xml_name = 'hsrl_mode1'
        train_cfg = deepcopy(cfg)
        train_cfg['wandb_activate'] = False
        if max_epoch:
            train_cfg['max_iterations'] = max_epoch

        train_cfg['train']['params']['config']['start_frame'] =  1
        srl_params = design_params
        # 生成xml模型
        self.generate_SRL_xml(xml_name,'mode1',srl_params,pretrain=False)
        # 设置xml路径
        train_cfg['task']['env']['asset']['assetFileName'] = self.mjcf_folder + '/' + xml_name + '.xml'  # XML模型路径
        # 设置hsrl预训练
        train_cfg['train']['params']['config']['hsrl_checkpoint'] = 'runs/SRL_walk_v1.8.3_4090_03-17-37-52/nn/SRL_walk_v1.8.3_4090_03-17-37-58.pth'   # 预训练加载点
        train_cfg['train']['params']['config']['hsrl_checkpoint'] = False   # 预训练加载点
        if train_cfg['task']['env']['design_param_obs']:
            train_cfg['task']['env']['design_params']['first_leg_lenth']  = float(design_params['first_leg_lenth'])
            train_cfg['task']['env']['design_params']['first_leg_size']   = float(design_params['first_leg_size'])
            train_cfg['task']['env']['design_params']['second_leg_lenth'] = float(design_params['second_leg_lenth'])
            train_cfg['task']['env']['design_params']['second_leg_size']  = float(design_params['second_leg_size'])
            train_cfg['task']['env']['design_params']['third_leg_size']   = float(design_params['third_leg_size'])
        # 设置模型输出路径
        model_name = 'mode1_id'
        model_output_path =  os.path.join(self.experiment_dir,  'nn')
        os.makedirs(model_output_path, exist_ok=True)  # 创建输出文件夹
        model_output_file = os.path.join(model_output_path, model_name)
        train_cfg['train']['params']['config']['model_output_file'] = model_output_file  # 模型输出路径
        train_cfg['train']['params']['config']['train_dir'] =  os.path.join(self.experiment_dir, 'logs')
        subproc_cls_runner = subproc_worker(SRLGym_process, ctx="spawn", daemon=True)
        runner = subproc_cls_runner(train_cfg)
        try:
            evaluate_reward, _, frame, summary_dir = runner.rlgpu(self.wandb_exp_name,design_params=srl_params).results
            print('frame=',frame)
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            runner.close()
            print('close runner')
        self.curr_frame = self.curr_frame + frame

        self._log_design_param(srl_params, self.iteration)
        wandb.log({'Evolution/reward':evaluate_reward, 'iteration': self.iteration} )
        self.iteration = self.iteration+1

        # save best model
        if evaluate_reward > self.best_evaluate_reward:
            self.best_evaluate_reward = evaluate_reward  # 更新最佳评估分数
            self.best_design_param = design_params 
            best_model_path = os.path.join(model_output_path, 'best_model.pth')
            shutil.copy(model_output_file+'.pth', best_model_path)  # 复制当前最优模型为 best_model.pth
            print(f"Best model saved with reward {evaluate_reward} at {best_model_path}")
        return evaluate_reward
    
    def design_evaluate_with_general_model(self, design_params, max_epoch=False):
        # put design param into observation, build a general model
        cfg = self.cfg
        xml_name = 'hsrl_mode1'
        train_cfg = deepcopy(cfg)
        train_cfg['wandb_activate'] = False
        if max_epoch:
            train_cfg['max_iterations'] = max_epoch

        train_cfg['train']['params']['config']['start_frame'] =  1
        srl_params = design_params
        # 生成xml模型
        self.generate_SRL_xml(xml_name,'mode1',srl_params,pretrain=False)
        # 设置xml路径
        train_cfg['task']['env']['asset']['assetFileName'] = self.mjcf_folder + '/' + xml_name + '.xml'  # XML模型路径
        # 设置hsrl预训练
        train_cfg['train']['params']['config']['hsrl_checkpoint'] = self.hsrl_checkpoint   # only first train use the pretrain model. after that, use general model
        
        if train_cfg['task']['env']['design_param_obs']:
            train_cfg['task']['env']['design_params']['first_leg_lenth']  = float(design_params['first_leg_lenth'])
            train_cfg['task']['env']['design_params']['first_leg_size']   = float(design_params['first_leg_size'])
            train_cfg['task']['env']['design_params']['second_leg_lenth'] = float(design_params['second_leg_lenth'])
            train_cfg['task']['env']['design_params']['second_leg_size']  = float(design_params['second_leg_size'])
            train_cfg['task']['env']['design_params']['third_leg_size']   = float(design_params['third_leg_size'])
        
        # 设置模型输出路径
        model_name = 'train_model'
        model_output_path =  os.path.join(self.experiment_dir,  'nn')
        os.makedirs(model_output_path, exist_ok=True)  # 创建输出文件夹
        model_output_file = os.path.join(model_output_path, model_name)
        train_cfg['train']['params']['config']['model_output_file'] = model_output_file  # 模型输出路径
        train_cfg['train']['params']['config']['train_dir'] =  os.path.join(self.experiment_dir, 'logs')
        subproc_cls_runner = subproc_worker(SRLGym_process, ctx="spawn", daemon=True)
        runner = subproc_cls_runner(train_cfg)
        try:
            evaluate_reward, _, frame, summary_dir = runner.rlgpu(self.wandb_exp_name,design_params=srl_params).results
            print('frame=',frame)
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            runner.close()
            print('close runner')
        self.curr_frame = self.curr_frame + frame

        # save the model if it is better enough
        if evaluate_reward > -100: 
            general_model_name = 'general_model'
            general_model_output_file = os.path.join(model_output_path, general_model_name)
            shutil.copy(model_output_file+'.pth', general_model_output_file+'.pth')  # 复制当前最优模型为 best_model.pth
            print(f"Saved model with reward {evaluate_reward} to General Model.")
            self.hsrl_checkpoint = general_model_output_file + '.pth'

        self._log_design_param(srl_params, self.iteration)
        wandb.log({'Evolution/reward':evaluate_reward, 'iteration': self.iteration} )
        self.iteration = self.iteration+1

        # save best model
        if evaluate_reward > self.best_evaluate_reward:
            self.best_evaluate_reward = evaluate_reward  # 更新最佳评估分数
            self.best_design_param = design_params 
            best_model_path = os.path.join(model_output_path, 'best_model.pth')
            shutil.copy(model_output_file+'.pth', best_model_path)  # 复制当前最优模型为 best_model.pth
            print(f"Best model saved with reward {evaluate_reward} at {best_model_path}")
        return evaluate_reward

    def final_design_train(self,max_epoch):
        # at the end, train the best design

        cfg = self.cfg
        train_cfg = deepcopy(cfg)
        train_cfg['wandb_activate'] = False
        if max_epoch:
            train_cfg['max_iterations'] = max_epoch

        train_cfg['train']['params']['config']['start_frame'] =  1
        srl_params = self.best_design_param
        # 生成xml模型
        xml_name = 'hsrl_best_design'
        self.generate_SRL_xml(xml_name,'mode1',srl_params,pretrain=False)
        xml_save_path =  os.path.join(self.experiment_dir,  'mjcf')
        self.generate_SRL_xml(xml_name,'mode1',srl_params,pretrain=False,save_path=xml_save_path)
        # 设置xml路径
        train_cfg['task']['env']['asset']['assetFileName'] = self.mjcf_folder + '/' + xml_name + '.xml'  # XML模型路径
        # 设置hsrl预训练
        best_model_path =  os.path.join(self.experiment_dir,  'nn','best_model.pth')
        train_cfg['train']['params']['config']['hsrl_checkpoint'] = best_model_path   # only first train use the pretrain model. after that, use general model
        design_params = self.best_design_param
        if train_cfg['task']['env']['design_param_obs']:
            train_cfg['task']['env']['design_params']['first_leg_lenth']  = float(design_params['first_leg_lenth'])
            train_cfg['task']['env']['design_params']['first_leg_size']   = float(design_params['first_leg_size'])
            train_cfg['task']['env']['design_params']['second_leg_lenth'] = float(design_params['second_leg_lenth'])
            train_cfg['task']['env']['design_params']['second_leg_size']  = float(design_params['second_leg_size'])
            train_cfg['task']['env']['design_params']['third_leg_size']   = float(design_params['third_leg_size'])
        
        # 设置模型输出路径
        model_name = 'final_best_model'
        model_output_path =  os.path.join(self.experiment_dir,  'nn')
        os.makedirs(model_output_path, exist_ok=True)  # 创建输出文件夹
        model_output_file = os.path.join(model_output_path, model_name)
        train_cfg['train']['params']['config']['model_output_file'] = model_output_file  # 模型输出路径
        train_cfg['train']['params']['config']['train_dir'] =  os.path.join(self.experiment_dir, 'logs')
        subproc_cls_runner = subproc_worker(SRLGym_process, ctx="spawn", daemon=True)
        runner = subproc_cls_runner(train_cfg)
        try:
            evaluate_reward, _, frame, summary_dir = runner.rlgpu(self.wandb_exp_name,design_params=srl_params).results
            print('frame=',frame)
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            runner.close()
            print('close runner')

        self._log_design_param(srl_params, self.iteration)
        wandb.log({'Evolution/reward':evaluate_reward, 'iteration': self.iteration} )
        self.iteration = self.iteration+1

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