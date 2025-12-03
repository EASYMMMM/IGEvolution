# ================================================================
# 2025 ICRA SRL-Gym
# 外肢体形态-控制 联合优化框架（结构化重构版）
# ================================================================

import os
import sys
import time
import math
import glob
import csv
import random
import shutil
import wandb
from copy import deepcopy
from datetime import datetime
from omegaconf import OmegaConf
# 导入基于 Jinja2 的 MJCF 生成方法
from isaacgymenvs.srl_mjcf_generator.generator.generate_hsrl import generate_hsrl_model
from .srl_continuous import SRLAgent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model_grammar')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from model_grammar import SRL_mode1, ModelGenerator

from isaacgymenvs.learning.SRLEvo.srl_gym_mp import SRLGym_process
from isaacgymenvs.learning.SRLEvo.designer_opt import (
    GeneticAlgorithmOptimizer, GeneticAlgorithmOptimizer_v2,
    BayesianOptimizer, RandomOptimizer
)
from isaacgymenvs.learning.SRLEvo.mp_util import subproc_worker
from isaacgymenvs.utils.reformat import omegaconf_to_dict


# ================================================================
# Helper Functions
# ================================================================

def sync_tensorboard_logs(main_log_dir):
    log_files = glob.glob(os.path.join(main_log_dir, '**', 'summaries', 'events.out.tfevents.*'), recursive=True)
    for log_file in log_files:
        print(f"Syncing file: {log_file}")
        wandb.save(log_file)


# ================================================================
# SRLGym Main Class
# ================================================================

class SRLGym():
    def __init__(self, cfg):
        self.cfg = cfg
        self.mjcf_folder = 'mjcf/hsrl_auto_gen'
        self.hsrl_checkpoint = cfg['train']['params']['config']['hsrl_checkpoint']
        self.process_cls = SRLGym_process
        self.wandb_group_name = cfg['experiment'] + 'Group' + datetime.now().strftime("_%d-%H-%M-%S")
        self.wandb_exp_name = cfg['experiment'] + datetime.now().strftime("_%d-%H-%M-%S")

        self.init_cfg()
        self.curr_frame = 0
        self.best_evaluate_reward = -100500
        self.best_design_param = {}

        # Subprocess Runner
        self.SubprocRunner = subproc_worker(SRLGym_process, ctx="spawn", daemon=True)

    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------

    def init_cfg(self):
        self.cfg['wandb_project'] = 'SRLGym'
        self.experiment_dir = os.path.join(
            'runs',
            self.cfg.train.params.config.name + '_Evo' +
            '_{date:%d-%H-%M-%S}'.format(date=datetime.now())
        )
        OmegaConf.update(self.cfg, "experiment_dir", self.experiment_dir)

    def init_wandb(self, cfg, wandb_experiment_name):
        wandb_unique_id = f"uid_{wandb_experiment_name}"
        print(f"[WandB] Using unique id {wandb_unique_id}")

        def my_init():
            wandb.init(
                project=cfg.wandb_project,
                group=cfg.wandb_group,
                tags=cfg.wandb_tags,
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=wandb_experiment_name,
                resume=True,
                settings=wandb.Settings(start_method='spawn'),
            )

        try:
            my_init()
        except Exception as exc:
            print(f"[WandB] init failed: {exc}")

        if isinstance(cfg, dict):
            wandb.config.update(cfg, allow_val_change=True)
        else:
            wandb.config.update(omegaconf_to_dict(cfg), allow_val_change=True)

    # ================================================================
    # Unified Optimize Pipeline
    # ================================================================

    def train(self):
        opt_type = self.cfg["train"]["gym"]["design_opt"]
        self.train_design_opt(opt_type)

    def train_design_opt(self, opt_type):
        """统一优化流程：构建优化器 → 运行 → 保存历史 → 最终训练最优设计"""
        cfg = self.cfg
        wandb_exp_name = self.wandb_exp_name
        self.init_wandb(cfg, wandb_exp_name)
        self.iteration = 1

        # 1. 构建优化器
        optimizer = self.build_optimizer(opt_type)

        # 2. 执行外层优化
        best_individuals = optimizer.optimize()

        # 3. 保存优化历史 (CSV)
        self.save_opt_history(best_individuals, opt_type)

        # 4. Final training
        logs_output_path = os.path.join(self.experiment_dir, 'logs')
        self.final_design_train(self.get_final_train_epoch(opt_type))
        sync_tensorboard_logs(logs_output_path)

        print(f"[{opt_type}] Optimization Finished.")

    # --------------------------- build optimizer ----------------------

    def build_optimizer(self, opt_type):
        gym_cfg = self.cfg["train"]["gym"]
        base_params = self.default_SRL_design_parameters()
        param_space = self.SRL_param_space()

        # choose evaluation fn
        if self.cfg['task']['env']['design_param_obs']:
            evaluate_fn = self.design_evaluate_with_general_model
        else:
            evaluate_fn = self.design_evaluate

        # GA
        if opt_type == "GA":
            return GeneticAlgorithmOptimizer(
                base_params, param_space, evaluate_fn,
                population_size=gym_cfg['GA_population_size'],
                num_iterations=gym_cfg['GA_num_iterations'],
                mutation_rate=gym_cfg['GA_mutation_rate'],
                crossover_rate=gym_cfg['GA_crossover_rate'],
                bounds_scale=gym_cfg['GA_bounds_scale']
            )

        # GA_v2
        if opt_type == "GA_v2":
            return GeneticAlgorithmOptimizer_v2(
                base_params, param_space, evaluate_fn,
                population_size=gym_cfg['GA_population_size'],
                num_iterations=gym_cfg['GA_num_iterations'],
                mutation_rate=gym_cfg['GA_mutation_rate'],
                crossover_rate=gym_cfg['GA_crossover_rate'],
                bounds_scale=gym_cfg['GA_bounds_scale']
            )

        # Random Search
        if opt_type == "RA":
            return RandomOptimizer(
                base_params, param_space, evaluate_fn,
                num_iterations=gym_cfg['RA_num_iterations']
            )

        # Bayesian Optimization
        if opt_type == "BO":
            return BayesianOptimizer(
                base_params, param_space, evaluate_fn,
                n_initial_points=gym_cfg['BO_n_initial_points'],
                num_iterations=gym_cfg['BO_num_iterations'],
                initial_eval_epoch=gym_cfg['BO_initial_eval_epoch']
            )

        raise ValueError(f"Unknown optimization type {opt_type}")

    # -------------------------- training epochs -----------------------

    def get_final_train_epoch(self, opt_type):
        return {
            "GA": 600,
            "GA_v2": 800,
            "RA": 600,
            "BO": 1000
        }.get(opt_type, 600)

    # -------------------------- save CSV ------------------------------

    def save_opt_history(self, best_individuals, opt_type):
        best_params = best_individuals[-1][0]
        csv_name = f"{opt_type}_optimize_result.csv"
        csv_file = os.path.join(self.experiment_dir, csv_name)

        with open(csv_file, 'w', newline='') as f:
            w = csv.writer(f)
            header = ['Generation', 'Best_Score'] + list(best_params.keys())
            w.writerow(header)

            for i, (params, score) in enumerate(best_individuals):
                w.writerow([i + 1, score] + list(params.values()))

        print(f"[{opt_type}] Optimization results saved to {csv_file}")

    # ================================================================
    # 以下部分全部保持原样（evaluate, general model, cost,
    # XML generation, final training, random/default params）
    # ================================================================
    def generate_SRL_xml(self, xml_name, srl_params, save_path=False):
        """
        使用 Jinja2 模板生成 humanoid + SRL 的 MJCF 模型。
        返回:
            输出 XML 文件绝对路径
        """

        # ---- 1. 参数名检查 ---- 
        required_keys = [
            "leg1_length",
            "leg2_length",
            "enable_freejoint_z",
            "enable_freejoint_y",
            "enable_freejoint_x",
            "base_width",
            "base_distance",
        ]

        missing = [k for k in required_keys if k not in srl_params]
        if len(missing) > 0:
            raise KeyError(
                f"[ERROR] Missing SRL design parameters: {missing}\n"
                f"Required keys = {required_keys}"
            )

        # ---- 2. 选择输出目录（优先 save_path，其次 self.mjcf_folder） ----
        if save_path:
            output_dir = save_path
        else:
            # 使用 SRLGym 初始化时定义的 self.mjcf_folder
            # 确保这是绝对路径
            output_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../../assets/", self.mjcf_folder)
            )

        os.makedirs(output_dir, exist_ok=True)

        # ---- 3. 文件后缀处理 ----
        if not xml_name.endswith(".xml"):
            xml_name = xml_name + ".xml"

        # ---- 4. 参数取值 ----
        leg1 = float(srl_params["leg1_length"])
        leg2 = float(srl_params["leg2_length"])
        free_z = int(srl_params["enable_freejoint_z"])
        free_y = int(srl_params["enable_freejoint_y"])
        free_x = int(srl_params["enable_freejoint_x"])
        base_w = float(srl_params["base_width"])
        base_d = float(srl_params["base_distance"])

        # ---- 5. 调用 Jinja2 模型生成函数 ----
        ok = generate_hsrl_model(
            leg1_length=leg1,
            leg2_length=leg2,
            base_width=base_w,
            base_distance=base_d,
            enable_freejoint_z=free_z,
            enable_freejoint_y=free_y,
            enable_freejoint_x=free_x,
            output_name=xml_name,
            output_dir=output_dir,
        )

        if not ok:
            raise RuntimeError(f"[ERROR] Failed to generate MJCF model: {xml_name}")

        # ---- 6. 返回最终输出路径 ----
        out_path = os.path.join(output_dir, xml_name)
        print(f"[MJCF] Generated: {out_path}")

        return out_path


    # ---------------------------------------------------------------

    def _log_design_param(self, design_param, step):
        info = {f"design/{k}": v for k, v in design_param.items()}
        info["iteration"] = step
        wandb.log(info)

    # ---------------------------------------------------------------
    # ⬇️ evaluate(), evaluate_with_general_model(), cost()
    # 完整保留，无修改
    # ---------------------------------------------------------------

    def design_evaluate(self, design_params, max_epoch=False):
        cfg = self.cfg
        xml_name = 'hsrl_mode1'

        # 深拷贝配置
        train_cfg = deepcopy(cfg)
        train_cfg['wandb_activate'] = False

        if max_epoch:
            train_cfg['max_iterations'] = max_epoch

        train_cfg['train']['params']['config']['start_frame'] = 1
        srl_params = design_params

        # 生成 XML
        self.generate_SRL_xml(xml_name, srl_params)
        train_cfg['task']['env']['asset']['assetFileName'] = \
            self.mjcf_folder + '/' + xml_name + '.xml'
        train_cfg['train']['params']['config']['hsrl_checkpoint'] = False

        # 设计参数打进 observation
        if train_cfg['task']['env']['design_param_obs']:
            for k in train_cfg['task']['env']['design_params']:
                train_cfg['task']['env']['design_params'][k] = float(design_params[k])

        # 输出模型路径设置
        model_output_path = os.path.join(self.experiment_dir, 'nn')
        os.makedirs(model_output_path, exist_ok=True)
        model_output_file = os.path.join(model_output_path, 'mode1_id')

        train_cfg['train']['params']['config']['model_output_file'] = model_output_file
        train_cfg['train']['params']['config']['train_dir'] = \
            os.path.join(self.experiment_dir, 'logs')

        # ---- Subprocess Call  ----
        runner = self.SubprocRunner(train_cfg)
        job = runner.rlgpu(self.wandb_exp_name, design_params=srl_params)
        res = job.join()  
        runner.close()

        # 子进程异常检查
        if isinstance(res, Exception):
            print("Subprocess Error:\n", res)
            return -99999

        # unpack
        evaluate_reward, amp_reward, frame, _ = res

        self.curr_frame += frame

        # compute score
        design_cost = self.calc_design_cost(srl_params)
        wandb.log({
            "Evolution/srl_torque_cost": evaluate_reward,
            "Evolution/design_cost": design_cost * 500,
            "Evolution/amp_reward": amp_reward,
            "iteration": self.iteration
        })

        evaluate_reward += design_cost * 500

        if amp_reward < 150:
            final_score = -150
        else:
            final_score = evaluate_reward

        self._log_design_param(srl_params, self.iteration)
        wandb.log({"Evolution/evaluate_value": final_score})

        self.iteration += 1

        # best model update
        if final_score > self.best_evaluate_reward:
            self.best_evaluate_reward = final_score
            self.best_design_param = design_params
            shutil.copy(
                model_output_file + '.pth',
                os.path.join(model_output_path, 'best_model.pth')
            )

        return final_score


    # ---------------------------------------------------------------

    def design_evaluate_with_general_model(self, design_params, max_epoch=False):
        cfg = self.cfg
        xml_name = 'hsrl_mode1'
        train_cfg = deepcopy(cfg)
        train_cfg['wandb_activate'] = False

        if max_epoch:
            train_cfg['max_iterations'] = max_epoch

        train_cfg['train']['params']['config']['start_frame'] = 1
        srl_params = design_params

        self.generate_SRL_xml(xml_name, srl_params, pretrain=False)
        train_cfg['task']['env']['asset']['assetFileName'] = self.mjcf_folder + '/' + xml_name + '.xml'
        train_cfg['train']['params']['config']['hsrl_checkpoint'] = self.hsrl_checkpoint

        if train_cfg['task']['env']['design_param_obs']:
            for k in train_cfg['task']['env']['design_params']:
                train_cfg['task']['env']['design_params'][k] = float(design_params[k])

        model_output_path = os.path.join(self.experiment_dir, 'nn')
        os.makedirs(model_output_path, exist_ok=True)
        model_output_file = os.path.join(model_output_path, 'train_model')

        train_cfg['train']['params']['config']['model_output_file'] = model_output_file
        train_cfg['train']['params']['config']['train_dir'] = os.path.join(self.experiment_dir, 'logs')

        runner = self.SubprocRunner(train_cfg)
        try:
            evaluate_reward, amp_reward, frame, _ = runner.rlgpu(
                self.wandb_exp_name, design_params=srl_params
            ) 
        except Exception as e:
            print("Error:", e)
            return -9999
        finally:
            runner.close()

        self.curr_frame += frame

        design_cost = self.calc_design_cost(srl_params)
        evaluate_reward += design_cost * 500

        if amp_reward < 150:
            final_score = -150
        else:
            final_score = evaluate_reward

        wandb.log({
            "Evolution/srl_torque_cost": evaluate_reward,
            "Evolution/design_cost": design_cost * 500,
            "Evolution/amp_reward": amp_reward,
            "iteration": self.iteration
        })

        # 若训练稳定，则更新 general model
        if amp_reward > 400:
            shutil.copy(
                model_output_file + '.pth',
                os.path.join(model_output_path, 'general_model.pth')
            )
            self.hsrl_checkpoint = os.path.join(model_output_path, 'general_model.pth')

        # Best model
        if final_score > self.best_evaluate_reward:
            self.best_evaluate_reward = final_score
            self.best_design_param = design_params
            shutil.copy(
                model_output_file + '.pth',
                os.path.join(model_output_path, 'best_model.pth')
            )

        self._log_design_param(srl_params, self.iteration)
        wandb.log({"Evolution/evaluate_value": final_score})
        self.iteration += 1

        return final_score

    # ---------------------------------------------------------------

    def calc_design_cost(self, design_params):
        a = math.pi * (0.03 ** 2) * design_params['leg1_length']
        b = math.pi * (0.03 ** 2) * design_params['leg2_length']
        return -( a + b )

    # ---------------------------------------------------------------

    def final_design_train(self, max_epoch):
        cfg = self.cfg
        train_cfg = deepcopy(cfg)
        train_cfg['wandb_activate'] = False

        if max_epoch:
            train_cfg['max_iterations'] = max_epoch

        train_cfg['train']['params']['config']['start_frame'] = 1
        srl_params = self.best_design_param

        xml_name = 'hsrl_best_design'
        self.generate_SRL_xml(xml_name,  srl_params)
        save_path = os.path.join(self.experiment_dir, 'mjcf')
        self.generate_SRL_xml(xml_name,  srl_params, save_path=save_path)

        train_cfg['task']['env']['asset']['assetFileName'] = self.mjcf_folder + '/' + xml_name + '.xml'
        train_cfg['train']['params']['config']['hsrl_checkpoint'] = os.path.join(self.experiment_dir, 'nn', 'best_model.pth')

        if train_cfg['task']['env']['design_param_obs']:
            for k in train_cfg['task']['env']['design_params']:
                train_cfg['task']['env']['design_params'][k] = float(srl_params[k])

        model_output_path = os.path.join(self.experiment_dir, 'nn')
        os.makedirs(model_output_path, exist_ok=True)
        model_output_file = os.path.join(model_output_path, 'final_best_model')

        train_cfg['train']['params']['config']['model_output_file'] = model_output_file
        train_cfg['train']['params']['config']['train_dir'] = os.path.join(self.experiment_dir, 'logs')

        runner = self.SubprocRunner(train_cfg)
        try:
            evaluate_reward, amp_reward, frame, _ = runner.rlgpu(
                self.wandb_exp_name, design_params=srl_params
            )
        except Exception as e:
            print("Error:", e)
            return -9999
        finally:
            runner.close()

        self.curr_frame += frame
        design_cost = self.calc_design_cost(srl_params)
        evaluate_reward += design_cost * 500

        wandb.log({
            "Evolution/srl_torque_cost": evaluate_reward,
            "Evolution/design_cost": design_cost * 500,
            "Evolution/amp_reward": amp_reward,
            "iteration": self.iteration
        })

        final_score = -150 if amp_reward < 150 else evaluate_reward
        wandb.log({"Evolution/evaluate_value": final_score})
        self.iteration += 1

        return final_score

    # ---------------------------------------------------------------

    def default_SRL_design_parameters(self):
        return {
            "leg1_length": 0.60,
            "leg2_length": 0.55,
            "base_width": 0.095,
            "base_distance": 0.60,
            "enable_freejoint_z": 1,
            "enable_freejoint_y": 1,
            "enable_freejoint_x": 0,
        }

    def SRL_param_space(self):
        """定义 SRL 形态参数的取值空间，用于 GA / BO / Random Search."""
        return {
            "leg1_length":     ("real", 0.40, 0.80),
            "leg2_length":     ("real", 0.35, 0.75),
            "base_width":      ("real", 0.085, 0.130),
            "base_distance":   ("real", 0.30, 0.70),

            # 离散参数固定，不优化
            "enable_freejoint_z": ("fixed", 1),
            "enable_freejoint_y": ("fixed", 1),
            "enable_freejoint_x": ("fixed", 0),
        }



# ================================================================

if __name__ == '__main__':
    # For testing model generation
    srl_mode = 'mode1'
    name = 'humanoid_srl_mode1'
    pretrain = False
    srl_params = {
        "first_leg_lenth": 0.40,
        "first_leg_size": 0.03,
        "second_leg_lenth": 0.80,
        "second_leg_size": 0.03,
        "third_leg_size": 0.03,
    }

    srl_gen = SRL_mode1(name=name, pretrain=pretrain, **srl_params)
    base_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_path, '../../../assets/mjcf/humanoid_srl/')

    mjcf = ModelGenerator(srl_gen, save_path=save_path)
    mjcf.gen_basic_humanoid_xml()
    mjcf.get_SRL_dfs(back_load=not pretrain)
    mjcf.generate()
