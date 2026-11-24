"""SRL morphology search with a separate Isaac Gym training process per evaluation."""
import csv
from copy import deepcopy
from datetime import datetime
import math
import os
from pathlib import Path
import shutil

import wandb
from omegaconf import OmegaConf

from isaacgymenvs.srl_mjcf_generator.generator.generate_hsrl import TEMPLATE_DIR, generate_hsrl_model
from isaacgymenvs.srl_mjcf_generator.generator.real_hri_v2 import (
    generate_real_hri_v2_model, read_real_hri_v2_design,
)
from isaacgymenvs.utils.utils import set_seed
from .designer_opt import (
    BayesianOptimizer, GeneticAlgorithmOptimizer, GeneticAlgorithmOptimizer_v2, RandomOptimizer,
)
from .mp_util import subproc_worker
from .srl_gym_mp import SRLGym_process


ASSET_ROOT = Path(__file__).resolve().parents[3] / 'assets'


class SRLGym:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gym_cfg = cfg.train.gym
        self.is_real_hri = cfg.task.name == 'SRL_Real_HRI'
        self.source_asset = ASSET_ROOT / cfg.task.env.asset.assetFileName
        self.hsrl_checkpoint = cfg.train.params.config.hsrl_checkpoint
        self.best_evaluate_reward = float('-inf')
        self.best_design_param = None
        self.iteration = 1
        self.curr_frame = 0
        name = cfg.train.params.config.name + datetime.now().strftime('_Evo_%d-%H-%M-%S-%f')
        self.experiment_dir = Path(cfg.experiment_dir or Path('runs') / name).resolve()
        self.wandb_exp_name = name
        self.SubprocRunner = subproc_worker(SRLGym_process, ctx="spawn", daemon=True)

    def train(self):
        if self.cfg.test or self.cfg.checkpoint:
            raise ValueError('Morphology search trains new candidates. Use train.params.config.hsrl_checkpoint '
                             'to seed them, or SRL_Evo_train.py for playback/resume of a single policy.')
        self.cfg.seed = set_seed(self.cfg.seed, self.cfg.torch_deterministic)
        opt_type = self.gym_cfg.design_opt
        optimizer = self.build_optimizer(opt_type)
        # Never pick up checkpoints or scores from an earlier search.
        self.experiment_dir.mkdir(parents=True, exist_ok=False)
        (self.experiment_dir / 'nn').mkdir()
        OmegaConf.save(self.cfg, self.experiment_dir / 'config.yaml')
        if self.cfg.wandb_activate:
            wandb.init(project=('SRLGym' if self.cfg.wandb_project == 'isaacgymenvs'
                                else self.cfg.wandb_project),
                       group=self.cfg.wandb_group, tags=list(self.cfg.wandb_tags),
                       name=self.wandb_exp_name, config=OmegaConf.to_container(self.cfg, resolve=True))
        try:
            history = optimizer.optimize()
            self.save_opt_history(history, opt_type)
            default_epochs = {'GA': 600, 'GA_v2': 800, 'RA': 600, 'BO': 1000}
            final_epochs = self.gym_cfg.get('final_eval_epoch') or default_epochs[opt_type]
            self.final_design_train(final_epochs)
            if self.cfg.wandb_activate:
                for log in (self.experiment_dir / 'logs').rglob('events.out.tfevents.*'):
                    wandb.save(str(log), base_path=str(self.experiment_dir))
        finally:
            if self.cfg.wandb_activate:
                wandb.finish()
        print(f'[{opt_type}] Optimization finished: {self.experiment_dir}')

    def build_optimizer(self, opt_type):
        args = (self.default_SRL_design_parameters(), self.SRL_param_space(), self.design_evaluate)
        log = wandb.log if self.cfg.wandb_activate else None
        cfg = self.gym_cfg
        if opt_type in ('GA', 'GA_v2'):
            cls = GeneticAlgorithmOptimizer if opt_type == 'GA' else GeneticAlgorithmOptimizer_v2
            return cls(*args, population_size=cfg.GA_population_size, num_iterations=cfg.GA_num_iterations,
                       mutation_rate=cfg.GA_mutation_rate, crossover_rate=cfg.GA_crossover_rate,
                       initial_eval_epoch=cfg.get('GA_initial_eval_epoch', 700), wandb_log=log)
        if opt_type == 'RA':
            return RandomOptimizer(*args, num_iterations=cfg.RA_num_iterations,
                                   initial_eval_epoch=cfg.get('RA_initial_eval_epoch', 700), wandb_log=log)
        if opt_type == 'BO':
            return BayesianOptimizer(*args, n_initial_points=cfg.BO_n_initial_points,
                                     num_iterations=cfg.BO_num_iterations,
                                     initial_eval_epoch=cfg.BO_initial_eval_epoch, wandb_log=log)
        raise ValueError(f'Unknown optimization type {opt_type}')

    def save_opt_history(self, best_individuals, opt_type):
        names = list(best_individuals[-1][0])
        with (self.experiment_dir / f'{opt_type}_optimize_result.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Generation', 'Best_Score', *names])
            for i, (params, score) in enumerate(best_individuals, 1):
                writer.writerow([i, score, *(params[name] for name in names)])

    def generate_SRL_xml(self, xml_name, srl_params, save_path):
        if self.is_real_hri:
            return generate_real_hri_v2_model(srl_params, xml_name, save_path, self.source_asset)
        output = generate_hsrl_model(
            **srl_params, output_name=xml_name, output_dir=str(save_path),
            base_xml_path=Path(TEMPLATE_DIR).parent / 'base/base_humanoid.xml',
            template_name='SRL_template.xml.j2',
        )
        if not output:
            raise RuntimeError(f'Failed to generate {xml_name}')
        return output

    def _train_design(self, params, max_epoch, checkpoint, label, model_name):
        cfg = deepcopy(self.cfg)
        cfg.wandb_activate = False
        cfg.experiment_dir = str(self.experiment_dir / 'logs' / label)
        if max_epoch:
            cfg.max_iterations = max_epoch
        env = cfg.task.env
        xml = self.generate_SRL_xml(label, params, self.experiment_dir / 'mjcf')
        env.asset.assetFileName = os.path.relpath(xml, ASSET_ROOT)
        hinges = sum(int(params['enable_freejoint_' + axis]) for axis in 'xyz')
        env.srl_free_actions_num = hinges + (3 if self.is_real_hri else 0)
        env.design_params = dict(params)

        train = cfg.train.params.config
        train.start_frame = 0
        train.hsrl_checkpoint = checkpoint
        if checkpoint:
            # The joint weights already include the human policy.
            train.humanoid_checkpoint = False
        output = self.experiment_dir / 'nn' / model_name
        train.model_output_file = str(output)
        train.train_dir = cfg.experiment_dir
        runner = self.SubprocRunner(cfg)
        try:
            result = runner.rlgpu(self.wandb_exp_name, design_params=params).join()
        finally:
            runner.close()
        if isinstance(result, Exception):
            raise RuntimeError(f'{label} training failed: {result}') from result
        _, info, frames, epoch = result
        if not output.with_suffix('.pth').is_file():
            raise RuntimeError(f'{label} did not save its policy: {output}.pth')
        if not all(math.isfinite(info[k]) for k in ('amp_rewards', 'humanoid_rewards', 'srl_rewards', 'ep_length')):
            raise ValueError(f'{label} returned non-finite episode metrics: {info}')
        if info['ep_length'] <= 0:
            raise ValueError(f'{label} completed no episodes. Increase the candidate training budget.')
        self.curr_frame += frames
        score = self.calc_evaluate_reward(info, params)
        metrics = {'candidate': label, 'score': score, **params, **info, 'frames': frames, 'epoch': epoch}
        with (self.experiment_dir / 'evaluations.csv').open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(metrics)
        if self.cfg.wandb_activate:
            wandb.log({'iteration': self.iteration, **{f'design/{k}': v for k, v in params.items()},
                       'Evolution/humanoid_task_reward': info['humanoid_rewards'],
                       'Evolution/srl_reward': info['srl_rewards'], 'Evolution/ep_length': info['ep_length'],
                       'Evolution/design_cost': self.calc_design_cost(params) * 10,
                       'Evolution/amp_reward': info['amp_rewards'], 'Evolution/evaluate_value': score})
        return score, info, output.with_suffix('.pth')

    def design_evaluate(self, design_params, max_epoch=False):
        label = f'candidate_{self.iteration:04d}'
        score, info, model = self._train_design(design_params, max_epoch, self.hsrl_checkpoint, label, 'train_model')
        if self.cfg.task.env.design_param_obs and info['amp_rewards'] > 400:
            general = self.experiment_dir / 'nn/general_model.pth'
            shutil.copyfile(model, general)
            self.hsrl_checkpoint = str(general)
        if score > self.best_evaluate_reward:
            self.best_evaluate_reward = score
            self.best_design_param = dict(design_params)
            shutil.copyfile(model, self.experiment_dir / 'nn/best_model.pth')
        self.iteration += 1
        return score

    def final_design_train(self, max_epoch):
        checkpoint = str(self.experiment_dir / 'nn/best_model.pth')
        score, _, _ = self._train_design(self.best_design_param, max_epoch, checkpoint,
                                         'srl_real_hri_best_design', 'final_best_model')
        return score

    @staticmethod
    def calc_design_cost(params):
        return -(0.12 * params['leg1_length'] + 0.12 * params['leg2_length'] + 0.60 * params['base_distance'])

    def calc_evaluate_reward(self, info, params):
        if info['amp_rewards'] < 1500:
            return -1
        return 0.002 * info['humanoid_rewards'] + 10 * self.calc_design_cost(params) + 0.002 * info['ep_length']

    def default_SRL_design_parameters(self):
        if self.is_real_hri:
            return read_real_hri_v2_design(self.source_asset)
        return dict(leg1_length=0.661, leg2_length=0.590, base_width=0.095, base_distance=0.60,
                    enable_freejoint_z=1, enable_freejoint_y=1, enable_freejoint_x=0)

    def SRL_param_space(self):
        gym_cfg = self.cfg["train"]["gym"]
        cfg_space = gym_cfg.get("morphology_space", None)

        # 如果 cfg 里没配，就用默认 hard-code 一份，防止报错
        if cfg_space is None:
            return {
                "leg1_length": ("real", 0.40, 0.80),
                "leg2_length": ("real", 0.35, 0.75),
                "base_width": ("real", 0.085, 0.130),
                "base_distance": ("real", 0.30, 0.70),
                "enable_freejoint_z": ("fixed", 1),
                "enable_freejoint_y": ("fixed", 1),
                "enable_freejoint_x": ("fixed", 0),
            }

        param_space = {}

        for name, spec in cfg_space.items():
            ptype = spec.get("type", "real")
            ptype = ptype.lower()

            if ptype in ("real", "int"):
                low = float(spec["low"])
                high = float(spec["high"])
                param_space[name] = (ptype, low, high)
            elif ptype == "fixed":
                value = spec["value"]
                param_space[name] = (ptype, value)
            else:
                raise ValueError(f"Unknown param type '{ptype}' for '{name}' in morphology_space")

        return param_space
