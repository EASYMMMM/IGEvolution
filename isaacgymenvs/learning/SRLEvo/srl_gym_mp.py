'''
SRL Gym train/evaluate multi-process
用于创建形态-控制联合优化过程中的 Isaac Gym 子进程
'''

import hydra
from omegaconf import DictConfig, OmegaConf
import time
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import _restore, _override_sigma
from isaacgymenvs.utils.utils import retry
from isaacgymenvs.utils.reformat import omegaconf_to_dict
import wandb


def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']

    train_cfg['device'] = cfg.rl_device

    train_cfg['population_based_training'] = cfg.pbt.enabled
    train_cfg['pbt_idx'] = cfg.pbt.policy_idx if cfg.pbt.enabled else None

    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    print(f'Using rl_device: {cfg.rl_device}')
    print(f'Using sim_device: {cfg.sim_device}')
    print(train_cfg)

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict


class MyWandbAlgoObserver(AlgoObserver):
    """Need this to propagate the correct experiment name after initialization."""

    def __init__(self, cfg, wandb_exp_name):
        self.wandb_exp_name = wandb_exp_name
        super().__init__()
        self.cfg = cfg

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """

        import wandb
        experiment_name = self.wandb_exp_name
        wandb_unique_id = f"uid_{experiment_name}"
        print(f"Wandb using unique id {wandb_unique_id}")

        cfg = self.cfg

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception,))
        def init_wandb():
            wandb.init(
                project=cfg.wandb_project,
                # entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                tags=cfg.wandb_tags,
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=experiment_name,
                resume=True,
                settings=wandb.Settings(start_method='fork'),
            )

            if cfg.wandb_logcode_dir:
                wandb.run.log_code(root=cfg.wandb_logcode_dir)
                print('wandb running directory........', wandb.run.dir)

        print('Initializing WandB...')
        try:
            init_wandb()
        except Exception as exc:
            print(f'Could not initialize WandB! {exc}')


class SRLGym_process():
    """
    子进程内实际跑 Isaac Gym + RL-Games 训练的类。
    会被 mp_util.subproc_worker 包一层，在子进程中实例化并调用 rlgpu()。
    """

    def __init__(self, cfg):
        self.cfg = cfg

    # -----------------------
    # 一些保留的接口（兼容用）
    # -----------------------
    def test(self):
        print('SUBPROCESS TEST')

    def init_wandb(self, wandb_experiment_name):
        """
        单独使用时可以在子进程内初始化一个 WandB run。
        注意：在 GA/BO 外层你目前是用主进程 log，所以一般不需要调用这个函数。
        """
        cfg = self.cfg
        wandb_unique_id = f"uid_{wandb_experiment_name}"
        print(f"Wandb using unique id {wandb_unique_id}")

        @retry(3, exceptions=(Exception,))
        def my_init_wandb():
            wandb.init(
                project=cfg.wandb_project,
                # entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                tags=cfg.wandb_tags,
                sync_tensorboard=False,
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

    def finish_wandb(self):
        wandb.finish()

    def _log_design_param(self, design_param, step):
        info_dict = {
            "design/leg1_lenth": design_param["first_leg_lenth"],
            "design/leg1_size": design_param["first_leg_size"],
            "design/leg2_lenth": design_param["second_leg_lenth"],
            "design/leg2_size": design_param["second_leg_size"],
            "design/end_size": design_param["third_leg_size"],
            "global_step": step,
        }
        wandb.log(info_dict)

    # -----------------------
    # 内部工具函数（重构部分）
    # -----------------------

    def _setup_env_and_vecenv(self, cfg, run_name):
        """
        注册 IsaacGym env / RL-Games vecenv。
        原来这些逻辑是直接写在 rlgpu 里，现在抽出来。
        """
        import os
        from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
        from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
        from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank
        from isaacgymenvs.tasks import isaacgym_task_map
        from isaacgymenvs.utils.utils import set_np_formatting, set_seed
        from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
        import gym
        import isaacgymenvs
        from rl_games.common import env_configurations, vecenv

        # pbt 检查
        if cfg.pbt.enabled:
            initial_pbt_check(cfg)

        # 仅做 numpy 打印设置
        set_np_formatting()

        # global rank of the GPU
        global_rank = int(os.getenv("RANK", "0"))

        # 设置随机种子
        cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

        # 创建 env 的工厂函数
        def create_isaacgym_env(**kwargs):
            envs = isaacgymenvs.make(
                cfg.seed,
                cfg.task_name,
                cfg.task.env.numEnvs,
                cfg.sim_device,
                cfg.rl_device,
                cfg.graphics_device_id,
                cfg.headless,
                cfg.multi_gpu,
                cfg.capture_video,
                cfg.force_render,
                cfg,
                **kwargs,
            )
            if cfg.capture_video:
                envs.is_vector_env = True
                envs = gym.wrappers.RecordVideo(
                    envs,
                    f"videos/{run_name}",
                    step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                    video_length=cfg.capture_video_len,
                )
            return envs

        # 在 rl_games 中注册 env 类型
        env_configurations.register('rlgpu', {
            'vecenv_type': 'RLGPU',
            'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
        })

        # 根据任务设置 obs 类型（dict_obs 或普通）
        ige_env_cls = isaacgym_task_map[cfg.task_name]  # 环境 class
        dict_cls = getattr(ige_env_cls, 'dict_obs_cls', False)

        if dict_cls:
            obs_spec = {}
            actor_net_cfg = cfg.train.params.network
            obs_spec['obs'] = {
                'names': list(actor_net_cfg.inputs.keys()),
                'concat': not actor_net_cfg.name == "complex_net",
                'space_name': 'observation_space',
            }
            if "central_value_config" in cfg.train.params.config:
                critic_net_cfg = cfg.train.params.config.central_value_config.network
                obs_spec['states'] = {
                    'names': list(critic_net_cfg.inputs.keys()),
                    'concat': not critic_net_cfg.name == "complex_net",
                    'space_name': 'state_space',
                }

            vecenv.register(
                'RLGPU',
                lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(
                    config_name, num_actors, obs_spec, **kwargs
                )
            )
        else:
            vecenv.register(
                'RLGPU',
                lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs)
            )

        return global_rank

    def _create_observers(self, cfg, wandb_exp_name, global_rank):
        """
        创建 RL-Games 的 observer 列表，包含：
        - RLGPUAlgoObserver
        - PbtAlgoObserver（可选）
        - MyWandbAlgoObserver（可选）
        """
        from isaacgymenvs.pbt.pbt import PbtAlgoObserver
        from isaacgymenvs.utils.rlgames_utils import RLGPUAlgoObserver, MultiObserver

        observers = [RLGPUAlgoObserver()]

        if cfg.pbt.enabled:
            pbt_observer = PbtAlgoObserver(cfg)
            observers.append(pbt_observer)

        if cfg.wandb_activate:
            # 多 GPU 时给不同 rank 不同 seed
            cfg.seed += global_rank
            if global_rank == 0:
                # 只在 rank0 上初始化 WandB
                wandb_observer = MyWandbAlgoObserver(cfg, wandb_exp_name)
                observers.append(wandb_observer)

        # 返回 MultiObserver 封装后的对象
        from isaacgymenvs.utils.rlgames_utils import MultiObserver as _MultiObs
        return _MultiObs(observers)

    def _register_algos_and_networks(self, runner):
        """
        在 RL-Games 中注册 AMP 和 SRL 的算法 / 网络 / 模型。
        原代码里这部分逻辑是写在 build_runner 里的。
        """
        from rl_games.algos_torch import model_builder
        from isaacgymenvs.learning import amp_continuous, amp_players, amp_models, amp_network_builder
        from isaacgymenvs.learning.SRLEvo import srl_continuous, srl_models, srl_players, srl_continuous_marl, srl_bot_continuous
        from isaacgymenvs.learning.SRLEvo import srl_network_builder, srl_gym

        # AMP
        runner.algo_factory.register_builder('amp_continuous',
                                             lambda **kwargs: amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous',
                                               lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp',
                                     lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp',
                                       lambda **kwargs: amp_network_builder.AMPBuilder())

        # SRL 
        runner.algo_factory.register_builder('srl_bot_continuous', lambda **kwargs : srl_bot_continuous.SRL_Bot_Agent(**kwargs))
        runner.algo_factory.register_builder('srl_continuous', lambda **kwargs : srl_continuous.SRLAgent(**kwargs))
        runner.algo_factory.register_builder('srl_continuous_marl', lambda **kwargs : srl_continuous_marl.SRL_MultiAgent(**kwargs))
        runner.player_factory.register_builder('srl_continuous', lambda **kwargs : srl_players.SRLPlayerContinuous(**kwargs))
        runner.player_factory.register_builder('srl_continuous_marl', lambda **kwargs : srl_players.SRLPlayerContinuous(**kwargs))
        runner.player_factory.register_builder('srl_bot_continuous', lambda **kwargs : srl_players.SRL_Bot_PlayerContinuous(**kwargs))
        model_builder.register_model('continuous_srl', lambda network, **kwargs : srl_models.ModelSRLContinuous(network))
        model_builder.register_network('amp_humanoid', lambda **kwargs : srl_network_builder.HumanoidBuilder())
        model_builder.register_network('srl', lambda **kwargs : srl_network_builder.SRLBuilder())

    def _prepare_rlgames_config(self, cfg):
        """
        把 Hydra/IsaacGym 的 cfg.train 转成 RL-Games 使用的 dict，并调用 preprocess_train_config。
        """
        rlg_config_dict = omegaconf_to_dict(cfg.train)
        rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)
        return rlg_config_dict

    def _maybe_dump_experiment_cfg(self, cfg):
        """
        创建 experiment_dir，并把 cfg 写入 config.yaml。
        """
        import os
        from datetime import datetime

        if cfg.test:
            return None

        exp_dir = cfg.get('experiment_dir', False)
        if exp_dir:
            experiment_dir = exp_dir
        else:
            experiment_dir = os.path.join(
                'runs',
                cfg.train.params.config.name + '_{date:%d-%H-%M-%S}'.format(date=datetime.now())
            )

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

        return experiment_dir

    def _build_runner(self, algo_observer, rlg_config_dict):
        """
        构建 RL-Games Runner，并注册 AMP / SRL 等 builder。
        """
        from rl_games.torch_runner import Runner

        runner = Runner(algo_observer)
        # 注册算法 / 网络 / 模型
        self._register_algos_and_networks(runner)
        # 加载配置
        runner.load(rlg_config_dict)
        runner.reset()
        return runner

    # -----------------------
    # 主要对外接口：子进程训练入口
    # -----------------------

    def rlgpu(self, wandb_exp_name, design_params=None):
        """
        子进程中真正执行 Isaac Gym + RL-Games 训练的函数。
        外层通过 mp_util.subproc_worker 调用：runner.rlgpu(...).results
        """
        print('*************************************')
        print('     Create Subprocess RLGPU         ')
        print('*************************************')
        import logging
        import os
        from datetime import datetime
        from hydra.utils import to_absolute_path

        cfg = self.cfg

        # 生成 run_name（用于视频保存路径等）
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{cfg.wandb_name}_{time_str}"

        # 确保 checkpoint 路径为绝对路径
        if cfg.checkpoint:
            cfg.checkpoint = to_absolute_path(cfg.checkpoint)

        # 配置 env / vecenv / seed 等，并拿到 global_rank
        global_rank = self._setup_env_and_vecenv(cfg, run_name)

        # 构造 RL-Games 的 config dict
        rlg_config_dict = self._prepare_rlgames_config(cfg)

        # 创建 observers（含 WandB 和 PBT）
        algo_observer = self._create_observers(cfg, wandb_exp_name, global_rank)

        # 构建 Runner（注册所有算法和网络）
        runner = self._build_runner(algo_observer, rlg_config_dict)

        # 创建实验目录并保存 config.yaml
        self._maybe_dump_experiment_cfg(cfg)

        # 训练 / 测试
        evaluate_reward, evaluate_info, frame, epoch_num = self.run(
            runner,
            {
                'train': not cfg.test,
                'play': cfg.test,
                'checkpoint': cfg.checkpoint,
                'sigma': cfg.sigma if cfg.sigma != '' else None,
            }
        )

        # 这里可以按需把 design_params 打 log（现在在外层做了就没在这里重复 log）
        # if design_params:
        #     self._log_design_param(design_params, frame)

        return evaluate_reward, evaluate_info, frame, epoch_num

    def run(self, runner, args):
        """
        封装 RL-Games 的训练 / play 入口。
        原代码有 train / play / else 三种分支，这里统一成：
        - play: runner.run_play
        - 其它：一律当作训练，返回 (reward, amp_reward, frame, summary_dir)
        """

        if args['play']:
            runner.run_play(args)
            # 保持返回结构一致：4 个值
            return 0.0, 0.0, 0, None

        # 默认走训练分支
        print(runner.algo_name)
        agent = runner.algo_factory.create(runner.algo_name, base_name='run', params=runner.params)
        _restore(agent, args)
        _override_sigma(agent, args)

        # 对应你的 srl/amp agent.train() 返回 (evaluate_reward, evaluate_amp_reward, frame, summary_dir)
        evaluate_reward, evaluate_info, frame, epoch_num = agent.train()
        return evaluate_reward, evaluate_info, frame, epoch_num