'''
SRL Gym train/evaluate multi-process
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
                #entity=cfg.wandb_entity,
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
    def __init__(self, cfg):
        self.cfg = cfg

    def test(self):
        print('SUBPROCESS TEST')

    def init_wandb(self,wandb_experiment_name):
        
        cfg = self.cfg
        wandb_unique_id = f"uid_{wandb_experiment_name}"
        print(f"Wandb using unique id {wandb_unique_id}")
        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception,))
        def my_init_wandb():
            wandb.init(
                project=cfg.wandb_project,
                #entity=cfg.wandb_entity,
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

    def _log_design_param(self,design_param,step):
        info_dict = {
                "design/leg1_lenth" : design_param["first_leg_lenth"],
                "design/leg1_size"  : design_param["first_leg_size"],
                "design/leg2_lenth": design_param["second_leg_lenth"],
                "design/leg2_size" : design_param["second_leg_size"],
                "design/end_size"  : design_param["third_leg_size"],
                "global_step" :  step
        }
        wandb.log(info_dict )

    def rlgpu(self, wandb_exp_name, design_params=None):
        print('SUBPROCESS RLGPU')
        cfg = self.cfg
        import logging
        import os
        from datetime import datetime

        # noinspection PyUnresolvedReferences
        
        from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
        from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank
        from hydra.utils import to_absolute_path
        from isaacgymenvs.tasks import isaacgym_task_map
        import gym
        from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
        from isaacgymenvs.utils.utils import set_np_formatting, set_seed

        if cfg.pbt.enabled:
            initial_pbt_check(cfg)

        from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
        from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner
        from rl_games.algos_torch import model_builder
        from isaacgymenvs.learning import amp_continuous
        from isaacgymenvs.learning import amp_players
        from isaacgymenvs.learning import amp_models
        from isaacgymenvs.learning import amp_network_builder
        from isaacgymenvs.learning.SRLEvo import srl_continuous,srl_models,srl_players
        from isaacgymenvs.learning.SRLEvo import srl_network_builder
        from isaacgymenvs.learning.SRLEvo import srl_gym
        import isaacgymenvs


        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{cfg.wandb_name}_{time_str}"

        # ensure checkpoints can be specified as relative paths
        if cfg.checkpoint:
            cfg.checkpoint = to_absolute_path(cfg.checkpoint)

        cfg_dict = omegaconf_to_dict(cfg)
        # print('cfg_dict:')
        # print_dict(cfg_dict)

        # set numpy formatting for printing only
        set_np_formatting()

        # global rank of the GPU
        global_rank = int(os.getenv("RANK", "0"))

        # sets seed. if seed is -1 will pick a random one
        cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

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

        env_configurations.register('rlgpu', {
            'vecenv_type': 'RLGPU',
            'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
        })

        ige_env_cls = isaacgym_task_map[cfg.task_name] # 环境class
        dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, 'dict_obs_cls') and ige_env_cls.dict_obs_cls else False
        # 检查env环境中是否定义了dict_obs_cls=True
        if dict_cls:
            obs_spec = {}
            actor_net_cfg = cfg.train.params.network
            obs_spec['obs'] = {'names': list(actor_net_cfg.inputs.keys()), 'concat': not actor_net_cfg.name == "complex_net", 'space_name': 'observation_space'}
            if "central_value_config" in cfg.train.params.config:
                critic_net_cfg = cfg.train.params.config.central_value_config.network
                obs_spec['states'] = {'names': list(critic_net_cfg.inputs.keys()), 'concat': not critic_net_cfg.name == "complex_net", 'space_name': 'state_space'}
            
            vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs))
        else:
            vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

        # 处理rl_games训练用到的train.cfg配置文件
        rlg_config_dict = omegaconf_to_dict(cfg.train)
        rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)
        
        # 创建observers
        observers = [RLGPUAlgoObserver()]
        if cfg.pbt.enabled:
            pbt_observer = PbtAlgoObserver(cfg)
            observers.append(pbt_observer)
        if cfg.wandb_activate:
            cfg.seed += global_rank
            if global_rank == 0:
                # initialize wandb only once per multi-gpu run
                wandb_observer = MyWandbAlgoObserver(cfg, wandb_exp_name)
                observers.append(wandb_observer)

        # register new AMP network builder and agent
        def build_runner(algo_observer):
            runner = Runner(algo_observer)
            runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
            runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
            model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
            model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())
            # SRL 
            runner.algo_factory.register_builder('srl_continuous', lambda **kwargs : srl_continuous.SRLAgent(**kwargs))
            runner.algo_factory.register_builder('srl_gym', lambda **kwargs : srl_gym.SRLGym(**kwargs))
            runner.player_factory.register_builder('srl_continuous', lambda **kwargs : srl_players.SRLPlayerContinuous(**kwargs))
            model_builder.register_model('continuous_srl', lambda network, **kwargs : srl_models.ModelSRLContinuous(network))
            model_builder.register_network('amp_humanoid', lambda **kwargs : srl_network_builder.HumanoidBuilder())
            model_builder.register_network('srl', lambda **kwargs : srl_network_builder.SRLBuilder())
            return runner

        # convert CLI arguments into dictionary
        # create runner and set the settings
        runner = build_runner(MultiObserver(observers))
        runner.load(rlg_config_dict)
        runner.reset()

        # 创建保存文件夹 
        if not cfg.test:
            exp_dir = cfg.get('experiment_dir',False)
            if exp_dir:
                experiment_dir = exp_dir
            else:
                experiment_dir = os.path.join('runs', cfg.train.params.config.name + 
                '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
            # dump config dict 
            os.makedirs(experiment_dir, exist_ok=True)
            with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
                f.write(OmegaConf.to_yaml(cfg))

 
        evaluate_reward, epoch_num, frame, summary_dir = self.run(runner, { 'train': not cfg.test,
                                                        'play': cfg.test,
                                                        'checkpoint': cfg.checkpoint,
                                                        'sigma': cfg.sigma if cfg.sigma != '' else None
                                                        })
        
        # if design_params:
        #     self._log_design_param(design_params, frame)
        # wandb.log({'Evolution/reward':evaluate_reward, 'global_step': frame} )
                
        # wandb.finish()  # finish wandb in subprocess
        return evaluate_reward, epoch_num, frame, summary_dir
 
    def run(self, runner, args):
        if args['train']:
            print(runner.algo_name)
            agent = runner.algo_factory.create(runner.algo_name, base_name='run', params=runner.params)
            _restore(agent, args)
            _override_sigma(agent, args)
            evaluate_reward, epoch_num, frame, summary_dir = agent.train()
            return evaluate_reward, epoch_num, frame , summary_dir
        elif args['play']:
            runner.run_play(args)
            return 0, 0, 0
        else:
            print(runner.algo_name)
            agent = runner.algo_factory.create(runner.algo_name, base_name='run', params=runner.params)
            _restore(agent, args)
            _override_sigma(agent, args)
            last_mean_rewards, epoch_num, frame = agent.train()
            return last_mean_rewards, epoch_num, frame, summary_dir
         


 