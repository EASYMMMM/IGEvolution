# train.py
# Script to train policies in Isaac Gym
#
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

import hydra

from omegaconf import DictConfig, OmegaConf
from omegaconf import DictConfig, OmegaConf


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


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    import logging
    import os
    from datetime import datetime

    # noinspection PyUnresolvedReferences
    import isaacgym
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
    from isaacgymenvs.learning import common_agent
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    from isaacgymenvs.learning.SRLEvo import srl_continuous,srl_models,srl_players,srl_continuous_marl
    from isaacgymenvs.learning.SRLEvo import srl_network_builder
    from isaacgymenvs.learning.SRLEvo import srl_bot_continuous
    import isaacgymenvs


    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print('cfg_dict:')
    print_dict(cfg_dict)

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
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())
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
        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    #'''
    # 导出jit模型
    # =====================================================================
    import torch
    if cfg.test and cfg.checkpoint:
        print(f"\n[True Export] 正在从 {cfg.checkpoint} 导出真正的 JIT 模型...")
        try:
            # 1. 让 rl_games 根据 YAML 自动创建完整的 Player 和 Network
            player = runner.create_player()
            
            # 2. 将 .pth 权重注入到这个“活体”网络中
            player.restore(cfg.checkpoint)
            
            # 3. 提取真实的 PyTorch 模块！
            # 这里的 actor_mlp 已经自动包含了 YAML 里定义的所有 LayerNorm、激活函数等
            # 我们不需要知道里面有什么，直接把它整个端走！
            class TrueExportActor(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.actor_mlp = model.a2c_network.actor_mlp
                    self.mu = model.a2c_network.mu
                
                def forward(self, x):
                    x = self.actor_mlp(x)
                    return self.mu(x)
            
            # 实例化我们要导出的部分
            export_net = TrueExportActor(player.model).to(player.device).eval()
            
            # 4. 自动获取 YAML 里定义的观测维度
            obs_dim = player.obs_shape[0]
            dummy_input = torch.zeros(1, obs_dim, device=player.device)
            
            # 5. Trace：让 PyTorch 自动录制计算图
            traced_script_module = torch.jit.trace(export_net, dummy_input)
            
            # 6. 保存
            save_path = cfg.checkpoint.replace(".pth", "_TrueJIT.pt")
            traced_script_module.save(save_path)
            
            print(f"✅✅✅ 完美成功！模型已基于 YAML 结构准确导出为:")
            print(f"👉 {os.path.abspath(save_path)}\n")
            
        except Exception as e:
            print(f"❌ JIT 导出失败: {e}\n")
    # =====================================================================
        player.run()

    else:
        experiment_dir = os.path.join('runs', cfg.train.params.config.name + 
        '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
        # dump config dict 
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

        runner.run({
            'train': not cfg.test,
            'play': cfg.test,
            'checkpoint': cfg.checkpoint,
            'sigma': cfg.sigma if cfg.sigma != '' else None
        })


    '''
    # 创建保存文件夹 
    if not cfg.test:
        experiment_dir = os.path.join('runs', cfg.train.params.config.name + 
        '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
        # dump config dict 
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })
    '''

if __name__ == "__main__":
    launch_rlg_hydra()
