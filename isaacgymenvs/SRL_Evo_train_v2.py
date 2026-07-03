# train_v2.py
# Isolated training entrypoint for v2 task experiments.

import hydra

from omegaconf import DictConfig, OmegaConf


def preprocess_train_config(cfg, config_dict):
    train_cfg = config_dict["params"]["config"]

    train_cfg["device"] = cfg.rl_device
    train_cfg["population_based_training"] = cfg.pbt.enabled
    train_cfg["pbt_idx"] = cfg.pbt.policy_idx if cfg.pbt.enabled else None
    train_cfg["full_experiment_name"] = cfg.get("full_experiment_name")

    print(f"Using rl_device: {cfg.rl_device}")
    print(f"Using sim_device: {cfg.sim_device}")
    print(train_cfg)

    try:
        model_size_multiplier = config_dict["params"]["network"]["mlp"]["model_size_multiplier"]
        if model_size_multiplier != 1:
            units = config_dict["params"]["network"]["mlp"]["units"]
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

    import isaacgym
    import gym
    import isaacgymenvs
    from hydra.utils import to_absolute_path
    from rl_games.algos_torch import model_builder
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning.SRLEvo import srl_bot_continuous
    from isaacgymenvs.learning.SRLEvo import srl_continuous, srl_continuous_marl
    from isaacgymenvs.learning.SRLEvo import srl_models, srl_network_builder, srl_players
    from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
    from isaacgymenvs.tasks_v2 import isaacgym_task_map_v2
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.rlgames_utils import ComplexObsRLGPUEnv, MultiObserver, RLGPUAlgoObserver, RLGPUEnv
    from isaacgymenvs.utils.rlgames_utils_v2 import get_rlgames_env_creator_v2
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver

    del logging
    del isaacgymenvs

    if cfg.pbt.enabled:
        initial_pbt_check(cfg)

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print("cfg_dict:")
    print_dict(cfg_dict)
    set_np_formatting()

    global_rank = int(os.getenv("RANK", "0"))
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_isaacgym_env(**kwargs):
        task_cfg = omegaconf_to_dict(cfg.task)
        creator = get_rlgames_env_creator_v2(
            seed=cfg.seed,
            task_config=task_cfg,
            task_name=cfg.task_name,
            sim_device=cfg.sim_device,
            rl_device=cfg.rl_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless,
            multi_gpu=cfg.multi_gpu,
            virtual_screen_capture=cfg.capture_video,
            force_render=cfg.force_render,
            **kwargs,
        )
        envs = creator()
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env_configurations.register("rlgpu", {
        "vecenv_type": "RLGPU",
        "env_creator": lambda **kwargs: create_isaacgym_env(**kwargs),
    })

    ige_env_cls = isaacgym_task_map_v2[cfg.task_name]
    dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, "dict_obs_cls") and ige_env_cls.dict_obs_cls else False
    if dict_cls:
        obs_spec = {}
        actor_net_cfg = cfg.train.params.network
        obs_spec["obs"] = {
            "names": list(actor_net_cfg.inputs.keys()),
            "concat": not actor_net_cfg.name == "complex_net",
            "space_name": "observation_space",
        }
        if "central_value_config" in cfg.train.params.config:
            critic_net_cfg = cfg.train.params.config.central_value_config.network
            obs_spec["states"] = {
                "names": list(critic_net_cfg.inputs.keys()),
                "concat": not critic_net_cfg.name == "complex_net",
                "space_name": "state_space",
            }
        vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs))
    else:
        vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = [RLGPUAlgoObserver()]
    if cfg.pbt.enabled:
        observers.append(PbtAlgoObserver(cfg))
    if cfg.wandb_activate and global_rank == 0:
        cfg.seed += global_rank
        observers.append(WandbAlgoObserver(cfg))

    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder("amp_continuous", lambda **kwargs: amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder("amp_continuous", lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model("continuous_amp", lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
        model_builder.register_network("amp", lambda **kwargs: amp_network_builder.AMPBuilder())

        runner.algo_factory.register_builder("srl_bot_continuous", lambda **kwargs: srl_bot_continuous.SRL_Bot_Agent(**kwargs))
        runner.algo_factory.register_builder("srl_continuous", lambda **kwargs: srl_continuous.SRLAgent(**kwargs))
        runner.algo_factory.register_builder("srl_continuous_marl", lambda **kwargs: srl_continuous_marl.SRL_MultiAgent(**kwargs))
        runner.player_factory.register_builder("srl_continuous", lambda **kwargs: srl_players.SRLPlayerContinuous(**kwargs))
        runner.player_factory.register_builder("srl_continuous_marl", lambda **kwargs: srl_players.SRLPlayerContinuous(**kwargs))
        runner.player_factory.register_builder("srl_bot_continuous", lambda **kwargs: srl_players.SRL_Bot_PlayerContinuous(**kwargs))
        model_builder.register_model("continuous_srl", lambda network, **kwargs: srl_models.ModelSRLContinuous(network))
        model_builder.register_network("amp_humanoid", lambda **kwargs: srl_network_builder.HumanoidBuilder())
        model_builder.register_network("srl", lambda **kwargs: srl_network_builder.SRLBuilder())
        return runner

    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    if not cfg.test:
        experiment_dir = os.path.join(
            "runs",
            cfg.train.params.config.name + "_{date:%d-%H-%M-%S}".format(date=datetime.now()),
        )
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    runner.run({
        "train": not cfg.test,
        "play": cfg.test,
        "checkpoint": cfg.checkpoint,
        "sigma": cfg.sigma if cfg.sigma != "" else None,
    })


if __name__ == "__main__":
    launch_rlg_hydra()
