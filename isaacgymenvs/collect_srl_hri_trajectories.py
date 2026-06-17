import os
from datetime import datetime

import hydra
from omegaconf import DictConfig


def preprocess_train_config(cfg, config_dict):
    train_cfg = config_dict["params"]["config"]
    train_cfg["device"] = cfg.rl_device
    train_cfg["population_based_training"] = cfg.pbt.enabled
    train_cfg["pbt_idx"] = cfg.pbt.policy_idx if cfg.pbt.enabled else None
    train_cfg["full_experiment_name"] = cfg.get("full_experiment_name")

    try:
        model_size_multiplier = config_dict["params"]["network"]["mlp"]["model_size_multiplier"]
        if model_size_multiplier != 1:
            units = config_dict["params"]["network"]["mlp"]["units"]
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
    except KeyError:
        pass

    return config_dict


def _to_cpu_tensor(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    return torch.as_tensor(x).detach().cpu().clone()


def _maybe_get_attr(obj, name):
    return getattr(obj, name, None) if hasattr(obj, name) else None


def _maybe_call(obj, name, default=None):
    fn = getattr(obj, name, None)
    if fn is None:
        return default
    return fn()


def _collect_env_snapshot(env, include_heavy_state=False):
    snapshot = {
        "obs_full": _to_cpu_tensor(_maybe_get_attr(env, "obs_buf")),
        "srl_obs": _to_cpu_tensor(_maybe_get_attr(env, "srl_obs_buf")),
        "srl_full_obs": _to_cpu_tensor(_maybe_get_attr(env, "srl_full_obs_buf")),
        "teacher_srl_obs": _to_cpu_tensor(_maybe_get_attr(env, "teacher_srl_obs_buf")),
        "srl_priv_extra_obs": _to_cpu_tensor(_maybe_get_attr(env, "srl_priv_extra_obs_buf")),
        "root_states": _to_cpu_tensor(_maybe_get_attr(env, "_root_states")),
        "srl_root_states": _to_cpu_tensor(_maybe_get_attr(env, "srl_root_states")),
        "dof_pos": _to_cpu_tensor(_maybe_get_attr(env, "_dof_pos")),
        "dof_vel": _to_cpu_tensor(_maybe_get_attr(env, "_dof_vel")),
        "target_vel_x": _to_cpu_tensor(_maybe_get_attr(env, "target_vel_x")),
        "target_ang_vel_z": _to_cpu_tensor(_maybe_get_attr(env, "target_ang_vel_z")),
        "target_pelvis_height": _to_cpu_tensor(_maybe_get_attr(env, "target_pelvis_height")),
        "target_yaw": _to_cpu_tensor(_maybe_get_attr(env, "target_yaw")),
        "progress_buf": _to_cpu_tensor(_maybe_get_attr(env, "progress_buf")),
        "reset_buf": _to_cpu_tensor(_maybe_get_attr(env, "reset_buf")),
        "terminate_buf": _to_cpu_tensor(_maybe_get_attr(env, "_terminate_buf")),
        "srl_reward": _to_cpu_tensor(_maybe_get_attr(env, "srl_rew_buf")),
    }

    if hasattr(env, "_virtual_load_cell_from_dof") and hasattr(env, "_dof_pos") and hasattr(env, "_dof_vel"):
        snapshot["virtual_load_cell"] = _to_cpu_tensor(env._virtual_load_cell_from_dof(env._dof_pos, env._dof_vel))

    if hasattr(env, "_rigid_body_pos") and hasattr(env, "_srl_end_ids"):
        snapshot["srl_end_pos"] = _to_cpu_tensor(env._rigid_body_pos[:, env._srl_end_ids, :])
    if hasattr(env, "_rigid_body_vel") and hasattr(env, "_srl_end_ids"):
        snapshot["srl_end_vel"] = _to_cpu_tensor(env._rigid_body_vel[:, env._srl_end_ids, :])

    if include_heavy_state:
        snapshot["rigid_body_pos"] = _to_cpu_tensor(_maybe_get_attr(env, "_rigid_body_pos"))
        snapshot["rigid_body_vel"] = _to_cpu_tensor(_maybe_get_attr(env, "_rigid_body_vel"))
        snapshot["dof_force"] = _to_cpu_tensor(_maybe_get_attr(env, "dof_force_tensor"))

    return {k: v for k, v in snapshot.items() if v is not None}


def _stack_chunk(chunk_lists):
    stacked = {}
    for key, values in chunk_lists.items():
        if len(values) == 0:
            continue
        try:
            stacked[key] = torch.stack(values, dim=0)
        except RuntimeError:
            stacked[key] = values
    return stacked


def _append_snapshot(chunk_lists, snapshot):
    for key, value in snapshot.items():
        chunk_lists.setdefault(key, []).append(value)


def _get_action_with_srl_teacher(player, obs_dict, deterministic=True):
    from rl_games.algos_torch.players import rescale_actions

    obs = obs_dict["obs"]
    if player.has_batch_dimension is False and getattr(obs, "dim", lambda: 0)() == 1:
        from rl_games.common.tr_helpers import unsqueeze_obs
        obs = unsqueeze_obs(obs)

    player.model.eval()
    player.model_srl.eval()

    processed_obs = player._preproc_obs(obs)
    expected_obs_dim = player.obs_num_humanoid + player.obs_num_srl

    while processed_obs.dim() > 2 and processed_obs.shape[0] == 1:
        processed_obs = processed_obs.squeeze(0)

    if processed_obs.dim() == 2 and processed_obs.shape[1] != expected_obs_dim and processed_obs.shape[0] == expected_obs_dim:
        processed_obs = processed_obs.transpose(0, 1).contiguous()

    if processed_obs.dim() != 2 or processed_obs.shape[1] != expected_obs_dim:
        raise RuntimeError(
            f"Unexpected obs shape: got {tuple(processed_obs.shape)}, expected [num_envs, {expected_obs_dim}] "
            f"({player.obs_num_humanoid} humanoid + {player.obs_num_srl} srl)."
        )

    humanoid_obs = processed_obs[:, :player.obs_num_humanoid]
    srl_obs = processed_obs[:, -player.obs_num_srl:]
    priv_srl_obs = processed_obs[:, -player.priv_obs_num_srl:]

    humanoid_input = {
        "is_train": False,
        "prev_actions": None,
        "obs": humanoid_obs,
        "rnn_states": player.states,
    }
    srl_input = {
        "is_train": False,
        "prev_actions": None,
        "obs": srl_obs,
        "rnn_states": player.states,
        "priv_obs": priv_srl_obs,
    }

    with torch.no_grad():
        humanoid_res = player.model(humanoid_input)
        srl_res = player.model_srl(srl_input)

    mu_humanoid = humanoid_res["mus"]
    mu_srl = srl_res["mus"]
    sampled_humanoid = humanoid_res["actions"]
    sampled_srl = srl_res["actions"]

    raw_mu_full = torch.cat((mu_humanoid, mu_srl), dim=-1)
    raw_sampled_full = torch.cat((sampled_humanoid, sampled_srl), dim=-1)
    raw_current_action = raw_mu_full if deterministic else raw_sampled_full

    player.states = humanoid_res["rnn_states"]

    if player.has_batch_dimension is False:
        raw_current_action = torch.squeeze(raw_current_action.detach())
    raw_current_action = raw_current_action.detach()

    if player.clip_actions:
        action_full = rescale_actions(
            player.actions_low,
            player.actions_high,
            torch.clamp(raw_current_action, -1.0, 1.0),
        )
    else:
        action_full = raw_current_action

    if action_full.dim() == 1:
        action_full = action_full.unsqueeze(0)

    srl_action_size = player.env.get_srl_action_size()
    action_srl_applied = action_full[:, -srl_action_size:]

    return {
        "action_full": action_full,
        "action_srl_applied": action_srl_applied,
        "raw_mu_full": raw_mu_full,
        "raw_mu_srl": mu_srl,
        "raw_sampled_srl": sampled_srl,
        "srl_obs_from_player": srl_obs,
        "priv_srl_obs_from_player": priv_srl_obs,
    }


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def collect_srl_hri_trajectories(cfg: DictConfig):
    import gym
    import isaacgym  # noqa: F401
    global torch
    import torch
    import isaacgymenvs
    from hydra.utils import to_absolute_path
    from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
    from isaacgymenvs.tasks import isaacgym_task_map
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.rlgames_utils import (
        ComplexObsRLGPUEnv,
        MultiObserver,
        RLGPUAlgoObserver,
        RLGPUEnv,
    )
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.algos_torch import model_builder
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from isaacgymenvs.learning import amp_continuous, amp_models, amp_network_builder, amp_players
    from isaacgymenvs.learning.SRLEvo import (
        srl_bot_continuous,
        srl_continuous,
        srl_continuous_marl,
        srl_models,
        srl_players,
    )
    from isaacgymenvs.learning.SRLEvo import srl_network_builder

    if not cfg.checkpoint:
        raise ValueError("A trained checkpoint is required. Pass checkpoint=...")

    if cfg.pbt.enabled:
        initial_pbt_check(cfg)

    cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    collect_steps = int(cfg.get("collect_steps", 20000))
    chunk_steps = int(cfg.get("collect_chunk_steps", 1000))
    output_dir = to_absolute_path(str(cfg.get("collect_output_dir", "datasets/srl_hri")))
    deterministic = bool(cfg.get("collect_deterministic", True))
    include_heavy_state = bool(cfg.get("collect_heavy_state", False))

    os.makedirs(output_dir, exist_ok=True)

    cfg_dict = omegaconf_to_dict(cfg)
    print("cfg_dict:")
    print_dict(cfg_dict)
    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=0)

    run_name = f"collect_srl_hri_{datetime.now():%Y-%m-%d_%H-%M-%S}"

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

    env_configurations.register("rlgpu", {
        "vecenv_type": "RLGPU",
        "env_creator": lambda **kwargs: create_isaacgym_env(**kwargs),
    })

    ige_env_cls = isaacgym_task_map[cfg.task_name]
    print("task_name:", cfg.task_name)
    print("env_class:", ige_env_cls)
    print("env_module:", ige_env_cls.__module__)

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
        vecenv.register(
            "RLGPU",
            lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs),
        )
    else:
        vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = preprocess_train_config(cfg, omegaconf_to_dict(cfg.train))

    observers = [RLGPUAlgoObserver()]
    if cfg.pbt.enabled:
        observers.append(PbtAlgoObserver(cfg))
    if cfg.wandb_activate:
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

    player = runner.create_player()
    player.restore(cfg.checkpoint)
    player.model.eval()
    player.model_srl.eval()

    env = player.env
    print("checkpoint:", cfg.checkpoint)
    print("num_envs:", env.num_envs)
    print("srl_obs_size:", env.get_srl_obs_size())
    print("srl_full_obs_size:", _maybe_call(env, "get_srl_full_obs_size", default=None))
    print("teacher_srl_obs_size:", _maybe_call(env, "get_teacher_srl_obs_size", default=None))
    print("srl_action_size:", env.get_srl_action_size())
    print("collect_steps:", collect_steps)
    print("chunk_steps:", chunk_steps)
    print("output_dir:", os.path.abspath(output_dir))

    obs_dict = player.env_reset(env)
    player.get_batch_size(obs_dict["obs"], 1)
    if player.is_rnn:
        player.init_rnn()

    chunk_id = 0
    chunk_lists = {}
    global_step = 0

    with torch.no_grad():
        while global_step < collect_steps:
            obs_dict, done_env_ids = player._env_reset_done()
            action_pack = _get_action_with_srl_teacher(player, obs_dict, deterministic=deterministic)

            pre_snapshot = _collect_env_snapshot(env, include_heavy_state=include_heavy_state)
            _append_snapshot(chunk_lists, pre_snapshot)
            _append_snapshot(chunk_lists, {
                "action_full": _to_cpu_tensor(action_pack["action_full"]),
                "action_srl_applied": _to_cpu_tensor(action_pack["action_srl_applied"]),
                "raw_mu_full": _to_cpu_tensor(action_pack["raw_mu_full"]),
                "raw_mu_srl": _to_cpu_tensor(action_pack["raw_mu_srl"]),
                "raw_sampled_srl": _to_cpu_tensor(action_pack["raw_sampled_srl"]),
                "srl_obs_from_player": _to_cpu_tensor(action_pack["srl_obs_from_player"]),
                "priv_srl_obs_from_player": _to_cpu_tensor(action_pack["priv_srl_obs_from_player"]),
            })

            obs_dict, reward, done, info = player.env_step(env, action_pack["action_full"])

            _append_snapshot(chunk_lists, {
                "reward": _to_cpu_tensor(reward),
                "done": _to_cpu_tensor(done),
                "done_env_ids": _to_cpu_tensor(done_env_ids),
            })

            global_step += 1

            if (global_step % chunk_steps == 0) or (global_step >= collect_steps):
                chunk = _stack_chunk(chunk_lists)
                chunk["metadata"] = {
                    "task_name": str(cfg.task_name),
                    "checkpoint": str(cfg.checkpoint),
                    "num_envs": int(env.num_envs),
                    "start_step": int(global_step - len(next(iter(chunk_lists.values())))),
                    "end_step": int(global_step),
                    "deterministic": deterministic,
                    "srl_obs_size": int(env.get_srl_obs_size()),
                    "srl_full_obs_size": _maybe_call(env, "get_srl_full_obs_size", default=None),
                    "teacher_srl_obs_size": _maybe_call(env, "get_teacher_srl_obs_size", default=None),
                    "srl_action_size": int(env.get_srl_action_size()),
                }
                save_path = os.path.join(output_dir, f"srl_hri_traj_chunk_{chunk_id:05d}.pt")
                torch.save(chunk, save_path)
                print(f"[collect] saved {save_path} at global_step={global_step}")
                chunk_id += 1
                chunk_lists = {}

    print("[collect] done")


if __name__ == "__main__":
    collect_srl_hri_trajectories()
