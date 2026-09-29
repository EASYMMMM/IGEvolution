"""Collect Isaac Gym traces for cross-engine concurrent-estimator diagnostics.

Run from the ``IGEvolution-main/isaacgymenvs`` directory. The actor receives
clean simulator truth for the four explicit channels, while the concurrent
estimator runs in shadow mode and is recorded without controlling the robot.
"""

import json
import os
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from SRL_Evo_train import preprocess_train_config


def _option(cfg, name, default):
    return cfg.get("diagnostics", {}).get(name, default)


def _remove_randomization_schedules(node):
    if not isinstance(node, (dict, DictConfig)):
        return 0
    removed = 0
    for key in list(node.keys()):
        if key in ("schedule", "schedule_steps"):
            del node[key]
            removed += 1
        else:
            removed += _remove_randomization_schedules(node[key])
    return removed


def _configure(cfg):
    OmegaConf.set_struct(cfg, False)
    num_envs = int(_option(cfg, "num_envs", 64))
    collect_steps = int(_option(cfg, "collect_steps", 5000))
    startup_steps = int(_option(cfg, "startup_steps", 300))
    if num_envs <= 0 or collect_steps <= 0 or startup_steps < 0:
        raise ValueError("num_envs/collect_steps must be positive and startup_steps nonnegative")
    if not cfg.checkpoint:
        raise ValueError("Supply checkpoint=... for the concurrent policy")
    cfg.checkpoint = to_absolute_path(str(cfg.checkpoint))
    if not os.path.isfile(cfg.checkpoint):
        raise FileNotFoundError(cfg.checkpoint)
    cfg.test = True
    cfg.headless = True
    cfg.force_render = False
    cfg.capture_video = False
    cfg.wandb_activate = False
    cfg.num_envs = num_envs
    cfg.task.env.numEnvs = num_envs
    cfg.train.params.config.concurrent_eval_use_estimator = False
    if bool(cfg.task.task.randomize) and bool(
        _option(cfg, "full_strength_dr", True)
    ):
        removed = _remove_randomization_schedules(
            cfg.task.task.randomization_params
        )
        print(f"Removed {removed} DR schedule fields for full-strength collection")
    output = Path(
        to_absolute_path(
            str(_option(cfg, "output", "estimator_diagnostics/data/isaacgym_trace.npz"))
        )
    )
    return num_envs, collect_steps, startup_steps, output


def _build_runner(cfg):
    import isaacgym  # noqa: F401
    import gym  # noqa: F401
    import isaacgymenvs
    from isaacgymenvs.learning.SRLEvo import (
        srl_bot_concurrent,
        srl_bot_continuous,
        srl_models,
        srl_network_builder,
        srl_players,
    )
    from isaacgymenvs.tasks import isaacgym_task_map
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    from isaacgymenvs.utils.rlgames_utils import (
        ComplexObsRLGPUEnv,
        MultiObserver,
        RLGPUAlgoObserver,
        RLGPUEnv,
    )
    from rl_games.algos_torch import model_builder
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    def create_isaacgym_env(**kwargs):
        return isaacgymenvs.make(
            cfg.seed, cfg.task_name, cfg.task.env.numEnvs,
            cfg.sim_device, cfg.rl_device, cfg.graphics_device_id,
            cfg.headless, cfg.multi_gpu, False, False, cfg, **kwargs
        )

    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "RLGPU", "env_creator": lambda **kwargs: create_isaacgym_env(**kwargs)},
    )
    task_cls = isaacgym_task_map[cfg.task_name]
    dict_obs = bool(hasattr(task_cls, "dict_obs_cls") and task_cls.dict_obs_cls)
    if dict_obs:
        obs_spec = {}
        actor_cfg = cfg.train.params.network
        obs_spec["obs"] = {
            "names": list(actor_cfg.inputs.keys()),
            "concat": actor_cfg.name != "complex_net",
            "space_name": "observation_space",
        }
        if "central_value_config" in cfg.train.params.config:
            critic_cfg = cfg.train.params.config.central_value_config.network
            obs_spec["states"] = {
                "names": list(critic_cfg.inputs.keys()),
                "concat": critic_cfg.name != "complex_net",
                "space_name": "state_space",
            }
        vecenv.register(
            "RLGPU",
            lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(
                config_name, num_actors, obs_spec, **kwargs
            ),
        )
    else:
        vecenv.register(
            "RLGPU",
            lambda config_name, num_actors, **kwargs: RLGPUEnv(
                config_name, num_actors, **kwargs
            ),
        )

    runner = Runner(MultiObserver([RLGPUAlgoObserver()]))
    runner.algo_factory.register_builder(
        "srl_bot_continuous", lambda **kwargs: srl_bot_continuous.SRL_Bot_Agent(**kwargs)
    )
    runner.algo_factory.register_builder(
        "srl_bot_concurrent", lambda **kwargs: srl_bot_concurrent.SRL_Bot_Concurrent_Agent(**kwargs)
    )
    runner.player_factory.register_builder(
        "srl_bot_continuous", lambda **kwargs: srl_players.SRL_Bot_PlayerContinuous(**kwargs)
    )
    runner.player_factory.register_builder(
        "srl_bot_concurrent", lambda **kwargs: srl_bot_concurrent.SRL_Bot_Concurrent_Player(**kwargs)
    )
    model_builder.register_model(
        "continuous_srl", lambda network, **kwargs: srl_models.ModelSRLContinuous(network)
    )
    model_builder.register_network("srl", lambda **kwargs: srl_network_builder.SRLBuilder())
    train_config = preprocess_train_config(cfg, omegaconf_to_dict(cfg.train))
    runner.load(train_config)
    runner.reset()
    return runner


def _unwrap_task(env):
    current = env
    for _ in range(8):
        if hasattr(current, "full_obs_buffer") and hasattr(current, "phase_buf"):
            return current
        if not hasattr(current, "env"):
            break
        current = current.env
    raise RuntimeError("Could not locate SRL task under RL-Games wrapper")


def _quat_rotate_inverse(q, v):
    import torch

    q_vec = q[:, :3]
    q_w = q[:, 3:4]
    return (
        v * (2.0 * q_w.square() - 1.0)
        - 2.0 * q_w * torch.cross(q_vec, v, dim=-1)
        + 2.0 * q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True)
    )


def _to_numpy(value, dtype=None):
    result = value.detach().cpu().numpy()
    return result.astype(dtype, copy=True) if dtype is not None else result.copy()


def _append(chunks, values):
    for key, value in values.items():
        chunks[key].append(_to_numpy(value))


def _install_fixed_command(task, cfg):
    if not bool(_option(cfg, "fixed_command", True)):
        return
    vx = float(_option(cfg, "target_vx", 1.0))
    wz = float(_option(cfg, "target_wz", 0.0))
    height = float(_option(cfg, "target_height", 1.0))
    yaw = float(_option(cfg, "target_yaw", 0.0))

    def set_fixed_command():
        task.target_vel_x.fill_(vx)
        task.target_ang_vel_z.fill_(wz)
        task.target_pelvis_height.fill_(height)
        task.target_yaw.fill_(yaw)

    task.set_task_target = set_fixed_command
    set_fixed_command()


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def collect(cfg: DictConfig):
    import isaacgym  # noqa: F401
    import torch
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed

    num_envs, collect_steps, startup_steps, output = _configure(cfg)
    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=0)
    runner = _build_runner(cfg)
    player = runner.create_player()
    player.restore(cfg.checkpoint)
    player.model.eval()
    player.concurrent_estimator.eval()
    env = player.env
    task = _unwrap_task(env)
    _install_fixed_command(task, cfg)

    body_properties = task.gym.get_actor_rigid_body_properties(
        task.envs[0], task.humanoid_handles[0]
    )
    masses = torch.as_tensor(
        [prop.mass for prop in body_properties],
        device=task.device, dtype=torch.float32,
    )
    total_mass = masses.sum().clamp_min(1e-8)
    feet = task.feet_indices.long()
    contact_threshold = float(_option(cfg, "contact_force_threshold", 1.0))
    gait_period = int(getattr(task, "gait_period", 54))

    obs_dict = player.env_reset(env)
    player.get_batch_size(obs_dict["obs"], 1)
    if player.is_rnn:
        player.init_rnn()
    episode_serial = torch.arange(num_envs, device=task.device, dtype=torch.long)
    next_episode = num_envs
    step_in_episode = torch.zeros(num_envs, device=task.device, dtype=torch.long)
    chunks = {key: [] for key in (
        "episode", "env_id", "step", "terminal", "input_26",
        "input_normalized_26", "truth_4", "estimate_4", "phase",
        "contact_state", "stage", "velocity_free_world", "velocity_free_local",
        "velocity_base_world", "velocity_base_local", "velocity_com_world",
        "velocity_com_local", "height_free", "height_base", "height_com",
    )}

    print("Collecting Isaac Gym trace with GT supplied to actor and EST in shadow mode")
    print(f"  checkpoint={cfg.checkpoint}")
    print(f"  envs={num_envs} vector_steps={collect_steps} DR={bool(cfg.task.task.randomize)}")
    deterministic = bool(_option(cfg, "deterministic", True))
    env_ids = torch.arange(num_envs, device=task.device, dtype=torch.long)
    with torch.no_grad():
        for vector_step in range(collect_steps):
            raw_obs, _ = player._env_reset_done()
            raw_actor_obs = raw_obs["obs"]
            estimated_actor_obs = player._estimated_actor_observation(raw_actor_obs)
            estimate = estimated_actor_obs[:, -4:]
            # RL-Games does not expose central-critic ``states`` in every
            # player/test configuration. The task's clean full observation is
            # the source used to construct those states during training, and
            # is therefore the authoritative estimator target here as well.
            if tuple(task.full_obs_buffer.shape[1:]) != (5, 30):
                raise RuntimeError(
                    "Expected task.full_obs_buffer [num_envs, 5, 30], got {}"
                    .format(tuple(task.full_obs_buffer.shape))
                )
            truth = task.full_obs_buffer[:, 0, :4].clone()
            if "states" in raw_obs:
                critic_truth = raw_obs["states"][:, :4]
                truth_difference = (critic_truth - truth).abs().max().item()
                if truth_difference > 1e-5:
                    raise RuntimeError(
                        "Task truth and central-critic state differ: {:.3e}"
                        .format(truth_difference)
                    )
            deployable = raw_actor_obs[:, :133]
            actor_obs = dict(raw_obs)
            actor_obs["obs"] = torch.cat((deployable, truth), dim=-1)

            current_input = deployable[:, :26]
            normalized_input = (
                current_input - player.concurrent_estimator.input_mean
            ) / player.concurrent_estimator.input_std()
            root_rotation = task.root_states[:, 3:7]
            velocity_free_world = task.root_states[:, 7:10]
            velocity_base_world = task.srl_root_states[:, 7:10]
            velocity_com_world = (
                task._rigid_body_vel * masses.view(1, -1, 1)
            ).sum(dim=1) / total_mass
            position_com_world = (
                task._rigid_body_pos * masses.view(1, -1, 1)
            ).sum(dim=1) / total_mass
            velocity_free_local = _quat_rotate_inverse(root_rotation, velocity_free_world)
            velocity_base_local = _quat_rotate_inverse(root_rotation, velocity_base_world)
            velocity_com_local = _quat_rotate_inverse(root_rotation, velocity_com_world)
            foot_contact = torch.linalg.vector_norm(
                task.contact_forces[:, feet, :], dim=-1
            ) >= contact_threshold
            contact_state = foot_contact[:, 0].to(torch.int8) + 2 * foot_contact[:, 1].to(torch.int8)

            values = {
                "episode": episode_serial.clone(),
                "env_id": env_ids.clone(),
                "step": step_in_episode.clone(),
                "input_26": current_input.clone(),
                "input_normalized_26": normalized_input.clone(),
                "truth_4": truth.clone(),
                "estimate_4": estimate.clone(),
                "phase": task.phase_buf.remainder(gait_period).to(torch.int16).clone(),
                "contact_state": contact_state.clone(),
                "stage": (step_in_episode >= startup_steps).to(torch.int8).clone(),
                "velocity_free_world": velocity_free_world.clone(),
                "velocity_free_local": velocity_free_local.clone(),
                "velocity_base_world": velocity_base_world.clone(),
                "velocity_base_local": velocity_base_local.clone(),
                "velocity_com_world": velocity_com_world.clone(),
                "velocity_com_local": velocity_com_local.clone(),
                "height_free": task.root_states[:, 2].clone(),
                "height_base": task.srl_root_states[:, 2].clone(),
                "height_com": position_com_world[:, 2].clone(),
            }
            action = player.get_action(actor_obs, deterministic)
            _, _, done, _ = player.env_step(env, action)
            terminal = done.bool()
            values["terminal"] = terminal
            _append(chunks, values)
            step_in_episode += 1
            done_ids = terminal.nonzero(as_tuple=False).reshape(-1)
            if done_ids.numel() > 0:
                count = int(done_ids.numel())
                episode_serial[done_ids] = torch.arange(
                    next_episode, next_episode + count,
                    device=task.device, dtype=torch.long,
                )
                next_episode += count
                step_in_episode[done_ids] = 0
            if (vector_step + 1) % 100 == 0:
                error = estimate - truth
                rmse = torch.sqrt(torch.mean(error.square(), dim=0))
                print(
                    f"step={vector_step + 1}/{collect_steps} "
                    f"RMSE={_to_numpy(rmse)} completed={next_episode - num_envs}"
                )

    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    metadata = {
        "format_version": 1,
        "engine": "isaacgym",
        "checkpoint": cfg.checkpoint,
        "asset": str(cfg.task.env.asset.assetFileName),
        "control_dt": float(getattr(task, "control_dt", 0.015)),
        "gait_period": gait_period,
        "startup_steps": startup_steps,
        "actor_input": "ground_truth",
        "estimator_mode": "shadow",
        "domain_randomization": bool(cfg.task.task.randomize),
        "velocity_perturbation": bool(cfg.task.task.vel_pertubation),
        "num_envs": num_envs,
        "vector_steps": collect_steps,
        "target_names": ["root_height", "local_vx", "local_vy", "local_vz"],
        "estimator_input_mean": _to_numpy(player.concurrent_estimator.input_mean).tolist(),
        "estimator_input_std": _to_numpy(player.concurrent_estimator.input_std()).tolist(),
        "estimator_target_mean": _to_numpy(player.concurrent_estimator.target_mean).tolist(),
        "estimator_target_std": _to_numpy(player.concurrent_estimator.target_std()).tolist(),
    }
    if output.suffix.lower() != ".npz":
        output = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(output, **arrays)
    saved = output
    print(f"Saved {len(arrays['step'])} samples to {saved}")


if __name__ == "__main__":
    collect()
