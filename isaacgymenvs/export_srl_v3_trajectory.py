"""Export a deterministic SRL v3 trajectory from Isaac Gym.

Run this script from the ``IGEvolution/isaacgymenvs`` directory.  It uses the
same RL-Games player and SRL task as evaluation, but fixes the task command and
records the actor action, PD targets, measured joint state, and joint torque.
"""

import csv
import os
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from SRL_Evo_train import preprocess_train_config


DEFAULT_CHECKPOINT = (
    "runs/SRL_Real_Bot_v2_s4_24-15-30-46/nn/SRL_Real_Bot_v2_s4.pth"
)
JOINT_NAMES = (
    "left_hip_x",
    "left_hip_y",
    "left_knee",
    "right_hip_x",
    "right_hip_y",
    "right_knee",
)
VECTOR_FIELDS = (
    "raw_action",
    "raw_target_pos",
    "target_pos",
    "measured_q",
    "measured_qd",
    "commanded_torque",
    "applied_torque",
    "task_command",
)


def _export_option(cfg, name, default):
    export_cfg = cfg.get("export", {})
    return export_cfg.get(name, default)


def _configure_export(cfg):
    OmegaConf.set_struct(cfg, False)

    warmup_steps = int(_export_option(cfg, "warmup_steps", 1000))
    cycles = int(_export_option(cfg, "cycles", 100))
    gait_period = int(round(float(cfg.task.env.gait_period)))
    required_steps = warmup_steps + cycles * gait_period

    if warmup_steps < 0:
        raise ValueError("export.warmup_steps must be >= 0")
    if cycles <= 0:
        raise ValueError("export.cycles must be > 0")

    cfg.test = True
    cfg.headless = True
    cfg.force_render = False
    cfg.capture_video = False
    cfg.wandb_activate = False
    cfg.num_envs = 1
    cfg.task.env.numEnvs = 1
    cfg.task.env.episodeLength = max(
        int(cfg.task.env.episodeLength), required_steps + 100
    )
    cfg.task.task.randomize = False
    cfg.task.task.vel_pertubation = False

    if not cfg.checkpoint:
        cfg.checkpoint = DEFAULT_CHECKPOINT
    cfg.checkpoint = to_absolute_path(str(cfg.checkpoint))
    if not os.path.isfile(cfg.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint}")

    return warmup_steps, cycles, gait_period, required_steps


def _build_runner(cfg, run_name):
    # Imports stay inside the Hydra entry point so Isaac Gym is imported before
    # torch-dependent simulator modules, matching SRL_Evo_train.py.
    import isaacgym  # noqa: F401
    import gym
    import isaacgymenvs
    from isaacgymenvs.learning.SRLEvo import (
        srl_bot_continuous,
        srl_models,
        srl_network_builder,
        srl_players,
    )
    from isaacgymenvs.tasks import isaacgym_task_map
    from isaacgymenvs.utils.rlgames_utils import (
        ComplexObsRLGPUEnv,
        MultiObserver,
        RLGPUAlgoObserver,
        RLGPUEnv,
    )
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    from rl_games.algos_torch import model_builder
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    def create_isaacgym_env(**kwargs):
        return isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            False,
            False,
            cfg,
            **kwargs,
        )

    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: create_isaacgym_env(**kwargs),
        },
    )

    task_cls = isaacgym_task_map[cfg.task_name]
    dict_obs = bool(
        hasattr(task_cls, "dict_obs_cls") and task_cls.dict_obs_cls
    )
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
        "srl_bot_continuous",
        lambda **kwargs: srl_bot_continuous.SRL_Bot_Agent(**kwargs),
    )
    runner.player_factory.register_builder(
        "srl_bot_continuous",
        lambda **kwargs: srl_players.SRL_Bot_PlayerContinuous(**kwargs),
    )
    model_builder.register_model(
        "continuous_srl",
        lambda network, **kwargs: srl_models.ModelSRLContinuous(network),
    )
    model_builder.register_network(
        "srl", lambda **kwargs: srl_network_builder.SRLBuilder()
    )

    train_config = preprocess_train_config(
        cfg, omegaconf_to_dict(cfg.train)
    )
    runner.load(train_config)
    runner.reset()
    return runner


def _fixed_command_hook(task, vx, wz, height, yaw):
    def set_fixed_command():
        task.target_vel_x.fill_(vx)
        task.target_ang_vel_z.fill_(wz)
        task.target_pelvis_height.fill_(height)
        task.target_yaw.fill_(yaw)

    task.set_task_target = set_fixed_command
    set_fixed_command()
    return set_fixed_command


def _cpu_row(tensor):
    return tensor[0].detach().cpu().numpy().astype(np.float32, copy=True)


def _unwrap_srl_task(env):
    """Accept both direct SRL tasks and RL-Games environment wrappers."""
    current = env
    visited = set()
    for _ in range(8):
        if hasattr(current, "raw_pd_targets") and hasattr(current, "phase_buf"):
            return current
        object_id = id(current)
        if object_id in visited or not hasattr(current, "env"):
            break
        visited.add(object_id)
        current = current.env
    raise RuntimeError(
        "Could not locate the SRL task behind player.env. "
        f"Top-level environment type: {type(env).__name__}"
    )


def _phase_aggregates(arrays, gait_period):
    output = dict(arrays)
    output["phase_index"] = np.arange(gait_period, dtype=np.int64)
    phase_ids = arrays["phase"] % gait_period
    for key in VECTOR_FIELDS:
        means = []
        medians = []
        for phase in range(gait_period):
            values = arrays[key][phase_ids == phase]
            if values.shape[0] == 0:
                raise RuntimeError(f"No recorded samples for phase={phase}")
            means.append(np.mean(values, axis=0))
            medians.append(np.median(values, axis=0))
        output[f"{key}_phase_mean"] = np.asarray(means, dtype=np.float32)
        output[f"{key}_phase_median"] = np.asarray(
            medians, dtype=np.float32
        )
    return output


def _write_csv(path, arrays, control_dt):
    command_names = ("target_vx", "target_wz", "target_height")
    headers = ["step", "time_s", "phase"]
    for field in VECTOR_FIELDS:
        names = command_names if field == "task_command" else JOINT_NAMES
        headers.extend(f"{field}_{name}" for name in names)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index in range(arrays["step"].shape[0]):
            row = [
                int(arrays["step"][index]),
                float(arrays["step"][index]) * control_dt,
                int(arrays["phase"][index]),
            ]
            for field in VECTOR_FIELDS:
                row.extend(arrays[field][index].tolist())
            writer.writerow(row)


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def export_trajectory(cfg: DictConfig):
    import isaacgym  # noqa: F401
    import torch
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed

    warmup_steps, cycles, gait_period, required_steps = _configure_export(cfg)
    set_np_formatting()
    cfg.seed = set_seed(
        cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=0
    )

    runner = _build_runner(cfg, "srl_v3_trajectory_export")
    player = runner.create_player()
    player.restore(cfg.checkpoint)
    player.model.eval()

    env = player.env
    task = _unwrap_srl_task(env)
    if task.num_envs != 1:
        raise RuntimeError(f"Exporter requires one environment, got {task.num_envs}")

    target_vx = float(_export_option(cfg, "target_vx", 0.0))
    target_wz = float(_export_option(cfg, "target_wz", 0.0))
    target_height = float(_export_option(cfg, "target_height", 1.0))
    target_yaw = float(_export_option(cfg, "target_yaw", 0.0))
    initial_phase = int(_export_option(cfg, "initial_phase", 0)) % gait_period
    save_csv = bool(_export_option(cfg, "save_csv", True))

    fixed_command = _fixed_command_hook(
        task, target_vx, target_wz, target_height, target_yaw
    )
    task.phase_buf.fill_(initial_phase)
    task.progress_buf.zero_()
    task.obs_buffer.zero_()
    task.obs_mirrored_buffer.zero_()
    task.full_obs_buffer.zero_()
    fixed_command()
    task.compute_observations()

    obs_dict = player.env_reset(env)
    player.get_batch_size(obs_dict["obs"], 1)
    if player.is_rnn:
        player.init_rnn()

    samples = {
        "step": [],
        "phase": [],
        "raw_action": [],
        "raw_target_pos": [],
        "target_pos": [],
        "measured_q": [],
        "measured_qd": [],
        "commanded_torque": [],
        "applied_torque": [],
        "task_command": [],
    }

    print("Isaac Gym SRL v3 trajectory export")
    print(f"  checkpoint: {cfg.checkpoint}")
    print(f"  output samples: {cycles} x {gait_period} = {cycles * gait_period}")
    print(f"  warmup: {warmup_steps} control steps")
    print(
        "  command: "
        f"vx={target_vx:.3f}, wz={target_wz:.3f}, "
        f"height={target_height:.3f}, yaw={target_yaw:.3f}"
    )
    print(f"  initial phase: {initial_phase}")

    with torch.no_grad():
        for step in range(required_steps):
            phase = int(task.phase_buf[0].item()) % gait_period
            action = player.get_action(obs_dict, is_determenistic=True)
            obs_dict, _, done, _ = player.env_step(env, action)

            if bool(done[0].item()):
                raise RuntimeError(
                    "Environment terminated before export completed at "
                    f"step={step}. Try a stable command/checkpoint or reduce "
                    "export.warmup_steps/export.cycles."
                )

            if step >= warmup_steps:
                samples["step"].append(np.int64(step - warmup_steps))
                samples["phase"].append(np.int64(phase))
                samples["raw_action"].append(_cpu_row(task.raw_actions))
                samples["raw_target_pos"].append(_cpu_row(task.raw_pd_targets))
                samples["target_pos"].append(_cpu_row(task.filtered_pd_targets))
                samples["measured_q"].append(_cpu_row(task.dof_pos))
                samples["measured_qd"].append(_cpu_row(task.dof_vel))
                samples["commanded_torque"].append(_cpu_row(task.torques))
                samples["applied_torque"].append(_cpu_row(task.dof_force_tensor))
                samples["task_command"].append(
                    np.asarray(
                        [target_vx, target_wz, target_height],
                        dtype=np.float32,
                    )
                )

    arrays = {key: np.asarray(value) for key, value in samples.items()}
    output = _phase_aggregates(arrays, gait_period)
    output["control_dt"] = np.asarray(float(task.control_dt), dtype=np.float64)
    output["gait_period_steps"] = np.asarray(gait_period, dtype=np.int64)
    output["warmup_steps"] = np.asarray(warmup_steps, dtype=np.int64)
    output["recorded_cycles"] = np.asarray(cycles, dtype=np.int64)
    output["initial_phase"] = np.asarray(initial_phase, dtype=np.int64)
    output["checkpoint"] = np.asarray(str(cfg.checkpoint))
    output["source"] = np.asarray("isaacgym")

    output_path = Path(
        to_absolute_path(
            str(
                _export_option(
                    cfg,
                    "output",
                    "trajectories/inplace_isaacgym_v3_s4.npz",
                )
            )
        )
    )
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    print(f"Saved NPZ: {output_path}")

    if save_csv:
        csv_path = output_path.with_suffix(".csv")
        _write_csv(csv_path, arrays, float(task.control_dt))
        print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    export_trajectory()
