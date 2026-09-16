"""Parallel first-episode evaluation for the original Isaac Gym SRL policy.

Run this script from the ``isaacgymenvs`` directory. Each vectorized
environment contributes exactly one episode, so environments that fall early
cannot be reset and counted repeatedly. The run is reproducible from the
global Hydra seed; environment IDs correspond to independent random samples
drawn from that seeded Isaac Gym run.
"""

import csv
import json
import os
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from SRL_Evo_train import preprocess_train_config


def _eval_option(cfg, name, default):
    eval_cfg = cfg.get("eval", {})
    return eval_cfg.get(name, default)


def _remove_randomization_schedules(node):
    """Make every configured DR component use its final distribution."""
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


def _configure_evaluation(cfg):
    OmegaConf.set_struct(cfg, False)

    num_envs = int(_eval_option(cfg, "num_envs", 100))
    episode_steps = int(
        _eval_option(cfg, "episode_steps", cfg.task.env.episodeLength)
    )
    if num_envs <= 0:
        raise ValueError("eval.num_envs must be > 0")
    if episode_steps <= 0:
        raise ValueError("eval.episode_steps must be > 0")
    if not cfg.checkpoint:
        raise ValueError("A policy checkpoint must be supplied with checkpoint=...")

    cfg.test = True
    cfg.headless = True
    cfg.force_render = False
    cfg.capture_video = False
    cfg.wandb_activate = False
    cfg.num_envs = num_envs
    cfg.task.env.numEnvs = num_envs
    cfg.task.env.episodeLength = episode_steps

    randomize_override = _eval_option(cfg, "domain_randomization", None)
    if randomize_override is not None:
        cfg.task.task.randomize = bool(randomize_override)
    cfg.eval_full_strength_dr = bool(cfg.task.task.randomize)
    cfg.eval_removed_dr_schedule_fields = 0
    if cfg.eval_full_strength_dr:
        # The evaluator records only the first episode. Without this override,
        # physical DR with a linear schedule is sampled at simulation frame 0
        # with zero strength; setup-only mass randomization then stays nominal.
        cfg.eval_removed_dr_schedule_fields = _remove_randomization_schedules(
            cfg.task.task.randomization_params
        )
    perturb_override = _eval_option(cfg, "velocity_perturbation", None)
    if perturb_override is not None:
        cfg.task.task.vel_pertubation = bool(perturb_override)

    cfg.checkpoint = to_absolute_path(str(cfg.checkpoint))
    if not os.path.isfile(cfg.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint}")

    output_path = Path(
        to_absolute_path(
            str(
                _eval_option(
                    cfg,
                    "output",
                    "eval_reports/isaacgym_srl_real_bot_100.csv",
                )
            )
        )
    )
    if output_path.suffix.lower() != ".csv":
        output_path = output_path.with_suffix(".csv")
    return num_envs, episode_steps, output_path


def _build_runner(cfg):
    # Keep imports in this function so Isaac Gym is imported before simulator
    # modules that depend on its bundled PyTorch bindings.
    import isaacgym  # noqa: F401
    import gym  # noqa: F401
    import isaacgymenvs
    from isaacgymenvs.learning.SRLEvo import (
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

    train_config = preprocess_train_config(cfg, omegaconf_to_dict(cfg.train))
    runner.load(train_config)
    runner.reset()
    return runner


def _unwrap_srl_task(env):
    current = env
    visited = set()
    for _ in range(8):
        if hasattr(current, "phase_buf") and hasattr(current, "_terminate_buf"):
            return current
        object_id = id(current)
        if object_id in visited or not hasattr(current, "env"):
            break
        visited.add(object_id)
        current = current.env
    raise RuntimeError(
        "Could not locate the SRL task behind the RL-Games environment. "
        f"Top-level type: {type(env).__name__}"
    )


def _install_fixed_command(task, vx, wz, height, yaw):
    def set_fixed_command():
        task.target_vel_x.fill_(vx)
        task.target_ang_vel_z.fill_(wz)
        task.target_pelvis_height.fill_(height)
        task.target_yaw.fill_(yaw)

    task.set_task_target = set_fixed_command
    set_fixed_command()


def _to_cpu(value, dtype=None):
    array = value.detach().cpu().numpy()
    return array.astype(dtype, copy=True) if dtype is not None else array.copy()


def _write_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "env_id",
        "global_seed",
        "initial_phase",
        "success",
        "termination_reason",
        "episode_steps",
        "episode_time_s",
        "return",
        "min_root_height",
        "final_root_height",
        "final_target_vx",
        "final_target_wz",
        "final_target_height",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def evaluate(cfg: DictConfig):
    import isaacgym  # noqa: F401
    import torch
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed

    num_envs, episode_steps, output_path = _configure_evaluation(cfg)
    set_np_formatting()
    cfg.seed = set_seed(
        cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=0
    )

    runner = _build_runner(cfg)
    player = runner.create_player()
    player.restore(cfg.checkpoint)
    player.model.eval()

    env = player.env
    task = _unwrap_srl_task(env)
    if int(task.num_envs) != num_envs:
        raise RuntimeError(
            f"Expected {num_envs} environments, task created {task.num_envs}."
        )

    fixed_command = bool(_eval_option(cfg, "fixed_command", False))
    if fixed_command:
        _install_fixed_command(
            task,
            float(_eval_option(cfg, "target_vx", 0.0)),
            float(_eval_option(cfg, "target_wz", 0.0)),
            float(_eval_option(cfg, "target_height", 1.0)),
            float(_eval_option(cfg, "target_yaw", 0.0)),
        )

    # Match SRL_Bot_PlayerContinuous.run(): initialize the RL-Games wrapper,
    # establish its batch size, then perform the actual VecTask reset through
    # reset_done() -> reset_idx().
    obs_dict = player.env_reset(env)
    player.get_batch_size(obs_dict["obs"], 1)
    if player.is_rnn:
        player.init_rnn()
    obs_dict, reset_env_ids = player._env_reset_done()
    if int(reset_env_ids.numel()) != num_envs:
        raise RuntimeError(
            "Initial reset did not include every environment: "
            f"reset={int(reset_env_ids.numel())}, expected={num_envs}."
        )
    initial_phase = _to_cpu(task.phase_buf, np.int64)
    active = torch.ones(num_envs, dtype=torch.bool, device=task.device)
    episode_step_count = torch.zeros(
        num_envs, dtype=torch.long, device=task.device
    )
    episode_return = torch.zeros(
        num_envs, dtype=torch.float32, device=task.device
    )
    min_root_height = task.srl_root_states[:, 2].clone()
    rows_by_env = {}

    dr_enabled = bool(cfg.task.task.get("randomize", False))
    perturb_enabled = bool(cfg.task.task.get("vel_pertubation", False))
    deterministic_actor = bool(
        _eval_option(cfg, "deterministic", player.is_deterministic)
    )
    print("Isaac Gym SRL first-episode evaluation")
    print(f"  checkpoint: {cfg.checkpoint}")
    print(f"  environments/samples: {num_envs}")
    print(f"  global seed: {cfg.seed}")
    print(f"  episode steps: {episode_steps}")
    print(f"  asset: {cfg.task.env.asset.assetFileName}")
    print(
        "  controller: "
        f"forceControl={bool(cfg.task.env.forceControl)}, "
        f"pdControl={bool(cfg.task.env.pdControl)}, "
        f"action_filter={bool(cfg.task.env.srl_action_filter_enable)}"
    )
    print(f"  kp: {_to_cpu(task.p_gains).tolist()}")
    print(f"  kd: {_to_cpu(task.d_gains).tolist()}")
    print(f"  deterministic actor: {deterministic_actor}")
    print(f"  domain randomization: {dr_enabled}")
    print(
        "  full-strength DR from first episode: "
        f"{bool(cfg.eval_full_strength_dr)} "
        f"(removed schedule fields: {int(cfg.eval_removed_dr_schedule_fields)})"
    )
    print(f"  velocity perturbation: {perturb_enabled}")
    print(f"  fixed command: {fixed_command}")

    with torch.no_grad():
        for step_index in range(episode_steps + 5):
            if step_index > 0:
                obs_dict, _ = player._env_reset_done()
            action = player.get_action(
                obs_dict, is_determenistic=deterministic_actor
            )
            obs_dict, reward, done, info = player.env_step(env, action)

            active_before = active.clone()
            episode_step_count[active_before] += 1
            reward_on_task_device = reward.to(task.device)
            episode_return[active_before] += reward_on_task_device[active_before]
            current_height = task.srl_root_states[:, 2]
            min_root_height[active_before] = torch.minimum(
                min_root_height[active_before], current_height[active_before]
            )

            done_mask = done.to(task.device).bool() & active_before
            if done_mask.any():
                done_ids = done_mask.nonzero(as_tuple=False).flatten()
                terminate = info.get("terminate", task._terminate_buf)
                terminate = terminate.to(task.device).bool()
                time_outs = info.get("time_outs", torch.zeros_like(done))
                time_outs = time_outs.to(task.device).bool()

                for env_id in done_ids.detach().cpu().tolist():
                    failed = bool(terminate[env_id].item())
                    timed_out = bool(time_outs[env_id].item())
                    success = timed_out and not failed
                    if success:
                        reason = "timeout_success"
                    elif failed:
                        reason = "height_termination"
                    elif timed_out:
                        reason = "timeout_with_termination"
                    else:
                        reason = "other_done"

                    steps = int(episode_step_count[env_id].item())
                    rows_by_env[env_id] = {
                        "sample_id": env_id + 1,
                        "env_id": env_id,
                        "global_seed": int(cfg.seed),
                        "initial_phase": int(initial_phase[env_id]),
                        "success": int(success),
                        "termination_reason": reason,
                        "episode_steps": steps,
                        "episode_time_s": steps * float(task.control_dt),
                        "return": float(episode_return[env_id].item()),
                        "min_root_height": float(min_root_height[env_id].item()),
                        "final_root_height": float(current_height[env_id].item()),
                        "final_target_vx": float(task.target_vel_x[env_id].item()),
                        "final_target_wz": float(
                            task.target_ang_vel_z[env_id].item()
                        ),
                        "final_target_height": float(
                            task.target_pelvis_height[env_id].item()
                        ),
                    }
                active[done_ids] = False

            if not active.any():
                break

    # An unresolved environment indicates an evaluator/task mismatch, not a
    # successful episode. Preserve it in the report rather than hiding it.
    for env_id in active.nonzero(as_tuple=False).flatten().cpu().tolist():
        steps = int(episode_step_count[env_id].item())
        rows_by_env[env_id] = {
            "sample_id": env_id + 1,
            "env_id": env_id,
            "global_seed": int(cfg.seed),
            "initial_phase": int(initial_phase[env_id]),
            "success": 0,
            "termination_reason": "incomplete",
            "episode_steps": steps,
            "episode_time_s": steps * float(task.control_dt),
            "return": float(episode_return[env_id].item()),
            "min_root_height": float(min_root_height[env_id].item()),
            "final_root_height": float(task.srl_root_states[env_id, 2].item()),
            "final_target_vx": float(task.target_vel_x[env_id].item()),
            "final_target_wz": float(task.target_ang_vel_z[env_id].item()),
            "final_target_height": float(
                task.target_pelvis_height[env_id].item()
            ),
        }

    rows = [rows_by_env[index] for index in range(num_envs)]
    _write_results(output_path, rows)

    successes = sum(row["success"] for row in rows)
    failures = num_envs - successes
    failure_steps = [
        row["episode_steps"] for row in rows if not row["success"]
    ]
    phase_stats = {}
    for phase in sorted({row["initial_phase"] for row in rows}):
        phase_rows = [row for row in rows if row["initial_phase"] == phase]
        phase_successes = sum(row["success"] for row in phase_rows)
        phase_stats[str(phase)] = {
            "samples": len(phase_rows),
            "successes": phase_successes,
            "success_rate": phase_successes / len(phase_rows),
        }
    summary = {
        "checkpoint": str(cfg.checkpoint),
        "global_seed": int(cfg.seed),
        "samples": num_envs,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / num_envs,
        "episode_steps": episode_steps,
        "control_dt": float(task.control_dt),
        "domain_randomization": dr_enabled,
        "full_strength_domain_randomization": bool(cfg.eval_full_strength_dr),
        "removed_dr_schedule_fields": int(
            cfg.eval_removed_dr_schedule_fields
        ),
        "velocity_perturbation": perturb_enabled,
        "fixed_command": fixed_command,
        "deterministic_actor": deterministic_actor,
        "failure_step_mean": (
            float(np.mean(failure_steps)) if failure_steps else None
        ),
        "failure_step_min": min(failure_steps) if failure_steps else None,
        "phase_stats": phase_stats,
        "csv": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(
        "[Isaac Gym multi-sample summary] "
        f"samples={num_envs} success={100.0 * successes / num_envs:.1f}% "
        f"successes={successes} failures={failures} "
        f"failure_step_mean={summary['failure_step_mean']} "
        f"failure_step_min={summary['failure_step_min']}"
    )
    print("Initial-phase success rates:")
    for phase, stats in phase_stats.items():
        print(
            f"  phase={int(phase):2d} samples={stats['samples']:3d} "
            f"success={100.0 * stats['success_rate']:5.1f}%"
        )
    print(f"Saved CSV: {output_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    evaluate()
