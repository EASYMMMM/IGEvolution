import json
import os
from datetime import datetime

import torch

try:
    from isaacgymenvs.learning.SRLEvo.privileged_estimator import load_privileged_estimator
    from isaacgymenvs.learning.SRLEvo.privileged_estimator_adapter import (
        EstimatedObservationAdapter,
    )
except ModuleNotFoundError:
    from learning.SRLEvo.privileged_estimator import load_privileged_estimator
    from learning.SRLEvo.privileged_estimator_adapter import (
        EstimatedObservationAdapter,
    )


TARGET_NAMES = ("root_height", "local_vx", "local_vy", "local_vz")


def _unwrap_task(env):
    current = env
    for _ in range(8):
        if hasattr(current, "full_obs_buffer") and hasattr(current, "obs_buf"):
            return current
        if not hasattr(current, "env"):
            break
        current = current.env
    raise RuntimeError("Could not find the Isaac Gym task below the rl-games wrapper")


def _error_metrics(errors):
    absolute = errors.abs()
    result = {}
    for index, name in enumerate(TARGET_NAMES):
        error = errors[:, index]
        abs_error = absolute[:, index]
        result[name] = {
            "bias": float(error.mean()),
            "mae": float(abs_error.mean()),
            "rmse": float(torch.sqrt(error.square().mean())),
            "p95_abs": float(torch.quantile(abs_error, 0.95)),
            "p99_abs": float(torch.quantile(abs_error, 0.99)),
            "max_abs": float(abs_error.max()),
        }
    return result


def evaluate_privileged_estimator_closed_loop(player):
    config = player.config
    use_ground_truth = bool(config.get("estimator_closed_loop_use_ground_truth", False))
    alpha = float(config.get("estimator_closed_loop_alpha", 1.0))
    if use_ground_truth:
        alpha = 0.0
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("estimator_closed_loop_alpha must be in [0, 1]")
    estimator_checkpoint = config.get("privileged_estimator_checkpoint", "")
    if alpha > 0.0 and not estimator_checkpoint:
        raise ValueError("privileged_estimator_checkpoint must be provided")
    total_steps = int(config.get("estimator_closed_loop_steps", 6000))
    deterministic = bool(config.get("estimator_closed_loop_deterministic", True))
    output_root = config.get("estimator_closed_loop_output_dir", "estimator_closed_loop")
    if alpha == 0.0:
        run_prefix = "ground_truth_policy"
    elif alpha == 1.0:
        run_prefix = "estimated_policy"
    else:
        run_prefix = "blended_alpha_{}".format(str(alpha).replace(".", "p"))
    run_name = datetime.now().strftime(run_prefix + "_%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, run_name)
    os.makedirs(output_dir, exist_ok=False)

    task = _unwrap_task(player.env)
    obs_dict = player.env_reset(player.env)
    player.get_batch_size(obs_dict["obs"], 1)
    observation = obs_dict["obs"]
    num_envs = observation.shape[0]
    if alpha == 0.0:
        model = None
        estimator_state = {}
        adapter = None
    else:
        model, estimator_state = load_privileged_estimator(
            estimator_checkpoint, device=player.device
        )
        adapter = EstimatedObservationAdapter(model, num_envs, player.device)

    current_returns = torch.zeros(num_envs, device=player.device)
    current_lengths = torch.zeros(num_envs, dtype=torch.long, device=player.device)
    completed_returns = []
    completed_lengths = []
    completed_terminated = []
    estimator_errors = []
    first = torch.ones(num_envs, dtype=torch.bool, device=player.device)

    if alpha == 0.0:
        observation_mode = "ground_truth"
    elif alpha == 1.0:
        observation_mode = "estimated"
    else:
        observation_mode = "blended(alpha={:.2f})".format(alpha)
    print(
        "Running {}-observation policy for {} steps with {} envs".format(
            observation_mode, total_steps, num_envs
        )
    )
    with torch.no_grad():
        for step in range(total_steps):
            if alpha == 0.0:
                actor_observation = observation
            else:
                actor_observation, estimate = adapter.transform(
                    observation, first, alpha=alpha
                )
                clean_target = task.full_obs_buffer[:, 0, 0:4].to(player.device)
                estimator_errors.append((estimate - clean_target).detach().cpu())

            actor_obs_dict = dict(obs_dict)
            actor_obs_dict["obs"] = actor_observation
            action = player.get_action(actor_obs_dict, deterministic)
            obs_dict, reward, done, info = player.env_step(player.env, action)
            observation = obs_dict["obs"]

            reward = reward.to(player.device)
            done_mask = done.to(player.device).bool()
            current_returns += reward
            current_lengths += 1
            if done_mask.any():
                terminated = info.get("terminate", torch.zeros_like(done_mask))
                terminated = terminated.to(player.device).bool()
                completed_returns.extend(current_returns[done_mask].cpu().tolist())
                completed_lengths.extend(current_lengths[done_mask].cpu().tolist())
                completed_terminated.extend(terminated[done_mask].cpu().tolist())
                current_returns[done_mask] = 0.0
                current_lengths[done_mask] = 0
            first = done_mask

            if (step + 1) % 500 == 0 or step + 1 == total_steps:
                print(
                    "step {}/{} completed_episodes={}".format(
                        step + 1, total_steps, len(completed_lengths)
                    )
                )

    if estimator_errors:
        errors = torch.cat(estimator_errors, dim=0)
        estimation = _error_metrics(errors)
    else:
        estimation = None
    episode_count = len(completed_lengths)
    terminated_count = int(sum(bool(value) for value in completed_terminated))
    timeout_count = episode_count - terminated_count
    mean_length = (
        float(sum(completed_lengths)) / episode_count if episode_count else 0.0
    )
    mean_return = (
        float(sum(completed_returns)) / episode_count if episode_count else 0.0
    )
    summary = {
        "observation_mode": observation_mode,
        "estimator_closed_loop_alpha": alpha,
        "policy_checkpoint": str(config.get("load_path", "")),
        "estimator_checkpoint": (
            os.path.abspath(estimator_checkpoint) if estimator_checkpoint else ""
        ),
        "estimator_epoch": int(estimator_state.get("epoch", -1)),
        "num_envs": int(num_envs),
        "total_steps": total_steps,
        "completed_episodes": episode_count,
        "terminated_episodes": terminated_count,
        "timeout_episodes": timeout_count,
        "termination_rate": float(terminated_count) / max(episode_count, 1),
        "timeout_rate": float(timeout_count) / max(episode_count, 1),
        "mean_episode_length": mean_length,
        "mean_episode_return": mean_return,
        "active_episode_mean_length": float(current_lengths.float().mean()),
        "active_episode_min_length": int(current_lengths.min()),
        "estimation": estimation,
    }
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w") as file:
        json.dump(summary, file, indent=2)

    print("\n{}-observation closed-loop evaluation".format(observation_mode))
    print(
        "episodes={} terminated={} timeout={} termination_rate={:.4f}".format(
            episode_count,
            terminated_count,
            timeout_count,
            summary["termination_rate"],
        )
    )
    print(
        "mean_length={:.2f} mean_return={:.4f} active_min_length={}".format(
            mean_length, mean_return, summary["active_episode_min_length"]
        )
    )
    if estimation is not None:
        for name in TARGET_NAMES:
            item = estimation[name]
            print(
                "{}: bias={:.6f} MAE={:.6f} RMSE={:.6f} P95={:.6f} P99={:.6f}".format(
                    name,
                    item["bias"],
                    item["mae"],
                    item["rmse"],
                    item["p95_abs"],
                    item["p99_abs"],
                )
            )
    print("Saved {}".format(path))
    return summary
