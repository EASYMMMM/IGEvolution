import os
from datetime import datetime

import torch

try:
    from isaacgymenvs.learning.SRLEvo.privileged_estimator import (
        load_privileged_estimator,
    )
    from isaacgymenvs.learning.SRLEvo.privileged_estimator_adapter import (
        EstimatedObservationAdapter,
    )
except ModuleNotFoundError:
    from learning.SRLEvo.privileged_estimator import load_privileged_estimator
    from learning.SRLEvo.privileged_estimator_adapter import (
        EstimatedObservationAdapter,
    )


def _unwrap_task(env):
    current = env
    for _ in range(8):
        if hasattr(current, "full_obs_buffer") and hasattr(current, "obs_buf"):
            return current
        if not hasattr(current, "env"):
            break
        current = current.env
    raise RuntimeError("Could not find the Isaac Gym task below the rl-games wrapper")


def _extract_sample(task):
    if tuple(task.full_obs_buffer.shape[1:]) != (5, 30):
        raise RuntimeError(
            "Expected full_obs_buffer [num_envs, 5, 30], got {}".format(
                tuple(task.full_obs_buffer.shape)
            )
        )
    if task.obs_buf.shape[1] != 153:
        raise RuntimeError(
            "The behavior policy must use 153 observations. Set "
            "'task.env.srl_policy_obs_remove_ids=[]'; got obs dim {}".format(
                task.obs_buf.shape[1]
            )
        )

    # obs_buf includes configured observation randomization. Labels stay clean.
    noisy_frames = task.obs_buf[:, :150].reshape(task.num_envs, 5, 30)
    estimator_input = noisy_frames[:, 0, 4:30]
    estimator_target = task.full_obs_buffer[:, 0, 0:4]
    return estimator_input.detach().cpu(), estimator_target.detach().cpu()


def _save_chunk(output_dir, chunk_index, prefix, samples, history_len, metadata):
    input_chunk = torch.stack(samples["input"], dim=0)
    target_chunk = torch.stack(samples["target"], dim=0)
    first_chunk = torch.stack(samples["first"], dim=0)
    payload = {
        "input": torch.cat((prefix["input"], input_chunk), dim=0).float(),
        "target": torch.cat((prefix["target"], target_chunk), dim=0).float(),
        "first": torch.cat((prefix["first"], first_chunk), dim=0).bool(),
        "loss_start": history_len - 1,
        "metadata": metadata,
    }
    path = os.path.join(output_dir, "chunk_{:05d}.pt".format(chunk_index))
    torch.save(payload, path)

    keep = history_len - 1
    next_prefix = {
        "input": payload["input"][-keep:].clone(),
        "target": payload["target"][-keep:].clone(),
        "first": payload["first"][-keep:].clone(),
    }
    print("Saved {} with {} new steps".format(path, input_chunk.shape[0]))
    return next_prefix


def collect_privileged_estimator_data(player):
    """Run a loaded 153-D policy and save causal supervised trajectories."""
    config = player.config
    history_len = int(config.get("estimator_history_len", 64))
    total_steps = int(config.get("estimator_collect_steps", 5000))
    chunk_steps = int(config.get("estimator_chunk_steps", 512))
    deterministic = bool(config.get("estimator_collect_deterministic", True))
    alpha = float(config.get("estimator_collect_alpha", 0.0))
    estimator_checkpoint = config.get("estimator_collect_checkpoint", "")
    output_root = config.get("estimator_data_dir", "estimator_data")
    alpha_name = str(alpha).replace(".", "p")
    run_name = datetime.now().strftime(
        "srl_privileged_alpha_{}_%Y%m%d_%H%M%S".format(alpha_name)
    )
    output_dir = os.path.join(output_root, run_name)
    os.makedirs(output_dir, exist_ok=False)

    if history_len < 29:
        raise ValueError("estimator_history_len must be at least 29 for this CNN")
    if total_steps <= 0 or chunk_steps <= 0:
        raise ValueError("collection and chunk step counts must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("estimator_collect_alpha must be in [0, 1]")
    if alpha > 0.0 and not estimator_checkpoint:
        raise ValueError(
            "estimator_collect_checkpoint must be provided when "
            "estimator_collect_alpha is greater than zero"
        )

    task = _unwrap_task(player.env)
    obs_dict = player.env_reset(player.env)
    # BasePlayer starts in single-observation mode. Mirror the initialization
    # performed by the normal player.run() so [num_envs, obs_dim] is preserved.
    player.get_batch_size(obs_dict["obs"], 1)
    input_now, target_now = _extract_sample(task)
    num_envs = input_now.shape[0]

    if alpha > 0.0:
        model, estimator_state = load_privileged_estimator(
            estimator_checkpoint, device=player.device
        )
        if model.history_len != history_len:
            raise ValueError(
                "estimator_history_len={} does not match checkpoint history_len={}".format(
                    history_len, model.history_len
                )
            )
        adapter = EstimatedObservationAdapter(model, num_envs, player.device)
    else:
        estimator_state = {}
        adapter = None

    # Repeat the reset observation when no past history is available.
    prefix = {
        "input": input_now.unsqueeze(0).repeat(history_len - 1, 1, 1),
        "target": target_now.unsqueeze(0).repeat(history_len - 1, 1, 1),
        "first": torch.zeros(history_len - 1, num_envs, dtype=torch.bool),
    }
    prefix["first"][0] = True
    metadata = {
        "history_len": history_len,
        "input_dim": 26,
        "target_dim": 4,
        "target_names": ["root_height", "local_vx", "local_vy", "local_vz"],
        "num_envs": num_envs,
        "control_dt": float(getattr(task, "control_dt", 0.015)),
        "checkpoint": str(config.get("load_path", "")),
        "collection_alpha": alpha,
        "estimator_checkpoint": (
            os.path.abspath(estimator_checkpoint) if estimator_checkpoint else ""
        ),
        "estimator_epoch": int(estimator_state.get("epoch", -1)),
    }
    torch.save(metadata, os.path.join(output_dir, "metadata.pt"))

    samples = {"input": [], "target": [], "first": []}
    chunk_index = 0
    print(
        "Collecting {} steps from {} envs at alpha={} into {}".format(
            total_steps, num_envs, alpha, output_dir
        )
    )

    first = torch.ones(num_envs, dtype=torch.bool, device=player.device)

    with torch.no_grad():
        for step in range(total_steps):
            obs_dict, _ = player._env_reset_done()
            if adapter is None:
                actor_observation = obs_dict["obs"]
            else:
                actor_observation, _ = adapter.transform(
                    obs_dict["obs"], first, alpha=alpha
                )
            actor_obs_dict = dict(obs_dict)
            actor_obs_dict["obs"] = actor_observation
            action = player.get_action(actor_obs_dict, deterministic)
            obs_dict, _, done, _ = player.env_step(player.env, action)
            input_now, target_now = _extract_sample(task)
            samples["input"].append(input_now)
            samples["target"].append(target_now)
            samples["first"].append(done.detach().cpu().bool())
            first = done.to(player.device).bool()

            if len(samples["input"]) >= chunk_steps or step + 1 == total_steps:
                prefix = _save_chunk(
                    output_dir, chunk_index, prefix, samples, history_len, metadata
                )
                samples = {"input": [], "target": [], "first": []}
                chunk_index += 1

    print("Estimator dataset collection complete: {}".format(output_dir))
    return output_dir
