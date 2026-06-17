from __future__ import annotations

import os
import re

import torch
import torch.nn as nn

import mujoco_ppo.train_mujoco_hri_finetune as base_train
from mujoco_ppo.models import ActorCritic, ModelConfig
from mujoco_ppo.srl_mujoco_hri218 import EnvConfig, SRLMujocoHRI218Env


_BASE_LOAD_HRI_CHECKPOINT = base_train.load_hri_checkpoint


def _state_linear_layers(model_state, prefix: str):
    pattern = re.compile(rf"(?:^|\.){re.escape(prefix)}\.(\d+)\.weight$")
    layers = []
    for key, value in model_state.items():
        match = pattern.search(str(key))
        if match is None or getattr(value, "ndim", 0) != 2:
            continue
        idx = int(match.group(1))
        bias_key = str(key)[:-len("weight")] + "bias"
        layers.append((idx, value, model_state.get(bias_key)))
    layers.sort(key=lambda item: item[0])
    return [(weight, bias) for _idx, weight, bias in layers]


def _infer_hidden_sizes_from_state(model_state):
    layers = _state_linear_layers(model_state, "actor_mlp")
    hidden_sizes = [int(weight.shape[0]) for weight, _bias in layers]
    return tuple(hidden_sizes) if hidden_sizes else None


def _infer_checkpoint_hidden_sizes(checkpoint_path: str, device: str, checkpoint_key: str):
    checkpoint = base_train.safe_torch_load(checkpoint_path, map_location=device)
    model_state, _selected_key = base_train._selected_model_state(checkpoint, checkpoint_key)
    return _infer_hidden_sizes_from_state(model_state)


def _infer_source_obs_dim(model_state):
    layers = _state_linear_layers(model_state, "actor_mlp")
    if layers:
        return int(layers[0][0].shape[1])
    for key in ("obs_norm.running_mean", "running_mean_std.running_mean"):
        if key in model_state:
            return int(model_state[key].numel())
    return None


def _copy_mapped_first_layer(dst: nn.Linear, src: nn.Linear):
    """Map old 153D obs [5*30+3] into new 218D obs [5*43+3]."""
    if src.weight.shape[1] != 153 or dst.weight.shape[1] != 218:
        raise ValueError(f"Expected 153->218 first layer, got {src.weight.shape} -> {dst.weight.shape}")

    with torch.no_grad():
        dst.weight.zero_()
        dst.bias.copy_(src.bias)

        old_frame = 30
        new_frame = 43
        frame_stack = 5
        for frame_idx in range(frame_stack):
            old_start = frame_idx * old_frame
            new_start = frame_idx * new_frame
            dst.weight[:, new_start:new_start + old_frame].copy_(
                src.weight[:, old_start:old_start + old_frame]
            )

        dst.weight[:, frame_stack * new_frame:frame_stack * new_frame + 3].copy_(
            src.weight[:, frame_stack * old_frame:frame_stack * old_frame + 3]
        )


def _copy_mapped_obs_norm(dst_policy: ActorCritic, src_policy: ActorCritic):
    with torch.no_grad():
        dst_policy.obs_norm.running_mean.zero_()
        dst_policy.obs_norm.running_var.fill_(1.0)

        old_frame = 30
        new_frame = 43
        frame_stack = 5
        for frame_idx in range(frame_stack):
            old_start = frame_idx * old_frame
            new_start = frame_idx * new_frame
            dst_policy.obs_norm.running_mean[new_start:new_start + old_frame].copy_(
                src_policy.obs_norm.running_mean[old_start:old_start + old_frame]
            )
            dst_policy.obs_norm.running_var[new_start:new_start + old_frame].copy_(
                src_policy.obs_norm.running_var[old_start:old_start + old_frame]
            )

        dst_task = frame_stack * new_frame
        src_task = frame_stack * old_frame
        dst_policy.obs_norm.running_mean[dst_task:dst_task + 3].copy_(
            src_policy.obs_norm.running_mean[src_task:src_task + 3]
        )
        dst_policy.obs_norm.running_var[dst_task:dst_task + 3].copy_(
            src_policy.obs_norm.running_var[src_task:src_task + 3]
        )


def _load_mujoco_153_to_218(checkpoint_path: str, model_cfg: ModelConfig, device: str, checkpoint_key: str):
    checkpoint = base_train.safe_torch_load(checkpoint_path, map_location=device)
    model_state, selected_key = base_train._selected_model_state(checkpoint, checkpoint_key)
    hidden_sizes = _infer_hidden_sizes_from_state(model_state) or model_cfg.hidden_sizes

    src_cfg = ModelConfig(
        obs_dim=153,
        act_dim=model_cfg.act_dim,
        hidden_sizes=hidden_sizes,
        activation=model_cfg.activation,
        init_log_std=model_cfg.init_log_std,
    )
    dst_cfg = ModelConfig(
        obs_dim=218,
        act_dim=model_cfg.act_dim,
        hidden_sizes=hidden_sizes,
        activation=model_cfg.activation,
        init_log_std=model_cfg.init_log_std,
    )
    src_policy = ActorCritic(src_cfg).to(device)
    src_policy.load_state_dict(model_state, strict=True)
    dst_policy = ActorCritic(dst_cfg).to(device)

    src_actor = [module for module in src_policy.actor_mlp if isinstance(module, nn.Linear)]
    dst_actor = [module for module in dst_policy.actor_mlp if isinstance(module, nn.Linear)]
    src_critic = [module for module in src_policy.critic_mlp if isinstance(module, nn.Linear)]
    dst_critic = [module for module in dst_policy.critic_mlp if isinstance(module, nn.Linear)]

    _copy_mapped_first_layer(dst_actor[0], src_actor[0])
    _copy_mapped_first_layer(dst_critic[0], src_critic[0])
    base_train._copy_same_shape_linears(dst_actor[1:], src_actor[1:])
    base_train._copy_same_shape_linears(dst_critic[1:], src_critic[1:])

    dst_policy.mu.weight.data.copy_(src_policy.mu.weight.data)
    dst_policy.mu.bias.data.copy_(src_policy.mu.bias.data)
    dst_policy.value.weight.data.copy_(src_policy.value.weight.data)
    dst_policy.value.bias.data.copy_(src_policy.value.bias.data)
    dst_policy.log_std.data.copy_(src_policy.log_std.data)
    _copy_mapped_obs_norm(dst_policy, src_policy)

    metadata = {
        "checkpoint_type": "mujoco_153_to_hri218",
        "selected_key": selected_key,
        "checkpoint_keys": sorted(list(checkpoint.keys())) if isinstance(checkpoint, dict) else [],
        "critic_loaded": True,
        "obs_norm_loaded": True,
        "source_obs_dim": 153,
        "target_obs_dim": 218,
        "model_hidden_sizes": hidden_sizes,
        "new_obs_init": "HRI per-frame weights zero; old 30D frame and task weights mapped",
    }
    return dst_policy, metadata


def load_hri218_checkpoint(checkpoint_path: str, model_cfg: ModelConfig, device: str, checkpoint_key: str):
    checkpoint = base_train.safe_torch_load(checkpoint_path, map_location=device)
    model_state, _selected_key = base_train._selected_model_state(checkpoint, checkpoint_key)
    src_obs_dim = _infer_source_obs_dim(model_state)
    if src_obs_dim == 153 and model_cfg.obs_dim == 218:
        return _load_mujoco_153_to_218(checkpoint_path, model_cfg, device, checkpoint_key)
    return _BASE_LOAD_HRI_CHECKPOINT(checkpoint_path, model_cfg, device, checkpoint_key)


def main():
    args = base_train.parse_args()

    base_train.EnvConfig = EnvConfig
    base_train.SRLMujocoHRIEnv = SRLMujocoHRI218Env
    base_train.load_hri_checkpoint = load_hri218_checkpoint
    base_train._infer_checkpoint_hidden_sizes = _infer_checkpoint_hidden_sizes

    cfg = base_train.PPOConfig(
        checkpoint_path=args.checkpoint,
        checkpoint_key=args.checkpoint_key,
        isaac_dataset_dir=args.dataset,
        isaac_replay_seq_len=args.isaac_replay_seq_len,
        xml_path=args.xml,
        total_updates=args.updates,
        rollout_steps=args.rollout_steps,
        learning_rate=args.lr,
        target_kl=args.target_kl,
        bc_coef=args.bc_coef,
        bc_warmup_updates=args.bc_warmup_updates,
        init_log_std=args.init_log_std,
        device=args.device,
        seed=args.seed,
        save_every=args.save_every,
        eval_only=args.eval_only,
        eval_steps=args.eval_steps,
        debug_rollout=args.debug_rollout,
        resume_optimizer=args.resume_optimizer,
        torque_update_mode=args.torque_update_mode,
        srl_action_filter=args.srl_action_filter,
        srl_action_filter_cutoff_hz=args.srl_action_filter_cutoff_hz,
        apply_isaac_load_cell_wrench=not args.no_isaac_load_cell_wrench,
        use_isaac_human_obs=not args.zero_isaac_human_obs,
        isaac_load_cell_force_scale=args.isaac_load_cell_force_scale,
        isaac_load_cell_torque_scale=args.isaac_load_cell_torque_scale,
        isaac_load_cell_force_clip_x=args.isaac_load_cell_force_clip_x,
        isaac_load_cell_force_clip_y=args.isaac_load_cell_force_clip_y,
        isaac_load_cell_force_clip_z=args.isaac_load_cell_force_clip_z,
        hri_wrench_ramp_time=args.hri_wrench_ramp_time,
        hri_command_ramp_time=args.hri_command_ramp_time,
        base_wobble_penalty_scale=args.base_wobble_penalty_scale,
        base_ang_acc_penalty_scale=args.base_ang_acc_penalty_scale,
        yaw_drift_penalty_scale=args.yaw_drift_penalty_scale,
        foot_impact_penalty_scale=args.foot_impact_penalty_scale,
        foot_force_threshold_bw=args.foot_force_threshold_bw,
        foot_force_penalty_power=args.foot_force_penalty_power,
        hri_wrench_penalty_scale=args.hri_wrench_penalty_scale,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )
    base_train.train(cfg)


if __name__ == "__main__":
    main()
