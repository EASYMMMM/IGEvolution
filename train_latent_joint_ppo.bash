#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/isaacgymenvs"

MODE="${1:-smoke}"
JOINT="${LATENT_PPO_UPDATE_ENCODER:-True}"
CHECKPOINT="${CHECKPOINT:-runs/SRL_Real_Bot_concurrent_latent_revised_s3_10hz_width_x3_seed48_07-23-18-27/nn/SRL_Real_Bot_concurrent_latent_revised_s3_10hz_width_x3_seed48.pth}"
ENCODER_LR="${LATENT_ENCODER_LR:-0.0001}"
case "$JOINT" in
  True|true) VARIANT=joint ;;
  False|false) VARIANT=aux_only ;;
  *) printf 'LATENT_PPO_UPDATE_ENCODER must be True or False\n' >&2; exit 2 ;;
esac
case "$MODE" in
  smoke) ENVS="${NUM_ENVS:-64}"; EXTRA="${EXTRA_EPOCHS:-5}"; MINI=1024; MINI_EPOCHS=2; WANDB=False ;;
  train) ENVS="${NUM_ENVS:-16384}"; EXTRA="${EXTRA_EPOCHS:-1500}"; MINI=65536; MINI_EPOCHS=5; WANDB="${WANDB_ACTIVATE:-True}" ;;
  *) printf 'Usage: bash train_latent_joint_ppo.bash smoke|train\n' >&2; exit 2 ;;
esac
if [[ ! -f "$CHECKPOINT" ]]; then
  printf 'Checkpoint not found: %s\nSet CHECKPOINT to a v2 latent checkpoint.\n' "$CHECKPOINT" >&2
  exit 2
fi
START_EPOCH="$(python - "$CHECKPOINT" <<'PY'
import inspect
import sys
import torch
kwargs = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
checkpoint = torch.load(sys.argv[1], map_location="cpu", **kwargs)
if "concurrent_latent_model" not in checkpoint:
    raise ValueError("Expected an Isaac Gym ConcurrentLatent checkpoint")
print(int(checkpoint.get("epoch", 0)))
PY
)"
if [[ ! "$EXTRA" =~ ^[1-9][0-9]*$ ]]; then
  printf 'EXTRA_EPOCHS must be a positive integer\n' >&2; exit 2
fi
# rl_games restores epoch_num: max_iterations is an absolute end epoch.
END_EPOCH=$((START_EPOCH + EXTRA))
printf 'mode=%s variant=%s start=%s end=%s extra=%s envs=%s encoder_lr=%s\n' \
  "$MODE" "$VARIANT" "$START_EPOCH" "$END_EPOCH" "$EXTRA" "$ENVS" "$ENCODER_LR"

python SRL_Evo_train.py \
  task=SRL_Real_Bot_ConcurrentLatent train=SRL_Real_Bot_ConcurrentLatentPPO \
  "checkpoint=$CHECKPOINT" "experiment=SRL_Real_Bot_latent_${VARIANT}_s4_${MODE}_10hz_seed48" \
  wandb_project=SRL_Real "wandb_activate=$WANDB" headless=True \
  seed=48 "num_envs=$ENVS" "max_iterations=$END_EPOCH" \
  "sim_device=${SIM_DEVICE:-cuda:0}" "rl_device=${RL_DEVICE:-cuda:0}" \
  task.env.task_training_stage=3 task.env.episodeLength=5000 \
  task.task.randomize=True task.task.vel_pertubation=True \
  task.env.vel_tracking_reward_scale=6.0 task.env.pelvis_height_reward_scale=5.0 \
  task.env.orientation_reward_scale=3.0 task.env.tracking_ang_vel_reward_scale=2.0 \
  task.env.progress_reward_scale=0.0 task.env.alive_reward_scale=0.0 task.env.srl_motor_cost_scale=0.0 \
  task.env.asset.assetFileName=mjcf/srl_real/srl_real_bot_v2.xml \
  task.env.forceControl=False task.env.pdControl=True task.env.action_scale=0.55 \
  task.env.srl_action_filter_enable=True task.env.srl_action_filter_cutoff_hz=10.0 task.env.srl_action_filter_order=2 \
  task.env.srl_dynamics_dr.enable=True \
  'task.env.srl_dynamics_dr.action_delay_ms=[0.0,5.0,10.0]' \
  'task.env.srl_dynamics_dr.armature_range=[0.8,1.2]' \
  'task.env.srl_dynamics_dr.inertia_range=[0.8,1.2]' \
  'task.env.default_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]' \
  'task.env.srl_effort_limits=[90,90,350,90,90,350]' \
  task.env.stable_gait_reward.enable=True task.env.stable_gait_reward.foot_width_penalty_multiplier=1.0 \
  task.env.stable_gait_reward.landing_width_enable=False \
  task.env.stable_gait_reward.swing_placement_enable=False task.env.stable_gait_reward.contact_width_enable=False \
  task.env.stable_gait_reward.stance_slip_scale=10.0 \
  task.env.stable_gait_reward.stance_slip_deadband=0.05 task.env.stable_gait_reward.stance_slip_grace_steps=2 \
  train.params.config.critic_coef=0 train.params.config.save_frequency=100 train.params.config.a_sym_loss_coef=1.0 \
  "train.params.config.minibatch_size=$MINI" "train.params.config.mini_epochs=$MINI_EPOCHS" \
  "train.params.config.latent_ppo_update_encoder=$JOINT" \
  "train.params.config.latent_ppo_learning_rate=$ENCODER_LR" "train.params.config.latent_learning_rate=$ENCODER_LR" \
  train.params.config.latent_ppo_aux_loss_coef=1.0 \
  train.params.config.latent_history_len=10 train.params.config.latent_dim=16 \
  train.params.config.latent_use_next_observation_loss=True \
  train.params.config.latent_explicit_loss_coef=1.0 train.params.config.latent_next_observation_loss_coef=1.0 \
  train.params.config.latent_l2_loss_coef=0.0001 train.params.config.latent_use_mirror_consistency=False \
  train.params.config.latent_use_com_cop_auxiliary=False train.params.config.latent_use_contact_auxiliary=False \
  train.params.config.concurrent_bootstrap_auto_advance=True \
  train.params.config.concurrent_bootstrap_reset_on_restore=True train.params.config.concurrent_bootstrap_reset_probability=0.5 \
  train.params.config.concurrent_bootstrap_clear_metrics_on_reset=True train.params.config.latent_est_only_after_epochs=-1
