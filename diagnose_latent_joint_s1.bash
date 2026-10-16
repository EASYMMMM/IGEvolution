#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/isaacgymenvs"

# Reproduce the current from-scratch Stage 1, changing only length and probes.
SEED="${SEED:-48}"
ENVS="${NUM_ENVS:-16384}"
EPOCHS="${MAX_EPOCHS:-150}"
AMP="${MIXED_PRECISION:-True}"
WANDB="${WANDB_ACTIVATE:-True}"

python SRL_Evo_train.py \
  task=SRL_Real_Bot_ConcurrentLatent \
  train=SRL_Real_Bot_ConcurrentLatentPPO \
  'checkpoint=' \
  "experiment=SRL_Real_Bot_latent_joint_s1_numerics_amp${AMP}_seed${SEED}" \
  wandb_project=SRL_Real "wandb_activate=$WANDB" headless=True \
  "seed=$SEED" "num_envs=$ENVS" "max_iterations=$EPOCHS" \
  "sim_device=${SIM_DEVICE:-cuda:0}" "rl_device=${RL_DEVICE:-cuda:0}" \
  task.env.task_training_stage=1 \
  task.task.randomize=False task.task.vel_pertubation=False \
  task.env.vel_tracking_reward_scale=8.0 \
  task.env.progress_reward_scale=1.0 task.env.alive_reward_scale=1.0 \
  task.env.srl_motor_cost_scale=0.05 \
  task.env.asset.assetFileName=mjcf/srl_real/srl_real_bot_v2.xml \
  task.env.forceControl=False task.env.pdControl=True \
  task.env.srl_action_filter_enable=False \
  task.env.stable_gait_reward.enable=False \
  'task.env.default_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]' \
  'task.env.srl_effort_limits=[90,90,350,90,90,350]' \
  train.params.config.critic_coef=0 train.params.config.save_frequency=100 \
  train.params.config.concurrent_bootstrap_initial_level=0 \
  train.params.config.concurrent_bootstrap_auto_advance=True \
  train.params.config.concurrent_bootstrap_warmup_epochs=300 \
  train.params.config.concurrent_bootstrap_min_epochs_per_level=100 \
  train.params.config.latent_use_mirror_consistency=False \
  train.params.config.latent_ppo_update_encoder=True \
  "train.params.config.mixed_precision=$AMP" \
  train.params.config.latent_ppo_numerics_enable=True \
  train.params.config.latent_ppo_numerics_every=10 \
  train.params.config.latent_ppo_numerics_samples=1024
