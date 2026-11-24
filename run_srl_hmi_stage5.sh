#!/usr/bin/env bash
# Standalone stage5 entry point; does not execute the earlier stages in srl_real_hri.bash.
# Optional arguments override Hydra settings, e.g. checkpoint=/path/to/stage5.pth to resume.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
srl_python="${SRL_PYTHON:-${repo_root}/../.miniconda3/envs/srlgym/bin/python}"
export PATH="$(dirname -- "$srl_python"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$srl_python")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$repo_root/isaacgymenvs"

exec "$srl_python" -u SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True \
    train.params.config.hsrl_checkpoint=runs/SRL_Real_HRI_v2_stage4_HMI_27-12-15-43/nn/SRL_Real_HRI_v2_stage4_HMI_27-12-15-49.pth \
    task.env.train_stage=4 task.env.srl_hmi_control=True task.env.HMI_obs_enable=True \
    train.params.config.hmi_encoder.enabled=True \
    train.params.config.hmi_encoder.humanoid_lr_scale=0.3 train.params.config.hmi_encoder.learning_rate=0.0003 \
    train.params.config.hmi_encoder.microbatch_size=2048 \
    experiment=SRL_Real_HRI_v2_stage5_HMIEncoder max_iterations=4000 seed=42 \
    sim_device=cuda:0 rl_device=cuda:0 num_envs=8192 \
    train.params.config.humanoid_checkpoint=False train.params.config.srl_teacher_checkpoint=False \
    train.params.config.dagger_loss_coef=0.0 train.params.config.sym_a_loss_coef=1.0 \
    train.params.config.train_humanoid=True train.params.config.train_srl=True \
    train.params.config.train_humanoid_freq=10 train.params.config.learning_rate=0.0001 \
    train.params.config.central_critic=True \
    task.env.srl_max_effort=150 task.env.srl_motor_cost_scale=0.5 \
    task.env.pelvis_height_reward_scale=0.0 task.env.orientation_reward_scale=5.0 \
    task.env.no_fly_penalty_scale=2.0 task.env.gait_similarity_penalty_scale=5.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0 \
    task.env.tracking_ang_vel_reward_scale=2.0 \
    task.env.srl_free_actions_num=5 task.env.clearance_penalty_scale=10 \
    task.env.humanoid_share_reward_scale=3.0 task.env.contact_force_cost_scale=1.0 \
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_mesh.xml" \
    "$@"
