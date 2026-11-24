#!/usr/bin/env bash
# Continue the three runs in srl_real_hri.bash with HMI-driven SRL decisions.
# Extra Hydra overrides may be supplied as arguments, including test=True.
set -euo pipefail

SRL_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRL_PYTHON="${SRL_PYTHON:-$SRL_REPO_ROOT/../.miniconda3/envs/srlgym/bin/python}"
SRL_ENV_DIR="$(dirname -- "$(dirname -- "$SRL_PYTHON")")"
SRL_STAGE3_CHECKPOINT="${SRL_STAGE3_CHECKPOINT:-$SRL_REPO_ROOT/isaacgymenvs/runs/SRL_Real_HRI_v2_stage3_19-20-10-40/nn/SRL_Real_HRI_v2_stage3_19-20-10-46.pth}"
SRL_STAGE4_LOG_DIR="$SRL_REPO_ROOT/run_data/srl_hmi_stage4"

[[ -x "$SRL_PYTHON" ]] || { echo "Missing Python: $SRL_PYTHON" >&2; exit 1; }
[[ -s "$SRL_STAGE3_CHECKPOINT" ]] || { echo "Missing checkpoint: $SRL_STAGE3_CHECKPOINT" >&2; exit 1; }
mkdir -p "$SRL_STAGE4_LOG_DIR"
exec 9>"$SRL_STAGE4_LOG_DIR/train.lock"
flock -n 9 || { echo 'Another HMI stage-4 run is already active.' >&2; exit 1; }

export PATH="$SRL_ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$SRL_ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export C_INCLUDE_PATH="$SRL_ENV_DIR/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$SRL_ENV_DIR/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export PYTHONUNBUFFERED=1
cd "$SRL_REPO_ROOT/isaacgymenvs"

# Training phase 4 still uses trajectory curriculum S4 (train_stage=4).
# Keep the stage-3 rewards and the humanoid's existing small, periodic updates.
# Actor command masking happens after normalization, including mirrored inputs.
exec "$SRL_PYTHON" SRL_Evo_train.py \
    task=SRL_Real_HRI headless=True \
    wandb_project=SRL_Evo wandb_activate=True \
    experiment=SRL_Real_HRI_v2_stage4_HMI max_iterations=4000 seed=42 \
    sim_device=cuda:0 rl_device=cuda:0 num_envs=8192 \
    "train.params.config.hsrl_checkpoint=$SRL_STAGE3_CHECKPOINT" \
    train.params.config.humanoid_checkpoint=False \
    train.params.config.srl_teacher_checkpoint=False \
    train.params.config.dagger_loss_coef=0.0 \
    train.params.config.train_humanoid=True \
    train.params.config.train_srl=True \
    train.params.config.train_humanoid_freq=10 \
    train.params.config.learning_rate=0.0001 \
    train.params.config.central_critic=True \
    train.params.config.sym_a_loss_coef=1.0 \
    task.env.train_stage=4 task.env.srl_hmi_control=True task.env.HMI_obs_enable=True \
    task.env.srl_max_effort=150 task.env.srl_motor_cost_scale=0.5 \
    task.env.pelvis_height_reward_scale=0.0 task.env.orientation_reward_scale=5.0 \
    task.env.no_fly_penalty_scale=2.0 task.env.gait_similarity_penalty_scale=5.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0 \
    task.env.tracking_ang_vel_reward_scale=2.0 \
    task.env.srl_free_actions_num=5 task.env.clearance_penalty_scale=10 \
    task.env.humanoid_share_reward_scale=3.0 task.env.contact_force_cost_scale=1.0 \
    task.env.asset.assetFileName=mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_mesh.xml \
    "$@"
