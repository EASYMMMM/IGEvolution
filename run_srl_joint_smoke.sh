#!/usr/bin/env bash
# Short Real HRI V2 morphology search. Additional arguments are Hydra overrides.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
srl_python="${SRL_PYTHON:-$repo_root/../.miniconda3/envs/srlgym/bin/python}"
if [[ ! -x "$srl_python" && -z "${SRL_PYTHON:-}" ]]; then
    srl_python="$(command -v python)"
fi
srl_python="$(realpath -s "$srl_python")"
srl_env="$(dirname -- "$(dirname -- "$srl_python")")"
seed_policy="$(realpath -m "${SRL_SEED_POLICY:-$repo_root/run_data/v2_03dis_run1/stage_5_HMIEncoder/checkpoints/policy_best.pth}")"
export PATH="$srl_env/bin:$PATH"
export LD_LIBRARY_PATH="$srl_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export C_INCLUDE_PATH="$srl_env/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$srl_env/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
cd "$repo_root/isaacgymenvs"
exec "$srl_python" -u SRLGym_train.py task=SRL_Real_HRI experiment=V2_HMI_joint_smoke \
    headless=True wandb_activate=False seed=42 num_envs=16 \
    task.env.asset.assetFileName=mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_03dis_mesh.xml \
    task.env.train_stage=4 task.env.srl_hmi_control=True task.env.HMI_obs_enable=True \
    task.env.srl_max_effort=150 task.env.srl_motor_cost_scale=0.5 \
    task.env.pelvis_height_reward_scale=0.0 task.env.orientation_reward_scale=5.0 \
    task.env.no_fly_penalty_scale=2.0 task.env.gait_similarity_penalty_scale=5.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0 \
    task.env.tracking_ang_vel_reward_scale=2.0 task.env.clearance_penalty_scale=10.0 \
    task.env.humanoid_share_reward_scale=3.0 task.env.contact_force_cost_scale=1.0 \
    task.env.design_param_obs=False task.env.episodeLength=2048 \
    "train.params.config.hsrl_checkpoint=\"$seed_policy\"" \
    train.params.config.hmi_encoder.enabled=True train.params.config.hmi_encoder.microbatch_size=128 \
    train.params.config.minibatch_size=256 train.params.config.mini_epochs=1 \
    train.params.config.amp_obs_demo_buffer_size=512 train.params.config.amp_replay_buffer_size=1024 \
    train.params.config.amp_batch_size=64 train.params.config.amp_minibatch_size=64 \
    train.gym.design_opt=RA train.gym.RA_num_iterations=2 train.gym.RA_initial_eval_epoch=64 \
    train.gym.GA_num_iterations=1 train.gym.GA_population_size=2 train.gym.GA_initial_eval_epoch=64 \
    train.gym.BO_num_iterations=2 train.gym.BO_n_initial_points=1 train.gym.BO_initial_eval_epoch=64 \
    max_iterations=64 train.gym.final_eval_epoch=64 "$@"
