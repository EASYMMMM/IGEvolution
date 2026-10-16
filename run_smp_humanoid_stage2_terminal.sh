#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/joemeng/SRL/IGEvolution"
WORKDIR="$ROOT/isaacgymenvs"
CONDA_PREFIX="/home/joemeng/SRL/.miniconda3/envs/srlgym"
PYTHON_BIN="$CONDA_PREFIX/bin/python"
STAGE1_CHECKPOINT="$WORKDIR/runs/Humanoid_175_Pretrain_s1_smp_13-20-08-10/nn/Humanoid_175_Pretrain_s1_smp_13-20-08-14.pth"
LOG_DIR="$ROOT/run_data/smp_humanoid_pretrain_chain"
TERMINAL_LOG="$LOG_DIR/stage2_terminal.log"

export CONDA_DEFAULT_ENV="srlgym"
export CONDA_PREFIX
export PATH="$CONDA_PREFIX/bin:/home/joemeng/SRL/.miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export C_INCLUDE_PATH="$CONDA_PREFIX/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
# W&B is reachable directly; do not inherit a stale localhost proxy.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"
exec > >(tee -a "$TERMINAL_LOG") 2>&1

if [[ ! -s "$STAGE1_CHECKPOINT" ]]; then
    printf 'Stage 1 checkpoint is missing: %s\n' "$STAGE1_CHECKPOINT" >&2
    exec /bin/bash -i
fi

stage2_pattern="^$PYTHON_BIN SRL_Evo_train.py .*experiment=Humanoid_175_Pretrain_s2_smp( |$)"
if pgrep -af "$stage2_pattern" >/dev/null; then
    printf 'Stage 2 is already running; refusing to start a duplicate.\n' >&2
    pgrep -af "$stage2_pattern"
    exec /bin/bash -i
fi

cd "$WORKDIR" || {
    printf 'Cannot enter %s\n' "$WORKDIR" >&2
    exec /bin/bash -i
}

seed="$RANDOM"
printf 'Starting Humanoid SMP pretraining stage 2\n'
printf 'Initialization checkpoint: %s\n' "$STAGE1_CHECKPOINT"
printf 'Seed: %s\n\n' "$seed"

"$PYTHON_BIN" SRL_Evo_train.py \
    task=HumanoidAMPSRLGym_s1_Smpl \
    experiment=Humanoid_175_Pretrain_s2_smp \
    task.env.stateInit=Hybrid \
    wandb_project=Humanoid_Pretrain \
    wandb_activate=True \
    max_iterations=4500 \
    task.env.episodeLength=4000 \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    num_envs=4096 \
    headless=True \
    "seed=$seed" \
    task.env.train_stage=3 \
    "checkpoint=$STAGE1_CHECKPOINT" \
    task.env.motion_file=smp_humanoid_walk_175.npy

exit_code=$?
printf '\nStage 2 exited with code %s at %s\n' "$exit_code" "$(date '+%F %T %Z')"
printf 'The terminal will remain open for inspection.\n'
exec /bin/bash -i
