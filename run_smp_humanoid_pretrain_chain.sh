#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/joemeng/SRL/IGEvolution"
WORKDIR="$ROOT/isaacgymenvs"
PYTHON_BIN="/home/joemeng/SRL/.miniconda3/envs/srlgym/bin/python"
STATE_DIR="$ROOT/run_data/smp_humanoid_pretrain_chain"
CHAIN_LOG="$STATE_DIR/chain.log"
LOCK_FILE="$STATE_DIR/chain.lock"
PID_FILE="$STATE_DIR/chain.pid"

STAGE1_PID="736327"
STAGE1_RUN="$WORKDIR/runs/Humanoid_175_Pretrain_s1_smp_13-20-08-10"
STAGE1_OUTPUT="$WORKDIR/wandb/run-20261013_200811-uid_Humanoid_175_Pretrain_s1_smp_13-20-08-10/files/output.log"

mkdir -p "$STATE_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf '%s Another SMP pretraining chain monitor is already active.\n' "$(date '+%F %T %Z')" >>"$CHAIN_LOG"
    exit 0
fi

printf '%s\n' "$$" >"$PID_FILE"

log() {
    printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >>"$CHAIN_LOG"
}

fail() {
    log "ERROR: $*"
    exit 1
}

latest_final_checkpoint() {
    local run_dir="$1"
    find "$run_dir/nn" -maxdepth 1 -type f -name '*.pth' \
        ! -name '*_[0-9].pth' \
        ! -name '*_[0-9][0-9].pth' \
        ! -name '*_[0-9][0-9][0-9].pth' \
        ! -name '*_[0-9][0-9][0-9][0-9].pth' \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
}

validate_checkpoint() {
    local checkpoint="$1"
    if [[ -z "$checkpoint" || ! -s "$checkpoint" ]]; then
        return 1
    fi
    "$PYTHON_BIN" - "$checkpoint" <<'PY'
import os
import sys
import torch

path = sys.argv[1]
if os.path.getsize(path) < 1_000_000:
    raise SystemExit(1)
obj = torch.load(path, map_location="cpu")
if not isinstance(obj, dict) or not obj:
    raise SystemExit(1)
PY
}

run_stage() {
    local stage_name="$1"
    shift
    local console_log="$STATE_DIR/${stage_name}_console.log"
    log "Starting $stage_name; console log: $console_log"
    "$PYTHON_BIN" SRL_Evo_train.py "$@" >>"$console_log" 2>&1
    local exit_code=$?
    log "$stage_name exited with code $exit_code"
    return "$exit_code"
}

export CONDA_DEFAULT_ENV="srlgym"
export CONDA_PREFIX="/home/joemeng/SRL/.miniconda3/envs/srlgym"
export PATH="$CONDA_PREFIX/bin:/home/joemeng/SRL/.miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export PYTHONUNBUFFERED=1

cd "$WORKDIR" || fail "Cannot enter $WORKDIR"
log "Chain monitor started; waiting for stage 1 PID $STAGE1_PID"

while kill -0 "$STAGE1_PID" 2>/dev/null; do
    stage1_cmd="$(ps -p "$STAGE1_PID" -o args= 2>/dev/null || true)"
    if [[ "$stage1_cmd" != *"Humanoid_175_Pretrain_s1_smp"* ]]; then
        fail "PID $STAGE1_PID no longer belongs to the expected stage 1 process"
    fi
    sleep 30
done

log "Stage 1 process exited; validating successful completion"
grep -q 'MAX EPOCHS NUM!' "$STAGE1_OUTPUT" || fail "Stage 1 lacks the MAX EPOCHS NUM completion marker"
stage1_checkpoint="$(latest_final_checkpoint "$STAGE1_RUN")"
validate_checkpoint "$stage1_checkpoint" || fail "Stage 1 final checkpoint is missing or invalid: $stage1_checkpoint"
log "Stage 1 checkpoint validated: $stage1_checkpoint"

if pgrep -af 'SRL_Evo_train.py.*Humanoid_175_Pretrain_s2_smp' >/dev/null; then
    fail "A stage 2 process is already running"
fi

stage2_seed="$RANDOM"
run_stage "stage2" \
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
    "seed=$stage2_seed" \
    task.env.train_stage=3 \
    "checkpoint=$stage1_checkpoint" \
    task.env.motion_file=smp_humanoid_walk_175.npy \
    || fail "Stage 2 training command failed"

stage2_run="$(find "$WORKDIR/runs" -mindepth 1 -maxdepth 1 -type d -name 'Humanoid_175_Pretrain_s2_smp_*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
[[ -n "$stage2_run" ]] || fail "Cannot locate the stage 2 run directory"
stage2_checkpoint="$(latest_final_checkpoint "$stage2_run")"
validate_checkpoint "$stage2_checkpoint" || fail "Stage 2 final checkpoint is missing or invalid: $stage2_checkpoint"
log "Stage 2 checkpoint validated: $stage2_checkpoint"

if pgrep -af 'SRL_Evo_train.py.*Humanoid_175_Pretrain_s3_smp' >/dev/null; then
    fail "A stage 3 process is already running"
fi

stage3_seed="$RANDOM"
run_stage "stage3" \
    task=HumanoidAMPSRLGym_s1_Smpl \
    experiment=Humanoid_175_Pretrain_s3_smp \
    wandb_project=Humanoid_Pretrain \
    wandb_activate=True \
    max_iterations=6500 \
    task.env.stateInit=Default \
    "checkpoint=$stage2_checkpoint" \
    task.env.episodeLength=4000 \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    headless=True \
    "seed=$stage3_seed" \
    task.env.train_stage=4 \
    task.env.motion_file=smp_humanoid_walk_175.npy \
    || fail "Stage 3 training command failed"

stage3_run="$(find "$WORKDIR/runs" -mindepth 1 -maxdepth 1 -type d -name 'Humanoid_175_Pretrain_s3_smp_*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
[[ -n "$stage3_run" ]] || fail "Cannot locate the stage 3 run directory"
stage3_checkpoint="$(latest_final_checkpoint "$stage3_run")"
validate_checkpoint "$stage3_checkpoint" || fail "Stage 3 final checkpoint is missing or invalid: $stage3_checkpoint"
log "Stage 3 checkpoint validated: $stage3_checkpoint"
log "All three SMP Humanoid pretraining stages completed successfully"
