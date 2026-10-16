#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}/isaacgymenvs"

HUMANOID_TEACHER="${HRI_HUMANOID_TEACHER:-runs/Humanoid_175_Pretrain_s3_20-13-56-52/nn/Humanoid_175_Pretrain_s3_20-13-56-58.pth}"
SRL_TEACHER="${HRI_SRL_TEACHER:-runs/SRL_Real_Bot_v2_s4_09-13-35-18/nn/SRL_Real_Bot_v2_s4.pth}"
HRI_ASSET="mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_mesh.xml"

require_checkpoint() {
  local checkpoint_path="${1:?checkpoint path is required}"
  local checkpoint_label="${2:?checkpoint label is required}"
  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "Missing ${checkpoint_label}: ${checkpoint_path}" >&2
    return 1
  fi
}

common_overrides=(
  task=SRL_Real_HRI
  task.env.asset.assetFileName="${HRI_ASSET}"
  task.env.pdControl=True
  task.env.forceControl=False
  task.env.srl_free_actions_num=5
  task.env.srl_action_filter_enable=True
  task.env.srl_action_filter_cutoff_hz=10.0
  "task.env.default_srl_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]"
  "task.env.srl_effort_limits=[90,90,350,90,90,350]"
)

train_stage_1() {
  require_checkpoint "${HUMANOID_TEACHER}" "humanoid teacher checkpoint"
  require_checkpoint "${SRL_TEACHER}" "Real Bot V2 teacher checkpoint"
  python SRL_Evo_train.py "${common_overrides[@]}" \
    headless=True wandb_project=SRL_Evo wandb_activate=True \
    experiment=SRL_Real_HRI_v2_s1 max_iterations=2000 \
    task.env.train_stage=2 \
    train.params.config.humanoid_checkpoint="${HUMANOID_TEACHER}" \
    train.params.config.srl_teacher_checkpoint="${SRL_TEACHER}" \
    train.params.config.dagger_loss_coef=1.0 \
    train.params.config.dagger_anneal_k=1e-5 \
    train.params.config.dagger_min_coef=0.0 \
    train.params.config.sym_a_loss_coef=1.0 \
    task.env.humanoid_share_reward_scale=2.0 \
    task.env.contact_force_cost_scale=0.5 \
    task.env.srl_motor_cost_scale=0.0
}

train_stage_2() {
  local hri_checkpoint="${1:?Usage: $0 train-s2 <stage-1 HRI checkpoint>}"
  require_checkpoint "${hri_checkpoint}" "stage-1 HRI checkpoint"
  python SRL_Evo_train.py "${common_overrides[@]}" \
    headless=True wandb_project=SRL_Evo wandb_activate=True \
    experiment=SRL_Real_HRI_v2_s2 max_iterations=4000 \
    task.env.train_stage=3 \
    train.params.config.hsrl_checkpoint="${hri_checkpoint}" \
    train.params.config.sym_a_loss_coef=1.0 \
    task.env.humanoid_share_reward_scale=3.0 \
    task.env.contact_force_cost_scale=0.5 \
    task.env.srl_motor_cost_scale=0.3
}

train_stage_3() {
  local hri_checkpoint="${1:?Usage: $0 train-s3 <stage-2 HRI checkpoint>}"
  require_checkpoint "${hri_checkpoint}" "stage-2 HRI checkpoint"
  python SRL_Evo_train.py "${common_overrides[@]}" \
    headless=True wandb_project=SRL_Evo wandb_activate=True \
    experiment=SRL_Real_HRI_v2_s3 max_iterations=4000 \
    task.env.train_stage=4 \
    train.params.config.hsrl_checkpoint="${hri_checkpoint}" \
    train.params.config.sym_a_loss_coef=1.0 \
    task.env.humanoid_share_reward_scale=3.0 \
    task.env.contact_force_cost_scale=1.0 \
    task.env.srl_motor_cost_scale=0.5
}

test_checkpoint() {
  local hri_checkpoint="${1:?Usage: $0 test <HRI checkpoint>}"
  require_checkpoint "${hri_checkpoint}" "HRI checkpoint"
  python SRL_Evo_train.py "${common_overrides[@]}" \
    test=True force_render=True num_envs=1 \
    task.env.cameraFollow=True task.env.enableDebugVis=True \
    task.env.episodeLength=2000 \
    checkpoint="${hri_checkpoint}"
}

case "${1:-}" in
  train-s1)
    train_stage_1
    ;;
  train-s2)
    train_stage_2 "${2:-}"
    ;;
  train-s3)
    train_stage_3 "${2:-}"
    ;;
  test)
    test_checkpoint "${2:-}"
    ;;
  *)
    echo "Usage: $0 {train-s1|train-s2 <checkpoint>|train-s3 <checkpoint>|test <checkpoint>}"
    exit 2
    ;;
esac
