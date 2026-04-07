#!/usr/bin/env bash

# =========================================================
# 毕业论文实验：6个实验 × 3个seed
# 双卡并行：GPU0 / GPU1 各跑一个worker，每卡始终只跑一个训练
# 放置目录：main/isaacgymenvs
# 运行方式：bash run_all_thesis_exp.sh
# =========================================================

set -u
set -o pipefail

# ========= 1. 固定随机种子（推荐，便于论文复现） =========
SEEDS=(21 42 84)

# ========= 2. 公共配置 =========
PROJECT="SRL_Master_Thesis"
TASK="SRL_Real_HRI"
ASSET="mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml"

HUMANOID_CKPT="runs/Humanoid_175_Pretrain_s2_30-13-53-51/nn/Humanoid_175_Pretrain_s2_30-13-53-56.pth"
TEACHER_CKPT="runs/SRL_Real_s4_25-14-45-53/nn/SRL_Real_s4.pth"
HSRL_STAGE2_CKPT="runs/SRL_Real_HRI_v1_02-20-54-41/nn/SRL_Real_HRI_v1_02-20-54-47.pth"

# 统一训练步数
MAX_ITERS=3000

LOG_DIR="logs_thesis_exp"
STATUS_DIR="status_thesis_exp"
FAILED_FILE="${LOG_DIR}/failed_runs.txt"

mkdir -p "${LOG_DIR}"
mkdir -p "${STATUS_DIR}"
: > "${FAILED_FILE}"

# ========= 3. 清理函数 =========
cleanup() {
    echo "IDLE" > "${STATUS_DIR}/gpu0.status" 2>/dev/null || true
    echo "IDLE" > "${STATUS_DIR}/gpu1.status" 2>/dev/null || true

    if [ -n "${MONITOR_PID:-}" ]; then
        kill "${MONITOR_PID}" 2>/dev/null || true
    fi
    if [ -n "${PID0:-}" ]; then
        kill "${PID0}" 2>/dev/null || true
    fi
    if [ -n "${PID1:-}" ]; then
        kill "${PID1}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'echo "[INTERRUPTED] Script interrupted."; exit 130' INT TERM

# ========= 4. 状态监视器：周期显示每张卡当前实验 =========
monitor_status () {
    while true; do
        local s0="IDLE"
        local s1="IDLE"

        [ -f "${STATUS_DIR}/gpu0.status" ] && s0=$(cat "${STATUS_DIR}/gpu0.status")
        [ -f "${STATUS_DIR}/gpu1.status" ] && s1=$(cat "${STATUS_DIR}/gpu1.status")

        echo "[`date '+%F %T'`] STATUS | GPU0: ${s0} | GPU1: ${s1}"
        sleep 30
    done
}

# ========= 5. 实验任务列表 =========
# 格式：实验类型:seed
TASKS=(
    "ours_stage2:21"
    "ours_stage2:42"
    "ours_stage2:84"

    "ours:21"
    "ours:42"
    "ours:84"

    "wo_teacher_student:21"
    "wo_teacher_student:42"
    "wo_teacher_student:84"

    "wo_central_critic:21"
    "wo_central_critic:42"
    "wo_central_critic:84"

    "wo_HMI_obs:21"
    "wo_HMI_obs:42"
    "wo_HMI_obs:84"

    "wo_coupled_reward:21"
    "wo_coupled_reward:42"
    "wo_coupled_reward:84"
)

# ========= 6. 运行单个实验 =========
run_one_exp () {
    local exp_type="$1"
    local seed="$2"
    local gpu_id="$3"

    local exp_name=""
    local log_file=""

    case "${exp_type}" in
        ours_stage2)
            exp_name="MS_ours_stage2_seed${seed}"
            ;;
        ours)
            exp_name="MS_ours_seed${seed}"
            ;;
        wo_teacher_student)
            exp_name="MS_wo_teacher_student_seed${seed}"
            ;;
        wo_central_critic)
            exp_name="MS_wo_central_critic_seed${seed}"
            ;;
        wo_HMI_obs)
            exp_name="MS_wo_HMI_obs_seed${seed}"
            ;;
        wo_coupled_reward)
            exp_name="MS_wo_coupled_reward_seed${seed}"
            ;;
        *)
            echo "Unknown exp_type: ${exp_type}"
            return 1
            ;;
    esac

    log_file="${LOG_DIR}/${exp_name}.log"

    echo "============================================================"
    echo "[`date '+%F %T'`] START: ${exp_name}"
    echo "PHYSICAL GPU: ${gpu_id}"
    echo "VISIBLE DEVICE INSIDE PROCESS: cuda:0"
    echo "Log file: ${log_file}"
    echo "============================================================"

    echo "${exp_name}" > "${STATUS_DIR}/gpu${gpu_id}.status"

    case "${exp_type}" in
        ours_stage2)
            CUDA_VISIBLE_DEVICES=${gpu_id} python SRL_Evo_train.py \
                task=${TASK} \
                headless=True \
                wandb_activate=True \
                wandb_project=${PROJECT} \
                experiment=${exp_name} \
                seed=${seed} \
                rl_device=cuda:0 \
                sim_device=cuda:0 \
                train.params.config.hsrl_checkpoint=${HSRL_STAGE2_CKPT} \
                task.env.srl_max_effort=150 \
                task.env.srl_motor_cost_scale=0.3 \
                max_iterations=${MAX_ITERS} \
                train.params.config.sym_a_loss_coef=1.0 \
                task.env.pelvis_height_reward_scale=0.0 \
                task.env.orientation_reward_scale=5.0 \
                task.env.no_fly_penalty_scale=2.0 \
                task.env.gait_similarity_penalty_scale=2.0 \
                task.env.progress_reward_scale=0.0 \
                task.env.vel_tracking_reward_scale=3.0 \
                task.env.srl_free_actions_num=5 \
                task.env.clearance_penalty_scale=10 \
                task.env.humanoid_share_reward_scale=3.0 \
                task.env.contact_force_cost_scale=0.5 \
                task.env.asset.assetFileName=${ASSET} \
                2>&1 | awk -v tag="[GPU${gpu_id} ${exp_name}] " '{print tag $0; fflush();}' | tee "${log_file}"
            ;;
        ours)
            CUDA_VISIBLE_DEVICES=${gpu_id} python SRL_Evo_train.py \
                task=${TASK} \
                headless=True \
                wandb_activate=True \
                wandb_project=${PROJECT} \
                experiment=${exp_name} \
                seed=${seed} \
                rl_device=cuda:0 \
                sim_device=cuda:0 \
                train.params.config.humanoid_checkpoint=${HUMANOID_CKPT} \
                task.env.srl_max_effort=150 \
                task.env.srl_motor_cost_scale=0.0 \
                max_iterations=${MAX_ITERS} \
                train.params.config.srl_teacher_checkpoint=${TEACHER_CKPT} \
                train.params.config.dagger_loss_coef=1 \
                train.params.config.sym_a_loss_coef=1.0 \
                task.env.pelvis_height_reward_scale=2.0 \
                task.env.no_fly_penalty_scale=2.0 \
                task.env.gait_similarity_penalty_scale=2.0 \
                task.env.progress_reward_scale=0.0 \
                task.env.vel_tracking_reward_scale=3.0 \
                train.params.config.dagger_anneal_k=1e-5 \
                task.env.srl_free_actions_num=5 \
                task.env.clearance_penalty_scale=10 \
                task.env.humanoid_share_reward_scale=2.0 \
                task.env.contact_force_cost_scale=0.5 \
                task.env.asset.assetFileName=${ASSET} \
                2>&1 | awk -v tag="[GPU${gpu_id} ${exp_name}] " '{print tag $0; fflush();}' | tee "${log_file}"
            ;;
        wo_teacher_student)
            CUDA_VISIBLE_DEVICES=${gpu_id} python SRL_Evo_train.py \
                task=${TASK} \
                headless=True \
                wandb_activate=True \
                wandb_project=${PROJECT} \
                experiment=${exp_name} \
                seed=${seed} \
                rl_device=cuda:0 \
                sim_device=cuda:0 \
                train.params.config.humanoid_checkpoint=${HUMANOID_CKPT} \
                task.env.srl_max_effort=150 \
                task.env.srl_motor_cost_scale=0.0 \
                max_iterations=${MAX_ITERS} \
                train.params.config.dagger_loss_coef=0.0 \
                train.params.config.sym_a_loss_coef=1.0 \
                task.env.pelvis_height_reward_scale=2.0 \
                task.env.no_fly_penalty_scale=2.0 \
                task.env.gait_similarity_penalty_scale=2.0 \
                task.env.progress_reward_scale=0.0 \
                task.env.vel_tracking_reward_scale=3.0 \
                train.params.config.dagger_anneal_k=1e-5 \
                task.env.srl_free_actions_num=5 \
                task.env.clearance_penalty_scale=10 \
                task.env.humanoid_share_reward_scale=2.0 \
                task.env.contact_force_cost_scale=0.5 \
                task.env.asset.assetFileName=${ASSET} \
                2>&1 | awk -v tag="[GPU${gpu_id} ${exp_name}] " '{print tag $0; fflush();}' | tee "${log_file}"
            ;;
        wo_central_critic)
            CUDA_VISIBLE_DEVICES=${gpu_id} python SRL_Evo_train.py \
                task=${TASK} \
                headless=True \
                wandb_activate=True \
                wandb_project=${PROJECT} \
                experiment=${exp_name} \
                seed=${seed} \
                rl_device=cuda:0 \
                sim_device=cuda:0 \
                train.params.config.central_critic=False \
                train.params.config.humanoid_checkpoint=${HUMANOID_CKPT} \
                task.env.srl_max_effort=150 \
                task.env.srl_motor_cost_scale=0.0 \
                max_iterations=${MAX_ITERS} \
                train.params.config.srl_teacher_checkpoint=${TEACHER_CKPT} \
                train.params.config.dagger_loss_coef=1 \
                train.params.config.sym_a_loss_coef=1.0 \
                task.env.pelvis_height_reward_scale=2.0 \
                task.env.no_fly_penalty_scale=2.0 \
                task.env.gait_similarity_penalty_scale=2.0 \
                task.env.progress_reward_scale=0.0 \
                task.env.vel_tracking_reward_scale=3.0 \
                train.params.config.dagger_anneal_k=1e-5 \
                task.env.srl_free_actions_num=5 \
                task.env.clearance_penalty_scale=10 \
                task.env.humanoid_share_reward_scale=2.0 \
                task.env.contact_force_cost_scale=0.5 \
                task.env.asset.assetFileName=${ASSET} \
                2>&1 | awk -v tag="[GPU${gpu_id} ${exp_name}] " '{print tag $0; fflush();}' | tee "${log_file}"
            ;;
        wo_HMI_obs)
            CUDA_VISIBLE_DEVICES=${gpu_id} python SRL_Evo_train.py \
                task=${TASK} \
                headless=True \
                wandb_activate=True \
                wandb_project=${PROJECT} \
                experiment=${exp_name} \
                seed=${seed} \
                rl_device=cuda:0 \
                sim_device=cuda:0 \
                task.env.HMI_obs_enable=False \
                train.params.config.humanoid_checkpoint=${HUMANOID_CKPT} \
                task.env.srl_max_effort=150 \
                task.env.srl_motor_cost_scale=0.0 \
                max_iterations=${MAX_ITERS} \
                train.params.config.srl_teacher_checkpoint=${TEACHER_CKPT} \
                train.params.config.dagger_loss_coef=1 \
                train.params.config.sym_a_loss_coef=1.0 \
                task.env.pelvis_height_reward_scale=2.0 \
                task.env.no_fly_penalty_scale=2.0 \
                task.env.gait_similarity_penalty_scale=2.0 \
                task.env.progress_reward_scale=0.0 \
                task.env.vel_tracking_reward_scale=3.0 \
                train.params.config.dagger_anneal_k=1e-5 \
                task.env.srl_free_actions_num=5 \
                task.env.clearance_penalty_scale=10 \
                task.env.humanoid_share_reward_scale=2.0 \
                task.env.contact_force_cost_scale=0.5 \
                task.env.asset.assetFileName=${ASSET} \
                2>&1 | awk -v tag="[GPU${gpu_id} ${exp_name}] " '{print tag $0; fflush();}' | tee "${log_file}"
            ;;
        wo_coupled_reward)
            CUDA_VISIBLE_DEVICES=${gpu_id} python SRL_Evo_train.py \
                task=${TASK} \
                headless=True \
                wandb_activate=True \
                wandb_project=${PROJECT} \
                experiment=${exp_name} \
                seed=${seed} \
                rl_device=cuda:0 \
                sim_device=cuda:0 \
                train.params.config.humanoid_checkpoint=${HUMANOID_CKPT} \
                task.env.srl_max_effort=150 \
                task.env.srl_motor_cost_scale=0.0 \
                max_iterations=${MAX_ITERS} \
                train.params.config.srl_teacher_checkpoint=${TEACHER_CKPT} \
                train.params.config.dagger_loss_coef=1 \
                train.params.config.sym_a_loss_coef=1.0 \
                task.env.pelvis_height_reward_scale=2.0 \
                task.env.no_fly_penalty_scale=2.0 \
                task.env.gait_similarity_penalty_scale=2.0 \
                task.env.progress_reward_scale=0.0 \
                task.env.vel_tracking_reward_scale=3.0 \
                train.params.config.dagger_anneal_k=1e-5 \
                task.env.srl_free_actions_num=5 \
                task.env.clearance_penalty_scale=10 \
                task.env.humanoid_share_reward_scale=0.0 \
                task.env.contact_force_cost_scale=0.5 \
                task.env.asset.assetFileName=${ASSET} \
                2>&1 | awk -v tag="[GPU${gpu_id} ${exp_name}] " '{print tag $0; fflush();}' | tee "${log_file}"
            ;;
    esac

    local exit_code=${PIPESTATUS[0]}

    if [ ${exit_code} -eq 0 ]; then
        echo "[`date '+%F %T'`] DONE: ${exp_name}"
    else
        echo "[`date '+%F %T'`] FAILED: ${exp_name} (exit code: ${exit_code})"
        echo "${exp_name}" >> "${FAILED_FILE}"
    fi

    echo "IDLE" > "${STATUS_DIR}/gpu${gpu_id}.status"

    echo
    return ${exit_code}
}

# ========= 7. worker：固定一张GPU，顺序执行分配到自己的任务 =========
worker_run () {
    local gpu_id="$1"
    shift
    local worker_tasks=("$@")

    echo "Worker on GPU ${gpu_id} started. Total tasks: ${#worker_tasks[@]}"

    for item in "${worker_tasks[@]}"; do
        IFS=':' read -r exp_type seed <<< "${item}"
        run_one_exp "${exp_type}" "${seed}" "${gpu_id}"
    done

    echo "Worker on GPU ${gpu_id} finished."
}

# ========= 8. 把任务轮流分给两张GPU =========
GPU0_TASKS=()
GPU1_TASKS=()

for i in "${!TASKS[@]}"; do
    if (( i % 2 == 0 )); then
        GPU0_TASKS+=("${TASKS[$i]}")
    else
        GPU1_TASKS+=("${TASKS[$i]}")
    fi
done

echo "============================================================"
echo "GPU0 will run ${#GPU0_TASKS[@]} tasks"
printf '  %s\n' "${GPU0_TASKS[@]}"
echo "------------------------------------------------------------"
echo "GPU1 will run ${#GPU1_TASKS[@]} tasks"
printf '  %s\n' "${GPU1_TASKS[@]}"
echo "============================================================"

# ========= 9. 双卡并行启动 =========
echo "IDLE" > "${STATUS_DIR}/gpu0.status"
echo "IDLE" > "${STATUS_DIR}/gpu1.status"

monitor_status &
MONITOR_PID=$!

worker_run 0 "${GPU0_TASKS[@]}" &
PID0=$!

worker_run 1 "${GPU1_TASKS[@]}" &
PID1=$!

wait ${PID0}
STATUS0=$?

wait ${PID1}
STATUS1=$?

kill "${MONITOR_PID}" 2>/dev/null || true

# ========= 10. 汇总 =========
echo "============================================================"
echo "ALL RUNS FINISHED"
echo "Seeds used: ${SEEDS[*]}"
echo "Max iterations: ${MAX_ITERS}"
echo "GPU0 worker exit code: ${STATUS0}"
echo "GPU1 worker exit code: ${STATUS1}"
echo "Logs saved in: ${LOG_DIR}"
echo "Status files in: ${STATUS_DIR}"

if [ -s "${FAILED_FILE}" ]; then
    echo "Failed runs:"
    cat "${FAILED_FILE}"
else
    echo "All runs finished without shell-level failure."
fi
echo "============================================================"