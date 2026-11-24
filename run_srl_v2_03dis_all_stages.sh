#!/usr/bin/env bash
# V2 0.3 m: 直线 -> 曲线 -> 全向 -> HMI -> HMI Encoder。
# 不包含人体/SRL 单体预训练；默认使用仓库现有的两个前置 checkpoint。
# 一键训练：bash run_srl_v2_03dis_all_stages.sh --run-dir run_data/v2_03dis_run1
# 仅检查并打印五阶段完整命令：追加 --dry-run
# 中断续训：相同 --run-dir，追加 --resume（自动读取保存的配置）
# 小规模五阶段联通检查：--smoke --run-dir run_data/v2_03dis_smoke
# 参数帮助：--help；日志、命令、checkpoint 和状态保存在 --run-dir。
set -euo pipefail
SRL_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRL_PYTHON="${SRL_PYTHON:-$SRL_REPO_ROOT/../.miniconda3/envs/srlgym/bin/python}"
[[ -x "$SRL_PYTHON" ]] || { echo "Missing Python: $SRL_PYTHON" >&2; exit 1; }
SRL_ENV_DIR="$(cd -- "$(dirname -- "$SRL_PYTHON")/.." && pwd)"
export PATH="$SRL_ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$SRL_ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export C_INCLUDE_PATH="$SRL_ENV_DIR/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$SRL_ENV_DIR/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
exec "$SRL_PYTHON" -u "$SRL_REPO_ROOT/tools/srl_training/train_five_stages.py" "$@"
