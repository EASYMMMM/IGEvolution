#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
srl_env="$repo_root/../.miniconda3/envs/srlgym"
export PATH="$srl_env/bin:$PATH"
export LD_LIBRARY_PATH="$srl_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export C_INCLUDE_PATH="$srl_env/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$srl_env/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TORCH_EXTENSIONS_DIR=/tmp/srl_progress_torch_extensions
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/srl_eval_mpl
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
cd "$repo_root"
out="${1:-reports/hmi_encoder_eval_20261110/validated}"
mkdir -p "$out/logs"
python -u tools/hmi_evaluation/evaluate.py --seed 17 --steps 1500 --calibration --output "$out" > "$out/logs/calibration.log" 2>&1
for seed in 101 202 303; do
    python -u tools/hmi_evaluation/evaluate.py --condition suite --seed "$seed" --steps 5000 --collect --randomize-layout --output "$out" > "$out/logs/suite_$seed.log" 2>&1
done
