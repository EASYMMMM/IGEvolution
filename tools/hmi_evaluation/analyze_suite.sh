#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
srl_env="$repo_root/../.miniconda3/envs/srlgym"
export PATH="$srl_env/bin:$PATH"
export LD_LIBRARY_PATH="$srl_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/srl_eval_mpl
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
cd "$repo_root"
out="${1:-reports/hmi_encoder_eval_20261110/validated}"
python -u tools/hmi_evaluation/analyze.py --output "$out" > "$out/logs/analysis.log" 2>&1
python -u tools/hmi_evaluation/history_controls.py --output "$out" > "$out/logs/history_controls.log" 2>&1
python tools/hmi_evaluation/summary_intervals.py --output "$out"
python tools/hmi_evaluation/render_report.py --output "$out"
