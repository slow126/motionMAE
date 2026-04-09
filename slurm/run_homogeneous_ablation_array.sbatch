#!/usr/bin/env bash
# Homogeneous source efficiency ablation — SLURM array job
#
# Prerequisites (run locally before submitting):
#   1. Build subsets:
#        python scripts/build_homogeneous_source_ablation_subsets.py ...
#   2. Generate RC configs:
#        python scripts/generate_homogeneous_configs.py --rc \
#          --summary analysis/homogeneous_source_ablation/homogeneous_ablation_summary.json \
#          --template src/configs/CorrespondenceConfigs/pointodyssey_spair_pfpascal_pooled_multitarget_clustercov_k1024_norm_noshortlist_dedup_0p5pct_lr1e-4.yaml \
#          --output-dir src/configs/CorrespondenceConfigs/homogeneous_ablation \
#          --lr 1e-4 --lr-backbone 3e-6 --max-steps 20000 --validation-step-interval 1000
#
# Usage:
#   # List configs first to check count:
#   ls src/configs/CorrespondenceConfigs/homogeneous_ablation/homogeneous_*.yaml | wc -l
#
#   # Submit (set --array upper bound to config count - 1):
#   sbatch --array=0-25 slurm/run_homogeneous_ablation_array.sbatch
#
# Optional env overrides:
#   HOMOG_CONFIG_DIR=src/configs/CorrespondenceConfigs/homogeneous_ablation
#   HOMOG_CONDA_ENV=cuda

#SBATCH --job-name=homog_abl
#SBATCH --output=slurm_logs/homog_abl_%A_%a.out
#SBATCH --error=slurm_logs/homog_abl_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --partition=cs

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_DIR="${HOMOG_CONFIG_DIR:-$REPO_ROOT/src/configs/CorrespondenceConfigs/homogeneous_ablation}"
LOG_DIR="$REPO_ROOT/slurm_logs"
mkdir -p "$LOG_DIR"

# Activate conda if specified
if [[ -n "${HOMOG_CONDA_ENV:-}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$HOMOG_CONDA_ENV"
fi

# Collect configs into a sorted array for deterministic index mapping
mapfile -t CONFIGS < <(find "$CONFIG_DIR" -name 'homogeneous_*.yaml' -type f | sort)
TOTAL=${#CONFIGS[@]}

if [[ $TOTAL -eq 0 ]]; then
    echo "ERROR: No homogeneous_*.yaml configs found in $CONFIG_DIR"
    exit 1
fi

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [[ $IDX -ge $TOTAL ]]; then
    echo "Array index $IDX >= total configs $TOTAL, nothing to do."
    exit 0
fi

CONFIG="${CONFIGS[$IDX]}"
CONFIG_NAME="$(basename "${CONFIG%.yaml}")"

echo "========================================"
echo "[homog_abl] job_id    = ${SLURM_JOB_ID:-none}"
echo "[homog_abl] array_idx = $IDX / $((TOTAL - 1))"
echo "[homog_abl] config    = $(basename "$CONFIG")"
echo "[homog_abl] node      = $(hostname)"
echo "[homog_abl] gpu       = ${CUDA_VISIBLE_DEVICES:-unset}"
echo "[homog_abl] started   = $(date)"
echo "========================================"

cd "$REPO_ROOT"
python3 -u train_lightning.py --config "$CONFIG" \
    2>&1 | tee "$LOG_DIR/${CONFIG_NAME}_${SLURM_ARRAY_TASK_ID}.log"

echo "[homog_abl] finished = $(date)"
