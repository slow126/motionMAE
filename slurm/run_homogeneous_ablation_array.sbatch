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
#   # List filtered configs first to check count:
#   HOMOG_SOURCES=pfpascal,spair bash slurm/run_homogeneous_ablation_array.sbatch
#
#   # Submit (set --array upper bound to filtered config count - 1):
#   sbatch --array=0-15 slurm/run_homogeneous_ablation_array.sbatch
#
# Optional env overrides:
#   HOMOG_CONFIG_DIR=src/configs/CorrespondenceConfigs/homogeneous_ablation
#   HOMOG_CONDA_ENV=cuda
#   HOMOG_SOURCES=pfpascal,spair

#SBATCH --job-name=homog_abl
#SBATCH --output=slurm_logs/homog_abl_%A_%a.out
#SBATCH --error=slurm_logs/homog_abl_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --qos=cs

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_DIR="${HOMOG_CONFIG_DIR:-$REPO_ROOT/src/configs/CorrespondenceConfigs/homogeneous_ablation}"
LOG_DIR="$REPO_ROOT/slurm_logs"
SOURCE_FILTER="${HOMOG_SOURCES:-pfpascal,spair}"
mkdir -p "$LOG_DIR"

# Activate conda if specified
if [[ -n "${HOMOG_CONDA_ENV:-}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$HOMOG_CONDA_ENV"
fi

# Collect configs into a sorted array for deterministic index mapping.
# By default, only submit PF-PASCAL and SPAIR ablations.
IFS=',' read -r -a FILTER_SOURCES <<< "$SOURCE_FILTER"
mapfile -t ALL_CONFIGS < <(find "$CONFIG_DIR" -name 'homogeneous_*.yaml' -type f | sort)
CONFIGS=()
for cfg in "${ALL_CONFIGS[@]}"; do
    base="$(basename "$cfg")"
    for src in "${FILTER_SOURCES[@]}"; do
        if [[ "$base" == homogeneous_"$src"_* ]]; then
            CONFIGS+=("$cfg")
            break
        fi
    done
done
TOTAL=${#CONFIGS[@]}

if [[ $TOTAL -eq 0 ]]; then
    echo "ERROR: No homogeneous configs found in $CONFIG_DIR for sources: $SOURCE_FILTER"
    exit 1
fi

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [[ $IDX -ge $TOTAL ]]; then
    echo "Array index $IDX >= total configs $TOTAL, nothing to do."
    exit 0
fi

CONFIG="${CONFIGS[$IDX]}"
CONFIG_NAME="$(basename "${CONFIG%.yaml}")"
ARRAY_GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
RUN_PREFIX="homog_abl_${ARRAY_GROUP_ID}_${IDX}_${CONFIG_NAME}"
TRAIN_STDOUT="$LOG_DIR/${RUN_PREFIX}.train.out"
TRAIN_STDERR="$LOG_DIR/${RUN_PREFIX}.train.err"
TASK_META="$LOG_DIR/${RUN_PREFIX}.meta"
INDEX_MAP="$LOG_DIR/homog_abl_${ARRAY_GROUP_ID}_index_map.tsv"
TMP_INDEX_MAP="${INDEX_MAP}.tmp.${SLURM_JOB_ID:-$$}"
WRAPPER_STDOUT="$LOG_DIR/homog_abl_${ARRAY_GROUP_ID}_${IDX}.out"
WRAPPER_STDERR="$LOG_DIR/homog_abl_${ARRAY_GROUP_ID}_${IDX}.err"
WRAPPER_STDOUT_LINK="$LOG_DIR/${RUN_PREFIX}.slurm.out"
WRAPPER_STDERR_LINK="$LOG_DIR/${RUN_PREFIX}.slurm.err"

# Write a deterministic index -> config map for the whole array.
{
    printf "array_idx\tconfig_name\tconfig_path\n"
    for i in "${!CONFIGS[@]}"; do
        printf "%s\t%s\t%s\n" \
            "$i" \
            "$(basename "${CONFIGS[$i]%.yaml}")" \
            "${CONFIGS[$i]}"
    done
} > "$TMP_INDEX_MAP"
mv -f "$TMP_INDEX_MAP" "$INDEX_MAP"

{
    echo "job_id=${SLURM_JOB_ID:-none}"
    echo "array_job_id=${SLURM_ARRAY_JOB_ID:-none}"
    echo "array_task_id=$IDX"
    echo "source_filter=$SOURCE_FILTER"
    echo "config_name=$CONFIG_NAME"
    echo "config_path=$CONFIG"
    echo "wrapper_stdout=$WRAPPER_STDOUT"
    echo "wrapper_stderr=$WRAPPER_STDERR"
    echo "train_stdout=$TRAIN_STDOUT"
    echo "train_stderr=$TRAIN_STDERR"
    echo "index_map=$INDEX_MAP"
} > "$TASK_META"

ln -sfn "$(basename "$WRAPPER_STDOUT")" "$WRAPPER_STDOUT_LINK"
ln -sfn "$(basename "$WRAPPER_STDERR")" "$WRAPPER_STDERR_LINK"

echo "========================================"
echo "[homog_abl] job_id    = ${SLURM_JOB_ID:-none}"
echo "[homog_abl] array_job = ${SLURM_ARRAY_JOB_ID:-none}"
echo "[homog_abl] array_idx = $IDX / $((TOTAL - 1))"
echo "[homog_abl] sources   = $SOURCE_FILTER"
echo "[homog_abl] config    = $(basename "$CONFIG")"
echo "[homog_abl] node      = $(hostname)"
echo "[homog_abl] gpu       = ${CUDA_VISIBLE_DEVICES:-unset}"
echo "[homog_abl] map_file  = $INDEX_MAP"
echo "[homog_abl] slurm_out = $WRAPPER_STDOUT_LINK"
echo "[homog_abl] slurm_err = $WRAPPER_STDERR_LINK"
echo "[homog_abl] train_out = $TRAIN_STDOUT"
echo "[homog_abl] train_err = $TRAIN_STDERR"
echo "[homog_abl] started   = $(date)"
echo "========================================"

cd "$REPO_ROOT"
srun --unbuffered --output="$TRAIN_STDOUT" --error="$TRAIN_STDERR" \
    python3 -u train_lightning.py --config "$CONFIG"

echo "[homog_abl] finished = $(date)"
