#!/usr/bin/env bash
# Usage:
#   sbatch slurm/run_correspondence_repr_fullmask_rc_long.sbatch
#
# Optional overrides:
#   CORR_REPR_CONFIG=/abs/path/to/config.yaml
#   CORR_REPR_RESUME_CKPT=/abs/path/to/checkpoint.ckpt
#   CORR_REPR_MAX_EPOCHS=200
#   CORR_REPR_CONDA_ENV=cuda

#SBATCH --job-name=corr_repr_fullmask
#SBATCH --output=slurm_logs/corr_repr_fullmask_%j.out
#SBATCH --error=slurm_logs/corr_repr_fullmask_%j.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=128g
#SBATCH --qos=cs

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${CORR_REPR_CONFIG:-$REPO_ROOT/src/configs/flow_mae/flyingthings_correspondence_repr_fullmask_rc_long.yaml}"
RESUME_CKPT="${CORR_REPR_RESUME_CKPT:-}"
MAX_EPOCHS="${CORR_REPR_MAX_EPOCHS:-}"
LOG_DIR="$REPO_ROOT/slurm_logs"
mkdir -p "$LOG_DIR"

if [[ -n "${CORR_REPR_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CORR_REPR_CONDA_ENV"
fi

echo "[corr_repr_fullmask] repo_root=$REPO_ROOT"
echo "[corr_repr_fullmask] config=$CONFIG_PATH"
if [[ -n "$RESUME_CKPT" ]]; then
  echo "[corr_repr_fullmask] resume_ckpt=$RESUME_CKPT"
fi
if [[ -n "$MAX_EPOCHS" ]]; then
  echo "[corr_repr_fullmask] max_epochs=$MAX_EPOCHS"
fi
echo "[corr_repr_fullmask] job_id=${SLURM_JOB_ID:-none}"

cd "$REPO_ROOT"
CMD=(
  python3 -u scripts/train_correspondence_repr.py
  --config "$CONFIG_PATH"
)
if [[ -n "$RESUME_CKPT" ]]; then
  CMD+=(--resume-ckpt "$RESUME_CKPT")
fi
if [[ -n "$MAX_EPOCHS" ]]; then
  CMD+=(--max-epochs "$MAX_EPOCHS")
fi

srun --unbuffered --output="$LOG_DIR/corr_repr_fullmask_${SLURM_JOB_ID}.log" \
  "${CMD[@]}"
