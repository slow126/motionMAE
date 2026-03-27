#!/usr/bin/env bash
# Restart a consistency run by warm-starting from a prior snapshot directory.
#
# Required:
#   FLOW_MAE_CONSISTENCY_RUN_DIR=/path/to/prior/run_dir
#
# Optional:
#   FLOW_MAE_CONDA_ENV=cuda
#   FLOW_MAE_CONSISTENCY_MAX_EPOCHS=200
#   FLOW_MAE_CONSISTENCY_CONFIG=/path/to/config.yaml
#   FLOW_MAE_CONSISTENCY_RESUME_CKPT=/path/to/checkpoint.ckpt
#
# Examples:
#   FLOW_MAE_CONSISTENCY_RUN_DIR=$PWD/snapshots/flyingthings_flow_mae_dinov3_ctx_consistency_norgb_smallprobes_long_2026_03_26_12_59 \
#   sbatch slurm/run_flow_mae_dino_context_consistency_resume_rc_long.sbatch
#
#   FLOW_MAE_CONSISTENCY_RUN_DIR=$PWD/snapshots/flyingthings_flow_mae_dinov3_ctx_consistency_rgb_smallprobes_long_2026_03_26_12_59 \
#   FLOW_MAE_CONSISTENCY_MAX_EPOCHS=100 \
#   sbatch slurm/run_flow_mae_dino_context_consistency_resume_rc_long.sbatch

#SBATCH --job-name=flow_mae_ctx_cons_resume
#SBATCH --output=slurm_logs/flow_mae_ctx_cons_resume_%j.out
#SBATCH --error=slurm_logs/flow_mae_ctx_cons_resume_%j.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=128g
#SBATCH --qos=cs

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="$REPO_ROOT/slurm_logs"
mkdir -p "$LOG_DIR"

RUN_DIR="${FLOW_MAE_CONSISTENCY_RUN_DIR:-}"
if [[ -z "$RUN_DIR" ]]; then
  echo "FLOW_MAE_CONSISTENCY_RUN_DIR is required." >&2
  exit 1
fi
if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run dir does not exist: $RUN_DIR" >&2
  exit 1
fi

CONFIG_PATH="${FLOW_MAE_CONSISTENCY_CONFIG:-$RUN_DIR/config.yaml}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file does not exist: $CONFIG_PATH" >&2
  exit 1
fi

if [[ -n "${FLOW_MAE_CONSISTENCY_RESUME_CKPT:-}" ]]; then
  RESUME_CKPT="$FLOW_MAE_CONSISTENCY_RESUME_CKPT"
elif [[ -f "$RUN_DIR/checkpoints/last.ckpt" ]]; then
  RESUME_CKPT="$RUN_DIR/checkpoints/last.ckpt"
else
  shopt -s nullglob
  CKPT_CANDIDATES=("$RUN_DIR"/checkpoints/epoch*.ckpt)
  shopt -u nullglob
  if [[ ${#CKPT_CANDIDATES[@]} -eq 0 ]]; then
    echo "No checkpoint files found under $RUN_DIR/checkpoints" >&2
    exit 1
  fi
  RESUME_CKPT="$(ls -1t "${CKPT_CANDIDATES[@]}" | head -n 1)"
fi

if [[ ! -f "$RESUME_CKPT" ]]; then
  echo "Resume checkpoint does not exist: $RESUME_CKPT" >&2
  exit 1
fi

if [[ -n "${FLOW_MAE_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$FLOW_MAE_CONDA_ENV"
fi

echo "[flow_mae_ctx_consistency_resume] repo_root=$REPO_ROOT"
echo "[flow_mae_ctx_consistency_resume] run_dir=$RUN_DIR"
echo "[flow_mae_ctx_consistency_resume] config=$CONFIG_PATH"
echo "[flow_mae_ctx_consistency_resume] resume_ckpt=$RESUME_CKPT"
echo "[flow_mae_ctx_consistency_resume] max_epochs=${FLOW_MAE_CONSISTENCY_MAX_EPOCHS:-<config default>}"
echo "[flow_mae_ctx_consistency_resume] job_id=${SLURM_JOB_ID:-none}"

cd "$REPO_ROOT"

CMD=(
  python3 -u scripts/train_flow_mae_dino_context_consistency.py
  --config "$CONFIG_PATH"
  --resume-ckpt "$RESUME_CKPT"
)

if [[ -n "${FLOW_MAE_CONSISTENCY_MAX_EPOCHS:-}" ]]; then
  CMD+=(--max-epochs "$FLOW_MAE_CONSISTENCY_MAX_EPOCHS")
fi

srun --unbuffered --output="$LOG_DIR/flow_mae_ctx_cons_resume_${SLURM_JOB_ID}.log" "${CMD[@]}"
