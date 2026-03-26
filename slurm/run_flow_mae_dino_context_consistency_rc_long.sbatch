#!/usr/bin/env bash
# Usage:
#   sbatch slurm/run_flow_mae_dino_context_consistency_rc_long.sbatch
#
# Optional overrides:
#   FLOW_MAE_CONSISTENCY_CONFIG=/abs/path/to/config.yaml
#   FLOW_MAE_CONDA_ENV=cuda
#
# Example RGB run:
#   FLOW_MAE_CONSISTENCY_CONFIG=$PWD/src/configs/flow_mae/flyingthings_dinov3_ctx_consistency_rgb_smallprobes_rc_long.yaml \
#   sbatch slurm/run_flow_mae_dino_context_consistency_rc_long.sbatch

#SBATCH --job-name=flow_mae_ctx_consistency
#SBATCH --output=slurm_logs/flow_mae_ctx_consistency_%j.out
#SBATCH --error=slurm_logs/flow_mae_ctx_consistency_%j.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=128g
#SBATCH --qos=cs

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${FLOW_MAE_CONSISTENCY_CONFIG:-$REPO_ROOT/src/configs/flow_mae/flyingthings_dinov3_ctx_consistency_norgb_smallprobes_rc_long.yaml}"
LOG_DIR="$REPO_ROOT/slurm_logs"
mkdir -p "$LOG_DIR"

if [[ -n "${FLOW_MAE_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$FLOW_MAE_CONDA_ENV"
fi

echo "[flow_mae_ctx_consistency] repo_root=$REPO_ROOT"
echo "[flow_mae_ctx_consistency] config=$CONFIG_PATH"
echo "[flow_mae_ctx_consistency] job_id=${SLURM_JOB_ID:-none}"

cd "$REPO_ROOT"
srun --unbuffered --output="$LOG_DIR/flow_mae_ctx_consistency_${SLURM_JOB_ID}.log" \
  python3 -u scripts/train_flow_mae_dino_context_consistency.py --config "$CONFIG_PATH"
