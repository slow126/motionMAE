#!/usr/bin/env bash
#SBATCH --job-name=flow_mae_dino_ctx
#SBATCH --output=slurm_logs/flow_mae_dino_ctx_%j.out
#SBATCH --error=slurm_logs/flow_mae_dino_ctx_%j.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128g
#SBATCH --qos=cs

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${FLOW_MAE_DINO_CTX_CONFIG:-$REPO_ROOT/src/configs/flow_mae/flyingthings_dinov3_ctx_rc.yaml}"
LOG_DIR="$REPO_ROOT/slurm_logs"
mkdir -p "$LOG_DIR"

if [[ -n "${FLOW_MAE_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$FLOW_MAE_CONDA_ENV"
fi

cd "$REPO_ROOT"
python3 -u scripts/train_flow_mae_dino_context.py --config "$CONFIG_PATH" | tee "$LOG_DIR/flow_mae_dino_ctx_${SLURM_JOB_ID}.log"
