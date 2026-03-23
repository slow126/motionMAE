#!/usr/bin/env bash
#SBATCH --job-name=flow_mae_vits_long
#SBATCH --output=slurm_logs/flow_mae_vits_long_%j.out
#SBATCH --error=slurm_logs/flow_mae_vits_long_%j.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=128g
#SBATCH --partition=cs

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${FLOW_MAE_CONFIG:-$REPO_ROOT/src/configs/flow_mae/flyingthings_vits_rc_long.yaml}"
LOG_DIR="$REPO_ROOT/slurm_logs"
mkdir -p "$LOG_DIR"

if [[ -n "${FLOW_MAE_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$FLOW_MAE_CONDA_ENV"
fi

cd "$REPO_ROOT"
python3 -u scripts/train_flow_mae.py --config "$CONFIG_PATH" | tee "$LOG_DIR/flow_mae_vits_long_${SLURM_JOB_ID}.log"
