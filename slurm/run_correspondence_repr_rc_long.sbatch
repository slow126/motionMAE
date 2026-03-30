#!/usr/bin/env bash
# Usage:
#   sbatch slurm/run_correspondence_repr_rc_long.sbatch
#
# Optional overrides:
#   CORR_REPR_CONFIG=/abs/path/to/config.yaml
#   CORR_REPR_CONDA_ENV=cuda

#SBATCH --job-name=corr_repr
#SBATCH --output=slurm_logs/corr_repr_%j.out
#SBATCH --error=slurm_logs/corr_repr_%j.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=128g
#SBATCH --qos=cs

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${CORR_REPR_CONFIG:-$REPO_ROOT/src/configs/flow_mae/flyingthings_correspondence_repr_rc_long.yaml}"
LOG_DIR="$REPO_ROOT/slurm_logs"
mkdir -p "$LOG_DIR"

if [[ -n "${CORR_REPR_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CORR_REPR_CONDA_ENV"
fi

echo "[corr_repr] repo_root=$REPO_ROOT"
echo "[corr_repr] config=$CONFIG_PATH"
echo "[corr_repr] job_id=${SLURM_JOB_ID:-none}"

cd "$REPO_ROOT"
srun --unbuffered --output="$LOG_DIR/corr_repr_${SLURM_JOB_ID}.log" \
  python3 -u scripts/train_correspondence_repr.py --config "$CONFIG_PATH"
