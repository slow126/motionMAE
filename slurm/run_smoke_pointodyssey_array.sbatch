#!/usr/bin/env bash
#SBATCH --job-name=po_smoke
#SBATCH --output=slurm_logs/po_smoke_%A_%a.out
#SBATCH --error=slurm_logs/po_smoke_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --partition=cs
#SBATCH --array=0-2

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_DIR="$REPO_ROOT/src/configs/CorrespondenceConfigs"
LOG_DIR="$REPO_ROOT/slurm_logs"
mkdir -p "$LOG_DIR"
CFG_SUFFIX="${SMOKE_CFG_SUFFIX:-}"
CFG_PREFIX="${SMOKE_CFG_PREFIX:-pointodyssey_smoke}"

CONFIGS=(
  "${CFG_PREFIX}_full${CFG_SUFFIX}.yaml"
  "${CFG_PREFIX}_random${CFG_SUFFIX}.yaml"
  "${CFG_PREFIX}_heuristic${CFG_SUFFIX}.yaml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running smoke config: $CONFIG"

cd "$REPO_ROOT"

python3 -u train_lightning.py --config "$CONFIG_DIR/$CONFIG" | tee "$LOG_DIR/${CONFIG%.yaml}_$SLURM_ARRAY_TASK_ID.log"
