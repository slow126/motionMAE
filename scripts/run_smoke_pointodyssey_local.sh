#!/usr/bin/env bash
set -euo pipefail

# Run the 3-cell smoke suite locally.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
CONFIG_DIR="$REPO_ROOT/src/configs/CorrespondenceConfigs"

CONFIGS=(
  "pointodyssey_smoke_full.yaml"
  "pointodyssey_smoke_random.yaml"
  "pointodyssey_smoke_heuristic.yaml"
)

cd "$REPO_ROOT"

for cfg in "${CONFIGS[@]}"; do
  echo "===== Running $cfg ====="
  "$PYTHON" -u train_lightning.py --config "$CONFIG_DIR/$cfg"
done
