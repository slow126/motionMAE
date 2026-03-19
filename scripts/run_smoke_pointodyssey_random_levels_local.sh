#!/usr/bin/env bash
set -euo pipefail

# Run random-level PointOdyssey smoke suite locally (50%, 30%, 10%).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
CONFIG_DIR="$REPO_ROOT/src/configs/CorrespondenceConfigs"

CONFIGS=(
  "pointodyssey_smoke_random_50.yaml"
  "pointodyssey_smoke_random.yaml"
  "pointodyssey_smoke_random_10.yaml"
)

cd "$REPO_ROOT"

for cfg in "${CONFIGS[@]}"; do
  echo "===== Running $cfg ====="
  "$PYTHON" -u train_lightning.py --config "$CONFIG_DIR/$cfg"
done
