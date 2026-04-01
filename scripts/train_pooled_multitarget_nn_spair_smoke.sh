#!/usr/bin/env bash
# Train on the pooled PointOdyssey+SPAIR candidate subset.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-$REPO_ROOT/src/configs/CorrespondenceConfigs/pointodyssey_spair_pooled_multitarget_nn_k24_5pct.yaml}"

cd "$REPO_ROOT"
"$PYTHON" train_lightning.py --config "$CONFIG"
