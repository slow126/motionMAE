#!/usr/bin/env bash

set -euo pipefail

MANIFEST=${1:-/path/to/pointodyssey_pair_manifest.jsonl}
PO_ROOT=${2:-/path/to/pointodyssey_root}
MODEL=${3:-det}

python src/flow_smoke/train_flow_smoke.py \
  --manifest-path "${MANIFEST}" \
  --pointodyssey-root "${PO_ROOT}" \
  --model "${MODEL}" \
  --dt-values 1,2,3,4 \
  --batch-size 4 \
  --epochs 12 \
  --size 256 \
  --max-train-samples 0 \
  --base-channels 24 \
  --lr 2e-4 \
  --beta-max 1e-3 \
  --z-dim 32 \
  --val-fraction 0.1 \
  --output-dir snapshots/flow_smoke \
  --exp-name "${MODEL}_dt1_4"

