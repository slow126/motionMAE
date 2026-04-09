#!/usr/bin/env bash
# Build PointOdyssey-only, SPAIR-only, and PF-PASCAL-only ablation subsets
# from the pooled PointOdyssey+SPAIR+PF-PASCAL manifest.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

MANIFEST_PATH="${MANIFEST_PATH:-$REPO_ROOT/analysis/pooled_candidates_plus_pfpascal/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/analysis/pooled_candidates_plus_pfpascal_source_ablations}"
POINTODYSSEY_FRACTION="${POINTODYSSEY_FRACTION:-0.005}"
SPAIR_COUNT="${SPAIR_COUNT:-0}"
PFPASCAL_SPLIT="${PFPASCAL_SPLIT:-trn}"
SEED="${SEED:-2021}"

cd "$REPO_ROOT"

"$PYTHON" scripts/build_pooled_source_ablation_subsets.py \
  --manifest-path "$MANIFEST_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --pointodyssey-fraction "$POINTODYSSEY_FRACTION" \
  --spair-count "$SPAIR_COUNT" \
  --pfpascal-split "$PFPASCAL_SPLIT" \
  --seed "$SEED"
