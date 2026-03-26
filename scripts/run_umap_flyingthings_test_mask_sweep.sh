#!/usr/bin/env bash
# Run one joint FlyingThings3D test-only latent UMAP analysis across a masking sweep.
#
# Usage:
#   ./scripts/run_umap_flyingthings_test_mask_sweep.sh <snapshot_path>
#
# Example:
#   ./scripts/run_umap_flyingthings_test_mask_sweep.sh /home/spencer/Downloads/mae_mixed/snapshots/flyingthings_flow_mae_vits_long_resume_2026_03_24_16_51

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${REPO_ROOT}/.venv/bin/python"

# Fall back to conda env python if no local venv
if [ ! -f "$PYTHON" ]; then
    PYTHON="/home/spencer/miniconda3/envs/cuda/bin/python"
fi

SNAPSHOT="${1:-}"
if [ -z "$SNAPSHOT" ]; then
    echo "Usage: $0 <snapshot_path>" >&2
    exit 1
fi

cd "$REPO_ROOT"

DATASET_LABEL="FlyingThings3D [test]"
SNAP_NAME="$(basename "$SNAPSHOT")"

echo "=== FlyingThings3D test | joint mask sweep (10%..100%) ==="
"$PYTHON" scripts/mae_latent_umap.py \
    --mode full \
    --snapshot "$SNAPSHOT" \
    --dataset-label "$DATASET_LABEL" \
    --mask-percent 10 \
    --mask-percent 20 \
    --mask-percent 30 \
    --mask-percent 40 \
    --mask-percent 50 \
    --mask-percent 60 \
    --mask-percent 70 \
    --mask-percent 80 \
    --mask-percent 90 \
    --mask-percent 100 \
    --output-tag "flyingthings_test_mask_sweep"

echo "Done. Outputs in:"
printf "  scripts/mae_latent_umap_out_%s_mask_sweep_flyingthings_test_mask_sweep/\n" \
    "$SNAP_NAME"
