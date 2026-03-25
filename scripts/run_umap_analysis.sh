#!/usr/bin/env bash
# Run all three UMAP analysis modes (full, rgb_only, compare) for a given snapshot.
#
# Usage:
#   ./scripts/run_umap_analysis.sh <snapshot_path>
#
# Example:
#   ./scripts/run_umap_analysis.sh snapshots_mae/snapshots/flyingthings_flow_mae_vits_long_resume_2026_03_24_16_51
#
# Outputs land in scripts/mae_latent_umap_out_<snapshot_name>{,_rgb_only,_compare}/

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

echo "=== [1/3] full observation ==="
"$PYTHON" scripts/mae_latent_umap.py --mode full --snapshot "$SNAPSHOT"

echo ""
echo "=== [2/3] rgb only ==="
"$PYTHON" scripts/mae_latent_umap.py --mode rgb_only --snapshot "$SNAPSHOT"

echo ""
echo "=== [3/3] compare (joint UMAP) ==="
"$PYTHON" scripts/mae_latent_umap.py --mode compare --snapshot "$SNAPSHOT"

echo ""
echo "Done. Outputs in:"
SNAP_NAME="$(basename "$SNAPSHOT")"
echo "  scripts/mae_latent_umap_out_${SNAP_NAME}/"
echo "  scripts/mae_latent_umap_out_${SNAP_NAME}_rgb_only/"
echo "  scripts/mae_latent_umap_out_${SNAP_NAME}_compare/"
