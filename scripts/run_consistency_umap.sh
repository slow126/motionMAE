#!/usr/bin/env bash
# Run both UMAP ablations for a consistency MAE snapshot.
#
# Usage:
#   ./scripts/run_consistency_umap.sh <snapshot_path> [dino_model_dir]
#
# Example:
#   ./scripts/run_consistency_umap.sh \
#     /home/spencer/Downloads/vicrg/flyingthings_flow_mae_dinov3_ctx_consistency_norgb_smallprobes_long_2026_03_26_12_59
#
# If dino_model_dir is omitted, it is read from the snapshot's config.yaml.
# Outputs land in scripts/consistency_umap_out_<snapshot_name>_{all_datasets,ft3d_sweep}/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON="$PYTHON_BIN"
elif [ -f "${REPO_ROOT}/.venv/bin/python" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
else
    PYTHON="$(command -v python)"
fi

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    echo "Could not find a usable python interpreter." >&2
    echo "Set PYTHON_BIN=/path/to/python or activate the intended environment first." >&2
    exit 1
fi

SNAPSHOT="${1:-}"
if [ -z "$SNAPSHOT" ]; then
    echo "Usage: $0 <snapshot_path> [dino_model_dir]" >&2
    exit 1
fi

DINO_ARG=""
if [ -n "${2:-}" ]; then
    DINO_ARG="--dino-model-dir ${2}"
fi

FT3D_MASK_MODE_ARG=""
if [ -n "${FT3D_MASK_MODE:-}" ]; then
    FT3D_MASK_MODE_ARG="--ft3d-mask-mode ${FT3D_MASK_MODE}"
fi

cd "$REPO_ROOT"

echo "[python] $PYTHON"
echo "=== [1/2] all_datasets (full vs fully-masked, all datasets) ==="
"$PYTHON" scripts/consistency_umap.py \
    --snapshot "$SNAPSHOT" \
    --mode all_datasets \
    $DINO_ARG

echo ""
echo "=== [2/2] ft3d_sweep (FlyingThings, 0%–100% masked in 10% steps) ==="
"$PYTHON" scripts/consistency_umap.py \
    --snapshot "$SNAPSHOT" \
    --mode ft3d_sweep \
    $FT3D_MASK_MODE_ARG \
    $DINO_ARG

echo ""
SNAP_NAME="$(basename "$SNAPSHOT")"
echo "Done. Outputs in:"
echo "  scripts/consistency_umap_out_${SNAP_NAME}_all_datasets/"
echo "  scripts/consistency_umap_out_${SNAP_NAME}_ft3d_sweep/"
