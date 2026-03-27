#!/usr/bin/env bash
# Build a 5-benchmark multi-target NN subset for the smoke manifest.
#
# Pipeline per benchmark:
#   1. extract_benchmark_vectors.py  → target_{bench}_vectors.npy
#   2. score_source_samples_raw_nn.py → pointodyssey_smoke_vs_{bench}_rawnn.csv
#
# Then:
#   3. build_multitarget_nn_subset.py → subset_multitarget_nn_5_seed2021.json
#
# Existing kitti2012 scored CSV is reused if present (takes ~10h to regenerate).
#
# Usage:
#   bash scripts/run_multitarget_nn_smoke.sh
#
# Env overrides (all optional):
#   PYTHON, MANIFEST_PATH, ANALYSIS_DIR, SUBSET_DIR,
#   KITTI_ROOT, TSS_ROOT, CATS_DATAPATH,
#   FRACTION, SCORE_COL, NUM_WORKERS, MAX_POINTS_PER_PAIR
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

MANIFEST_PATH="${MANIFEST_PATH:-$REPO_ROOT/analysis/pointodyssey_pairs_smoke/manifest.jsonl}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$REPO_ROOT/analysis}"
SUBSET_DIR="${SUBSET_DIR:-$REPO_ROOT/analysis/pointodyssey_pairs_smoke_5pct}"

KITTI_ROOT="${KITTI_ROOT:-/home/spencer/Data/correspondence/kitti}"
TSS_ROOT="${TSS_ROOT:-/home/spencer/Data/correspondence/TSS_CVPR2016}"
CATS_DATAPATH="${CATS_DATAPATH:-$REPO_ROOT/models/Datasets_CATs}"

FRACTION="${FRACTION:-0.05}"
SCORE_COL="${SCORE_COL:-target_knn1_mean_dist_raw}"
MAX_POINTS_PER_PAIR="${MAX_POINTS_PER_PAIR:-128}"
SCORE_QUERY_BATCH_SIZE="${SCORE_QUERY_BATCH_SIZE:-8192}"

cd "$REPO_ROOT"
mkdir -p "$ANALYSIS_DIR" "$SUBSET_DIR"

# ---------------------------------------------------------------------------
# Benchmark definitions: name, vector-npy path, scored-csv path
# ---------------------------------------------------------------------------
declare -a BENCHMARKS=(kitti2012 kitti2015 pfpascal pfwillow tss)

vector_path() { echo "$ANALYSIS_DIR/target_${1}_vectors.npy"; }
score_path()  { echo "$ANALYSIS_DIR/pointodyssey_smoke_vs_${1}_rawnn.csv"; }

SUBSET_OUT="$SUBSET_DIR/subset_multitarget_nn_5_seed2021.json"

# ---------------------------------------------------------------------------
# Step 1 + 2: extract vectors + score for each benchmark
# ---------------------------------------------------------------------------
for bench in "${BENCHMARKS[@]}"; do
    VECTORS="$(vector_path "$bench")"
    SCORES="$(score_path "$bench")"

    # --- Step 1: extract benchmark vectors (skip if exists) ---
    if [[ -f "$VECTORS" ]]; then
        echo "[$bench] Vectors already exist: $VECTORS — skipping extraction"
    else
        echo "[$bench] Extracting benchmark vectors..."
        "$PYTHON" scripts/extract_benchmark_vectors.py \
            --benchmark      "$bench" \
            --output         "$VECTORS" \
            --kitti-root     "$KITTI_ROOT" \
            --tss-root       "$TSS_ROOT" \
            --cats-datapath  "$CATS_DATAPATH" \
            --size           512
        echo "[$bench] Vectors saved: $VECTORS"
    fi

    # --- Step 2: score PO smoke pairs against this benchmark (skip if exists) ---
    if [[ -f "$SCORES" ]]; then
        echo "[$bench] Scored CSV already exists: $SCORES — skipping scoring"
    else
        echo "[$bench] Scoring smoke manifest vs $bench..."
        "$PYTHON" scripts/score_source_samples_raw_nn.py \
            --manifest-path        "$MANIFEST_PATH" \
            --target-vectors       "$VECTORS" \
            --output               "$SCORES" \
            --max-points-per-pair  "$MAX_POINTS_PER_PAIR" \
            --query-batch-size     "$SCORE_QUERY_BATCH_SIZE" \
            --raw-space            joint \
            --normalize-norm2x1 \
            --trust-manifest \
            --use-faiss \
            --faiss-gpu \
            --faiss-index-type     ivf_flat \
            --save-format          csv
        echo "[$bench] Scores saved: $SCORES"
    fi
done

# ---------------------------------------------------------------------------
# Step 3: build multi-target union subset
# ---------------------------------------------------------------------------
echo ""
echo "Building multi-target NN subset (fraction=$FRACTION)..."

SCORES_ARGS=()
for bench in "${BENCHMARKS[@]}"; do
    SCORES_ARGS+=("${bench}:$(score_path "$bench")")
done

"$PYTHON" scripts/build_multitarget_nn_subset.py \
    --manifest-path "$MANIFEST_PATH" \
    --scores        "${SCORES_ARGS[@]}" \
    --fraction      "$FRACTION" \
    --score-col     "$SCORE_COL" \
    --output        "$SUBSET_OUT"

echo ""
echo "Done. Subset: $SUBSET_OUT"
