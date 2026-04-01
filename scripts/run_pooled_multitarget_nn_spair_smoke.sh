#!/usr/bin/env bash
# Build a pooled candidate manifest, score it against the 5-benchmark target
# set, and build a multi-target NN subset using fixed-K=24.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

POINTODYSSEY_MANIFEST="${POINTODYSSEY_MANIFEST:-$REPO_ROOT/analysis/pointodyssey_pairs_smoke/manifest.jsonl}"
POOLED_MANIFEST_DIR="${POOLED_MANIFEST_DIR:-$REPO_ROOT/analysis/pooled_candidates_smoke}"
POOLED_MANIFEST="${POOLED_MANIFEST:-$POOLED_MANIFEST_DIR/manifest.jsonl}"
SUBSET_DIR="${SUBSET_DIR:-$REPO_ROOT/analysis/pooled_candidates_smoke_5pct}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$REPO_ROOT/analysis}"

SPAIR_ROOT="${SPAIR_ROOT:-$REPO_ROOT/models/Datasets_CATs/SPair-71k}"
PFPASCAL_DATAPATH="${PFPASCAL_DATAPATH:-$REPO_ROOT/models/Datasets_CATs}"
KITTI_ROOT="${KITTI_ROOT:-/home/spencer/Data/correspondence/kitti}"
TSS_ROOT="${TSS_ROOT:-/home/spencer/Data/correspondence/TSS_CVPR2016}"
CATS_DATAPATH="${CATS_DATAPATH:-$REPO_ROOT/models/Datasets_CATs}"
INCLUDE_PFPASCAL="${INCLUDE_PFPASCAL:-0}"
PFPASCAL_SPLIT="${PFPASCAL_SPLIT:-trn}"

FRACTION="${FRACTION:-0.05}"
FIXED_QUERY_K="${FIXED_QUERY_K:-24}"
SCORE_COL="${SCORE_COL:-target_knn1_mean_dist_raw}"
MAX_POINTS_PER_PAIR="${MAX_POINTS_PER_PAIR:-128}"
SCORE_QUERY_BATCH_SIZE="${SCORE_QUERY_BATCH_SIZE:-8192}"
SEED="${SEED:-2021}"

cd "$REPO_ROOT"
mkdir -p "$POOLED_MANIFEST_DIR" "$SUBSET_DIR" "$ANALYSIS_DIR"

declare -a BENCHMARKS=(kitti2012 kitti2015 pfpascal pfwillow tss)

vector_path() { echo "$ANALYSIS_DIR/target_${1}_vectors.npy"; }
score_path()  { echo "$ANALYSIS_DIR/pooled_candidates_vs_${1}_rawnn_k${FIXED_QUERY_K}.csv"; }

SUBSET_OUT="$SUBSET_DIR/subset_multitarget_nn_k${FIXED_QUERY_K}_5_seed${SEED}.json"

echo "Building pooled manifest: $POOLED_MANIFEST"
BUILD_ARGS=(
  --output "$POOLED_MANIFEST"
  --pointodyssey-manifest "$POINTODYSSEY_MANIFEST"
  --include-spair
  --spair-root "$SPAIR_ROOT"
)
if [[ "$INCLUDE_PFPASCAL" == "1" ]]; then
  BUILD_ARGS+=(
    --include-pfpascal
    --pfpascal-datapath "$PFPASCAL_DATAPATH"
    --pfpascal-split "$PFPASCAL_SPLIT"
  )
fi
"$PYTHON" scripts/build_pooled_candidate_manifest.py "${BUILD_ARGS[@]}"

for bench in "${BENCHMARKS[@]}"; do
    VECTORS="$(vector_path "$bench")"
    SCORES="$(score_path "$bench")"

    if [[ -f "$VECTORS" ]]; then
        echo "[$bench] Target vectors already exist: $VECTORS"
    else
        echo "[$bench] Extracting benchmark vectors..."
        "$PYTHON" scripts/extract_benchmark_vectors.py \
            --benchmark "$bench" \
            --output "$VECTORS" \
            --kitti-root "$KITTI_ROOT" \
            --tss-root "$TSS_ROOT" \
            --cats-datapath "$CATS_DATAPATH" \
            --size 512
    fi

    if [[ -f "$SCORES" ]]; then
        echo "[$bench] Scored CSV already exists: $SCORES"
    else
        echo "[$bench] Scoring pooled candidates vs $bench..."
        "$PYTHON" scripts/score_source_samples_raw_nn.py \
            --manifest-path "$POOLED_MANIFEST" \
            --target-vectors "$VECTORS" \
            --output "$SCORES" \
            --max-points-per-pair "$MAX_POINTS_PER_PAIR" \
            --query-batch-size "$SCORE_QUERY_BATCH_SIZE" \
            --raw-space joint \
            --normalize-norm2x1 \
            --trust-manifest \
            --use-faiss \
            --faiss-gpu \
            --faiss-index-type ivf_flat \
            --fixed-query-k "$FIXED_QUERY_K" \
            --seed "$SEED" \
            --save-format csv
    fi
done

SCORES_ARGS=()
for bench in "${BENCHMARKS[@]}"; do
    SCORES_ARGS+=("${bench}:$(score_path "$bench")")
done

echo "Building pooled subset: $SUBSET_OUT"
"$PYTHON" scripts/build_multitarget_nn_subset.py \
    --manifest-path "$POOLED_MANIFEST" \
    --scores "${SCORES_ARGS[@]}" \
    --fraction "$FRACTION" \
    --score-col "$SCORE_COL" \
    --output "$SUBSET_OUT"

echo "Done."
echo "Manifest: $POOLED_MANIFEST"
echo "Subset:   $SUBSET_OUT"
