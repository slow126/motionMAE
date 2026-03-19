#!/usr/bin/env bash
set -euo pipefail

# Prepare PointOdyssey pair manifest + fixed subset files for the 3-cell smoke test.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POINTODYSSEY_ROOT="${POINTODYSSEY_ROOT:-/home/spencer/Data/PointOdyssey}"
SUBSET_SEED="${SUBSET_SEED:-2021}"
MAX_SEQUENCES="${MAX_SEQUENCES:-3}"
MIN_VALID_POINTS="${MIN_VALID_POINTS:-8}"
SUBSET_FRACTION="${SUBSET_FRACTION:-0.30}"
FAIL_STREAK_STOP="${FAIL_STREAK_STOP:-3}"
MAX_DT="${MAX_DT:-}"
NUM_WORKERS="${NUM_WORKERS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/analysis/pointodyssey_pairs_smoke}"
PCT_LABEL="$(python3 -c "print(int(round(float('$SUBSET_FRACTION')*100)))")"

cd "$REPO_ROOT"

CMD=(python3 -u scripts/build_pointodyssey_pair_manifest.py
    --pointodyssey_root "$POINTODYSSEY_ROOT"
    --output_dir "$OUTPUT_DIR"
    --split train
    --max_sequences "$MAX_SEQUENCES"
    --min_valid_points "$MIN_VALID_POINTS"
    --subset_fraction "$SUBSET_FRACTION"
    --seed "$SUBSET_SEED"
    --fail_streak_stop "$FAIL_STREAK_STOP"
)
if [[ -n "$MAX_DT" ]]; then
    CMD+=(--max_dt "$MAX_DT")
fi
if [[ -n "$NUM_WORKERS" && "$NUM_WORKERS" -gt 0 ]]; then
    CMD+=(--num_workers "$NUM_WORKERS")
fi
"${CMD[@]}"

cat <<EOF
Prepared:
  manifest: $OUTPUT_DIR/manifest.jsonl
  stats: $OUTPUT_DIR/manifest_stats.json
  random subset: $OUTPUT_DIR/subset_random_${PCT_LABEL}_seed${SUBSET_SEED}.json
  heuristic subset: $OUTPUT_DIR/subset_heuristic_balanced_${PCT_LABEL}_seed${SUBSET_SEED}.json
EOF
