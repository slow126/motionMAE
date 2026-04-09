#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANALYSIS="$REPO_ROOT/analysis"
MANIFEST="$ANALYSIS/pooled_candidates_plus_pfpascal/manifest.jsonl"
S="$ANALYSIS"
OUT="$ANALYSIS/pooled_candidates_plus_pfpascal_0p5pct/subset_multitarget_clustercov_k1024_norm_noshortlist_dedup.json"

python3 "$REPO_ROOT/scripts/build_multitarget_cluster_subset.py" \
    --manifest-path "$MANIFEST" \
    --scores \
        "kitti2012:$S/pooled_candidates_plus_pfpascal_vs_kitti2012_clustercov_k1024.csv:24" \
        "kitti2015:$S/pooled_candidates_plus_pfpascal_vs_kitti2015_clustercov_k1024.csv:24" \
        "pfpascal:$S/pooled_candidates_plus_pfpascal_vs_pfpascal_clustercov_k1024.csv:8" \
        "pfwillow:$S/pooled_candidates_plus_pfpascal_vs_pfwillow_clustercov_k1024.csv:10" \
        "tss:$S/pooled_candidates_plus_pfpascal_vs_tss_clustercov_k1024.csv:24" \
    --fraction 0.005 \
    --shortlist-count 0 \
    --alpha 1.0 \
    --deduplicate-across-benchmarks \
    --output "$OUT"
