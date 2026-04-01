#!/usr/bin/env python3
"""Report source-dataset breakdowns for pooled multi-target score CSVs.

Example:
  python3 scripts/report_pooled_target_source_breakdown.py \
    --manifest-path analysis/pooled_candidates_smoke/manifest.jsonl \
    --scores \
      kitti2012:analysis/pooled_candidates_vs_kitti2012_rawnn_k24.csv \
      kitti2015:analysis/pooled_candidates_vs_kitti2015_rawnn_k24.csv \
      pfpascal:analysis/pooled_candidates_vs_pfpascal_rawnn_k24.csv \
      pfwillow:analysis/pooled_candidates_vs_pfwillow_rawnn_k24.csv \
      tss:analysis/pooled_candidates_vs_tss_rawnn_k24.csv \
    --fraction-total 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_SCORE_COL = "target_knn1_mean_dist_raw"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report source-dataset breakdown for pooled target selections")
    p.add_argument("--manifest-path", type=Path, required=True, help="Pooled manifest JSONL path.")
    p.add_argument(
        "--scores",
        nargs="+",
        required=True,
        metavar="NAME:PATH",
        help="Per-target score CSVs in name:path format.",
    )
    p.add_argument(
        "--score-col",
        type=str,
        default=DEFAULT_SCORE_COL,
        help=f"Score column to rank ascending. Default: {DEFAULT_SCORE_COL}",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--fraction-total",
        type=float,
        help="Total fraction used by the multi-target builder; split equally across all targets.",
    )
    group.add_argument(
        "--per-target-fraction",
        type=float,
        help="Fraction to use independently for each target.",
    )
    group.add_argument(
        "--per-target-count",
        type=int,
        help="Exact top-K count to use independently for each target.",
    )
    return p.parse_args()


def parse_score_args(items: List[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Expected NAME:PATH, got {item!r}")
        name, path = item.split(":", 1)
        out.append((name.strip(), Path(path.strip())))
    return out


def count_manifest_rows(path: Path) -> int:
    total = 0
    with path.open("r") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def manifest_source_counts(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source = str(row.get("source_dataset", "pointodyssey"))
            counts[source] = counts.get(source, 0) + 1
    return counts


def percent(count: int, total: int) -> float:
    return 100.0 * float(count) / float(max(1, total))


def main() -> None:
    args = parse_args()
    scores = parse_score_args(args.scores)
    n_targets = len(scores)
    manifest_total = count_manifest_rows(args.manifest_path)
    pool_counts = manifest_source_counts(args.manifest_path)

    if args.per_target_count is not None:
        per_target_budget = int(max(1, args.per_target_count))
        budget_desc = f"{per_target_budget} rows per target"
    elif args.per_target_fraction is not None:
        per_target_budget = int(max(1, round(manifest_total * float(args.per_target_fraction))))
        budget_desc = f"{args.per_target_fraction:.6f} per target -> {per_target_budget} rows"
    else:
        total_budget = int(max(1, round(manifest_total * float(args.fraction_total))))
        per_target_budget = int(max(1, total_budget // max(1, n_targets)))
        budget_desc = (
            f"total fraction {args.fraction_total:.6f} over {n_targets} targets "
            f"-> {per_target_budget} rows per target"
        )

    print("Pool Baseline")
    print("-------------")
    print(f"Manifest rows: {manifest_total}")
    print(f"Selection rule: {budget_desc}")
    for source, count in sorted(pool_counts.items()):
        print(f"{source:14} {count:9d}  {percent(count, manifest_total):6.2f}%")

    print("")
    print("Per Target Breakdown")
    print("--------------------")
    for target_name, score_path in scores:
        if not score_path.exists():
            raise FileNotFoundError(f"Score CSV not found: {score_path}")

        df = pd.read_csv(score_path, usecols=["source_dataset", args.score_col])
        df["source_dataset"] = df["source_dataset"].fillna("pointodyssey").astype(str)
        df = df.dropna(subset=[args.score_col])
        df = df.sort_values(args.score_col, ascending=True)
        top = df.head(min(per_target_budget, len(df)))
        top_total = len(top)
        top_counts = top["source_dataset"].value_counts().to_dict()

        print(f"{target_name}  top={top_total}")
        for source, count in sorted(top_counts.items()):
            pool_count = pool_counts.get(source, 0)
            sel_pct = percent(count, top_total)
            pool_pct = percent(pool_count, manifest_total)
            enrich = (sel_pct / pool_pct) if pool_pct > 0 else float("inf")
            print(
                f"  {source:12} {count:8d}  {sel_pct:6.2f}%"
                f"   pool={pool_pct:6.2f}%   enrich={enrich:6.2f}x"
            )
        if "spair" not in top_counts:
            print("  spair               0    0.00%   pool="
                  f"{percent(pool_counts.get('spair', 0), manifest_total):6.2f}%   enrich=  0.00x")
        print("")


if __name__ == "__main__":
    main()
