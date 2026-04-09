#!/usr/bin/env python3
"""Build a multi-target NN subset by unioning the top-K nearest neighbours
from each benchmark's scored CSV.

For each benchmark, load its scored manifest (output of score_source_samples_raw_nn.py),
select the top (budget // num_benchmarks) pairs by ascending NN distance, then
take the union across all benchmarks.  Ties within a benchmark are broken by the
score; ties across benchmarks are kept (the union de-duplicates by manifest index).

Output is a flat JSON list of manifest pair indices, matching the format of
subset_random_*.json and subset_heuristic_balanced_*.json so
PointOdysseyPairManifestDataset can load it without modification.

Usage:
  python scripts/build_multitarget_nn_subset.py \\
    --manifest-path analysis/pointodyssey_pairs_smoke/manifest.jsonl \\
    --scores \\
        kitti2012:analysis/pointodyssey_smoke_vs_kitti2012_rawnn.csv \\
        kitti2015:analysis/pointodyssey_smoke_vs_kitti2015_rawnn.csv \\
        pfpascal:analysis/pointodyssey_smoke_vs_pfpascal_rawnn.csv  \\
        pfwillow:analysis/pointodyssey_smoke_vs_pfwillow_rawnn.csv  \\
        tss:analysis/pointodyssey_smoke_vs_tss_rawnn.csv            \\
    --fraction 0.05 \\
    --score-col target_knn1_mean_dist_raw \\
    --output analysis/pointodyssey_pairs_smoke_5pct/subset_multitarget_nn_5_seed2021.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


MANIFEST_IDX_COL = "manifest_idx"
DEFAULT_SCORE_COL = "target_knn1_mean_dist_raw"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-target NN subset")
    p.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to manifest.jsonl (used only to determine total pair count).",
    )
    p.add_argument(
        "--scores",
        nargs="+",
        required=True,
        metavar="NAME:PATH",
        help=(
            "Per-benchmark scored CSVs in 'name:path' format. "
            "The budget is split equally across all benchmarks given here."
        ),
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=0.05,
        help="Total fraction of pairs to select (default: 0.05). "
             "Each benchmark contributes fraction/num_benchmarks.",
    )
    p.add_argument(
        "--score-col",
        default=DEFAULT_SCORE_COL,
        help=f"Score column to rank by (ascending). Default: {DEFAULT_SCORE_COL}",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    return p.parse_args()


def _parse_scores_arg(raw: List[str]) -> List[Tuple[str, Path]]:
    result = []
    for item in raw:
        if ":" not in item:
            raise ValueError(f"--scores entries must be 'name:path', got: {item!r}")
        name, path_str = item.split(":", 1)
        result.append((name.strip(), Path(path_str.strip())))
    return result


def _count_manifest(path: Path) -> int:
    count = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _summarize_selected_sources(manifest_path: Path, selected: List[int]) -> Dict[str, int]:
    wanted = set(int(x) for x in selected)
    counts: Dict[str, int] = {}
    with manifest_path.open("r") as f:
        for idx, line in enumerate(f):
            if idx not in wanted:
                continue
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source = str(row.get("source_dataset", "pointodyssey"))
            counts[source] = counts.get(source, 0) + 1
    return counts


def _summarize_manifest_sources(manifest_path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with manifest_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source = str(row.get("source_dataset", "pointodyssey"))
            counts[source] = counts.get(source, 0) + 1
    return counts


def _build_source_selection_summary(
    selected_counts: Dict[str, int],
    manifest_counts: Dict[str, int],
    selected_total: int,
    manifest_total: int,
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    all_sources = sorted(set(manifest_counts) | set(selected_counts))
    for source in all_sources:
        selected = int(selected_counts.get(source, 0))
        total = int(manifest_counts.get(source, 0))
        summary[source] = {
            "selected_count": selected,
            "total_count": total,
            "selected_fraction_of_subset": (float(selected) / float(selected_total)) if selected_total > 0 else 0.0,
            "selected_fraction_of_source": (float(selected) / float(total)) if total > 0 else 0.0,
            "source_fraction_of_manifest": (float(total) / float(manifest_total)) if manifest_total > 0 else 0.0,
        }
    return summary


def main() -> None:
    args = parse_args()

    benchmarks = _parse_scores_arg(args.scores)
    n_benchmarks = len(benchmarks)
    if n_benchmarks == 0:
        raise ValueError("At least one --scores entry required.")

    print(f"Counting pairs in manifest: {args.manifest_path}")
    total = _count_manifest(args.manifest_path)
    print(f"  {total} pairs total")

    total_budget = max(1, int(round(args.fraction * total)))
    per_bench_budget = max(1, total_budget // n_benchmarks)
    pct = int(round(args.fraction * 100))
    print(
        f"  Total budget: {total_budget} ({pct}% of {total}), "
        f"{n_benchmarks} benchmarks × {per_bench_budget} each"
    )

    selected_per_bench: Dict[str, List[int]] = {}
    all_selected: set[int] = set()

    for name, csv_path in benchmarks:
        print(f"\nLoading scores for [{name}]: {csv_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"Scored CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, usecols=[MANIFEST_IDX_COL, args.score_col])
        df = df.dropna(subset=[args.score_col])
        df = df.sort_values(args.score_col, ascending=True)

        budget_this = min(per_bench_budget, len(df))
        top = df.head(budget_this)[MANIFEST_IDX_COL].astype(int).tolist()
        selected_per_bench[name] = top
        all_selected.update(top)
        print(
            f"  {len(df)} scored pairs → selected top {budget_this} "
            f"(score range [{df[args.score_col].iloc[0]:.4f}, "
            f"{df[args.score_col].iloc[budget_this-1]:.4f}])"
        )

    selected = sorted(all_selected)
    print(f"\nUnion: {len(selected)} unique pairs selected")

    # Overlap report
    overlap_total = sum(len(v) for v in selected_per_bench.values()) - len(selected)
    if overlap_total > 0:
        print(f"  ({overlap_total} pairs appeared in multiple benchmarks and were deduplicated)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(selected, f)
    print(f"Saved {len(selected)} pair indices → {args.output}")

    source_counts = _summarize_selected_sources(args.manifest_path, selected)
    if source_counts:
        manifest_source_counts = _summarize_manifest_sources(args.manifest_path)
        summary_path = args.output.with_name(args.output.stem + "_source_counts.json")
        summary = {
            "manifest_path": str(args.manifest_path),
            "subset_path": str(args.output),
            "selected_count": len(selected),
            "manifest_count": total,
            "source_counts": source_counts,
            "manifest_source_counts": manifest_source_counts,
            "source_selection_summary": _build_source_selection_summary(
                selected_counts=source_counts,
                manifest_counts=manifest_source_counts,
                selected_total=len(selected),
                manifest_total=total,
            ),
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved source-count summary → {summary_path}")


if __name__ == "__main__":
    main()
