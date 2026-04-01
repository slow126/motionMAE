#!/usr/bin/env python3
"""Build a multi-target subset from sparse centroid-hit score CSVs.

This is a diversity-aware alternative to top-K-by-distance selection. For each
target benchmark:
1. shortlist candidates by base distance
2. greedily select samples with maximum marginal gain over uncovered / lightly
   covered target centroids
3. union the selections across targets

The marginal gain uses a diminishing-returns feature objective:
  gain(sample) = sum_{c in hits(sample)} 1 / (1 + cover_count[c])^alpha
where alpha >= 0 controls how strongly repeated coverage is penalized.
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


MANIFEST_IDX_COL = "manifest_idx"
DEFAULT_SCORE_COL = "target_cluster_mean_min_dist"
DEFAULT_HITS_COL = "covered_centroids"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-target cluster-coverage subset")
    p.add_argument("--manifest-path", type=Path, required=True)
    p.add_argument("--scores", nargs="+", required=True, metavar="NAME:PATH")
    p.add_argument("--fraction", type=float, default=0.05)
    p.add_argument("--score-col", type=str, default=DEFAULT_SCORE_COL)
    p.add_argument("--hits-col", type=str, default=DEFAULT_HITS_COL)
    p.add_argument(
        "--shortlist-count",
        type=int,
        default=200000,
        help="Per-target shortlist size after sorting by score-col (default: 200000).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Diminishing-returns strength for repeated centroid coverage (default: 1.0).",
    )
    p.add_argument("--output", type=Path, required=True)
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


def _parse_hits(raw: object) -> np.ndarray:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.zeros((0,), dtype=np.int32)
    text = str(raw).strip()
    if not text:
        return np.zeros((0,), dtype=np.int32)
    return np.asarray([int(x) for x in text.split()], dtype=np.int32)


def _marginal_gain(hits: np.ndarray, cover_counts: Dict[int, int], alpha: float) -> float:
    if hits.size == 0:
        return 0.0
    gain = 0.0
    for cid in hits.tolist():
        gain += 1.0 / float((1 + int(cover_counts.get(int(cid), 0))) ** alpha)
    return gain


def _select_for_target(
    df: pd.DataFrame,
    budget: int,
    score_col: str,
    hits_col: str,
    shortlist_count: int,
    alpha: float,
) -> List[int]:
    df = df.dropna(subset=[score_col]).copy()
    df = df.sort_values(score_col, ascending=True)
    if shortlist_count > 0:
        df = df.head(min(int(shortlist_count), len(df)))
    if len(df) == 0 or budget <= 0:
        return []

    manifest_idx = df[MANIFEST_IDX_COL].astype(int).to_numpy()
    base_scores = df[score_col].astype(float).to_numpy()
    hits_list = [_parse_hits(x) for x in df[hits_col].tolist()]
    n = len(df)
    budget = min(int(budget), n)

    cover_counts: Dict[int, int] = {}
    selected_local = np.zeros((n,), dtype=bool)
    selected_manifest: List[int] = []
    heap: List[Tuple[float, float, int]] = []

    for local_idx in range(n):
        init_gain = _marginal_gain(hits_list[local_idx], cover_counts, alpha)
        heapq.heappush(heap, (-init_gain, float(base_scores[local_idx]), int(local_idx)))

    while len(selected_manifest) < budget and heap:
        neg_gain, base_score, local_idx = heapq.heappop(heap)
        if selected_local[local_idx]:
            continue
        actual_gain = _marginal_gain(hits_list[local_idx], cover_counts, alpha)
        if heap and (-actual_gain, base_score, local_idx) > heap[0]:
            heapq.heappush(heap, (-actual_gain, base_score, local_idx))
            continue

        selected_local[local_idx] = True
        selected_manifest.append(int(manifest_idx[local_idx]))
        for cid in hits_list[local_idx].tolist():
            cid = int(cid)
            cover_counts[cid] = int(cover_counts.get(cid, 0)) + 1

    if len(selected_manifest) < budget:
        for local_idx in range(n):
            if len(selected_manifest) >= budget:
                break
            if selected_local[local_idx]:
                continue
            selected_local[local_idx] = True
            selected_manifest.append(int(manifest_idx[local_idx]))

    return selected_manifest


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
    print(
        f"  Total budget: {total_budget}, "
        f"{n_benchmarks} benchmarks x {per_bench_budget} each, "
        f"shortlist_count={int(args.shortlist_count)}, alpha={float(args.alpha):.3f}"
    )

    selected_per_bench: Dict[str, List[int]] = {}
    all_selected: set[int] = set()

    for name, csv_path in benchmarks:
        print(f"\nLoading cluster coverage scores for [{name}]: {csv_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"Coverage CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, usecols=[MANIFEST_IDX_COL, args.score_col, args.hits_col])
        chosen = _select_for_target(
            df=df,
            budget=per_bench_budget,
            score_col=args.score_col,
            hits_col=args.hits_col,
            shortlist_count=int(args.shortlist_count),
            alpha=float(args.alpha),
        )
        selected_per_bench[name] = chosen
        all_selected.update(chosen)
        print(f"  shortlisted {min(int(args.shortlist_count), len(df))} -> selected {len(chosen)}")

    selected = sorted(all_selected)
    print(f"\nUnion: {len(selected)} unique pairs selected")
    overlap_total = sum(len(v) for v in selected_per_bench.values()) - len(selected)
    if overlap_total > 0:
        print(f"  ({overlap_total} pairs appeared in multiple benchmarks and were deduplicated)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(selected, f)
    print(f"Saved {len(selected)} pair indices -> {args.output}")

    source_counts = _summarize_selected_sources(args.manifest_path, selected)
    if source_counts:
        summary_path = args.output.with_name(args.output.stem + "_source_counts.json")
        summary = {
            "manifest_path": str(args.manifest_path),
            "subset_path": str(args.output),
            "selected_count": len(selected),
            "source_counts": source_counts,
            "shortlist_count": int(args.shortlist_count),
            "alpha": float(args.alpha),
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved source-count summary -> {summary_path}")


if __name__ == "__main__":
    main()
