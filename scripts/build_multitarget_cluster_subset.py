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
from typing import Dict, List, Optional, Tuple

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
    p.add_argument(
        "--normalize-by-n-valid",
        action="store_true",
        default=True,
        help="Normalize marginal gain by n_valid to remove bias toward samples with more query vectors (default: True).",
    )
    p.add_argument("--no-normalize-by-n-valid", dest="normalize_by_n_valid", action="store_false")
    p.add_argument(
        "--deduplicate-across-benchmarks",
        action="store_true",
        default=False,
        help="Prevent later benchmarks from re-selecting samples already chosen by earlier benchmarks.",
    )
    return p.parse_args()


def _parse_scores_arg(raw: List[str]) -> List[Tuple[str, Path, int]]:
    """Parse 'name:path' or 'name:path:TARGET_N_VALID' entries.

    The optional third field is the target benchmark's average points per pair,
    used as a ceiling for n_valid normalization. 0 means no ceiling (use source
    n_valid directly).
    """
    result = []
    for item in raw:
        if ":" not in item:
            raise ValueError(f"--scores entries must be 'name:path[:target_n_valid]', got: {item!r}")
        parts = item.split(":")
        if len(parts) == 2:
            name, path_str = parts
            target_n_valid = 0
        elif len(parts) == 3:
            name, path_str, n_str = parts
            try:
                target_n_valid = int(n_str.strip())
            except ValueError:
                raise ValueError(f"Third field must be an integer (target avg n_valid), got: {n_str!r}")
        else:
            raise ValueError(f"--scores entries must be 'name:path[:target_n_valid]', got: {item!r}")
        result.append((name.strip(), Path(path_str.strip()), target_n_valid))
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


def _parse_hits(raw: object) -> np.ndarray:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.zeros((0,), dtype=np.int32)
    text = str(raw).strip()
    if not text:
        return np.zeros((0,), dtype=np.int32)
    return np.asarray([int(float(x)) for x in text.split()], dtype=np.int32)


def _marginal_gain(
    hits: np.ndarray, cover_counts: Dict[int, int], alpha: float, n_valid: int = 0, n_valid_ceiling: int = 0,
) -> float:
    if hits.size == 0:
        return 0.0
    gain = 0.0
    for cid in hits.tolist():
        gain += 1.0 / float((1 + int(cover_counts.get(int(cid), 0))) ** alpha)
    if n_valid > 0:
        denom = min(n_valid, n_valid_ceiling) if n_valid_ceiling > 0 else n_valid
        gain /= float(denom)
    return gain


def _select_for_target(
    df: pd.DataFrame,
    budget: int,
    score_col: str,
    hits_col: str,
    shortlist_count: int,
    alpha: float,
    normalize_by_n_valid: bool = True,
    n_valid_ceiling: int = 0,
    exclude: Optional[set] = None,
) -> List[int]:
    df = df.dropna(subset=[score_col]).copy()
    df = df.sort_values(score_col, ascending=True)
    if shortlist_count > 0:
        df = df.head(min(int(shortlist_count), len(df)))
    if len(df) == 0 or budget <= 0:
        return []

    manifest_idx = df[MANIFEST_IDX_COL].astype(int).to_numpy()
    base_scores = df[score_col].astype(float).to_numpy()
    source_arr = df["source_dataset"].fillna("unknown").to_numpy() if "source_dataset" in df.columns else None
    print(f"  [greedy] parsing hits for {len(df)} candidates...")
    hits_list = [_parse_hits(x) for x in df[hits_col].tolist()]
    n = len(df)
    budget = min(int(budget), n)

    n_valid_arr: np.ndarray
    if normalize_by_n_valid and "n_valid" in df.columns:
        n_valid_arr = df["n_valid"].fillna(0).astype(int).to_numpy()
    else:
        n_valid_arr = np.zeros((n,), dtype=int)

    cover_counts: Dict[int, int] = {}
    source_selected: Dict[str, int] = {}
    selected_local = np.zeros((n,), dtype=bool)
    if exclude:
        for local_idx in range(n):
            if int(manifest_idx[local_idx]) in exclude:
                selected_local[local_idx] = True
        n_excluded = int(selected_local.sum())
        if n_excluded > 0:
            print(f"  [greedy] excluded {n_excluded} already-selected samples")
    selected_manifest: List[int] = []
    heap: List[Tuple[float, float, int]] = []

    print(f"  [greedy] building initial heap for {n} candidates...")
    for local_idx in range(n):
        init_gain = _marginal_gain(hits_list[local_idx], cover_counts, alpha, n_valid=int(n_valid_arr[local_idx]), n_valid_ceiling=int(n_valid_ceiling))
        heapq.heappush(heap, (-init_gain, float(base_scores[local_idx]), int(local_idx)))
    print(f"  [greedy] heap ready, selecting {budget} samples...")

    while len(selected_manifest) < budget and heap:
        neg_gain, base_score, local_idx = heapq.heappop(heap)
        if selected_local[local_idx]:
            continue
        actual_gain = _marginal_gain(hits_list[local_idx], cover_counts, alpha, n_valid=int(n_valid_arr[local_idx]), n_valid_ceiling=int(n_valid_ceiling))
        if heap and (-actual_gain, base_score, local_idx) > heap[0]:
            heapq.heappush(heap, (-actual_gain, base_score, local_idx))
            continue

        selected_local[local_idx] = True
        selected_manifest.append(int(manifest_idx[local_idx]))
        if source_arr is not None:
            src = str(source_arr[local_idx])
            source_selected[src] = source_selected.get(src, 0) + 1
        for cid in hits_list[local_idx].tolist():
            cid = int(cid)
            cover_counts[cid] = int(cover_counts.get(cid, 0)) + 1
        if len(selected_manifest) % 1000 == 0:
            src_str = json.dumps(source_selected) if source_selected else ""
            print(f"  [greedy] {len(selected_manifest)}/{budget} selected  {src_str}")

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

    for name, csv_path, target_n_valid in benchmarks:
        if not csv_path.exists():
            raise FileNotFoundError(f"Coverage CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, usecols=[MANIFEST_IDX_COL, args.score_col, args.hits_col, "n_valid", "source_dataset"])
        ceiling_str = f", ceiling={target_n_valid}" if target_n_valid > 0 else ""
        print(f"\nLoading cluster coverage scores for [{name}]: {csv_path}  (normalize={args.normalize_by_n_valid}{ceiling_str})")
        chosen = _select_for_target(
            df=df,
            budget=per_bench_budget,
            score_col=args.score_col,
            hits_col=args.hits_col,
            shortlist_count=int(args.shortlist_count),
            alpha=float(args.alpha),
            normalize_by_n_valid=bool(args.normalize_by_n_valid),
            n_valid_ceiling=int(target_n_valid),
            exclude=all_selected if args.deduplicate_across_benchmarks else None,
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

    # Write per-benchmark selection lists
    bench_detail_path = args.output.with_name(args.output.stem + "_per_benchmark.json")
    bench_detail_path.write_text(json.dumps(
        {name: idxs for name, idxs in selected_per_bench.items()},
    ))
    print(f"Saved per-benchmark selections -> {bench_detail_path}")

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
            "shortlist_count": int(args.shortlist_count),
            "alpha": float(args.alpha),
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved source-count summary -> {summary_path}")


if __name__ == "__main__":
    main()
