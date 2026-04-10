#!/usr/bin/env python3
"""Build per-source homogeneous ablation subsets (clustercov + random).

For each training source (pointodyssey, spair, pfpascal), filters the existing
pooled scoring CSVs to that source's rows only, then runs:
  1. Multi-target clustercov greedy selection at each requested budget fraction
  2. Random sampling at the same budget fractions

Budget fractions are relative to each source's pool size, not the total manifest.

Reuses the greedy selection logic from build_multitarget_cluster_subset.py.
"""

from __future__ import annotations

import argparse
import heapq
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


MANIFEST_IDX_COL = "manifest_idx"
DEFAULT_SCORE_COL = "target_cluster_mean_min_dist"
DEFAULT_HITS_COL = "covered_centroids"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build per-source homogeneous ablation subsets (clustercov + random)"
    )
    p.add_argument("--manifest-path", type=Path, required=True, help="Pooled manifest JSONL.")
    p.add_argument(
        "--scores",
        nargs="+",
        required=True,
        metavar="NAME:PATH[:TARGET_N_VALID]",
        help="Per-benchmark cluster coverage scoring CSVs (same format as build_multitarget_cluster_subset.py).",
    )
    p.add_argument(
        "--fractions",
        type=str,
        default="0.005,0.01,0.02,0.05,0.10",
        help="Comma-separated budget fractions relative to each source pool (default: 0.005,0.01,0.02,0.05,0.10).",
    )
    p.add_argument(
        "--source-fractions",
        nargs="*",
        metavar="SOURCE=FRACS",
        default=None,
        help="Per-source fraction overrides, e.g. 'spair=0.02,0.05,0.10,0.25,0.50 pfpascal=0.10,0.25,0.50,0.75,1.0'. "
             "Overrides --fractions for the specified sources.",
    )
    p.add_argument(
        "--sources",
        type=str,
        default="pointodyssey,spair,pfpascal",
        help="Comma-separated source datasets to process (default: pointodyssey,spair,pfpascal).",
    )
    p.add_argument("--score-col", type=str, default=DEFAULT_SCORE_COL)
    p.add_argument("--hits-col", type=str, default=DEFAULT_HITS_COL)
    p.add_argument(
        "--shortlist-count",
        type=int,
        default=0,
        help="Per-target shortlist size (0 = no shortlist, use all source rows).",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument(
        "--normalize-by-n-valid",
        action="store_true",
        default=True,
    )
    p.add_argument("--no-normalize-by-n-valid", dest="normalize_by_n_valid", action="store_false")
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument(
        "--min-budget",
        type=int,
        default=50,
        help="Skip source/fraction combos where budget < this (default: 50).",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _parse_scores_arg(raw: List[str]) -> List[Tuple[str, Path, int]]:
    result = []
    for item in raw:
        parts = item.split(":")
        if len(parts) == 2:
            result.append((parts[0].strip(), Path(parts[1].strip()), 0))
        elif len(parts) == 3:
            result.append((parts[0].strip(), Path(parts[1].strip()), int(parts[2].strip())))
        else:
            raise ValueError(f"--scores entries must be 'name:path[:target_n_valid]', got: {item!r}")
    return result


def _parse_hits(raw: object) -> np.ndarray:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.zeros((0,), dtype=np.int32)
    text = str(raw).strip()
    if not text:
        return np.zeros((0,), dtype=np.int32)
    return np.asarray([int(float(x)) for x in text.split()], dtype=np.int32)


def _marginal_gain(
    hits: np.ndarray, cover_counts: Dict[int, int], alpha: float,
    n_valid: int = 0, n_valid_ceiling: int = 0,
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


def _greedy_select(
    df: pd.DataFrame,
    budget: int,
    score_col: str,
    hits_col: str,
    shortlist_count: int,
    alpha: float,
    normalize_by_n_valid: bool,
    n_valid_ceiling: int = 0,
) -> List[int]:
    """Greedy coverage-maximizing selection on a single-source filtered DataFrame."""
    df = df.dropna(subset=[score_col]).copy()
    df = df.sort_values(score_col, ascending=True)
    if shortlist_count > 0:
        df = df.head(min(shortlist_count, len(df)))
    if len(df) == 0 or budget <= 0:
        return []

    manifest_idx = df[MANIFEST_IDX_COL].astype(int).to_numpy()
    hits_list = [_parse_hits(x) for x in df[hits_col].tolist()]
    n = len(df)
    budget = min(budget, n)

    if normalize_by_n_valid and "n_valid" in df.columns:
        n_valid_arr = df["n_valid"].fillna(0).astype(int).to_numpy()
    else:
        n_valid_arr = np.zeros((n,), dtype=int)

    cover_counts: Dict[int, int] = {}
    selected_local = np.zeros((n,), dtype=bool)
    selected_manifest: List[int] = []
    heap: List[Tuple[float, float, int]] = []

    for local_idx in range(n):
        init_gain = _marginal_gain(
            hits_list[local_idx], cover_counts, alpha,
            n_valid=int(n_valid_arr[local_idx]), n_valid_ceiling=n_valid_ceiling,
        )
        heapq.heappush(heap, (-init_gain, float(df.iloc[local_idx][score_col]), local_idx))

    while len(selected_manifest) < budget and heap:
        neg_gain, base_score, local_idx = heapq.heappop(heap)
        if selected_local[local_idx]:
            continue
        actual_gain = _marginal_gain(
            hits_list[local_idx], cover_counts, alpha,
            n_valid=int(n_valid_arr[local_idx]), n_valid_ceiling=n_valid_ceiling,
        )
        if heap and (-actual_gain, base_score, local_idx) > heap[0]:
            heapq.heappush(heap, (-actual_gain, base_score, local_idx))
            continue
        selected_local[local_idx] = True
        selected_manifest.append(int(manifest_idx[local_idx]))
        for cid in hits_list[local_idx].tolist():
            cover_counts[int(cid)] = cover_counts.get(int(cid), 0) + 1
        if len(selected_manifest) % 1000 == 0:
            print(f"    [greedy] {len(selected_manifest)}/{budget}")

    # Fill remaining budget with best-distance fallback
    if len(selected_manifest) < budget:
        for local_idx in range(n):
            if len(selected_manifest) >= budget:
                break
            if selected_local[local_idx]:
                continue
            selected_local[local_idx] = True
            selected_manifest.append(int(manifest_idx[local_idx]))

    return selected_manifest


def _multitarget_clustercov(
    source_indices: set,
    benchmarks: List[Tuple[str, Path, int]],
    budget: int,
    score_col: str,
    hits_col: str,
    shortlist_count: int,
    alpha: float,
    normalize_by_n_valid: bool,
) -> Tuple[List[int], Dict[str, List[int]]]:
    """Run per-benchmark greedy selection, then union. Returns (union, per_bench)."""
    n_benchmarks = len(benchmarks)
    per_bench_budget = max(1, budget // n_benchmarks)
    per_bench: Dict[str, List[int]] = {}
    all_selected: set = set()

    for name, csv_path, target_n_valid in benchmarks:
        df = pd.read_csv(csv_path, usecols=[MANIFEST_IDX_COL, score_col, hits_col, "n_valid", "source_dataset"])
        # Filter to source
        df = df[df[MANIFEST_IDX_COL].isin(source_indices)]
        print(f"    [{name}] {len(df)} source rows, budget {per_bench_budget}")
        chosen = _greedy_select(
            df=df,
            budget=per_bench_budget,
            score_col=score_col,
            hits_col=hits_col,
            shortlist_count=shortlist_count,
            alpha=alpha,
            normalize_by_n_valid=normalize_by_n_valid,
            n_valid_ceiling=target_n_valid,
        )
        per_bench[name] = chosen
        all_selected.update(chosen)

    return sorted(all_selected), per_bench


def _random_select(source_indices: List[int], budget: int, seed: int) -> List[int]:
    budget = min(budget, len(source_indices))
    if budget <= 0:
        return []
    if budget == len(source_indices):
        return sorted(source_indices)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(source_indices, dtype=np.int64), size=budget, replace=False)
    return sorted(int(x) for x in chosen.tolist())


def fraction_label(frac: float) -> str:
    pct = frac * 100.0
    rounded = round(pct)
    if abs(pct - rounded) < 1e-9 and rounded == int(rounded):
        return f"{int(rounded)}pct"
    return f"{pct:.2f}".rstrip("0").rstrip(".").replace(".", "p") + "pct"


def main() -> None:
    args = parse_args()
    benchmarks = _parse_scores_arg(args.scores)
    default_fractions = [float(x.strip()) for x in args.fractions.split(",")]
    sources = [s.strip().lower() for s in args.sources.split(",")]

    # Parse per-source fraction overrides
    source_fractions: Dict[str, List[float]] = {}
    if args.source_fractions:
        for entry in args.source_fractions:
            src_name, frac_str = entry.split("=", 1)
            source_fractions[src_name.strip().lower()] = [
                float(x.strip()) for x in frac_str.split(",")
            ]

    # Index manifest by source
    print(f"Indexing manifest by source: {args.manifest_path}")
    source_indices: Dict[str, List[int]] = defaultdict(list)
    total = 0
    with args.manifest_path.open("r") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            src = str(row.get("source_dataset", "pointodyssey")).strip().lower()
            source_indices[src].append(idx)
            total += 1
    print(f"  Total: {total}")
    for src in sorted(source_indices):
        print(f"  {src}: {len(source_indices[src])}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict = {
        "manifest_path": str(args.manifest_path),
        "manifest_total": total,
        "source_pool_sizes": {s: len(source_indices.get(s, [])) for s in sources},
        "fractions_default": default_fractions,
        "fractions_per_source": {s: source_fractions.get(s, default_fractions) for s in sources},
        "sources": sources,
        "benchmarks": [name for name, _, _ in benchmarks],
        "runs": [],
    }

    for source in sources:
        src_indices = source_indices.get(source, [])
        if not src_indices:
            print(f"\nSkipping {source}: no rows in manifest")
            continue
        pool_size = len(src_indices)
        src_index_set = set(src_indices)
        src_dir = args.output_dir / source
        src_dir.mkdir(parents=True, exist_ok=True)

        fractions = source_fractions.get(source, default_fractions)
        print(f"\n  Fractions for {source}: {fractions}")

        for frac in fractions:
            budget = max(1, int(round(pool_size * frac)))
            label = fraction_label(frac)

            if budget < args.min_budget:
                print(f"\nSkipping {source} @ {label}: budget {budget} < min_budget {args.min_budget}")
                continue

            print(f"\n{'='*60}")
            print(f"Source: {source} | Fraction: {label} | Budget: {budget} / {pool_size}")
            print(f"{'='*60}")

            # --- Clustercov ---
            print(f"  Running clustercov selection...")
            clustercov_subset, clustercov_per_bench = _multitarget_clustercov(
                source_indices=src_index_set,
                benchmarks=benchmarks,
                budget=budget,
                score_col=args.score_col,
                hits_col=args.hits_col,
                shortlist_count=args.shortlist_count,
                alpha=args.alpha,
                normalize_by_n_valid=args.normalize_by_n_valid,
            )
            cc_path = src_dir / f"subset_clustercov_{label}.json"
            cc_path.write_text(json.dumps(clustercov_subset))
            cc_bench_path = src_dir / f"subset_clustercov_{label}_per_benchmark.json"
            cc_bench_path.write_text(json.dumps(clustercov_per_bench))
            print(f"  Clustercov: {len(clustercov_subset)} selected -> {cc_path}")

            # --- Random ---
            print(f"  Running random selection...")
            random_subset = _random_select(src_indices, budget, seed=args.seed)
            rand_path = src_dir / f"subset_random_{label}_seed{args.seed}.json"
            rand_path.write_text(json.dumps(random_subset))
            print(f"  Random: {len(random_subset)} selected -> {rand_path}")

            summary["runs"].append({
                "source": source,
                "fraction": frac,
                "fraction_label": label,
                "pool_size": pool_size,
                "budget": budget,
                "clustercov_count": len(clustercov_subset),
                "clustercov_path": str(cc_path),
                "random_count": len(random_subset),
                "random_path": str(rand_path),
            })

    summary_path = args.output_dir / "homogeneous_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary -> {summary_path}")
    print(f"Total runs: {len(summary['runs'])}")


if __name__ == "__main__":
    main()
