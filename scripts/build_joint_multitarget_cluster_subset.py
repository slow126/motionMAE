#!/usr/bin/env python3
"""Build a multi-target subset via joint greedy selection across all benchmarks.

Instead of selecting per-benchmark and then merging (with or without dedup),
this script runs a single greedy pass where each sample's marginal gain is the
sum of its gains across ALL target benchmarks:

  gain(sample) = sum_{b in benchmarks} w_b * sum_{c in hits_b(sample)}
                    1 / (1 + cover_count_b[c])^alpha

Each benchmark maintains its own independent cover counts, so covering centroid
5 in KITTI doesn't affect centroid 5 in PF-PASCAL.

Advantages over per-benchmark-then-merge:
  - Budget is exactly N (no overlap ambiguity)
  - No benchmark ordering effects
  - Samples that help multiple benchmarks are explicitly rewarded
  - Source allocation emerges from the coverage geometry
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
    p = argparse.ArgumentParser(description="Build multi-target subset via joint greedy selection")
    p.add_argument("--manifest-path", type=Path, required=True)
    p.add_argument(
        "--scores",
        nargs="+",
        required=True,
        metavar="NAME:PATH[:TARGET_N_VALID[:WEIGHT]]",
        help=(
            "Per-benchmark cluster coverage scoring CSVs. Format: "
            "name:path[:target_n_valid[:weight]]. "
            "weight defaults to 1.0 (equal across benchmarks)."
        ),
    )
    p.add_argument("--fraction", type=float, default=0.05)
    p.add_argument("--score-col", type=str, default=DEFAULT_SCORE_COL)
    p.add_argument("--hits-col", type=str, default=DEFAULT_HITS_COL)
    p.add_argument(
        "--shortlist-count",
        type=int,
        default=0,
        help="Per-benchmark shortlist size (0 = no shortlist, use all candidates).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Diminishing-returns strength for repeated centroid coverage (default: 1.0).",
    )
    p.add_argument(
        "--normalize-by-n-valid",
        action="store_true",
        default=True,
        help="Normalize each benchmark's gain contribution by n_valid (default: True).",
    )
    p.add_argument("--no-normalize-by-n-valid", dest="normalize_by_n_valid", action="store_false")
    p.add_argument(
        "--normalize-by-n-centroids",
        action="store_true",
        default=False,
        help="Normalize each benchmark's gain by its total centroid count to equalize scale.",
    )
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def _parse_scores_arg(raw: List[str]) -> List[Tuple[str, Path, int, float]]:
    """Parse 'name:path[:target_n_valid[:weight]]' entries."""
    result = []
    for item in raw:
        parts = item.split(":")
        if len(parts) < 2 or len(parts) > 4:
            raise ValueError(
                f"--scores entries must be 'name:path[:target_n_valid[:weight]]', got: {item!r}"
            )
        name = parts[0].strip()
        path = Path(parts[1].strip())
        target_n_valid = int(parts[2].strip()) if len(parts) >= 3 else 0
        weight = float(parts[3].strip()) if len(parts) >= 4 else 1.0
        result.append((name, path, target_n_valid, weight))
    return result


def _count_manifest(path: Path) -> int:
    count = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _parse_hits(raw: object) -> np.ndarray:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.zeros((0,), dtype=np.int32)
    text = str(raw).strip()
    if not text:
        return np.zeros((0,), dtype=np.int32)
    return np.asarray([int(float(x)) for x in text.split()], dtype=np.int32)


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


def main() -> None:
    args = parse_args()
    benchmarks = _parse_scores_arg(args.scores)
    n_benchmarks = len(benchmarks)
    if n_benchmarks == 0:
        raise ValueError("At least one --scores entry required.")

    print(f"Counting pairs in manifest: {args.manifest_path}")
    total = _count_manifest(args.manifest_path)
    budget = max(1, int(round(args.fraction * total)))
    print(f"  {total} pairs total, budget = {budget}")

    # ── Load and merge per-benchmark data ──────────────────────────────────
    # For each manifest_idx we need: per-benchmark hits, per-benchmark n_valid,
    # and a tiebreak score (mean of per-benchmark distance scores).
    #
    # We index everything by manifest_idx.

    # Per-benchmark data keyed by manifest_idx
    bench_hits: Dict[str, Dict[int, np.ndarray]] = {}       # bench -> {midx -> hits}
    bench_n_valid: Dict[str, Dict[int, int]] = {}            # bench -> {midx -> n_valid}
    bench_n_centroids: Dict[str, int] = {}                   # bench -> total centroids in cluster index
    bench_weights: Dict[str, float] = {}
    bench_ceilings: Dict[str, int] = {}
    all_candidates: set = set()
    tiebreak_scores: Dict[int, float] = {}                   # midx -> mean distance score
    source_dataset: Dict[int, str] = {}

    for name, csv_path, target_n_valid, weight in benchmarks:
        if not csv_path.exists():
            raise FileNotFoundError(f"Coverage CSV not found: {csv_path}")

        print(f"Loading [{name}]: {csv_path}  (weight={weight})")
        cols = [MANIFEST_IDX_COL, args.score_col, args.hits_col, "n_valid"]
        if "source_dataset" not in cols:
            cols.append("source_dataset")
        df = pd.read_csv(csv_path, usecols=cols)
        df = df.dropna(subset=[args.score_col])

        if args.shortlist_count > 0:
            df = df.sort_values(args.score_col, ascending=True).head(args.shortlist_count)
            print(f"  Shortlisted to {len(df)}")

        hits_dict: Dict[int, np.ndarray] = {}
        nv_dict: Dict[int, int] = {}
        max_centroid = 0

        for _, row in df.iterrows():
            midx = int(row[MANIFEST_IDX_COL])
            hits = _parse_hits(row[args.hits_col])
            hits_dict[midx] = hits
            nv_dict[midx] = int(row["n_valid"]) if not pd.isna(row["n_valid"]) else 0
            if hits.size > 0:
                max_centroid = max(max_centroid, int(hits.max()))
            all_candidates.add(midx)

            # Accumulate tiebreak score (mean across benchmarks)
            score = float(row[args.score_col])
            if midx in tiebreak_scores:
                tiebreak_scores[midx] = (tiebreak_scores[midx] + score) / 2.0
            else:
                tiebreak_scores[midx] = score

            if "source_dataset" in df.columns and midx not in source_dataset:
                source_dataset[midx] = str(row["source_dataset"])

        bench_hits[name] = hits_dict
        bench_n_valid[name] = nv_dict
        bench_n_centroids[name] = max_centroid + 1
        bench_weights[name] = weight
        bench_ceilings[name] = target_n_valid
        print(f"  {len(hits_dict)} candidates, ~{max_centroid + 1} centroids")

    candidates = sorted(all_candidates)
    n = len(candidates)
    budget = min(budget, n)
    print(f"\nTotal unique candidates across all benchmarks: {n}")
    print(f"Budget: {budget}")

    # ── Build index mapping ────────────────────────────────────────────────
    midx_to_local = {midx: i for i, midx in enumerate(candidates)}
    bench_names = [name for name, _, _, _ in benchmarks]

    # ── Joint marginal gain function ───────────────────────────────────────
    def joint_gain(
        midx: int,
        cover_counts: Dict[str, Dict[int, int]],
    ) -> float:
        total_gain = 0.0
        for bname in bench_names:
            hits = bench_hits[bname].get(midx)
            if hits is None or hits.size == 0:
                continue
            bg = 0.0
            cc = cover_counts[bname]
            for cid in hits.tolist():
                bg += 1.0 / float((1 + cc.get(cid, 0)) ** args.alpha)
            # Normalize by n_valid
            if args.normalize_by_n_valid:
                nv = bench_n_valid[bname].get(midx, 0)
                ceiling = bench_ceilings[bname]
                if nv > 0:
                    denom = min(nv, ceiling) if ceiling > 0 else nv
                    bg /= float(denom)
            # Normalize by centroid count to equalize benchmark scale
            if args.normalize_by_n_centroids:
                nc = bench_n_centroids[bname]
                if nc > 0:
                    bg /= float(nc)
            # Apply benchmark weight
            bg *= bench_weights[bname]
            total_gain += bg
        return total_gain

    # ── Greedy selection with lazy heap ────────────────────────────────────
    cover_counts: Dict[str, Dict[int, int]] = {bname: {} for bname in bench_names}
    selected_set = set()
    selected_list: List[int] = []
    source_selected: Dict[str, int] = {}

    print("Building initial heap...")
    heap: List[Tuple[float, float, int]] = []
    for midx in candidates:
        g = joint_gain(midx, cover_counts)
        tb = tiebreak_scores.get(midx, 999.0)
        heapq.heappush(heap, (-g, tb, midx))

    print(f"Heap ready, selecting {budget} samples...")
    while len(selected_list) < budget and heap:
        neg_gain, tb, midx = heapq.heappop(heap)
        if midx in selected_set:
            continue
        # Lazy re-evaluation
        actual_gain = joint_gain(midx, cover_counts)
        if heap and (-actual_gain, tb, midx) > heap[0]:
            heapq.heappush(heap, (-actual_gain, tb, midx))
            continue

        # Select this sample
        selected_set.add(midx)
        selected_list.append(midx)
        src = source_dataset.get(midx, "unknown")
        source_selected[src] = source_selected.get(src, 0) + 1

        # Update per-benchmark cover counts
        for bname in bench_names:
            hits = bench_hits[bname].get(midx)
            if hits is not None:
                cc = cover_counts[bname]
                for cid in hits.tolist():
                    cc[cid] = cc.get(cid, 0) + 1

        if len(selected_list) % 1000 == 0:
            src_str = json.dumps(source_selected)
            # Compute per-benchmark coverage
            cov_str = ", ".join(
                f"{bname}={len(cover_counts[bname])}/{bench_n_centroids[bname]}"
                for bname in bench_names
            )
            print(f"  [joint] {len(selected_list)}/{budget}  sources={src_str}  coverage=[{cov_str}]")

    # Fallback fill (shouldn't happen with enough candidates)
    if len(selected_list) < budget:
        for midx in candidates:
            if len(selected_list) >= budget:
                break
            if midx not in selected_set:
                selected_set.add(midx)
                selected_list.append(midx)

    selected = sorted(selected_list)
    print(f"\nSelected {len(selected)} samples")
    print(f"Source breakdown: {json.dumps(source_selected, indent=2)}")

    # Per-benchmark coverage stats
    print("\nPer-benchmark centroid coverage:")
    for bname in bench_names:
        n_covered = len(cover_counts[bname])
        n_total = bench_n_centroids[bname]
        print(f"  {bname}: {n_covered}/{n_total} centroids covered ({100*n_covered/n_total:.1f}%)")

    # ── Save outputs ──────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(selected, f)
    print(f"\nSaved {len(selected)} pair indices -> {args.output}")

    # Source count summary
    manifest_source_counts = _summarize_manifest_sources(args.manifest_path)
    summary_path = args.output.with_name(args.output.stem + "_source_counts.json")
    summary = {
        "manifest_path": str(args.manifest_path),
        "subset_path": str(args.output),
        "method": "joint_multitarget_clustercov",
        "selected_count": len(selected),
        "manifest_count": total,
        "fraction": args.fraction,
        "budget": budget,
        "alpha": args.alpha,
        "normalize_by_n_valid": args.normalize_by_n_valid,
        "normalize_by_n_centroids": args.normalize_by_n_centroids,
        "shortlist_count": args.shortlist_count,
        "benchmarks": {
            name: {
                "weight": weight,
                "target_n_valid_ceiling": tnv,
                "candidates": len(bench_hits[name]),
                "centroids_covered": len(cover_counts[name]),
                "centroids_total": bench_n_centroids[name],
            }
            for name, _, tnv, weight in benchmarks
        },
        "source_counts": source_selected,
        "manifest_source_counts": manifest_source_counts,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
