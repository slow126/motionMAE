#!/usr/bin/env python3
"""Summarize subset composition and optional per-benchmark multitarget selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from build_multitarget_cluster_subset import (
    DEFAULT_HITS_COL as CLUSTER_DEFAULT_HITS_COL,
    DEFAULT_SCORE_COL as CLUSTER_DEFAULT_SCORE_COL,
    MANIFEST_IDX_COL,
    _parse_scores_arg,
    _select_for_target,
)
from build_multitarget_nn_subset import DEFAULT_SCORE_COL as NN_DEFAULT_SCORE_COL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Summarize a subset JSON against a manifest, optionally reconstructing "
            "per-benchmark multitarget selection from score CSVs."
        )
    )
    p.add_argument("--manifest-path", type=Path, required=True, help="Manifest JSONL path.")
    p.add_argument("--subset-path", type=Path, required=True, help="Subset JSON path.")
    p.add_argument(
        "--scores",
        nargs="+",
        default=None,
        metavar="NAME:PATH",
        help="Optional per-benchmark score CSVs in NAME:PATH format.",
    )
    p.add_argument(
        "--selection-mode",
        choices=["auto", "cluster", "nn"],
        default="auto",
        help="How to reconstruct per-benchmark selection when --scores is provided.",
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Total selection fraction used when building the subset.",
    )
    p.add_argument(
        "--total-budget",
        type=int,
        default=None,
        help="Total selection budget used when building the subset.",
    )
    p.add_argument(
        "--per-bench-budget",
        type=int,
        default=None,
        help="Directly specify the per-benchmark budget used when building the subset.",
    )
    p.add_argument(
        "--score-col",
        type=str,
        default=None,
        help="Optional score column override for --scores CSVs.",
    )
    p.add_argument(
        "--hits-col",
        type=str,
        default=CLUSTER_DEFAULT_HITS_COL,
        help=f"Cluster mode hits column (default: {CLUSTER_DEFAULT_HITS_COL}).",
    )
    p.add_argument(
        "--shortlist-count",
        type=int,
        default=200000,
        help="Cluster mode shortlist size (default: 200000).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Cluster mode diminishing-returns alpha (default: 1.0).",
    )
    p.add_argument(
        "--normalize-by-n-valid",
        action="store_true",
        default=True,
        help="Cluster mode: normalize marginal gain by n_valid (default: True).",
    )
    p.add_argument("--no-normalize-by-n-valid", dest="normalize_by_n_valid", action="store_false")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the full summary.",
    )
    return p.parse_args()


def load_subset_indices(path: Path) -> tuple[list[int], dict]:
    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        return sorted(int(x) for x in obj), {}
    if isinstance(obj, dict):
        if "indices" not in obj:
            raise ValueError(f"Unsupported subset JSON object format in {path}")
        meta = {k: v for k, v in obj.items() if k != "indices"}
        return sorted(int(x) for x in obj["indices"]), meta
    raise ValueError(f"Unsupported subset JSON format in {path}")


def source_key(row: dict) -> str:
    return str(row.get("source_dataset", "unknown")).strip().lower()


def split_key(row: dict) -> str:
    split = str(row.get("source_split", "")).strip().lower()
    return split or "unknown"


def count_sources(rows: Iterable[dict]) -> dict[str, int]:
    counts = Counter(source_key(row) for row in rows)
    return {k: int(v) for k, v in sorted(counts.items())}


def count_source_splits(rows: Iterable[dict]) -> dict[str, dict[str, int]]:
    nested: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        nested[source_key(row)][split_key(row)] += 1
    return {source: {split: int(count) for split, count in sorted(counter.items())}
            for source, counter in sorted(nested.items())}


def build_source_selection_summary(
    selected_counts: dict[str, int],
    manifest_counts: dict[str, int],
    selected_total: int,
    manifest_total: int,
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    all_sources = sorted(set(manifest_counts) | set(selected_counts))
    for source in all_sources:
        selected = int(selected_counts.get(source, 0))
        total = int(manifest_counts.get(source, 0))
        summary[source] = {
            "selected_count": selected,
            "total_count": total,
            "selected_fraction_of_subset": float(selected) / float(selected_total) if selected_total > 0 else 0.0,
            "selected_fraction_of_source": float(selected) / float(total) if total > 0 else 0.0,
            "source_fraction_of_manifest": float(total) / float(manifest_total) if manifest_total > 0 else 0.0,
        }
    return summary


def determine_mode(args: argparse.Namespace, score_entries: list[tuple[str, Path]]) -> str:
    if args.selection_mode != "auto":
        return args.selection_mode
    sample_path = score_entries[0][1]
    sample_df = pd.read_csv(sample_path, nrows=1)
    if args.hits_col in sample_df.columns:
        return "cluster"
    return "nn"


def determine_budgets(args: argparse.Namespace, manifest_total: int, n_benchmarks: int) -> tuple[int, int]:
    if args.per_bench_budget is not None:
        per_bench_budget = max(1, int(args.per_bench_budget))
        return per_bench_budget * n_benchmarks, per_bench_budget
    if args.total_budget is not None:
        total_budget = max(1, int(args.total_budget))
        return total_budget, max(1, total_budget // n_benchmarks)
    if args.fraction is not None:
        total_budget = max(1, int(round(float(args.fraction) * manifest_total)))
        return total_budget, max(1, total_budget // n_benchmarks)
    raise ValueError("When using --scores, provide one of --fraction, --total-budget, or --per-bench-budget.")


def summarize_rows(rows: list[dict]) -> dict[str, object]:
    source_counts = count_sources(rows)
    return {
        "count": int(len(rows)),
        "source_counts": source_counts,
        "source_split_counts": count_source_splits(rows),
    }


def scan_manifest(
    manifest_path: Path,
    wanted_indices: set[int],
) -> tuple[int, dict[str, int], dict[str, dict[str, int]], dict[str, int], dict[str, dict[str, int]]]:
    manifest_source_counter: Counter = Counter()
    subset_source_counter: Counter = Counter()
    manifest_split_counter: dict[str, Counter] = defaultdict(Counter)
    subset_split_counter: dict[str, Counter] = defaultdict(Counter)
    total = 0

    with manifest_path.open("r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source = source_key(row)
            split = split_key(row)
            manifest_source_counter[source] += 1
            manifest_split_counter[source][split] += 1
            total += 1

            if idx in wanted_indices:
                subset_source_counter[source] += 1
                subset_split_counter[source][split] += 1

    manifest_source_counts = {k: int(v) for k, v in sorted(manifest_source_counter.items())}
    subset_source_counts = {k: int(v) for k, v in sorted(subset_source_counter.items())}
    manifest_source_split_counts = {
        source: {split: int(count) for split, count in sorted(counter.items())}
        for source, counter in sorted(manifest_split_counter.items())
    }
    subset_source_split_counts = {
        source: {split: int(count) for split, count in sorted(counter.items())}
        for source, counter in sorted(subset_split_counter.items())
    }
    return total, manifest_source_counts, manifest_source_split_counts, subset_source_counts, subset_source_split_counts


def reconstruct_cluster_selection(
    csv_path: Path,
    score_col: str,
    hits_col: str,
    shortlist_count: int,
    alpha: float,
    budget: int,
    normalize_by_n_valid: bool,
) -> tuple[list[int], dict[str, float], dict[str, int], dict[str, dict[str, int]]]:
    df = pd.read_csv(csv_path, usecols=[MANIFEST_IDX_COL, score_col, hits_col, "n_valid", "source_dataset", "source_split"])
    chosen = _select_for_target(
        df=df,
        budget=budget,
        score_col=score_col,
        hits_col=hits_col,
        shortlist_count=shortlist_count,
        alpha=alpha,
        normalize_by_n_valid=normalize_by_n_valid,
    )
    chosen_df = df[df[MANIFEST_IDX_COL].isin(chosen)].copy()
    score_stats = {}
    if not chosen_df.empty:
        values = chosen_df[score_col].astype(float)
        score_stats = {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
        }
    rows = chosen_df.to_dict(orient="records")
    return chosen, score_stats, count_sources(rows), count_source_splits(rows)


def reconstruct_nn_selection(
    csv_path: Path,
    score_col: str,
    budget: int,
) -> tuple[list[int], dict[str, float], dict[str, int], dict[str, dict[str, int]]]:
    df = pd.read_csv(csv_path, usecols=[MANIFEST_IDX_COL, score_col, "source_dataset", "source_split"])
    df = df.dropna(subset=[score_col]).sort_values(score_col, ascending=True)
    budget = min(int(budget), len(df))
    chosen_df = df.head(budget).copy()
    chosen = chosen_df[MANIFEST_IDX_COL].astype(int).tolist()
    score_stats = {}
    if not chosen_df.empty:
        values = chosen_df[score_col].astype(float)
        score_stats = {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
        }
    rows = chosen_df.to_dict(orient="records")
    return chosen, score_stats, count_sources(rows), count_source_splits(rows)


def build_overlap_matrix(selected_per_bench: dict[str, set[int]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for left_name, left_set in selected_per_bench.items():
        out[left_name] = {}
        for right_name, right_set in selected_per_bench.items():
            out[left_name][right_name] = int(len(left_set & right_set))
    return out


def summarize_multitarget_selection(
    args: argparse.Namespace,
    manifest_total: int,
    subset_indices: list[int],
) -> dict[str, object]:
    score_entries = _parse_scores_arg(list(args.scores or []))
    mode = determine_mode(args, score_entries)
    total_budget, per_bench_budget = determine_budgets(args, manifest_total, len(score_entries))
    subset_set = set(int(x) for x in subset_indices)

    score_col = args.score_col or (CLUSTER_DEFAULT_SCORE_COL if mode == "cluster" else NN_DEFAULT_SCORE_COL)
    selected_per_bench_lists: dict[str, list[int]] = {}
    score_stats_by_bench: dict[str, dict[str, float]] = {}
    source_counts_by_bench: dict[str, dict[str, int]] = {}
    source_splits_by_bench: dict[str, dict[str, dict[str, int]]] = {}

    for name, csv_path in score_entries:
        if mode == "cluster":
            chosen, score_stats, source_counts, source_splits = reconstruct_cluster_selection(
                csv_path=csv_path,
                score_col=score_col,
                hits_col=args.hits_col,
                shortlist_count=int(args.shortlist_count),
                alpha=float(args.alpha),
                budget=per_bench_budget,
                normalize_by_n_valid=bool(args.normalize_by_n_valid),
            )
        else:
            chosen, score_stats, source_counts, source_splits = reconstruct_nn_selection(
                csv_path=csv_path,
                score_col=score_col,
                budget=per_bench_budget,
            )
        selected_per_bench_lists[name] = [int(x) for x in chosen]
        score_stats_by_bench[name] = score_stats
        source_counts_by_bench[name] = source_counts
        source_splits_by_bench[name] = source_splits

    selected_per_bench_sets = {name: set(vals) for name, vals in selected_per_bench_lists.items()}
    reconstructed_union = set().union(*selected_per_bench_sets.values()) if selected_per_bench_sets else set()

    per_bench_summary: dict[str, dict[str, object]] = {}
    for name, chosen_set in selected_per_bench_sets.items():
        other_union = set().union(*(s for other_name, s in selected_per_bench_sets.items() if other_name != name))
        unique_set = chosen_set - other_union
        shared_set = chosen_set & other_union
        overlap_subset = chosen_set & subset_set
        per_bench_summary[name] = {
            "budget": int(per_bench_budget),
            "chosen_count": int(len(chosen_set)),
            "unique_to_reconstructed_union_count": int(len(unique_set)),
            "shared_with_other_benchmarks_count": int(len(shared_set)),
            "overlap_with_subset_count": int(len(overlap_subset)),
            "source_counts": source_counts_by_bench.get(name, {}),
            "source_split_counts": source_splits_by_bench.get(name, {}),
            "score_stats": score_stats_by_bench.get(name, {}),
        }

    overlap_total = sum(len(v) for v in selected_per_bench_lists.values()) - len(reconstructed_union)
    missing_from_subset = sorted(int(x) for x in (reconstructed_union - subset_set))
    extra_in_subset = sorted(int(x) for x in (subset_set - reconstructed_union))

    return {
        "selection_mode": mode,
        "score_col": score_col,
        "hits_col": args.hits_col if mode == "cluster" else None,
        "benchmark_count": int(len(score_entries)),
        "total_budget": int(total_budget),
        "per_benchmark_budget": int(per_bench_budget),
        "reconstructed_union_count": int(len(reconstructed_union)),
        "deduplicated_overlap_count": int(overlap_total),
        "subset_matches_reconstructed_union": not missing_from_subset and not extra_in_subset,
        "missing_from_subset_count": int(len(missing_from_subset)),
        "extra_in_subset_count": int(len(extra_in_subset)),
        "missing_from_subset_examples": missing_from_subset[:20],
        "extra_in_subset_examples": extra_in_subset[:20],
        "benchmark_overlap_matrix": build_overlap_matrix(selected_per_bench_sets),
        "per_benchmark": per_bench_summary,
    }


def print_text_summary(summary: dict[str, object]) -> None:
    print(f"Subset: {summary['subset_path']}")
    print(f"Manifest: {summary['manifest_path']}")
    print(f"Selected: {summary['subset_count']} / {summary['manifest_count']}")

    print("\nSource counts:")
    source_summary = summary["source_selection_summary"]
    for source, stats in source_summary.items():
        print(
            f"  {source}: selected={stats['selected_count']} "
            f"subset_frac={stats['selected_fraction_of_subset']:.4f} "
            f"source_frac={stats['selected_fraction_of_source']:.4f}"
        )

    print("\nSource split counts:")
    for source, splits in summary["subset_source_split_counts"].items():
        split_text = ", ".join(f"{split}={count}" for split, count in splits.items())
        print(f"  {source}: {split_text}")

    multitarget = summary.get("multitarget_selection")
    if not multitarget:
        return

    print("\nMultitarget reconstruction:")
    print(
        f"  mode={multitarget['selection_mode']} "
        f"benchmarks={multitarget['benchmark_count']} "
        f"budget={multitarget['total_budget']} "
        f"per_bench={multitarget['per_benchmark_budget']}"
    )
    print(
        f"  reconstructed_union={multitarget['reconstructed_union_count']} "
        f"deduplicated_overlap={multitarget['deduplicated_overlap_count']} "
        f"exact_match={multitarget['subset_matches_reconstructed_union']}"
    )

    print("\nPer-benchmark:")
    for name, stats in multitarget["per_benchmark"].items():
        source_text = ", ".join(f"{k}={v}" for k, v in stats["source_counts"].items())
        print(
            f"  {name}: chosen={stats['chosen_count']} "
            f"unique={stats['unique_to_reconstructed_union_count']} "
            f"shared={stats['shared_with_other_benchmarks_count']} "
            f"subset_overlap={stats['overlap_with_subset_count']} "
            f"sources[{source_text}]"
        )


def main() -> None:
    args = parse_args()
    if not args.manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest_path}")
    if not args.subset_path.exists():
        raise FileNotFoundError(f"Subset not found: {args.subset_path}")

    subset_indices, subset_meta = load_subset_indices(args.subset_path)
    subset_wanted = set(int(x) for x in subset_indices)
    (
        manifest_total,
        manifest_source_counts,
        manifest_source_split_counts,
        subset_source_counts,
        subset_source_split_counts,
    ) = scan_manifest(args.manifest_path, subset_wanted)

    summary: dict[str, object] = {
        "manifest_path": str(args.manifest_path),
        "subset_path": str(args.subset_path),
        "subset_metadata": subset_meta,
        "manifest_count": int(manifest_total),
        "subset_count": int(len(subset_indices)),
        "manifest_source_counts": manifest_source_counts,
        "manifest_source_split_counts": manifest_source_split_counts,
        "subset_source_counts": subset_source_counts,
        "subset_source_split_counts": subset_source_split_counts,
        "source_selection_summary": build_source_selection_summary(
            selected_counts=subset_source_counts,
            manifest_counts=manifest_source_counts,
            selected_total=len(subset_indices),
            manifest_total=manifest_total,
        ),
    }

    if args.scores:
        summary["multitarget_selection"] = summarize_multitarget_selection(
            args=args,
            manifest_total=manifest_total,
            subset_indices=subset_indices,
        )

    print_text_summary(summary)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved JSON summary -> {args.output}")


if __name__ == "__main__":
    main()
