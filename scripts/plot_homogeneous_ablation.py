#!/usr/bin/env python3
"""Plot homogeneous source ablation results.

Scans snapshots/ for homogeneous_* directories, extracts validation PCK,
and produces two PNGs:

1. Convergence curves: PCK vs training steps (one subplot per benchmark,
   lines grouped by source+method+budget). Good for checking progress.

2. Efficiency curves: converged PCK vs subset size, per source, with
   clustercov vs random comparison. This is the main result figure.

Designed to be re-run as jobs finish — picks up whatever snapshots exist.

Usage:
    python scripts/plot_homogeneous_ablation.py
    python scripts/plot_homogeneous_ablation.py --snapshots-dir ./snapshots --output-dir analysis/plots
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BENCHMARKS = ["kitti2012", "kitti2015", "pfpascal", "pfwillow", "tss"]

# fraction label -> approximate count for x-axis ordering
FRACTION_ORDER = {
    "0p5pct": 0.005,
    "1pct": 0.01,
    "2pct": 0.02,
    "5pct": 0.05,
    "10pct": 0.10,
}

SOURCE_COLORS = {
    "pointodyssey": "#1f77b4",
    "spair": "#ff7f0e",
    "pfpascal": "#2ca02c",
}

METHOD_STYLES = {
    "clustercov": ("-", "o"),   # solid, circle
    "random": ("--", "x"),      # dashed, x
}


class RunInfo(NamedTuple):
    source: str
    method: str
    fraction_label: str
    fraction: float
    snapshot_dir: Path
    label: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot homogeneous source ablation results")
    p.add_argument(
        "--snapshots-dir",
        type=Path,
        default=Path("snapshots"),
        help="Directory containing snapshot folders (default: snapshots/).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/plots"),
        help="Directory to write PNGs (default: analysis/plots/).",
    )
    p.add_argument(
        "--tail-n",
        type=int,
        default=5,
        help="Number of final evaluations to average for converged PCK (default: 5).",
    )
    p.add_argument(
        "--scope",
        default="step",
        choices=["step", "epoch", "all"],
        help="Validation scope to use (default: step).",
    )
    return p.parse_args()


def discover_runs(snapshots_dir: Path) -> List[RunInfo]:
    """Find all homogeneous_* snapshot dirs and parse their metadata."""
    pattern = re.compile(
        r"^homogeneous_(?P<source>pointodyssey|spair|pfpascal)"
        r"_(?P<method>clustercov|random)"
        r"_(?P<frac>\d+p?\d*pct)"
        r"_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}$"
    )
    runs = []
    if not snapshots_dir.exists():
        return runs

    for d in sorted(snapshots_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        csv_path = d / "validation_results.csv"
        if not csv_path.exists():
            continue
        source = m.group("source")
        method = m.group("method")
        frac_label = m.group("frac")
        frac = FRACTION_ORDER.get(frac_label, 0.0)
        if frac == 0.0:
            continue
        label = f"{source}_{method}_{frac_label}"
        runs.append(RunInfo(source, method, frac_label, frac, d, label))

    return runs


def load_pck(snapshot_dir: Path, scope: str) -> Optional[pd.DataFrame]:
    """Load validation_results.csv and filter to PCK rows."""
    csv_path = snapshot_dir / "validation_results.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "pck" not in df.columns or "benchmark" not in df.columns:
        return None
    df["training_steps"] = pd.to_numeric(df["training_steps"], errors="coerce")
    df["pck"] = pd.to_numeric(df["pck"], errors="coerce")
    if scope != "all" and "validation_scope" in df.columns:
        df = df[df["validation_scope"] == scope]
    return df.dropna(subset=["training_steps", "pck"])


def get_converged_pck(df: pd.DataFrame, benchmark: str, tail_n: int) -> Optional[float]:
    """Get median of last tail_n PCK values for a benchmark."""
    bench_df = df[df["benchmark"] == benchmark].sort_values("training_steps")
    if len(bench_df) < 2:
        return None
    tail = bench_df.tail(tail_n)
    return float(tail["pck"].median())


def plot_convergence(runs: List[RunInfo], args: argparse.Namespace) -> Path:
    """Plot PCK vs training steps: one subplot per benchmark, all runs overlaid."""
    n_bench = len(BENCHMARKS)
    cols = min(3, n_bench)
    rows = math.ceil(n_bench / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4), squeeze=False)
    axes_flat = axes.flatten()

    # Group runs by source for coloring
    for run in runs:
        df = load_pck(run.snapshot_dir, args.scope)
        if df is None:
            continue
        color = SOURCE_COLORS.get(run.source, "gray")
        ls, marker = METHOD_STYLES.get(run.method, ("-", "o"))
        # Lighten color for smaller fractions
        alpha = 0.4 + 0.6 * (run.fraction / 0.10)

        for idx, bench in enumerate(BENCHMARKS):
            if idx >= len(axes_flat):
                break
            bench_df = df[df["benchmark"] == bench].sort_values("training_steps")
            if bench_df.empty:
                continue
            axes_flat[idx].plot(
                bench_df["training_steps"],
                bench_df["pck"],
                color=color,
                linestyle=ls,
                alpha=alpha,
                linewidth=1.2,
                label=run.label,
            )

    for idx, bench in enumerate(BENCHMARKS):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        ax.set_title(bench, fontsize=11)
        ax.set_xlabel("training steps")
        ax.set_ylabel("PCK")
        ax.grid(alpha=0.3)

    for j in range(len(BENCHMARKS), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Build legend with unique entries
    handles, labels = [], []
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)

    # Put legend below figure
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(4, len(labels)),
        fontsize=7,
        frameon=True,
    )
    fig.suptitle(f"Homogeneous ablation — convergence curves ({len(runs)} runs found)", fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])

    out = args.output_dir / "homogeneous_ablation_convergence.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_efficiency(runs: List[RunInfo], args: argparse.Namespace) -> Path:
    """Plot converged PCK vs subset fraction: one subplot per benchmark,
    lines per source, solid=clustercov dashed=random."""
    n_bench = len(BENCHMARKS)
    cols = min(3, n_bench)
    rows = math.ceil(n_bench / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4), squeeze=False)
    axes_flat = axes.flatten()

    # Collect converged PCK per (source, method, fraction, benchmark)
    records: List[Dict] = []
    for run in runs:
        df = load_pck(run.snapshot_dir, args.scope)
        if df is None:
            continue
        for bench in BENCHMARKS:
            pck = get_converged_pck(df, bench, args.tail_n)
            if pck is not None:
                records.append({
                    "source": run.source,
                    "method": run.method,
                    "fraction": run.fraction,
                    "fraction_label": run.fraction_label,
                    "benchmark": bench,
                    "pck": pck,
                })

    if not records:
        print("No converged results yet — skipping efficiency plot.")
        return args.output_dir / "homogeneous_ablation_efficiency.png"

    results = pd.DataFrame(records)

    for idx, bench in enumerate(BENCHMARKS):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        bench_df = results[results["benchmark"] == bench]

        for source in sorted(bench_df["source"].unique()):
            color = SOURCE_COLORS.get(source, "gray")
            for method in ["clustercov", "random"]:
                subset = bench_df[
                    (bench_df["source"] == source) & (bench_df["method"] == method)
                ].sort_values("fraction")
                if subset.empty:
                    continue
                ls, marker = METHOD_STYLES.get(method, ("-", "o"))
                ax.plot(
                    subset["fraction"] * 100,
                    subset["pck"],
                    color=color,
                    linestyle=ls,
                    marker=marker,
                    markersize=6,
                    linewidth=1.5,
                    label=f"{source} {method}",
                )

        ax.set_title(bench, fontsize=11)
        ax.set_xlabel("budget (% of source pool)")
        ax.set_ylabel("converged PCK")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5, 10])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(alpha=0.3)

    for j in range(len(BENCHMARKS), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Deduplicated legend
    handles, labels = [], []
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)

    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(4, len(labels)),
        fontsize=8,
        frameon=True,
    )
    fig.suptitle(
        f"Homogeneous ablation — efficiency curves (tail-{args.tail_n} median PCK)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])

    out = args.output_dir / "homogeneous_ablation_efficiency.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.snapshots_dir)
    print(f"Found {len(runs)} homogeneous ablation runs:")
    for r in runs:
        print(f"  {r.label}  ({r.snapshot_dir.name})")

    if not runs:
        print("No runs found yet. Re-run as jobs complete.")
        return

    plot_convergence(runs, args)
    plot_efficiency(runs, args)
    print("\nDone. Re-run anytime to pick up new results.")


if __name__ == "__main__":
    main()
