#!/usr/bin/env python3
"""Plot validation metric curves from snapshot `validation_results.csv` files."""

from __future__ import annotations

import argparse
import math
import pathlib
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-benchmark validation curves from 2+ snapshot folders."
    )
    parser.add_argument(
        "snapshots",
        nargs="+",
        help="Paths to snapshot directories (each must contain validation_results.csv).",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        metavar="LABEL",
        help="Labels for runs in legend; if set, count must match snapshots.",
    )
    parser.add_argument(
        "--scope",
        default="auto",
        choices=["auto", "step", "epoch", "initial", "all"],
        help="Validation scope filter. 'auto' merges per x-axis using step>epoch>initial priority.",
    )
    parser.add_argument(
        "--metric",
        default="pck",
        choices=["loss", "pck", "pck_motion_aware"],
        help="Column to plot from validation_results.csv (default: pck).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["loss", "pck", "pck_motion_aware"],
        default=None,
        metavar="METRIC",
        help="Plot multiple metrics side by side (one column per metric). Overrides --metric.",
    )
    parser.add_argument(
        "--include-step-zero",
        action="store_true",
        help="Prepend a step=0 point for each run/benchmark using first measured value.",
    )
    parser.add_argument(
        "--x-axis",
        default="training_steps",
        choices=["training_steps", "epoch"],
        help="X-axis metric to plot against.",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Optional single benchmark to plot in a single axis.",
    )
    parser.add_argument(
        "--target",
        default=None,
        type=float,
        help="Optional exact validation_target value to filter (e.g. 5000).",
    )
    parser.add_argument(
        "--cols",
        default=2,
        type=int,
        help="Number of columns in per-benchmark grid (default: 2).",
    )
    parser.add_argument(
        "--compare-step-zero",
        action="store_true",
        help="Show side-by-side plots per benchmark: without step-0 and with step-0.",
    )
    parser.add_argument(
        "--exclude-step-zero",
        action="store_true",
        help="Drop rows where selected x-axis equals 0 (applies across all scopes).",
    )
    parser.add_argument(
        "--output",
        default="validation_pck_curve.png",
        help="Path to save figure (default: validation_pck_curve.png).",
    )
    return parser.parse_args()


def load_validation_csv(snapshot: pathlib.Path, metric: str) -> pd.DataFrame:
    csv_path = snapshot / "validation_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing validation_results.csv in {snapshot}")

    df = pd.read_csv(csv_path)
    if "validation_scope" not in df.columns or metric not in df.columns:
        raise ValueError(f"Unexpected CSV format in {csv_path}")

    df["training_steps"] = pd.to_numeric(df["training_steps"], errors="coerce")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df["validation_scope"] = df["validation_scope"].astype(str)
    return df


def _apply_scope(series: pd.DataFrame, scope: str, x_axis: str) -> pd.DataFrame:
    if series.empty:
        return series
    if scope == "all":
        return series
    if scope in {"step", "epoch", "initial"}:
        return series[series["validation_scope"] == scope].copy()

    # auto: keep one row per x-axis, preferring denser "step" logs, then epoch, then initial.
    priority = {"step": 0, "epoch": 1, "initial": 2}
    scoped = series.copy()
    scoped["_scope_pri"] = scoped["validation_scope"].map(priority).fillna(99).astype(int)
    scoped = scoped.sort_values([x_axis, "_scope_pri"])
    scoped = scoped.groupby(x_axis, as_index=False).first()
    if "_scope_pri" in scoped.columns:
        scoped = scoped.drop(columns=["_scope_pri"])
    return scoped


def _add_step_zero(df: pd.DataFrame, x_axis: str, metric: str) -> pd.DataFrame:
    if x_axis != "training_steps" or (df[x_axis] == 0).any():
        return df
    if df.empty:
        return df

    first = df.sort_values(x_axis).iloc[[0]].copy()
    first[x_axis] = 0
    if pd.isna(first[metric].iloc[0]):
        return df
    return pd.concat([first, df], ignore_index=True)


def _prepare_series(
    df: pd.DataFrame,
    benchmark: str,
    args: argparse.Namespace,
    *,
    include_zero: bool = False,
    include_explicit_zero: bool = True,
) -> pd.DataFrame:
    series = df[df["benchmark"] == benchmark].copy()
    if args.target is not None:
        series = series[series["validation_target"] == args.target]
    series = series.sort_values(args.x_axis).dropna(subset=[args.x_axis, args.metric])
    series = _apply_scope(series, args.scope, args.x_axis)
    if args.exclude_step_zero:
        series = series[series[args.x_axis] != 0]
    if not include_explicit_zero:
        series = series[series[args.x_axis] != 0]
    if include_zero:
        series = _add_step_zero(series, args.x_axis, args.metric)
    return series


def plot_single(
    ax: plt.Axes,
    series_by_run: Sequence[tuple[str, pd.DataFrame]],
    args: argparse.Namespace,
    bench: str,
    suffix: str = "",
    show_legend: bool = False,
) -> None:
    if not series_by_run:
        ax.set_visible(False)
        return

    all_empty = all(series.empty for _, series in series_by_run)
    if all_empty:
        ax.set_visible(False)
        return

    markers = ["o", "x", "^", "s", "D", "v", "P", "*"]
    linestyles = ["-", "--", "-.", ":"]
    for idx, (label, series) in enumerate(series_by_run):
        if series.empty:
            continue
        ax.plot(
            series[args.x_axis],
            series[args.metric],
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            label=label,
        )

    title = f"{bench} ({suffix})" if suffix else bench
    ax.set_title(title)
    ax.set_xlabel(args.x_axis)
    ax.set_ylabel(args.metric)
    ax.grid(alpha=0.3)
    if show_legend:
        ax.legend(fontsize=8)


def plot_metric_curves(
    runs: Sequence[tuple[str, pd.DataFrame]],
    args: argparse.Namespace,
) -> None:
    if len(runs) < 2:
        raise ValueError("Need at least 2 runs to compare.")
    run_names = [name for name, _ in runs]
    run_dfs = [df for _, df in runs]

    if args.benchmark:
        fig, ax = plt.subplots(figsize=(9, 5))
        if args.compare_step_zero:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            no_series = [
                (
                    name,
                    _prepare_series(
                        df,
                        args.benchmark,
                        args,
                        include_zero=False,
                        include_explicit_zero=False,
                    ),
                )
                for name, df in zip(run_names, run_dfs)
            ]
            with_series = [
                (
                    name,
                    _prepare_series(
                        df,
                        args.benchmark,
                        args,
                        include_zero=args.include_step_zero,
                        include_explicit_zero=True,
                    ),
                )
                for name, df in zip(run_names, run_dfs)
            ]
            plot_single(axes[0], no_series, args, args.benchmark, suffix="no step-0")
            plot_single(
                axes[1],
                with_series,
                args,
                args.benchmark,
                suffix="with step-0",
                show_legend=True,
            )
            handles, labels = axes[1].get_legend_handles_labels()
            fig.tight_layout()
            if handles:
                fig.legend(handles, labels, loc="upper right", fontsize=8)
            fig.savefig(args.output, dpi=200)
            print(f"Saved plot: {args.output}")
            return

        series = [
            (name, _prepare_series(df, args.benchmark, args))
            for name, df in zip(run_names, run_dfs)
        ]
        plot_single(ax, series, args, args.benchmark, show_legend=True)
        fig.tight_layout()
        fig.savefig(args.output, dpi=200)
        print(f"Saved plot: {args.output}")
        return

    common_benchmarks = set(run_dfs[0]["benchmark"])
    for df in run_dfs[1:]:
        common_benchmarks &= set(df["benchmark"])
    common_benchmarks = sorted(common_benchmarks)
    if not common_benchmarks:
        raise ValueError("No common benchmarks found between snapshots.")

    if args.compare_step_zero:
        fig, axes = plt.subplots(
            len(common_benchmarks),
            2,
            figsize=(12, len(common_benchmarks) * 3.5),
            squeeze=False,
        )

        for idx, bench in enumerate(common_benchmarks):
            no_series = [
                (
                    name,
                    _prepare_series(
                        df,
                        bench,
                        args,
                        include_zero=False,
                        include_explicit_zero=False,
                    ),
                )
                for name, df in zip(run_names, run_dfs)
            ]
            with_series = [
                (
                    name,
                    _prepare_series(
                        df,
                        bench,
                        args,
                        include_zero=args.include_step_zero,
                        include_explicit_zero=True,
                    ),
                )
                for name, df in zip(run_names, run_dfs)
            ]

            show_legend = (idx == 0)
            plot_single(
                axes[idx][0],
                no_series,
                args,
                bench,
                suffix="no step-0",
                show_legend=False,
            )
            plot_single(
                axes[idx][1],
                with_series,
                args,
                bench,
                suffix="with step-0",
                show_legend=show_legend,
            )

            # Hide the right plot if no data was available.
            if all(series.empty for _, series in no_series):
                axes[idx][0].set_visible(False)
            if all(series.empty for _, series in with_series):
                axes[idx][1].set_visible(False)

        handles, labels = axes[0][1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=8)
        fig.suptitle(f"Validation {args.metric} by benchmark ({args.scope}), no-vs-with step-0")
        fig.tight_layout()
        fig.savefig(args.output, dpi=200)
        print(f"Saved plot: {args.output}")
        return

    cols = max(1, int(args.cols))
    rows = math.ceil(len(common_benchmarks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5), squeeze=False)
    axes = axes.flatten()

    for idx, bench in enumerate(common_benchmarks):
        series = [(name, _prepare_series(df, bench, args)) for name, df in zip(run_names, run_dfs)]
        plot_single(axes[idx], series, args, bench)

    for j in range(len(common_benchmarks), len(axes)):
        axes[j].set_visible(False)

    # Build a shared legend from the first valid axis
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle(f"Validation {args.metric} by benchmark ({args.scope})")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved plot: {args.output}")


def plot_multi_metric(
    snapshots: list[pathlib.Path],
    labels: list[str],
    metrics: list[str],
    args: argparse.Namespace,
) -> None:
    """Plot multiple metrics side by side: rows=benchmarks, cols=metrics."""
    # Load all CSVs for each metric
    runs_per_metric = {}
    for metric in metrics:
        runs_per_metric[metric] = [
            (label, load_validation_csv(snap, metric))
            for label, snap in zip(labels, snapshots)
        ]

    # Find common benchmarks across all metrics and runs
    all_dfs = [df for runs in runs_per_metric.values() for _, df in runs]
    common_benchmarks = set(all_dfs[0]["benchmark"])
    for df in all_dfs[1:]:
        common_benchmarks &= set(df["benchmark"])
    common_benchmarks = sorted(common_benchmarks)
    if not common_benchmarks:
        raise ValueError("No common benchmarks found.")

    n_rows = len(common_benchmarks)
    n_cols = len(metrics)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5, n_rows * 3.5),
        squeeze=False,
    )

    for col, metric in enumerate(metrics):
        args_copy = argparse.Namespace(**vars(args))
        args_copy.metric = metric
        runs = runs_per_metric[metric]
        run_names = [name for name, _ in runs]
        run_dfs = [df for _, df in runs]

        for row, bench in enumerate(common_benchmarks):
            series = [
                (name, _prepare_series(df, bench, args_copy))
                for name, df in zip(run_names, run_dfs)
            ]
            show_legend = (row == 0 and col == n_cols - 1)
            plot_single(axes[row][col], series, args_copy, bench, show_legend=show_legend)
            if row == 0:
                axes[row][col].set_title(f"{bench}\n({metric})")
            else:
                axes[row][col].set_title(f"{bench} ({metric})")

    handles, labels_legend = axes[0][n_cols - 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_legend, loc="upper right", fontsize=8)
    fig.suptitle(f"Validation: {' vs '.join(metrics)} ({args.scope})")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved plot: {args.output}")


def main() -> None:
    args = parse_args()
    if len(args.snapshots) < 2:
        raise ValueError("Provide at least 2 snapshot paths.")

    snapshots = [pathlib.Path(p).expanduser().resolve() for p in args.snapshots]
    if args.labels is not None and len(args.labels) != len(snapshots):
        raise ValueError("--labels count must match number of snapshots.")

    labels = list(args.labels) if args.labels is not None else [snap.name for snap in snapshots]

    if args.metrics is not None:
        plot_multi_metric(snapshots, labels, args.metrics, args)
        return

    runs = [(label, load_validation_csv(snap, args.metric)) for label, snap in zip(labels, snapshots)]
    plot_metric_curves(runs, args)


if __name__ == "__main__":
    main()
