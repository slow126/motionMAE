#!/usr/bin/env python3
"""Plot pooled 0.5% validation comparisons and converged summaries."""

from __future__ import annotations

import argparse
import pathlib
from types import SimpleNamespace

from export_validation_tail import (
    build_summary_export,
    build_summary_markdown,
    determine_benchmarks,
    print_summary_table,
    write_summary_bar_plot,
)
from plot_validation_loss import load_validation_csv, plot_metric_curves, plot_multi_metric


METHODS: dict[str, dict[str, str]] = {
    "clustercov": {
        "label": "ClusterCov (shortlist)",
        "prefix": "pointodyssey_spair_pfpascal_pooled_multitarget_clustercov_k1024_0p5pct",
    },
    "clustercov_norm_lr1e4": {
        "label": "ClusterCov + n_valid",
        "prefix": "pointodyssey_spair_pfpascal_pooled_multitarget_clustercov_k1024_norm_0p5pct_lr1e-4",
    },
    "joint_clustercov_norm_lr1e4": {
        "label": "Joint ClusterCov + n_valid",
        "prefix": "pointodyssey_spair_pfpascal_pooled_joint_multitarget_clustercov_k1024_norm_0p5pct_lr1e-4",
    },
    "clustercov_norm_noshortlist_lr1e4": {
        "label": "ClusterCov + n_valid",
        "prefix": "pointodyssey_spair_pfpascal_pooled_multitarget_clustercov_k1024_norm_noshortlist_0p5pct_lr1e-4",
    },
    "clustercov_norm_noshortlist_dedup_lr1e4": {
        "label": "ClusterCov + n_valid + dedup",
        "prefix": "pointodyssey_spair_pfpascal_pooled_multitarget_clustercov_k1024_norm_noshortlist_dedup_0p5pct_lr1e-4",
    },
    "mixed_balanced_lr1e4": {
        "label": "Hand-tuned mixed + random",
        "prefix": "pointodyssey_spair_pfpascal_pooled_mixed_balanced_0p5pct_lr1e-4",
    },
    "pfpascal_only": {
        "label": "PF-PASCAL only",
        "prefix": "pointodyssey_spair_pfpascal_pooled_pfpascal_only",
    },
    "pointodyssey_only": {
        "label": "PointOdyssey only",
        "prefix": "pointodyssey_spair_pfpascal_pooled_pointodyssey_only_0p5pct",
    },
    "clustercov_pointodyssey_only": {
        "label": "ClusterCov PointOdyssey-only",
        "prefix": "pointodyssey_spair_pfpascal_pooled_multitarget_clustercov_k1024_pointodyssey_only_0p5pct",
    },
    "pooled_random_shuffled": {
        "label": "Pooled random",
        "prefix": "pointodyssey_spair_pfpascal_pooled_random_0p5pct_shuffled",
    },
    "spair_only_match_pointodyssey": {
        "label": "SPair-71k only",
        "prefix": "pointodyssey_spair_pfpascal_pooled_spair_only_match_pointodyssey_0p5pct",
    },
}

DEFAULT_METHODS = [
    "joint_clustercov_norm_lr1e4",
    "clustercov_norm_noshortlist_dedup_lr1e4",
    "clustercov_norm_noshortlist_lr1e4",
    "mixed_balanced_lr1e4",
    "pfpascal_only",
    "pointodyssey_only",
    "pooled_random_shuffled",
    "spair_only_match_pointodyssey",
]

SOLID_CLUSTER_COV_LABELS = {
    METHODS["joint_clustercov_norm_lr1e4"]["label"],
    METHODS["clustercov_norm_noshortlist_dedup_lr1e4"]["label"],
    METHODS["clustercov_norm_noshortlist_lr1e4"]["label"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pooled 0.5%% validation comparisons and converged summaries."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHODS.keys()),
        default=DEFAULT_METHODS,
        metavar="METHOD",
        help="Methods to include in the comparison.",
    )
    parser.add_argument(
        "--mode",
        choices=["curves", "summary", "both"],
        default="summary",
        help="Which outputs to generate (default: summary).",
    )
    parser.add_argument(
        "--snapshots-root",
        default="snapshots",
        help="Directory containing snapshot folders (default: snapshots).",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/plots",
        help="Directory for generated outputs (default: analysis/plots).",
    )
    parser.add_argument(
        "--metric",
        default="pck",
        choices=["loss", "pck", "pck_motion_aware"],
        help="Metric to plot/export (default: pck).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["loss", "pck", "pck_motion_aware"],
        default=None,
        metavar="METRIC",
        help="Plot multiple metrics side by side for curve plots.",
    )
    parser.add_argument(
        "--scope",
        default="auto",
        choices=["auto", "step", "epoch", "initial", "all"],
        help="Validation scope filter (default: auto).",
    )
    parser.add_argument(
        "--x-axis",
        default="training_steps",
        choices=["training_steps", "epoch"],
        help="X-axis for plots/export (default: training_steps).",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Optional single benchmark filter.",
    )
    parser.add_argument(
        "--target",
        default=None,
        type=float,
        help="Optional exact validation_target filter.",
    )
    parser.add_argument(
        "--cols",
        default=2,
        type=int,
        help="Columns in the per-benchmark curve grid (default: 2).",
    )
    parser.add_argument(
        "--include-step-zero",
        action="store_true",
        help="Prepend a step=0 point when absent in curve plots.",
    )
    parser.add_argument(
        "--exclude-step-zero",
        action="store_true",
        help="Drop step-0 rows from plots and summaries.",
    )
    parser.add_argument(
        "--compare-step-zero",
        action="store_true",
        help="Show side-by-side curve plots without and with step-0.",
    )
    parser.add_argument(
        "--last-n",
        default=8,
        type=int,
        help="Tail window for converged summary export (default: 8).",
    )
    parser.add_argument(
        "--curve-output",
        default="pooled_0p5pct_validation_pck.png",
        help="Filename for the curve plot inside --output-dir.",
    )
    parser.add_argument(
        "--smooth-window",
        default=1,
        type=int,
        help="Optional trailing rolling-average window for curve plots (default: 1 = no smoothing).",
    )
    parser.add_argument(
        "--hide-markers",
        action="store_true",
        help="Hide per-point markers in curve plots for a cleaner paper-style figure.",
    )
    parser.add_argument(
        "--summary-stem",
        default=None,
        help=(
            "Summary output stem inside --output-dir. "
            "Defaults to validation_converged_summary_last{N}_{scope}."
        ),
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print resolved snapshots without rendering outputs.",
    )
    return parser.parse_args()


def latest_snapshot(snapshots_root: pathlib.Path, prefix: str) -> pathlib.Path:
    matches = [
        path
        for path in snapshots_root.glob(f"{prefix}_*")
        if path.is_dir() and (path / "validation_results.csv").exists()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No snapshot with validation_results.csv found for prefix '{prefix}' in {snapshots_root}"
        )
    return max(matches, key=lambda path: path.name)


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def resolve_runs(
    snapshots_root: pathlib.Path,
    method_ids: list[str],
) -> list[tuple[str, pathlib.Path]]:
    runs: list[tuple[str, pathlib.Path]] = []
    for method_id in ordered_unique(method_ids):
        cfg = METHODS[method_id]
        runs.append((cfg["label"], latest_snapshot(snapshots_root, cfg["prefix"])))
    return runs


def build_plot_args(args: argparse.Namespace, output_path: pathlib.Path) -> SimpleNamespace:
    return SimpleNamespace(
        benchmark=args.benchmark,
        cols=args.cols,
        compare_step_zero=args.compare_step_zero,
        exclude_step_zero=args.exclude_step_zero,
        hide_markers=args.hide_markers,
        include_step_zero=args.include_step_zero,
        metric=args.metric,
        metrics=args.metrics,
        output=str(output_path),
        scope=args.scope,
        smooth_window=args.smooth_window,
        snapshots=[],
        target=args.target,
        x_axis=args.x_axis,
    )


def render_curves(
    runs: list[tuple[str, pathlib.Path]],
    args: argparse.Namespace,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    if len(runs) < 2:
        raise ValueError(f"Need at least 2 runs to compare, got {len(runs)}.")

    output_path = output_dir / args.curve_output
    plot_args = build_plot_args(args, output_path)
    labels = [label for label, _ in runs]
    snapshots = [snapshot for _, snapshot in runs]

    if args.metrics:
        plot_multi_metric(snapshots, labels, args.metrics, plot_args)
    else:
        loaded_runs = [(label, load_validation_csv(snapshot, args.metric)) for label, snapshot in runs]
        plot_metric_curves(loaded_runs, plot_args)
    return output_path


def summary_output_stem(args: argparse.Namespace) -> str:
    if args.summary_stem:
        return args.summary_stem
    return f"validation_converged_summary_last{args.last_n}_{args.scope}"


def render_summary(
    runs: list[tuple[str, pathlib.Path]],
    args: argparse.Namespace,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    if args.last_n <= 0:
        raise ValueError("--last-n must be a positive integer.")

    run_rows = [
        (label, snapshot, load_validation_csv(snapshot, args.metric))
        for label, snapshot in runs
    ]
    run_dfs = [df for _, _, df in run_rows]
    benchmarks = determine_benchmarks(run_dfs, args.benchmark)
    summary_df = build_summary_export(run_rows, benchmarks, args)

    stem = summary_output_stem(args)
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    png_path = output_dir / f"{stem}.png"
    mean_png_path = output_dir / f"{stem}_mean.png"

    summary_df.to_csv(csv_path, index=False)
    md_path.write_text(build_summary_markdown(summary_df))
    write_summary_bar_plot(
        summary_df,
        png_path,
        solid_run_labels=SOLID_CLUSTER_COV_LABELS,
    )
    write_summary_bar_plot(
        summary_df,
        mean_png_path,
        value_column="tail_mean",
        solid_run_labels=SOLID_CLUSTER_COV_LABELS,
    )
    print_summary_table(summary_df)

    return csv_path, md_path, png_path, mean_png_path


def main() -> None:
    args = parse_args()
    snapshots_root = pathlib.Path(args.snapshots_root).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = resolve_runs(snapshots_root, list(args.methods))
    for label, snapshot in runs:
        print(f"{label}: {snapshot}")

    if args.print_only:
        return

    if args.mode in {"curves", "both"}:
        curve_path = render_curves(runs, args, output_dir)
        print(f"Wrote curve plot -> {curve_path}")

    if args.mode in {"summary", "both"}:
        csv_path, md_path, png_path, mean_png_path = render_summary(runs, args, output_dir)
        print(f"Wrote summary CSV -> {csv_path}")
        print(f"Wrote summary Markdown -> {md_path}")
        print(f"Wrote summary plot -> {png_path}")
        print(f"Wrote mean summary plot -> {mean_png_path}")


if __name__ == "__main__":
    main()
