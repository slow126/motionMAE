#!/usr/bin/env python3
"""Plot grouped 5% PointOdyssey smoke validation comparisons.

This wrapper resolves the latest snapshot for each run prefix and reuses the
generic validation plotting helpers from `plot_validation_loss.py`.
"""

from __future__ import annotations

import argparse
import pathlib
from types import SimpleNamespace

from plot_validation_loss import load_validation_csv, plot_metric_curves, plot_multi_metric


METHODS: dict[str, dict[str, object]] = {
    "random": {
        "label": "Random",
        "prefix": "pointodyssey_smoke_random_5pct",
        "tags": ("baseline",),
    },
    "heuristic": {
        "label": "Heuristic",
        "prefix": "pointodyssey_smoke_heuristic_5pct",
        "tags": ("baseline",),
    },
    "fps_bfv": {
        "label": "FPS BFV",
        "prefix": "pointodyssey_smoke_fps_bfv_5pct",
        "tags": ("fps", "representation"),
    },
    "fps_mean_mag": {
        "label": "FPS mean_mag",
        "prefix": "pointodyssey_smoke_fps_mean_mag_5pct",
        "tags": ("fps", "representation"),
    },
    "fps_median_mag": {
        "label": "FPS median_mag",
        "prefix": "pointodyssey_smoke_fps_median_mag_5pct",
        "tags": ("fps", "representation"),
    },
    "fps_p90_mag": {
        "label": "FPS p90_mag",
        "prefix": "pointodyssey_smoke_fps_p90_mag_5pct",
        "tags": ("fps", "representation"),
    },
    "single_target_joint": {
        "label": "Single-target joint",
        "prefix": "pointodyssey_smoke_top_joint_5pct",
        "tags": ("target_aware", "single_target"),
    },
    "single_target_p90": {
        "label": "Single-target p90",
        "prefix": "pointodyssey_smoke_top_p90_5pct",
        "tags": ("target_aware", "single_target"),
    },
    "multi_target_nn": {
        "label": "Multi-target NN (equal-weight)",
        "prefix": "pointodyssey_smoke_multitarget_nn_5pct",
        "tags": ("target_aware", "multi_target"),
    },
}

GROUPS: dict[str, list[str]] = {
    "sampling": ["random", "heuristic"],
    "representation": ["fps_bfv", "fps_mean_mag", "fps_median_mag", "fps_p90_mag"],
    "target_aware": ["single_target_joint", "single_target_p90", "multi_target_nn"],
    "all_methods": list(METHODS.keys()),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 5%% PointOdyssey smoke comparisons for selected method subsets."
    )
    parser.add_argument(
        "--group",
        nargs="+",
        choices=["sampling", "representation", "target_aware", "all_methods", "all"],
        default=["all"],
        help="Comparison group(s) to render. Ignored when --methods is set.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHODS.keys()),
        default=None,
        metavar="METHOD",
        help="Explicit methods to include in a single custom plot.",
    )
    parser.add_argument(
        "--plot-name",
        default="custom",
        help="Output stem suffix when using --methods (default: custom).",
    )
    parser.add_argument(
        "--no-fps",
        action="store_true",
        help="Exclude all FPS methods from the selected group(s) or explicit method list.",
    )
    parser.add_argument(
        "--snapshots-root",
        default="snapshots",
        help="Directory containing snapshot folders (default: snapshots).",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/plots",
        help="Directory for generated plot files (default: analysis/plots).",
    )
    parser.add_argument(
        "--metric",
        default="pck",
        choices=["loss", "pck", "pck_motion_aware"],
        help="Metric to plot when --metrics is not used (default: pck).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["loss", "pck", "pck_motion_aware"],
        default=None,
        metavar="METRIC",
        help="Plot multiple metrics side by side.",
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
        help="X-axis to plot against (default: training_steps).",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Optional single benchmark to plot.",
    )
    parser.add_argument(
        "--target",
        default=None,
        type=float,
        help="Optional exact validation_target value filter.",
    )
    parser.add_argument(
        "--cols",
        default=2,
        type=int,
        help="Columns in the per-benchmark grid (default: 2).",
    )
    parser.add_argument(
        "--include-step-zero",
        action="store_true",
        help="Prepend a step=0 point when absent.",
    )
    parser.add_argument(
        "--exclude-step-zero",
        action="store_true",
        help="Drop step-0 rows from the plot.",
    )
    parser.add_argument(
        "--compare-step-zero",
        action="store_true",
        help="Show side-by-side plots without and with step-0.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print resolved snapshots without rendering plots.",
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


def selected_groups(raw_groups: list[str]) -> list[str]:
    if "all" in raw_groups:
        return list(GROUPS.keys())
    return raw_groups


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def filter_method_ids(method_ids: list[str], no_fps: bool) -> list[str]:
    if not no_fps:
        return method_ids
    return [method_id for method_id in method_ids if "fps" not in METHODS[method_id]["tags"]]


def resolve_runs(snapshots_root: pathlib.Path, method_ids: list[str]) -> list[tuple[str, pathlib.Path]]:
    runs: list[tuple[str, pathlib.Path]] = []
    for method_id in method_ids:
        cfg = METHODS[method_id]
        runs.append((str(cfg["label"]), latest_snapshot(snapshots_root, str(cfg["prefix"]))))
    return runs


def build_plot_args(args: argparse.Namespace, output_path: pathlib.Path) -> SimpleNamespace:
    return SimpleNamespace(
        benchmark=args.benchmark,
        cols=args.cols,
        compare_step_zero=args.compare_step_zero,
        exclude_step_zero=args.exclude_step_zero,
        include_step_zero=args.include_step_zero,
        metric=args.metric,
        metrics=args.metrics,
        output=str(output_path),
        scope=args.scope,
        snapshots=[],
        target=args.target,
        x_axis=args.x_axis,
    )


def render_plot(
    plot_name: str,
    runs: list[tuple[str, pathlib.Path]],
    args: argparse.Namespace,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    if len(runs) < 2:
        raise ValueError(f"Need at least 2 runs to compare, got {len(runs)} for '{plot_name}'.")

    metric_suffix = "-".join(args.metrics) if args.metrics else args.metric
    output_path = output_dir / f"pointodyssey_smoke_5pct_{plot_name}_{metric_suffix}.png"
    plot_args = build_plot_args(args, output_path)
    labels = [label for label, _ in runs]
    snapshots = [snapshot for _, snapshot in runs]

    if args.metrics:
        plot_multi_metric(snapshots, labels, args.metrics, plot_args)
    else:
        loaded_runs = [(label, load_validation_csv(snapshot, args.metric)) for label, snapshot in runs]
        plot_metric_curves(loaded_runs, plot_args)

    return output_path


def main() -> None:
    args = parse_args()
    snapshots_root = pathlib.Path(args.snapshots_root).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.methods is not None:
        method_ids = filter_method_ids(ordered_unique(list(args.methods)), args.no_fps)
        runs = resolve_runs(snapshots_root, method_ids)
        print(f"[{args.plot_name}]")
        for label, snapshot in runs:
            print(f"  {label}: {snapshot}")
        if not args.print_only:
            output_path = render_plot(args.plot_name, runs, args, output_dir)
            print(f"  wrote: {output_path}")
        return

    for group_name in selected_groups(args.group):
        method_ids = filter_method_ids(ordered_unique(GROUPS[group_name]), args.no_fps)
        runs = resolve_runs(snapshots_root, method_ids)
        print(f"[{group_name}]")
        for label, snapshot in runs:
            print(f"  {label}: {snapshot}")

        if args.print_only:
            continue

        output_path = render_plot(group_name, runs, args, output_dir)
        print(f"  wrote: {output_path}")


if __name__ == "__main__":
    main()
