#!/usr/bin/env python3
"""Export the last N filtered validation points from snapshot CSV logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from validation_curve_utils import load_validation_csv, prepare_series

BENCHMARK_LABELS = {
    "kitti2012": "KITTI-2012",
    "kitti2015": "KITTI-2015",
    "pfpascal": "PF-PASCAL",
    "pfwillow": "PF-WILLOW",
    "tss": "TSS",
}

METRIC_LABELS = {
    "loss": "Loss",
    "pck": "PCK",
    "pck_motion_aware": "Motion-Aware PCK",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the last N validation points per run/benchmark from "
            "snapshot validation_results.csv files."
        )
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
        help="Labels for runs in output; if set, count must match snapshots.",
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
        help="Metric used for filtering/export defaults and NA dropping (default: pck).",
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
        help="X-axis metric used to order the exported tail.",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Optional single benchmark to export.",
    )
    parser.add_argument(
        "--target",
        default=None,
        type=float,
        help="Optional exact validation_target value to filter (e.g. 5000).",
    )
    parser.add_argument(
        "--exclude-step-zero",
        action="store_true",
        help="Drop rows where selected x-axis equals 0 (applies across all scopes).",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=4,
        help="Number of most recent filtered validation points per run/benchmark (default: 4).",
    )
    parser.add_argument(
        "--format",
        choices=["wide", "long", "summary"],
        default="wide",
        help=(
            "Export shape: 'wide' writes one row per run/validation event with benchmark "
            "metrics as columns; 'long' writes one row per run/benchmark/event; "
            "'summary' writes one row per run/benchmark with tail statistics."
        ),
    )
    parser.add_argument(
        "--output",
        default="validation_tail.csv",
        help="Path to save CSV output (default: validation_tail.csv).",
    )
    parser.add_argument(
        "--readme-output",
        default=None,
        help=(
            "Optional Markdown summary path for --format summary. "
            "Defaults to OUTPUT with a .md suffix."
        ),
    )
    parser.add_argument(
        "--bar-plot-output",
        default=None,
        help=(
            "Optional grouped bar plot path for --format summary. "
            "Defaults to OUTPUT with a .png suffix."
        ),
    )
    return parser.parse_args()


def determine_benchmarks(run_dfs: list[pd.DataFrame], benchmark: str | None) -> list[str]:
    if benchmark is not None:
        return [benchmark]

    common_benchmarks = set(run_dfs[0]["benchmark"])
    for df in run_dfs[1:]:
        common_benchmarks &= set(df["benchmark"])
    benchmarks = sorted(common_benchmarks)
    if not benchmarks:
        raise ValueError("No common benchmarks found between snapshots.")
    return benchmarks


def pretty_benchmark(bench: str) -> str:
    return BENCHMARK_LABELS.get(bench, bench)


def pretty_metric(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _filter_base_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.benchmark is not None:
        out = out[out["benchmark"] == args.benchmark]
    if args.target is not None:
        out = out[out["validation_target"] == args.target]
    out = out.dropna(subset=[args.x_axis, args.metric]).copy()
    if args.exclude_step_zero:
        out = out[out[args.x_axis] != 0]
    return out


def _select_tail_event_keys(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    base = _filter_base_rows(df, args)
    if base.empty:
        return base

    if args.scope == "all":
        scoped = base.copy()
    elif args.scope in {"step", "epoch", "initial"}:
        scoped = base[base["validation_scope"] == args.scope].copy()
    else:
        priority = {"step": 0, "epoch": 1, "initial": 2}
        scoped = base.copy()
        scoped["_scope_pri"] = scoped["validation_scope"].map(priority).fillna(99).astype(int)
        scoped = scoped.sort_values(
            [args.x_axis, "_scope_pri", "training_steps", "epoch", "validation_target"]
        )
        scoped = scoped.groupby(args.x_axis, as_index=False).first()
        scoped = scoped.drop(columns=["_scope_pri"])

    if scoped.empty:
        return scoped

    key_cols = ["training_steps", "epoch", "validation_scope", "validation_target"]
    return (
        scoped[key_cols]
        .drop_duplicates()
        .sort_values(args.x_axis)
        .tail(args.last_n)
        .reset_index(drop=True)
    )


def build_long_export(
    runs: list[tuple[str, Path, pd.DataFrame]],
    benchmarks: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    exported = []
    for benchmark in benchmarks:
        for label, snapshot, df in runs:
            series = prepare_series(df, benchmark, args)
            if series.empty:
                continue
            tail = series.sort_values(args.x_axis).tail(args.last_n).copy()
            tail.insert(0, "snapshot", str(snapshot))
            tail.insert(0, "run_label", label)
            exported.append(tail)

    if not exported:
        raise ValueError("No validation rows matched the requested filters.")

    out_df = pd.concat(exported, ignore_index=True)
    return out_df.sort_values(["benchmark", "run_label", args.x_axis, "training_steps", "epoch"])


def build_wide_export(
    runs: list[tuple[str, Path, pd.DataFrame]],
    benchmarks: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    exported = []
    key_cols = ["training_steps", "epoch", "validation_scope", "validation_target"]

    for label, snapshot, df in runs:
        event_keys = _select_tail_event_keys(df, args)
        if event_keys.empty:
            continue

        rows = _filter_base_rows(df, args)
        rows = rows[rows["benchmark"].isin(benchmarks)].copy()
        rows = rows.merge(event_keys, on=key_cols, how="inner")
        if rows.empty:
            continue

        wide = (
            rows.pivot_table(
                index=key_cols,
                columns="benchmark",
                values=args.metric,
                aggfunc="first",
            )
            .reset_index()
        )
        wide.columns = [
            f"{args.metric}__{col}" if col not in key_cols else col
            for col in wide.columns
        ]
        wide.insert(0, "snapshot", str(snapshot))
        wide.insert(0, "run_label", label)
        exported.append(wide)

    if not exported:
        raise ValueError("No validation rows matched the requested filters.")

    out_df = pd.concat(exported, ignore_index=True)
    return out_df.sort_values(["run_label", args.x_axis, "training_steps", "epoch"])


def build_summary_export(
    runs: list[tuple[str, Path, pd.DataFrame]],
    benchmarks: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    exported = []

    for benchmark in benchmarks:
        for label, snapshot, df in runs:
            series = prepare_series(df, benchmark, args)
            if series.empty:
                continue

            tail = series.sort_values(args.x_axis).tail(args.last_n).copy()
            metric_values = tail[args.metric].astype(float)
            last_row = tail.sort_values(args.x_axis).iloc[-1]
            tail_std = metric_values.std(ddof=0) if len(metric_values) > 1 else 0.0

            exported.append(
                {
                    "run_label": label,
                    "snapshot": str(snapshot),
                    "benchmark": benchmark,
                    "tail_count": int(len(tail)),
                    "tail_median": float(metric_values.median()),
                    "tail_mean": float(metric_values.mean()),
                    "tail_std": float(tail_std),
                    "tail_min": float(metric_values.min()),
                    "tail_max": float(metric_values.max()),
                    "last_value": float(last_row[args.metric]),
                    "last_training_steps": int(last_row["training_steps"]),
                    "last_epoch": int(last_row["epoch"]),
                    "summary_scope": args.scope,
                    "summary_metric": args.metric,
                    "summary_x_axis": args.x_axis,
                    "summary_last_n": int(args.last_n),
                }
            )

    if not exported:
        raise ValueError("No validation rows matched the requested filters.")

    out_df = pd.DataFrame(exported)
    return out_df.sort_values(["benchmark", "run_label"])


def print_summary_table(summary_df: pd.DataFrame) -> None:
    compare = summary_df.pivot(index="benchmark", columns="run_label", values="tail_median")
    compare = compare.sort_index(axis=0).sort_index(axis=1)
    compare.loc["macro_avg"] = compare.mean(axis=0)
    print("\nTail median comparison:")
    print(compare.to_string(float_format=lambda x: f"{x:.4f}"))


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not headers:
        return ""
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def _format_float(value: float) -> str:
    return f"{float(value):.4f}"


def build_summary_markdown(summary_df: pd.DataFrame) -> str:
    compare = summary_df.pivot(index="benchmark", columns="run_label", values="tail_median")
    compare = compare.sort_index(axis=0).sort_index(axis=1)
    compare.loc["macro_avg"] = compare.mean(axis=0)

    rank_df = (
        compare.loc[["macro_avg"]]
        .T
        .reset_index()
        .rename(columns={"index": "run_label", "macro_avg": "macro_avg_tail_median"})
        .sort_values("macro_avg_tail_median", ascending=False)
        .reset_index(drop=True)
    )

    summary_meta = summary_df.iloc[0]
    lines: list[str] = [
        "# Validation Converged Summary",
        "",
        f"- Metric: `{summary_meta['summary_metric']}`",
        f"- Scope: `{summary_meta['summary_scope']}`",
        f"- X-axis: `{summary_meta['summary_x_axis']}`",
        f"- Tail window: last `{int(summary_meta['summary_last_n'])}` points",
        "",
        "## Overall Ranking",
        "",
    ]

    rank_rows = [
        [str(idx + 1), row["run_label"], _format_float(row["macro_avg_tail_median"])]
        for idx, (_, row) in enumerate(rank_df.iterrows())
    ]
    lines.append(_markdown_table(["rank", "run_label", "macro_avg_tail_median"], rank_rows))
    lines.extend(["", "## Tail Median Comparison", ""])

    compare_headers = ["benchmark", *[str(col) for col in compare.columns]]
    compare_rows = []
    for benchmark, row in compare.iterrows():
        compare_rows.append([str(benchmark), *[_format_float(v) for v in row.tolist()]])
    lines.append(_markdown_table(compare_headers, compare_rows))

    lines.extend(["", "## Detailed Summary", ""])
    detail_df = summary_df.sort_values(["benchmark", "tail_median"], ascending=[True, False]).reset_index(drop=True)
    detail_headers = [
        "benchmark",
        "run_label",
        "tail_median",
        "tail_mean",
        "tail_std",
        "last_value",
        "last_training_steps",
        "last_epoch",
    ]
    detail_rows = []
    for _, row in detail_df.iterrows():
        detail_rows.append(
            [
                str(row["benchmark"]),
                str(row["run_label"]),
                _format_float(row["tail_median"]),
                _format_float(row["tail_mean"]),
                _format_float(row["tail_std"]),
                _format_float(row["last_value"]),
                str(int(row["last_training_steps"])),
                str(int(row["last_epoch"])),
            ]
        )
    lines.append(_markdown_table(detail_headers, detail_rows))
    lines.append("")
    return "\n".join(lines)


def write_summary_bar_plot(
    summary_df: pd.DataFrame,
    output: Path,
    *,
    value_column: str = "tail_median",
    solid_run_labels: set[str] | None = None,
) -> None:
    compare = summary_df.pivot(index="benchmark", columns="run_label", values=value_column)
    compare = compare.sort_index(axis=0)

    macro_order = (
        summary_df.groupby("run_label", as_index=True)[value_column]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    compare = compare.reindex(columns=macro_order)
    compare.loc["macro_avg"] = compare.mean(axis=0)
    macro_winner_label: str | None = None
    if not compare.loc["macro_avg"].isna().all():
        macro_winner_label = compare.loc["macro_avg"].astype(float).idxmax()

    benchmarks = compare.index.tolist()
    run_labels = compare.columns.tolist()
    n_benchmarks = len(benchmarks)
    n_runs = len(run_labels)
    if n_benchmarks == 0 or n_runs == 0:
        return

    fig_width = max(10.0, 1.4 * n_benchmarks + 0.9 * n_runs)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    x_positions = list(range(n_benchmarks))
    total_width = 0.82
    bar_width = total_width / max(1, n_runs)
    x_offset_start = -0.5 * total_width + 0.5 * bar_width

    for idx, run_label in enumerate(run_labels):
        offsets = [x + x_offset_start + idx * bar_width for x in x_positions]
        heights = compare[run_label].tolist()
        solid = solid_run_labels is not None and run_label in solid_run_labels
        bars = ax.bar(
            offsets,
            heights,
            width=bar_width,
            label=f"{run_label}*" if macro_winner_label == run_label else run_label,
            linewidth=0.9,
            edgecolor="0.35",
            hatch=None if solid else "xx",
        )
        for bar in bars:
            bar.set_joinstyle("miter")

    if "macro_avg" in benchmarks:
        macro_row = compare.loc["macro_avg"]
        if not macro_row.isna().all():
            y_max = max(float(compare.max().max()), 0.0)
            star_offset = max(0.8, y_max * 0.02)
            best_label = macro_row.astype(float).idxmax()
            run_idx = run_labels.index(best_label)
            for bench_idx, benchmark in enumerate(benchmarks):
                height = float(compare.loc[benchmark, best_label])
                star_x = x_positions[bench_idx] + x_offset_start + run_idx * bar_width
                ax.text(
                    star_x,
                    height + star_offset,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    fontweight="bold",
                    color="black",
                    clip_on=False,
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        ["Macro Avg" if bench == "macro_avg" else pretty_benchmark(bench) for bench in benchmarks]
    )
    summary_meta = summary_df.iloc[0]
    metric_label = pretty_metric(str(summary_meta["summary_metric"]))
    tail_n = int(summary_meta["summary_last_n"])
    stat_label = value_column.replace("tail_", "").replace("_", " ").title()
    ax.set_ylabel(f"{metric_label} ({stat_label.lower()} of last {tail_n})")
    ax.set_title(f"Converged Validation {metric_label} by Benchmark ({stat_label})")
    ax.grid(axis="y", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if macro_winner_label is not None:
        handles.append(Line2D([], [], linestyle="none"))
        labels.append("* = max macro mean")
    ax.legend(handles, labels, fontsize=8, ncol=max(1, min(3, (n_runs + 2) // 3)))
    ax.margins(y=0.08)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.last_n <= 0:
        raise ValueError("--last-n must be a positive integer.")

    snapshots = [Path(p).expanduser().resolve() for p in args.snapshots]
    if args.labels is not None and len(args.labels) != len(snapshots):
        raise ValueError("--labels count must match number of snapshots.")

    labels = list(args.labels) if args.labels is not None else [snap.name for snap in snapshots]
    runs = [(label, snap, load_validation_csv(snap, args.metric)) for label, snap in zip(labels, snapshots)]
    run_dfs = [df for _, _, df in runs]
    benchmarks = determine_benchmarks(run_dfs, args.benchmark)
    if args.format == "long":
        out_df = build_long_export(runs, benchmarks, args)
    elif args.format == "wide":
        out_df = build_wide_export(runs, benchmarks, args)
    else:
        out_df = build_summary_export(runs, benchmarks, args)

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, index=False)
    print(f"Wrote {len(out_df)} rows -> {output}")
    if args.format == "summary":
        print_summary_table(out_df)
        readme_output = Path(args.readme_output).expanduser() if args.readme_output else output.with_suffix(".md")
        readme_output.parent.mkdir(parents=True, exist_ok=True)
        readme_output.write_text(build_summary_markdown(out_df))
        print(f"Wrote Markdown summary -> {readme_output}")
        bar_plot_output = Path(args.bar_plot_output).expanduser() if args.bar_plot_output else output.with_suffix(".png")
        write_summary_bar_plot(out_df, bar_plot_output)
        print(f"Wrote grouped bar plot -> {bar_plot_output}")


if __name__ == "__main__":
    main()
