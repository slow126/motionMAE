#!/usr/bin/env python3
"""Plot training loss curves from TensorBoard event files in snapshot folders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


TAG_MAP = {
    "step": "train/loss_step",
    "epoch": "train/loss_epoch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train loss curves from snapshot TensorBoard logs.")
    parser.add_argument(
        "snapshots",
        nargs="+",
        help="Paths to snapshot directories containing TensorBoard event files.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        metavar="LABEL",
        help="Labels for runs in legend; if set, count must match snapshots.",
    )
    parser.add_argument(
        "--series",
        choices=["step", "epoch"],
        default="step",
        help="Which TensorBoard train-loss series to plot (default: step).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Optional rolling mean window over plotted points (default: 1 = no smoothing).",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=None,
        help="Optional EMA smoothing factor in (0, 1]; e.g. 0.05 or 0.1.",
    )
    parser.add_argument(
        "--output",
        default="training_loss_curve.png",
        help="Path to save figure.",
    )
    return parser.parse_args()


def find_event_file(snapshot: Path) -> Path:
    candidates = sorted(snapshot.rglob("events.out.tfevents.*"))
    if not candidates:
        candidates = sorted(snapshot.rglob("*.tfevents.*"))
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event file found under {snapshot}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_scalar_series(snapshot: Path, tag: str) -> pd.DataFrame:
    event_path = find_event_file(snapshot)
    ea = event_accumulator.EventAccumulator(str(event_path))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        raise KeyError(f"Scalar tag {tag!r} not found in {event_path}")
    events = ea.Scalars(tag)
    if not events:
        raise ValueError(f"No scalar events for tag {tag!r} in {event_path}")
    return pd.DataFrame(
        {
            "step": [int(e.step) for e in events],
            "value": [float(e.value) for e in events],
        }
    )


def maybe_smooth(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1 or df.empty:
        return df
    out = df.copy()
    out["value"] = out["value"].rolling(window=window, min_periods=1).mean()
    return out


def maybe_ema_smooth(df: pd.DataFrame, alpha: float | None) -> pd.DataFrame:
    if alpha is None or df.empty:
        return df
    if not (0.0 < alpha <= 1.0):
        raise ValueError("--ema-alpha must be in the range (0, 1].")
    out = df.copy()
    out["value"] = out["value"].ewm(alpha=alpha, adjust=False).mean()
    return out


def plot_curves(runs: Sequence[tuple[str, pd.DataFrame]], series: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "x", "^", "s", "D", "v", "P", "*"]
    linestyles = ["-", "--", "-.", ":"]

    for idx, (label, df) in enumerate(runs):
        ax.plot(
            df["step"],
            df["value"],
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            markersize=3,
            linewidth=1.5,
            label=label,
        )

    ax.set_title(f"Training Loss ({series})")
    ax.set_xlabel("training_steps" if series == "step" else "epoch_step")
    ax.set_ylabel("train_loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    snapshots = [Path(p) for p in args.snapshots]
    if args.labels is not None and len(args.labels) != len(snapshots):
        raise ValueError("--labels count must match snapshot count")
    if args.ema_alpha is not None and args.smooth_window > 1:
        raise ValueError("Use either --smooth-window or --ema-alpha, not both.")

    tag = TAG_MAP[args.series]
    labels = args.labels or [p.name for p in snapshots]
    runs = []
    for label, snapshot in zip(labels, snapshots):
        df = load_scalar_series(snapshot, tag=tag)
        df = maybe_smooth(df, window=max(1, int(args.smooth_window)))
        df = maybe_ema_smooth(df, alpha=args.ema_alpha)
        runs.append((label, df))

    output = Path(args.output)
    plot_curves(runs, series=args.series, output=output)
    print(f"Saved training-loss plot -> {output}")


if __name__ == "__main__":
    main()
