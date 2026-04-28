#!/usr/bin/env python3
"""
Generate separate toy 2D point-set illustrations for the four directed-coverage regimes.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


OUT_DIR = Path(__file__).resolve().parent / "figures"

BG = "#F0F4F8"
TRAIN_C = "#00B4D8"
EVAL_C = "#F4A261"
GRID_C = "#D7E3F1"
TEXT_C = "#0D1B2A"
MUTED_C = "#4A6785"


def hard_coverage(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    return float((d.min(axis=1) <= eps).mean())


def make_sets():
    rng = np.random.default_rng(4)
    eval_core = np.array([
        [0.28, 0.68], [0.36, 0.62], [0.46, 0.66], [0.42, 0.53],
        [0.31, 0.47], [0.54, 0.49], [0.62, 0.58], [0.59, 0.72],
    ], dtype=np.float32)
    return {
        "well_aligned": (
            "Well-Aligned",
            eval_core + rng.normal(scale=0.025, size=eval_core.shape),
            eval_core,
        ),
        "excess_train_mass": (
            "Excess Train Mass",
            np.concatenate([
                eval_core + rng.normal(scale=0.03, size=eval_core.shape),
                np.array([
                    [0.10, 0.15], [0.22, 0.18], [0.84, 0.18], [0.90, 0.28],
                    [0.14, 0.88], [0.82, 0.86], [0.72, 0.24], [0.18, 0.78],
                ], dtype=np.float32)
            ], axis=0),
            eval_core,
        ),
        "eval_under_covered": (
            "Eval Under-Covered",
            np.array([
                [0.27, 0.67], [0.36, 0.61], [0.45, 0.65], [0.32, 0.47],
                [0.43, 0.54],
            ], dtype=np.float32),
            eval_core,
        ),
        "complete_mismatch": (
            "Complete Mismatch",
            np.array([
                [0.12, 0.18], [0.18, 0.28], [0.24, 0.20], [0.76, 0.18],
                [0.84, 0.26], [0.74, 0.30], [0.18, 0.84], [0.82, 0.78],
            ], dtype=np.float32),
            eval_core,
        ),
    }


def draw_case(title: str, train_pts: np.ndarray, eval_pts: np.ndarray, out_path: Path, eps: float = 0.12):
    fig, ax = plt.subplots(figsize=(4.4, 1.9), facecolor=BG)
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.11, top=0.78)

    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#B9CDE3")
        spine.set_linewidth(1.1)

    for x in np.linspace(0.2, 0.8, 4):
        ax.axvline(x, color=GRID_C, lw=0.6, zorder=0)
        ax.axhline(x, color=GRID_C, lw=0.6, zorder=0)

    for p in eval_pts:
        ax.add_patch(Circle(p, eps, fill=False, ec=EVAL_C, lw=1.0, alpha=0.23, zorder=1))

    ax.scatter(train_pts[:, 0], train_pts[:, 1], s=34, c=TRAIN_C, edgecolors="white", linewidths=0.8, zorder=3)
    ax.scatter(eval_pts[:, 0], eval_pts[:, 1], s=34, c=EVAL_C, edgecolors="white", linewidths=0.8, zorder=4)

    et = hard_coverage(eval_pts, train_pts, eps)
    te = hard_coverage(train_pts, eval_pts, eps)

    def tag(val: float) -> str:
        return "High" if val >= 0.75 else "Low"

    fig.text(0.05, 0.93, title, fontsize=10.5, fontweight="bold", color=TEXT_C)
    fig.text(0.05, 0.865, f"E→T: {tag(et)}   T→E: {tag(te)}", fontsize=8.5, color=MUTED_C, fontweight="bold")
    fig.text(0.58, 0.865, "●", fontsize=10, color=TRAIN_C, fontweight="bold")
    fig.text(0.615, 0.865, "Train", fontsize=7.8, color=MUTED_C)
    fig.text(0.76, 0.865, "●", fontsize=10, color=EVAL_C, fontweight="bold")
    fig.text(0.795, 0.865, "Eval", fontsize=7.8, color=MUTED_C)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    for stem, (title, train_pts, eval_pts) in make_sets().items():
        out = OUT_DIR / f"fig_coverage_regime__{stem}.png"
        draw_case(title, train_pts, eval_pts, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
