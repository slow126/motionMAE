"""
Compact illustrative figure for the ClusterCov preliminary coverage probe.

This piggybacks on the visual language of the density-mismatch and GIST figures:
dark navy panels, gold target BFV points, teal selection/coverage overlays, and
simple synthetic geometry that communicates the method steps without requiring
heavy data dependencies.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch


np.random.seed(11)

BG = "#0D1B2A"
NAVY_M = "#132237"
TEAL = "#00B4D8"
GOLD = "#F4A261"
RED = "#E76F51"
GREEN = "#52B788"
WHITE = "#FFFFFF"
MUTED = "#8BA3BF"
OFF_W = "#E8EFF7"
LILAC = "#CDB4DB"


def style_panel(ax) -> None:
    ax.set_facecolor(NAVY_M)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(TEAL)
        spine.set_linewidth(1.4)


def add_panel_title(ax, title: str, subtitle: str) -> None:
    ax.text(
        0.0,
        1.045,
        title,
        transform=ax.transAxes,
        color=WHITE,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    if subtitle:
        ax.text(
            0.01,
            0.925,
            subtitle,
            transform=ax.transAxes,
            color=MUTED,
            fontsize=8.9,
            va="top",
            ha="left",
        )


def score_box(ax, xy: tuple[float, float], title: str, lines: list[str], fc: str) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        3.0,
        1.15,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=NAVY_M,
        edgecolor=fc,
        linewidth=1.4,
        alpha=0.98,
        zorder=6,
    )
    ax.add_patch(box)
    ax.text(x + 0.12, y + 0.95, title, color=fc, fontsize=9.5, fontweight="bold", zorder=7)
    for idx, line in enumerate(lines):
        ax.text(
            x + 0.12,
            y + 0.66 - idx * 0.27,
            line,
            color=OFF_W,
            fontsize=8.5,
            zorder=7,
        )


def text_box(
    ax,
    xy: tuple[float, float],
    text: str,
    *,
    fc: str = OFF_W,
    ec: str = MUTED,
    fontsize: float = 8.8,
    ha: str = "left",
) -> None:
    x, y = xy
    ax.text(
        x,
        y,
        text,
        color=fc,
        fontsize=fontsize,
        ha=ha,
        va="center",
        zorder=8,
        bbox=dict(
            boxstyle="round,pad=0.28,rounding_size=0.06",
            facecolor=NAVY_M,
            edgecolor=ec,
            linewidth=1.0,
            alpha=0.96,
        ),
    )


def main() -> None:
    fig = plt.figure(figsize=(15.5, 5.7), facecolor=BG)
    gs = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        left=0.035,
        right=0.985,
        top=0.84,
        bottom=0.14,
        wspace=0.18,
        width_ratios=[1.0, 1.0, 1.0],
    )

    ax_clusters = fig.add_subplot(gs[0, 0])
    ax_scoring = fig.add_subplot(gs[0, 1])
    ax_greedy = fig.add_subplot(gs[0, 2])
    for ax in (ax_clusters, ax_scoring, ax_greedy):
        style_panel(ax)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")

    target_centers = np.array(
        [
            [1.4, 5.7],
            [3.0, 6.3],
            [4.8, 5.1],
            [6.6, 6.1],
            [7.8, 4.5],
            [6.2, 2.3],
            [3.8, 2.0],
            [1.8, 3.1],
        ]
    )
    target_pts = []
    for c in target_centers:
        pts = np.random.randn(18, 2) * [0.26, 0.23] + c
        target_pts.append(pts)
    target_pts = np.vstack(target_pts)

    # Panel 2 coverage state: clusters already covered by previous greedy picks
    p2_covered = {0, 2, 5, 6, 7}
    p2_uncovered = sorted(set(range(len(target_centers))) - p2_covered)  # [1, 3, 4]

    # Panel 3 final coverage state
    greedy_final = {0, 2, 3, 5, 6, 7}

    # ── PANEL 1: Cluster Target BFV Space ────────────────────────────
    # At this step we are only clustering — no pairs have been selected yet,
    # so all cluster centroids are shown uniformly (teal).
    ax_clusters.scatter(
        target_pts[:, 0],
        target_pts[:, 1],
        s=18,
        color=GOLD,
        alpha=0.72,
        linewidths=0,
        zorder=2,
    )
    for idx, center in enumerate(target_centers):
        ax_clusters.add_patch(
            Circle(
                center,
                radius=0.55,
                facecolor=TEAL,
                edgecolor=WHITE,
                linewidth=1.2,
                alpha=0.22,
                zorder=3,
            )
        )
        ax_clusters.scatter(
            [center[0]],
            [center[1]],
            s=120,
            marker="X",
            color=TEAL,
            edgecolors=WHITE,
            linewidths=0.8,
            zorder=4,
        )
        ax_clusters.text(
            center[0],
            center[1] + 0.52,
            str(idx + 1),
            color=WHITE,
            fontsize=8.5,
            ha="center",
            va="bottom",
            fontweight="bold",
            zorder=5,
        )
    add_panel_title(
        ax_clusters,
        "1. Cluster Target BFV Space",
        "",
    )
    text_box(
        ax_clusters,
        (0.35, 7.45),
        f"k-means  →  K = {len(target_centers)} motion clusters",
        fc=MUTED,
        ec=MUTED,
        fontsize=8.9,
    )
    ax_clusters.text(
        0.03,
        0.03,
        "Gold = target BFV descriptors     Teal X = cluster centroid",
        transform=ax_clusters.transAxes,
        color=MUTED,
        fontsize=8.5,
        ha="left",
        va="bottom",
    )

    # ── PANEL 2: Score Candidate Pairs ───────────────────────────────
    # Pair A (dense, 128 correspondences): BFV descriptors land near
    #   uncovered clusters 1 and 3  →  covers 2 new clusters
    #   normalized gain = 2 / 128 = 0.016  (low — many corr, few new modes)
    # Pair B (sparse, 12 correspondences): BFV descriptors land near
    #   uncovered cluster 4 only  →  covers 1 new cluster
    #   normalized gain = 1 / 12 = 0.083  (high — selected)
    add_panel_title(
        ax_scoring,
        "2. Score Candidate Pairs",
        "",
    )

    # Formula banner
    ax_scoring.add_patch(FancyBboxPatch(
        (0.55, 7.03),
        6.8,
        0.5,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=NAVY_M,
        edgecolor=GOLD,
        linewidth=1.1,
        alpha=0.98,
        zorder=6,
    ))
    ax_scoring.text(
        0.78,
        7.27,
        "score(pair) = new target clusters hit  ÷  |pair|",
        color=OFF_W,
        fontsize=8.3,
        ha="left",
        va="center",
        zorder=7,
    )

    # Faint target BFV cloud for spatial context
    ax_scoring.scatter(
        target_pts[:, 0],
        target_pts[:, 1],
        s=10,
        color=GOLD,
        alpha=0.18,
        linewidths=0,
        zorder=1,
    )

    # Cluster circles: uncovered = red dashed + red X, already covered = muted dim
    for idx, center in enumerate(target_centers):
        if idx in p2_uncovered:
            ax_scoring.add_patch(
                Circle(
                    center,
                    radius=0.52,
                    fill=False,
                    edgecolor=RED,
                    linewidth=1.4,
                    linestyle="--",
                    alpha=0.9,
                    zorder=2,
                )
            )
            ax_scoring.scatter(
                [center[0]],
                [center[1]],
                s=75,
                marker="X",
                color=RED,
                edgecolors=WHITE,
                linewidths=0.7,
                zorder=3,
            )
        else:
            ax_scoring.add_patch(
                Circle(
                    center,
                    radius=0.46,
                    fill=False,
                    edgecolor=MUTED,
                    linewidth=0.7,
                    linestyle=":",
                    alpha=0.3,
                    zorder=2,
                )
            )

    # Coverage halos: semi-transparent fill shows which clusters each pair would cover
    # Pair A → teal halos on clusters 1 and 3
    for cid in [1, 3]:
        ax_scoring.add_patch(
            Circle(
                target_centers[cid],
                radius=0.60,
                facecolor=TEAL,
                edgecolor="none",
                alpha=0.20,
                zorder=2,
            )
        )
    # Pair B → green halo on cluster 4
    ax_scoring.add_patch(
        Circle(
            target_centers[4],
            radius=0.60,
            facecolor=GREEN,
            edgecolor="none",
            alpha=0.20,
            zorder=2,
        )
    )

    # Pair A: dense — two groups of teal circles near clusters 1 and 3
    pair_a_c1 = np.array([
        [2.78, 6.15], [2.88, 6.42], [3.05, 6.52], [3.22, 6.18],
        [2.82, 6.05], [3.18, 6.40], [2.65, 6.28], [3.12, 6.25],
    ])
    pair_a_c3 = np.array([
        [6.38, 6.22], [6.52, 6.42], [6.78, 6.32], [6.95, 6.05],
        [6.42, 5.92], [6.88, 5.96], [6.62, 6.08],
    ])
    pair_a = np.vstack([pair_a_c1, pair_a_c3])

    # Pair B: sparse — one small group of green squares near cluster 4
    pair_b = np.array([
        [7.58, 4.62], [7.88, 4.72], [8.05, 4.32], [7.72, 4.25],
    ])

    ax_scoring.scatter(
        pair_a[:, 0],
        pair_a[:, 1],
        s=42,
        color=TEAL,
        alpha=0.95,
        edgecolors=WHITE,
        linewidths=0.5,
        marker="o",
        zorder=5,
    )
    # Label both sub-groups so it's clear they belong to the same pair
    for sub_pts in [pair_a_c1, pair_a_c3]:
        ax_scoring.text(
            sub_pts[:, 0].mean(),
            sub_pts[:, 1].mean() - 0.52,
            "A",
            color=TEAL,
            fontsize=9.0,
            ha="center",
            fontweight="bold",
            zorder=6,
        )

    ax_scoring.scatter(
        pair_b[:, 0],
        pair_b[:, 1],
        s=52,
        color=GREEN,
        alpha=0.95,
        edgecolors=WHITE,
        linewidths=0.6,
        marker="s",
        zorder=5,
    )
    ax_scoring.text(
        pair_b[:, 0].mean(),
        pair_b[:, 1].mean() - 0.52,
        "B",
        color=GREEN,
        fontsize=9.0,
        ha="center",
        fontweight="bold",
        zorder=6,
    )

    # Score boxes: Pair A (teal border = lower gain) vs Pair B (gold border = winner)
    score_box(
        ax_scoring,
        (0.2, 1.2),
        "Pair A  (dense)",
        ["2 new clusters", "2 \u00f7 128 = 0.016"],
        TEAL,
    )
    score_box(
        ax_scoring,
        (3.7, 1.2),
        "Pair B  (sparse)",
        ["1 new cluster", "1 \u00f7 12 = 0.083  \u2713"],
        GOLD,
    )

    ax_scoring.text(
        0.03,
        0.03,
        "Red dashed = uncovered cluster     Muted = already covered",
        transform=ax_scoring.transAxes,
        color=MUTED,
        fontsize=8.5,
        ha="left",
        va="bottom",
    )

    add_panel_title(
        ax_greedy,
        "3. Greedy Selection Covers Modes",
        "",
    )
    text_box(
        ax_greedy,
        (0.65, 7.05),
        "Greedy picks convert red to teal",
        fc=MUTED,
        ec=MUTED,
        fontsize=8.0,
    )
    step_positions = [1.9, 5.0, 8.1]
    step_labels = ["start", "mid", "final"]
    step_status = [
        {0, 6},
        {0, 2, 3, 6},
        greedy_final,
    ]
    y_rows = np.linspace(5.8, 1.6, len(target_centers))
    for x, label, status in zip(step_positions, step_labels, step_status):
        ax_greedy.text(x, 6.55, label, color=WHITE, fontsize=9.0, fontweight="bold", ha="center", va="bottom")
        for idx, y in enumerate(y_rows):
            covered = idx in status
            fill = GREEN if covered else RED
            ax_greedy.add_patch(
                Circle(
                    (x, y),
                    radius=0.34,
                    facecolor=fill,
                    edgecolor=WHITE,
                    linewidth=1.0,
                    alpha=0.9 if covered else 0.45,
                    zorder=3,
                )
            )
            ax_greedy.text(
                x,
                y,
                str(idx + 1),
                color=WHITE,
                fontsize=8.2,
                ha="center",
                va="center",
                fontweight="bold",
                zorder=4,
            )
        if x != step_positions[-1]:
            ax_greedy.annotate(
                "",
                xy=(x + 1.25, 4.0),
                xytext=(x + 0.45, 4.0),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.4),
                zorder=4,
            )

    ax_greedy.text(0.3, 0.58, "uncovered", color=RED, fontsize=8.5, fontweight="bold")
    ax_greedy.text(3.2, 0.58, "covered", color=GREEN, fontsize=8.5, fontweight="bold")
    text_box(
        ax_greedy,
        (5.8, 0.58),
        "Coverage grows across modes.",
        fc=GOLD,
        ec=GOLD,
        fontsize=8.1,
    )

    legend_handles = [
        Line2D([], [], marker="o", linestyle="None", color=GOLD, markersize=8, label="Target BFV descriptors"),
        Line2D([], [], marker="X", linestyle="None", color=TEAL, markersize=9, label="Target cluster centroid"),
        Line2D([], [], marker="o", linestyle="None", color=TEAL, markeredgecolor=WHITE, markersize=7, label="Dense pair"),
        Line2D([], [], marker="s", linestyle="None", color=GREEN, markeredgecolor=WHITE, markersize=7, label="Sparse pair"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=10.2,
        facecolor=NAVY_M,
        edgecolor=TEAL,
        labelcolor=OFF_W,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.suptitle(
        "ClusterCov: Cluster Target Motion, Score Uncovered Modes, Greedily Build Coverage",
        color=WHITE,
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fig_clustercov_probe.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"Saved {out_path.name}")


if __name__ == "__main__":
    main()
