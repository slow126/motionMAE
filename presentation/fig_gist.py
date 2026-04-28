"""
Illustrative GIST figure:
Left panel  — coverage-only greedy: picks near-duplicates, wastes budget
Right panel — GIST (coverage + diversity): spread across target modes
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.patheffects import withStroke

np.random.seed(42)

BG      = "#0D1B2A"
NAVY_M  = "#132237"
TEAL    = "#00B4D8"
GOLD    = "#F4A261"
RED     = "#E76F51"
GREEN   = "#52B788"
WHITE   = "#FFFFFF"
MUTED   = "#8BA3BF"
OFF_W   = "#E8EFF7"

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG)

# ── Target distribution: 5 clusters of orange dots ──────────────
target_centers = np.array([
    [1.5, 3.5], [3.5, 4.5], [5.5, 2.5], [7.5, 4.0], [6.0, 6.5]
])
target_pts = []
for cx, cy in target_centers:
    pts = np.random.randn(18, 2) * 0.35 + [cx, cy]
    target_pts.append(pts)
target_pts = np.vstack(target_pts)

# ── Pool of candidate training pairs (blue, scattered) ───────────
pool_pts = np.random.uniform(0.5, 8.5, (120, 2))
# Cluster many near center-left (simulating PointOdyssey near-dups)
dense_blob = np.random.randn(60, 2) * 0.5 + [2.0, 2.0]
pool_pts = np.vstack([pool_pts, dense_blob])

def draw_panel(ax, title, selected_idx, subtitle, note_text):
    ax.set_facecolor(NAVY_M)
    ax.set_xlim(0, 9); ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(TEAL); spine.set_linewidth(1.5)

    # Pool points (unselected)
    mask = np.ones(len(pool_pts), dtype=bool)
    mask[selected_idx] = False
    ax.scatter(pool_pts[mask, 0], pool_pts[mask, 1],
               s=22, color=MUTED, alpha=0.4, zorder=2, linewidths=0)

    # Target points
    ax.scatter(target_pts[:, 0], target_pts[:, 1],
               s=38, color=GOLD, alpha=0.85, zorder=3,
               marker="D", linewidths=0)

    # Coverage radius circles around selected training points
    for i in selected_idx:
        circ = Circle(pool_pts[i], radius=0.9,
                      fill=True, facecolor=TEAL, alpha=0.12,
                      edgecolor=TEAL, linewidth=0.8, linestyle="--", zorder=3)
        ax.add_patch(circ)

    # Selected training points
    ax.scatter(pool_pts[selected_idx, 0], pool_pts[selected_idx, 1],
               s=90, color=TEAL, zorder=5, linewidths=1.5,
               edgecolors=WHITE)

    # Count covered target points
    covered = 0
    for tp in target_pts:
        for i in selected_idx:
            if np.linalg.norm(tp - pool_pts[i]) <= 0.9:
                covered += 1
                break

    ax.set_title(title, color=WHITE, fontsize=15, fontweight="bold",
                 pad=10, loc="left")
    ax.text(0.5, -0.06, subtitle, transform=ax.transAxes,
            color=MUTED, fontsize=11, ha="center", style="italic")

    # Coverage stat
    ax.text(8.7, 0.3, f"{covered}/{len(target_pts)}\ncovered",
            color=TEAL, fontsize=10, fontweight="bold", ha="right", va="bottom")

    # Note annotation
    ax.text(0.3, 7.5, note_text, color=GOLD, fontsize=10,
            fontweight="bold", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=NAVY_M,
                      edgecolor=GOLD, linewidth=1.2))

# ── Panel A: coverage-only greedy ────────────────────────────────
# Coverage-only greedy naturally covers all target modes first (one pair
# near each cluster), then the remaining budget collapses into the densest
# pool region (the near-duplicate blob) because there is no diversity penalty.
cov_only_idx = []
# Step 1: one pair near each of the 5 target clusters (covers all modes)
for cx, cy in target_centers:
    dists = np.linalg.norm(pool_pts[:120] - np.array([cx, cy]), axis=1)
    for j in np.argsort(dists):
        if j not in cov_only_idx:
            cov_only_idx.append(int(j))
            break
# Step 2: remaining 15 picks collapse into the dense near-duplicate blob
for j in range(120, 180):
    if len(cov_only_idx) >= 20:
        break
    cov_only_idx.append(j)

draw_panel(axes[0],
           "A.  Coverage-Only Greedy",
           cov_only_idx,
           "Budget = 20 pairs   |   All modes covered once, then 15 collapse into dense blob",
           "All modes covered,\nbut 15/20 pairs\nwasted on near-dups")

# ── Panel B: GIST — diverse + covering ───────────────────────────
# Pick one near each target center, plus some diversity
gist_idx = []
for cx, cy in target_centers:
    dists = np.linalg.norm(pool_pts - np.array([cx, cy]), axis=1)
    # pick closest pool point not already selected
    order = np.argsort(dists)
    for j in order:
        if j not in gist_idx:
            gist_idx.append(j)
            break
# Fill remaining budget with spread-out points
extra_centers = [[4.5, 1.5], [2.0, 6.0], [7.5, 1.5], [8.0, 6.5], [4.0, 7.5]]
for ec in extra_centers:
    dists = np.linalg.norm(pool_pts - np.array(ec), axis=1)
    order = np.argsort(dists)
    for j in order:
        if j not in gist_idx:
            gist_idx.append(j)
            break
# pad to 20
remaining = [i for i in range(len(pool_pts)) if i not in gist_idx]
np.random.shuffle(remaining)
for j in remaining:
    if len(gist_idx) >= 20: break
    # only add if far enough from all selected
    far = all(np.linalg.norm(pool_pts[j] - pool_pts[k]) > 1.2 for k in gist_idx)
    if far:
        gist_idx.append(j)

draw_panel(axes[1],
           "B.  GIST (Coverage + Diversity)",
           gist_idx,
           "Budget = 20 pairs   |   Selected pairs spread across target modes",
           "Covers diverse\ntarget modes")

# ── Legend ────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=GOLD,  label="Target distribution (eval BFV)", alpha=0.85),
    mpatches.Patch(facecolor=TEAL,  label="Selected training pairs",        alpha=0.9),
    mpatches.Patch(facecolor=MUTED, label="Unselected pool",               alpha=0.5),
    mpatches.Patch(facecolor=TEAL,  label="Coverage radius ε",
                   alpha=0.15, edgecolor=TEAL, linewidth=1),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4,
           facecolor=NAVY_M, edgecolor=TEAL, labelcolor=OFF_W,
           fontsize=11, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.0))

fig.suptitle("Coverage-Only vs. GIST: Why Diversity Constraints Matter",
             color=WHITE, fontsize=16, fontweight="bold", y=1.01)

plt.tight_layout(rect=[0, 0.08, 1, 1])
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

plt.savefig(OUT_DIR / "fig_gist_illustration.png",
            dpi=180, bbox_inches="tight", facecolor=BG)
print("Saved fig_gist_illustration.png")
