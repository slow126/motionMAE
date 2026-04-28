"""
Illustrative density mismatch figure:
Shows the core problem with heterogeneous correspondence pools —
sparse keypoints vs dense optical flow look completely different
after rasterization into BFV space.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle

np.random.seed(7)

BG      = "#0D1B2A"
NAVY_M  = "#132237"
TEAL    = "#00B4D8"
GOLD    = "#F4A261"
RED     = "#E76F51"
GREEN   = "#52B788"
WHITE   = "#FFFFFF"
MUTED   = "#8BA3BF"
OFF_W   = "#E8EFF7"

fig = plt.figure(figsize=(16, 7), facecolor=BG)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.35,
                       left=0.05, right=0.97, top=0.88, bottom=0.12)

# ── Helper: draw a fake image pair with correspondences ──────────
def draw_image_pair(ax, n_pts, color, title, subtitle, alpha=0.85):
    ax.set_facecolor(NAVY_M)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(1.5)

    # Fake "image" as grey texture
    tex = np.random.rand(20, 20) * 0.15 + 0.1
    ax.imshow(tex, extent=[0,1,0,1], cmap="Blues", alpha=0.25,
              aspect="auto", origin="lower", vmin=0, vmax=1)

    # Draw correspondences as colored dots + lines
    src = np.random.uniform(0.1, 0.9, (n_pts, 2))
    disp = np.random.randn(n_pts, 2) * 0.08
    dst = np.clip(src + disp, 0.05, 0.95)

    for s, d in zip(src, dst):
        ax.annotate("", xy=d, xytext=s,
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=0.8, mutation_scale=6), zorder=4)
    ax.scatter(src[:,0], src[:,1], s=18, color=color, zorder=5, linewidths=0)

    ax.set_title(title, color=WHITE, fontsize=10.5, fontweight="bold", pad=4)
    ax.text(0.5, -0.16, subtitle, transform=ax.transAxes,
            color=MUTED, fontsize=9, ha="center", style="italic")

# ── Helper: draw BFV scatter ──────────────────────────────────────
def draw_bfv(ax, n_pts, color, title, compact=False, alpha=0.6):
    ax.set_facecolor(NAVY_M)
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(1.2)
    ax.set_facecolor(NAVY_M)

    if compact:
        # Sparse: a few tight clusters in specific BFV locations
        centers = np.random.uniform(0.15, 0.85, (max(1, n_pts//5), 2))
        pts = []
        for c in centers:
            pts.append(np.random.randn(5, 2)*0.04 + c)
        pts = np.vstack(pts)[:n_pts]
    else:
        # Dense: fills BFV space more uniformly (many flow vectors)
        pts = np.random.uniform(0.05, 0.95, (n_pts, 2))
        # Add spatial structure — more vectors in some directions
        bias = np.random.randn(n_pts//3, 2)*0.12 + [0.6, 0.7]
        pts = np.vstack([pts, bias])[:n_pts]

    ax.scatter(pts[:,0], pts[:,1], s=8, color=color, alpha=alpha,
               zorder=3, linewidths=0)

    # Count annotation
    ax.text(0.97, 0.03, f"n = {n_pts:,}", transform=ax.transAxes,
            color=color, fontsize=9, ha="right", va="bottom", fontweight="bold")

    ax.set_title(title, color=WHITE, fontsize=10.5, fontweight="bold", pad=4)
    ax.set_xlabel("x̂  (norm. position)", color=MUTED, fontsize=8)
    ax.set_ylabel("Δx̂  (norm. disp.)",   color=MUTED, fontsize=8)

# ── Top row: image pair examples ─────────────────────────────────
ax_sp_img  = fig.add_subplot(gs[0, 0])
ax_sp_bfv  = fig.add_subplot(gs[0, 1])
ax_dn_img  = fig.add_subplot(gs[0, 2])
ax_dn_bfv  = fig.add_subplot(gs[0, 3])

draw_image_pair(ax_sp_img, 12, GOLD,
                "SPair-71k pair", "~12 keypoints / pair")
draw_bfv(ax_sp_bfv, 12, GOLD,
         "BFV cloud (sparse)", compact=True)

draw_image_pair(ax_dn_img, 180, TEAL,
                "PointOdyssey pair", "~128+ flow vectors / pair")
draw_bfv(ax_dn_bfv, 400, TEAL,
         "BFV cloud (dense)", compact=False)

# ── Bottom row: the problem + consequence ─────────────────────────
ax_prob = fig.add_subplot(gs[1, :2])
ax_cons = fig.add_subplot(gs[1, 2:])

# Problem panel — bar chart showing density ratio
ax_prob.set_facecolor(NAVY_M)
for sp in ax_prob.spines.values():
    sp.set_edgecolor(MUTED); sp.set_linewidth(0.8)

sources = ["PF-PASCAL", "SPair-71k", "PointOdyssey"]
kpts    = [12, 12, 128]
colors  = [RED, GOLD, TEAL]
bars = ax_prob.barh(sources, kpts, color=colors, alpha=0.85, height=0.5)
for bar, v in zip(bars, kpts):
    ax_prob.text(v + 2, bar.get_y() + bar.get_height()/2,
                 f"~{v} kpts/pair", va="center", color=WHITE, fontsize=10)
ax_prob.set_xlim(0, 165)
ax_prob.set_xlabel("Keypoints / pair (supervision density)", color=MUTED, fontsize=10)
ax_prob.set_title("Supervision Density Mismatch Across Sources",
                  color=WHITE, fontsize=12, fontweight="bold", pad=6)
ax_prob.tick_params(colors=MUTED)
ax_prob.set_facecolor(NAVY_M)
ax_prob.axvline(x=128, color=TEAL, linestyle="--", linewidth=1, alpha=0.5)

# Consequence panel — latent space clustering
ax_cons.set_facecolor(NAVY_M)
for sp in ax_cons.spines.values():
    sp.set_edgecolor(MUTED); sp.set_linewidth(0.8)

# Simulate UMAP-like 2D embedding that clusters by dataset identity
n = 80
sp_pts  = np.random.randn(n, 2)*0.6 + [-2.5,  1.5]
po_pts  = np.random.randn(n, 2)*0.8 + [ 2.0, -1.0]
pf_pts  = np.random.randn(n//2, 2)*0.4 + [-1.5, -2.5]

ax_cons.scatter(sp_pts[:,0],  sp_pts[:,1],  s=18, color=GOLD,  alpha=0.7,
                label="SPair-71k",    linewidths=0, zorder=3)
ax_cons.scatter(po_pts[:,0],  po_pts[:,1],  s=18, color=TEAL,  alpha=0.7,
                label="PointOdyssey", linewidths=0, zorder=3)
ax_cons.scatter(pf_pts[:,0],  pf_pts[:,1],  s=18, color=RED,   alpha=0.7,
                label="PF-PASCAL",   linewidths=0, zorder=3)

# Ellipse annotations
from matplotlib.patches import Ellipse
for center, w, h, angle, col, lbl in [
    ([-2.5, 1.5], 2.8, 2.2, 10,  GOLD, "Dataset A"),
    ([ 2.0,-1.0], 3.5, 2.8, -5,  TEAL, "Dataset B"),
    ([-1.5,-2.5], 2.0, 1.8,  0,  RED,  "Dataset C"),
]:
    e = Ellipse(center, w, h, angle=angle,
                fill=False, edgecolor=col, linewidth=1.5, linestyle="--",
                alpha=0.6, zorder=2)
    ax_cons.add_patch(e)

ax_cons.set_xticks([]); ax_cons.set_yticks([])
ax_cons.set_title("MAE Latent Space Clusters by Dataset Identity\n(not motion structure)",
                  color=WHITE, fontsize=12, fontweight="bold", pad=6)
ax_cons.legend(facecolor=NAVY_M, edgecolor=MUTED, labelcolor=OFF_W,
               fontsize=9, loc="upper right", framealpha=0.8)
ax_cons.text(0.5, -0.08,
             "Density mismatch causes encoder to rely on dataset statistics, not motion",
             transform=ax_cons.transAxes,
             color=GOLD, fontsize=10, ha="center", style="italic")

# ── Super title ───────────────────────────────────────────────────
fig.suptitle("The Density Mismatch Problem: Sparse Keypoints vs. Dense Optical Flow",
             color=WHITE, fontsize=15, fontweight="bold", y=0.98)

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

plt.savefig(OUT_DIR / "fig_density_mismatch.png",
            dpi=180, bbox_inches="tight", facecolor=BG)
print("Saved fig_density_mismatch.png")
