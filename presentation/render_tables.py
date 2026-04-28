"""
Render three LaTeX tables as clean matplotlib figures for embedding in slides.
Tables are simplified for presentation clarity — key rows/columns only, bold callouts.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

BG      = "#FFFFFF"
NAVY    = "#0D1B2A"
TEAL    = "#00B4D8"
GOLD    = "#F4A261"
GREEN   = "#52B788"
MUTED   = "#8BA3BF"
LIGHT   = "#F0F4F8"
HEADER  = "#132237"
BOLD_ROW= "#DDF3FA"   # stronger highlight row
RED     = "#E76F51"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


def parse_numeric(cell):
    text = str(cell).replace("%", "").replace("+", "").replace("−", "-").strip()
    try:
        return float(text)
    except ValueError:
        return None


def compute_best_cells_global(rows, metric_cols):
    best_cells = []
    for col in metric_cols:
        best_val = None
        best_row = None
        for i, row in enumerate(rows):
            if isinstance(row[0], str) and row[0].startswith("__"):
                continue
            val = parse_numeric(row[col])
            if val is None:
                continue
            if best_val is None or val > best_val:
                best_val = val
                best_row = i
        if best_row is not None:
            best_cells.append((best_row, col))
    return best_cells


def compute_best_cells_by_section(rows, metric_cols):
    best_cells = []
    current_block = []

    def flush_block(block):
        if not block:
            return
        for col in metric_cols:
            best_val = None
            best_row = None
            for i in block:
                val = parse_numeric(rows[i][col])
                if val is None:
                    continue
                if best_val is None or val > best_val:
                    best_val = val
                    best_row = i
            if best_row is not None:
                best_cells.append((best_row, col))

    for i, row in enumerate(rows):
        if isinstance(row[0], str) and row[0].startswith("__"):
            flush_block(current_block)
            current_block = []
            continue
        current_block.append(i)
    flush_block(current_block)
    return best_cells


def make_table(ax, headers, rows, col_widths, row_colors=None,
               header_bg=HEADER, header_fg="white",
               fontsize=9, title="", bold_rows=None, col_aligns=None,
               star_cells=None, row_h=0.74, section_row_h=0.58,
               title_pad=3, focal_bold_cols=None,
               section_bg="#E2ECF4", section_fg=NAVY, section_bold=False):
    """star_cells: list of (row_idx, col_idx) tuples to render in TEAL bold."""
    ax.axis("off")
    ax.set_xlim(0, sum(col_widths))
    n_rows = len(rows)
    row_heights = [section_row_h if (isinstance(row[0], str) and row[0].startswith("__")) else row_h for row in rows]
    total_h = sum(row_heights) + row_h
    title_space = 0.0 if not title else 0.36
    ax.set_ylim(0, total_h + title_space)

    x_starts = [sum(col_widths[:i]) for i in range(len(col_widths))]
    if col_aligns is None:
        col_aligns = ["left"] + ["center"] * (len(headers) - 1)
    if star_cells is None:
        star_cells = []
    if focal_bold_cols is None:
        focal_bold_cols = []

    # Header row
    for j, (hdr, w, x0) in enumerate(zip(headers, col_widths, x_starts)):
        ax.add_patch(FancyBboxPatch((x0, total_h - row_h), w, row_h,
            boxstyle="square,pad=0", fc=header_bg, ec="white", lw=0.5))
        ax.text(x0 + w*0.5, total_h - row_h*0.5, hdr,
            ha="center", va="center", fontsize=fontsize,
            color=header_fg, fontweight="bold", wrap=True,
            multialignment="center")

    # Data rows
    y_cursor = total_h - row_h
    for i, row in enumerate(rows):
        current_h = row_heights[i]
        y = y_cursor - current_h
        is_bold = bold_rows and i in bold_rows
        is_section = isinstance(row[0], str) and row[0].startswith("__")

        if is_section:
            label = row[0][2:]
            ax.add_patch(FancyBboxPatch((0, y), sum(col_widths), current_h,
                boxstyle="square,pad=0", fc=section_bg, ec="white", lw=0.3))
            ax.text(0.18, y + current_h*0.5, label,
                ha="left", va="center", fontsize=fontsize - (0 if section_bold else 0.5),
                color=section_fg,
                fontstyle="normal",
                fontweight="bold" if section_bold else "normal")
            y_cursor = y
            continue

        bg = BOLD_ROW if is_bold else (LIGHT if i % 2 == 0 else BG)
        if row_colors and i < len(row_colors) and row_colors[i]:
            bg = row_colors[i]

        for j, (cell, w, x0) in enumerate(zip(row, col_widths, x_starts)):
            ax.add_patch(FancyBboxPatch((x0, y), w, current_h,
                boxstyle="square,pad=0", fc=bg, ec="white", lw=0.3))
            align = col_aligns[j] if j < len(col_aligns) else "center"
            xpos = x0 + (0.1 if align == "left" else w * 0.5)
            cell_str = str(cell)
            if (i, j) in star_cells:
                fc = TEAL; fw = "bold"
            elif is_bold and j in focal_bold_cols:
                fc = NAVY; fw = "bold"
            else:
                fc = NAVY; fw = "normal"
            ax.text(xpos, y + current_h*0.5, cell_str,
                ha=align, va="center", fontsize=fontsize,
                color=fc, fontweight=fw)
        y_cursor = y

    # Horizontal lines
    line_positions = [total_h, total_h - row_h]
    y_cursor = total_h - row_h
    for h in row_heights:
        y_cursor -= h
        line_positions.append(y_cursor)
    for i, y_line in enumerate(line_positions):
        lw = 1.2 if i in (0, 1, len(line_positions) - 1) else 0.3
        ax.axhline(y_line, color="#CCDDEE", lw=lw)

    # Light vertical separators make wide tables easier to scan.
    for x in x_starts[1:]:
        ax.axvline(x, ymin=0, ymax=1, color="#EEF4FA", lw=0.45)

    if title:
        ax.set_title(title, fontsize=fontsize+1, color=NAVY,
                     fontweight="bold", loc="left", pad=title_pad)


# ── TABLE 1: Transfer Grid — compact but complete for shown train sources ────
# Include each train source's own validation target so the slide doesn't hide
# the "easy" columns, but keep the layout tight enough for presentation use.
fig1, ax1 = plt.subplots(figsize=(13.8, 2.4))
fig1.patch.set_facecolor(BG)

headers1 = [
    "Train Set", "Type",
    "KITTI-12", "KITTI-15",
    "SPair", "TSS", "PF-PASCAL", "PF-WILLOW",
    "PointOd.", "FlyingThings",
    "Middlebury", "SDF-Fractal3D",
]
rows1 = [
    ["Sintel",                  "Flow",     "94.8", "91.4", "10.0", "44.6", "27.9", "30.7", "59.7", "73.5", "53.8", "26.8"],
    ["FlyingThings3D",          "Flow",     "96.2", "93.4", "10.0", "44.8", "29.0", "31.6", "55.0", "80.6", "53.9", "27.0"],
    ["SPair-71k",               "Semantic", "40.4", "41.9", "13.0", "27.6", "18.1", "21.7", "23.6", "28.5", "23.3", "17.7"],
    ["PointOdyssey",            "Tracking", "91.2", "88.9", "10.5", "41.4", "25.6", "28.2", "70.4", "68.9", "54.1", "25.6"],
    ["ImageNet 2D Warp",        "Flow",     "68.5", "72.7", "10.0", "34.4", "20.5", "23.9", "36.4", "56.3", "41.1", "24.2"],
    ["SDF-Fractal3D",           "Flow",     "89.0", "90.5", "11.3", "42.7", "27.4", "29.9", "46.6", "68.9", "48.1", "34.0"],
    ["SDF-Fractal3D (2D Warp)", "Flow",     "86.1", "88.9", "11.6", "39.3", "27.7", "30.0", "52.0", "70.8", "49.2", "29.8"],
    ["SDF-Fractal3D (Zoom) ★",  "Flow",     "95.7", "94.6", "11.0", "46.3", "28.4", "32.0", "48.8", "72.3", "51.6", "54.1"],
]
col_w1 = [2.25, 1.0, 0.82, 0.82, 0.72, 0.72, 0.88, 0.88, 0.88, 0.98, 0.88, 1.02]

star1 = compute_best_cells_global(rows1, metric_cols=list(range(2, len(headers1))))

make_table(ax1, headers1, rows1, col_w1,
           title="",
           bold_rows=[7], star_cells=star1,
           col_aligns=["left","center"] + ["center"]*(len(headers1)-2),
           fontsize=8.6, row_h=0.53, section_row_h=0.45, title_pad=1,
           focal_bold_cols=[0, 1])
plt.tight_layout(pad=0.04)
plt.savefig(OUT_DIR / "table_transfer_grid.png", dpi=180,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved table_transfer_grid.png")


# ── TABLE 2: Ranking — simplified (key configurations only) ──────────────────
fig2, ax2 = plt.subplots(figsize=(12, 3.7))
fig2.patch.set_facecolor(BG)

headers2 = ["Configuration", "Predictors", "Hold-out Train", "Hold-out Eval", "Hold-out Both"]
rows2 = [
    ["__Symmetric (baseline)"],
    ["Appearance MMD",           "MMD×1",   "46.2%", "44.3%", "41.3%"],
    ["Flow MMD",                 "MMD×1",   "58.7%", "58.7%", "58.7%"],
    ["__Directed — single direction"],
    ["Flow (Eval→Train)",        "F×1",     "61.7%", "61.7%", "61.7%"],
    ["Appearance (Eval→Train)",  "A×1",     "54.0%", "58.7%", "54.6%"],
    ["__Directed — bidirectional"],
    ["Flow, bidirectional",      "F×2",     "65.4%", "65.6%", "65.5%"],
    ["Appearance, bidirectional","A×2",     "63.5%", "63.9%", "61.9%"],
    ["__Best model"],
    ["Flow + Appearance bidir. ★","F×2+A×2","70.1%", "69.8%", "69.4%"],
]
col_w2 = [4.5, 1.6, 1.55, 1.55, 1.55]
bold2 = [9]  # F×2+A×2 row
star2 = compute_best_cells_global(rows2, metric_cols=[2, 3, 4])
make_table(ax2, headers2, rows2, col_w2,
           title="",
           bold_rows=bold2, star_cells=star2,
           col_aligns=["left","center","center","center","center"],
           fontsize=9.9, row_h=0.60, section_row_h=0.46, title_pad=2,
           focal_bold_cols=[0, 1])
plt.tight_layout(pad=0.04)
plt.savefig(OUT_DIR / "table_ranking.png", dpi=180,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved table_ranking.png")


# ── TABLE 3: Motion Tuning — simplified (key benchmarks, key columns) ────────
fig3, ax3 = plt.subplots(figsize=(12, 3.05))
fig3.patch.set_facecolor(BG)

headers3 = ["Training Variant", "Type", "PCK@5%", "Δ PCK", "F(E→T) coverage ↑"]
rows3 = [
    ["__Eval = KITTI-2015"],
    ["SDF-Fractal3D",  "base",  "90.43", "+0.00", "0.197"],
    ["+ zoom ★",       "zoom",  "94.94", "+4.51", "0.653"],
    ["+ flip",         "flip",  "84.72", "−5.71", "0.132"],
    ["__Eval = TSS"],
    ["SDF-Fractal3D",  "base",  "46.58", "+0.00", "0.061"],
    ["+ zoom ★",       "zoom",  "53.42", "+6.84", "0.070"],
    ["+ flip",         "flip",  "39.89", "−6.69", "0.044"],
    ["__Eval = PF-PASCAL"],
    ["SDF-Fractal3D",  "base",  "30.72", "+0.00", "0.031"],
    ["+ zoom ★",       "zoom",  "34.57", "+3.85", "0.048"],
    ["+ flip",         "flip",  "22.27", "−8.44", "0.029"],
]
col_w3 = [2.8, 1.4, 1.1, 1.1, 2.0]
# Bold the best row within each benchmark block.
bold3 = [2, 6, 10]
star3 = compute_best_cells_by_section(rows3, metric_cols=[2, 3, 4])
make_table(ax3, headers3, rows3, col_w3,
           title="",
           bold_rows=bold3, star_cells=star3,
           col_aligns=["left","left","center","center","center"],
           fontsize=9.9, row_h=0.57, section_row_h=0.52, title_pad=2,
           focal_bold_cols=[0, 1],
           section_bg=HEADER, section_fg=TEAL, section_bold=True)
plt.tight_layout(pad=0.04)
plt.savefig(OUT_DIR / "table_motion_tuning.png", dpi=180,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved table_motion_tuning.png")
