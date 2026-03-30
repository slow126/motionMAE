#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a side-by-side comparison image from consistency UMAP ft3d_sweep grids."
    )
    parser.add_argument(
        "sweep_dir",
        type=str,
        help="Path to scripts/consistency_umap_out_<snapshot>_ft3d_sweep",
    )
    parser.add_argument(
        "--panel",
        choices=["src", "tgt", "gt", "pred", "mask"],
        default="pred",
        help="Which panel column to extract from each saved grid.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=16,
        help="Maximum number of sample rows to include from each grid.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.6,
        help="Resize factor applied to each extracted cell.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path. Defaults inside the sweep dir.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["auto", "iid", "train_like"],
        default="auto",
        help="Which saved sweep grids to use. "
             "auto prefers train_like grids when present, otherwise iid.",
    )
    return parser.parse_args()


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, mask.shape[0]))
    return runs


def _detect_panel_boxes(image: Image.Image) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    arr = np.asarray(image.convert("RGB"))
    nonwhite = np.any(arr < 245, axis=2)

    col_mask = nonwhite.mean(axis=0) > 0.08
    col_runs = _contiguous_runs(col_mask)
    col_runs = [run for run in col_runs if (run[1] - run[0]) > arr.shape[1] * 0.08]
    if len(col_runs) == 4:
        widths = [run[1] - run[0] for run in col_runs]
        gaps = [col_runs[i + 1][0] - col_runs[i][1] for i in range(len(col_runs) - 1)]
        width = int(round(float(np.median(widths))))
        gap = int(round(float(np.median(gaps)))) if gaps else 0
        start = col_runs[-1][1] + gap
        end = min(arr.shape[1], start + width)
        col_runs.append((start, end))
    if len(col_runs) < 5:
        raise RuntimeError(f"Could not detect 5 panel columns; found {len(col_runs)}")
    col_runs = sorted(col_runs, key=lambda run: run[0])[:5]

    row_mask = nonwhite.mean(axis=1) > 0.10
    row_runs = _contiguous_runs(row_mask)
    row_runs = [run for run in row_runs if (run[1] - run[0]) > arr.shape[0] * 0.03]
    if not row_runs:
        raise RuntimeError("Could not detect any sample rows.")
    return col_runs, row_runs


def _panel_index(name: str) -> int:
    return {
        "src": 0,
        "tgt": 1,
        "gt": 2,
        "pred": 3,
        "mask": 4,
    }[name]


def _load_sweep_grids(grids_dir: Path, mask_mode: str) -> tuple[list[tuple[int, Path]], str]:
    iid_entries: list[tuple[int, Path]] = []
    train_like_entries: list[tuple[int, Path]] = []
    pattern = re.compile(r"_(\d+)pct_masked(?:_\(train-like\))?_grid\.png$")
    for path in sorted(grids_dir.glob("*pct_masked*_grid.png")):
        match = pattern.search(path.name)
        if match is None:
            continue
        pct = int(match.group(1))
        if "(train-like)" in path.name:
            train_like_entries.append((pct, path))
        else:
            iid_entries.append((pct, path))

    if mask_mode == "train_like":
        entries = train_like_entries
        chosen = "train_like"
    elif mask_mode == "iid":
        entries = iid_entries
        chosen = "iid"
    else:
        if train_like_entries:
            entries = train_like_entries
            chosen = "train_like"
        else:
            entries = iid_entries
            chosen = "iid"

    if not entries:
        raise RuntimeError(
            f"No sweep grids found under {grids_dir} for mask_mode={mask_mode!r}"
        )
    entries.sort(key=lambda item: item[0])
    return entries, chosen


def _resize(image: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0:
        return image
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    return image.resize((width, height), Image.Resampling.BICUBIC)


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    grids_dir = sweep_dir / "grids"
    entries, chosen_mask_mode = _load_sweep_grids(grids_dir, args.mask_mode)

    first_image = Image.open(entries[0][1]).convert("RGB")
    col_runs, row_runs = _detect_panel_boxes(first_image)
    col_idx = _panel_index(args.panel)
    panel_x0, panel_x1 = col_runs[col_idx]
    row_runs = row_runs[: args.rows]

    font = ImageFont.load_default()
    header_h = 28
    label_w = 54
    pad = 8

    sample_cell = _resize(first_image.crop((panel_x0, row_runs[0][0], panel_x1, row_runs[0][1])), args.scale)
    cell_w, cell_h = sample_cell.size

    canvas_w = label_w + pad + len(entries) * cell_w + max(0, len(entries) - 1) * pad
    canvas_h = header_h + pad + len(row_runs) * cell_h + max(0, len(row_runs) - 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for col, (pct, path) in enumerate(entries):
        image = Image.open(path).convert("RGB")
        x = label_w + pad + col * (cell_w + pad)
        label = f"{pct}%"
        text_w = draw.textbbox((0, 0), label, font=font)[2]
        draw.text((x + (cell_w - text_w) // 2, 6), label, fill=(0, 0, 0), font=font)

        for row, (y0, y1) in enumerate(row_runs):
            crop = image.crop((panel_x0, y0, panel_x1, y1))
            crop = _resize(crop, args.scale)
            y = header_h + pad + row * (cell_h + pad)
            canvas.paste(crop, (x, y))

    for row in range(len(row_runs)):
        y = header_h + pad + row * (cell_h + pad) + cell_h // 2 - 4
        label = f"{row:02d}"
        draw.text((10, y), label, fill=(0, 0, 0), font=font)

    title = f"{sweep_dir.name} | {args.panel} panels | {chosen_mask_mode}"
    draw.text((10, 6), title, fill=(0, 0, 0), font=font)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_path = sweep_dir / f"sweep_{args.panel}_side_by_side_{chosen_mask_mode}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
