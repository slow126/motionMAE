#!/usr/bin/env python3
"""
PointOdyssey mining visualizer (single-file).

Examples:
  python scripts/pointodyssey_mine_viz.py \
    --seq /home/spencer/Data/PointOdyssey/seq_train_0001 \
    --pair 120 130 --out /tmp/po_mining_viz

  python scripts/pointodyssey_mine_viz.py \
    --seq /home/spencer/Data/PointOdyssey/seq_train_0001 \
    --manifest analysis/pointodyssey_pairs_smoke/manifest.jsonl \
    --manifest-index 0 --out /tmp/po_mining_viz
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def read_image_rgb(path: Path) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[-1] > 3:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def find_frame_path(frame_dir: Path, frame_idx: int) -> Path:
    stem = f"{frame_idx:06d}"
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = frame_dir / f"{stem}{ext}"
        if p.exists():
            return p

    cands = sorted(
        [p for p in frame_dir.iterdir() if p.is_file()],
        key=lambda x: (x.stem)
    )
    # fallback: try any filename containing the index token
    for p in cands:
        nums = re.findall(r"\d+", p.stem)
        if nums and int(nums[-1]) == frame_idx:
            return p
    if 0 <= frame_idx < len(cands):
        return cands[frame_idx]
    raise FileNotFoundError(f"Could not find frame {frame_idx} in {frame_dir}")


def load_manifest_pair(manifest_path: Path, seq_path: Path, idx: int | None = None, random_pick: bool = False):
    seq_name = seq_path.name
    seq_norm = str(seq_path)
    matches = []
    with manifest_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec_seq = str(rec.get("seq_path", ""))
            if (rec_seq == str(seq_path)) or (rec_seq == seq_norm) or (Path(rec_seq).name == seq_name) or (seq_name in rec_seq):
                matches.append(rec)

    if not matches:
        raise RuntimeError(f"No manifest rows found for sequence {seq_path}")

    if random_pick:
        rec = random.choice(matches)
    else:
        if idx is None:
            idx = 0
        rec = matches[idx]

    return rec, len(matches)


def mining_mask(
    pts_i: np.ndarray,
    pts_j: np.ndarray,
    valids_i: np.ndarray,
    valids_j: np.ndarray,
    h: int,
    w: int,
    visibs_i: np.ndarray | None = None,
    visibs_j: np.ndarray | None = None,
    max_displacement: float | None = None,
    min_valid_points: int = 8,
):
    pts_i = np.asarray(pts_i, dtype=np.float32)
    pts_j = np.asarray(pts_j, dtype=np.float32)
    valids_i = valids_i.astype(bool)
    valids_j = valids_j.astype(bool)

    stages = {}
    mask = valids_i & valids_j
    stages["1) valids"] = mask.copy()

    finite = np.isfinite(pts_i).all(axis=1) & np.isfinite(pts_j).all(axis=1)
    mask = mask & finite
    stages["2) finite"] = mask.copy()

    in_bounds = (
        (pts_i[:, 0] >= 0) & (pts_i[:, 0] < w) &
        (pts_i[:, 1] >= 0) & (pts_i[:, 1] < h) &
        (pts_j[:, 0] >= 0) & (pts_j[:, 0] < w) &
        (pts_j[:, 1] >= 0) & (pts_j[:, 1] < h)
    )
    mask = mask & in_bounds
    stages["3) in bounds"] = mask.copy()

    if visibs_i is not None and visibs_j is not None:
        mask = mask & visibs_i.astype(bool) & visibs_j.astype(bool)
        stages["4) visibs"] = mask.copy()

    not_zero = ~(np.all(pts_i == 0, axis=1) | np.all(pts_j == 0, axis=1))
    mask = mask & not_zero
    stages["5) non-zero"] = mask.copy()

    if max_displacement is not None and np.isfinite(max_displacement):
        disp = np.linalg.norm(pts_j - pts_i, axis=1)
        mask = mask & (disp <= float(max_displacement))
        stages[f"6) disp <= {max_displacement}"] = mask.copy()

    disp = np.linalg.norm(pts_j - pts_i, axis=1)
    valid_disp = disp[mask]
    valid_disp = valid_disp[np.isfinite(valid_disp)]
    accepted = int(mask.sum()) >= min_valid_points

    stats = {
        "valid_points": int(mask.sum()),
        "min_valid_points": int(min_valid_points),
        "accepted": bool(accepted),
        "motion_mean": float(valid_disp.mean()) if valid_disp.size else 0.0,
        "motion_median": float(np.median(valid_disp)) if valid_disp.size else 0.0,
        "stage_counts": {k: int(v.sum()) for k, v in stages.items()},
    }
    return stages, mask, stats


def save_stage_plot(img_i, img_j, pts_i, pts_j, mask, title, out_png):
    idx = np.where(mask)[0]
    # subsample for speed/readability
    if len(idx) > 2000:
        idx = np.random.choice(idx, size=2000, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    p_i = pts_i[idx]
    p_j = pts_j[idx]

    h, w = img_i.shape[:2]
    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

    axes[0].imshow(img_i)
    axes[0].set_title("frame_i")
    axes[1].imshow(img_j)
    axes[1].set_title("frame_j")

    axes[0].scatter(p_i[:, 0], p_i[:, 1], s=4, c="lime", alpha=0.45)
    axes[1].scatter(p_j[:, 0], p_j[:, 1], s=4, c="cyan", alpha=0.45)

    if len(idx):
        segs = np.stack([p_i, p_j], axis=1)
        lc = LineCollection(segs, colors="yellow", linewidths=0.35, alpha=0.35)
        axes[1].add_collection(lc)

    fig.suptitle(f"{title} (kept={len(idx)})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=Path, required=True)
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--anno", type=Path, default=None)
    parser.add_argument("--pair", type=int, nargs=2, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifest-index", type=int, default=0)
    parser.add_argument("--manifest-random", action="store_true")
    parser.add_argument("--min-valid-points", type=int, default=8)
    parser.add_argument("--max-displacement", type=float, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/pointodyssey_mine_viz"),
        help="Output directory (default: analysis/pointodyssey_mine_viz under project root)",
    )
    args = parser.parse_args()

    seq = args.seq.expanduser().resolve()
    frame_dir = (args.frame_dir or (seq / "rgbs")).resolve()
    anno_path = (args.anno or (seq / "anno.npz")).resolve()
    out = args.out.expanduser().resolve()

    anno = np.load(anno_path)
    trajs_2d = anno["trajs_2d"]
    valids = anno["valids"]
    visibs = anno["visibs"] if "visibs" in anno.files else None

    if args.manifest:
        if args.pair is not None:
            raise ValueError("Use either --pair or --manifest (not both).")
        rec, n = load_manifest_pair(
            args.manifest.expanduser().resolve(),
            seq,
            idx=args.manifest_index,
            random_pick=args.manifest_random,
        )
        i = int(rec["frame_i"])
        j = int(rec["frame_j"])
        print(f"Loaded pair from manifest: frame_i={i}, frame_j={j}, matches_for_seq={n}")
    else:
        if not args.pair:
            raise ValueError("Need either --pair or --manifest")
        i, j = args.pair

    if i < 0 or j <= i:
        raise ValueError("Require frame_j > frame_i >= 0")
    if j >= trajs_2d.shape[0] or i >= trajs_2d.shape[0]:
        raise ValueError(f"frames out of range; total frames={trajs_2d.shape[0]}")

    img_i = read_image_rgb(find_frame_path(frame_dir, i))
    img_j = read_image_rgb(find_frame_path(frame_dir, j))
    h, w = img_i.shape[:2]
    if img_j.shape[:2] != (h, w):
        raise ValueError("Source/target image sizes differ")

    pts_i = trajs_2d[i]
    pts_j = trajs_2d[j]

    stages, final_mask, stats = mining_mask(
        pts_i=pts_i,
        pts_j=pts_j,
        valids_i=valids[i],
        valids_j=valids[j],
        h=h,
        w=w,
        visibs_i=None if visibs is None else visibs[i],
        visibs_j=None if visibs is None else visibs[j],
        max_displacement=args.max_displacement,
        min_valid_points=args.min_valid_points,
    )

    out.mkdir(parents=True, exist_ok=True)
    for k, (name, m) in enumerate(stages.items(), start=1):
        save_stage_plot(
            img_i, img_j, pts_i, pts_j, m,
            f"{k}) {name}",
            out / f"{seq.name}_pair_{i:06d}_{j:06d}_stage_{k:02d}.png",
        )

    meta = {
        "seq_path": str(seq),
        "pair": [i, j],
        "min_valid_points": args.min_valid_points,
        "max_displacement": args.max_displacement,
        **stats,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    print("Saved:", out)
    print(f"accepted: {stats['accepted']}, final valid points: {stats['valid_points']}")
    print(f"motion mean/median: {stats['motion_mean']:.4f}, {stats['motion_median']:.4f}")


if __name__ == "__main__":
    main()
