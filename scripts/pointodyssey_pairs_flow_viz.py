#!/usr/bin/env python3
"""
Quick PointOdyssey pair flow visualization from manifest entries.

Creates a dense per-pixel flow image for a small random sample of pairs.
Flow channels are stored as:
  - Red  = dx
  - Green = dy
  - Blue  = 0
Raw float flow tensors are also written to disk for exact inspection.

Example:
  python scripts/pointodyssey_pairs_flow_viz.py \
    --manifest analysis/pointodyssey_pairs_smoke/manifest.jsonl \
    --num-samples 4 \
    --seed 42 \
    --out analysis/pointodyssey_flow_viz
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np


def find_frame_path(frame_dir: Path, frame_idx: int) -> Path:
    stem = f"{frame_idx:06d}"
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = frame_dir / f"{stem}{ext}"
        if p.exists():
            return p

    cands = sorted([p for p in frame_dir.iterdir() if p.is_file()], key=lambda x: x.name)
    for p in cands:
        nums = re.findall(r"\d+", p.stem)
        if nums and int(nums[-1]) == frame_idx:
            return p

    if 0 <= frame_idx < len(cands):
        return cands[frame_idx]

    raise FileNotFoundError(f"Could not resolve frame {frame_idx} in {frame_dir}")


def sample_manifest_records(manifest_path: Path, num_samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    with manifest_path.open("r") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if len(out) < num_samples:
                out.append(rec)
                continue

            j = rng.randrange(n)
            if j < num_samples:
                out[j] = rec

    if not out:
        raise RuntimeError(f"No valid lines found in manifest: {manifest_path}")
    return out


def load_annotation_cache(anno_cache: dict[Path, tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],], anno_path: Path):
    if anno_path in anno_cache:
        return anno_cache[anno_path]

    anno = np.load(anno_path, allow_pickle=True)
    trajs_2d = anno["trajs_2d"]
    valids = anno["valids"]
    visibs = anno["visibs"] if "visibs" in anno.files else None
    anno_cache[anno_path] = (trajs_2d, valids, visibs)
    return anno_cache[anno_path]


def build_pixel_flow(
    trajs_2d: np.ndarray,
    valids: np.ndarray,
    visibs: Optional[np.ndarray],
    frame_i: int,
    frame_j: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if frame_i >= trajs_2d.shape[0] or frame_j >= trajs_2d.shape[0]:
        raise ValueError(f"Frame index out of range for sequence: {frame_i}, {frame_j}")

    mask = valids[frame_i] & valids[frame_j]
    if visibs is not None:
        mask = mask & visibs[frame_i] & visibs[frame_j]

    p_i = trajs_2d[frame_i][mask]
    p_j = trajs_2d[frame_j][mask]
    if p_i.size == 0:
        return (
            np.zeros((height, width, 3), dtype=np.float32),
            np.zeros((height, width), dtype=np.int32),
            0,
        )

    finite = np.isfinite(p_i).all(axis=1) & np.isfinite(p_j).all(axis=1)
    p_i = p_i[finite]
    p_j = p_j[finite]
    if p_i.size == 0:
        return (
            np.zeros((height, width, 3), dtype=np.float32),
            np.zeros((height, width), dtype=np.int32),
            0,
        )

    xi = np.rint(p_i[:, 0]).astype(np.int32)
    yi = np.rint(p_i[:, 1]).astype(np.int32)
    in_bounds = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    xi = xi[in_bounds]
    yi = yi[in_bounds]
    p_i = p_i[in_bounds]
    p_j = p_j[in_bounds]
    if xi.size == 0:
        return (
            np.zeros((height, width, 3), dtype=np.float32),
            np.zeros((height, width), dtype=np.int32),
            0,
        )

    dx = p_j[:, 0] - p_i[:, 0]
    dy = p_j[:, 1] - p_i[:, 1]

    flow = np.zeros((height, width, 3), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.int32)
    np.add.at(flow[..., 0], (yi, xi), dx)
    np.add.at(flow[..., 1], (yi, xi), dy)
    np.add.at(count, (yi, xi), 1)

    valid = count > 0
    flow[valid, 0] = flow[valid, 0] / count[valid]
    flow[valid, 1] = flow[valid, 1] / count[valid]
    # channel 2 remains zero

    return flow, count, int(count.sum())


def flow_to_rgb(flow: np.ndarray, valid_pixels: np.ndarray) -> np.ndarray:
    rgb = np.zeros(flow.shape, dtype=np.uint8)
    h, w, _ = flow.shape

    for ch, dst in ((0, 0), (1, 1)):
        comp = flow[..., ch]
        if not np.any(valid_pixels):
            continue
        c = comp[valid_pixels]
        scale = float(np.percentile(np.abs(c), 99.0))
        if not np.isfinite(scale) or scale < 1e-6:
            continue
        v = np.clip((c / scale) * 127.0 + 127.0, 0.0, 255.0).astype(np.uint8)
        rgb[..., dst][valid_pixels] = v

    # Blue channel left at 0
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/pointodyssey_flow_viz"),
        help="Output folder",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = sample_manifest_records(manifest_path, args.num_samples, args.seed)
    anno_cache: dict[Path, tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}

    for sample_idx, rec in enumerate(records):
        seq_path = Path(rec["seq_path"]).expanduser().resolve()
        frame_dir = seq_path / "rgbs"
        frame_i = int(rec["frame_i"])
        frame_j = int(rec["frame_j"])
        anno_path = Path(rec["anno_path"]).expanduser().resolve()

        frame_i_path = find_frame_path(frame_dir, frame_i)
        frame_j_path = find_frame_path(frame_dir, frame_j)
        img_i = imageio.imread(frame_i_path)
        img_j = imageio.imread(frame_j_path)
        if img_i.shape[:2] != img_j.shape[:2]:
            raise ValueError(f"Frame shape mismatch for sample {sample_idx}: {img_i.shape} vs {img_j.shape}")
        h, w = img_i.shape[:2]

        trajs_2d, valids, visibs = load_annotation_cache(anno_cache, anno_path)
        flow, count, n_px = build_pixel_flow(trajs_2d, valids, visibs, frame_i, frame_j, h, w)
        valid_pixels = count > 0
        rgb = flow_to_rgb(flow, valid_pixels)

        base = (
            f"{seq_path.name}_"
            f"pair_{frame_i:06d}_{frame_j:06d}_"
            f"sample_{sample_idx:02d}"
        )
        flow_path = out_dir / f"{base}_flow.npy"
        vis_path = out_dir / f"{base}_flow_vis.png"

        np.save(flow_path, flow)
        imageio.imwrite(vis_path, rgb)

        meta = {
            "seq_path": str(seq_path),
            "frame_i": frame_i,
            "frame_j": frame_j,
            "pair_id": rec.get("pair_id", -1),
            "seed": args.seed,
            "sample_idx": sample_idx,
            "valid_pixel_count": int(n_px),
        }
        (out_dir / f"{base}_meta.json").write_text(json.dumps(meta, indent=2))
        print(
            f"[{sample_idx + 1}/{len(records)}] {seq_path.name}: {frame_i}->{frame_j} "
            f"pixels={n_px} saved={vis_path.name}"
        )


if __name__ == "__main__":
    main()
