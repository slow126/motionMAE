#!/usr/bin/env python3
"""Extract (x, y, dx, dy) flow vectors from a correspondence benchmark dataset.

Output is a (N, 4) float32 .npy file where each row is one keypoint:
    [x_src, y_src, dx, dy]  (dx = x_trg - x_src, dy = y_trg - y_src)

This file can be passed directly to score_source_samples_raw_nn.py
--target-vectors to score PointOdyssey pairs against a given benchmark.

Supported benchmarks:
  kitti2012  -- KittiSimpleDataset, val split
  kitti2015  -- KittiSimpleDataset, val split
  tss        -- TSSSimpleDataset
  pfpascal   -- CorrespondenceDataset (pfpascal, val split)
  pfwillow   -- CorrespondenceDataset (pfwillow, test split)

Usage:
  python scripts/extract_benchmark_vectors.py \\
      --benchmark kitti2015 \\
      --output    analysis/target_kitti2015_vectors.npy \\
      --kitti-root /home/spencer/Data/correspondence/kitti \\
      --tss-root   /home/spencer/Data/correspondence/TSS_CVPR2016 \\
      --cats-datapath ./models/Datasets_CATs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract benchmark flow vectors to .npy")
    p.add_argument(
        "--benchmark",
        required=True,
        choices=["kitti2012", "kitti2015", "tss", "pfpascal", "pfwillow"],
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--kitti-root",
        default="/home/spencer/Data/correspondence/kitti",
        help="Root containing kitti-2012/ and kitti-2015/ subdirs",
    )
    p.add_argument(
        "--tss-root",
        default="/home/spencer/Data/correspondence/TSS_CVPR2016",
    )
    p.add_argument(
        "--cats-datapath",
        default="./models/Datasets_CATs",
        help="datapath arg for CorrespondenceDataset (pfpascal / pfwillow)",
    )
    p.add_argument(
        "--size",
        type=int,
        default=512,
        help="Resize images to this square size before extracting kps (default: 512)",
    )
    p.add_argument(
        "--max-kps-per-pair",
        type=int,
        default=0,
        help="Cap keypoints per pair (0 = unlimited)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-benchmark loaders
# ---------------------------------------------------------------------------

def _extract_kitti(root: Path, split: str, max_kps: int) -> np.ndarray:
    from src.data.synth.datasets.KittiDataset import KittiSimpleDataset

    dataset = KittiSimpleDataset(root=str(root), split=split)
    return _collect_from_flow_dataset(dataset, max_kps, name=f"kitti/{split}")


def _extract_tss(root: Path, max_kps: int) -> np.ndarray:
    from src.data.synth.datasets.TSSDataset import TSSSimpleDataset

    dataset = TSSSimpleDataset(root=root)
    return _collect_from_flow_dataset(dataset, max_kps, name="tss")


def _extract_corr(benchmark: str, datapath: str, split: str, size: int, max_kps: int) -> np.ndarray:
    from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset

    dataset = CorrespondenceDataset(
        dataset_name=benchmark,
        datapath=datapath,
        split=split,
        size=(size, size),
    )
    return _collect_from_kps_dataset(dataset, max_kps, name=benchmark)


def _collect_from_flow_dataset(dataset, max_kps: int, name: str) -> np.ndarray:
    """For datasets that return dense flow (2, H, W): sample valid pixels → (N, 4)."""
    all_vecs: List[np.ndarray] = []
    n = len(dataset)
    print(f"  {name}: {n} pairs (dense flow)", flush=True)

    for i in range(n):
        sample = dataset[i]
        flow = sample.get("flow")  # (2, H, W) tensor, pixel-space dx/dy
        if flow is None:
            continue

        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()

        # flow: (2, H, W)  where [0]=dx, [1]=dy
        dx = flow[0].astype(np.float32)   # (H, W)
        dy = flow[1].astype(np.float32)

        H, W = dx.shape
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        xs = xs.astype(np.float32).ravel()
        ys = ys.astype(np.float32).ravel()
        dxf = dx.ravel()
        dyf = dy.ravel()

        valid = np.isfinite(dxf) & np.isfinite(dyf) & ((dxf != 0.0) | (dyf != 0.0))
        xs, ys, dxf, dyf = xs[valid], ys[valid], dxf[valid], dyf[valid]

        if len(xs) == 0:
            continue

        if max_kps > 0 and len(xs) > max_kps:
            idx = np.random.choice(len(xs), max_kps, replace=False)
            xs, ys, dxf, dyf = xs[idx], ys[idx], dxf[idx], dyf[idx]

        all_vecs.append(np.stack([xs, ys, dxf, dyf], axis=1))

        if (i + 1) % 20 == 0 or (i + 1) == n:
            total = sum(v.shape[0] for v in all_vecs)
            print(f"  [{i+1}/{n}] {total} vectors collected", flush=True)

    if not all_vecs:
        raise RuntimeError(f"No valid flow pixels extracted from {name}")

    return np.concatenate(all_vecs, axis=0).astype(np.float32)


def _collect_from_kps_dataset(dataset, max_kps: int, name: str) -> np.ndarray:
    """For datasets that return src_kps / trg_kps: derive dx/dy → (N, 4)."""
    all_vecs: List[np.ndarray] = []
    n = len(dataset)
    print(f"  {name}: {n} pairs (keypoints)", flush=True)

    for i in range(n):
        sample = dataset[i]
        src_kps = sample.src_kps if hasattr(sample, "src_kps") else sample.get("src_kps")
        trg_kps = sample.trg_kps if hasattr(sample, "trg_kps") else sample.get("trg_kps")
        n_pts   = sample.n_pts   if hasattr(sample, "n_pts")   else sample.get("n_pts")

        if src_kps is None or trg_kps is None:
            continue

        if isinstance(src_kps, torch.Tensor):
            src_kps = src_kps.cpu().numpy()
        if isinstance(trg_kps, torch.Tensor):
            trg_kps = trg_kps.cpu().numpy()
        if isinstance(n_pts, torch.Tensor):
            n_pts = int(n_pts.item())
        elif n_pts is not None:
            n_pts = int(n_pts)

        if src_kps.ndim == 2 and src_kps.shape[0] == 2:
            k = n_pts if n_pts is not None else src_kps.shape[1]
            src_x = src_kps[0, :k].astype(np.float32)
            src_y = src_kps[1, :k].astype(np.float32)
            trg_x = trg_kps[0, :k].astype(np.float32)
            trg_y = trg_kps[1, :k].astype(np.float32)
        else:
            continue

        valid = (
            np.isfinite(src_x) & np.isfinite(src_y) &
            np.isfinite(trg_x) & np.isfinite(trg_y)
        )
        src_x, src_y = src_x[valid], src_y[valid]
        trg_x, trg_y = trg_x[valid], trg_y[valid]

        if len(src_x) == 0:
            continue

        if max_kps > 0 and len(src_x) > max_kps:
            idx = np.random.choice(len(src_x), max_kps, replace=False)
            src_x, src_y = src_x[idx], src_y[idx]
            trg_x, trg_y = trg_x[idx], trg_y[idx]

        dx = trg_x - src_x
        dy = trg_y - src_y
        all_vecs.append(np.stack([src_x, src_y, dx, dy], axis=1))

        if (i + 1) % 50 == 0 or (i + 1) == n:
            total = sum(v.shape[0] for v in all_vecs)
            print(f"  [{i+1}/{n}] {total} vectors collected", flush=True)

    if not all_vecs:
        raise RuntimeError(f"No valid keypoints extracted from {name}")

    return np.concatenate(all_vecs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting vectors for benchmark: {args.benchmark}")

    if args.benchmark == "kitti2012":
        root = Path(args.kitti_root) / "kitti-2012"
        vectors = _extract_kitti(root, split="val", max_kps=args.max_kps_per_pair)
    elif args.benchmark == "kitti2015":
        root = Path(args.kitti_root) / "kitti-2015"
        vectors = _extract_kitti(root, split="val", max_kps=args.max_kps_per_pair)
    elif args.benchmark == "tss":
        vectors = _extract_tss(Path(args.tss_root), max_kps=args.max_kps_per_pair)
    elif args.benchmark == "pfpascal":
        vectors = _extract_corr("pfpascal", args.cats_datapath, split="val",
                                size=args.size, max_kps=args.max_kps_per_pair)
    elif args.benchmark == "pfwillow":
        vectors = _extract_corr("pfwillow", args.cats_datapath, split="test",
                                size=args.size, max_kps=args.max_kps_per_pair)
    else:
        raise ValueError(args.benchmark)

    np.save(args.output, vectors)
    print(f"Saved {vectors.shape[0]} vectors ({vectors.shape}) → {args.output}")


if __name__ == "__main__":
    main()
