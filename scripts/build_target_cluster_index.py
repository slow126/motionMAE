#!/usr/bin/env python3
"""Cluster target vectors and derive hard radii per centroid.

This script is the target-side setup for the cluster-coverage selector:
1. load target raw vectors (x, y, dx, dy)
2. optionally normalize into the same raw space used by the scorer
3. k-means cluster the target vectors
4. assign a hard radius to each centroid from the within-cluster distance quantile

The output is a compressed .npz file containing:
  - centroids: (K, D) float32
  - radii: (K,) float32
  - counts: (K,) int32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from score_source_samples_raw_nn import _prepare_raw_vectors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build target cluster index with hard radii")
    p.add_argument("--target-vectors", type=Path, required=True, help="Input target .npy with vec4 rows.")
    p.add_argument("--output", type=Path, required=True, help="Output .npz cluster index path.")
    p.add_argument("--num-centroids", type=int, required=True, help="Number of target centroids.")
    p.add_argument(
        "--radius-quantile",
        type=float,
        default=0.9,
        help="Quantile of within-cluster distances used as the hard radius (default: 0.9).",
    )
    p.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="Multiply each centroid radius by this factor after the quantile step (default: 1.0).",
    )
    p.add_argument(
        "--min-radius",
        type=float,
        default=0.0,
        help="Clamp each centroid radius to at least this value after scaling (default: 0.0).",
    )
    p.add_argument("--raw-space", type=str, choices=["flow", "joint"], default="joint")
    p.add_argument("--normalize-norm2x1", dest="normalize_norm2x1", action="store_true", default=True)
    p.add_argument("--no-normalize-norm2x1", dest="normalize_norm2x1", action="store_false")
    p.add_argument("--norm-width", type=float, default=512.0)
    p.add_argument("--norm-height", type=float, default=512.0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--max-iters", type=int, default=40)
    return p.parse_args()


def _run_kmeans_numpy(x: np.ndarray, k: int, seed: int, max_iters: int) -> Tuple[np.ndarray, np.ndarray]:
    n, d = x.shape
    if k <= 0 or k > n:
        raise ValueError(f"num_centroids must be in [1, {n}], got {k}")

    rng = np.random.default_rng(int(seed))
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = x[init_idx].astype(np.float32, copy=True)
    assignments = np.full(n, -1, dtype=np.int32)

    for _ in range(max(1, int(max_iters))):
        x2 = np.sum(x * x, axis=1, keepdims=True)
        c2 = np.sum(centroids * centroids, axis=1, keepdims=True).T
        d2 = np.maximum(x2 + c2 - 2.0 * (x @ centroids.T), 0.0)
        new_assignments = np.argmin(d2, axis=1).astype(np.int32)

        if np.array_equal(assignments, new_assignments):
            assignments = new_assignments
            break
        assignments = new_assignments

        for idx in range(k):
            members = x[assignments == idx]
            if members.shape[0] == 0:
                centroids[idx] = x[int(rng.integers(0, n))]
            else:
                centroids[idx] = members.mean(axis=0, dtype=np.float32)

    return centroids.astype(np.float32, copy=False), assignments


def main() -> None:
    args = parse_args()
    if not (0.0 < float(args.radius_quantile) <= 1.0):
        raise ValueError("--radius-quantile must be in (0, 1]")

    raw = np.load(args.target_vectors, mmap_mode="r")
    if raw.ndim != 2 or raw.shape[1] < 4:
        raise ValueError(f"Expected target vectors shape (N,4+), got {raw.shape}")
    x = _prepare_raw_vectors(
        raw[:, :4],
        raw_space=args.raw_space,
        normalize_norm2x1=args.normalize_norm2x1,
        norm_width=args.norm_width,
        norm_height=args.norm_height,
    ).astype(np.float32, copy=False)

    k = int(min(max(1, args.num_centroids), x.shape[0]))
    print(f"[cluster_index] clustering n={x.shape[0]} d={x.shape[1]} into k={k}")
    centroids, assignments = _run_kmeans_numpy(x, k=k, seed=int(args.seed), max_iters=int(args.max_iters))

    radii = np.zeros((k,), dtype=np.float32)
    counts = np.zeros((k,), dtype=np.int32)
    for idx in range(k):
        members = x[assignments == idx]
        counts[idx] = int(members.shape[0])
        if members.shape[0] == 0:
            radii[idx] = 0.0
            continue
        delta = members - centroids[idx : idx + 1]
        dist = np.sqrt(np.maximum(np.sum(delta * delta, axis=1), 0.0))
        radius = float(np.quantile(dist, float(args.radius_quantile)))
        radius *= float(args.radius_scale)
        radius = max(radius, float(args.min_radius))
        radii[idx] = radius

    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "target_vectors": str(args.target_vectors),
        "raw_space": args.raw_space,
        "normalize_norm2x1": bool(args.normalize_norm2x1),
        "norm_width": float(args.norm_width),
        "norm_height": float(args.norm_height),
        "num_centroids": int(k),
        "radius_quantile": float(args.radius_quantile),
        "radius_scale": float(args.radius_scale),
        "min_radius": float(args.min_radius),
        "seed": int(args.seed),
        "max_iters": int(args.max_iters),
    }
    np.savez_compressed(
        args.output,
        centroids=centroids,
        radii=radii,
        counts=counts,
        meta_json=np.asarray(json.dumps(meta)),
    )
    min_radius = float(np.min(radii)) if radii.size > 0 else 0.0
    median_radius = float(np.median(radii)) if radii.size > 0 else 0.0
    max_radius = float(np.max(radii)) if radii.size > 0 else 0.0
    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_centroids": int(k),
                "min_radius": min_radius,
                "median_radius": median_radius,
                "max_radius": max_radius,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
