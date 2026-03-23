#!/usr/bin/env python3
"""Build a Farthest Point Sampling (FPS) pair subset from a PointOdyssey manifest.

Reads manifest.jsonl, computes a 17-D BFV summary descriptor per pair from the
raw annotation trajectories, standardizes descriptors with StandardScaler, then
runs greedy FPS in descriptor space to select a maximally diverse budget subset.

Descriptor computation is parallelized across sequences via --num_workers.
Descriptors are cached to disk so FPS can be rerun cheaply (different budget/seed)
without reloading any annotations.

The output is a JSON list of pair indices (into manifest.jsonl) in the same format
as subset_random_*.json and subset_heuristic_balanced_*.json.

Usage (full dataset, ~4M pairs, 5% budget, 8 workers):
    python scripts/build_pointodyssey_fps_subset.py \\
        --manifest_path analysis/pointodyssey_pairs_smoke/manifest.jsonl \\
        --output_dir   analysis/pointodyssey_pairs_smoke_5pct \\
        --fraction     0.05 \\
        --seed         42 \\
        --num_workers  8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FPS pair subset from manifest")
    p.add_argument("--manifest_path", required=True, help="Path to manifest.jsonl")
    p.add_argument("--output_dir", required=True, help="Directory for output JSON")
    p.add_argument(
        "--fraction",
        type=float,
        default=0.05,
        help="Fraction of pairs to select (default: 0.05)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for candidate pre-sampling and FPS (default: 42)",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for descriptor computation (default: 1)",
    )
    p.add_argument(
        "--pool_multiplier",
        type=float,
        default=None,
        help=(
            "If set, randomly pre-sample (pool_multiplier × budget) candidates from "
            "the manifest before computing descriptors or running FPS. "
            "Recommended: 5–10 for large manifests. Default: None (use all pairs)."
        ),
    )
    p.add_argument(
        "--descriptor_type",
        choices=["bfv", "mean_mag", "median_mag", "p90_mag"],
        default="bfv",
        help=(
            "Which descriptor to use for FPS. "
            "'bfv' (default): 17-D bag-of-flow-vectors summary (requires anno loading). "
            "'mean_mag'/'median_mag': 1-D scalar read directly from manifest (fast, no anno loading). "
            "'p90_mag': 1-D 90th-percentile magnitude (requires anno loading)."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# BFV summary descriptor
# ---------------------------------------------------------------------------


def _image_size(rgb_path: Path) -> Tuple[int, int]:
    """Return (W, H) for the given image."""
    with Image.open(rgb_path) as img:
        return img.size  # PIL: (width, height)


def _summarize_pair(
    trajs_2d: np.ndarray,
    valids: np.ndarray,
    visibs: Optional[np.ndarray],
    frame_i: int,
    frame_j: int,
    W: int,
    H: int,
) -> Optional[np.ndarray]:
    """Compute a 17-D BFV summary descriptor for a single frame pair.

    The 4-D BFV vector per valid point is:
        f = [2*x/W - 1,  2*y/H - 1,  2*dx/W,  2*dy/H]
    where (x, y) is the source position and (dx, dy) = trg - src.

    Summary = [mean(4), std(4), median(4), mag_stats(5)] → 17-D.
    Returns None if there are no valid points.
    """
    src = np.asarray(trajs_2d[frame_i], dtype=np.float32)  # (N, 2): x, y pixels
    trg = np.asarray(trajs_2d[frame_j], dtype=np.float32)

    src_valid = np.asarray(valids[frame_i], dtype=np.float32) > 0
    trg_valid = np.asarray(valids[frame_j], dtype=np.float32) > 0
    mask = src_valid & trg_valid
    mask &= np.isfinite(src).all(axis=1)
    mask &= np.isfinite(trg).all(axis=1)
    mask &= ~(np.logical_and(src[:, 0] == 0.0, src[:, 1] == 0.0))
    mask &= ~(np.logical_and(trg[:, 0] == 0.0, trg[:, 1] == 0.0))

    if visibs is not None:
        mask &= np.asarray(visibs[frame_i], dtype=np.float32) > 0
        mask &= np.asarray(visibs[frame_j], dtype=np.float32) > 0

    if not mask.any():
        return None

    sx, sy = src[mask, 0], src[mask, 1]
    dx = trg[mask, 0] - sx
    dy = trg[mask, 1] - sy

    x_hat = 2.0 * sx / W - 1.0
    y_hat = 2.0 * sy / H - 1.0
    dx_hat = 2.0 * dx / W
    dy_hat = 2.0 * dy / H

    bfv = np.stack([x_hat, y_hat, dx_hat, dy_hat], axis=1)  # (N, 4)

    mean = bfv.mean(axis=0)           # (4,)
    std = bfv.std(axis=0)             # (4,)
    median = np.median(bfv, axis=0)   # (4,)

    mag = np.sqrt(dx_hat**2 + dy_hat**2)
    mag_stats = np.array([
        mag.mean(),
        mag.std(),
        np.percentile(mag, 10),
        np.percentile(mag, 50),
        np.percentile(mag, 90),
    ])  # (5,)

    return np.concatenate([mean, std, median, mag_stats])  # (17,)


def _summarize_pair_p90(
    trajs_2d: np.ndarray,
    valids: np.ndarray,
    visibs: Optional[np.ndarray],
    frame_i: int,
    frame_j: int,
    W: int,
    H: int,
) -> Optional[np.ndarray]:
    """1-D descriptor: 90th-percentile of normalised flow magnitude."""
    src = np.asarray(trajs_2d[frame_i], dtype=np.float32)
    trg = np.asarray(trajs_2d[frame_j], dtype=np.float32)

    src_valid = np.asarray(valids[frame_i], dtype=np.float32) > 0
    trg_valid = np.asarray(valids[frame_j], dtype=np.float32) > 0
    mask = src_valid & trg_valid
    mask &= np.isfinite(src).all(axis=1)
    mask &= np.isfinite(trg).all(axis=1)
    mask &= ~(np.logical_and(src[:, 0] == 0.0, src[:, 1] == 0.0))
    mask &= ~(np.logical_and(trg[:, 0] == 0.0, trg[:, 1] == 0.0))

    if visibs is not None:
        mask &= np.asarray(visibs[frame_i], dtype=np.float32) > 0
        mask &= np.asarray(visibs[frame_j], dtype=np.float32) > 0

    if not mask.any():
        return None

    dx_hat = 2.0 * (trg[mask, 0] - src[mask, 0]) / W
    dy_hat = 2.0 * (trg[mask, 1] - src[mask, 1]) / H
    mag = np.sqrt(dx_hat**2 + dy_hat**2)
    return np.array([np.percentile(mag, 90)], dtype=np.float32)


def compute_descriptors_from_manifest(
    records: List[Dict],
    candidate_indices: List[int],
    field: str,
) -> Tuple[np.ndarray, List[int]]:
    """Extract a 1-D scalar descriptor directly from a manifest field (no anno loading).

    Used for 'mean_mag' and 'median_mag' which are already pre-computed in the manifest.
    """
    valid_indices = []
    values = []
    for idx in candidate_indices:
        v = records[idx].get(field)
        if v is not None and np.isfinite(float(v)):
            valid_indices.append(idx)
            values.append(float(v))
    descriptors = np.array(values, dtype=np.float32).reshape(-1, 1)
    return descriptors, valid_indices


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Per-sequence worker (runs in a subprocess when num_workers > 1)
# ---------------------------------------------------------------------------


def _process_sequence(
    seq_job: Tuple[str, List[Tuple[int, int, int, str]], str],
) -> Tuple[str, List[Tuple[int, np.ndarray]]]:
    """Load one anno.npz and compute descriptors for all its candidate pairs.

    Args:
        seq_job: (anno_path_str, [(manifest_idx, frame_i, frame_j, seq_path), ...], descriptor_type)

    Returns:
        (anno_path_str, [(manifest_idx, descriptor), ...])
    """
    anno_path_str, pair_specs, descriptor_type = seq_job
    anno_path = Path(anno_path_str)

    if not anno_path.exists():
        return anno_path_str, []

    first_frame_i, first_seq_path = pair_specs[0][1], pair_specs[0][3]
    sample_rgb = Path(first_seq_path) / "rgbs" / f"rgb_{first_frame_i:05d}.jpg"
    if not sample_rgb.exists():
        return anno_path_str, []

    W, H = _image_size(sample_rgb)
    results: List[Tuple[int, np.ndarray]] = []

    summarize_fn = _summarize_pair if descriptor_type == "bfv" else _summarize_pair_p90

    with np.load(anno_path, allow_pickle=True, mmap_mode="r") as ann:
        trajs_2d = ann["trajs_2d"]
        valids = ann["valids"]
        visibs = ann["visibs"] if "visibs" in ann.files else None

        for manifest_idx, frame_i, frame_j, _ in pair_specs:
            desc = summarize_fn(
                trajs_2d=trajs_2d,
                valids=valids,
                visibs=visibs,
                frame_i=frame_i,
                frame_j=frame_j,
                W=W,
                H=H,
            )
            if desc is not None:
                results.append((manifest_idx, desc))

    return anno_path_str, results


# ---------------------------------------------------------------------------
# Descriptor computation (parallel or sequential)
# ---------------------------------------------------------------------------


def compute_descriptors(
    records: List[Dict],
    candidate_indices: List[int],
    num_workers: int = 1,
    descriptor_type: str = "bfv",
) -> Tuple[np.ndarray, List[int]]:
    """Compute one descriptor per candidate pair by loading anno.npz files.

    Used for descriptor_type='bfv' and 'p90_mag'. Groups by anno_path so each
    file is loaded once, and optionally parallelizes across worker processes.

    Returns:
        descriptors   – (M, D) float32 array for valid candidates
        valid_indices – length-M list of manifest indices for each descriptor row
    """
    # Build per-sequence job list: anno_path → [(manifest_idx, frame_i, frame_j, seq_path)]
    by_anno: Dict[str, List[Tuple[int, int, int, str]]] = {}
    for idx in candidate_indices:
        r = records[idx]
        by_anno.setdefault(r["anno_path"], []).append(
            (idx, r["frame_i"], r["frame_j"], r["seq_path"])
        )

    seq_jobs = sorted(by_anno.items())  # deterministic order
    n_seqs = len(seq_jobs)
    num_workers = max(1, min(num_workers, n_seqs))

    desc_map: Dict[int, np.ndarray] = {}
    overall_t0 = time.time()
    seq_times: List[float] = []
    seqs_done = 0

    def _handle_result(anno_path_str: str, pairs_result: List[Tuple[int, np.ndarray]], seq_elapsed: float) -> None:
        nonlocal seqs_done
        for manifest_idx, desc in pairs_result:
            desc_map[manifest_idx] = desc
        seq_times.append(seq_elapsed)
        seqs_done += 1
        seqs_left = n_seqs - seqs_done
        avg_s = sum(seq_times) / len(seq_times)
        eta_s = avg_s * seqs_left
        wall_s = time.time() - overall_t0
        seq_name = Path(anno_path_str).parent.name
        n_ok = len(pairs_result)
        n_pairs = len(by_anno[anno_path_str])
        print(
            f"  [{seqs_done}/{n_seqs}] {seq_name}:"
            f" {n_ok}/{n_pairs} valid, {seq_elapsed:.1f}s/seq"
            f" | wall {_fmt_duration(wall_s)}"
            f" | ETA {_fmt_duration(eta_s)}",
            flush=True,
        )

    if num_workers <= 1:
        for anno_path_str, pair_specs in seq_jobs:
            t0 = time.time()
            _, pairs_result = _process_sequence((anno_path_str, pair_specs, descriptor_type))
            _handle_result(anno_path_str, pairs_result, time.time() - t0)
    else:
        print(f"  Using {num_workers} parallel workers", flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
            future_to_info = {}
            for anno_path_str, pair_specs in seq_jobs:
                t0 = time.time()
                fut = ex.submit(_process_sequence, (anno_path_str, pair_specs, descriptor_type))
                future_to_info[fut] = (anno_path_str, t0)

            for fut in concurrent.futures.as_completed(future_to_info):
                anno_path_str, t0 = future_to_info[fut]
                _, pairs_result = fut.result()
                _handle_result(anno_path_str, pairs_result, time.time() - t0)

    valid_indices = sorted(desc_map)
    descriptors = np.stack([desc_map[i] for i in valid_indices], axis=0).astype(np.float32)
    return descriptors, valid_indices


# ---------------------------------------------------------------------------
# Farthest Point Sampling
# ---------------------------------------------------------------------------


def farthest_point_sampling(
    descriptors: np.ndarray,
    budget: int,
    seed: int = 42,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 5_000,
    checkpoint_suffix: str = "",
) -> List[int]:
    """Greedy FPS in standardized descriptor space.

    Uses squared L2 via BLAS GEMV:
        ||a - b||² = ||a||² - 2(a·b) + ||b||²
    Precomputing ||a||² once reduces each iteration to a single matrix-vector
    multiply, which is 3-5× faster than elementwise subtract-square-sum.

    Args:
        descriptors:      (N, D) float32 array, already standardized.
        budget:           Number of points to select.
        seed:             Random seed for initial point selection.
        checkpoint_path:  Directory to save/load FPS state. Enables resume on crash.
        checkpoint_every: Save a checkpoint every this many iterations.

    Returns:
        List of selected row indices (length == min(budget, N)).
    """
    N = descriptors.shape[0]
    budget = min(budget, N)

    # Precompute squared norms once — reused every iteration
    sq_norms = (descriptors * descriptors).sum(axis=1)  # (N,) float32

    selected: List[int] = []
    min_dist_sq = np.full(N, np.inf, dtype=np.float32)
    start_k = 0

    # Resume from checkpoint if available
    if checkpoint_path is not None:
        ckpt_sel = checkpoint_path / f"fps_selected{checkpoint_suffix}.npy"
        ckpt_dist = checkpoint_path / f"fps_min_dist_sq{checkpoint_suffix}.npy"
        if ckpt_sel.exists() and ckpt_dist.exists():
            selected = np.load(ckpt_sel).tolist()
            min_dist_sq = np.load(ckpt_dist)
            start_k = len(selected)
            print(f"  Resumed FPS from checkpoint at iteration {start_k}", flush=True)

    if start_k == 0:
        rng = np.random.RandomState(seed)
        first_idx = int(rng.randint(N))
        selected = [first_idx]
        _update_min_dist_sq(descriptors, sq_norms, first_idx, min_dist_sq)
        start_k = 1

    t_last = time.time()
    for k in range(start_k, budget):
        next_idx = int(np.argmax(min_dist_sq))
        selected.append(next_idx)
        _update_min_dist_sq(descriptors, sq_norms, next_idx, min_dist_sq)
        min_dist_sq[next_idx] = -np.inf  # prevent re-selection

        if k % 1000 == 0:
            elapsed = time.time() - t_last
            rate = 1000 / elapsed  # iters/sec
            eta_s = (budget - k) / rate
            print(
                f"  FPS: {k}/{budget} ({100*k/budget:.1f}%)"
                f" {rate:.0f} iter/s | ETA {_fmt_duration(eta_s)}",
                flush=True,
            )
            t_last = time.time()

        if checkpoint_path is not None and k % checkpoint_every == 0:
            np.save(checkpoint_path / f"fps_selected{checkpoint_suffix}.npy", np.array(selected, dtype=np.int64))
            np.save(checkpoint_path / f"fps_min_dist_sq{checkpoint_suffix}.npy", min_dist_sq)

    # Final checkpoint save
    if checkpoint_path is not None:
        np.save(checkpoint_path / f"fps_selected{checkpoint_suffix}.npy", np.array(selected, dtype=np.int64))
        np.save(checkpoint_path / f"fps_min_dist_sq{checkpoint_suffix}.npy", min_dist_sq)

    return selected


def _update_min_dist_sq(
    descriptors: np.ndarray,
    sq_norms: np.ndarray,
    idx: int,
    min_dist_sq: np.ndarray,
) -> None:
    """In-place: min_dist_sq[i] = min(min_dist_sq[i], ||desc[i] - desc[idx]||²).

    Uses the identity ||a-b||² = ||a||² - 2(a·b) + ||b||² so the inner loop
    is a single BLAS GEMV (descriptors @ center) rather than an elementwise
    subtract-square-sum.
    """
    center = descriptors[idx]
    # descriptors @ center is BLAS GEMV — much faster than elementwise ops
    dot = descriptors @ center          # (N,) float32
    d_sq = sq_norms - 2.0 * dot + sq_norms[idx]
    np.maximum(d_sq, 0.0, out=d_sq)    # clamp numerical noise
    np.minimum(min_dist_sq, d_sq, out=min_dist_sq)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest: {manifest_path}")
    records = load_manifest(manifest_path)
    total = len(records)
    print(f"  {total} pairs in manifest")

    budget = max(1, int(round(args.fraction * total)))
    pct = int(round(args.fraction * 100))
    dtype = args.descriptor_type
    print(f"  Budget: {budget} ({pct}% of {total}), seed={args.seed}, descriptor={dtype}")

    # Cache files are namespaced by descriptor type.
    # 'bfv' keeps legacy names (no suffix) so a running BFv FPS job is unaffected.
    suffix = "" if dtype == "bfv" else f"_{dtype}"
    desc_cache = output_dir / f"fps_descriptors{suffix}.npy"
    idx_cache  = output_dir / f"fps_descriptor_indices{suffix}.npy"
    ckpt_sel   = output_dir / f"fps_selected{suffix}.npy"
    ckpt_dist  = output_dir / f"fps_min_dist_sq{suffix}.npy"

    if desc_cache.exists() and idx_cache.exists():
        print(f"Loading cached descriptors from {desc_cache}")
        descriptors = np.load(desc_cache)
        valid_indices = np.load(idx_cache).tolist()
        print(f"  {len(valid_indices)} descriptors loaded")
    else:
        if args.pool_multiplier is not None:
            pool_size = min(total, max(budget, int(round(args.pool_multiplier * budget))))
            print(
                f"  Pre-sampling candidate pool: {pool_size} pairs"
                f" ({args.pool_multiplier:.1f}× budget) from {total} total"
            )
            rng = np.random.RandomState(args.seed)
            candidate_indices = sorted(
                rng.choice(total, size=pool_size, replace=False).tolist()
            )
        else:
            print("  Using all pairs as candidates (no --pool_multiplier)")
            candidate_indices = list(range(total))

        if dtype in ("mean_mag", "median_mag"):
            # Read directly from manifest — no annotation loading needed
            manifest_field = "motion_mean" if dtype == "mean_mag" else "motion_median"
            print(f"Reading '{manifest_field}' from manifest (no anno loading)...")
            t0 = time.time()
            descriptors, valid_indices = compute_descriptors_from_manifest(
                records, candidate_indices, field=manifest_field
            )
            print(
                f"  {len(valid_indices)}/{len(candidate_indices)} valid in"
                f" {_fmt_duration(time.time() - t0)}"
            )
        else:
            # bfv or p90_mag — requires loading anno.npz per sequence
            print(
                f"Computing '{dtype}' descriptors for {len(candidate_indices)} candidates"
                f" ({args.num_workers} worker{'s' if args.num_workers > 1 else ''})..."
            )
            t0 = time.time()
            descriptors, valid_indices = compute_descriptors(
                records, candidate_indices,
                num_workers=args.num_workers,
                descriptor_type=dtype,
            )
            print(
                f"  {len(valid_indices)}/{len(candidate_indices)} valid descriptors"
                f" in {_fmt_duration(time.time() - t0)}"
            )

        np.save(desc_cache, descriptors)
        np.save(idx_cache, np.array(valid_indices, dtype=np.int64))
        print(f"  Descriptors cached to {desc_cache}")

    print("Standardizing descriptors...")
    scaler = StandardScaler()
    descriptors_normed = scaler.fit_transform(descriptors).astype(np.float32)

    budget = min(budget, len(valid_indices))
    print(f"Running FPS (N={len(valid_indices)}, budget={budget})...")
    t0 = time.time()
    fps_local = farthest_point_sampling(
        descriptors_normed,
        budget=budget,
        seed=args.seed,
        checkpoint_path=output_dir,
        checkpoint_every=5_000,
        checkpoint_suffix=suffix,
    )
    print(f"  FPS done in {_fmt_duration(time.time() - t0)}")

    selected = sorted(valid_indices[i] for i in fps_local)

    out_path = output_dir / f"subset_fps{suffix}_{pct}_seed{args.seed}.json"
    with out_path.open("w") as f:
        json.dump(selected, f)

    print(f"Saved {len(selected)} pairs → {out_path}")


if __name__ == "__main__":
    main()
