#!/usr/bin/env python3
"""Build deterministic PointOdyssey frame-pair manifests for smoke tests."""

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Build PointOdyssey pair manifest")
    parser.add_argument("--pointodyssey_root", type=str, required=True, help="PointOdyssey dataset root")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for manifest/subset outputs")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--max_sequences", type=int, default=3, help="Number of sequences to use")
    parser.add_argument("--min_valid_points", type=int, default=8, help="Min valid correspondences per pair")
    parser.add_argument("--subset_fraction", type=float, default=0.30, help="Fraction for random/heuristic subsets")
    parser.add_argument("--seed", type=int, default=2021, help="Seed for deterministic subset generation")
    parser.add_argument("--fail_streak_stop", type=int, default=3, help="Stop dt expansion after N consecutive fails")
    parser.add_argument(
        "--progress_every_frames",
        type=int,
        default=100,
        help="Print progress every N anchor frames within each sequence",
    )
    parser.add_argument(
        "--max_dt",
        type=int,
        default=None,
        help="Optional hard cap on dt (default: adaptive only)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for per-sequence manifest building (default: 1)",
    )
    return parser.parse_args()


def _resolve_sequences(root: Path, split: str, max_sequences: int) -> List[Path]:
    split_dir = root / split
    if split_dir.exists():
        seq_parent = split_dir
    else:
        seq_parent = root
    sequences = sorted([p for p in seq_parent.iterdir() if p.is_dir()])
    if max_sequences is not None and max_sequences > 0:
        sequences = sequences[:max_sequences]
    return sequences


def _pair_stats(
    trajs_2d: np.ndarray,
    valids: np.ndarray,
    visibs: Optional[np.ndarray],
    frame_i: int,
    frame_j: int,
) -> Tuple[int, float, float]:
    src = np.asarray(trajs_2d[frame_i], dtype=np.float32)
    trg = np.asarray(trajs_2d[frame_j], dtype=np.float32)
    src_valid = np.asarray(valids[frame_i], dtype=np.float32) > 0
    trg_valid = np.asarray(valids[frame_j], dtype=np.float32) > 0

    valid_mask = src_valid & trg_valid
    valid_mask &= np.isfinite(src).all(axis=1)
    valid_mask &= np.isfinite(trg).all(axis=1)
    valid_mask &= ~np.logical_and(src[:, 0] == 0.0, src[:, 1] == 0.0)
    valid_mask &= ~np.logical_and(trg[:, 0] == 0.0, trg[:, 1] == 0.0)

    if visibs is not None:
        src_vis = np.asarray(visibs[frame_i], dtype=np.float32) > 0
        trg_vis = np.asarray(visibs[frame_j], dtype=np.float32) > 0
        valid_mask &= src_vis & trg_vis

    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return 0, 0.0, 0.0

    deltas = trg[valid_mask] - src[valid_mask]
    mags = np.linalg.norm(deltas, axis=1)
    motion_mean = float(np.mean(mags))
    motion_median = float(np.median(mags))
    return n_valid, motion_mean, motion_median


def _build_manifest_records_for_sequence(
    seq_id: int,
    seq_path: str,
    pointodyssey_root: str,
    min_valid_points: int,
    fail_streak_stop: int,
    max_dt: Optional[int],
) -> Dict:
    seq_path = Path(seq_path)
    pointodyssey_root = Path(pointodyssey_root)
    anno_path = seq_path / "anno.npz"
    rgbs_dir = seq_path / "rgbs"
    if not anno_path.exists() or not rgbs_dir.exists():
        return {
            "seq_id": int(seq_id),
            "seq_name": str(seq_path.name),
            "status": "skipped",
            "reason": "missing anno.npz or rgbs/",
            "records": [],
            "pairs_added": 0,
            "seq_elapsed_s": 0.0,
        }

    seq_start = time.time()
    records: List[Dict] = []
    with np.load(anno_path, allow_pickle=True, mmap_mode="r") as annotations:
        trajs_2d = annotations["trajs_2d"]
        valids = annotations["valids"]
        visibs = annotations["visibs"] if "visibs" in annotations.files else None
        num_frames = int(trajs_2d.shape[0])

        for frame_i in range(0, num_frames - 1):
            fail_streak = 0
            dt = 1
            while frame_i + dt < num_frames:
                if max_dt is not None and dt > max_dt:
                    break
                frame_j = frame_i + dt
                rgb_i = rgbs_dir / f"rgb_{frame_i:05d}.jpg"
                rgb_j = rgbs_dir / f"rgb_{frame_j:05d}.jpg"
                if not rgb_i.exists() or not rgb_j.exists():
                    fail_streak += 1
                    if fail_streak >= fail_streak_stop:
                        break
                    dt += 1
                    continue

                n_valid, motion_mean, motion_median = _pair_stats(
                    trajs_2d=trajs_2d,
                    valids=valids,
                    visibs=visibs,
                    frame_i=frame_i,
                    frame_j=frame_j,
                )

                if n_valid >= min_valid_points:
                    records.append(
                        {
                            "seq_id": int(seq_id),
                            "seq_path": str(seq_path.resolve()),
                            "seq_rel_path": str(seq_path.relative_to(pointodyssey_root)),
                            "anno_path": str(anno_path.resolve()),
                            "anno_rel_path": str(anno_path.relative_to(pointodyssey_root)),
                            "frame_i": int(frame_i),
                            "frame_j": int(frame_j),
                            "dt": int(dt),
                            "valid_points": int(n_valid),
                            "motion_mean": float(motion_mean),
                            "motion_median": float(motion_median),
                        }
                    )
                    fail_streak = 0
                else:
                    fail_streak += 1
                    if fail_streak >= fail_streak_stop:
                        break
                dt += 1
    seq_elapsed = time.time() - seq_start

    return {
        "seq_id": int(seq_id),
        "seq_name": str(seq_path.name),
        "status": "ok",
        "records": records,
        "pairs_added": len(records),
        "seq_elapsed_s": float(seq_elapsed),
    }


def _build_manifest_records(
    pointodyssey_root: Path,
    split: str,
    max_sequences: int,
    min_valid_points: int,
    fail_streak_stop: int,
    max_dt: Optional[int],
    num_workers: int = 1,
) -> List[Dict]:
    records: List[Dict] = []
    sequences = _resolve_sequences(pointodyssey_root, split, max_sequences)
    overall_start = time.time()
    n_sequences = len(sequences)
    num_workers = max(1, min(int(num_workers), n_sequences)) if n_sequences > 0 else 1
    root_str = str(pointodyssey_root)
    if num_workers <= 1:
        for seq_id, seq_path in enumerate(sequences):
            result = _build_manifest_records_for_sequence(
                seq_id=seq_id,
                seq_path=str(seq_path),
                pointodyssey_root=root_str,
                min_valid_points=min_valid_points,
                fail_streak_stop=fail_streak_stop,
                max_dt=max_dt,
            )
            if result["status"] == "skipped":
                print(
                    f"[{seq_id + 1}/{n_sequences}] Skipping {result['seq_name']}: "
                    f"{result['reason']}",
                    flush=True,
                )
            else:
                print(
                    f"[{seq_id + 1}/{n_sequences}] Done {result['seq_name']}: "
                    f"pairs={result['pairs_added']}, seq_time={result['seq_elapsed_s']:6.1f}s",
                    flush=True,
                )
            records.extend(result["records"])
    else:
        args = [
            (
                seq_id,
                str(seq_path),
                root_str,
                min_valid_points,
                fail_streak_stop,
                max_dt,
            )
            for seq_id, seq_path in enumerate(sequences)
        ]

        completed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {
                ex.submit(_build_manifest_records_for_sequence, *seq_args): seq_args[0]
                for seq_args in args
            }
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                completed += 1
                seq_id = futures[fut]
                if result["status"] == "skipped":
                    print(
                        f"[{seq_id + 1}/{n_sequences}] Skipping {result['seq_name']}: {result['reason']}",
                        flush=True,
                    )
                    continue
                seq_elapsed = result.get("seq_elapsed_s", 0.0)
                seq_pairs = result.get("pairs_added", len(result["records"]))
                overall_elapsed = time.time() - overall_start
                seqs_done = completed
                avg_seq_time = overall_elapsed / max(seqs_done, 1)
                eta_sequences = avg_seq_time * max(0, n_sequences - seqs_done)
                print(
                    f"[{seq_id + 1}/{n_sequences}] Done {result['seq_name']}: "
                    f"pairs={seq_pairs}, seq_time={seq_elapsed:6.1f}s, "
                    f"overall_elapsed={overall_elapsed:6.1f}s, overall_eta={eta_sequences:6.1f}s",
                    flush=True,
                )
                records.extend(result["records"])

    records = sorted(records, key=lambda r: (r["seq_id"], r["frame_i"], r["frame_j"]))
    for pair_id, row in enumerate(records):
        row["pair_id"] = int(pair_id)
    return records


def _subset_random(total: int, k: int, seed: int) -> List[int]:
    if total <= 0 or k <= 0:
        return []
    rng = np.random.default_rng(seed)
    if k >= total:
        return list(range(total))
    chosen = rng.choice(np.arange(total), size=k, replace=False).tolist()
    chosen.sort()
    return [int(x) for x in chosen]


def _subset_heuristic_balanced(records: List[Dict], k: int, seed: int) -> List[int]:
    total = len(records)
    if total <= 0 or k <= 0:
        return []
    if k >= total:
        return list(range(total))

    rng = np.random.default_rng(seed)
    motion = np.asarray([r["motion_median"] for r in records], dtype=np.float64)

    edges = np.quantile(motion, [0.0, 0.25, 0.5, 0.75, 1.0])
    edges = np.maximum.accumulate(edges)
    bins = np.digitize(motion, edges[1:-1], right=False)  # 0..3

    base = k // 4
    rem = k % 4
    targets = [base + (1 if i < rem else 0) for i in range(4)]

    selected: List[int] = []
    selected_set = set()
    for b in range(4):
        bin_indices = np.where(bins == b)[0]
        if len(bin_indices) == 0:
            continue
        perm = rng.permutation(bin_indices)
        take_n = min(targets[b], len(perm))
        chosen = [int(x) for x in perm[:take_n].tolist()]
        selected.extend(chosen)
        selected_set.update(chosen)

    deficit = k - len(selected)
    if deficit > 0:
        remaining = [idx for idx in range(total) if idx not in selected_set]
        if remaining:
            rem_perm = rng.permutation(np.asarray(remaining, dtype=np.int64))
            selected.extend([int(x) for x in rem_perm[:deficit].tolist()])

    selected = sorted(selected[:k])
    return selected


def _subset_stats(records: List[Dict], indices: List[int]) -> Dict:
    if not indices:
        return {
            "count": 0,
            "avg_dt": 0.0,
            "avg_valid_points": 0.0,
            "motion_mean": 0.0,
            "motion_median": 0.0,
            "motion_bin_counts": [0, 0, 0, 0],
            "motion_quantiles": [0.0, 0.0, 0.0, 0.0, 0.0],
            "valid_points_bin_edges": [8, 16, 32, 64, 128],
            "valid_points_bin_counts": [0, 0, 0, 0, 0, 0],
            "valid_points_quantiles": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    sel = [records[i] for i in indices]
    dt_arr = np.asarray([r["dt"] for r in sel], dtype=np.float64)
    vp_arr = np.asarray([r["valid_points"] for r in sel], dtype=np.float64)
    mm_arr = np.asarray([r["motion_median"] for r in sel], dtype=np.float64)
    m_arr = np.asarray([r["motion_mean"] for r in sel], dtype=np.float64)
    q = np.quantile(mm_arr, [0.25, 0.5, 0.75])
    bins = np.digitize(mm_arr, q, right=False)
    bin_counts = [int((bins == i).sum()) for i in range(4)]
    motion_quantiles = np.quantile(mm_arr, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
    vp_edges = [8, 16, 32, 64, 128]
    vp_bins = np.digitize(vp_arr, vp_edges, right=False)
    vp_counts = [int((vp_bins == i).sum()) for i in range(len(vp_edges) + 1)]
    vp_quantiles = np.quantile(vp_arr, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
    return {
        "count": int(len(indices)),
        "avg_dt": float(dt_arr.mean()),
        "avg_valid_points": float(vp_arr.mean()),
        "motion_mean": float(m_arr.mean()),
        "motion_median": float(mm_arr.mean()),
        "motion_bin_counts": bin_counts,
        "motion_quantiles": [float(x) for x in motion_quantiles],
        "valid_points_bin_edges": vp_edges,
        "valid_points_bin_counts": vp_counts,
        "valid_points_quantiles": [float(x) for x in vp_quantiles],
    }


def _write_json(path: Path, payload):
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main():
    args = parse_args()
    root = Path(args.pointodyssey_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise FileNotFoundError(f"PointOdyssey root not found: {root}")

    records = _build_manifest_records(
        pointodyssey_root=root,
        split=args.split,
        max_sequences=args.max_sequences,
        min_valid_points=args.min_valid_points,
        fail_streak_stop=args.fail_streak_stop,
        max_dt=args.max_dt,
        num_workers=args.num_workers,
    )
    if not records:
        raise RuntimeError("No valid pairs generated for manifest")

    total = len(records)
    subset_k = int(round(float(args.subset_fraction) * total))
    subset_k = max(0, min(subset_k, total))
    pct = int(round(float(args.subset_fraction) * 100))

    random_indices = _subset_random(total=total, k=subset_k, seed=args.seed)
    heuristic_indices = _subset_heuristic_balanced(records=records, k=subset_k, seed=args.seed)

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for row in records:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    random_path = output_dir / f"subset_random_{pct}_seed{args.seed}.json"
    heuristic_path = output_dir / f"subset_heuristic_balanced_{pct}_seed{args.seed}.json"

    _write_json(random_path, random_indices)
    _write_json(heuristic_path, heuristic_indices)

    full_indices = list(range(total))
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    stats = {
        "config": {
            "pointodyssey_root": str(root),
            "split": args.split,
            "max_sequences": int(args.max_sequences),
            "min_valid_points": int(args.min_valid_points),
            "subset_fraction": float(args.subset_fraction),
            "seed": int(args.seed),
            "fail_streak_stop": int(args.fail_streak_stop),
            "max_dt": None if args.max_dt is None else int(args.max_dt),
        },
        "paths": {
            "manifest": str(manifest_path),
            "subset_random": str(random_path),
            "subset_heuristic_balanced": str(heuristic_path),
        },
        "hashes": {
            "manifest_sha256": manifest_sha256,
        },
        "subsets": {
            "full": _subset_stats(records, full_indices),
            "random": _subset_stats(records, random_indices),
            "heuristic_balanced": _subset_stats(records, heuristic_indices),
        },
    }
    stats_path = output_dir / "manifest_stats.json"
    _write_json(stats_path, stats)

    print("PointOdyssey pair-manifest build complete")
    print(f"  manifest: {manifest_path}")
    print(f"  stats: {stats_path}")
    print(f"  random subset: {random_path} ({len(random_indices)}/{total})")
    print(f"  heuristic subset: {heuristic_path} ({len(heuristic_indices)}/{total})")
    print("  subset stats:")
    for name in ["full", "random", "heuristic_balanced"]:
        s = stats["subsets"][name]
        print(
            f"    {name:18} count={s['count']:6d} avg_dt={s['avg_dt']:.2f} "
            f"avg_valid_points={s['avg_valid_points']:.2f} motion_mean={s['motion_mean']:.2f} "
            f"motion_q50={s['motion_quantiles'][2]:.2f} valid_q50={s['valid_points_quantiles'][2]:.2f}"
        )


if __name__ == "__main__":
    main()
