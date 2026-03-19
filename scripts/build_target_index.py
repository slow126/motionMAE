#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.flow_smoke.dataset import PointOdysseyFlowSmokeDataset, load_manifest
from src.pointodyssey_pairs.bfv import BFVConfig, flow_to_bfv, vectors_to_bfv
from src.pointodyssey_pairs.flow_stats import infer_mag_clip_from_magnitudes


def _infer_mag_clip_from_vector_file(
    vectors: np.ndarray,
    quantile: float,
    sample_size: int,
    seed: int,
) -> float:
    n = int(vectors.shape[0])
    if n == 0:
        return 1.0
    if vectors.shape[1] >= 4:
        start = 2
    elif vectors.shape[1] == 2:
        start = 0
    else:
        raise ValueError(f"Expected vectors with 2 or >=4 columns, got {vectors.shape}")

    rng = np.random.default_rng(seed)
    if sample_size > 0 and n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        arr = np.asarray(vectors[idx, start : start + 2], dtype=np.float32)
    else:
        arr = np.asarray(vectors[:, start : start + 2], dtype=np.float32)

    finite = np.isfinite(arr).all(axis=1)
    arr = arr[finite]
    if arr.size == 0:
        return 1.0
    mag = np.sqrt(arr[:, 0] * arr[:, 0] + arr[:, 1] * arr[:, 1])
    return infer_mag_clip_from_magnitudes(mag, quantile=quantile)


def _infer_mag_clip_from_manifest_dataset(
    dataset: PointOdysseyFlowSmokeDataset,
    max_samples: int,
    quantile: float,
) -> float:
    mags: List[np.ndarray] = []
    n = len(dataset) if max_samples <= 0 else min(len(dataset), int(max_samples))
    for i in range(n):
        sample = dataset[i]
        flow = sample["flow"].to(torch.float32)
        mask = sample.get("valid_flow_mask", torch.isfinite(flow).all(dim=0)).bool()
        if not mask.any():
            continue
        dx = flow[0][mask].detach().cpu().numpy()
        dy = flow[1][mask].detach().cpu().numpy()
        m = np.sqrt(dx * dx + dy * dy)
        if m.size > 0:
            mags.append(m.astype(np.float32, copy=False))
    if not mags:
        return 1.0
    cat = np.concatenate(mags, axis=0)
    return infer_mag_clip_from_magnitudes(cat, quantile=quantile)


def _build_from_vectors(
    vectors_path: Path,
    cfg: BFVConfig,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    vectors = np.load(vectors_path, mmap_mode="r")
    if vectors.ndim != 2:
        raise ValueError(f"Expected vectors shape (N,C), got {vectors.shape}")

    if chunk_size <= 0:
        bfv = vectors_to_bfv(vectors, cfg)[None, :]
        target_ids = np.array([0], dtype=np.int64)
    else:
        bfvs = []
        target_ids = []
        n = int(vectors.shape[0])
        for s in range(0, n, chunk_size):
            e = min(n, s + chunk_size)
            bfvs.append(vectors_to_bfv(vectors[s:e], cfg))
            target_ids.append(s // chunk_size)
        bfv = np.stack(bfvs, axis=0).astype(np.float32, copy=False)
        target_ids = np.asarray(target_ids, dtype=np.int64)

    meta = {
        "source": "vector_file",
        "vectors_path": str(vectors_path),
        "num_rows": int(vectors.shape[0]),
        "num_targets": int(bfv.shape[0]),
    }
    return bfv.astype(np.float32, copy=False), target_ids, meta


def _build_from_manifest(
    manifest_path: Path,
    pointodyssey_root: Optional[str],
    cfg: BFVConfig,
    size: int | None,
    trust_manifest: bool,
    max_target_samples: int,
    reverse_flow: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    entries = load_manifest(manifest_path)
    if max_target_samples > 0:
        entries = entries[:max_target_samples]
    idxs = list(range(len(entries)))
    dataset = PointOdysseyFlowSmokeDataset(
        manifest_path=manifest_path,
        indices=idxs,
        pointodyssey_root=pointodyssey_root,
        reverse_flow=reverse_flow,
        size=size,
        trust_manifest=trust_manifest,
        normalize_flow=False,
    )

    bfvs: List[np.ndarray] = []
    target_ids: List[int] = []
    manifest_indices: List[int] = []
    pair_ids: List[int] = []
    seq_ids: List[str] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        flow = sample["flow"]
        valid = sample.get("valid_flow_mask")
        bfv = flow_to_bfv(flow, valid, cfg)

        manifest_idx = int(sample["manifest_idx"].item())
        entry = dataset.entries[manifest_idx]
        pair_id = int(entry.get("pair_id", manifest_idx))
        seq_id = str(entry.get("seq_id", entry.get("seq_rel_path", entry.get("seq_path", ""))))

        bfvs.append(bfv)
        target_ids.append(pair_id)
        manifest_indices.append(manifest_idx)
        pair_ids.append(pair_id)
        seq_ids.append(seq_id)

        if (i + 1) % 500 == 0:
            print(f"[build_target_index] processed {i + 1}/{len(dataset)} target samples")

    bfv_mat = np.stack(bfvs, axis=0).astype(np.float32, copy=False) if bfvs else np.zeros((0, cfg.dim), dtype=np.float32)
    meta = {
        "source": "manifest",
        "manifest_path": str(manifest_path),
        "num_targets": int(bfv_mat.shape[0]),
    }
    return (
        bfv_mat,
        np.asarray(target_ids, dtype=np.int64),
        np.asarray(manifest_indices, dtype=np.int64),
        np.asarray(pair_ids, dtype=np.int64),
        np.asarray(seq_ids, dtype=str),
        meta,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build target BFV index for PointOdyssey pair scoring.")
    p.add_argument("--output", type=Path, required=True, help="Output .npz path for target BFV index.")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--target-vectors", type=Path, help="Target flow vectors .npy (e.g. *_flow.npy).")
    src.add_argument("--target-manifest", type=Path, help="Target pair manifest .jsonl.")

    p.add_argument("--pointodyssey-root", type=str, default=None)
    p.add_argument("--max-target-samples", type=int, default=0, help="0 = all")
    p.add_argument("--size", type=int, default=512, help="Resize for manifest-based flow extraction.")
    p.add_argument("--trust-manifest", action="store_true")
    p.add_argument("--reverse-flow", action="store_true", default=True)
    p.add_argument("--no-reverse-flow", dest="reverse_flow", action="store_false")

    p.add_argument("--angle-bins", type=int, default=8)
    p.add_argument("--mag-bins", type=int, default=4)
    p.add_argument("--mag-clip", type=float, default=None)
    p.add_argument("--mag-clip-quantile", type=float, default=0.99)
    p.add_argument("--mag-clip-sample-size", type=int, default=500000)
    p.add_argument(
        "--vector-chunk-size",
        type=int,
        default=8192,
        help="For vector-file mode: number of rows per target descriptor. Set 0 only if you explicitly want one global descriptor.",
    )
    p.add_argument("--seed", type=int, default=2021)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.target_vectors is not None:
        vec = np.load(args.target_vectors, mmap_mode="r")
        if args.mag_clip is None:
            mag_clip = _infer_mag_clip_from_vector_file(
                vec,
                quantile=args.mag_clip_quantile,
                sample_size=args.mag_clip_sample_size,
                seed=args.seed,
            )
        else:
            mag_clip = float(args.mag_clip)
        cfg = BFVConfig(
            angle_bins=int(args.angle_bins),
            mag_bins=int(args.mag_bins),
            mag_clip=float(mag_clip),
            l1_normalize=True,
        )
        bfv, target_ids, meta = _build_from_vectors(args.target_vectors, cfg, int(args.vector_chunk_size))
        target_manifest_idx = np.asarray([], dtype=np.int64)
        target_pair_id = np.asarray([], dtype=np.int64)
        target_seq_id = np.asarray([], dtype=str)
    else:
        manifest_path = args.target_manifest
        assert manifest_path is not None
        entries = load_manifest(manifest_path)
        idxs = list(range(len(entries if args.max_target_samples <= 0 else entries[: args.max_target_samples])))
        dataset_for_clip = PointOdysseyFlowSmokeDataset(
            manifest_path=manifest_path,
            indices=idxs,
            pointodyssey_root=args.pointodyssey_root,
            reverse_flow=args.reverse_flow,
            size=args.size,
            trust_manifest=args.trust_manifest,
            normalize_flow=False,
        )
        if args.mag_clip is None:
            mag_clip = _infer_mag_clip_from_manifest_dataset(
                dataset_for_clip,
                max_samples=min(len(dataset_for_clip), 1000),
                quantile=args.mag_clip_quantile,
            )
        else:
            mag_clip = float(args.mag_clip)
        cfg = BFVConfig(
            angle_bins=int(args.angle_bins),
            mag_bins=int(args.mag_bins),
            mag_clip=float(mag_clip),
            l1_normalize=True,
        )
        (
            bfv,
            target_ids,
            target_manifest_idx,
            target_pair_id,
            target_seq_id,
            meta,
        ) = _build_from_manifest(
            manifest_path=manifest_path,
            pointodyssey_root=args.pointodyssey_root,
            cfg=cfg,
            size=args.size,
            trust_manifest=args.trust_manifest,
            max_target_samples=args.max_target_samples,
            reverse_flow=args.reverse_flow,
        )

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        bfv=bfv.astype(np.float32, copy=False),
        target_ids=target_ids.astype(np.int64, copy=False),
        target_manifest_idx=target_manifest_idx,
        target_pair_id=target_pair_id,
        target_seq_id=target_seq_id,
        angle_bins=np.asarray([cfg.angle_bins], dtype=np.int32),
        mag_bins=np.asarray([cfg.mag_bins], dtype=np.int32),
        mag_clip=np.asarray([cfg.mag_clip], dtype=np.float32),
        metadata_json=np.asarray([json.dumps(meta)], dtype=str),
    )

    print(f"[build_target_index] saved: {out}")
    print(
        f"[build_target_index] target_bfv_shape={bfv.shape} angle_bins={cfg.angle_bins} "
        f"mag_bins={cfg.mag_bins} mag_clip={cfg.mag_clip:.6f}"
    )


if __name__ == "__main__":
    main()
