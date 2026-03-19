#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.flow_smoke.dataset import PointOdysseyFlowSmokeDataset, load_manifest
from src.pointodyssey_pairs.bfv import BFVConfig, flow_to_bfv, vectors_to_bfv
from src.pointodyssey_pairs.flow_stats import compute_scalar_stats, infer_mag_clip_from_magnitudes


def _parse_indices(path: Path) -> List[int]:
    suffix = path.suffix.lower()
    if suffix in [".json", ".jsn"]:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            if "indices" in data:
                data = data["indices"]
            elif "subset" in data:
                data = data["subset"]
        return [int(x) for x in data]
    if suffix in [".npy", ".npz"]:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            if "indices" in arr.files:
                arr = arr["indices"]
            elif "subset" in arr.files:
                arr = arr["subset"]
            else:
                arr = arr[arr.files[0]]
        return [int(x) for x in np.asarray(arr).reshape(-1)]
    if suffix in [".pt", ".pth"]:
        import torch

        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            if "indices" in data:
                data = data["indices"]
            elif "subset" in data:
                data = data["subset"]
        return [int(x) for x in data]
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def _parse_k_values(raw: str) -> List[int]:
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        vals = [1, 5]
    return vals


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


class TargetSearcher:
    def __init__(
        self,
        target_bfv: np.ndarray,
        target_ids: np.ndarray,
        metric: str = "l2",
        use_faiss: bool = False,
        faiss_gpu: bool = False,
        faiss_index_type: str = "flat",
        faiss_nlist: int = 1024,
        faiss_nprobe: int = 64,
    ):
        self.metric = metric
        self.target_ids = target_ids.astype(np.int64, copy=False)
        self.target_bfv = target_bfv.astype(np.float32, copy=False)
        self._faiss = None
        self._index = None

        if self.target_bfv.ndim != 2:
            raise ValueError(f"Expected target_bfv shape (N,D), got {self.target_bfv.shape}")

        if metric == "cosine":
            self.target_proc = _l2_normalize_rows(self.target_bfv)
        elif metric == "l2":
            self.target_proc = self.target_bfv
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if use_faiss:
            self._build_faiss(
                faiss_gpu=faiss_gpu,
                faiss_index_type=faiss_index_type,
                faiss_nlist=faiss_nlist,
                faiss_nprobe=faiss_nprobe,
            )

    def _build_faiss(self, faiss_gpu: bool, faiss_index_type: str, faiss_nlist: int, faiss_nprobe: int) -> None:
        try:
            import faiss
        except Exception as exc:
            print(f"[score_source_samples] FAISS requested but unavailable: {exc}. Falling back to exact search.")
            return

        self._faiss = faiss
        d = int(self.target_proc.shape[1])
        metric_const = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT

        kind = faiss_index_type.lower().strip()
        if kind == "flat":
            if self.metric == "l2":
                index = faiss.IndexFlatL2(d)
            else:
                index = faiss.IndexFlatIP(d)
        elif kind == "ivf_flat":
            quantizer = faiss.IndexFlatL2(d) if self.metric == "l2" else faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, int(max(1, faiss_nlist)), metric_const)
            index.train(self.target_proc)
            index.nprobe = int(max(1, faiss_nprobe))
        else:
            raise ValueError(f"Unsupported FAISS index type: {faiss_index_type}")

        if faiss_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception as exc:
                print(f"[score_source_samples] FAISS GPU setup failed: {exc}. Continuing on CPU.")

        index.add(self.target_proc)
        self._index = index

    def _exact_search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.metric == "l2":
            diff = self.target_proc - q[None, :]
            d = np.sqrt(np.maximum(np.sum(diff * diff, axis=1), 0.0))
            idx = np.argsort(d)[:k]
            return d[idx], idx

        sims = self.target_proc @ q
        d = 1.0 - sims
        idx = np.argsort(d)[:k]
        return d[idx], idx

    def _exact_search_many(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = q.astype(np.float32, copy=False)
        t = self.target_proc
        if self.metric == "l2":
            q2 = np.sum(q * q, axis=1, keepdims=True)
            t2 = np.sum(t * t, axis=1, keepdims=True).T
            d2 = np.maximum(q2 + t2 - 2.0 * (q @ t.T), 0.0)
            part_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
            part_d2 = np.take_along_axis(d2, part_idx, axis=1)
            order = np.argsort(part_d2, axis=1)
            idx = np.take_along_axis(part_idx, order, axis=1).astype(np.int64, copy=False)
            d = np.sqrt(np.maximum(np.take_along_axis(part_d2, order, axis=1), 0.0))
            return d.astype(np.float32, copy=False), idx

        sims = q @ t.T
        d = 1.0 - sims
        part_idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
        part_d = np.take_along_axis(d, part_idx, axis=1)
        order = np.argsort(part_d, axis=1)
        idx = np.take_along_axis(part_idx, order, axis=1).astype(np.int64, copy=False)
        d_sorted = np.take_along_axis(part_d, order, axis=1)
        return d_sorted.astype(np.float32, copy=False), idx

    def search(self, query_bfv: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = int(max(1, min(k, self.target_proc.shape[0])))
        q = query_bfv.astype(np.float32, copy=False).reshape(-1)
        if self.metric == "cosine":
            norm = float(np.linalg.norm(q))
            if norm > 0:
                q = q / norm

        if self._index is None:
            return self._exact_search(q, k)

        D, I = self._index.search(q[None, :], k)
        idx = I[0].astype(np.int64, copy=False)
        d = D[0].astype(np.float32, copy=False)
        if self.metric == "l2":
            d = np.sqrt(np.maximum(d, 0.0))
        else:
            d = 1.0 - d
        return d, idx

    def search_many(self, query_bfv: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = int(max(1, min(k, self.target_proc.shape[0])))
        q = np.asarray(query_bfv, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2:
            raise ValueError(f"Expected query_bfv shape (B,D) or (D,), got {q.shape}")

        if self.metric == "cosine":
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            q = q / norms

        if self._index is None:
            return self._exact_search_many(q, k)

        D, I = self._index.search(q, k)
        idx = I.astype(np.int64, copy=False)
        d = D.astype(np.float32, copy=False)
        if self.metric == "l2":
            d = np.sqrt(np.maximum(d, 0.0))
        else:
            d = 1.0 - d
        return d, idx


def _load_target_from_index(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[float], Optional[int], Optional[int], dict]:
    data = np.load(path, allow_pickle=True)
    bfv = np.asarray(data["bfv"], dtype=np.float32)
    target_ids = np.asarray(data["target_ids"], dtype=np.int64)
    mag_clip = None
    angle_bins = None
    mag_bins = None
    if "mag_clip" in data.files and data["mag_clip"].size > 0:
        mag_clip = float(data["mag_clip"].reshape(-1)[0])
    if "angle_bins" in data.files and data["angle_bins"].size > 0:
        angle_bins = int(data["angle_bins"].reshape(-1)[0])
    if "mag_bins" in data.files and data["mag_bins"].size > 0:
        mag_bins = int(data["mag_bins"].reshape(-1)[0])
    meta = {}
    if "metadata_json" in data.files and data["metadata_json"].size > 0:
        try:
            meta = json.loads(str(data["metadata_json"].reshape(-1)[0]))
        except Exception:
            meta = {}
    return bfv, target_ids, mag_clip, angle_bins, mag_bins, meta


def _load_target_from_vectors(path: Path, cfg: BFVConfig) -> Tuple[np.ndarray, np.ndarray]:
    vectors = np.load(path, mmap_mode="r")
    bfv = vectors_to_bfv(vectors, cfg)[None, :]
    target_ids = np.asarray([0], dtype=np.int64)
    return bfv.astype(np.float32, copy=False), target_ids


def _infer_mag_clip_from_target_vectors(path: Path, quantile: float = 0.99, sample_size: int = 500000, seed: int = 2021) -> float:
    vectors = np.load(path, mmap_mode="r")
    if vectors.shape[1] < 4:
        raise ValueError(f"Expected vector rows with >=4 columns in {path}, got {vectors.shape}")
    n = int(vectors.shape[0])
    rng = np.random.default_rng(seed)
    if sample_size > 0 and n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        dx = vectors[idx, 2].astype(np.float32, copy=False)
        dy = vectors[idx, 3].astype(np.float32, copy=False)
    else:
        dx = vectors[:, 2].astype(np.float32, copy=False)
        dy = vectors[:, 3].astype(np.float32, copy=False)
    finite = np.isfinite(dx) & np.isfinite(dy)
    if not np.any(finite):
        return 1.0
    mag = np.sqrt(dx[finite] * dx[finite] + dy[finite] * dy[finite])
    return infer_mag_clip_from_magnitudes(mag, quantile=quantile)


def _resolve_anno_path(entry: Dict, pointodyssey_root: Optional[str]) -> Path:
    root = Path(pointodyssey_root) if pointodyssey_root is not None else None
    if root is not None and entry.get("anno_rel_path"):
        return root / str(entry["anno_rel_path"])
    if entry.get("anno_path"):
        return Path(str(entry["anno_path"]))
    if root is not None and entry.get("seq_rel_path"):
        return root / str(entry["seq_rel_path"]) / "anno.npz"
    if entry.get("seq_path"):
        return Path(str(entry["seq_path"])) / "anno.npz"
    raise KeyError("Manifest entry missing anno path fields")


class _SparseAnnoCache:
    def __init__(self):
        self._cached_path: Optional[Path] = None
        self._cached_ann: Optional[Dict[str, np.ndarray]] = None

    def get(self, anno_path: Path) -> Dict[str, np.ndarray]:
        if self._cached_path == anno_path and self._cached_ann is not None:
            return self._cached_ann
        with np.load(anno_path, allow_pickle=True) as npz:
            ann = {
                "trajs_2d": np.asarray(npz["trajs_2d"]),
                "valids": np.asarray(npz["valids"]),
            }
            if "visibs" in npz.files:
                ann["visibs"] = np.asarray(npz["visibs"])
        self._cached_path = anno_path
        self._cached_ann = ann
        return ann


def _extract_sparse_dxdy(
    ann: Dict[str, np.ndarray],
    entry: Dict,
    reverse_flow: bool,
    trust_manifest: bool,
    max_displacement: Optional[float],
    max_points_per_pair: Optional[int],
) -> Tuple[np.ndarray, int, int]:
    frame_i = int(entry["frame_i"])
    frame_j = int(entry["frame_j"])
    if reverse_flow:
        src_idx, trg_idx = frame_j, frame_i
    else:
        src_idx, trg_idx = frame_i, frame_j

    trajs = ann["trajs_2d"]
    valids = ann["valids"]
    src_pts = np.asarray(trajs[src_idx], dtype=np.float32)
    trg_pts = np.asarray(trajs[trg_idx], dtype=np.float32)
    src_valid = np.asarray(valids[src_idx], dtype=np.float32) > 0
    trg_valid = np.asarray(valids[trg_idx], dtype=np.float32) > 0

    valid = src_valid & trg_valid
    valid &= np.isfinite(src_pts).all(axis=1)
    valid &= np.isfinite(trg_pts).all(axis=1)
    valid &= ~((src_pts[:, 0] == 0.0) & (src_pts[:, 1] == 0.0))
    valid &= ~((trg_pts[:, 0] == 0.0) & (trg_pts[:, 1] == 0.0))

    if (not trust_manifest) and ("visibs" in ann):
        vis = ann["visibs"]
        src_vis = np.asarray(vis[src_idx], dtype=np.float32)
        trg_vis = np.asarray(vis[trg_idx], dtype=np.float32)
        valid &= (src_vis > 0) & (trg_vis > 0)

    if max_displacement is not None and max_displacement > 0:
        idx = np.flatnonzero(valid)
        if idx.size > 0:
            disp = src_pts[idx] - trg_pts[idx]
            keep = np.linalg.norm(disp, axis=1) <= float(max_displacement)
            valid[idx] &= keep

    valid_idx = np.flatnonzero(valid)
    if max_points_per_pair is not None and max_points_per_pair > 0 and valid_idx.size > max_points_per_pair:
        valid_idx = valid_idx[: int(max_points_per_pair)]

    if valid_idx.size == 0:
        return np.zeros((0, 2), dtype=np.float32), 0, int(src_pts.shape[0])

    dxdy = (src_pts[valid_idx] - trg_pts[valid_idx]).astype(np.float32, copy=False)
    return dxdy, int(valid_idx.size), int(src_pts.shape[0])


def _compute_scalar_stats_from_dxdy(dxdy: np.ndarray, n_total: int) -> Dict[str, float]:
    n_valid = int(dxdy.shape[0])
    if n_valid == 0:
        return {
            "mean_mag": float("nan"),
            "median_mag": float("nan"),
            "p90_mag": float("nan"),
            "p95_mag": float("nan"),
            "valid_fraction": float(0.0 if n_total > 0 else float("nan")),
            "n_valid": float(0),
        }
    mag = np.sqrt(dxdy[:, 0] * dxdy[:, 0] + dxdy[:, 1] * dxdy[:, 1])
    return {
        "mean_mag": float(np.mean(mag)),
        "median_mag": float(np.quantile(mag, 0.5)),
        "p90_mag": float(np.quantile(mag, 0.9)),
        "p95_mag": float(np.quantile(mag, 0.95)),
        "valid_fraction": float(n_valid / max(1, n_total)),
        "n_valid": float(n_valid),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score PointOdyssey source samples with scalar and BFV target matching.")
    p.add_argument("--manifest-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p.add_argument("--pointodyssey-root", type=str, default=None)
    p.add_argument("--subset-indices-path", type=Path, default=None)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all active samples")

    p.add_argument("--size", type=int, default=512)
    p.add_argument("--trust-manifest", action="store_true")
    p.add_argument("--reverse-flow", action="store_true", default=True)
    p.add_argument("--no-reverse-flow", dest="reverse_flow", action="store_false")
    p.add_argument("--max-points-per-pair", type=int, default=None)
    p.add_argument("--max-displacement", type=float, default=None)
    p.add_argument(
        "--sparse-kp-mode",
        action="store_true",
        help="Fast path: compute stats/BFV directly from anno keypoint displacements (no image decode, no dense flow).",
    )

    p.add_argument("--target-index", type=Path, default=None, help="Path produced by build_target_index.py")
    p.add_argument("--target-vectors", type=Path, default=None, help="Direct target flow vectors .npy")
    p.add_argument("--distance", type=str, choices=["l2", "cosine"], default="l2")
    p.add_argument("--k-values", type=str, default="1,5,10")

    p.add_argument("--angle-bins", type=int, default=8)
    p.add_argument("--mag-bins", type=int, default=4)
    p.add_argument("--mag-clip", type=float, default=None)
    p.add_argument("--mag-clip-quantile", type=float, default=0.99)

    p.add_argument("--use-faiss", action="store_true")
    p.add_argument("--faiss-gpu", action="store_true")
    p.add_argument("--faiss-index-type", type=str, default="flat", choices=["flat", "ivf_flat"])
    p.add_argument("--faiss-nlist", type=int, default=1024)
    p.add_argument("--faiss-nprobe", type=int, default=64)

    p.add_argument("--save-format", type=str, default="csv", choices=["csv", "parquet"])
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--query-batch-size", type=int, default=8192)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_index is not None and args.target_vectors is not None:
        raise ValueError("Use only one of --target-index or --target-vectors.")

    k_values = _parse_k_values(args.k_values)
    k_max = max(k_values)

    target_bfv = None
    target_ids = None
    index_meta = {}

    angle_bins = int(args.angle_bins)
    mag_bins = int(args.mag_bins)

    if args.target_index is not None:
        (
            target_bfv,
            target_ids,
            index_mag_clip,
            index_angle_bins,
            index_mag_bins,
            index_meta,
        ) = _load_target_from_index(args.target_index)
        if index_angle_bins is not None:
            angle_bins = int(index_angle_bins)
        if index_mag_bins is not None:
            mag_bins = int(index_mag_bins)
        mag_clip = float(args.mag_clip) if args.mag_clip is not None else float(index_mag_clip if index_mag_clip is not None else 32.0)
    elif args.target_vectors is not None:
        if args.mag_clip is None:
            mag_clip = _infer_mag_clip_from_target_vectors(
                args.target_vectors,
                quantile=args.mag_clip_quantile,
                sample_size=500000,
                seed=args.seed,
            )
        else:
            mag_clip = float(args.mag_clip)
    else:
        mag_clip = float(args.mag_clip) if args.mag_clip is not None else 32.0

    cfg = BFVConfig(
        angle_bins=angle_bins,
        mag_bins=mag_bins,
        mag_clip=float(mag_clip),
        l1_normalize=True,
    )

    if args.target_vectors is not None:
        target_bfv, target_ids = _load_target_from_vectors(args.target_vectors, cfg)

    searcher = None
    if target_bfv is not None and target_ids is not None and target_bfv.shape[0] > 0:
        searcher = TargetSearcher(
            target_bfv=target_bfv,
            target_ids=target_ids,
            metric=args.distance,
            use_faiss=args.use_faiss,
            faiss_gpu=args.faiss_gpu,
            faiss_index_type=args.faiss_index_type,
            faiss_nlist=args.faiss_nlist,
            faiss_nprobe=args.faiss_nprobe,
        )
        print(
            f"[score_source_samples] target index loaded: n={target_bfv.shape[0]} d={target_bfv.shape[1]} "
            f"metric={args.distance}"
        )
        if int(target_bfv.shape[0]) == 1:
            print(
                "[score_source_samples] WARNING: target index has n=1. "
                "NN search is trivial and GPU utilization will stay near 0%. "
                "Rebuild target index with a positive --vector-chunk-size (e.g. 4096 or 8192)."
            )
    else:
        print("[score_source_samples] running scalar-only (no target index provided)")

    entries = load_manifest(args.manifest_path)
    indices = list(range(len(entries)))
    if args.subset_indices_path is not None:
        subset = set(_parse_indices(args.subset_indices_path))
        indices = [i for i in indices if i in subset]
    if args.max_samples > 0:
        indices = indices[: int(args.max_samples)]

    dataset = None
    if not args.sparse_kp_mode:
        dataset = PointOdysseyFlowSmokeDataset(
            manifest_path=args.manifest_path,
            indices=indices,
            pointodyssey_root=args.pointodyssey_root,
            reverse_flow=args.reverse_flow,
            size=args.size,
            trust_manifest=args.trust_manifest,
            max_points_per_pair=args.max_points_per_pair,
            max_displacement=args.max_displacement,
            normalize_flow=False,
            seed=args.seed,
        )
    else:
        print("[score_source_samples] sparse-kp-mode enabled: skipping image decode and dense flow construction")

    rows = []
    pending_rows: List[dict] = []
    pending_q: List[np.ndarray] = []
    query_batch_size = int(max(1, args.query_batch_size))
    if searcher is not None:
        print(
            f"[score_source_samples] query batch size: {query_batch_size}. "
            f"FAISS search launches every {query_batch_size} processed samples."
        )

    def _flush_pending() -> None:
        nonlocal pending_rows, pending_q, rows
        if searcher is None or not pending_q:
            rows.extend(pending_rows)
            pending_rows = []
            pending_q = []
            return
        q = np.stack(pending_q, axis=0).astype(np.float32, copy=False)
        d_mat, nn_idx_mat = searcher.search_many(q, k=min(k_max, searcher.target_bfv.shape[0]))
        for b, row in enumerate(pending_rows):
            d = d_mat[b]
            nn_idx = nn_idx_mat[b]
            nearest_target_id = int(searcher.target_ids[int(nn_idx[0])]) if len(nn_idx) > 0 else -1
            row["nearest_target_id"] = nearest_target_id
            row["target_nn_dist_bfv"] = float(d[0]) if len(d) > 0 else float("nan")
            for k in k_values:
                kk = min(k, len(d))
                val = float(np.mean(d[:kk])) if kk > 0 else float("nan")
                row[f"target_knn{k}_mean_dist_bfv"] = val
            rows.append(row)
        pending_rows = []
        pending_q = []

    if args.sparse_kp_mode:
        n_iter = len(indices)
        ann_cache = _SparseAnnoCache()
        for i, manifest_idx in enumerate(indices):
            entry = entries[manifest_idx]
            anno_path = _resolve_anno_path(entry, args.pointodyssey_root)
            ann = ann_cache.get(anno_path)
            dxdy, n_valid, n_total = _extract_sparse_dxdy(
                ann=ann,
                entry=entry,
                reverse_flow=args.reverse_flow,
                trust_manifest=args.trust_manifest,
                max_displacement=args.max_displacement,
                max_points_per_pair=args.max_points_per_pair,
            )
            stats = _compute_scalar_stats_from_dxdy(dxdy, n_total=n_total)

            sample_id = int(entry.get("pair_id", manifest_idx))
            clip_id = str(entry.get("seq_id", entry.get("seq_rel_path", entry.get("seq_path", ""))))

            row = {
                "sample_id": sample_id,
                "clip_id": clip_id,
                "manifest_idx": manifest_idx,
                "frame_i": int(entry.get("frame_i", -1)),
                "frame_j": int(entry.get("frame_j", -1)),
                "dt": int(entry.get("dt", int(entry.get("frame_j", 0)) - int(entry.get("frame_i", 0)))),
                "mean_mag": stats["mean_mag"],
                "median_mag": stats["median_mag"],
                "p90_mag": stats["p90_mag"],
                "p95_mag": stats["p95_mag"],
                "valid_fraction": stats["valid_fraction"],
                "n_valid": int(stats["n_valid"]),
            }

            if searcher is not None:
                query = vectors_to_bfv(dxdy, cfg)
                pending_rows.append(row)
                pending_q.append(query)
                if len(pending_q) >= query_batch_size:
                    _flush_pending()
            else:
                row["nearest_target_id"] = -1
                row["target_nn_dist_bfv"] = float("nan")
                for k in k_values:
                    row[f"target_knn{k}_mean_dist_bfv"] = float("nan")
                rows.append(row)

            if (i + 1) % 500 == 0:
                print(f"[score_source_samples] processed {i + 1}/{n_iter}")
    else:
        assert dataset is not None
        for i in range(len(dataset)):
            sample = dataset[i]
            flow = sample["flow"]
            valid_mask = sample.get("valid_flow_mask")
            stats = compute_scalar_stats(flow, valid_mask)

            manifest_idx = int(sample["manifest_idx"].item())
            entry = dataset.entries[manifest_idx]
            sample_id = int(entry.get("pair_id", manifest_idx))
            clip_id = str(entry.get("seq_id", entry.get("seq_rel_path", entry.get("seq_path", ""))))

            row = {
                "sample_id": sample_id,
                "clip_id": clip_id,
                "manifest_idx": manifest_idx,
                "frame_i": int(entry.get("frame_i", -1)),
                "frame_j": int(entry.get("frame_j", -1)),
                "dt": int(entry.get("dt", int(entry.get("frame_j", 0)) - int(entry.get("frame_i", 0)))),
                "mean_mag": stats["mean_mag"],
                "median_mag": stats["median_mag"],
                "p90_mag": stats["p90_mag"],
                "p95_mag": stats["p95_mag"],
                "valid_fraction": stats["valid_fraction"],
                "n_valid": int(stats["n_valid"]),
            }

            if searcher is not None:
                query = flow_to_bfv(flow, valid_mask, cfg)
                pending_rows.append(row)
                pending_q.append(query)
                if len(pending_q) >= query_batch_size:
                    _flush_pending()
            else:
                row["nearest_target_id"] = -1
                row["target_nn_dist_bfv"] = float("nan")
                for k in k_values:
                    row[f"target_knn{k}_mean_dist_bfv"] = float("nan")
                rows.append(row)
            if (i + 1) % 500 == 0:
                print(f"[score_source_samples] processed {i + 1}/{len(dataset)}")

    _flush_pending()

    df = pd.DataFrame(rows)
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_format == "parquet":
        try:
            df.to_parquet(out, index=False)
        except Exception as exc:
            csv_fallback = out.with_suffix(".csv")
            df.to_csv(csv_fallback, index=False)
            print(f"[score_source_samples] parquet write failed ({exc}), wrote CSV instead: {csv_fallback}")
    else:
        df.to_csv(out, index=False)

    summary = {
        "manifest_path": str(args.manifest_path),
        "n_rows": int(len(df)),
        "angle_bins": cfg.angle_bins,
        "mag_bins": cfg.mag_bins,
        "mag_clip": cfg.mag_clip,
        "distance": args.distance,
        "k_values": k_values,
        "target_meta": index_meta,
    }
    print(f"[score_source_samples] wrote {len(df)} rows -> {out}")
    print(f"[score_source_samples] summary: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
