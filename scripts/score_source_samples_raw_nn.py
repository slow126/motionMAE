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

from src.flow_smoke.dataset import load_manifest


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
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def _parse_k_values(raw: str) -> List[int]:
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    vals = sorted(set(v for v in vals if v > 0))
    return vals or [1]


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


def _extract_sparse_vec4(
    ann: Dict[str, np.ndarray],
    entry: Dict,
    reverse_flow: bool,
    trust_manifest: bool,
    max_displacement: Optional[float],
    max_points_per_pair: Optional[int],
) -> Tuple[np.ndarray, int]:
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
        return np.zeros((0, 4), dtype=np.float32), int(src_pts.shape[0])

    dxdy = (src_pts[valid_idx] - trg_pts[valid_idx]).astype(np.float32, copy=False)
    xy = trg_pts[valid_idx].astype(np.float32, copy=False)
    vec4 = np.concatenate([xy, dxdy], axis=1).astype(np.float32, copy=False)
    return vec4, int(src_pts.shape[0])


def _extract_sparse_pair_entry_vec4(
    entry: Dict,
    max_displacement: Optional[float],
    max_points_per_pair: Optional[int],
) -> Tuple[np.ndarray, int]:
    src_pts = np.asarray(entry["src_kps"], dtype=np.float32)
    trg_pts = np.asarray(entry["trg_kps"], dtype=np.float32)
    if src_pts.ndim != 2 or src_pts.shape[1] != 2 or trg_pts.shape != src_pts.shape:
        raise ValueError(
            f"Expected pooled sparse entry src_kps/trg_kps shape (N,2), got {src_pts.shape} and {trg_pts.shape}"
        )

    valid = np.isfinite(src_pts).all(axis=1)
    valid &= np.isfinite(trg_pts).all(axis=1)
    valid &= ~((src_pts[:, 0] == 0.0) & (src_pts[:, 1] == 0.0))
    valid &= ~((trg_pts[:, 0] == 0.0) & (trg_pts[:, 1] == 0.0))

    if max_displacement is not None and max_displacement > 0:
        idx = np.flatnonzero(valid)
        if idx.size > 0:
            disp = trg_pts[idx] - src_pts[idx]
            keep = np.linalg.norm(disp, axis=1) <= float(max_displacement)
            valid[idx] &= keep

    valid_idx = np.flatnonzero(valid)
    if max_points_per_pair is not None and max_points_per_pair > 0 and valid_idx.size > max_points_per_pair:
        valid_idx = valid_idx[: int(max_points_per_pair)]

    if valid_idx.size == 0:
        return np.zeros((0, 4), dtype=np.float32), int(src_pts.shape[0])

    src_keep = src_pts[valid_idx].astype(np.float32, copy=False)
    trg_keep = trg_pts[valid_idx].astype(np.float32, copy=False)
    dxdy = (trg_keep - src_keep).astype(np.float32, copy=False)
    vec4 = np.concatenate([src_keep, dxdy], axis=1).astype(np.float32, copy=False)
    return vec4, int(src_pts.shape[0])


def _maybe_fixed_k_subsample(
    vec4: np.ndarray,
    fixed_query_k: int,
    seed: int,
    manifest_idx: int,
    source_dataset: str,
) -> np.ndarray:
    if source_dataset != "pointodyssey" or fixed_query_k <= 0 or vec4.shape[0] <= fixed_query_k:
        return vec4
    rng = np.random.default_rng(int(seed) + int(manifest_idx) * 73856093 + vec4.shape[0] * 19349663)
    chosen = rng.choice(vec4.shape[0], size=int(fixed_query_k), replace=False)
    return vec4[np.sort(chosen)]


def _prepare_raw_vectors(
    vec4: np.ndarray,
    raw_space: str,
    normalize_norm2x1: bool,
    norm_width: float,
    norm_height: float,
) -> np.ndarray:
    vec4 = np.asarray(vec4, dtype=np.float32)
    out = vec4[:, :4].astype(np.float32, copy=True)

    if normalize_norm2x1:
        w = max(float(norm_width), 1.0)
        h = max(float(norm_height), 1.0)
        out[:, 0] = (2.0 * out[:, 0] / w) - 1.0
        out[:, 1] = (2.0 * out[:, 1] / h) - 1.0
        out[:, 2] = 2.0 * out[:, 2] / w
        out[:, 3] = 2.0 * out[:, 3] / h

    if raw_space == "flow":
        return out[:, 2:4]
    if raw_space == "joint":
        return out
    raise ValueError(f"Unsupported raw_space: {raw_space}")


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


class RawVectorSearcher:
    def __init__(
        self,
        target_vectors: np.ndarray,
        metric: str,
        use_faiss: bool,
        faiss_gpu: bool,
        faiss_index_type: str,
        faiss_nlist: int,
        faiss_nprobe: int,
    ):
        self.metric = metric
        self.target_vectors = np.asarray(target_vectors, dtype=np.float32)
        if self.target_vectors.ndim != 2:
            raise ValueError(f"Expected target vectors shape (N,D), got {self.target_vectors.shape}")
        self.target_proc = _l2_normalize_rows(self.target_vectors) if metric == "cosine" else self.target_vectors
        self._index = None

        if use_faiss:
            self._build_faiss(faiss_gpu, faiss_index_type, faiss_nlist, faiss_nprobe)

    def _build_faiss(self, faiss_gpu: bool, faiss_index_type: str, faiss_nlist: int, faiss_nprobe: int) -> None:
        try:
            import faiss
        except Exception as exc:
            print(f"[raw_nn] FAISS unavailable: {exc}. Using exact CPU search.")
            return

        d = int(self.target_proc.shape[1])
        metric_const = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT
        kind = faiss_index_type.lower().strip()
        if kind == "flat":
            index = faiss.IndexFlatL2(d) if self.metric == "l2" else faiss.IndexFlatIP(d)
        elif kind == "ivf_flat":
            quantizer = faiss.IndexFlatL2(d) if self.metric == "l2" else faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, int(max(1, faiss_nlist)), metric_const)
            index.train(self.target_proc)
            index.nprobe = int(max(1, faiss_nprobe))
        else:
            raise ValueError(f"Unsupported faiss index type: {faiss_index_type}")

        if faiss_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception as exc:
                print(f"[raw_nn] FAISS GPU setup failed: {exc}. Continuing on CPU.")
        index.add(self.target_proc)
        self._index = index

    def search_many(self, query_vectors: np.ndarray, k: int) -> np.ndarray:
        q = np.asarray(query_vectors, dtype=np.float32)
        if q.ndim != 2:
            raise ValueError(f"Expected query vectors shape (B,D), got {q.shape}")
        if q.shape[0] == 0:
            return np.zeros((0, 1), dtype=np.float32)
        k = int(max(1, min(k, self.target_proc.shape[0])))
        if self.metric == "cosine":
            q = _l2_normalize_rows(q)

        if self._index is not None:
            D, _ = self._index.search(q, k)
            d = D.astype(np.float32, copy=False)
            if self.metric == "l2":
                d = np.sqrt(np.maximum(d, 0.0))
            else:
                d = 1.0 - d
            return d

        if self.metric == "l2":
            q2 = np.sum(q * q, axis=1, keepdims=True)
            t2 = np.sum(self.target_proc * self.target_proc, axis=1, keepdims=True).T
            d2 = np.maximum(q2 + t2 - 2.0 * (q @ self.target_proc.T), 0.0)
            part_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
            part_d2 = np.take_along_axis(d2, part_idx, axis=1)
            order = np.argsort(part_d2, axis=1)
            return np.sqrt(np.maximum(np.take_along_axis(part_d2, order, axis=1), 0.0)).astype(np.float32, copy=False)
        sims = q @ self.target_proc.T
        d = 1.0 - sims
        part_idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
        part_d = np.take_along_axis(d, part_idx, axis=1)
        order = np.argsort(part_d, axis=1)
        return np.take_along_axis(part_d, order, axis=1).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raw NN scorer: average per-sample NN distance to target vectors.")
    p.add_argument("--manifest-path", type=Path, required=True)
    p.add_argument("--target-vectors", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pointodyssey-root", type=str, default=None)
    p.add_argument("--subset-indices-path", type=Path, default=None)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--reverse-flow", action="store_true", default=True)
    p.add_argument("--no-reverse-flow", dest="reverse_flow", action="store_false")
    p.add_argument("--trust-manifest", action="store_true")
    p.add_argument("--max-displacement", type=float, default=None)
    p.add_argument("--max-points-per-pair", type=int, default=128)
    p.add_argument("--distance", type=str, choices=["l2", "cosine"], default="l2")
    p.add_argument("--k-values", type=str, default="1")
    p.add_argument("--raw-space", type=str, choices=["flow", "joint"], default="flow")
    p.add_argument("--normalize-norm2x1", dest="normalize_norm2x1", action="store_true", default=True)
    p.add_argument("--no-normalize-norm2x1", dest="normalize_norm2x1", action="store_false")
    p.add_argument("--norm-width", type=float, default=512.0)
    p.add_argument("--norm-height", type=float, default=512.0)
    p.add_argument("--use-faiss", action="store_true")
    p.add_argument("--faiss-gpu", action="store_true")
    p.add_argument("--faiss-index-type", type=str, default="flat", choices=["flat", "ivf_flat"])
    p.add_argument("--faiss-nlist", type=int, default=1024)
    p.add_argument("--faiss-nprobe", type=int, default=64)
    p.add_argument("--query-batch-size", type=int, default=8192, help="Samples per batched search flush")
    p.add_argument("--save-format", type=str, default="csv", choices=["csv", "parquet"])
    p.add_argument("--top-fraction", type=float, default=None, help="Optional top fraction by k=1 distance")
    p.add_argument("--subset-output", type=Path, default=None, help="Optional JSON subset output path")
    p.add_argument(
        "--fixed-query-k",
        type=int,
        default=0,
        help="Optional fixed K for source-side query points. If >0, randomly subsample each query to K points when it has more than K.",
    )
    p.add_argument("--seed", type=int, default=2021)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    k_values = _parse_k_values(args.k_values)
    k_max = max(k_values)

    target_all = np.load(args.target_vectors, mmap_mode="r")
    if target_all.ndim != 2 or target_all.shape[1] < 4:
        raise ValueError(f"Expected target vectors shape (N,4+), got {target_all.shape}")
    target_vectors = _prepare_raw_vectors(
        target_all[:, :4],
        raw_space=args.raw_space,
        normalize_norm2x1=args.normalize_norm2x1,
        norm_width=args.norm_width,
        norm_height=args.norm_height,
    )
    searcher = RawVectorSearcher(
        target_vectors=target_vectors,
        metric=args.distance,
        use_faiss=args.use_faiss,
        faiss_gpu=args.faiss_gpu,
        faiss_index_type=args.faiss_index_type,
        faiss_nlist=args.faiss_nlist,
        faiss_nprobe=args.faiss_nprobe,
    )
    print(f"[raw_nn] target loaded: n={target_vectors.shape[0]} d={target_vectors.shape[1]} metric={args.distance}")

    entries = load_manifest(args.manifest_path)
    indices = list(range(len(entries)))
    if args.subset_indices_path is not None:
        subset = set(_parse_indices(args.subset_indices_path))
        indices = [i for i in indices if i in subset]
    if args.max_samples > 0:
        indices = indices[: int(args.max_samples)]

    cache = _SparseAnnoCache()
    rows: List[Dict] = []
    pending_rows: List[Dict] = []
    pending_queries: List[np.ndarray] = []
    batch_size = int(max(1, args.query_batch_size))
    print(f"[raw_nn] query batch size: {batch_size}")

    def flush() -> None:
        nonlocal pending_rows, pending_queries, rows
        if not pending_rows:
            return
        lengths = [q.shape[0] for q in pending_queries]
        total_q = int(sum(lengths))
        if total_q == 0:
            for row in pending_rows:
                row["target_nn_mean_dist_raw"] = float("nan")
                for k in k_values:
                    row[f"target_knn{k}_mean_dist_raw"] = float("nan")
                rows.append(row)
            pending_rows = []
            pending_queries = []
            return

        q = np.concatenate(pending_queries, axis=0).astype(np.float32, copy=False)
        d = searcher.search_many(q, k=k_max)
        start = 0
        for row, ln in zip(pending_rows, lengths):
            if ln <= 0:
                row["target_nn_mean_dist_raw"] = float("nan")
                for k in k_values:
                    row[f"target_knn{k}_mean_dist_raw"] = float("nan")
                rows.append(row)
                continue
            end = start + ln
            ds = d[start:end]
            start = end
            row["target_nn_mean_dist_raw"] = float(np.mean(ds[:, 0]))
            for k in k_values:
                kk = min(k, ds.shape[1])
                row[f"target_knn{k}_mean_dist_raw"] = float(np.mean(np.mean(ds[:, :kk], axis=1)))
            rows.append(row)
        pending_rows = []
        pending_queries = []

    for i, manifest_idx in enumerate(indices):
        entry = entries[manifest_idx]
        source_dataset = str(entry.get("source_dataset", "pointodyssey")).lower()
        if source_dataset == "pointodyssey":
            ann = cache.get(_resolve_anno_path(entry, args.pointodyssey_root))
            vec4, n_total = _extract_sparse_vec4(
                ann=ann,
                entry=entry,
                reverse_flow=args.reverse_flow,
                trust_manifest=args.trust_manifest,
                max_displacement=args.max_displacement,
                max_points_per_pair=args.max_points_per_pair,
            )
        elif source_dataset in {"spair", "pfpascal", "pfwillow"}:
            vec4, n_total = _extract_sparse_pair_entry_vec4(
                entry=entry,
                max_displacement=args.max_displacement,
                max_points_per_pair=args.max_points_per_pair,
            )
        else:
            raise ValueError(f"Unsupported source_dataset in manifest: {source_dataset}")

        vec4 = _maybe_fixed_k_subsample(
            vec4,
            fixed_query_k=int(args.fixed_query_k),
            seed=int(args.seed),
            manifest_idx=int(manifest_idx),
            source_dataset=source_dataset,
        )
        n_valid = int(vec4.shape[0])
        if n_valid > 0:
            mag = np.sqrt(vec4[:, 2] * vec4[:, 2] + vec4[:, 3] * vec4[:, 3])
            mean_mag = float(np.mean(mag))
            p90_mag = float(np.quantile(mag, 0.9))
        else:
            mean_mag = float("nan")
            p90_mag = float("nan")

        row = {
            "sample_id": int(entry.get("pair_id", manifest_idx)),
            "clip_id": str(entry.get("seq_id", entry.get("seq_rel_path", entry.get("seq_path", "")))),
            "manifest_idx": int(manifest_idx),
            "source_dataset": source_dataset,
            "source_split": str(entry.get("source_split", "train")),
            "source_sample_id": int(entry.get("source_sample_id", entry.get("pair_id", manifest_idx))),
            "frame_i": int(entry.get("frame_i", -1)),
            "frame_j": int(entry.get("frame_j", -1)),
            "dt": int(entry.get("dt", int(entry.get("frame_j", 0)) - int(entry.get("frame_i", 0)))),
            "n_valid": int(n_valid),
            "valid_fraction": float(n_valid / max(1, n_total)),
            "mean_mag": mean_mag,
            "p90_mag": p90_mag,
        }
        q = _prepare_raw_vectors(
            vec4,
            raw_space=args.raw_space,
            normalize_norm2x1=args.normalize_norm2x1,
            norm_width=args.norm_width,
            norm_height=args.norm_height,
        )
        pending_rows.append(row)
        pending_queries.append(q)
        if len(pending_rows) >= batch_size:
            flush()

        if (i + 1) % 500 == 0:
            print(f"[raw_nn] processed {i + 1}/{len(indices)}")

    flush()

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_format == "parquet":
        try:
            df.to_parquet(args.output, index=False)
        except Exception as exc:
            fallback = args.output.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            print(f"[raw_nn] parquet write failed ({exc}), wrote CSV: {fallback}")
    else:
        df.to_csv(args.output, index=False)
    print(f"[raw_nn] wrote {len(df)} rows -> {args.output}")

    if args.top_fraction is not None:
        frac = float(args.top_fraction)
        if not (0.0 < frac <= 1.0):
            raise ValueError("--top-fraction must be in (0, 1]")
        score_col = "target_knn1_mean_dist_raw" if "target_knn1_mean_dist_raw" in df.columns else "target_nn_mean_dist_raw"
        valid_df = df.dropna(subset=[score_col]).sort_values(score_col, ascending=True)
        budget = max(1, int(len(valid_df) * frac))
        chosen = valid_df.head(budget)
        subset_path = args.subset_output or args.output.with_name(args.output.stem + f"_top_{int(frac*100)}pct_indices.json")
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"indices": [int(x) for x in chosen["manifest_idx"].tolist()]}
        subset_path.write_text(json.dumps(payload, indent=2))
        print(f"[raw_nn] wrote subset {len(payload['indices'])} -> {subset_path}")


if __name__ == "__main__":
    main()
