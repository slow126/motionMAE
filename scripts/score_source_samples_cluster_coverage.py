#!/usr/bin/env python3
"""Score source samples by hard target-centroid coverage.

Each source sample is converted into raw vectors in the same space as the target
cluster index. Each query vector is matched to its nearest target centroid, and
that centroid is counted as covered when the distance is within the centroid's
precomputed hard radius.

The output is a CSV with per-sample sparse centroid hits plus a distance-based
base score that can be used for shortlisting and tie-breaking.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.flow_smoke.dataset import load_manifest
from score_source_samples_raw_nn import (
    RawVectorSearcher,
    _SparseAnnoCache,
    _extract_sparse_pair_entry_vec4,
    _extract_sparse_vec4,
    _maybe_fixed_k_subsample,
    _parse_indices,
    _prepare_raw_vectors,
    _resolve_anno_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hard target-centroid coverage scorer")
    p.add_argument("--manifest-path", type=Path, required=True)
    p.add_argument("--cluster-index-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pointodyssey-root", type=str, default=None)
    p.add_argument("--subset-indices-path", type=Path, default=None)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--reverse-flow", action="store_true", default=True)
    p.add_argument("--no-reverse-flow", dest="reverse_flow", action="store_false")
    p.add_argument("--trust-manifest", action="store_true")
    p.add_argument("--max-displacement", type=float, default=None)
    p.add_argument("--max-points-per-pair", type=int, default=128)
    p.add_argument("--use-faiss", action="store_true")
    p.add_argument("--faiss-gpu", action="store_true")
    p.add_argument("--faiss-index-type", type=str, default="flat", choices=["flat", "ivf_flat"])
    p.add_argument("--faiss-nlist", type=int, default=1024)
    p.add_argument("--faiss-nprobe", type=int, default=64)
    p.add_argument("--query-batch-size", type=int, default=8192, help="Samples per progress flush.")
    p.add_argument("--fixed-query-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    return p.parse_args()


def _hits_to_string(hits: np.ndarray) -> str:
    if hits.size == 0:
        return ""
    return " ".join(str(int(x)) for x in hits.tolist())


def main() -> None:
    args = parse_args()
    blob = np.load(args.cluster_index_path, allow_pickle=False)
    centroids = np.asarray(blob["centroids"], dtype=np.float32)
    radii = np.asarray(blob["radii"], dtype=np.float32).reshape(-1)
    counts = np.asarray(blob["counts"], dtype=np.int32).reshape(-1)
    meta = {}
    if "meta_json" in blob.files:
        meta = json.loads(str(np.asarray(blob["meta_json"]).item()))
    if centroids.ndim != 2 or radii.shape[0] != centroids.shape[0]:
        raise ValueError(
            f"Cluster index mismatch: centroids {centroids.shape}, radii {radii.shape}, counts {counts.shape}"
        )

    searcher = RawVectorSearcher(
        target_vectors=centroids,
        metric="l2",
        use_faiss=args.use_faiss,
        faiss_gpu=args.faiss_gpu,
        faiss_index_type=args.faiss_index_type,
        faiss_nlist=args.faiss_nlist,
        faiss_nprobe=args.faiss_nprobe,
    )
    print(f"[cluster_cov] cluster index loaded: k={centroids.shape[0]} d={centroids.shape[1]}")

    entries = load_manifest(args.manifest_path)
    indices = list(range(len(entries)))
    if args.subset_indices_path is not None:
        subset = set(_parse_indices(args.subset_indices_path))
        indices = [i for i in indices if i in subset]
    if args.max_samples > 0:
        indices = indices[: int(args.max_samples)]

    cache = _SparseAnnoCache()
    rows: List[Dict] = []
    batch_size = int(max(1, args.query_batch_size))
    print(f"[cluster_cov] sample batch size: {batch_size}")

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
        q = _prepare_raw_vectors(
            vec4,
            raw_space=str(meta.get("raw_space", "joint" if centroids.shape[1] == 4 else "flow")),
            normalize_norm2x1=bool(meta.get("normalize_norm2x1", True)),
            norm_width=float(meta.get("norm_width", 512.0)),
            norm_height=float(meta.get("norm_height", 512.0)),
        )
        n_valid = int(q.shape[0])

        if n_valid > 0:
            if searcher._index is not None:
                D, I = searcher._index.search(np.asarray(q, dtype=np.float32), 1)
                d = np.sqrt(np.maximum(D.reshape(-1), 0.0)).astype(np.float32, copy=False)
                nearest_idx = I.reshape(-1).astype(np.int32, copy=False)
            else:
                d2 = np.maximum(
                    np.sum(q * q, axis=1, keepdims=True)
                    + np.sum(centroids * centroids, axis=1, keepdims=True).T
                    - 2.0 * (q @ centroids.T),
                    0.0,
                )
                d = np.sqrt(np.min(d2, axis=1)).astype(np.float32, copy=False)
                nearest_idx = np.argmin(d2, axis=1).astype(np.int32, copy=False)

            hit_mask = d <= radii[nearest_idx]
            hit_centroids = np.unique(nearest_idx[hit_mask]).astype(np.int32, copy=False)
            hit_weight = int(np.sum(counts[hit_centroids])) if hit_centroids.size > 0 else 0
            mean_min_dist = float(np.mean(d))
            mean_hit_dist = float(np.mean(d[hit_mask])) if np.any(hit_mask) else float("nan")
            hit_fraction = float(np.mean(hit_mask.astype(np.float32)))
        else:
            hit_centroids = np.zeros((0,), dtype=np.int32)
            hit_weight = 0
            mean_min_dist = float("nan")
            mean_hit_dist = float("nan")
            hit_fraction = float("nan")

        rows.append(
            {
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
                "n_covered_centroids": int(hit_centroids.size),
                "covered_centroid_weight": int(hit_weight),
                "covered_centroids": _hits_to_string(hit_centroids),
                "target_cluster_mean_min_dist": mean_min_dist,
                "target_cluster_mean_hit_dist": mean_hit_dist,
                "target_cluster_hit_fraction": hit_fraction,
            }
        )

        if (i + 1) % 500 == 0:
            print(f"[cluster_cov] processed {i + 1}/{len(indices)}")
        if (i + 1) % batch_size == 0:
            pass

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[cluster_cov] wrote {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
