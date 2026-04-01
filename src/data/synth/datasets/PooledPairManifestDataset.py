"""
Generic manifest-backed pair dataset for pooled training candidates.

This is an opt-in companion to PointOdysseyPairManifestDataset. It supports
mixing PointOdyssey pair-manifest rows with sparse semantic-pair rows such as
SPAIR while preserving source provenance in each sample's metadata.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


class PooledPairManifestDataset(torch.utils.data.Dataset):
    """Manifest-backed dataset for mixed candidate pools."""

    def __init__(
        self,
        manifest_path: str,
        subset_mode: str = "full",
        subset_indices_path: Optional[str] = None,
        seed: int = 2021,
        reverse_flow: bool = True,
        pointodyssey_root: Optional[str] = None,
        verbose: bool = False,
        trust_manifest: bool = False,
        max_points_per_pair: Optional[int] = None,
        random_subsample_within_pair: bool = False,
        cache_arrays_in_memory: bool = True,
        max_displacement: Optional[float] = None,
        profile: bool = False,
        profile_every: int = 200,
    ):
        self.manifest_path = Path(manifest_path)
        self.subset_mode = (subset_mode or "full").lower()
        self.subset_indices_path = Path(subset_indices_path) if subset_indices_path else None
        self.seed = int(seed)
        self.reverse_flow = bool(reverse_flow)
        self.pointodyssey_root = Path(pointodyssey_root) if pointodyssey_root else None
        self.verbose = bool(verbose)
        self.trust_manifest = bool(trust_manifest)
        self.max_points_per_pair = None if max_points_per_pair is None else int(max_points_per_pair)
        self.random_subsample_within_pair = bool(random_subsample_within_pair)
        self.cache_arrays_in_memory = bool(cache_arrays_in_memory)
        self.max_displacement = None if max_displacement is None else float(max_displacement)
        self.profile = bool(profile)
        self.profile_every = max(1, int(profile_every))

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        self.entries = self._load_manifest(self.manifest_path)
        if not self.entries:
            raise RuntimeError(f"Manifest is empty: {self.manifest_path}")

        self.active_indices = self._resolve_active_indices()
        if not self.active_indices:
            raise RuntimeError(
                f"Subset mode '{self.subset_mode}' resolved to 0 samples for {self.manifest_path}"
            )

        self._cached_annotation_path: Optional[Path] = None
        self._cached_annotations = None
        self._window_start: int = 0
        self._window_length: Optional[int] = None

        self._profile_samples = 0
        self._profile_total_s = 0.0
        self._profile_img_s = 0.0
        self._profile_anno_s = 0.0
        self._profile_kps_s = 0.0

        if self.verbose:
            counts: Dict[str, int] = {}
            for idx in self.active_indices:
                name = str(self.entries[idx].get("source_dataset", "pointodyssey"))
                counts[name] = counts.get(name, 0) + 1
            print(
                f"[PooledPairs] Loaded {len(self.active_indices)} / {len(self.entries)} pairs "
                f"(mode={self.subset_mode}, counts={counts})"
            )

    @staticmethod
    def _load_manifest(path: Path) -> List[Dict]:
        entries: List[Dict] = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        return entries

    @staticmethod
    def _coerce_indices(raw) -> List[int]:
        if isinstance(raw, dict):
            if "indices" in raw:
                raw = raw["indices"]
            elif "subset" in raw:
                raw = raw["subset"]
            else:
                raise TypeError(f"Unsupported subset dict format: {list(raw.keys())}")
        return [int(x) for x in raw]

    def _default_subset_path(self) -> Optional[Path]:
        parent = self.manifest_path.parent
        if self.subset_mode == "random":
            return parent / f"subset_random_30_seed{self.seed}.json"
        if self.subset_mode == "heuristic":
            return parent / f"subset_heuristic_balanced_30_seed{self.seed}.json"
        return None

    def _load_subset_indices(self, subset_path: Path) -> List[int]:
        if not subset_path.exists():
            raise FileNotFoundError(f"Subset file not found: {subset_path}")
        with subset_path.open("r") as f:
            data = json.load(f)
        indices = self._coerce_indices(data)
        return [idx for idx in indices if 0 <= idx < len(self.entries)]

    def _resolve_active_indices(self) -> List[int]:
        if self.subset_mode == "full":
            return list(range(len(self.entries)))

        subset_path = self.subset_indices_path or self._default_subset_path()
        if subset_path is None:
            raise ValueError(
                f"subset_mode={self.subset_mode!r} requires subset_indices_path "
                "or a default subset file in the manifest directory"
            )
        return self._load_subset_indices(subset_path)

    def __len__(self) -> int:
        total = len(self.active_indices) - self._window_start
        if total <= 0:
            return 0
        if self._window_length is None:
            return total
        return min(total, self._window_length)

    def set_epoch_window(self, start_idx: int, length: Optional[int]) -> None:
        if length is not None and length < 0:
            raise ValueError(f"length must be >= 0, got {length}")
        if start_idx < 0:
            raise ValueError(f"start_idx must be >= 0, got {start_idx}")
        self._window_start = int(start_idx)
        self._window_length = None if length is None else int(length)
        if self._window_start > len(self.active_indices):
            self._window_start = len(self.active_indices)

    @staticmethod
    def _read_rgb(path: Path) -> torch.Tensor:
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img_rgb).permute(2, 0, 1).to(torch.float32) / 255.0

    def _resolve_seq_path(self, entry: Dict) -> Path:
        if self.pointodyssey_root is not None and entry.get("seq_rel_path"):
            return self.pointodyssey_root / str(entry["seq_rel_path"])
        if entry.get("seq_path"):
            return Path(str(entry["seq_path"]))
        raise KeyError("PointOdyssey manifest entry missing both seq_rel_path and seq_path")

    def _resolve_anno_path(self, entry: Dict, seq_path: Path) -> Path:
        if self.pointodyssey_root is not None and entry.get("anno_rel_path"):
            return self.pointodyssey_root / str(entry["anno_rel_path"])
        if entry.get("anno_path"):
            return Path(str(entry["anno_path"]))
        return seq_path / "anno.npz"

    def _load_annotations(self, anno_path: Path):
        if self._cached_annotation_path == anno_path and self._cached_annotations is not None:
            return self._cached_annotations
        if self.cache_arrays_in_memory:
            with np.load(anno_path, allow_pickle=True) as npz:
                annotations = {
                    "trajs_2d": np.asarray(npz["trajs_2d"]),
                    "valids": np.asarray(npz["valids"]),
                }
                if "visibs" in npz.files:
                    annotations["visibs"] = np.asarray(npz["visibs"])
        else:
            annotations = np.load(anno_path, allow_pickle=True, mmap_mode="r")
        self._cached_annotation_path = anno_path
        self._cached_annotations = annotations
        return annotations

    def _sample_valid_indices(self, valid_idx: np.ndarray, target_n: int, pair_seed: int) -> np.ndarray:
        if valid_idx.size <= target_n:
            return valid_idx
        if self.random_subsample_within_pair:
            rng = np.random.default_rng(pair_seed)
            chosen = rng.choice(valid_idx.size, size=target_n, replace=False)
            valid_idx = valid_idx[np.sort(chosen)]
        else:
            valid_idx = valid_idx[:target_n]
        return valid_idx

    def _limit_sparse_pair(
        self,
        src_pts: np.ndarray,
        trg_pts: np.ndarray,
        pair_seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        valid = np.isfinite(src_pts).all(axis=1) & np.isfinite(trg_pts).all(axis=1)
        valid &= ~((src_pts[:, 0] == 0.0) & (src_pts[:, 1] == 0.0))
        valid &= ~((trg_pts[:, 0] == 0.0) & (trg_pts[:, 1] == 0.0))
        if self.max_displacement is not None and self.max_displacement > 0:
            idx = np.flatnonzero(valid)
            if idx.size > 0:
                disp = src_pts[idx] - trg_pts[idx]
                keep = np.linalg.norm(disp, axis=1) <= self.max_displacement
                valid[idx] &= keep
        valid_idx = np.flatnonzero(valid)
        if self.max_points_per_pair is not None and self.max_points_per_pair > 0:
            valid_idx = self._sample_valid_indices(valid_idx, self.max_points_per_pair, pair_seed)

        if valid_idx.size == 0:
            return (
                torch.zeros((2, 0), dtype=torch.float32),
                torch.zeros((2, 0), dtype=torch.float32),
                0,
            )

        src_kps = torch.from_numpy(src_pts[valid_idx].T.copy()).to(torch.float32)
        trg_kps = torch.from_numpy(trg_pts[valid_idx].T.copy()).to(torch.float32)
        return src_kps, trg_kps, int(valid_idx.size)

    def _extract_pointodyssey_keypoints(
        self,
        annotations,
        src_idx: int,
        trg_idx: int,
        image_hw: Optional[Tuple[int, int]],
        pair_seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        trajs_2d = annotations["trajs_2d"]
        valids = annotations["valids"]

        src_pts = np.asarray(trajs_2d[src_idx], dtype=np.float32)
        trg_pts = np.asarray(trajs_2d[trg_idx], dtype=np.float32)
        src_valid = np.asarray(valids[src_idx], dtype=np.float32) > 0
        trg_valid = np.asarray(valids[trg_idx], dtype=np.float32) > 0

        valid = src_valid & trg_valid
        valid &= np.isfinite(src_pts).all(axis=1)
        valid &= np.isfinite(trg_pts).all(axis=1)
        valid &= ~((src_pts[:, 0] == 0.0) & (src_pts[:, 1] == 0.0))
        valid &= ~((trg_pts[:, 0] == 0.0) & (trg_pts[:, 1] == 0.0))

        if image_hw is not None:
            h, w = int(image_hw[0]), int(image_hw[1])
            valid &= (src_pts[:, 0] >= 0.0) & (src_pts[:, 0] < float(w))
            valid &= (src_pts[:, 1] >= 0.0) & (src_pts[:, 1] < float(h))
            valid &= (trg_pts[:, 0] >= 0.0) & (trg_pts[:, 0] < float(w))
            valid &= (trg_pts[:, 1] >= 0.0) & (trg_pts[:, 1] < float(h))

        if (not self.trust_manifest) and ("visibs" in annotations):
            visibs = annotations["visibs"]
            src_vis = np.asarray(visibs[src_idx], dtype=np.float32)
            trg_vis = np.asarray(visibs[trg_idx], dtype=np.float32)
            valid &= (src_vis > 0) & (trg_vis > 0)

        if self.max_displacement is not None and self.max_displacement > 0:
            idx = np.flatnonzero(valid)
            if idx.size > 0:
                disp = src_pts[idx] - trg_pts[idx]
                keep = np.linalg.norm(disp, axis=1) <= self.max_displacement
                valid[idx] &= keep

        valid_idx = np.flatnonzero(valid)
        if self.max_points_per_pair is not None and self.max_points_per_pair > 0:
            valid_idx = self._sample_valid_indices(valid_idx, self.max_points_per_pair, pair_seed)

        if valid_idx.size == 0:
            return (
                torch.zeros((2, 0), dtype=torch.float32),
                torch.zeros((2, 0), dtype=torch.float32),
                0,
            )

        src_kps = torch.from_numpy(src_pts[valid_idx].T.copy()).to(torch.float32)
        trg_kps = torch.from_numpy(trg_pts[valid_idx].T.copy()).to(torch.float32)
        return src_kps, trg_kps, int(valid_idx.size)

    def _get_pointodyssey_item(self, entry: Dict, manifest_idx: int) -> Dict:
        frame_i = int(entry["frame_i"])
        frame_j = int(entry["frame_j"])
        pair_id = int(entry.get("pair_id", manifest_idx))
        if self.reverse_flow:
            src_idx, trg_idx = frame_j, frame_i
        else:
            src_idx, trg_idx = frame_i, frame_j

        seq_path = self._resolve_seq_path(entry)
        anno_path = self._resolve_anno_path(entry, seq_path)
        src_frame_path = seq_path / "rgbs" / f"rgb_{src_idx:05d}.jpg"
        trg_frame_path = seq_path / "rgbs" / f"rgb_{trg_idx:05d}.jpg"

        t_img0 = time.perf_counter() if self.profile else 0.0
        src_img = self._read_rgb(src_frame_path)
        trg_img = self._read_rgb(trg_frame_path)
        t_img = time.perf_counter() - t_img0 if self.profile else 0.0

        t_anno0 = time.perf_counter() if self.profile else 0.0
        annotations = self._load_annotations(anno_path)
        t_anno = time.perf_counter() - t_anno0 if self.profile else 0.0

        t_kps0 = time.perf_counter() if self.profile else 0.0
        src_kps, trg_kps, n_pts = self._extract_pointodyssey_keypoints(
            annotations,
            src_idx,
            trg_idx,
            image_hw=(src_img.shape[1], src_img.shape[2]),
            pair_seed=self.seed + pair_id,
        )
        t_kps = time.perf_counter() - t_kps0 if self.profile else 0.0

        if self.profile:
            self._profile_samples += 1
            self._profile_total_s += (t_img + t_anno + t_kps)
            self._profile_img_s += t_img
            self._profile_anno_s += t_anno
            self._profile_kps_s += t_kps

        return {
            "src_img": src_img,
            "trg_img": trg_img,
            "src_kps": src_kps,
            "trg_kps": trg_kps,
            "n_pts": torch.tensor(n_pts, dtype=torch.int32),
            "meta": {
                "source_dataset": str(entry.get("source_dataset", "pointodyssey")),
                "source_split": str(entry.get("source_split", "train")),
                "source_sample_id": int(entry.get("source_sample_id", pair_id)),
                "manifest_idx": int(manifest_idx),
                "pair_id": pair_id,
            },
        }

    @staticmethod
    def _resolve_direct_path(entry: Dict, abs_key: str, rel_key: str, root_key: str) -> Path:
        if entry.get(abs_key):
            return Path(str(entry[abs_key]))
        root = entry.get(root_key)
        rel = entry.get(rel_key)
        if root is not None and rel is not None:
            return Path(str(root)) / str(rel)
        raise KeyError(f"Manifest entry missing path keys: {abs_key}, {rel_key}, {root_key}")

    def _get_sparse_item(self, entry: Dict, manifest_idx: int) -> Dict:
        src_img_path = self._resolve_direct_path(entry, "src_img_path", "src_img_rel_path", "source_root")
        trg_img_path = self._resolve_direct_path(entry, "trg_img_path", "trg_img_rel_path", "source_root")

        t_img0 = time.perf_counter() if self.profile else 0.0
        src_img = self._read_rgb(src_img_path)
        trg_img = self._read_rgb(trg_img_path)
        t_img = time.perf_counter() - t_img0 if self.profile else 0.0

        src_pts = np.asarray(entry["src_kps"], dtype=np.float32)
        trg_pts = np.asarray(entry["trg_kps"], dtype=np.float32)
        if src_pts.ndim != 2 or src_pts.shape[1] != 2 or trg_pts.shape != src_pts.shape:
            raise ValueError(
                f"Expected src_kps/trg_kps arrays with shape (N,2), got {src_pts.shape} and {trg_pts.shape}"
            )

        t_kps0 = time.perf_counter() if self.profile else 0.0
        src_kps, trg_kps, n_pts = self._limit_sparse_pair(
            src_pts=src_pts,
            trg_pts=trg_pts,
            pair_seed=self.seed + int(entry.get("pair_id", manifest_idx)),
        )
        t_kps = time.perf_counter() - t_kps0 if self.profile else 0.0

        if self.profile:
            self._profile_samples += 1
            self._profile_total_s += (t_img + t_kps)
            self._profile_img_s += t_img
            self._profile_kps_s += t_kps

        return {
            "src_img": src_img,
            "trg_img": trg_img,
            "src_kps": src_kps,
            "trg_kps": trg_kps,
            "n_pts": torch.tensor(n_pts, dtype=torch.int32),
            "meta": {
                "source_dataset": str(entry["source_dataset"]),
                "source_split": str(entry.get("source_split", "train")),
                "source_sample_id": int(entry.get("source_sample_id", manifest_idx)),
                "manifest_idx": int(manifest_idx),
                "pair_id": int(entry.get("pair_id", manifest_idx)),
            },
        }

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset length {len(self)}")
        manifest_idx = self.active_indices[self._window_start + idx]
        entry = self.entries[manifest_idx]
        source_dataset = str(entry.get("source_dataset", "pointodyssey")).lower()

        if source_dataset == "pointodyssey":
            return self._get_pointodyssey_item(entry, manifest_idx)

        if source_dataset in {"spair", "pfpascal", "pfwillow"}:
            return self._get_sparse_item(entry, manifest_idx)

        raise ValueError(f"Unsupported source_dataset in pooled manifest: {source_dataset}")
