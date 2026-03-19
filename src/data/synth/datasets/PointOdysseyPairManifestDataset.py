"""
PointOdyssey pair-manifest dataset for deterministic correspondence training.

This dataset reads a precomputed manifest of frame pairs and avoids runtime
retry/search logic. It supports full, random, and heuristic subset modes via
precomputed index files.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch


class PointOdysseyPairManifestDataset(torch.utils.data.Dataset):
    """Manifest-backed PointOdyssey correspondence dataset."""

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

        # Worker-local single-file cache.
        self._cached_annotation_path: Optional[Path] = None
        self._cached_annotations = None
        self._window_start: int = 0
        self._window_length: Optional[int] = None
        self._profile_samples = 0
        self._profile_total_s = 0.0
        self._profile_img_s = 0.0
        self._profile_anno_s = 0.0
        self._profile_kps_s = 0.0
        self._profile_cache_hits = 0
        self._profile_kps_valid_s = 0.0
        self._profile_kps_sample_s = 0.0
        self._profile_kps_gather_s = 0.0
        self._profile_kps_filter_s = 0.0
        self._profile_kps_fallback_s = 0.0
        self._profile_kps_tensor_s = 0.0

        if self.verbose:
            print(
                f"[PointOdysseyPairs] Loaded {len(self.active_indices)} / {len(self.entries)} "
                f"pairs (mode={self.subset_mode})"
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
        filtered = [idx for idx in indices if 0 <= idx < len(self.entries)]
        if self.verbose and len(filtered) != len(indices):
            print(
                f"[PointOdysseyPairs] Filtered {len(indices) - len(filtered)} out-of-range subset indices"
            )
        return filtered

    def _resolve_active_indices(self) -> List[int]:
        if self.subset_mode == "full":
            return list(range(len(self.entries)))

        subset_path = self.subset_indices_path
        if subset_path is None:
            subset_path = self._default_subset_path()

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
        """Restrict dataset to a contiguous window for this epoch."""
        if length is not None and length < 0:
            raise ValueError(f"length must be >= 0, got {length}")
        if start_idx < 0:
            raise ValueError(f"start_idx must be >= 0, got {start_idx}")
        self._window_start = int(start_idx)
        self._window_length = None if length is None else int(length)
        if self._window_start > len(self.active_indices):
            self._window_start = len(self.active_indices)
        if self.verbose:
            active_len = len(self.active_indices)
            end_idx = (
                min(self._window_start + (self._window_length or active_len), active_len)
                if self._window_length is not None
                else active_len
            )
            print(
                f"[PointOdysseyPairs] Epoch window set: start={self._window_start}, "
                f"length={self._window_length}, end={end_idx}, active_len={active_len}"
            )

    def _resolve_seq_path(self, entry: Dict) -> Path:
        if self.pointodyssey_root is not None and entry.get("seq_rel_path"):
            return self.pointodyssey_root / entry["seq_rel_path"]
        if entry.get("seq_path"):
            return Path(entry["seq_path"])
        raise KeyError("Manifest entry missing both seq_rel_path and seq_path")

    def _resolve_anno_path(self, entry: Dict, seq_path: Path) -> Path:
        if self.pointodyssey_root is not None and entry.get("anno_rel_path"):
            return self.pointodyssey_root / entry["anno_rel_path"]
        if entry.get("anno_path"):
            return Path(entry["anno_path"])
        return seq_path / "anno.npz"

    @staticmethod
    def _read_rgb(frame_path: Path) -> torch.Tensor:
        img_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {frame_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img_rgb).permute(2, 0, 1).to(torch.float32) / 255.0

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
            try:
                annotations = np.load(anno_path, allow_pickle=True, mmap_mode="r")
            except (OSError, ValueError, TypeError):
                annotations = np.load(anno_path, allow_pickle=True)
        self._cached_annotation_path = anno_path
        self._cached_annotations = annotations
        return annotations

    def _extract_keypoints(
        self,
        annotations,
        src_idx: int,
        trg_idx: int,
        image_hw: Optional[tuple] = None,
        trust_manifest: bool = False,
        pair_seed: Optional[int] = None,
    ):
        profile_kps = self.profile
        kps_timing = {
            "valid_s": 0.0,
            "sample_s": 0.0,
            "gather_s": 0.0,
            "filter_s": 0.0,
            "fallback_s": 0.0,
            "tensor_s": 0.0,
        }
        trajs_2d = annotations["trajs_2d"]
        valids = annotations["valids"]
        h = w = None
        if image_hw is not None:
            h = int(image_hw[0])
            w = int(image_hw[1])

        def _invalid_zero_xy(points: np.ndarray) -> np.ndarray:
            return np.logical_and(points[:, 0] == 0.0, points[:, 1] == 0.0)

        def _build_valid_mask(
            src_pts_in: np.ndarray,
            trg_pts_in: np.ndarray,
            src_valid_idx: np.ndarray,
            trg_valid_idx: np.ndarray,
            src_vis_idx: Optional[np.ndarray] = None,
            trg_vis_idx: Optional[np.ndarray] = None,
            apply_displacement: bool = True,
        ) -> np.ndarray:
            valid_mask = src_valid_idx & trg_valid_idx
            valid_mask &= np.isfinite(src_pts_in).all(axis=1)
            valid_mask &= np.isfinite(trg_pts_in).all(axis=1)
            valid_mask &= ~_invalid_zero_xy(src_pts_in)
            valid_mask &= ~_invalid_zero_xy(trg_pts_in)
            if h is not None and w is not None:
                valid_mask &= (src_pts_in[:, 0] >= 0.0) & (src_pts_in[:, 0] < float(w))
                valid_mask &= (src_pts_in[:, 1] >= 0.0) & (src_pts_in[:, 1] < float(h))
                valid_mask &= (trg_pts_in[:, 0] >= 0.0) & (trg_pts_in[:, 0] < float(w))
                valid_mask &= (trg_pts_in[:, 1] >= 0.0) & (trg_pts_in[:, 1] < float(h))
            if src_vis_idx is not None and trg_vis_idx is not None:
                valid_mask &= src_vis_idx > 0
                valid_mask &= trg_vis_idx > 0
            if apply_displacement and self.max_displacement is not None and self.max_displacement > 0:
                valid_idx = np.flatnonzero(valid_mask)
                if valid_idx.size > 0:
                    disp = src_pts_in[valid_idx] - trg_pts_in[valid_idx]
                    disp_mag = np.linalg.norm(disp, axis=1)
                    valid_mask[valid_idx] &= disp_mag <= self.max_displacement
            return valid_mask

        # Fast smoke path: trust manifest validity and cap before heavy trajectory gather.
        if trust_manifest and self.max_points_per_pair is not None and self.max_points_per_pair > 0:
            n_tracks = int(trajs_2d.shape[1])
            if n_tracks <= 0:
                src_kps = torch.zeros((2, 0), dtype=torch.float32)
                trg_kps = torch.zeros((2, 0), dtype=torch.float32)
                return src_kps, trg_kps, 0, kps_timing
            target_n = min(int(self.max_points_per_pair), n_tracks)
            oversample_n = min(n_tracks, max(target_n * 4, target_n))

            t0 = time.perf_counter() if profile_kps else 0.0
            if self.random_subsample_within_pair:
                rng = np.random.default_rng(int(pair_seed) if pair_seed is not None else self.seed)
                candidate_idx = rng.choice(n_tracks, size=oversample_n, replace=False)
            else:
                candidate_idx = np.arange(oversample_n, dtype=np.int64)
            if profile_kps:
                kps_timing["sample_s"] += time.perf_counter() - t0

            t0 = time.perf_counter() if profile_kps else 0.0
            src_pts_c = np.asarray(trajs_2d[src_idx, candidate_idx], dtype=np.float32)
            trg_pts_c = np.asarray(trajs_2d[trg_idx, candidate_idx], dtype=np.float32)
            src_valid_c = np.asarray(valids[src_idx, candidate_idx], dtype=np.float32) > 0
            trg_valid_c = np.asarray(valids[trg_idx, candidate_idx], dtype=np.float32) > 0
            if profile_kps:
                kps_timing["gather_s"] += time.perf_counter() - t0

            t0 = time.perf_counter() if profile_kps else 0.0
            valid_mask_c = _build_valid_mask(src_pts_c, trg_pts_c, src_valid_c, trg_valid_c)
            if profile_kps:
                kps_timing["filter_s"] += time.perf_counter() - t0

            src_pts = src_pts_c[valid_mask_c]
            trg_pts = trg_pts_c[valid_mask_c]

            # Rare fallback: if sampled subset is too sparse, recover with full scan.
            min_needed = min(8, target_n)
            if src_pts.shape[0] < min_needed and oversample_n < n_tracks:
                t0 = time.perf_counter() if profile_kps else 0.0
                src_trajs = np.asarray(trajs_2d[src_idx], dtype=np.float32)
                trg_trajs = np.asarray(trajs_2d[trg_idx], dtype=np.float32)
                src_valid = np.asarray(valids[src_idx], dtype=np.float32) > 0
                trg_valid = np.asarray(valids[trg_idx], dtype=np.float32) > 0
                valid_mask = src_valid & trg_valid
                valid_mask = _build_valid_mask(src_trajs, trg_trajs, src_valid, trg_valid)
                valid_idx = np.flatnonzero(valid_mask)
                if valid_idx.size == 0:
                    relaxed_mask = _build_valid_mask(
                        src_trajs,
                        trg_trajs,
                        src_valid,
                        trg_valid,
                        apply_displacement=False,
                    )
                    valid_idx = np.flatnonzero(relaxed_mask)
                if valid_idx.size == 0:
                    src_kps = torch.zeros((2, 0), dtype=torch.float32)
                    trg_kps = torch.zeros((2, 0), dtype=torch.float32)
                    if profile_kps:
                        kps_timing["fallback_s"] += time.perf_counter() - t0
                    return src_kps, trg_kps, 0, kps_timing
                if valid_idx.size > target_n:
                    if self.random_subsample_within_pair:
                        rng = np.random.default_rng(int(pair_seed) if pair_seed is not None else self.seed)
                        chosen = rng.choice(valid_idx.size, size=target_n, replace=False)
                        valid_idx = valid_idx[np.sort(chosen)]
                    else:
                        valid_idx = valid_idx[:target_n]
                src_pts = src_trajs[valid_idx]
                trg_pts = trg_trajs[valid_idx]
                if profile_kps:
                    kps_timing["fallback_s"] += time.perf_counter() - t0
            elif src_pts.shape[0] > target_n:
                src_pts = src_pts[:target_n]
                trg_pts = trg_pts[:target_n]
        else:
            t0 = time.perf_counter() if profile_kps else 0.0
            src_trajs = np.asarray(trajs_2d[src_idx], dtype=np.float32)
            trg_trajs = np.asarray(trajs_2d[trg_idx], dtype=np.float32)
            src_valid = np.asarray(valids[src_idx], dtype=np.float32) > 0
            trg_valid = np.asarray(valids[trg_idx], dtype=np.float32) > 0
            valid_mask = src_valid & trg_valid
            if profile_kps:
                kps_timing["valid_s"] += time.perf_counter() - t0

            t0 = time.perf_counter() if profile_kps else 0.0
            src_vis = None
            trg_vis = None
            if not trust_manifest:
                has_visibs = (
                    ("visibs" in annotations) if isinstance(annotations, dict)
                    else ("visibs" in annotations.files)
                )
                if has_visibs:
                    visibs = annotations["visibs"]
                    src_vis = np.asarray(visibs[src_idx], dtype=np.float32)
                    trg_vis = np.asarray(visibs[trg_idx], dtype=np.float32)

            valid_mask = _build_valid_mask(
                src_trajs,
                trg_trajs,
                src_valid,
                trg_valid,
                src_vis_idx=src_vis,
                trg_vis_idx=trg_vis,
            )

            if (not trust_manifest) and (not np.any(valid_mask)):
                relaxed_mask = _build_valid_mask(
                    src_trajs,
                    trg_trajs,
                    src_valid,
                    trg_valid,
                    src_vis_idx=src_vis,
                    trg_vis_idx=trg_vis,
                    apply_displacement=False,
                )
                valid_mask = relaxed_mask if np.any(relaxed_mask) else valid_mask
            if profile_kps:
                kps_timing["filter_s"] += time.perf_counter() - t0

            if not np.any(valid_mask):
                src_kps = torch.zeros((2, 0), dtype=torch.float32)
                trg_kps = torch.zeros((2, 0), dtype=torch.float32)
                return src_kps, trg_kps, 0, kps_timing

            src_pts = src_trajs[valid_mask]
            trg_pts = trg_trajs[valid_mask]
            if self.max_points_per_pair is not None and self.max_points_per_pair > 0 and src_pts.shape[0] > self.max_points_per_pair:
                src_pts = src_pts[: self.max_points_per_pair]
                trg_pts = trg_pts[: self.max_points_per_pair]

        t0 = time.perf_counter() if profile_kps else 0.0
        src_kps = torch.from_numpy(src_pts.T).to(torch.float32)
        trg_kps = torch.from_numpy(trg_pts.T).to(torch.float32)
        if profile_kps:
            kps_timing["tensor_s"] += time.perf_counter() - t0
        return src_kps, trg_kps, int(src_pts.shape[0]), kps_timing

    def __getitem__(self, idx: int):
        t_total0 = time.perf_counter() if self.profile else 0.0
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for windowed dataset of length {len(self)}")
        manifest_idx = self.active_indices[self._window_start + idx]
        entry = self.entries[manifest_idx]

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
        t_img = (time.perf_counter() - t_img0) if self.profile else 0.0

        cache_hit = bool(self._cached_annotation_path == anno_path and self._cached_annotations is not None)
        t_anno0 = time.perf_counter() if self.profile else 0.0
        annotations = self._load_annotations(anno_path)
        t_anno = (time.perf_counter() - t_anno0) if self.profile else 0.0
        t_kps0 = time.perf_counter() if self.profile else 0.0
        src_kps, trg_kps, n_pts, kps_timing = self._extract_keypoints(
            annotations,
            src_idx,
            trg_idx,
            image_hw=(src_img.shape[1], src_img.shape[2]),
            trust_manifest=self.trust_manifest,
            pair_seed=self.seed + pair_id,
        )
        t_kps = (time.perf_counter() - t_kps0) if self.profile else 0.0

        if self.profile:
            t_total = time.perf_counter() - t_total0
            self._profile_samples += 1
            self._profile_total_s += t_total
            self._profile_img_s += t_img
            self._profile_anno_s += t_anno
            self._profile_kps_s += t_kps
            self._profile_kps_valid_s += float(kps_timing.get("valid_s", 0.0))
            self._profile_kps_sample_s += float(kps_timing.get("sample_s", 0.0))
            self._profile_kps_gather_s += float(kps_timing.get("gather_s", 0.0))
            self._profile_kps_filter_s += float(kps_timing.get("filter_s", 0.0))
            self._profile_kps_fallback_s += float(kps_timing.get("fallback_s", 0.0))
            self._profile_kps_tensor_s += float(kps_timing.get("tensor_s", 0.0))
            if cache_hit:
                self._profile_cache_hits += 1
            if self._profile_samples % self.profile_every == 0:
                avg_total_ms = 1000.0 * self._profile_total_s / max(1, self._profile_samples)
                avg_img_ms = 1000.0 * self._profile_img_s / max(1, self._profile_samples)
                avg_anno_ms = 1000.0 * self._profile_anno_s / max(1, self._profile_samples)
                avg_kps_ms = 1000.0 * self._profile_kps_s / max(1, self._profile_samples)
                avg_valid_ms = 1000.0 * self._profile_kps_valid_s / max(1, self._profile_samples)
                avg_sample_ms = 1000.0 * self._profile_kps_sample_s / max(1, self._profile_samples)
                avg_gather_ms = 1000.0 * self._profile_kps_gather_s / max(1, self._profile_samples)
                avg_filter_ms = 1000.0 * self._profile_kps_filter_s / max(1, self._profile_samples)
                avg_fallback_ms = 1000.0 * self._profile_kps_fallback_s / max(1, self._profile_samples)
                avg_tensor_ms = 1000.0 * self._profile_kps_tensor_s / max(1, self._profile_samples)
                cache_hit_pct = 100.0 * float(self._profile_cache_hits) / max(1, self._profile_samples)
                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id if worker_info is not None else -1
                print(
                    f"[PointOdysseyPairsProfile][worker={worker_id}] "
                    f"samples={self._profile_samples} avg_total={avg_total_ms:.2f}ms "
                    f"(img={avg_img_ms:.2f}ms anno={avg_anno_ms:.2f}ms kps={avg_kps_ms:.2f}ms) "
                    f"kps_breakdown=(valid={avg_valid_ms:.2f}ms sample={avg_sample_ms:.2f}ms "
                    f"gather={avg_gather_ms:.2f}ms filter={avg_filter_ms:.2f}ms "
                    f"fallback={avg_fallback_ms:.2f}ms tensor={avg_tensor_ms:.2f}ms) "
                    f"cache_hit={cache_hit_pct:.1f}%",
                    flush=True,
                )

        return {
            "src_img": src_img,
            "trg_img": trg_img,
            "src_kps": src_kps,
            "trg_kps": trg_kps,
            "n_pts": torch.tensor(n_pts, dtype=torch.int32),
        }
