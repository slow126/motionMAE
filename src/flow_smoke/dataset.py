"""Lightweight Point Odyssey smoke dataset for flow prediction prototypes."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.synth.datasets.flow_utils import flow_from_kps


def load_manifest(manifest_path: Union[str, Path]) -> List[dict]:
    """Load a JSONL pair manifest into a list of dictionaries."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    entries: List[dict] = []
    with manifest_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    if not entries:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    return entries


def _as_tuple_size(size: Union[int, Sequence[int], None]) -> Optional[Tuple[int, int]]:
    if size is None:
        return None
    if isinstance(size, int):
        return (int(size), int(size))
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError(f"Expected int or (H, W), got {size!r}")


def split_manifest_indices_by_clip(
    entries: Sequence[dict],
    val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Split manifest rows by clip id for train/validation."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")

    clip_to_indices: Dict[str, List[int]] = {}
    for idx, entry in enumerate(entries):
        seq_id = entry.get("seq_id", None)
        seq_key = str(seq_id) if seq_id is not None else str(entry.get("seq_rel_path", entry.get("seq_path", idx)))
        clip_to_indices.setdefault(seq_key, []).append(idx)

    clip_keys = sorted(clip_to_indices.keys())
    if not clip_keys:
        return [], []

    rng = np.random.default_rng(seed)
    rng.shuffle(clip_keys)

    n_val_clips = int(round(len(clip_keys) * val_fraction)) if val_fraction > 0 else 0
    n_val_clips = max(0, min(n_val_clips, len(clip_keys)))
    if n_val_clips == 0 and len(clip_keys) > 1:
        n_val_clips = 1

    val_clip_keys = set(clip_keys[:n_val_clips])

    val_indices: List[int] = []
    train_indices: List[int] = []
    for clip_key, idxs in clip_to_indices.items():
        if clip_key in val_clip_keys:
            val_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    return train_indices, val_indices


class PointOdysseyFlowSmokeDataset(Dataset):
    """Dataset that emits ``(src_img, trg_img, flow, dt)`` for smoke testing."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        indices: Sequence[int],
        dt_values: Optional[Sequence[int]] = None,
        pointodyssey_root: Optional[Union[str, Path]] = None,
        reverse_flow: bool = False,
        size: Union[int, Sequence[int], None] = None,
        max_points_per_pair: Optional[int] = None,
        max_displacement: Optional[float] = None,
        use_grayscale: bool = False,
        trust_manifest: bool = False,
        min_valid_points: int = 0,
        normalize_flow: bool = True,
        cache_annotations: bool = True,
        seed: int = 2021,
    ):
        self.entries = load_manifest(manifest_path)
        self.pointodyssey_root = Path(pointodyssey_root) if pointodyssey_root is not None else None

        dt_filter = None if dt_values is None else {int(v) for v in dt_values}

        def _extract_manifest_valid_points(row: Dict) -> Optional[int]:
            for key in ("valid_points", "num_valid_kpts", "n_valid", "num_valid_pts", "num_kpts"):
                if key in row:
                    try:
                        return int(row[key])
                    except (TypeError, ValueError):
                        continue
            return None

        min_valid_points = int(min_valid_points) if min_valid_points is not None else 0
        min_valid_points = max(0, min_valid_points)

        selected = []
        for idx in indices:
            entry = self.entries[idx]
            row_dt = int(entry.get("dt", int(int(entry.get("frame_j", 0)) - int(entry.get("frame_i", 0)))))
            if dt_filter is not None and row_dt not in dt_filter:
                continue
            if min_valid_points > 0:
                row_valid = _extract_manifest_valid_points(entry)
                if row_valid is not None and row_valid < min_valid_points:
                    continue
            selected.append(idx)

        if not selected:
            raise RuntimeError("No manifest rows match requested dt/filter settings.")

        self.manifest_indices = selected
        self.reverse_flow = bool(reverse_flow)
        self.size = _as_tuple_size(size)
        self.max_points_per_pair = None if max_points_per_pair is None else int(max_points_per_pair)
        self.max_displacement = None if max_displacement is None else float(max_displacement)
        self.use_grayscale = bool(use_grayscale)
        self.trust_manifest = bool(trust_manifest)
        self.min_valid_points = int(min_valid_points)
        self.normalize_flow = bool(normalize_flow)
        self.seed = int(seed)
        self.cache_annotations = bool(cache_annotations)
        self._anno_cache: Dict[Path, dict] = {}
        self._rng = random.Random(self.seed)
        self.dt_values = sorted(set(int(v) for v in dt_filter)) if dt_filter is not None else []
        self._max_dt_for_norm = float(max(self.dt_values)) if self.dt_values else None

        if self.size is None:
            self.normalization_scale = 1.0
        else:
            self.normalization_scale = float(max(self.size))

    def __len__(self):
        return len(self.manifest_indices)

    @staticmethod
    def _read_rgb(path: Path) -> torch.Tensor:
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
        return img

    def _resolve_seq_path(self, entry: Dict) -> Path:
        if self.pointodyssey_root is not None and entry.get("seq_rel_path"):
            return self.pointodyssey_root / entry["seq_rel_path"]
        if entry.get("seq_path"):
            return Path(entry["seq_path"])
        raise KeyError("Manifest entry missing both seq_path and seq_rel_path")

    def _resolve_anno_path(self, entry: Dict, seq_path: Path) -> Path:
        if self.pointodyssey_root is not None and entry.get("anno_rel_path"):
            return self.pointodyssey_root / entry["anno_rel_path"]
        if entry.get("anno_path"):
            return Path(entry["anno_path"])
        return seq_path / "anno.npz"

    def _load_annotations(self, path: Path) -> dict:
        if self.cache_annotations and path in self._anno_cache:
            return self._anno_cache[path]

        with np.load(path, allow_pickle=True) as npz:
            ann = {
                "trajs_2d": np.asarray(npz["trajs_2d"]),
                "valids": np.asarray(npz["valids"]),
            }
            if "visibs" in npz.files:
                ann["visibs"] = np.asarray(npz["visibs"])

        if self.cache_annotations:
            self._anno_cache[path] = ann

        return ann

    @staticmethod
    def _normalize_mask(src_kps: np.ndarray, trg_kps: np.ndarray, frame_h: int, frame_w: int):
        valid = np.isfinite(src_kps).all(axis=1) & np.isfinite(trg_kps).all(axis=1)
        zero = (src_kps[:, 0] == 0.0) & (src_kps[:, 1] == 0.0)
        zero |= (trg_kps[:, 0] == 0.0) & (trg_kps[:, 1] == 0.0)
        valid &= ~zero
        valid &= (src_kps[:, 0] >= 0.0) & (src_kps[:, 0] < float(frame_w))
        valid &= (src_kps[:, 1] >= 0.0) & (src_kps[:, 1] < float(frame_h))
        valid &= (trg_kps[:, 0] >= 0.0) & (trg_kps[:, 0] < float(frame_w))
        valid &= (trg_kps[:, 1] >= 0.0) & (trg_kps[:, 1] < float(frame_h))
        return valid

    def _sample_kpts(self, src_kps: np.ndarray, trg_kps: np.ndarray, target: int, manifest_idx: int):
        n = src_kps.shape[0]
        if target is None or n <= target:
            return src_kps, trg_kps
        rng = np.random.default_rng(self.seed + int(manifest_idx) * 73856093 + n * 19349663)
        keep = rng.choice(n, size=target, replace=False)
        return src_kps[keep], trg_kps[keep]

    def __getitem__(self, index):
        manifest_idx = self.manifest_indices[index]
        entry = self.entries[manifest_idx]

        frame_i = int(entry["frame_i"])
        frame_j = int(entry["frame_j"])
        if self.reverse_flow:
            src_frame, trg_frame = frame_j, frame_i
        else:
            src_frame, trg_frame = frame_i, frame_j

        seq_path = self._resolve_seq_path(entry)
        anno_path = self._resolve_anno_path(entry, seq_path)

        src_img = self._read_rgb(seq_path / "rgbs" / f"rgb_{src_frame:05d}.jpg")
        trg_img = self._read_rgb(seq_path / "rgbs" / f"rgb_{trg_frame:05d}.jpg")

        source_h, source_w = src_img.shape[1], src_img.shape[2]
        original_h, original_w = source_h, source_w
        target_size = self.size
        if target_size is not None:
            target_h, target_w = target_size
            if (source_h, source_w) != (target_h, target_w):
                scale_x = float(target_w) / float(source_w)
                scale_y = float(target_h) / float(source_h)
                src_img = torch.nn.functional.interpolate(
                    src_img.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
                ).squeeze(0)
                trg_img = torch.nn.functional.interpolate(
                    trg_img.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
                ).squeeze(0)
            else:
                scale_x = scale_y = 1.0
            source_h, source_w = target_size
        else:
            scale_x = scale_y = 1.0

        if self.use_grayscale:
            src_img = src_img.mean(dim=0, keepdim=True)
            trg_img = trg_img.mean(dim=0, keepdim=True)

        ann = self._load_annotations(anno_path)
        trajs = ann["trajs_2d"]
        valids = ann["valids"]
        visibs = ann.get("visibs", None)

        src_kps_np = np.asarray(trajs[src_frame], dtype=np.float32)
        trg_kps_np = np.asarray(trajs[trg_frame], dtype=np.float32)

        src_valid = np.asarray(valids[src_frame], dtype=np.float32) > 0
        trg_valid = np.asarray(valids[trg_frame], dtype=np.float32) > 0
        valid = src_valid & trg_valid

        valid = self._normalize_mask(src_kps_np, trg_kps_np, original_h, original_w) & valid

        if visibs is not None and not self.trust_manifest:
            src_vis = np.asarray(visibs[src_frame], dtype=np.float32)
            trg_vis = np.asarray(visibs[trg_frame], dtype=np.float32)
            valid &= (src_vis > 0) & (trg_vis > 0)

        if self.max_displacement is not None and self.max_displacement > 0:
            idx = np.flatnonzero(valid)
            if idx.size > 0:
                disp = src_kps_np[idx] - trg_kps_np[idx]
                keep = np.linalg.norm(disp, axis=1) <= self.max_displacement
                valid[np.flatnonzero(valid)] &= keep

        src_kps_np = src_kps_np[valid]
        trg_kps_np = trg_kps_np[valid]

        if self.max_points_per_pair is not None and self.max_points_per_pair > 0:
            src_kps_np, trg_kps_np = self._sample_kpts(
                src_kps_np,
                trg_kps_np,
                int(self.max_points_per_pair),
                manifest_idx,
            )

        if src_kps_np.size == 0:
            flow = torch.full((2, source_h, source_w), float("inf"), dtype=torch.float32)
            n_pts = 0
        else:
            src_kps_np[:, 0] *= scale_x
            src_kps_np[:, 1] *= scale_y
            trg_kps_np[:, 0] *= scale_x
            trg_kps_np[:, 1] *= scale_y

            src_kps = torch.from_numpy(src_kps_np.T)
            trg_kps = torch.from_numpy(trg_kps_np.T)
            flow = flow_from_kps(src_kps, trg_kps, (source_h, source_w))
            n_pts = int(src_kps.shape[1])

        if self.normalize_flow and self.normalization_scale > 0:
            flow = flow / self.normalization_scale

        dt = int(entry.get("dt", trg_frame - src_frame))
        dt_norm = float(dt) / float(self._max_dt_for_norm) if self._max_dt_for_norm else 1.0
        dt_norm = torch.tensor(float(dt_norm), dtype=torch.float32)
        valid_flow_mask = torch.isfinite(flow).all(dim=0)

        return {
            "src_img": src_img,
            "trg_img": trg_img,
            "flow": flow,
            "valid_flow_mask": valid_flow_mask,
            "dt": dt_norm,
            "dt_raw": torch.tensor(float(dt), dtype=torch.float32),
            "flow_scale": torch.tensor(float(self.normalization_scale), dtype=torch.float32),
            "n_pts": torch.tensor(int(n_pts), dtype=torch.int32),
            "manifest_idx": torch.tensor(int(manifest_idx), dtype=torch.int64),
            "frame_i": torch.tensor(int(src_frame), dtype=torch.int32),
            "frame_j": torch.tensor(int(trg_frame), dtype=torch.int32),
        }
