from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from .flow_stats import _to_flow_chw


@dataclass(frozen=True)
class BFVConfig:
    angle_bins: int = 8
    mag_bins: int = 4
    mag_clip: float = 32.0
    l1_normalize: bool = True

    @property
    def dim(self) -> int:
        return int(self.angle_bins * self.mag_bins)


def _hist_from_dxdy(dx: np.ndarray, dy: np.ndarray, cfg: BFVConfig) -> np.ndarray:
    if cfg.angle_bins <= 0 or cfg.mag_bins <= 0:
        raise ValueError("angle_bins and mag_bins must be > 0")
    if cfg.mag_clip <= 0:
        raise ValueError("mag_clip must be > 0")

    if dx.size == 0:
        return np.zeros(cfg.dim, dtype=np.float32)

    mag = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)  # [-pi, pi]
    angle = (angle + (2.0 * np.pi)) % (2.0 * np.pi)  # [0, 2pi)

    angle_bin = np.floor((angle / (2.0 * np.pi)) * cfg.angle_bins).astype(np.int64)
    angle_bin = np.clip(angle_bin, 0, cfg.angle_bins - 1)

    mag_clip = np.minimum(mag, cfg.mag_clip)
    mag_bin = np.floor((mag_clip / cfg.mag_clip) * cfg.mag_bins).astype(np.int64)
    mag_bin = np.clip(mag_bin, 0, cfg.mag_bins - 1)

    flat_bin = mag_bin * cfg.angle_bins + angle_bin
    hist = np.bincount(flat_bin, minlength=cfg.dim).astype(np.float32)
    if cfg.l1_normalize and hist.sum() > 0:
        hist /= hist.sum()
    return hist


def flow_to_bfv(
    flow: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    cfg: BFVConfig,
) -> np.ndarray:
    flow = _to_flow_chw(flow).to(torch.float32)
    finite_mask = torch.isfinite(flow).all(dim=0)
    if valid_mask is not None:
        finite_mask = finite_mask & valid_mask.bool()
    if finite_mask.sum().item() == 0:
        return np.zeros(cfg.dim, dtype=np.float32)

    dx = flow[0][finite_mask].detach().cpu().numpy().astype(np.float32, copy=False)
    dy = flow[1][finite_mask].detach().cpu().numpy().astype(np.float32, copy=False)
    return _hist_from_dxdy(dx, dy, cfg)


def vectors_to_bfv(vectors: np.ndarray, cfg: BFVConfig) -> np.ndarray:
    vectors = np.asarray(vectors)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vectors array, got shape={vectors.shape}")
    if vectors.shape[1] >= 4:
        dx = vectors[:, 2].astype(np.float32, copy=False)
        dy = vectors[:, 3].astype(np.float32, copy=False)
    elif vectors.shape[1] == 2:
        dx = vectors[:, 0].astype(np.float32, copy=False)
        dy = vectors[:, 1].astype(np.float32, copy=False)
    else:
        raise ValueError(f"Expected vectors with 2 or >=4 channels, got shape={vectors.shape}")
    finite = np.isfinite(dx) & np.isfinite(dy)
    if not np.any(finite):
        return np.zeros(cfg.dim, dtype=np.float32)
    return _hist_from_dxdy(dx[finite], dy[finite], cfg)


def bfv_batch_from_flows(
    flows: Sequence[torch.Tensor],
    valid_masks: Sequence[Optional[torch.Tensor]],
    cfg: BFVConfig,
) -> np.ndarray:
    if len(flows) != len(valid_masks):
        raise ValueError("flows and valid_masks lengths must match")
    out = [flow_to_bfv(flow, mask, cfg) for flow, mask in zip(flows, valid_masks)]
    if not out:
        return np.zeros((0, cfg.dim), dtype=np.float32)
    return np.stack(out, axis=0).astype(np.float32, copy=False)
