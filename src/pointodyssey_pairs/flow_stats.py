from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def _to_flow_chw(flow: torch.Tensor) -> torch.Tensor:
    if flow.ndim != 3:
        raise ValueError(f"Expected 3D flow tensor, got shape={tuple(flow.shape)}")
    if flow.shape[0] == 2:
        return flow
    if flow.shape[-1] == 2:
        return flow.permute(2, 0, 1)
    raise ValueError(f"Expected flow in (2,H,W) or (H,W,2), got shape={tuple(flow.shape)}")


def compute_scalar_stats(
    flow: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute scalar magnitude statistics from a single flow sample.
    """
    flow = _to_flow_chw(flow).to(torch.float32)
    finite_mask = torch.isfinite(flow).all(dim=0)
    if valid_mask is not None:
        finite_mask = finite_mask & valid_mask.bool()

    total_px = int(finite_mask.numel())
    n_valid = int(finite_mask.sum().item())
    valid_fraction = float(n_valid / max(1, total_px))

    if n_valid == 0:
        return {
            "mean_mag": float("nan"),
            "median_mag": float("nan"),
            "p90_mag": float("nan"),
            "p95_mag": float("nan"),
            "valid_fraction": valid_fraction,
            "n_valid": float(n_valid),
        }

    dx = flow[0][finite_mask]
    dy = flow[1][finite_mask]
    mag = torch.sqrt(dx * dx + dy * dy)

    return {
        "mean_mag": float(mag.mean().item()),
        "median_mag": float(torch.quantile(mag, torch.tensor(0.5, dtype=mag.dtype)).item()),
        "p90_mag": float(torch.quantile(mag, torch.tensor(0.9, dtype=mag.dtype)).item()),
        "p95_mag": float(torch.quantile(mag, torch.tensor(0.95, dtype=mag.dtype)).item()),
        "valid_fraction": valid_fraction,
        "n_valid": float(n_valid),
    }


def infer_mag_clip_from_magnitudes(
    magnitudes: np.ndarray,
    quantile: float = 0.99,
    min_clip: float = 1e-6,
) -> float:
    if magnitudes.size == 0:
        return 1.0
    quantile = float(np.clip(quantile, 0.0, 1.0))
    clip = float(np.quantile(magnitudes, quantile))
    return float(max(min_clip, clip))
