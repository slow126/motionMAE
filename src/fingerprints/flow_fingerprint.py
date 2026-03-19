"""
flow_fingerprint.py
===================
Compact "flow fingerprints" with BOTH:
  - spatial_motion_prob (binary: P[m > τ])
  - spatial_mean_mag (mean magnitude)

Produces a JSON with fixed bin edges + histograms + spatial maps.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
import numpy as np
import json

# ------------------------------
# Config
# ------------------------------

@dataclass
class FlowFingerprintConfig:
    # Magnitude (log-spaced) histogram
    mag_min: float = 1e-3
    mag_max: float = 256.0
    mag_bins: int = 40

    # Angle histogram (in radians)
    ang_bins: int = 36
    ang_weight_clip: float = 5.0

    # Joint histogram (magnitude x angle)
    joint_mag_bins: int = 24
    joint_ang_bins: int = 36

    # Temporal delta histogram
    delta_min: float = 1e-3
    delta_max: float = 256.0
    delta_bins: int = 30

    # Divergence / curl histograms
    div_bins: int = 30
    div_min: float = -2.0
    div_max: float =  2.0
    curl_bins: int = 30
    curl_min: float = -2.0
    curl_max: float =  2.0

    # Spatial maps
    spatial_downsample_hw: Tuple[int, int] = (32, 32)
    motion_thresh: float = 0.1  # px

    # General
    accumulate_density: bool = True
    keep_counts: bool = False


# ------------------------------
# Helpers
# ------------------------------

def _logspace_edges(vmin: float, vmax: float, bins: int) -> np.ndarray:
    # Convert to float in case YAML loaded as string
    vmin = float(vmin)
    vmax = float(vmax)
    assert vmin > 0, f"vmin must be > 0, got {vmin} (type: {type(vmin)})"
    return np.exp(np.linspace(np.log(vmin), np.log(vmax), bins + 1))

def _linspace_edges(vmin: float, vmax: float, bins: int) -> np.ndarray:
    # Convert to float in case YAML loaded as string
    vmin = float(vmin)
    vmax = float(vmax)
    return np.linspace(vmin, vmax, bins + 1)

def _safe_angles(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.arctan2(v, u)

def _magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.hypot(u, v)

def _percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))

def _safe_sample(x: np.ndarray, max_keep: int = 200_000) -> np.ndarray:
    n = x.size
    if n <= max_keep:
        return x.copy()
    idx = np.random.choice(n, size=max_keep, replace=False)
    return x[idx]

def _density_to_prob(H: np.ndarray) -> np.ndarray:
    s = H.sum()
    return H / s if s > 0 else H

def _to_prob2d(H2: np.ndarray) -> np.ndarray:
    s = H2.sum()
    return (H2 / s) if s > 0 else H2

def _downsample_mean(x: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Area-average downsample to (H_out, W_out) for 2D arrays."""
    H, W = x.shape
    H_out, W_out = out_hw
    h_blk = H / H_out
    w_blk = W / W_out
    ys = (np.arange(H_out) + 0.5) * h_blk
    xs = (np.arange(W_out) + 0.5) * w_blk
    out = np.zeros((H_out, W_out), dtype=np.float64)
    for i, y in enumerate(ys):
        y0 = int(max(0, np.floor(y - 0.5 * h_blk))); y1 = int(min(H, np.ceil(y + 0.5 * h_blk)))
        for j, xj in enumerate(xs):
            x0 = int(max(0, np.floor(xj - 0.5 * w_blk))); x1 = int(min(W, np.ceil(xj + 0.5 * w_blk)))
            blk = x[y0:y1, x0:x1]
            out[i, j] = blk.mean() if blk.size else 0.0
    return out


# ------------------------------
# Accumulator
# ------------------------------

class FlowFingerprint:
    """
    add_frame(flow, prev_flow, valid_mask)
    finalize() -> stats dict

    flow[...,0]=u (+x right), flow[...,1]=v (+y down) in pixels.
    
    Invalid Flow Handling:
    - Automatically excludes inf/nan values (occlusion-based invalid pixels)
    - Zero flow (0, 0) is treated as valid (correspondence pixels that don't move)
    - If valid_mask is provided, it is combined with isfinite() check
    - If valid_mask is None, automatically creates mask from isfinite() check
    """

    def __init__(self, cfg: FlowFingerprintConfig):
        self.cfg = cfg

        # Bin edges
        self.mag_edges   = _logspace_edges(cfg.mag_min,   cfg.mag_max,   cfg.mag_bins)
        self.ang_edges   = _linspace_edges(-np.pi, np.pi, cfg.ang_bins)
        self.jmag_edges  = _logspace_edges(cfg.mag_min,   cfg.mag_max,   cfg.joint_mag_bins)
        self.jang_edges  = _linspace_edges(-np.pi, np.pi, cfg.joint_ang_bins)
        self.delta_edges = _logspace_edges(cfg.delta_min, cfg.delta_max, cfg.delta_bins)
        self.div_edges   = _linspace_edges(cfg.div_min,   cfg.div_max,   cfg.div_bins)
        self.curl_edges  = _linspace_edges(cfg.curl_min,  cfg.curl_max,  cfg.curl_bins)

        # Hists
        self.mag_hist   = np.zeros(cfg.mag_bins,  dtype=np.float64)
        self.ang_hist   = np.zeros(cfg.ang_bins,  dtype=np.float64)
        self.joint_hist = np.zeros((cfg.joint_mag_bins, cfg.joint_ang_bins), dtype=np.float64)
        self.delta_hist = np.zeros(cfg.delta_bins, dtype=np.float64)
        self.div_hist   = np.zeros(cfg.div_bins,  dtype=np.float64)
        self.curl_hist  = np.zeros(cfg.curl_bins, dtype=np.float64)

        # Spatial accumulators
        self.frames_seen = 0
        self.motion_hits_sum = None      # sum of 1[m>τ] per-pixel over frames
        self.mean_mag_sum    = None      # sum of magnitude per-pixel over frames
        self.valid_sum       = None      # count of valid pixels per location over frames

        # Moments
        self._mag_samples   = []
        self._delta_samples = []
        self._moving_total  = 0
        self._valid_total   = 0

    def add_frame(
        self,
        flow: np.ndarray,
        prev_flow: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
    ):
        assert flow.ndim == 3 and flow.shape[-1] == 2, "flow must be [H,W,2]"
        H, W, _ = flow.shape
        
        u = flow[..., 0]; v = flow[..., 1]
        
        # Auto-detect invalid flow (inf/nan) if no valid_mask provided
        # Distinguishes between:
        # - "no flow" (zero values) = valid correspondence pixels that don't move
        # - "invalid flow" (inf/nan) = occlusion-based invalid pixels (should be excluded)
        if valid_mask is None:
            # Check for finite values in both u and v components
            # Invalid flow is marked as inf/nan (occlusion-based)
            # Zero flow is valid (correspondence pixels that don't move)
            M = np.isfinite(u) & np.isfinite(v)
        else:
            # Combine provided mask with finite check to ensure inf/nan are excluded
            M = (valid_mask > 0) & np.isfinite(u) & np.isfinite(v)
        
        mag = _magnitude(u, v)
        ang = _safe_angles(u, v)

        # Flatten valid
        M_flat   = M.reshape(-1)
        mag_flat = mag.reshape(-1)[M_flat]
        ang_flat = ang.reshape(-1)[M_flat]

        # ---- Hists ----
        H_mag,  _ = np.histogram(mag_flat, bins=self.mag_edges, density=self.cfg.accumulate_density)
        self.mag_hist += H_mag

        ang_w = np.minimum(mag_flat, self.cfg.ang_weight_clip)
        H_ang, _ = np.histogram(ang_flat, bins=self.ang_edges, weights=ang_w, density=self.cfg.accumulate_density)
        self.ang_hist += H_ang

        H2, _, _ = np.histogram2d(mag_flat, ang_flat, bins=[self.jmag_edges, self.jang_edges],
                                  density=self.cfg.accumulate_density)
        self.joint_hist += H2

        # ---- Spatial maps (binary + mean) ----
        if self.motion_hits_sum is None:
            self.motion_hits_sum = np.zeros((H, W), dtype=np.float64)
            self.mean_mag_sum    = np.zeros((H, W), dtype=np.float64)
            self.valid_sum       = np.zeros((H, W), dtype=np.float64)

        # Ensure mag is finite where M is True (should already be, but be explicit)
        mag_valid = np.where(M, mag, 0.0)  # Set invalid locations to 0 for accumulation
        mag_finite = np.where(np.isfinite(mag_valid), mag_valid, 0.0)  # Handle any remaining inf/nan
        
        moving = (mag_finite > self.cfg.motion_thresh) & M
        self.motion_hits_sum += moving.astype(np.float64)
        self.mean_mag_sum    += mag_finite  # Already masked by M via mag_valid
        self.valid_sum       += M.astype(np.float64)

        # Moments & sparsity
        self._mag_samples.append(_safe_sample(mag_flat))
        self._moving_total += int(moving.sum())
        self._valid_total  += int(M.sum())

        # ---- Temporal delta ----
        if prev_flow is not None:
            assert prev_flow.shape == flow.shape
            # Check that prev_flow is also finite at valid locations
            prev_u = prev_flow[..., 0]
            prev_v = prev_flow[..., 1]
            M_prev = np.isfinite(prev_u) & np.isfinite(prev_v)
            M_temporal = M & M_prev  # Both current and previous must be valid
            
            du = (flow[..., 0] - prev_flow[..., 0])[M_temporal]
            dv = (flow[..., 1] - prev_flow[..., 1])[M_temporal]
            dmag = np.hypot(du, dv)
            H_d, _ = np.histogram(dmag, bins=self.delta_edges, density=self.cfg.accumulate_density)
            self.delta_hist += H_d
            self._delta_samples.append(_safe_sample(dmag))

        # ---- Divergence / Curl ----
        if H >= 3 and W >= 3:
            ux = 0.5 * (u[:, 2:] - u[:, :-2])
            uy = 0.5 * (u[2:, :] - u[:-2, :])
            vx = 0.5 * (v[:, 2:] - v[:, :-2])
            vy = 0.5 * (v[2:, :] - v[:-2, :])
            Mi = M[1:-1, 1:-1]
            div  = ux[1:-1, :] + vy[:, 1:-1]
            curl = vx[1:-1, :] - uy[:, 1:-1]
            
            # Ensure div/curl are finite (gradients of inf values will be inf/nan)
            div_finite = np.isfinite(div)
            curl_finite = np.isfinite(curl)
            Mi_div = Mi & div_finite
            Mi_curl = Mi & curl_finite
            
            H_div,  _ = np.histogram(div[Mi_div],  bins=self.div_edges,  density=self.cfg.accumulate_density)
            H_curl, _ = np.histogram(curl[Mi_curl], bins=self.curl_edges, density=self.cfg.accumulate_density)
            self.div_hist  += H_div
            self.curl_hist += H_curl

        self.frames_seen += 1

    def finalize(self) -> Dict[str, Any]:
        mag_all   = np.concatenate(self._mag_samples)   if self._mag_samples else np.array([], dtype=np.float64)
        delta_all = np.concatenate(self._delta_samples) if self._delta_samples else np.array([], dtype=np.float64)

        def to_prob(H: np.ndarray) -> np.ndarray:
            return _density_to_prob(H) if self.cfg.accumulate_density else (H / H.sum() if H.sum() > 0 else H)

        # Spatial: binary probability and mean magnitude
        if self.frames_seen == 0 or self.valid_sum is None:
            spatial_prob = np.zeros(self.cfg.spatial_downsample_hw, dtype=float)
            spatial_mean = np.zeros(self.cfg.spatial_downsample_hw, dtype=float)
        else:
            # Probability of motion > τ: motion_hits_sum / frames_seen  (implicitly over valid pixels)
            # Clamp by valid_sum>0 to avoid div/0; for probability we ignore exact valid fraction and treat missing as 0.
            prob_full = np.where(self.valid_sum > 0, self.motion_hits_sum / self.frames_seen, 0.0)
            mean_full = np.where(self.valid_sum > 0, self.mean_mag_sum    / np.maximum(self.valid_sum, 1e-12), 0.0)

            spatial_prob = _downsample_mean(prob_full, self.cfg.spatial_downsample_hw)
            spatial_mean = _downsample_mean(mean_full, self.cfg.spatial_downsample_hw)

        stats = {
            "config": asdict(self.cfg),
            "bins": {
                "mag_edges":   self.mag_edges.tolist(),
                "ang_edges":   self.ang_edges.tolist(),
                "joint_mag_edges": self.jmag_edges.tolist(),
                "joint_ang_edges": self.jang_edges.tolist(),
                "delta_edges": self.delta_edges.tolist(),
                "div_edges":   self.div_edges.tolist(),
                "curl_edges":  self.curl_edges.tolist(),
            },
            "hists": {
                "mag":   to_prob(self.mag_hist).tolist(),
                "angle": to_prob(self.ang_hist).tolist(),
                "joint_mag_angle": _to_prob2d(self.joint_hist).tolist(),
                "delta": to_prob(self.delta_hist).tolist(),
                "div":   to_prob(self.div_hist).tolist(),
                "curl":  to_prob(self.curl_hist).tolist(),
            },
            "moments": {
                "mag_mean":   float(np.mean(mag_all))   if mag_all.size else float("nan"),
                "mag_median": _percentile(mag_all, 50)  if mag_all.size else float("nan"),
                "mag_p95":    _percentile(mag_all, 95)  if mag_all.size else float("nan"),
                "delta_mean": float(np.mean(delta_all)) if delta_all.size else float("nan"),
                "delta_p90":  _percentile(delta_all, 90) if delta_all.size else float("nan"),
                "sparsity_motion_frac": (self._moving_total / self._valid_total) if self._valid_total > 0 else float("nan"),
            },
            "spatial": {
                "grid_hw": self.cfg.spatial_downsample_hw,
                "motion_prob": spatial_prob.astype(float).tolist(),
                "mean_magnitude": spatial_mean.astype(float).tolist(),
            }
        }

        if self.cfg.keep_counts:
            stats["counts"] = {
                "mag":   self.mag_hist.astype(float).tolist(),
                "angle": self.ang_hist.astype(float).tolist(),
                "joint_mag_angle": self.joint_hist.astype(float).tolist(),
                "delta": self.delta_hist.astype(float).tolist(),
                "div":   self.div_hist.astype(float).tolist(),
                "curl":  self.curl_hist.astype(float).tolist(),
            }

        return stats


# ------------------------------
# JSON I/O
# ------------------------------

def save_stats_json(path: str, stats: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

def load_stats_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
