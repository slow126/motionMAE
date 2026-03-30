#!/usr/bin/env python3
"""
Latent-space UMAP analysis for the DINO-context consistency MAE.

Two ablation modes (--mode):

  all_datasets   Run every configured dataset at full observation (0% masked)
                 AND fully masked (100%). Build a joint UMAP coloured by dataset,
                 shaped by masking regime (● full, ▲ masked).  Goal: do dataset
                 clusters still form independently of the masking regime?

  ft3d_sweep     Run FlyingThings3D train at 0%, 10%, 20%, …, 100% masked.
                 Build a joint UMAP coloured by mask percentage.  Goal: does mask
                 ratio dominate the embedding, or is masking invariant?

Latent extracted: pair_latent  (mean-pool of local encoder tokens, pre-projector)

Usage (from repo root):
    python scripts/consistency_umap.py \\
        --snapshot /path/to/snapshot \\
        --dino-model-dir /path/to/dinov3-vitb16 \\
        --mode all_datasets

    python scripts/consistency_umap.py \\
        --snapshot /path/to/snapshot \\
        --dino-model-dir /path/to/dinov3-vitb16 \\
        --mode ft3d_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO_ROOT / ".cache" / "numba"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets
from torchvision.transforms.functional import normalize as tv_normalize

# ── repo root ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(REPO_ROOT))

from src.flow_mae.dino_context_consistency_model import (
    FlowMAEDINOContextConsistencyModelConfig,
    FlowMaskedAutoencoderDINOPrependedContextConsistencyViT,
)
from src.flow_smoke.dataset import PointOdysseyFlowSmokeDataset
from src.data.synth.datasets.flow_utils import flow_from_kps
from src.data.synth.datasets.TSSDataset import TSSSimpleDataset
from src.data.synth.datasets.KittiDataset import KittiSimpleDataset
from src.data.real.datasets.semantic_pairs.spair import SPairDataset
from src.data.real.datasets.semantic_pairs.pfpascal import PFPascalDataset
from src.data.real.datasets.semantic_pairs.pfwillow import PFWillowDataset
from src.data.real.datasets.semantic_pairs.dataset import CorrespondenceDataset as _CorrBase

# ── global constants ──────────────────────────────────────────────────────────
IMAGE_SIZE           = (256, 256)
FLOW_SCALE           = float(max(IMAGE_SIZE))   # 256 – matches training
SAMPLES_PER_DATASET  = 128
GRID_SAMPLES         = 16
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN   = (0.485, 0.456, 0.406)
IMAGENET_STD    = (0.229, 0.224, 0.225)
IMAGENET_MEAN_T = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
IMAGENET_STD_T  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

# ── dataset paths ─────────────────────────────────────────────────────────────
FLYINGTHINGS_ROOT = "/home/spencer/Data/FlyingThings3D_10k"
SINTEL_ROOT       = "/home/spencer/Data"
POINTODYSSEY_ROOT = "/home/spencer/Data/PointOdyssey"
CORR_ROOT         = "/home/spencer/Data/correspondence"
CATS_ROOT         = "/home/spencer/Projects/OnlineSyntheticCorrespondence/models/Datasets_CATs"
TSS_ROOT          = f"{CORR_ROOT}/TSS_CVPR2016"
KITTI_ROOT        = f"{CORR_ROOT}/kitti/kitti-2015"

# ── dataset registry ──────────────────────────────────────────────────────────
DATASET_CONFIGS: List[Tuple[str, str, str, dict]] = [
    ("FlyingThings3D", "train",  "ft3d",        {"split": "train"}),
    ("FlyingThings3D", "test",   "ft3d",        {"split": "test"}),
    ("Sintel-clean",   "train",  "sintel",       {}),
    ("PointOdyssey",   "train",  "pointodyssey", {}),
    ("KITTI-2015",     "train",  "kitti",        {}),
    ("TSS",            "test",   "tss",          {}),
    ("SPair-71k",      "train",  "spair",        {"split": "trn"}),
    ("SPair-71k",      "test",   "spair",        {"split": "test"}),
    ("PF-Pascal",      "train",  "pfpascal",     {"split": "trn"}),
    ("PF-Pascal",      "test",   "pfpascal",     {"split": "val"}),
    ("PF-Willow",      "test",   "pfwillow",     {}),
]

FT3D_SWEEP_CONFIGS: List[Tuple[str, str, str, dict]] = [
    ("FlyingThings3D", "train", "ft3d", {"split": "train"}),
]


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint selection
# ══════════════════════════════════════════════════════════════════════════════

def pick_best_checkpoint(snapshot_dir: Path) -> Path:
    epoch_re   = re.compile(r"epoch(\d+)")
    best_epoch = -1
    best_ckpt: Optional[Path] = None

    run_dirs = [snapshot_dir] if (snapshot_dir / "checkpoints").exists() \
               else sorted(snapshot_dir.iterdir())

    for run_dir in run_dirs:
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        for ckpt in ckpt_dir.glob("epoch*.ckpt"):
            m = epoch_re.search(ckpt.name)
            if m and int(m.group(1)) > best_epoch:
                best_epoch = int(m.group(1))
                best_ckpt  = ckpt
    if best_ckpt is None:
        raise RuntimeError(f"No epoch checkpoints under {snapshot_dir}")
    print(f"[checkpoint] epoch {best_epoch}: {best_ckpt}")
    return best_ckpt


def _is_complete_dino_snapshot(path: Path) -> bool:
    required = ("preprocessor_config.json", "config.json")
    if not all((path / name).exists() for name in required):
        return False
    if (path / "model.safetensors").exists():
        return True
    if (path / "pytorch_model.bin").exists():
        return True
    if (path / "model.safetensors.index.json").exists():
        return True
    if any(path.glob("*.safetensors")):
        return True
    if any(path.glob("*.bin")):
        return True
    return False


def _find_dino_model_dir_from_config(snapshot_dir: Path) -> Optional[str]:
    """Try to extract dino_model_dir from config.yaml in the snapshot.

    Only returns the path if it exists locally; ignores RC/cluster paths.
    """
    cfg_path = snapshot_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        candidates = []
        for probe in (cfg.get("qualitative_probes") or []):
            model_dir = probe.get("dino_model_dir")
            if model_dir:
                candidates.append(str(model_dir))
        probe = cfg.get("pointodyssey_probe") or {}
        model_dir = probe.get("dino_model_dir")
        if model_dir:
            candidates.append(str(model_dir))
        for c in candidates:
            candidate = Path(c).expanduser()
            if candidate.exists():
                if candidate.is_dir() and not _is_complete_dino_snapshot(candidate):
                    print(f"[dino] config path exists but is not a complete DinoV3 snapshot: {candidate}")
                    continue
                return str(candidate)
            print(f"[dino] config path not accessible locally: {c}")
    except Exception as e:
        print(f"[dino] could not parse config.yaml: {e}")
    return None


# HuggingFace model ID for DINOv3 ViT-B/16.
_DINO_HF_DEFAULT = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def _resolve_local_hf_dino_snapshot(model_id: str) -> Optional[str]:
    try:
        snapshot_path = Path(snapshot_download(model_id, local_files_only=True))
    except Exception as e:
        print(f"[dino] no local HuggingFace cache for {model_id}: {e}")
        return None

    if _is_complete_dino_snapshot(snapshot_path):
        print(f"[dino] using local HuggingFace snapshot: {snapshot_path}")
        return str(snapshot_path)

    print(f"[dino] local HuggingFace snapshot is incomplete: {snapshot_path}")
    return None


def _download_hf_dino_snapshot(model_id: str) -> str:
    print(f"[dino] downloading DinoV3 snapshot from Hugging Face: {model_id}")
    snapshot_path = Path(snapshot_download(model_id, local_files_only=False))
    if not _is_complete_dino_snapshot(snapshot_path):
        print("[dino] initial download did not produce a complete local snapshot; retrying with force_download=True")
        snapshot_path = Path(snapshot_download(model_id, local_files_only=False, force_download=True))
    if not _is_complete_dino_snapshot(snapshot_path):
        raise RuntimeError(
            f"Downloaded DinoV3 snapshot is still incomplete: {snapshot_path}"
        )
    print(f"[dino] download complete: {snapshot_path}")
    return str(snapshot_path)


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: Path) -> FlowMaskedAutoencoderDINOPrependedContextConsistencyViT:
    print(f"[model] loading {ckpt_path.name} …")
    ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp    = ckpt["hyper_parameters"]
    cfg   = FlowMAEDINOContextConsistencyModelConfig(**hp["model"])
    model = FlowMaskedAutoencoderDINOPrependedContextConsistencyViT(cfg)
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
             if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(DEVICE)
    print(f"[model] encoder_dim={cfg.encoder_dim} depth={cfg.encoder_depth} "
          f"use_rgb={cfg.use_rgb_inputs} image_size={cfg.image_size}")
    return model


class _DinoWrapper:
    """
    Thin wrapper that extracts spatial patch tokens from a DINO-family model.
    Accepts already-ImageNet-normalized (B, 3, H, W) tensors.
    Returns (B, N, D) where N = num_patch_tokens and D = feature dim.
    """
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _DinoV3Wrapper(_DinoWrapper):
    def __init__(self, dino):
        self._dino = dino

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self._dino.get_spatial_features(images).float()  # (B, N, D)


def load_dino(dino_model_dir: str) -> _DinoWrapper:
    try:
        import transformers
        if not transformers.is_torch_available():
            raise RuntimeError(
                "transformers has PyTorch model support disabled in this environment. "
                f"transformers={transformers.__version__}, torch={torch.__version__}. "
                "Install a compatible combination before loading DinoV3."
            )
        from models.DinoV3.DinoV3 import DinoV3
        print(f"[dino] loading DinoV3 from {dino_model_dir} …")
        dino = DinoV3(pretrained_model_name=str(dino_model_dir), resize_size=IMAGE_SIZE[0])
        dino.model.to(DEVICE)
        dino.model.eval()
        print("[dino] DinoV3 ready")
        return _DinoV3Wrapper(dino)
    except Exception as e:
        raise RuntimeError(
            "Failed to load DinoV3. This UMAP path requires the same DinoV3 family used "
            f"during training and will not fall back to DinoV2.\n"
            f"Requested model dir/id: {dino_model_dir}\n"
            f"Original error: {e}\n"
            "If you are running locally, activate a Python environment with DinoV3 support "
            "(new enough `transformers` to include `dinov3_vit`) and pass a complete local "
            "DinoV3 snapshot via --dino-model-dir if needed."
        ) from e


@torch.no_grad()
def compute_dino_tokens(
    dino: _DinoWrapper,
    src_rgb: torch.Tensor,
    tgt_rgb: torch.Tensor,
    grid_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    src_rgb / tgt_rgb: (B, 3, H, W) ImageNet-normalized float32, already on DEVICE.
    Returns src_dino, tgt_dino each of shape (B, grid_size, grid_size, context_dim).
    """
    B = src_rgb.shape[0]
    src_feats = dino(src_rgb)   # (B, N, D)
    tgt_feats = dino(tgt_rgb)   # (B, N, D)
    expected_tokens = grid_size * grid_size
    if src_feats.shape[1] != expected_tokens or tgt_feats.shape[1] != expected_tokens:
        raise RuntimeError(
            "DinoV3 token grid mismatch: "
            f"expected {expected_tokens} patch tokens ({grid_size}x{grid_size}), "
            f"got src={src_feats.shape[1]} tgt={tgt_feats.shape[1]}. "
            "Check that the loaded DinoV3 model and image resize match training."
        )
    D = src_feats.shape[-1]
    src_dino = src_feats.reshape(B, grid_size, grid_size, D)
    tgt_dino = tgt_feats.reshape(B, grid_size, grid_size, D)
    return src_dino, tgt_dino


# ══════════════════════════════════════════════════════════════════════════════
# Shared tensor helpers (same as mae_latent_umap)
# ══════════════════════════════════════════════════════════════════════════════

def _resize_rgb(img: torch.Tensor) -> torch.Tensor:
    if img.shape[-2:] == IMAGE_SIZE:
        return img
    return F.interpolate(img.unsqueeze(0), size=IMAGE_SIZE,
                         mode="bilinear", align_corners=False).squeeze(0)


def _resize_flow_valid(flow: torch.Tensor,
                       valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = flow.shape[-2:]
    if (h, w) == IMAGE_SIZE:
        return flow, valid
    sh, sw = IMAGE_SIZE[0] / h, IMAGE_SIZE[1] / w
    flow = F.interpolate(flow.unsqueeze(0), size=IMAGE_SIZE,
                         mode="bilinear", align_corners=False).squeeze(0)
    flow[0] *= sw
    flow[1] *= sh
    valid = (F.interpolate(valid.float().unsqueeze(0).unsqueeze(0),
                           size=IMAGE_SIZE, mode="nearest").squeeze() > 0.5)
    flow = torch.where(valid.unsqueeze(0), flow, torch.zeros_like(flow))
    return flow, valid


def _imagenet_normalize(img: torch.Tensor) -> torch.Tensor:
    return tv_normalize(img.clamp(0, 1), IMAGENET_MEAN, IMAGENET_STD)


def _sample_observed_override(valid: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """
    Build an observed-flow mask.
      mask_ratio=0.0  → all valid pixels observed (full observation)
      mask_ratio=1.0  → no pixels observed (fully masked)
    Returns a float tensor with 1=observed, 0=masked.
    """
    valid = valid.bool()
    observed = valid.float().clone()
    if float(valid.sum()) <= 0.0:
        return observed
    if mask_ratio <= 0.0:
        return observed
    if mask_ratio >= 1.0:
        return torch.zeros_like(observed)

    valid_idx  = torch.nonzero(valid, as_tuple=False)
    n_valid    = int(valid_idx.shape[0])
    num_mask   = int(round(n_valid * mask_ratio))
    num_mask   = max(0, min(num_mask, n_valid - 1))  # keep at least 1 visible
    if num_mask <= 0:
        return observed
    choice = valid_idx[torch.randperm(n_valid)[:num_mask]]
    observed[choice[:, 0], choice[:, 1]] = 0.0
    return observed


def _finalize_sample(
    src_rgb: torch.Tensor,
    tgt_rgb: torch.Tensor,
    flow_px: torch.Tensor,
    valid: torch.Tensor,
    already_normalized: bool,
) -> dict:
    flow_px = torch.nan_to_num(flow_px, 0.0, 0.0, 0.0)
    max_mag = FLOW_SCALE * 2.0
    mag     = torch.linalg.vector_norm(flow_px, dim=0)
    valid   = valid.bool() & (mag <= max_mag)
    flow_px = torch.where(valid.unsqueeze(0), flow_px, torch.zeros_like(flow_px))

    src_rgb = _resize_rgb(src_rgb)
    tgt_rgb = _resize_rgb(tgt_rgb)
    flow_px, valid = _resize_flow_valid(flow_px, valid)

    if not already_normalized:
        src_rgb = _imagenet_normalize(src_rgb)
        tgt_rgb = _imagenet_normalize(tgt_rgb)

    return {
        "src_rgb":    src_rgb,
        "tgt_rgb":    tgt_rgb,
        "flow":       flow_px / FLOW_SCALE,
        "valid":      valid.float(),
        "flow_scale": torch.tensor(FLOW_SCALE, dtype=torch.float32),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Dataset wrappers  (same as mae_latent_umap, minus observed_valid_override —
# that is built in the inference loop where we control the mask ratio)
# ══════════════════════════════════════════════════════════════════════════════

class FlyingThingsDataset(Dataset):
    def __init__(self, split: str, n: int):
        base       = tv_datasets.FlyingThings3D(FLYINGTHINGS_ROOT, split=split,
                                                pass_name="clean", camera="left")
        self._base = base
        self._idx  = list(range(min(n, len(base))))

    def __len__(self): return len(self._idx)

    def __getitem__(self, i):
        item = self._base[self._idx[i]]
        src  = torch.from_numpy(np.array(item[0])).permute(2, 0, 1).float() / 255.0
        tgt  = torch.from_numpy(np.array(item[1])).permute(2, 0, 1).float() / 255.0
        flow = torch.from_numpy(np.array(item[2])).float()
        valid = torch.isfinite(flow).all(dim=0)
        return _finalize_sample(src, tgt, flow, valid, already_normalized=False)


class SintelDataset(Dataset):
    def __init__(self, n: int):
        self._base = tv_datasets.Sintel(SINTEL_ROOT, split="train", pass_name="clean")
        self._idx  = list(range(min(n, len(self._base))))

    def __len__(self): return len(self._idx)

    def __getitem__(self, i):
        item = self._base[self._idx[i]]
        src  = torch.from_numpy(np.array(item[0])).permute(2, 0, 1).float() / 255.0
        tgt  = torch.from_numpy(np.array(item[1])).permute(2, 0, 1).float() / 255.0
        flow = torch.from_numpy(np.array(item[2])).float()
        valid = torch.isfinite(flow).all(dim=0)
        return _finalize_sample(src, tgt, flow, valid, already_normalized=False)


class PointOdysseyDataset(Dataset):
    def __init__(self, manifest_path: str, n: int):
        with open(manifest_path) as f:
            n_entries = sum(1 for _ in f)
        self._base = PointOdysseyFlowSmokeDataset(
            manifest_path=manifest_path,
            indices=list(range(min(n, n_entries))),
            pointodyssey_root=POINTODYSSEY_ROOT,
            reverse_flow=False,
            size=IMAGE_SIZE,
            min_valid_points=64,
            normalize_flow=False,
            trust_manifest=True,
        )

    def __len__(self): return len(self._base)

    def __getitem__(self, i):
        s     = self._base[i]
        src   = s["src_img"]
        tgt   = s["trg_img"]
        flow  = -s["flow"]
        valid = s["valid_flow_mask"].bool()
        return _finalize_sample(src, tgt, flow, valid, already_normalized=False)


class TSSDataset(Dataset):
    def __init__(self, n: int):
        self._base = TSSSimpleDataset(root=TSS_ROOT)
        self._idx  = list(range(min(n, len(self._base))))

    def __len__(self): return len(self._idx)

    def __getitem__(self, i):
        s    = self._base[self._idx[i]]
        src  = s["trg_img"]
        tgt  = s["src_img"]
        flow = s["flow"]
        valid = torch.isfinite(flow).all(dim=0)
        return _finalize_sample(src, tgt, flow, valid, already_normalized=False)


class KittiDataset(Dataset):
    def __init__(self, n: int):
        self._base = KittiSimpleDataset(root=KITTI_ROOT, split="train", version="2015")
        self._idx  = list(range(min(n, len(self._base))))

    def __len__(self): return len(self._idx)

    def __getitem__(self, i):
        s    = self._base[self._idx[i]]
        src  = s["trg_img"]
        tgt  = s["src_img"]
        flow = s["flow"]
        valid = torch.isfinite(flow).all(dim=0)
        return _finalize_sample(src, tgt, flow, valid, already_normalized=False)


class SparseKpsDataset(Dataset):
    def __init__(self, raw_dataset, n: int, use_parent_getitem: bool = False):
        self._base       = raw_dataset
        self._idx        = list(range(min(n, len(raw_dataset))))
        self._use_parent = use_parent_getitem

    def __len__(self): return len(self._idx)

    def __getitem__(self, i):
        real_idx = self._idx[i]
        if self._use_parent:
            s = _CorrBase.__getitem__(self._base, real_idx)
        else:
            s = self._base[real_idx]
        src     = s["src_img"]
        tgt     = s["trg_img"]
        n_pts   = int(s["n_pts"])
        src_kps = s["src_kps"][:, :n_pts]
        trg_kps = s["trg_kps"][:, :n_pts]
        flow    = flow_from_kps(trg_kps, src_kps, IMAGE_SIZE)
        valid   = torch.isfinite(flow).all(dim=0)
        return _finalize_sample(src, tgt, flow, valid, already_normalized=True)


def _build_dataset(factory_key: str, factory_kwargs: dict, n: int) -> Dataset:
    if factory_key == "ft3d":
        return FlyingThingsDataset(split=factory_kwargs["split"], n=n)
    if factory_key == "sintel":
        return SintelDataset(n=n)
    if factory_key == "pointodyssey":
        manifest = REPO_ROOT / "analysis" / "pointodyssey_pairs_smoke" / "manifest.jsonl"
        if not manifest.exists():
            snap_dir = REPO_ROOT / "snapshots_mae" / "snapshots"
            manifest = next(snap_dir.rglob("pointodyssey_probe_manifest.jsonl"), None)
            if manifest is None:
                raise RuntimeError("No PointOdyssey manifest found")
        return PointOdysseyDataset(str(manifest), n)
    if factory_key == "tss":
        return TSSDataset(n=n)
    if factory_key == "kitti":
        return KittiDataset(n=n)
    if factory_key == "spair":
        raw = SPairDataset("spair", CORR_ROOT, "bbox", factory_kwargs["split"], augmentation=False)
        return SparseKpsDataset(raw, n)
    if factory_key == "pfpascal":
        raw = PFPascalDataset("pfpascal", CORR_ROOT, "bbox", factory_kwargs["split"], augmentation=False)
        return SparseKpsDataset(raw, n)
    if factory_key == "pfwillow":
        raw = PFWillowDataset("pfwillow", CATS_ROOT, "bbox", "test",
                              augmentation=False, feature_size=64)
        return SparseKpsDataset(raw, n, use_parent_getitem=True)
    raise ValueError(f"Unknown factory key: {factory_key!r}")


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _denorm_rgb(img: torch.Tensor) -> np.ndarray:
    img = (img.cpu() * IMAGENET_STD_T + IMAGENET_MEAN_T).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def _flow_to_rgb_hsv(flow: torch.Tensor) -> np.ndarray:
    from matplotlib.colors import hsv_to_rgb as _hsv_to_rgb
    flow  = torch.nan_to_num(flow.cpu().float(), 0.0, 0.0, 0.0)
    u, v  = flow[0].numpy(), flow[1].numpy()
    angle = np.arctan2(v, u)
    mag   = np.hypot(u, v)
    p99   = max(float(np.percentile(mag, 99)), 1e-6)
    hue   = (angle / (2.0 * np.pi)) % 1.0
    val   = np.clip(mag / p99, 0.0, 1.0)
    return _hsv_to_rgb(np.stack([hue, np.ones_like(hue), val], axis=-1)).astype(np.float32)


def save_prediction_grid(samples: List[dict], label: str, out_dir: Path) -> None:
    N   = len(samples)
    fig = plt.figure(figsize=(22, N * 4), dpi=100)
    gs  = gridspec.GridSpec(N, 5, figure=fig, hspace=0.04, wspace=0.04)
    col_titles = ["Source RGB", "Target RGB", "GT Flow", "Pred Flow", "Observed Mask"]

    for row, s in enumerate(samples):
        obs_frac  = float(s["obs_mask"].float().mean())
        obs_label = (f"{obs_frac*100:.0f}% obs"
                     if 0.0 < obs_frac < 1.0
                     else ("all obs" if obs_frac >= 1.0 else "none obs"))
        panels = [
            _denorm_rgb(s["src_rgb"]),
            _denorm_rgb(s["tgt_rgb"]),
            _flow_to_rgb_hsv(s["gt_flow_px"]),
            _flow_to_rgb_hsv(s["pred_flow_px"]),
            s["obs_mask"].float().numpy()[..., np.newaxis].repeat(3, axis=-1),
        ]
        for col, (img, title) in enumerate(zip(panels, col_titles)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img, vmin=0.0, vmax=1.0)
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=11, pad=4)
            if col == 0:
                ax.set_ylabel(
                    f"#{row}  {obs_label}",
                    fontsize=7, rotation=0, labelpad=68, va="center",
                )

    fig.suptitle(label, fontsize=13, y=1.002)
    slug = label.replace(" ", "_").replace("/", "-").replace("%", "pct")
    out  = out_dir / "grids" / f"{slug}_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [grid] saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Inference loop
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model: FlowMaskedAutoencoderDINOPrependedContextConsistencyViT,
    dino: "DinoV3",
    dataset: Dataset,
    label: str,
    mask_ratio: float,
    out_dir: Path,
    *,
    observation_mode: str = "iid",
) -> dict:
    """
    mask_ratio: fraction of valid pixels to mask.
      0.0 = full observation, 1.0 = fully masked.
    observation_mode:
      "iid"        = per-pixel iid masking override used by the original script
      "train_like" = patch masking + speckle observation matching student train-time structure
    Returns dict with latents (N, D), epe, and a grid.
    """
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    all_latents:  List[np.ndarray] = []
    all_epe:      List[float]      = []
    grid_samples: List[dict]       = []

    for batch in loader:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        src_rgb    = batch["src_rgb"]
        tgt_rgb    = batch["tgt_rgb"]
        flow       = batch["flow"]
        valid      = batch["valid"]
        flow_scale = batch["flow_scale"].view(-1, 1, 1, 1)

        if observation_mode == "iid":
            obs_list = []
            for b in range(valid.shape[0]):
                obs_list.append(_sample_observed_override(valid[b], mask_ratio))
            observed_valid = torch.stack(obs_list, dim=0).to(DEVICE)  # (B, H, W)
        elif observation_mode == "train_like":
            if mask_ratio <= 0.0:
                observed_valid = valid.clone()
            elif mask_ratio >= 1.0:
                observed_valid = torch.zeros_like(valid)
            else:
                view = model.build_observation(
                    valid,
                    mask_ratio=mask_ratio,
                    speckle_keep_ratio=float(model.config.student_speckle_keep_ratio),
                    speckle_dilation_kernel=int(model.config.student_speckle_dilation_kernel),
                    allow_full_mask=False,
                    ensure_nonempty=True,
                    is_full_mask=False,
                )
                observed_valid = view["observed_valid"]
        else:
            raise ValueError(f"Unknown observation_mode: {observation_mode!r}")

        # Compute DINO tokens on the fly
        src_dino, tgt_dino = compute_dino_tokens(
            dino, src_rgb, tgt_rgb, model.grid_size
        )

        # Masked observed flow
        obs_flow = flow * observed_valid.unsqueeze(1)

        outputs = model.forward_branch(
            src_rgb=src_rgb,
            tgt_rgb=tgt_rgb,
            src_dino=src_dino,
            tgt_dino=tgt_dino,
            observed_flow=obs_flow,
            observed_valid=observed_valid,
            return_latent=True,
            decode=True,
        )

        pair_latent = outputs["pair_latent"]       # (B, D)
        pred_flow   = outputs["pred_flow"]         # (B, 2, H, W) normalised

        all_latents.append(pair_latent.cpu().float().numpy())

        # EPE in pixel space
        pred_px = pred_flow * flow_scale
        gt_px   = flow      * flow_scale
        safe_gt = torch.where(valid.unsqueeze(1) > 0, gt_px, pred_px.detach())
        epe     = torch.linalg.vector_norm(pred_px - safe_gt, dim=1)
        denom   = valid.sum(dim=(-2, -1)).clamp_min(1.0)
        epe_b   = (epe * valid).sum(dim=(-2, -1)) / denom
        all_epe.extend(epe_b.cpu().tolist())

        # Grid accumulation
        for b in range(src_rgb.shape[0]):
            if len(grid_samples) < GRID_SAMPLES:
                grid_samples.append({
                    "src_rgb":      src_rgb[b].cpu(),
                    "tgt_rgb":      tgt_rgb[b].cpu(),
                    "gt_flow_px":   gt_px[b].cpu(),
                    "pred_flow_px": pred_px[b].cpu(),
                    "valid":        valid[b].cpu(),
                    "obs_mask":     observed_valid[b].cpu(),
                })

    latents  = np.concatenate(all_latents, axis=0)
    mean_epe = float(np.mean(all_epe)) if all_epe else 0.0
    mean_observed_ratio = float(np.mean([float(s["obs_mask"].float().mean()) for s in grid_samples])) if grid_samples else 0.0
    print(
        f"  [{label}] n={len(all_epe)}  mean_epe={mean_epe:.2f}px  "
        f"mean_observed_ratio={mean_observed_ratio:.3f}"
    )
    save_prediction_grid(grid_samples, label, out_dir)
    return {
        "latents": latents,
        "epe": mean_epe,
        "all_epe": all_epe,
        "mean_observed_ratio": mean_observed_ratio,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UMAP helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fit_umap(latents: np.ndarray) -> np.ndarray:
    try:
        import umap as umap_lib
    except ImportError:
        raise RuntimeError("umap-learn not installed — pip install umap-learn")
    print(f"[UMAP] fitting {latents.shape[0]}×{latents.shape[1]} …")
    reducer = umap_lib.UMAP(n_components=2, random_state=42,
                            n_neighbors=15, min_dist=0.1, metric="cosine")
    emb = reducer.fit_transform(latents)
    print("[UMAP] done")
    return emb


# ══════════════════════════════════════════════════════════════════════════════
# Mode: all_datasets
#   For every configured dataset: embed at full (0%) and fully-masked (100%).
#   Joint UMAP: colour = dataset, marker shape = mode.
# ══════════════════════════════════════════════════════════════════════════════

def run_all_datasets(
    model: FlowMaskedAutoencoderDINOPrependedContextConsistencyViT,
    dino: "DinoV3",
    out_dir: Path,
    n_samples: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grids").mkdir(exist_ok=True)

    # Collect (label_str, mask_ratio, factory_key, factory_kwargs, display_name)
    entries = []
    for display_name, split_label, factory_key, factory_kwargs in DATASET_CONFIGS:
        dataset_label = f"{display_name} [{split_label}]"
        entries.append((dataset_label, 0.0,  factory_key, factory_kwargs, display_name))
        entries.append((dataset_label, 1.0,  factory_key, factory_kwargs, display_name))

    # dataset_name → int label (shared across modes so same dataset = same colour)
    dataset_to_label: Dict[str, int] = {}
    label_names: List[str]           = []

    all_latents:       List[np.ndarray] = []
    all_dataset_labels: List[int]       = []
    all_is_masked:     List[bool]       = []   # True = fully-masked branch
    metrics: dict = {}

    for dataset_label, mask_ratio, factory_key, factory_kwargs, display_name in entries:
        mode_str  = "masked" if mask_ratio >= 1.0 else "full"
        label_str = f"{dataset_label} | {mode_str}"
        print(f"\n[all_datasets] {label_str}")

        try:
            ds = _build_dataset(factory_key, factory_kwargs, n_samples)
        except Exception as exc:
            print(f"  SKIP – {exc}")
            continue

        if len(ds) == 0:
            print("  SKIP – empty dataset")
            continue

        if dataset_label not in dataset_to_label:
            dataset_to_label[dataset_label] = len(label_names)
            label_names.append(dataset_label)

        result = run_inference(model, dino, ds, label_str, mask_ratio, out_dir)
        all_latents.append(result["latents"])
        all_dataset_labels.extend([dataset_to_label[dataset_label]] * result["latents"].shape[0])
        all_is_masked.extend([mask_ratio >= 1.0] * result["latents"].shape[0])
        metrics[label_str] = {"n_samples": len(result["all_epe"]), "mean_epe_px": result["epe"]}

    if not all_latents:
        raise RuntimeError("No datasets succeeded.")

    latents_all     = np.concatenate(all_latents, axis=0)
    ds_labels_arr   = np.array(all_dataset_labels, dtype=np.int32)
    is_masked_arr   = np.array(all_is_masked, dtype=bool)

    np.savez(out_dir / "latents.npz",
             latents=latents_all,
             dataset_labels=ds_labels_arr,
             is_masked=is_masked_arr.astype(np.int32),
             label_names=np.array(label_names))

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[metrics] → {out_dir / 'metrics.json'}")

    emb = _fit_umap(latents_all)
    np.save(out_dir / "umap_embedding.npy", emb)

    unique_ds = sorted(set(ds_labels_arr.tolist()))
    cmap      = plt.colormaps.get_cmap("tab10")
    colours   = {lbl: cmap(i / max(len(unique_ds) - 1, 1)) for i, lbl in enumerate(unique_ds)}

    fig, ax = plt.subplots(figsize=(12, 8), dpi=130)

    # ● full observation
    full_mask = ~is_masked_arr
    for lbl in unique_ds:
        m = (ds_labels_arr == lbl) & full_mask
        if m.any():
            ax.scatter(emb[m, 0], emb[m, 1],
                       c=[colours[lbl]], marker="o",
                       s=60, alpha=0.80, edgecolors="none",
                       label=label_names[lbl])

    # ▲ fully masked
    for lbl in unique_ds:
        m = (ds_labels_arr == lbl) & is_masked_arr
        if m.any():
            ax.scatter(emb[m, 0], emb[m, 1],
                       c=[colours[lbl]], marker="^",
                       s=60, alpha=0.80, edgecolors="none",
                       label=None)

    from matplotlib.lines import Line2D
    marker_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markersize=9, label="● full observation (0% masked)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
               markersize=9, label="▲ fully masked (100%)"),
    ]
    leg1 = ax.legend(title="Dataset / split", framealpha=0.9,
                     bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, title="Mode", framealpha=0.9,
              bbox_to_anchor=(1.02, 0), loc="lower left", fontsize=8)

    ax.set_title("Consistency MAE pair_latent — UMAP (cosine)\n"
                 "● full observation   ▲ fully masked\n"
                 "Same colour = same dataset. Ideal: shapes overlap per cluster.",
                 fontsize=10)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    out = out_dir / "umap_all_datasets.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[UMAP] saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Mode: ft3d_sweep
#   FlyingThings3D train: embed at 0%, 10%, …, 100% masked.
#   Joint UMAP: colour = mask percentage.
#   Ideal: all points intermixed (masking invariant).
#   Bad sign: points cluster by mask% instead of (or in addition to) content.
# ══════════════════════════════════════════════════════════════════════════════

def run_ft3d_sweep(
    model: FlowMaskedAutoencoderDINOPrependedContextConsistencyViT,
    dino: "DinoV3",
    out_dir: Path,
    n_samples: int,
    observation_mode: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grids").mkdir(exist_ok=True)

    mask_percents = list(range(0, 101, 10))   # 0, 10, 20, …, 100

    all_latents:  List[np.ndarray] = []
    all_pct_labels: List[int]      = []
    metrics: dict = {}

    for pct in mask_percents:
        mask_ratio = pct / 100.0
        mode_label = "train-like" if observation_mode == "train_like" else "iid"
        label_str  = f"FlyingThings3D train | {pct}% masked ({mode_label})"
        print(f"\n[ft3d_sweep] {label_str}")

        # Use the same indices every time so content is constant across mask levels.
        # We load fresh each time (Dataset re-seeds per sample), but indices are
        # fixed from [0, n_samples).
        ds = FlyingThingsDataset(split="train", n=n_samples)

        result = run_inference(
            model,
            dino,
            ds,
            label_str,
            mask_ratio,
            out_dir,
            observation_mode=observation_mode,
        )
        all_latents.append(result["latents"])
        all_pct_labels.extend([pct] * result["latents"].shape[0])
        metrics[label_str] = {
            "n_samples": len(result["all_epe"]),
            "mean_epe_px": result["epe"],
            "mean_observed_ratio": result["mean_observed_ratio"],
            "nominal_mask_ratio": mask_ratio,
        }

    latents_all  = np.concatenate(all_latents, axis=0)
    pct_arr      = np.array(all_pct_labels, dtype=np.int32)

    np.savez(out_dir / "latents.npz",
             latents=latents_all,
             mask_percents=pct_arr)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[metrics] → {out_dir / 'metrics.json'}")

    emb = _fit_umap(latents_all)
    np.save(out_dir / "umap_embedding.npy", emb)

    # Colour by mask percentage using a perceptually-uniform colormap
    cmap       = plt.colormaps.get_cmap("plasma")
    unique_pct = sorted(set(pct_arr.tolist()))
    norm_pct   = {p: p / 100.0 for p in unique_pct}

    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)
    for pct in unique_pct:
        m = pct_arr == pct
        sc = ax.scatter(emb[m, 0], emb[m, 1],
                        c=[cmap(norm_pct[pct])],
                        s=50, alpha=0.75, edgecolors="none",
                        label=f"{pct}%")

    ax.legend(title="Mask %", framealpha=0.9,
              bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
              ncol=2)
    ax.set_title("Consistency MAE pair_latent — UMAP (cosine)\n"
                 "FlyingThings3D train, colour = nominal patch-mask ratio\n"
                 f"{mode_label} sweep.\n"
                 "Ideal: colours fully intermixed (masking-invariant embedding).",
                 fontsize=10)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    out = out_dir / "umap_ft3d_sweep.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[UMAP] saved → {out}")

    # Also save a small 2×6 grid of (mask%, UMAP scatter) thumbnails — handy
    # for quick visual scanning without opening the big plot.
    fig2, axes = plt.subplots(2, 6, figsize=(18, 6), dpi=110)
    for i, pct in enumerate(unique_pct[:12]):  # up to 12 panels
        ax2   = axes.flat[i]
        m     = pct_arr == pct
        other = ~m
        ax2.scatter(emb[other, 0], emb[other, 1], c="lightgrey", s=10, alpha=0.4, edgecolors="none")
        ax2.scatter(emb[m,     0], emb[m,     1], c=[cmap(norm_pct[pct])],
                    s=20, alpha=0.9, edgecolors="none")
        ax2.set_title(f"{pct}% masked", fontsize=8)
        ax2.set_xticks([])
        ax2.set_yticks([])
    for j in range(len(unique_pct), len(axes.flat)):
        axes.flat[j].axis("off")
    fig2.suptitle("FlyingThings sweep — each panel highlights one mask level (grey = rest)", fontsize=10)
    fig2.tight_layout()
    out2 = out_dir / "umap_ft3d_sweep_panels.png"
    fig2.savefig(out2, bbox_inches="tight")
    plt.close(fig2)
    print(f"[UMAP] panels saved → {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Consistency MAE UMAP analysis")
    parser.add_argument("--snapshot", type=str, required=True,
                        help="Path to snapshot directory (must contain checkpoints/).")
    parser.add_argument("--dino-model-dir", type=str, default=None,
                        help="Path to DinoV3 pretrained weights directory. "
                             "If omitted, tries to read from snapshot config.yaml.")
    parser.add_argument("--mode", choices=["all_datasets", "ft3d_sweep"],
                        default="all_datasets",
                        help="all_datasets: full vs fully-masked across all datasets. "
                             "ft3d_sweep: FlyingThings3D at 0%%–100%% masked in 10%% steps.")
    parser.add_argument(
        "--ft3d-mask-mode",
        choices=["iid", "train_like"],
        default="train_like",
        help="Masking structure for ft3d_sweep. "
             "iid = per-pixel salt-and-pepper masking. "
             "train_like = patch masking + speckle observation matching student training.",
    )
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_DATASET,
                        help=f"Samples per dataset/masking-level (default {SAMPLES_PER_DATASET}).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Defaults to "
                             "scripts/consistency_umap_out_<snapshot_name>_<mode>/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    snap_path = Path(args.snapshot)
    if not snap_path.is_absolute():
        snap_path = REPO_ROOT / snap_path

    # Resolve DINO model dir
    dino_model_dir = args.dino_model_dir
    if dino_model_dir:
        candidate = Path(dino_model_dir).expanduser()
        if candidate.exists():
            if candidate.is_dir() and not _is_complete_dino_snapshot(candidate):
                print(
                    "[dino] provided --dino-model-dir is not a complete DinoV3 snapshot; "
                    f"ignoring it and falling back to auto-resolution: {candidate}"
                )
                dino_model_dir = None
            else:
                dino_model_dir = str(candidate)
        else:
            # Treat non-path strings as Hugging Face model IDs.
            dino_model_dir = str(dino_model_dir)
    if not dino_model_dir:
        dino_model_dir = _find_dino_model_dir_from_config(snap_path)
    if not dino_model_dir:
        dino_model_dir = _resolve_local_hf_dino_snapshot(_DINO_HF_DEFAULT)
    if not dino_model_dir:
        try:
            model_id = args.dino_model_dir if args.dino_model_dir and not Path(args.dino_model_dir).expanduser().exists() else _DINO_HF_DEFAULT
            dino_model_dir = _download_hf_dino_snapshot(model_id)
        except Exception as e:
            raise RuntimeError(
                "Could not resolve or download a usable DinoV3 snapshot.\n"
                "The RC snapshot config points to cluster-only paths, and no complete local cache was found.\n"
                f"Download attempt failed for {model_id}: {e}"
            ) from e

    # Resolve output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
    else:
        out_dir = REPO_ROOT / "scripts" / f"consistency_umap_out_{snap_path.name}_{args.mode}"

    print(f"[mode] {args.mode}")
    print(f"[output] {out_dir}")

    ckpt_path = pick_best_checkpoint(snap_path)
    model     = load_model(ckpt_path)
    dino      = load_dino(dino_model_dir)

    if args.mode == "all_datasets":
        run_all_datasets(model, dino, out_dir, args.samples)
    elif args.mode == "ft3d_sweep":
        run_ft3d_sweep(model, dino, out_dir, args.samples, args.ft3d_mask_mode)
    else:
        raise ValueError(f"Unknown mode: {args.mode!r}")

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
