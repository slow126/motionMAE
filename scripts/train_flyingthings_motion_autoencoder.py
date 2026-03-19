#!/usr/bin/env python3
"""Train a small masked motion autoencoder on FlyingThings dense flow.

Inputs to the model:
  - channel 0: observed dx
  - channel 1: observed dy
  - channel 2: observation mask (1 for observed, 0 for masked)

Targets:
  - full dense flow (dx, dy)

Masking modes:
  - random_pixel
  - structured
  - none
  - random_block
  - pointodyssey
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from src.data.synth.datasets.FlyingThingsDataset import FlyingThingsSimpleDataset


def _as_size(size: str | int | None) -> Optional[tuple[int, int]]:
    if size is None:
        return None
    if isinstance(size, int):
        return int(size), int(size)
    if isinstance(size, str):
        parts = [p.strip() for p in size.replace("x", ",").split(",") if p.strip()]
        if len(parts) == 1 and parts[0].isdigit():
            return int(parts[0]), int(parts[0])
        if len(parts) != 2:
            raise ValueError(f"Expected --size as int or HxW, got {size!r}")
        return int(parts[0]), int(parts[1])
    if isinstance(size, Sequence):
        if len(size) != 2:
            raise ValueError(f"Expected 2 numbers for --size, got {size!r}")
        return int(size[0]), int(size[1])
    raise TypeError(f"Unsupported --size value: {type(size)!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Motion masked autoencoder on FlyingThings")
    p.add_argument("--flyingthings-root", type=str, default="/home/spencer/Data/FlyingThings3D_tiny")
    p.add_argument("--split", type=str, default="train", help="train/val split passed to FlyingThingsSimpleDataset")
    p.add_argument("--reverse-flow", action="store_true", help="Use reversed flow direction")
    p.add_argument("--size", type=str, default="256", help="Resize flow to HxW (default 256)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--print-every", type=int, default=20)

    p.add_argument("--mask-ratio", type=float, default=0.9, help="Fraction to mask (e.g. 0.9 = 90% masked)")
    p.add_argument(
        "--mask-curriculum-epochs",
        type=int,
        default=5,
        help="Linearly ramp mask ratio from 0 to --mask-ratio over this many epochs.",
    )
    p.add_argument(
        "--mask-curriculum-start-epoch",
        type=int,
        default=0,
        help="Epoch index to begin masking ramp. Before this epoch, mask ratio is 0.",
    )
    p.add_argument(
        "--mask-modes",
        type=str,
        default="none,random_pixel,structured,pointodyssey",
        help="Comma-separated subset from {none,random_pixel,random_block,structured,pointodyssey}",
    )
    p.add_argument("--mask-block-jitter", type=float, default=0.15, help="Random jitter factor for block sizing")
    p.add_argument("--pointodyssey-mask-ratio-min", type=float, default=None, help="Min masked ratio for pointodyssey masks. If both min/max are set, sample per sample.")
    p.add_argument("--pointodyssey-mask-ratio-max", type=float, default=None, help="Max masked ratio for pointodyssey masks. If both min/max are set, sample per sample.")

    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--lambda-full", type=float, default=1.0, help="Weight on full-flow reconstruction")
    p.add_argument("--lambda-observed", type=float, default=0.25, help="Weight on observed sparse support recon.")
    p.add_argument("--lambda-tv", type=float, default=0.0, help="Optional flow smoothness loss weight")
    p.add_argument("--tv-eps", type=float, default=1e-3, help="Charbonnier-like epsilon for smoothness")

    p.add_argument("--max-train", type=int, default=0, help="Optional hard cap on train samples")
    p.add_argument("--max-val", type=int, default=0, help="Optional hard cap on val samples")
    p.add_argument("--output-dir", type=str, default="snapshots/flyingthings_motion_ae")
    p.add_argument("--exp-name", type=str, default="motion_ae_flyingthings")
    p.add_argument("--num-val-batches", type=int, default=8, help="Number of validation batches per epoch")
    p.add_argument("--no-tensorboard", action="store_true", help="Disable tensorboard logging")
    p.add_argument("--no-snapshots", action="store_true", help="Disable model checkpoints")
    p.add_argument(
        "--save-final-only",
        action="store_true",
        help="Only save one final checkpoint at the end (supersedes best/epoch checkpoints).",
    )
    p.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["none", "step", "cosine", "exponential", "plateau"],
        default="none",
        help="Learning-rate scheduler.",
    )
    p.add_argument("--lr-step-size", type=int, default=250, help="Step size for step LR scheduler")
    p.add_argument("--lr-gamma", type=float, default=0.5, help="Gamma for step/exponential LR schedulers")
    p.add_argument("--lr-eta-min", type=float, default=1e-6, help="Minimum LR for cosine scheduler")
    p.add_argument(
        "--lr-patience",
        type=int,
        default=20,
        help="Patience for ReduceLROnPlateau scheduler",
    )
    p.add_argument("--tb-logdir", type=str, default="", help="TensorBoard log dir (defaults to <output-dir>/<exp-name>/tb)")
    p.add_argument("--tb-log-images", type=int, default=4, help="Number of validation examples to log per epoch")
    p.add_argument("--tb-train-log-images", type=int, default=4, help="Number of training examples to log per epoch")
    return p


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _scale_flow(flow: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Resize dense flow and scale vectors for new spatial resolution."""
    _, h, w = flow.shape
    H, W = target_hw
    if (h, w) == (H, W):
        return flow
    scale_x = W / float(w)
    scale_y = H / float(h)
    flow_u = F.interpolate(flow[0:1, :, :].unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
    flow_v = F.interpolate(flow[1:2, :, :].unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
    flow = torch.cat([
        flow_u * scale_x,
        flow_v * scale_y,
    ], dim=0)
    return flow


def build_mask(
    flow: torch.Tensor,
    strategy: str,
    sparse_ratio: float,
    rng: random.Random,
    jitter: float = 0.15,
) -> torch.Tensor:
    """Build [H,W] mask where 1 = observed, 0 = masked."""
    _, H, W = flow.shape
    sparse_ratio = float(np.clip(sparse_ratio, 0.0, 1.0))
    obs_ratio = 1.0 - sparse_ratio
    if obs_ratio <= 0.0:
        return torch.zeros((H, W), dtype=torch.float32, device=flow.device)

    valid = torch.isfinite(flow).all(dim=0)
    mask = torch.zeros((H, W), dtype=torch.float32, device=flow.device)

    if strategy == "none":
        mask[valid] = 1.0
        return mask

    n_obs_target = int((obs_ratio * valid.float().sum()).item())
    n_obs_target = int(max(1, min(n_obs_target, int(valid.sum().item()))))

    if strategy in {"random_pixel", "pointwise", "bernoulli"}:
        selected = (torch.rand((H, W), device=flow.device) < obs_ratio) & valid
        mask[selected] = 1.0
        return mask

    if strategy == "pointodyssey":
        valid_cnt = max(1.0, float(valid.sum().item()))
        p = float(min(1.0, n_obs_target / valid_cnt))
        sampled = (torch.rand((H, W), device=flow.device) < p)
        mask[valid & sampled] = 1.0
        return mask

    if strategy == "random_block":
        # Start from a single random block and add small jitter around target coverage.
        target = float(obs_ratio)
        target = max(0.02, min(0.95, target))
        j = max(0.0, min(1.0, jitter))
        target *= float(rng.uniform(1.0 - j, 1.0 + j))
        target = max(0.02, min(0.95, target))
        area = int(target * H * W)
        aspect = rng.uniform(0.35, 2.8)
        bh = max(1, int(math.sqrt(area / aspect)))
        bw = max(1, int(area / max(1, bh)))
        bh = min(H, max(1, bh))
        bw = min(W, max(1, bw))
        y0 = rng.randint(0, max(0, H - bh))
        x0 = rng.randint(0, max(0, W - bw))
        mask[y0 : y0 + bh, x0 : x0 + bw] = 1.0
        return mask * valid.float()

    if strategy == "structured":
        stripe = max(2, int(max(2, min(H, W) * 0.03)))
        period = max(4, int(round(stripe * 4)))
        yy = torch.arange(H, device=flow.device).view(H, 1)
        xx = torch.arange(W, device=flow.device).view(1, W)
        block = ((yy // stripe + xx // stripe) % 2).float()
        if n_obs_target > H * W // 2:
            block = 1.0 - block
        mask[valid] = block[valid]
        # enforce exact sparse ratio only approximately, no hard requirement
        current = float(mask.sum().item())
        if current <= 1e-4:
            return mask
        return mask

    # fallback
    flat = torch.rand((H, W), device=flow.device)
    selected = (flat < obs_ratio) & valid
    mask[selected] = 1.0
    return mask


def scheduled_mask_ratio(args: argparse.Namespace, epoch: int) -> float:
    """Keep unmasked for initial epochs, then ramp mask ratio linearly to target."""
    target = float(np.clip(args.mask_ratio, 0.0, 1.0))
    start_epoch = max(0, int(args.mask_curriculum_start_epoch))
    ramp_epochs = max(0, int(args.mask_curriculum_epochs))

    if epoch < start_epoch or target <= 0.0:
        return 0.0
    if ramp_epochs <= 0 or epoch >= start_epoch + ramp_epochs:
        return target

    p = float(epoch - start_epoch) / float(ramp_epochs)
    p = float(np.clip(p, 0.0, 1.0))
    return target * p


def sample_sparse_ratio(args: argparse.Namespace, strategy: str, rng: random.Random, epoch: int) -> float:
    if strategy == "none":
        return 0.0

    target_ratio = scheduled_mask_ratio(args, epoch)
    base_ratio = float(np.clip(args.mask_ratio, 0.0, 1.0))

    if strategy != "pointodyssey":
        return target_ratio

    if args.pointodyssey_mask_ratio_min is None or args.pointodyssey_mask_ratio_max is None:
        return target_ratio

    lo = float(np.clip(min(args.pointodyssey_mask_ratio_min, args.pointodyssey_mask_ratio_max), 0.0, 1.0))
    hi = float(np.clip(max(args.pointodyssey_mask_ratio_min, args.pointodyssey_mask_ratio_max), 0.0, 1.0))
    if np.isclose(lo, hi):
        return target_ratio * (lo / max(1e-8, base_ratio))

    if base_ratio <= 0.0:
        return 0.0

    scale = target_ratio / base_ratio
    lo = float(np.clip(lo * scale, 0.0, 1.0))
    hi = float(np.clip(hi * scale, 0.0, 1.0))
    if np.isclose(lo, hi):
        return lo
    return float(rng.uniform(lo, hi))


def observed_ratio(mask: torch.Tensor, valid: torch.Tensor) -> float:
    valid = valid.to(dtype=mask.dtype)
    denom = valid.sum().clamp_min(1.0)
    return float((mask * valid).sum() / denom)


def flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
    """Map [2,H,W] flow tensor to [3,H,W] [0,1] RGB where B channel is 0."""
    flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    u, v = flow[0], flow[1]
    flat = torch.cat([u.reshape(-1), v.reshape(-1)])
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        scale = torch.tensor(1.0, device=flow.device, dtype=flow.dtype)
    else:
        scale = torch.quantile(finite.abs(), 0.99)
        if not torch.isfinite(scale) or scale.item() < 1e-6:
            scale = torch.tensor(1.0, device=flow.device, dtype=flow.dtype)
    rgb = torch.zeros((3, u.shape[0], u.shape[1]), device=flow.device, dtype=flow.dtype)
    rgb[0] = (0.5 + u / (2.0 * scale)).clamp(0.0, 1.0)
    rgb[1] = (0.5 + v / (2.0 * scale)).clamp(0.0, 1.0)
    return rgb


def flow_panel_figure(
    gt_flow: torch.Tensor,
    pred_flow: torch.Tensor,
    input_flow: torch.Tensor,
    obs_mask: torch.Tensor,
    *,
    split: str,
    mode: str,
    sparse_ratio: float,
    idx: int,
) -> plt.Figure:
    gt_np = flow_to_rgb(gt_flow).permute(1, 2, 0).detach().cpu().numpy()
    pred_np = flow_to_rgb(pred_flow).permute(1, 2, 0).detach().cpu().numpy()
    in_np = flow_to_rgb(input_flow).permute(1, 2, 0).detach().cpu().numpy()
    H, W = obs_mask.shape[-2:]
    mask_np = np.zeros((H, W, 3), dtype=np.float32)
    mask_np[..., 0] = obs_mask.detach().cpu().numpy().astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=130)
    panels = [
        ("Input (masked)", in_np),
        ("Ground Truth", gt_np),
        ("Prediction", pred_np),
        ("Observed mask", mask_np),
    ]

    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(name, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{split} sample {idx} | mode={mode} | masked={sparse_ratio:.2f}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def masked_charbonnier(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    # mask: [B,H,W] bool-like
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    m = mask.unsqueeze(1).to(dtype=pred.dtype)
    safe_target = torch.where(torch.isfinite(target), target, pred.detach())
    diff = torch.where(m > 0.0, pred - safe_target, torch.zeros_like(pred))
    err = charbonnier(diff.abs().sum(dim=1, keepdim=True), eps=eps)
    if bool((m > 0).sum()):
        return (err * m).sum() / m.sum().clamp_min(1.0)
    return pred.new_tensor(0.0)


def tv_smooth(flow: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return charbonnier(dx, eps).mean() + charbonnier(dy, eps).mean()


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False))


class MotionMaskedAutoEncoder(nn.Module):
    """Compact convolutional autoencoder with a global latent."""

    def __init__(self, in_channels: int = 3, base: int = 32, latent_dim: int = 128):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.to_latent = nn.Sequential(
            nn.Conv2d(base * 8, latent_dim, kernel_size=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_dim, base * 8, kernel_size=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )

        self.dec1 = UpBlock(base * 8, base * 4)
        self.dec2 = UpBlock(base * 4, base * 2)
        self.dec3 = UpBlock(base * 2, max(base, 1))
        self.out = nn.Conv2d(max(base, 1), 2, kernel_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        pooled = F.adaptive_avg_pool2d(e4, (1, 1)).flatten(1)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        z_map = self.from_latent(self.to_latent(e4))
        d = self.dec1(z_map)
        d = self.dec2(d)
        d = self.dec3(d)
        return self.out(F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False))


class FlyingThingsMaskedFlowDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        reverse_flow: bool = False,
        size: Optional[tuple[int, int]] = None,
    ):
        self.base = FlyingThingsSimpleDataset(root=root, split=split, reverse_flow=reverse_flow)
        self.size = size
        self.reverse_flow = bool(reverse_flow)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        flow = item["flow"].float()
        if self.size is not None:
            flow = _scale_flow(flow, self.size)
        valid_flow_mask = torch.isfinite(flow).all(dim=0)
        # For numeric stability in losses, clamp absurd values.
        flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "flow": flow,
            "valid_flow_mask": valid_flow_mask,
            "idx": torch.tensor(int(idx), dtype=torch.int64),
        }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    modes: list[str],
    args,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
    epoch: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total_epe = 0.0
    total = 0
    total_loss = 0.0
    total_ratio = 0.0
    ratio_count = 0
    vis_samples: list[tuple[int, plt.Figure]] = []
    rng = random.Random(args.seed + 77)
    current_epoch = 0 if epoch is None else epoch
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= args.num_val_batches:
                break
            flow = batch["flow"].to(device, non_blocking=True)
            valid = batch["valid_flow_mask"].to(device, non_blocking=True)
            B, _, H, W = flow.shape
            mode = modes[i % len(modes)]
            loss = torch.zeros((B,), device=flow.device, dtype=flow.dtype)
            epe = torch.zeros((B,), device=flow.device, dtype=flow.dtype)
            for b in range(B):
                sparse_ratio = sample_sparse_ratio(args=args, strategy=mode, rng=rng, epoch=current_epoch)
                fm = build_mask(
                    flow[b],
                    strategy=mode,
                    sparse_ratio=sparse_ratio,
                    rng=rng,
                    jitter=args.mask_block_jitter,
                )
                fm = fm.to(flow.device)
                sparse_ratio_actual = float(1.0 - observed_ratio(fm, valid[b]))
                input_flow = flow[b] * fm.unsqueeze(0)
                inp = torch.cat([input_flow, fm.unsqueeze(0)], dim=0).unsqueeze(0)
                pred = model(inp)
                pred = pred.squeeze(0)
                valid_b = valid[b].to(flow.device)
                l_full = masked_charbonnier(pred, flow[b], valid_b)
                l_obs = masked_charbonnier(pred, flow[b], fm * valid_b, eps=1e-3)
                err = (flow[b] - pred).pow(2).sum(dim=0).sqrt()
                err[~valid_b] = 0.0
                if valid_b.any():
                    epe[b] = err[valid_b].mean()
                else:
                    epe[b] = 0.0
                loss[b] = args.lambda_full * l_full + args.lambda_observed * l_obs
                total_ratio += sparse_ratio_actual
                ratio_count += 1
                if writer is not None and epoch is not None and len(vis_samples) < max(0, args.tb_log_images):
                    sample_idx = i * args.batch_size + b
                    panel = flow_panel_figure(
                        gt_flow=flow[b],
                        pred_flow=pred,
                        input_flow=input_flow,
                        obs_mask=fm,
                        split="val",
                        mode=mode,
                        sparse_ratio=sparse_ratio_actual,
                        idx=sample_idx,
                    )
                    vis_samples.append((sample_idx, panel))
            total_loss += float(loss.mean().item())
            total_epe += float(epe.mean().item())
            total += 1
    if writer is not None and epoch is not None and vis_samples:
        for sample_idx, fig in vis_samples:
            writer.add_figure(f"val/panels/idx_{sample_idx}", fig, global_step=epoch)
            plt.close(fig)
    if total == 0:
        return {"val_loss": float("nan"), "val_epe": float("nan"), "val_mask_ratio": float("nan")}
    avg_mask_ratio = total_ratio / max(1.0, float(ratio_count))
    return {
        "val_loss": total_loss / total,
        "val_epe": total_epe / total,
        "val_mask_ratio": avg_mask_ratio,
    }


def run_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    modes: list[str],
    args,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    model.train()
    running = {"loss": 0.0, "epe": 0.0}
    cnt = 0
    rng = random.Random(args.seed + epoch * 991)
    vis_samples: list[tuple[int, plt.Figure]] = []
    mask_sum = 0.0
    mask_count = 0

    for step, batch in enumerate(dataloader):
        flow = batch["flow"].to(device, non_blocking=True)
        valid = batch["valid_flow_mask"].to(device, non_blocking=True)
        B = flow.shape[0]

        optimizer.zero_grad(set_to_none=True)
        total_loss = flow.new_tensor(0.0)
        batch_epe = flow.new_tensor(0.0)

        for b in range(B):
            mode = modes[rng.randrange(len(modes))]
            sparse_ratio = sample_sparse_ratio(args=args, strategy=mode, rng=rng, epoch=epoch)
            obs = build_mask(flow[b], strategy=mode, sparse_ratio=sparse_ratio, rng=rng, jitter=args.mask_block_jitter).to(device)
            sparse_ratio_actual = float(1.0 - observed_ratio(obs, valid[b]))
            in_flow = flow[b] * obs.unsqueeze(0)
            model_in = torch.cat([in_flow, obs.unsqueeze(0)], dim=0).unsqueeze(0)
            pred = model(model_in)
            pred = pred.squeeze(0)

            l_full = masked_charbonnier(pred, flow[b], valid[b], eps=1e-3)
            l_obs = masked_charbonnier(pred, flow[b], obs * valid[b], eps=1e-3)
            if args.lambda_tv > 0:
                l_tv = tv_smooth(pred, eps=args.tv_eps)
            else:
                l_tv = flow.new_tensor(0.0)

            pred_e = (flow[b] - pred).pow(2).sum(dim=0).sqrt()
            pred_e[~valid[b]] = 0.0
            e = pred_e[valid[b]].mean() if bool(valid[b].any()) else flow.new_tensor(0.0)
            batch_epe = batch_epe + e
            total_loss = total_loss + (args.lambda_full * l_full + args.lambda_observed * l_obs + args.lambda_tv * l_tv)
            if writer is not None and len(vis_samples) < max(0, args.tb_train_log_images):
                panel = flow_panel_figure(
                    gt_flow=flow[b],
                    pred_flow=pred,
                    input_flow=in_flow,
                    obs_mask=obs,
                    split="train",
                    mode=mode,
                    sparse_ratio=sparse_ratio_actual,
                    idx=step * args.batch_size + b,
                )
                vis_samples.append((step * args.batch_size + b, panel))
            mask_sum += sparse_ratio_actual
            mask_count += 1

        total_loss = total_loss / float(B)
        batch_epe = batch_epe / float(B)
        total_loss.backward()
        optimizer.step()

        running["loss"] += float(total_loss.item())
        running["epe"] += float(batch_epe.item())
        cnt += 1
        if (step + 1) % args.print_every == 0:
            print(f"[train] epoch={epoch:03d} step={step+1:05d} loss={running['loss']/cnt:.4f} epe={running['epe']/cnt:.4f}")

    if writer is not None and vis_samples:
        for sample_idx, fig in vis_samples:
            writer.add_figure(f"train/panels/idx_{sample_idx}", fig, global_step=epoch)
            plt.close(fig)

    avg_mask = mask_sum / max(1.0, float(mask_count))
    return {
        "loss": running["loss"] / max(1, cnt),
        "epe": running["epe"] / max(1, cnt),
        "mask_ratio": avg_mask,
    }


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(args.seed)

    modes = [m.strip() for m in args.mask_modes.split(",") if m.strip()]
    valid_modes = {"none", "random_pixel", "random_block", "structured", "pointodyssey"}
    if not modes or not set(modes).issubset(valid_modes):
        raise ValueError(f"--mask-modes must be a comma-separated subset of {sorted(valid_modes)}")
    if (args.pointodyssey_mask_ratio_min is None) != (args.pointodyssey_mask_ratio_max is None):
        raise ValueError("Both --pointodyssey-mask-ratio-min and --pointodyssey-mask-ratio-max must be set together")
    if args.pointodyssey_mask_ratio_min is not None and args.pointodyssey_mask_ratio_max is not None:
        if args.pointodyssey_mask_ratio_min > args.pointodyssey_mask_ratio_max:
            args.pointodyssey_mask_ratio_min, args.pointodyssey_mask_ratio_max = (
                args.pointodyssey_mask_ratio_max,
                args.pointodyssey_mask_ratio_min,
            )

    target_size = _as_size(args.size)
    if target_size is None:
        target_size = (256, 256)

    full_set = FlyingThingsMaskedFlowDataset(
        root=args.flyingthings_root,
        split=args.split,
        reverse_flow=args.reverse_flow,
        size=target_size,
    )
    n = len(full_set)
    if n < 2:
        raise RuntimeError(f"FlyingThings split '{args.split}' appears empty at {args.flyingthings_root}")

    val_start = int(0.9 * n)
    train_idx = list(range(0, val_start))
    val_idx = list(range(val_start, n))
    if args.max_train and args.max_train > 0:
        train_idx = train_idx[: args.max_train]
    if args.max_val and args.max_val > 0:
        val_idx = val_idx[: args.max_val]

    g = torch.Generator().manual_seed(args.seed)
    train_set = torch.utils.data.Subset(full_set, train_idx)
    val_set = torch.utils.data.Subset(full_set, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=g,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    model = MotionMaskedAutoEncoder(
        in_channels=3,
        base=args.base_channels,
        latent_dim=args.latent_dim,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, args.lr_step_size),
            gamma=args.lr_gamma,
        )
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.lr_eta_min,
        )
    elif args.lr_scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_gamma,
            patience=args.lr_patience,
        )

    out_dir = Path(args.output_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_writer: Optional[SummaryWriter] = None
    if not args.no_tensorboard:
        tb_root = Path(args.tb_logdir) if args.tb_logdir else (out_dir / "tb")
        tb_root.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_root))
    log_path = out_dir / "train_log.jsonl"

    best = float("inf")
    train_log: list[dict] = []
    for epoch in range(args.epochs):
        train_stats = run_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            modes=modes,
            args=args,
            device=device,
            epoch=epoch,
            writer=tb_writer,
        )
        val_stats = evaluate(
            model=model,
            dataloader=val_loader,
            modes=modes,
            args=args,
            device=device,
            writer=tb_writer,
            epoch=epoch,
        )
        merged = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_epe": train_stats["epe"],
            **{f"{k}": v for k, v in val_stats.items()},
        }
        train_log.append(merged)
        log_path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in train_log) + "\n")

        if not args.no_snapshots and not args.save_final_only and val_stats["val_loss"] < best:
            best = val_stats["val_loss"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "args": vars(args)}, out_dir / "best.pt")
        if tb_writer is not None:
            tb_writer.add_scalar("loss/train", train_stats["loss"], epoch)
            tb_writer.add_scalar("loss/val", val_stats["val_loss"], epoch)
            tb_writer.add_scalar("epe/train", train_stats["epe"], epoch)
            tb_writer.add_scalar("epe/val", val_stats["val_epe"], epoch)
            tb_writer.add_scalar("mask/ratio_train", train_stats["mask_ratio"], epoch)
            tb_writer.add_scalar("mask/ratio", val_stats["val_mask_ratio"], epoch)
            tb_writer.add_scalar("lr/current", optimizer.param_groups[0]["lr"], epoch)
            tb_writer.flush()

        print(
            f"[epoch {epoch:03d}] train_loss={train_stats['loss']:.4f} "
            f"train_epe={train_stats['epe']:.4f} val_loss={val_stats['val_loss']:.4f} val_epe={val_stats['val_epe']:.4f}"
        )
        if not args.no_snapshots and not args.save_final_only:
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "args": vars(args)}, out_dir / f"epoch_{epoch:03d}.pt")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_stats["val_loss"])
            else:
                scheduler.step()

    if tb_writer is not None:
        tb_writer.close()

    if not args.no_snapshots and args.save_final_only:
        torch.save(
            {"model_state": model.state_dict(), "epoch": args.epochs - 1, "args": vars(args)},
            out_dir / "final.pt",
        )


if __name__ == "__main__":
    main()
