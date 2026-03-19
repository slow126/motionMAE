"""Minimal end-to-end smoke test training script for p(flow | image, dt)."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.flow_smoke.dataset import (
    PointOdysseyFlowSmokeDataset,
    load_manifest,
    split_manifest_indices_by_clip,
)
from src.flow_smoke.models import ConditionalFlowVAE, DeterministicUNet


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_dt_values(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return sorted({int(x) for x in raw})
    values: List[int] = []
    for token in str(raw).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return sorted(set(values))


def _subsample_indices(indices: Sequence[int], max_items: int, seed: int) -> List[int]:
    if max_items <= 0 or max_items >= len(indices):
        return list(indices)
    rng = random.Random(seed)
    return sorted(rng.sample(list(indices), int(max_items)))


def _prune_empty_pairs(dataset: PointOdysseyFlowSmokeDataset, min_n_pts: int, split_name: str) -> int:
    if min_n_pts <= 0:
        return 0
    kept = []
    for local_idx in range(len(dataset)):
        sample = dataset[local_idx]
        if int(sample["n_pts"].item()) >= min_n_pts:
            kept.append(dataset.manifest_indices[local_idx])
    removed = len(dataset.manifest_indices) - len(kept)
    if not kept:
        raise RuntimeError(f"No {split_name} samples pass min_n_pts={min_n_pts}.")
    dataset.manifest_indices = kept
    return removed


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask is None:
        raise ValueError("mask cannot be None")
    while mask.dim() > 4 and mask.size(1) == 1:
        mask = mask[:, 0]
    if mask.dim() == 2:
        return mask.unsqueeze(0).unsqueeze(1)
    if mask.dim() == 3:
        return mask.unsqueeze(1)
    if mask.dim() == 5 and mask.size(1) == 1:
        return mask[:, 0]
    return mask


def _charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    if valid_mask is None or not bool(valid_mask.any()):
        return pred[:, :, :0, :0].sum()
    valid_mask = _normalize_mask(valid_mask).to(dtype=torch.bool)
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    finite_target = torch.isfinite(target).all(dim=1, keepdim=True)
    finite_pred = torch.isfinite(pred).all(dim=1, keepdim=True)
    valid_mask = valid_mask & finite_target & finite_pred
    if not bool(valid_mask.any()):
        return pred[:, :, :0, :0].sum()
    err = torch.sqrt((pred - target).pow(2).sum(dim=1) + eps * eps)
    return err[valid_mask[:, 0]].mean()


def _epe(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> Optional[torch.Tensor]:
    if valid_mask is None or not bool(valid_mask.any()):
        return None
    valid_mask = _normalize_mask(valid_mask).to(dtype=torch.bool)
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    finite_target = torch.isfinite(target).all(dim=1)
    finite_pred = torch.isfinite(pred).all(dim=1)
    valid_mask = valid_mask[:, 0] & finite_target & finite_pred
    if not bool(valid_mask.any()):
        return None
    err = torch.sqrt((pred - target).pow(2).sum(dim=1))
    return err[valid_mask].mean()


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)


def _flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Convert [2,H,W] flow to RGB image for quick sanity checks."""
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"flow must be [2,H,W], got {flow.shape}")
    flow = flow.astype(np.float32, copy=False)
    valid = np.isfinite(flow[0]) & np.isfinite(flow[1])
    u = flow[0]
    v = flow[1]
    u[~valid] = 0.0
    v[~valid] = 0.0
    mag = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)
    hue = ((ang + np.pi) / (2.0 * np.pi)) * 179.0
    mag_scale = float(np.percentile(mag[valid], 98)) if bool(valid.any()) else 1.0
    if mag_scale <= 1e-8:
        mag_scale = 1e-8
    sat = np.ones_like(hue) * 255.0
    val = np.clip(mag / mag_scale, 0.0, 1.0) * 255.0
    hsv = np.stack([hue, sat, val], axis=-1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[~valid] = 0
    return rgb


def _save_vis_grid(
    vis_dir: Path,
    epoch: int,
    step_name: str,
    src_img: torch.Tensor,
    gt_flow: torch.Tensor,
    pred_flow: torch.Tensor,
    pred_samples: Optional[torch.Tensor] = None,
    limit: int = 3,
) -> None:
    vis_dir.mkdir(parents=True, exist_ok=True)
    src_np = src_img[:limit].detach().cpu().numpy()
    gt_np = gt_flow[:limit].detach().cpu().numpy()
    pred_np = pred_flow[:limit].detach().cpu().numpy()
    for idx in range(src_np.shape[0]):
        img = src_np[idx]
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8).transpose(1, 2, 0)

        gt_rgb = _flow_to_rgb(gt_np[idx])
        pred_rgb = _flow_to_rgb(pred_np[idx])
        canvas = np.concatenate([img, gt_rgb, pred_rgb], axis=1)
        out_path = vis_dir / f"{step_name}_e{epoch:03d}_i{idx}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        if pred_samples is not None and pred_samples.ndim == 5:
            # Save first sample in the stack as a coarse diversity proxy.
            sample_np = pred_samples[:, idx].detach().cpu().numpy()
            sample_rgb = _flow_to_rgb(sample_np[0])
            sample_path = vis_dir / f"{step_name}_e{epoch:03d}_i{idx}_sample0.png"
            cv2.imwrite(str(sample_path), cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2BGR))


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    beta: float,
    model_type: str,
) -> Dict[str, float]:
    model.train()
    num_loss_batches = 0
    sum_loss = 0.0
    sum_recon = 0.0
    sum_kl = 0.0
    sum_epe = 0.0
    sum_epe_count = 0

    for batch in dataloader:
        batch = _to_device(batch, device)
        src_img = batch["src_img"]
        flow = batch["flow"]
        dt = batch["dt"]
        valid = batch.get("valid_flow_mask", torch.isfinite(flow).all(dim=1))
        valid = _normalize_mask(valid)

        optimizer.zero_grad(set_to_none=True)

        if model_type == "det":
            pred = model(src_img, dt)
            recon = _charbonnier_loss(pred, flow, valid)
            loss = recon
            kl = torch.tensor(0.0, device=flow.device, dtype=flow.dtype)
        else:
            vae_out = model(src_img, dt, flow_gt=flow)
            pred = vae_out.flow
            if pred.dim() == 5:
                pred = pred[0]
            recon = _charbonnier_loss(pred, flow, valid)
            mu = vae_out.mu
            logvar = vae_out.logvar
            assert mu is not None and logvar is not None
            kl = _kl_divergence(mu, logvar)
            loss = recon + beta * kl

        loss.backward()
        optimizer.step()

        num_loss_batches += 1
        sum_loss += float(loss.detach().item())
        sum_recon += float(recon.detach().item())
        sum_kl += float(kl.detach().item())
        with torch.no_grad():
            epe = _epe(pred, flow, valid)
            if epe is not None and torch.isfinite(epe):
                sum_epe += float(epe.item())
                sum_epe_count += 1

    denom = max(1, num_loss_batches)
    denom_epe = max(1, sum_epe_count)
    return {
        "epoch": int(epoch),
        "train/loss": sum_loss / denom,
        "train/recon": sum_recon / denom,
        "train/kl": sum_kl / denom,
        "train/epe": sum_epe / denom_epe,
    }


@torch.no_grad()
def evaluate_epoch(
    model,
    dataloader: DataLoader,
    device: torch.device,
    model_type: str,
    k_samples: int = 8,
) -> Dict[str, float]:
    model.eval()
    total_epe = 0.0
    total_epe_count = 0
    best_epe_total = 0.0
    best_epe_count = 0
    mean_sample_epe_total = 0.0
    mean_sample_count = 0
    diversity_total = 0.0
    diversity_count = 0

    for batch in dataloader:
        batch = _to_device(batch, device)
        src_img = batch["src_img"]
        flow = batch["flow"]
        dt = batch["dt"]
        valid = batch.get("valid_flow_mask", torch.isfinite(flow).all(dim=1))
        valid = _normalize_mask(valid)

        if model_type == "det":
            pred = model(src_img, dt)
            epe = _epe(pred, flow, valid)
            if epe is not None and torch.isfinite(epe):
                total_epe += float(epe.item())
                total_epe_count += 1
            continue

        vae_out = model(src_img, dt, flow_gt=flow, n_samples=k_samples)
        pred_samples = vae_out.flow
        if pred_samples is None:
            pred_samples = model(src_img, dt, n_samples=k_samples).flow

        if pred_samples.dim() != 5:
            pred_samples = pred_samples.unsqueeze(0)

        batch_epe = []
        for sample in pred_samples:
            e = _epe(sample, flow, valid)
            if e is not None and torch.isfinite(e):
                batch_epe.append(e)
        if not batch_epe:
            continue
        batch_epe_t = torch.stack(batch_epe)
        min_epe = torch.min(batch_epe_t).item()
        mean_epe = torch.mean(batch_epe_t).item()

        mean_pred = pred_samples.mean(dim=0)
        mean_pred_epe = _epe(mean_pred, flow, valid)
        if mean_pred_epe is None:
            continue

        flow_var = pred_samples.var(dim=0, unbiased=False).sum(dim=1).sqrt()
        finite_var = flow_var[valid[:, 0]].mean()
        if torch.isfinite(finite_var):
            diversity_total += float(finite_var.item())
            diversity_count += 1

        best_epe_total += float(min_epe)
        best_epe_count += 1
        mean_sample_epe_total += float(mean_pred_epe.item())
        mean_sample_count += 1
        total_epe += float(mean_pred_epe.item())
        total_epe_count += 1

    return {
        "val/epe": total_epe / max(1, total_epe_count),
        "val/best_of_k_epe": best_epe_total / max(1, best_epe_count),
        "val/sample_mean_epe": mean_sample_epe_total / max(1, mean_sample_count),
        "val/diversity": diversity_total / max(1, diversity_count),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Point Odyssey flow smoke test")
    parser.add_argument("--manifest-path", type=str, required=True, help="JSONL pair manifest path")
    parser.add_argument("--pointodyssey-root", type=str, required=True, help="Point Odyssey root dir")
    parser.add_argument("--dt-values", type=str, default="1,2,3,4", help="Comma-separated dt list")
    parser.add_argument("--trust-manifest", action="store_true", help="Use manifest validity fields and skip visibility filtering in annotation loading")
    parser.add_argument("--min-valid-points", type=int, default=1, help="Skip manifest rows with fewer valid points than this")
    parser.add_argument("--prune-empty", action="store_true", help="Materialize data once and drop pairs with n_pts below min-valid-points")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation clip fraction")
    parser.add_argument("--size", type=int, default=256, help="Resize H=W target size")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--model", choices=["det", "vae"], default="det")
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--beta-max", type=float, default=1e-3)
    parser.add_argument("--beta-warmup-epochs", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--k-samples", type=int, default=8, help="VAE best-of-K at validation")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional capped train pairs")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Optional capped val pairs")
    parser.add_argument("--max-points-per-pair", type=int, default=0, help="Optional dense-flow sparsification cap")
    parser.add_argument("--use-grayscale", action="store_true", help="Train with grayscale image input")
    parser.add_argument("--output-dir", type=str, default="snapshots/flow_smoke")
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--vis-dir", type=str, default="", help="Optional visualization directory")
    parser.add_argument("--vis-every", type=int, default=0, help="Save vis every N epochs if >0")
    parser.add_argument("--vis-limit", type=int, default=3, help="Images per visualization")
    return parser


def _beta_schedule(epoch: int, beta_max: float, warmup_epochs: int) -> float:
    if beta_max <= 0:
        return 0.0
    if warmup_epochs <= 0:
        return beta_max
    return beta_max * min(1.0, (epoch + 1) / float(max(1, warmup_epochs)))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _set_seed(args.seed)

    dt_values = _parse_dt_values(args.dt_values)
    if not dt_values:
        raise ValueError("--dt-values must include at least one integer")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_name = args.exp_name.strip() or f"{args.model}_dt_{'_'.join(map(str, dt_values))}"
    ckpt_dir = output_dir / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = ckpt_dir / "train_log.jsonl"
    train_log: List[dict] = []

    entries = load_manifest(args.manifest_path)
    train_indices, val_indices = split_manifest_indices_by_clip(entries, args.val_fraction, args.seed)
    if args.max_train_samples > 0:
        train_indices = _subsample_indices(train_indices, args.max_train_samples, args.seed + 1)
    if args.max_val_samples > 0:
        val_indices = _subsample_indices(val_indices, args.max_val_samples, args.seed + 2)

    if not train_indices:
        raise RuntimeError("No train indices found for requested split/settings.")
    if not val_indices:
        raise RuntimeError("No val indices found for requested split/settings.")

    in_channels = 1 if args.use_grayscale else 3
    train_ds = PointOdysseyFlowSmokeDataset(
        manifest_path=args.manifest_path,
        indices=train_indices,
        dt_values=dt_values,
        pointodyssey_root=args.pointodyssey_root,
        size=(args.size, args.size),
        max_points_per_pair=(None if args.max_points_per_pair <= 0 else args.max_points_per_pair),
        use_grayscale=args.use_grayscale,
        trust_manifest=args.trust_manifest,
        min_valid_points=args.min_valid_points,
    )
    val_ds = PointOdysseyFlowSmokeDataset(
        manifest_path=args.manifest_path,
        indices=val_indices,
        dt_values=dt_values,
        pointodyssey_root=args.pointodyssey_root,
        size=(args.size, args.size),
        max_points_per_pair=(None if args.max_points_per_pair <= 0 else args.max_points_per_pair),
        use_grayscale=args.use_grayscale,
        trust_manifest=args.trust_manifest,
        min_valid_points=args.min_valid_points,
    )

    removed_train = removed_val = 0
    if args.prune_empty:
        removed_train = _prune_empty_pairs(train_ds, args.min_valid_points, "train")
        removed_val = _prune_empty_pairs(val_ds, args.min_valid_points, "val")

    print(
        f"[data] train_pairs={len(train_ds)} (removed {removed_train} empty by n_pts) "
        f"val_pairs={len(val_ds)} (removed {removed_val} empty by n_pts)"
    )

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    if args.model == "det":
        model = DeterministicUNet(in_channels=in_channels, base_channels=args.base_channels)
    else:
        model = ConditionalFlowVAE(
            in_channels=in_channels,
            base_channels=args.base_channels,
            z_dim=args.z_dim,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_epe = math.inf
    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        beta = _beta_schedule(epoch, args.beta_max, args.beta_warmup_epochs) if args.model == "vae" else 0.0
        train_stats = train_one_epoch(model, train_loader, optimizer, device, epoch, beta, args.model)
        metrics = dict(train_stats)
        print(
            f"[train] epoch={epoch:03d} model={args.model} "
            f"loss={train_stats['train/loss']:.6f} recon={train_stats['train/recon']:.6f} "
            f"kl={train_stats['train/kl']:.6f} epe={train_stats['train/epe']:.6f} beta={beta:.6f}"
        )

        if (epoch + 1) % args.eval_every == 0:
            val_stats = evaluate_epoch(model, val_loader, device, args.model, args.k_samples)
            metrics.update(val_stats)
            if "val/epe" in val_stats and val_stats["val/epe"] < best_val_epe:
                best_val_epe = float(val_stats["val/epe"])
                checkpoint = {
                    "epoch": int(epoch),
                    "model": args.model,
                    "state_dict": model.state_dict(),
                }
                torch.save(checkpoint, ckpt_dir / "best.pt")
            print(
                f"[val] epoch={epoch:03d} epe={val_stats.get('val/epe', float('nan')):.6f} "
                f"best_k={val_stats.get('val/best_of_k_epe', float('nan')):.6f} "
                f"sample={val_stats.get('val/sample_mean_epe', float('nan')):.6f} "
                f"div={val_stats.get('val/diversity', float('nan')):.6f}"
            )

        checkpoint = {"epoch": int(epoch), "model": args.model, "state_dict": model.state_dict(), "args": vars(args)}
        torch.save(checkpoint, ckpt_dir / f"epoch_{epoch:03d}.pt")
        train_log.append(metrics)
        log_path.write_text("\n".join(json.dumps(entry, sort_keys=True) for entry in train_log) + "\n")

        if vis_dir is not None and args.vis_every > 0 and (epoch + 1) % args.vis_every == 0:
            with torch.no_grad():
                val_batch = next(iter(val_loader))
                val_batch = _to_device(val_batch, device)
                src_img = val_batch["src_img"]
                dt = val_batch["dt"]
                gt_flow = val_batch["flow"]
                if args.model == "det":
                    pred_flow = model(src_img, dt)
                    _save_vis_grid(vis_dir, epoch, args.model, src_img, gt_flow, pred_flow, limit=args.vis_limit)
                else:
                    vae_eval = model(src_img, dt, n_samples=max(1, min(4, args.k_samples)))
                    pred_samples = vae_eval.flow
                    if pred_samples.dim() == 5:
                        pred_mean = pred_samples.mean(dim=0)
                    else:
                        pred_mean = pred_samples
                        pred_samples = None
                    _save_vis_grid(
                        vis_dir,
                        epoch,
                        args.model,
                        src_img,
                        gt_flow,
                        pred_mean,
                        pred_samples=pred_samples,
                        limit=args.vis_limit,
                    )

    print(f"Done. Outputs in: {ckpt_dir}")


if __name__ == "__main__":
    main()
