from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .model import FlowMAEModelConfig, FlowMaskedAutoencoderViT

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class FlowMAELightningModule(pl.LightningModule):
    def __init__(self, model_config: FlowMAEModelConfig, training_config: dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "model": vars(model_config),
                "training": training_config,
            }
        )
        self.model = FlowMaskedAutoencoderViT(model_config)
        self.training_config = training_config
        self.example_batch = None

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        return self.model(
            src_rgb=batch["src_rgb"],
            tgt_rgb=batch["tgt_rgb"],
            flow=batch["flow"],
            valid=batch["valid"],
        )

    @staticmethod
    def flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
        flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0).float()
        u, v = flow[0], flow[1]
        finite = torch.cat([u.reshape(-1), v.reshape(-1)])
        finite = finite[torch.isfinite(finite)]
        if finite.numel() == 0:
            scale = torch.tensor(1.0, device=flow.device, dtype=flow.dtype)
        else:
            scale = torch.quantile(finite.abs(), 0.99)
            if not torch.isfinite(scale) or scale.item() < 1e-6:
                scale = torch.tensor(1.0, device=flow.device, dtype=flow.dtype)
        rgb = torch.zeros((3, flow.shape[-2], flow.shape[-1]), device=flow.device, dtype=flow.dtype)
        rgb[0] = (0.5 + u / (2.0 * scale)).clamp(0.0, 1.0)
        rgb[1] = (0.5 + v / (2.0 * scale)).clamp(0.0, 1.0)
        return rgb

    @staticmethod
    def denormalize_rgb(image: torch.Tensor) -> torch.Tensor:
        mean = IMAGENET_MEAN.to(device=image.device, dtype=image.dtype)
        std = IMAGENET_STD.to(device=image.device, dtype=image.dtype)
        return (image * std + mean).clamp(0.0, 1.0)

    @staticmethod
    def endpoint_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe_target = torch.where(mask.unsqueeze(1) > 0, target, pred.detach())
        epe = torch.linalg.vector_norm(pred - safe_target, dim=1)
        denom = mask.sum().clamp_min(1.0)
        return (epe * mask).sum() / denom

    @staticmethod
    def _pixel_space_flow(flow: torch.Tensor, flow_scale: torch.Tensor | None) -> torch.Tensor:
        if flow_scale is None:
            return flow
        return flow * flow_scale.view(-1, 1, 1, 1).to(device=flow.device, dtype=flow.dtype)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        outputs = self(batch)
        pred_flow = outputs["pred_flow"]
        flow_scale = batch.get("flow_scale")
        valid = batch["valid"]
        masked_pixels = outputs["masked_pixels"] * valid
        observed_pixels = outputs["observed_pixels"] * valid
        pred_flow_px = self._pixel_space_flow(pred_flow, flow_scale)
        target_flow_px = self._pixel_space_flow(batch["flow"], flow_scale)
        input_flow_px = self._pixel_space_flow(outputs["flow_input"], flow_scale)

        if not torch.isfinite(pred_flow).all() or not torch.isfinite(outputs["loss"]):
            pred_finite = torch.isfinite(pred_flow)
            raise RuntimeError(
                f"Non-finite tensors in {stage} step: "
                f"loss={float(outputs['loss'].detach().cpu()) if torch.numel(outputs['loss']) else 'nan'}, "
                f"pred_finite_frac={float(pred_finite.float().mean().detach().cpu()):.6f}, "
                f"input_abs_max={float(input_flow_px.abs().max().detach().cpu()):.4f}, "
                f"target_abs_max={float(target_flow_px.abs().max().detach().cpu()):.4f}"
            )

        epe = self.endpoint_error(pred_flow_px, target_flow_px, valid)
        masked_epe = self.endpoint_error(pred_flow_px, target_flow_px, masked_pixels)
        observed_epe = self.endpoint_error(pred_flow_px, target_flow_px, observed_pixels)
        mask_ratio = masked_pixels.sum() / valid.sum().clamp_min(1.0)
        latent_norm = outputs["encoded_tokens"].norm(dim=-1).mean()
        pred_abs_max = pred_flow_px.abs().max()
        target_abs_max = target_flow_px.abs().max()
        input_abs_max = input_flow_px.abs().max()

        self.log(f"{stage}/loss", outputs["loss"], prog_bar=stage == "train", on_step=stage == "train", on_epoch=True, batch_size=batch["flow"].shape[0])
        self.log(f"{stage}_loss", outputs["loss"], prog_bar=False, on_step=False, on_epoch=True, batch_size=batch["flow"].shape[0])
        self.log(f"{stage}/epe", epe, prog_bar=stage != "train", on_step=False, on_epoch=True, batch_size=batch["flow"].shape[0])
        self.log(f"{stage}_epe", epe, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch["flow"].shape[0])
        self.log(f"{stage}/masked_epe", masked_epe, on_step=False, on_epoch=True, batch_size=batch["flow"].shape[0])
        self.log(f"{stage}/observed_epe", observed_epe, on_step=False, on_epoch=True, batch_size=batch["flow"].shape[0])
        self.log(f"{stage}/mask_ratio", mask_ratio, on_step=False, on_epoch=True, batch_size=batch["flow"].shape[0])
        self.log(f"{stage}/latent_norm", latent_norm, on_step=False, on_epoch=True, batch_size=batch["flow"].shape[0])
        self.log(f"{stage}/pred_abs_max", pred_abs_max, on_step=False, on_epoch=True, batch_size=batch["flow"].shape[0])
        self.log(f"{stage}/target_abs_max", target_abs_max, on_step=False, on_epoch=True, batch_size=batch["flow"].shape[0])
        self.log(f"{stage}/input_abs_max", input_abs_max, on_step=False, on_epoch=True, batch_size=batch["flow"].shape[0])

        if stage == "val" and self.example_batch is None:
            self.example_batch = {
                "src_rgb": batch["src_rgb"].detach().cpu(),
                "tgt_rgb": batch["tgt_rgb"].detach().cpu(),
                "flow": target_flow_px.detach().cpu(),
                "valid": batch["valid"].detach().cpu(),
                "pred_flow": pred_flow_px.detach().cpu(),
                "flow_input": input_flow_px.detach().cpu(),
                "observed_valid": outputs["observed_valid"].detach().cpu(),
            }

        return outputs["loss"]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def on_validation_epoch_start(self) -> None:
        self.example_batch = None

    def on_validation_epoch_end(self) -> None:
        if self.example_batch is None or self.logger is None:
            return
        max_images = int(self.training_config.get("tb_log_images", 4))
        if max_images <= 0:
            return
        figures = []
        count = min(max_images, self.example_batch["flow"].shape[0])
        for idx in range(count):
            fig = self._build_figure(idx)
            figures.append(fig)
            self.logger.experiment.add_figure(f"val/examples_{idx}", fig, global_step=self.current_epoch)
        for fig in figures:
            plt.close(fig)

    def _build_figure(self, idx: int) -> plt.Figure:
        src_rgb = self.denormalize_rgb(self.example_batch["src_rgb"][idx])
        tgt_rgb = self.denormalize_rgb(self.example_batch["tgt_rgb"][idx])
        input_rgb = self.flow_to_rgb(self.example_batch["flow_input"][idx])
        gt_rgb = self.flow_to_rgb(self.example_batch["flow"][idx])
        pred_rgb = self.flow_to_rgb(self.example_batch["pred_flow"][idx])
        valid_rgb = self.example_batch["observed_valid"][idx].unsqueeze(0).repeat(3, 1, 1)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=130)
        panels = [
            ("Source RGB", src_rgb),
            ("Target RGB", tgt_rgb),
            ("Observed Flow", input_rgb),
            ("Ground Truth", gt_rgb),
            ("Prediction", pred_rgb),
            ("Observed Mask", valid_rgb),
        ]
        for ax, (title, image) in zip(axes.flat, panels):
            ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Flow-only reconstruction on valid pixels", fontsize=10)
        fig.tight_layout()
        return fig

    def configure_optimizers(self) -> dict[str, Any]:
        lr = float(self.training_config.get("lr", 3e-4))
        weight_decay = float(self.training_config.get("weight_decay", 0.05))
        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        warmup_epochs = int(self.training_config.get("warmup_epochs", 0))
        max_epochs = int(self.training_config.get("max_epochs", 1))
        min_lr = float(self.training_config.get("min_lr", 1e-6))
        if warmup_epochs > 0 and max_epochs > warmup_epochs:
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=min_lr)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs), eta_min=min_lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
