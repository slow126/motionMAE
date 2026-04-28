from __future__ import annotations

import os
from typing import Any, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from percieverIO_Exp.model import GaussianLatentReg, VariableFlowConfig, VariableFlowPerceiverIO


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class VariableFlowLightningModule(pl.LightningModule):
    def __init__(self, model_config: VariableFlowConfig, training_config: dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters({"model": model_config, "training": training_config})
        self.model = VariableFlowPerceiverIO(model_config)
        self.training_config = training_config
        self.gaussian_reg = GaussianLatentReg()
        self.example_batch: Optional[dict[str, Any]] = None

    @staticmethod
    def charbonnier(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        diff = pred - target
        err = torch.sqrt(diff.pow(2).sum(dim=-1) + eps**2)
        denom = mask.sum().clamp_min(1.0)
        return (err * mask).sum() / denom

    @staticmethod
    def endpoint_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        epe = torch.linalg.vector_norm(pred - target, dim=-1)
        denom = mask.sum().clamp_min(1.0)
        return (epe * mask).sum() / denom

    @staticmethod
    def denormalize_rgb(image: torch.Tensor) -> torch.Tensor:
        mean = IMAGENET_MEAN.to(image.device, image.dtype)
        std = IMAGENET_STD.to(image.device, image.dtype)
        return (image * std + mean).clamp(0.0, 1.0)

    @staticmethod
    def flow_to_rgb(flow_hw2: torch.Tensor) -> torch.Tensor:
        flow = torch.nan_to_num(flow_hw2, nan=0.0, posinf=0.0, neginf=0.0).float()
        u = flow[..., 0]
        v = flow[..., 1]
        finite = torch.cat([u.reshape(-1), v.reshape(-1)])
        finite = finite[torch.isfinite(finite)]
        scale = torch.tensor(1.0, device=flow.device)
        if finite.numel() > 0:
            candidate = torch.quantile(finite.abs(), 0.99)
            if torch.isfinite(candidate) and candidate.item() > 1e-6:
                scale = candidate
        rgb = torch.zeros((3, flow.shape[0], flow.shape[1]), device=flow.device, dtype=flow.dtype)
        rgb[0] = (0.5 + u / (2.0 * scale)).clamp(0.0, 1.0)
        rgb[1] = (0.5 + v / (2.0 * scale)).clamp(0.0, 1.0)
        return rgb

    @staticmethod
    def q_to_hw(flow_q: torch.Tensor, image_size: tuple[int, int], stride: int) -> torch.Tensor:
        b, q, _ = flow_q.shape
        hq = image_size[0] // stride
        wq = image_size[1] // stride
        return flow_q.reshape(b, hq, wq, 2)

    @staticmethod
    def valid_q_to_hw(valid_q: torch.Tensor, image_size: tuple[int, int], stride: int) -> torch.Tensor:
        hq = image_size[0] // stride
        wq = image_size[1] // stride
        return valid_q.reshape(valid_q.shape[0], hq, wq)

    def _run_view(self, view: dict[str, Any], query_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(view["tokens"], query_inputs, pad_mask=view["pad_mask"])

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        phase = int(batch["phase"])
        out_a = self._run_view(batch["view_a"], batch["query_inputs"])
        losses = {}
        recon_a = self.charbonnier(out_a["pred_flow"], batch["target_flow_q"], batch["target_valid_q"])
        losses["recon_a"] = recon_a
        pred_for_metrics = out_a["pred_flow"]

        if phase in (3, 4):
            out_b = self._run_view(batch["view_b"], batch["query_inputs"])
            recon_b = self.charbonnier(out_b["pred_flow"], batch["target_flow_q"], batch["target_valid_q"])
            losses["recon_b"] = recon_b
            losses["alignment"] = torch.mean((out_a["z_content"] - out_b["z_content"]) ** 2)
            pred_for_metrics = 0.5 * (out_a["pred_flow"] + out_b["pred_flow"])
            if phase == 4:
                losses["gaussian"] = self.gaussian_reg(torch.cat([out_a["latents"], out_b["latents"]], dim=0))

        loss = losses["recon_a"]
        if "recon_b" in losses:
            loss = 0.5 * (loss + losses["recon_b"])
        if "alignment" in losses:
            loss = loss + float(self.training_config.get("alignment_weight", 1.0)) * losses["alignment"]
        if "gaussian" in losses:
            loss = loss + float(self.training_config.get("gaussian_weight", 1e-3)) * losses["gaussian"]

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss in {stage} step.")

        epe = self.endpoint_error(pred_for_metrics, batch["target_flow_q"], batch["target_valid_q"])
        self.log(f"{stage}/loss", loss, prog_bar=stage != "train", on_step=stage == "train", on_epoch=True, batch_size=batch["image1"].shape[0])
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, batch_size=batch["image1"].shape[0])
        self.log(f"{stage}/epe", epe, prog_bar=stage != "train", on_step=False, on_epoch=True, batch_size=batch["image1"].shape[0])
        self.log(f"{stage}_epe", epe, on_step=False, on_epoch=True, batch_size=batch["image1"].shape[0])
        for name, value in losses.items():
            self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True, batch_size=batch["image1"].shape[0])

        if stage == "val" and self.example_batch is None:
            cached = {
                "phase": phase,
                "image1": batch["image1"].detach().cpu(),
                "image2": batch["image2"].detach().cpu(),
                "target_flow_q": batch["target_flow_q"].detach().cpu(),
                "target_valid_q": batch["target_valid_q"].detach().cpu(),
                "pred_a": out_a["pred_flow"].detach().cpu(),
                "obs_a": batch["view_a"]["observed_mask"].detach().cpu(),
                "image_size": tuple(int(v) for v in self.training_config.get("image_size", (256, 256))),
                "stride": int(self.training_config.get("query_stride", 4)),
            }
            if phase in (3, 4):
                cached["pred_b"] = out_b["pred_flow"].detach().cpu()
                cached["obs_b"] = batch["view_b"]["observed_mask"].detach().cpu()
            self.example_batch = cached
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def on_validation_epoch_start(self) -> None:
        self.example_batch = None

    def on_validation_epoch_end(self) -> None:
        if self.example_batch is None or self.logger is None:
            return
        self._log_example_panel()

    def _log_example_panel(self) -> None:
        batch = self.example_batch
        max_images = int(self.training_config.get("tb_log_images", 2))
        image_size = tuple(batch["image_size"])
        stride = int(batch["stride"])
        count = min(max_images, batch["image1"].shape[0])
        figs = []
        for idx in range(count):
            target_hw = self.q_to_hw(batch["target_flow_q"][idx : idx + 1], image_size, stride)[0]
            valid_hw = self.valid_q_to_hw(batch["target_valid_q"][idx : idx + 1], image_size, stride)[0]
            pred_a_hw = self.q_to_hw(batch["pred_a"][idx : idx + 1], image_size, stride)[0]
            obs_a = batch["obs_a"][idx]
            fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=130)
            panels = [
                ("Frame 1", self.denormalize_rgb(batch["image1"][idx]).numpy().transpose(1, 2, 0)),
                ("Frame 2", self.denormalize_rgb(batch["image2"][idx]).numpy().transpose(1, 2, 0)),
                ("Observed A", np.repeat(obs_a.numpy()[..., None], 3, axis=-1)),
                ("Target Flow", self.flow_to_rgb(target_hw * torch.tensor([image_size[1], image_size[0]])).numpy().transpose(1, 2, 0)),
                ("Pred Flow A", self.flow_to_rgb(pred_a_hw * torch.tensor([image_size[1], image_size[0]])).numpy().transpose(1, 2, 0)),
                ("Valid Q", np.repeat(valid_hw.numpy()[..., None], 3, axis=-1)),
            ]
            if batch["phase"] in (3, 4):
                pred_b_hw = self.q_to_hw(batch["pred_b"][idx : idx + 1], image_size, stride)[0]
                obs_b = batch["obs_b"][idx]
                panels[2] = ("Observed A/B", np.repeat(torch.maximum(obs_a, obs_b).numpy()[..., None], 3, axis=-1))
                panels[5] = ("Pred Flow B", self.flow_to_rgb(pred_b_hw * torch.tensor([image_size[1], image_size[0]])).numpy().transpose(1, 2, 0))
            for ax, (title, image) in zip(axes.flat, panels):
                ax.imshow(np.clip(image, 0.0, 1.0))
                ax.set_title(title, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            self.logger.experiment.add_figure(f"val/examples_{idx}", fig, global_step=self.current_epoch)
            figs.append(fig)
        for fig in figs:
            plt.close(fig)

    def configure_optimizers(self) -> dict[str, Any]:
        lr = float(self.training_config.get("lr", 1e-4))
        weight_decay = float(self.training_config.get("weight_decay", 1e-4))
        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(self.training_config.get("max_epochs", 1))))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
