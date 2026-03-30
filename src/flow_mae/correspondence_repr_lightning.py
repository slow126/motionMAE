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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .correspondence_repr_model import (
    CorrespondenceReprModel,
    CorrespondenceReprModelConfig,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class CorrespondenceReprLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: CorrespondenceReprModelConfig,
        training_config: dict[str, Any],
        pointodyssey_probe_config: Optional[dict[str, Any]] = None,
        qualitative_probe_configs: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters({
            "model": vars(model_config),
            "training": training_config,
            "pointodyssey_probe": pointodyssey_probe_config,
            "qualitative_probes": qualitative_probe_configs,
        })
        self.model = CorrespondenceReprModel(model_config)
        self.training_config = training_config
        self.pointodyssey_probe_config = pointodyssey_probe_config
        self.qualitative_probe_configs = qualitative_probe_configs or []
        self.example_batch = None
        self._probe_dino = None

        # EMA buffers for VICReg-style losses (smooth over small batches)
        proj_dim = int(model_config.projector_output_dim)
        self.register_buffer("running_z_var",
                             torch.ones(proj_dim))
        self.register_buffer("running_z_ratio_corr",
                             torch.zeros(proj_dim))
        self._ema_decay = float(training_config.get("ema_decay", 0.99))

    # -------------------------------------------------------------------
    # EMA-smoothed variance & decorrelation losses
    # -------------------------------------------------------------------

    def _update_ema_stats(self, z: torch.Tensor,
                          visible_ratio: torch.Tensor) -> None:
        """Update running statistics from current batch."""
        decay = self._ema_decay

        # Per-dim variance
        batch_var = z.var(dim=0, unbiased=False)
        self.running_z_var = (decay * self.running_z_var
                              + (1 - decay) * batch_var.detach())

        # Per-dim correlation with visible_ratio
        if visible_ratio.shape[0] > 1:
            z_c = z - z.mean(0)
            r_c = visible_ratio - visible_ratio.mean()
            r_std = r_c.std().clamp_min(1e-6)
            z_std = z_c.std(0).clamp_min(1e-6)
            batch_corr = (z_c * r_c.unsqueeze(1)).mean(0) / (z_std * r_std)
            self.running_z_ratio_corr = (
                decay * self.running_z_ratio_corr
                + (1 - decay) * batch_corr.detach())

    def variance_loss(self, z: torch.Tensor,
                      target_std: float = 1.0,
                      eps: float = 1e-4) -> torch.Tensor:
        """VICReg variance regularizer using EMA variance estimate.

        Gradients flow through *z* (current batch) but the target threshold
        is compared against the smoothed running variance.
        """
        if z.shape[0] <= 1:
            return z.new_zeros(())
        batch_std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        # Use running estimate for the threshold comparison but let
        # gradients flow through the batch computation
        running_std = torch.sqrt(self.running_z_var + eps)
        # Blend: penalise where the *running* estimate is below target,
        # but backprop through the batch std so the model gets gradients
        below_target = (running_std < target_std).float()
        return (F.relu(target_std - batch_std) * below_target).mean()

    def decorrelation_loss(self, z: torch.Tensor,
                           visible_ratio: torch.Tensor) -> torch.Tensor:
        """Penalise correlation between z_corr dims and visible_ratio.

        Uses the EMA correlation estimate for stability but gradients
        flow through the current batch's z.
        """
        if z.shape[0] <= 1:
            return z.new_zeros(())
        z_c = z - z.mean(0)
        r_c = visible_ratio - visible_ratio.mean()
        r_std = r_c.std().clamp_min(1e-6)
        z_std = z_c.std(0).clamp_min(1e-6)
        batch_corr = (z_c * r_c.unsqueeze(1)).mean(0) / (z_std * r_std)
        # Weight by running estimate magnitude — focus on persistently
        # correlated dimensions
        weight = self.running_z_ratio_corr.abs().clamp(0.0, 1.0).detach()
        return (batch_corr.pow(2) * (0.5 + 0.5 * weight)).mean()

    # -------------------------------------------------------------------
    # Static helpers (adapted from existing pipeline)
    # -------------------------------------------------------------------

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
        rgb = torch.zeros((3, flow.shape[-2], flow.shape[-1]),
                           device=flow.device, dtype=flow.dtype)
        rgb[0] = (0.5 + u / (2.0 * scale)).clamp(0.0, 1.0)
        rgb[1] = (0.5 + v / (2.0 * scale)).clamp(0.0, 1.0)
        return rgb

    @staticmethod
    def denormalize_rgb(image: torch.Tensor) -> torch.Tensor:
        mean = IMAGENET_MEAN.to(device=image.device, dtype=image.dtype)
        std = IMAGENET_STD.to(device=image.device, dtype=image.dtype)
        return (image * std + mean).clamp(0.0, 1.0)

    @staticmethod
    def mask_rgb_black(image: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
        return image * mask.to(device=image.device,
                               dtype=image.dtype).unsqueeze(0)

    @staticmethod
    def dino_to_rgb_pca(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"Expected [gh, gw, dim], got {tuple(tokens.shape)}")
        gh, gw, dim = tokens.shape
        flat = tokens.reshape(gh * gw, dim).float()
        flat = flat - flat.mean(dim=0, keepdim=True)
        try:
            _, _, v = torch.pca_lowrank(flat, q=min(3, flat.shape[0], flat.shape[1]))
            projected = flat @ v[:, :3]
        except RuntimeError:
            projected = flat[:, :3]
        if projected.shape[1] < 3:
            projected = F.pad(projected, (0, 3 - projected.shape[1]))
        projected = projected.reshape(gh, gw, 3)
        projected = projected - projected.amin(dim=(0, 1), keepdim=True)
        denom = projected.amax(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        projected = projected / denom
        return projected.permute(2, 0, 1).contiguous()

    @staticmethod
    def _pixel_space_flow(flow: torch.Tensor,
                          flow_scale: torch.Tensor | None) -> torch.Tensor:
        if flow_scale is None:
            return flow
        return flow * flow_scale.view(-1, 1, 1, 1).to(
            device=flow.device, dtype=flow.dtype)

    # -------------------------------------------------------------------
    # DINO probe helper (for datasets without precomputed DINO features)
    # -------------------------------------------------------------------

    def _get_probe_dino(self):
        if self._probe_dino is not None:
            return self._probe_dino
        model_dir = None
        if self.pointodyssey_probe_config:
            model_dir = self.pointodyssey_probe_config.get("dino_model_dir")
        if not model_dir:
            for probe_cfg in self.qualitative_probe_configs:
                model_dir = probe_cfg.get("dino_model_dir")
                if model_dir:
                    break
        if not model_dir:
            return None
        from models.DinoV3.DinoV3 import DinoV3
        resize_size = int(self.hparams["model"]["image_size"])
        self._probe_dino = DinoV3(pretrained_model_name=str(model_dir),
                                   resize_size=resize_size)
        try:
            self._probe_dino.model.to(self.device)
            self._probe_dino.model.eval()
        except Exception:
            pass
        return self._probe_dino

    def _augment_batch_with_dino(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if "src_dino" in batch and "tgt_dino" in batch:
            return batch
        dino = self._get_probe_dino()
        if dino is None:
            raise RuntimeError("Probe requires dino_model_dir to compute DINO tokens.")
        grid = self.model.grid_size
        with torch.inference_mode():
            src_dino = dino.get_spatial_features(batch["src_rgb"]).reshape(
                batch["src_rgb"].shape[0], grid, grid, -1)
            tgt_dino = dino.get_spatial_features(batch["tgt_rgb"]).reshape(
                batch["tgt_rgb"].shape[0], grid, grid, -1)
        batch = dict(batch)
        batch["src_dino"] = src_dino
        batch["tgt_dino"] = tgt_dino
        return batch

    # -------------------------------------------------------------------
    # Training step
    # -------------------------------------------------------------------

    def training_step(self, batch: dict[str, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        valid = batch["valid"]
        flow = batch["flow"]
        sampler = self.model.evidence_sampler

        # Sample two views of the evidence
        anchor_view = sampler.sample_anchor(valid)
        student_view = sampler.sample_student(valid)

        anchor_obs_flow = flow * anchor_view["observed_valid"].unsqueeze(1)
        student_obs_flow = flow * student_view["observed_valid"].unsqueeze(1)

        # Anchor branch: no_grad, latent only (no decode)
        with torch.no_grad():
            anchor_out = self.model.forward_branch(
                src_rgb=batch["src_rgb"],
                tgt_rgb=batch["tgt_rgb"],
                src_dino=batch["src_dino"],
                tgt_dino=batch["tgt_dino"],
                observed_flow=anchor_obs_flow,
                observed_valid=anchor_view["observed_valid"],
                visible_ratio=anchor_view["visible_ratio"],
                return_latent=True,
                decode=False,
                return_pre_encoder=False,
            )

        # Student branch: with grad, decode + pre-encoder readouts
        student_out = self.model.forward_branch(
            src_rgb=batch["src_rgb"],
            tgt_rgb=batch["tgt_rgb"],
            src_dino=batch["src_dino"],
            tgt_dino=batch["tgt_dino"],
            observed_flow=student_obs_flow,
            observed_valid=student_view["observed_valid"],
            visible_ratio=student_view["visible_ratio"],
            return_latent=True,
            decode=True,
            return_pre_encoder=True,
        )

        # -- Losses ----------------------------------------------------------

        # 1. Flow reconstruction on all valid pixels
        loss_recon = self.model.compute_reconstruction_loss(
            student_out["pred_flow"], flow, valid)

        # 2. Consistency: align student and anchor z_corr
        loss_align = F.mse_loss(
            student_out["projected_latent"],
            anchor_out["projected_latent"].detach())

        # 3. RGB reconstruction (pre-encoder grounding)
        loss_rgb_recon = torch.tensor(0.0, device=valid.device)
        if self.model.config.rgb_recon_enabled:
            loss_rgb_recon = self.model.compute_rgb_reconstruction_loss(
                student_out["src_rgb_tokens_raw"],
                student_out["tgt_rgb_tokens_raw"],
                batch["src_rgb"], batch["tgt_rgb"])

        # 4. Sup-state prediction
        loss_sup = torch.tensor(0.0, device=valid.device)
        if (self.model.config.sup_token_enabled
                and "pred_visible_ratio" in student_out):
            loss_sup = F.mse_loss(
                student_out["pred_visible_ratio"],
                student_view["visible_ratio"])

        # 5. Update EMA stats and compute VICReg losses
        z = student_out["projected_latent"]
        vis_ratio = student_view["visible_ratio"]
        self._update_ema_stats(z, vis_ratio)

        loss_var = self.variance_loss(
            z,
            target_std=float(self.training_config.get("variance_target_std", 1.0)),
            eps=float(self.training_config.get("variance_eps", 1e-4)))

        loss_decorr = self.decorrelation_loss(z, vis_ratio)

        # -- Weighted total --------------------------------------------------

        lam = self.training_config
        loss_total = (
            loss_recon
            + float(lam.get("lambda_align", 0.05)) * loss_align
            + float(lam.get("lambda_rgb_recon", 0.01)) * loss_rgb_recon
            + float(lam.get("lambda_sup", 0.01)) * loss_sup
            + float(lam.get("lambda_var", 0.01)) * loss_var
            + float(lam.get("lambda_decorr", 0.01)) * loss_decorr
        )

        # -- Logging ---------------------------------------------------------

        B = flow.shape[0]
        flow_scale = batch.get("flow_scale")
        pred_flow_px = self._pixel_space_flow(student_out["pred_flow"],
                                               flow_scale)
        target_flow_px = self._pixel_space_flow(flow, flow_scale)

        epe = self.model.endpoint_error(pred_flow_px, target_flow_px, valid)

        self.log("train/loss", loss_total, prog_bar=True, on_step=True,
                 on_epoch=True, batch_size=B)
        self.log("train/loss_recon", loss_recon, on_step=True, on_epoch=True,
                 batch_size=B)
        self.log("train/loss_align", loss_align, on_step=True, on_epoch=True,
                 batch_size=B)
        self.log("train/loss_rgb_recon", loss_rgb_recon, on_step=False,
                 on_epoch=True, batch_size=B)
        self.log("train/loss_sup", loss_sup, on_step=False, on_epoch=True,
                 batch_size=B)
        self.log("train/loss_var", loss_var, on_step=True, on_epoch=True,
                 batch_size=B)
        self.log("train/loss_decorr", loss_decorr, on_step=True, on_epoch=True,
                 batch_size=B)
        self.log("train/epe", epe, on_step=False, on_epoch=True, batch_size=B)
        self.log("train_loss", loss_total, prog_bar=False, on_step=False,
                 on_epoch=True, batch_size=B)

        self.log("train/anchor_visible_ratio",
                 anchor_view["visible_ratio"].mean(),
                 on_step=False, on_epoch=True, batch_size=B)
        self.log("train/student_visible_ratio",
                 student_view["visible_ratio"].mean(),
                 on_step=False, on_epoch=True, batch_size=B)
        self.log("train/projected_latent_norm",
                 z.norm(dim=-1).mean(),
                 on_step=False, on_epoch=True, batch_size=B)
        self.log("train/projected_alignment_mse",
                 (z - anchor_out["projected_latent"]).pow(2).mean(),
                 on_step=False, on_epoch=True, batch_size=B)

        # EMA stat monitoring
        self.log("train/ema_z_std_mean",
                 torch.sqrt(self.running_z_var + 1e-4).mean(),
                 on_step=False, on_epoch=True, batch_size=B)
        self.log("train/ema_z_ratio_corr_abs_mean",
                 self.running_z_ratio_corr.abs().mean(),
                 on_step=False, on_epoch=True, batch_size=B)

        if not torch.isfinite(loss_total):
            raise RuntimeError(
                f"Non-finite training loss: total={float(loss_total):.6f}, "
                f"recon={float(loss_recon):.6f}, align={float(loss_align):.6f}")

        return loss_total

    # -------------------------------------------------------------------
    # Validation step
    # -------------------------------------------------------------------

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Default forward for validation/probes.

        Probe datasets can provide an explicit observation mask via
        observed_valid_override; otherwise we sample a student-style mask.
        """
        valid = batch["valid"]
        observed_valid_override = batch.get("observed_valid_override")
        if observed_valid_override is not None:
            observed_valid = observed_valid_override.to(
                device=valid.device, dtype=valid.dtype) * valid
            valid_pixels = valid.sum(dim=(1, 2))
            visible_ratio = torch.where(
                valid_pixels > 0,
                observed_valid.sum(dim=(1, 2)) / valid_pixels.clamp_min(1.0),
                torch.zeros_like(valid_pixels),
            )
            view = {
                "observed_valid": observed_valid,
                "visible_ratio": visible_ratio,
            }
        else:
            view = self.model.evidence_sampler.sample_student(valid)
        obs_flow = batch["flow"] * view["observed_valid"].unsqueeze(1)
        out = self.model.forward_branch(
            src_rgb=batch["src_rgb"],
            tgt_rgb=batch["tgt_rgb"],
            src_dino=batch["src_dino"],
            tgt_dino=batch["tgt_dino"],
            observed_flow=obs_flow,
            observed_valid=view["observed_valid"],
            visible_ratio=view["visible_ratio"],
            return_latent=True,
            decode=True,
            return_pre_encoder=False,
        )
        out["observed_valid"] = view["observed_valid"]
        out["visible_ratio"] = view["visible_ratio"]
        loss = self.model.compute_reconstruction_loss(
            out["pred_flow"], batch["flow"], valid)
        out["loss"] = loss
        return out

    def validation_step(self, batch: dict[str, torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        B = batch["flow"].shape[0]
        flow_scale = batch.get("flow_scale")
        pred_flow_px = self._pixel_space_flow(outputs["pred_flow"], flow_scale)
        target_flow_px = self._pixel_space_flow(batch["flow"], flow_scale)
        epe = self.model.endpoint_error(pred_flow_px, target_flow_px,
                                         batch["valid"])

        self.log("val/loss", outputs["loss"], prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=B)
        self.log("val_loss", outputs["loss"], prog_bar=False, on_step=False,
                 on_epoch=True, batch_size=B)
        self.log("val/epe", epe, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=B)
        self.log("val_epe", epe, prog_bar=False, on_step=False, on_epoch=True,
                 batch_size=B)

        if self.example_batch is None:
            obs_flow_px = self._pixel_space_flow(
                batch["flow"] * outputs["observed_valid"].unsqueeze(1),
                flow_scale)
            self.example_batch = {
                "src_rgb": batch["src_rgb"].detach().cpu(),
                "tgt_rgb": batch["tgt_rgb"].detach().cpu(),
                "src_dino": batch["src_dino"].detach().cpu(),
                "tgt_dino": batch["tgt_dino"].detach().cpu(),
                "flow": target_flow_px.detach().cpu(),
                "valid": batch["valid"].detach().cpu(),
                "pred_flow": pred_flow_px.detach().cpu(),
                "flow_input": obs_flow_px.detach().cpu(),
                "observed_valid": outputs["observed_valid"].detach().cpu(),
            }
        return outputs["loss"]

    def on_validation_epoch_start(self) -> None:
        self.example_batch = None

    def on_validation_epoch_end(self) -> None:
        if self.logger is None:
            return
        if self.example_batch is not None:
            max_images = int(self.training_config.get("tb_log_images", 4))
            if max_images > 0:
                figures = []
                count = min(max_images, self.example_batch["flow"].shape[0])
                for idx in range(count):
                    fig = self._build_figure(idx)
                    figures.append(fig)
                    self.logger.experiment.add_figure(
                        f"val/examples_{idx}", fig,
                        global_step=self.current_epoch)
                for fig in figures:
                    plt.close(fig)
        self._run_pointodyssey_probe()
        self._run_qualitative_probes()

    # -------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------

    def _build_figure(self, idx: int) -> plt.Figure:
        src_rgb = self.denormalize_rgb(self.example_batch["src_rgb"][idx])
        tgt_rgb = self.denormalize_rgb(self.example_batch["tgt_rgb"][idx])
        src_dino_rgb = self.dino_to_rgb_pca(self.example_batch["src_dino"][idx])
        tgt_dino_rgb = self.dino_to_rgb_pca(self.example_batch["tgt_dino"][idx])
        observed_valid = self.example_batch["observed_valid"][idx]
        input_rgb = self.mask_rgb_black(
            self.flow_to_rgb(self.example_batch["flow_input"][idx]),
            observed_valid)
        gt_rgb = self.flow_to_rgb(self.example_batch["flow"][idx])
        pred_rgb = self.flow_to_rgb(self.example_batch["pred_flow"][idx])
        valid_rgb = observed_valid.unsqueeze(0).repeat(3, 1, 1)

        fig, axes = plt.subplots(2, 4, figsize=(15, 8), dpi=130)
        panels = [
            ("Source RGB", src_rgb),
            ("Target RGB", tgt_rgb),
            ("Source DINO PCA", src_dino_rgb),
            ("Target DINO PCA", tgt_dino_rgb),
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
        fig.suptitle("Correspondence repr validation", fontsize=10)
        fig.tight_layout()
        return fig

    # -------------------------------------------------------------------
    # Probes (adapted from existing pipeline)
    # -------------------------------------------------------------------

    def _run_pointodyssey_probe(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule,
                                              "get_pointodyssey_probe_dataloader"):
            return
        probe_loader = datamodule.get_pointodyssey_probe_dataloader()
        if probe_loader is None:
            return
        max_images = int(self.training_config.get(
            "pointodyssey_probe_log_images", 4))
        if max_images <= 0:
            return

        probe_examples = []
        with torch.no_grad():
            for pbatch in probe_loader:
                pbatch = {
                    k: v.to(self.device, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in pbatch.items()
                }
                pbatch = self._augment_batch_with_dino(pbatch)
                outputs = self(pbatch)
                pred_px = self._pixel_space_flow(
                    outputs["pred_flow"], pbatch.get("flow_scale"))
                obs_flow_px = self._pixel_space_flow(
                    pbatch["flow"] * outputs["observed_valid"].unsqueeze(1),
                    pbatch.get("flow_scale"))
                for i in range(pred_px.shape[0]):
                    probe_examples.append({
                        "src_rgb": pbatch["src_rgb"][i].cpu(),
                        "tgt_rgb": pbatch["tgt_rgb"][i].cpu(),
                        "src_dino": pbatch["src_dino"][i].cpu(),
                        "tgt_dino": pbatch["tgt_dino"][i].cpu(),
                        "pred_flow": pred_px[i].cpu(),
                        "flow_input": obs_flow_px[i].cpu(),
                        "observed_valid": outputs["observed_valid"][i].cpu(),
                    })
                    if len(probe_examples) >= max_images:
                        break
                if len(probe_examples) >= max_images:
                    break

        if not probe_examples:
            return
        figures = []
        for idx, ex in enumerate(probe_examples):
            fig = self._build_probe_figure(ex)
            figures.append(fig)
            self.logger.experiment.add_figure(
                f"pointodyssey_probe/examples_{idx}", fig,
                global_step=self.current_epoch)
        for fig in figures:
            plt.close(fig)

    def _build_probe_figure(self, ex: dict[str, torch.Tensor]) -> plt.Figure:
        src_rgb = self.denormalize_rgb(ex["src_rgb"])
        tgt_rgb = self.denormalize_rgb(ex["tgt_rgb"])
        src_dino_rgb = self.dino_to_rgb_pca(ex["src_dino"])
        tgt_dino_rgb = self.dino_to_rgb_pca(ex["tgt_dino"])
        input_rgb = self.mask_rgb_black(
            self.flow_to_rgb(ex["flow_input"]), ex["observed_valid"])
        pred_rgb = self.flow_to_rgb(ex["pred_flow"])

        fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=130)
        panels = [
            ("Source RGB", src_rgb), ("Target RGB", tgt_rgb),
            ("Source DINO PCA", src_dino_rgb),
            ("Target DINO PCA", tgt_dino_rgb),
            ("Observed Flow", input_rgb), ("Prediction", pred_rgb),
        ]
        for ax, (title, image) in zip(axes.flat[:6], panels):
            ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("PointOdyssey probe", fontsize=10)
        fig.tight_layout()
        return fig

    def _run_qualitative_probes(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not hasattr(
                datamodule, "get_qualitative_probe_dataloaders"):
            return
        probe_loaders = datamodule.get_qualitative_probe_dataloaders()
        if not probe_loaders:
            return
        max_images = int(self.training_config.get(
            "qualitative_probe_log_images", 4))
        if max_images <= 0:
            return

        for probe_name, probe_loader in probe_loaders.items():
            probe_examples = []
            with torch.no_grad():
                for pbatch in probe_loader:
                    pbatch = {
                        k: v.to(self.device, non_blocking=True)
                        if isinstance(v, torch.Tensor) else v
                        for k, v in pbatch.items()
                    }
                    pbatch = self._augment_batch_with_dino(pbatch)
                    outputs = self(pbatch)
                    pred_px = self._pixel_space_flow(
                        outputs["pred_flow"], pbatch.get("flow_scale"))
                    obs_flow_px = self._pixel_space_flow(
                        pbatch["flow"] * outputs["observed_valid"].unsqueeze(1),
                        pbatch.get("flow_scale"))
                    target_px = self._pixel_space_flow(
                        pbatch["flow"], pbatch.get("flow_scale"))
                    for i in range(pred_px.shape[0]):
                        probe_examples.append({
                            "src_rgb": pbatch["src_rgb"][i].cpu(),
                            "tgt_rgb": pbatch["tgt_rgb"][i].cpu(),
                            "src_dino": pbatch["src_dino"][i].cpu(),
                            "tgt_dino": pbatch["tgt_dino"][i].cpu(),
                            "pred_flow": pred_px[i].cpu(),
                            "flow_input": obs_flow_px[i].cpu(),
                            "target_flow": target_px[i].cpu(),
                            "observed_valid": outputs["observed_valid"][i].cpu(),
                            "valid": pbatch["valid"][i].cpu(),
                        })
                        if len(probe_examples) >= max_images:
                            break
                    if len(probe_examples) >= max_images:
                        break

            if not probe_examples:
                continue
            figures = []
            for idx, ex in enumerate(probe_examples):
                fig = self._build_qual_probe_figure(probe_name, ex)
                figures.append(fig)
                self.logger.experiment.add_figure(
                    f"qualitative_probe/{probe_name}/examples_{idx}", fig,
                    global_step=self.current_epoch)
            for fig in figures:
                plt.close(fig)

    def _build_qual_probe_figure(self, probe_name: str,
                                  ex: dict[str, torch.Tensor]) -> plt.Figure:
        src_rgb = self.denormalize_rgb(ex["src_rgb"])
        tgt_rgb = self.denormalize_rgb(ex["tgt_rgb"])
        src_dino_rgb = self.dino_to_rgb_pca(ex["src_dino"])
        tgt_dino_rgb = self.dino_to_rgb_pca(ex["tgt_dino"])
        obs_valid = ex["observed_valid"]
        input_rgb = self.mask_rgb_black(
            self.flow_to_rgb(ex["flow_input"]), obs_valid)
        gt_rgb = self.mask_rgb_black(
            self.flow_to_rgb(ex["target_flow"]), ex["valid"])
        pred_rgb = self.flow_to_rgb(ex["pred_flow"])
        valid_rgb = obs_valid.unsqueeze(0).repeat(3, 1, 1)

        fig, axes = plt.subplots(2, 4, figsize=(15, 8), dpi=130)
        panels = [
            ("Source RGB", src_rgb), ("Target RGB", tgt_rgb),
            ("Source DINO PCA", src_dino_rgb),
            ("Target DINO PCA", tgt_dino_rgb),
            ("Observed Flow", input_rgb), ("Ground Truth", gt_rgb),
            ("Prediction", pred_rgb), ("Observed Mask", valid_rgb),
        ]
        for ax, (title, image) in zip(axes.flat, panels):
            ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{probe_name} probe", fontsize=10)
        fig.tight_layout()
        return fig

    # -------------------------------------------------------------------
    # Optimizer / scheduler
    # -------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        lr = float(self.training_config.get("lr", 7.5e-5))
        weight_decay = float(self.training_config.get("weight_decay", 0.05))
        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        warmup_epochs = int(self.training_config.get("warmup_epochs", 10))
        max_epochs = int(self.training_config.get("max_epochs", 200))
        min_lr = float(self.training_config.get("min_lr", 1e-6))

        if warmup_epochs > 0 and max_epochs > warmup_epochs:
            warmup = LinearLR(optimizer, start_factor=0.1,
                              total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(optimizer,
                                        T_max=max_epochs - warmup_epochs,
                                        eta_min=min_lr)
            scheduler = SequentialLR(optimizer,
                                      schedulers=[warmup, cosine],
                                      milestones=[warmup_epochs])
        else:
            scheduler = CosineAnnealingLR(optimizer,
                                           T_max=max(1, max_epochs),
                                           eta_min=min_lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
