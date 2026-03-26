from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino_context_model import PatchEmbed, TransformerStack


@dataclass
class FlowMAEDINOContextConsistencyModelConfig:
    image_size: int = 256
    patch_size: int = 16
    rgb_channels: int = 6
    use_rgb_inputs: bool = False
    flow_channels: int = 2
    valid_channels: int = 1
    context_feature_dim: int = 768
    encoder_dim: int = 768
    encoder_depth: int = 8
    encoder_heads: int = 12
    mlp_ratio: float = 4.0
    decoder_dim: int = 512
    decoder_depth: int = 4
    decoder_heads: int = 8
    dropout: float = 0.0
    attention_dropout: float = 0.0
    loss: str = "smooth_l1"
    smooth_l1_beta: float = 1.0
    reconstruction_loss_mask: str = "masked"
    projector_hidden_dim: int = 768
    projector_output_dim: int = 256
    normalize_projector_output: bool = True
    anchor_full_visible_prob: float = 0.80
    anchor_light_mask_prob: float = 0.20
    anchor_light_mask_ratio_min: float = 0.0
    anchor_light_mask_ratio_max: float = 0.10
    anchor_speckle_keep_ratio: float = 1.0
    anchor_speckle_dilation_kernel: int = 1
    student_moderate_mask_prob: float = 0.20
    student_heavy_mask_prob: float = 0.60
    student_full_mask_prob: float = 0.20
    student_moderate_mask_ratio_min: float = 0.40
    student_moderate_mask_ratio_max: float = 0.70
    student_heavy_mask_ratio_min: float = 0.85
    student_heavy_mask_ratio_max: float = 0.95
    student_speckle_keep_ratio: float = 0.20
    student_speckle_dilation_kernel: int = 1


class ProjectorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class FlowMaskedAutoencoderDINOPrependedContextConsistencyViT(nn.Module):
    def __init__(self, config: FlowMAEDINOContextConsistencyModelConfig) -> None:
        super().__init__()
        self.config = config
        image_size = int(config.image_size)
        patch_size = int(config.patch_size)
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.flow_patch_dim = int(config.flow_channels) * patch_size * patch_size
        self.context_feature_dim = int(config.context_feature_dim)

        self.local_in_channels = (
            (int(config.rgb_channels) if bool(config.use_rgb_inputs) else 0)
            + int(config.flow_channels)
            + int(config.valid_channels)
        )
        self.local_embed = PatchEmbed(self.local_in_channels, patch_size, int(config.encoder_dim))
        self.context_norm = nn.LayerNorm(self.context_feature_dim)
        self.context_proj = nn.Linear(self.context_feature_dim, int(config.encoder_dim))

        self.local_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, int(config.encoder_dim)))
        self.context_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, int(config.encoder_dim)))
        self.src_type_embed = nn.Parameter(torch.zeros(1, 1, int(config.encoder_dim)))
        self.tgt_type_embed = nn.Parameter(torch.zeros(1, 1, int(config.encoder_dim)))
        self.local_type_embed = nn.Parameter(torch.zeros(1, 1, int(config.encoder_dim)))

        self.decoder_local_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, int(config.decoder_dim)))
        self.decoder_local_type_embed = nn.Parameter(torch.zeros(1, 1, int(config.decoder_dim)))

        self.encoder = TransformerStack(
            dim=int(config.encoder_dim),
            depth=int(config.encoder_depth),
            num_heads=int(config.encoder_heads),
            mlp_ratio=float(config.mlp_ratio),
            dropout=float(config.dropout),
            attention_dropout=float(config.attention_dropout),
        )
        self.encoder_to_decoder = nn.Linear(int(config.encoder_dim), int(config.decoder_dim))
        self.decoder = TransformerStack(
            dim=int(config.decoder_dim),
            depth=int(config.decoder_depth),
            num_heads=int(config.decoder_heads),
            mlp_ratio=float(config.mlp_ratio),
            dropout=float(config.dropout),
            attention_dropout=float(config.attention_dropout),
        )
        self.flow_head = nn.Linear(int(config.decoder_dim), self.flow_patch_dim)
        self.projector = ProjectorMLP(
            input_dim=int(config.encoder_dim),
            hidden_dim=int(config.projector_hidden_dim),
            output_dim=int(config.projector_output_dim),
            dropout=float(config.dropout),
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.local_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.context_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.src_type_embed, std=0.02)
        nn.init.trunc_normal_(self.tgt_type_embed, std=0.02)
        nn.init.trunc_normal_(self.local_type_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_local_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_local_type_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def patchify_mask(self, valid: torch.Tensor) -> torch.Tensor:
        patch_valid = F.avg_pool2d(valid.unsqueeze(1), kernel_size=self.patch_size, stride=self.patch_size)
        return (patch_valid[:, 0] > 0).float().reshape(valid.shape[0], self.num_patches)

    def expand_patch_mask(self, patch_mask: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            patch_mask.view(patch_mask.shape[0], 1, self.grid_size, self.grid_size),
            scale_factor=self.patch_size,
            mode="nearest",
        )[:, 0]

    def sample_patch_mask_with_ratio(
        self,
        valid: torch.Tensor,
        ratio: float,
        *,
        allow_full_mask: bool,
    ) -> torch.Tensor:
        device = valid.device
        valid_patches = self.patchify_mask(valid)
        bsz, num_patches = valid_patches.shape
        masked = torch.zeros((bsz, num_patches), device=device, dtype=torch.float32)
        ratio = max(0.0, min(1.0, float(ratio)))

        for b in range(bsz):
            valid_idx = torch.nonzero(valid_patches[b] > 0, as_tuple=False).flatten()
            if valid_idx.numel() == 0 or ratio <= 0.0:
                continue
            num_mask = int(math.ceil(valid_idx.numel() * ratio))
            if allow_full_mask:
                num_mask = min(num_mask, valid_idx.numel())
            elif valid_idx.numel() > 1:
                num_mask = min(num_mask, valid_idx.numel() - 1)
            else:
                num_mask = 0
            if num_mask <= 0:
                continue
            perm = torch.randperm(valid_idx.numel(), device=device)[:num_mask]
            masked[b, valid_idx[perm]] = 1.0
        return masked

    @staticmethod
    def _uniform_ratio(min_ratio: float, max_ratio: float) -> float:
        min_ratio = max(0.0, min(1.0, float(min_ratio)))
        max_ratio = max(0.0, min(1.0, float(max_ratio)))
        if max_ratio < min_ratio:
            min_ratio, max_ratio = max_ratio, min_ratio
        if max_ratio == min_ratio:
            return min_ratio
        return float(torch.empty(1).uniform_(min_ratio, max_ratio).item())

    def sample_speckle_observation(
        self,
        valid: torch.Tensor,
        keep_ratio: float,
        dilation_kernel: int,
        *,
        ensure_nonempty: bool,
    ) -> torch.Tensor:
        keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
        if keep_ratio >= 1.0:
            return valid.clone()

        observed = ((torch.rand_like(valid) < keep_ratio) & (valid > 0)).float()
        kernel = max(1, int(dilation_kernel))
        if kernel > 1:
            observed = F.max_pool2d(
                observed.unsqueeze(1),
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            )[:, 0]
            observed = (observed > 0).float()

        observed = observed * valid
        if ensure_nonempty:
            for b in range(valid.shape[0]):
                if float(valid[b].sum()) <= 0.0 or float(observed[b].sum()) > 0.0:
                    continue
                valid_idx = torch.nonzero(valid[b] > 0, as_tuple=False)
                if valid_idx.numel() == 0:
                    continue
                choice = valid_idx[torch.randint(valid_idx.shape[0], (1,), device=valid.device).item()]
                observed[b, choice[0], choice[1]] = 1.0
        return observed

    def build_observation(
        self,
        valid: torch.Tensor,
        *,
        mask_ratio: float,
        speckle_keep_ratio: float,
        speckle_dilation_kernel: int,
        allow_full_mask: bool,
        ensure_nonempty: bool,
        is_full_mask: bool,
    ) -> dict[str, torch.Tensor]:
        if is_full_mask:
            patch_mask = self.patchify_mask(valid)
            observed_valid = torch.zeros_like(valid)
        else:
            patch_mask = self.sample_patch_mask_with_ratio(valid, mask_ratio, allow_full_mask=allow_full_mask)
            visible_patch_mask = 1.0 - self.expand_patch_mask(patch_mask)
            patch_visible_valid = valid * visible_patch_mask
            observed_valid = self.sample_speckle_observation(
                patch_visible_valid,
                keep_ratio=speckle_keep_ratio,
                dilation_kernel=speckle_dilation_kernel,
                ensure_nonempty=ensure_nonempty,
            )

        masked_pixels = valid * (1.0 - observed_valid)
        valid_pixels = valid.sum(dim=(1, 2))
        observed_pixels = observed_valid.sum(dim=(1, 2))
        visible_ratio = torch.where(
            valid_pixels > 0,
            observed_pixels / valid_pixels.clamp_min(1.0),
            torch.zeros_like(valid_pixels),
        )
        return {
            "observed_valid": observed_valid,
            "patch_mask": patch_mask,
            "masked_pixels": masked_pixels,
            "visible_ratio": visible_ratio,
        }

    def sample_anchor_observation(self, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz = valid.shape[0]
        observed_valid = torch.zeros_like(valid)
        patch_mask = torch.zeros((bsz, self.num_patches), device=valid.device, dtype=torch.float32)
        masked_pixels = torch.zeros_like(valid)
        visible_ratio = torch.zeros((bsz,), device=valid.device, dtype=valid.dtype)
        is_full_mask = torch.zeros((bsz,), device=valid.device, dtype=valid.dtype)

        full_prob = max(0.0, float(self.config.anchor_full_visible_prob))
        light_prob = max(0.0, float(self.config.anchor_light_mask_prob))
        total_prob = full_prob + light_prob
        if total_prob <= 0.0:
            full_prob = 1.0
            total_prob = 1.0

        for b in range(bsz):
            sample_prob = float(torch.rand(1).item()) * total_prob
            if sample_prob < full_prob:
                view = self.build_observation(
                    valid[b : b + 1],
                    mask_ratio=0.0,
                    speckle_keep_ratio=1.0,
                    speckle_dilation_kernel=1,
                    allow_full_mask=False,
                    ensure_nonempty=False,
                    is_full_mask=False,
                )
            else:
                view = self.build_observation(
                    valid[b : b + 1],
                    mask_ratio=self._uniform_ratio(
                        self.config.anchor_light_mask_ratio_min,
                        self.config.anchor_light_mask_ratio_max,
                    ),
                    speckle_keep_ratio=float(self.config.anchor_speckle_keep_ratio),
                    speckle_dilation_kernel=int(self.config.anchor_speckle_dilation_kernel),
                    allow_full_mask=False,
                    ensure_nonempty=True,
                    is_full_mask=False,
                )
            observed_valid[b] = view["observed_valid"][0]
            patch_mask[b] = view["patch_mask"][0]
            masked_pixels[b] = view["masked_pixels"][0]
            visible_ratio[b] = view["visible_ratio"][0]

        return {
            "observed_valid": observed_valid,
            "patch_mask": patch_mask,
            "masked_pixels": masked_pixels,
            "visible_ratio": visible_ratio,
            "is_full_mask": is_full_mask,
        }

    def sample_student_observation(self, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz = valid.shape[0]
        observed_valid = torch.zeros_like(valid)
        patch_mask = torch.zeros((bsz, self.num_patches), device=valid.device, dtype=torch.float32)
        masked_pixels = torch.zeros_like(valid)
        visible_ratio = torch.zeros((bsz,), device=valid.device, dtype=valid.dtype)
        is_full_mask = torch.zeros((bsz,), device=valid.device, dtype=valid.dtype)

        moderate_prob = max(0.0, float(self.config.student_moderate_mask_prob))
        heavy_prob = max(0.0, float(self.config.student_heavy_mask_prob))
        full_prob = max(0.0, float(self.config.student_full_mask_prob))
        total_prob = moderate_prob + heavy_prob + full_prob
        if total_prob <= 0.0:
            heavy_prob = 1.0
            total_prob = 1.0

        for b in range(bsz):
            sample_prob = float(torch.rand(1).item()) * total_prob
            if sample_prob < moderate_prob:
                mask_ratio = self._uniform_ratio(
                    self.config.student_moderate_mask_ratio_min,
                    self.config.student_moderate_mask_ratio_max,
                )
                full_mask = False
            elif sample_prob < moderate_prob + heavy_prob:
                mask_ratio = self._uniform_ratio(
                    self.config.student_heavy_mask_ratio_min,
                    self.config.student_heavy_mask_ratio_max,
                )
                full_mask = False
            else:
                mask_ratio = 1.0
                full_mask = True

            view = self.build_observation(
                valid[b : b + 1],
                mask_ratio=mask_ratio,
                speckle_keep_ratio=float(self.config.student_speckle_keep_ratio),
                speckle_dilation_kernel=int(self.config.student_speckle_dilation_kernel),
                allow_full_mask=True,
                ensure_nonempty=not full_mask,
                is_full_mask=full_mask,
            )
            observed_valid[b] = view["observed_valid"][0]
            patch_mask[b] = view["patch_mask"][0]
            masked_pixels[b] = view["masked_pixels"][0]
            visible_ratio[b] = view["visible_ratio"][0]
            is_full_mask[b] = float(full_mask)

        return {
            "observed_valid": observed_valid,
            "patch_mask": patch_mask,
            "masked_pixels": masked_pixels,
            "visible_ratio": visible_ratio,
            "is_full_mask": is_full_mask,
        }

    def unpatchify_flow(self, flow_tokens: torch.Tensor) -> torch.Tensor:
        bsz = flow_tokens.shape[0]
        p = self.patch_size
        flow_channels = int(self.config.flow_channels)
        x = flow_tokens.view(bsz, self.grid_size, self.grid_size, flow_channels, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(bsz, flow_channels, self.image_size, self.image_size)

    def build_local_inputs(
        self,
        src_rgb: torch.Tensor,
        tgt_rgb: torch.Tensor,
        observed_flow: torch.Tensor,
        observed_valid: torch.Tensor,
    ) -> torch.Tensor:
        _, _, height, width = observed_flow.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(f"Expected {self.image_size}x{self.image_size} inputs, got {height}x{width}")
        pieces = []
        if bool(self.config.use_rgb_inputs):
            pieces.append(torch.cat([src_rgb, tgt_rgb], dim=1))
        pieces.extend([observed_flow, observed_valid.unsqueeze(1)])
        return torch.cat(pieces, dim=1)

    def flatten_context_tokens(self, context: torch.Tensor) -> torch.Tensor:
        if context.dim() == 4:
            bsz, grid_h, grid_w, dim = context.shape
            if (grid_h, grid_w) != (self.grid_size, self.grid_size):
                raise ValueError(
                    f"Expected DINO grid {(self.grid_size, self.grid_size)}, got {(grid_h, grid_w)}."
                )
            context = context.reshape(bsz, grid_h * grid_w, dim)
        elif context.dim() != 3:
            raise ValueError(f"Expected [B, grid_h, grid_w, dim] or [B, num_patches, dim], got {tuple(context.shape)}")

        if context.shape[1] != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} DINO tokens, got {context.shape[1]}.")
        if context.shape[2] != self.context_feature_dim:
            raise ValueError(f"Expected DINO feature dim {self.context_feature_dim}, got {context.shape[2]}.")
        return context

    def build_context_sequence(self, src_dino: torch.Tensor, tgt_dino: torch.Tensor) -> torch.Tensor:
        src_tokens = self.context_proj(self.context_norm(self.flatten_context_tokens(src_dino)))
        tgt_tokens = self.context_proj(self.context_norm(self.flatten_context_tokens(tgt_dino)))
        src_tokens = src_tokens + self.context_pos_embed + self.src_type_embed
        tgt_tokens = tgt_tokens + self.context_pos_embed + self.tgt_type_embed
        return torch.cat([src_tokens, tgt_tokens], dim=1)

    def pool_local_tokens(self, local_tokens: torch.Tensor) -> torch.Tensor:
        return local_tokens.mean(dim=1)

    def project_pair_latent(self, pair_latent: torch.Tensor) -> torch.Tensor:
        projected = self.projector(pair_latent)
        if bool(self.config.normalize_projector_output):
            projected = F.normalize(projected, dim=-1)
        return projected

    def compute_reconstruction_loss(
        self,
        pred_flow: torch.Tensor,
        target_flow: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.unsqueeze(1)
        if self.config.loss == "l1":
            per_pixel = (pred_flow - target_flow).abs()
        else:
            per_pixel = F.smooth_l1_loss(
                pred_flow,
                target_flow,
                beta=float(self.config.smooth_l1_beta),
                reduction="none",
            )
        denom = mask.sum().clamp_min(1.0) * pred_flow.shape[1]
        return (per_pixel * mask).sum() / denom

    def resolve_reconstruction_mask(self, valid: torch.Tensor, masked_pixels: torch.Tensor) -> torch.Tensor:
        mode = str(self.config.reconstruction_loss_mask).lower()
        if mode == "masked":
            return masked_pixels
        if mode == "valid":
            return valid
        raise ValueError(
            f"Unsupported reconstruction_loss_mask={self.config.reconstruction_loss_mask!r}. "
            "Expected one of {'masked', 'valid'}."
        )

    def forward_branch(
        self,
        src_rgb: torch.Tensor,
        tgt_rgb: torch.Tensor,
        src_dino: torch.Tensor,
        tgt_dino: torch.Tensor,
        observed_flow: torch.Tensor,
        observed_valid: torch.Tensor,
        *,
        return_latent: bool = True,
        decode: bool = True,
    ) -> dict[str, Any]:
        local_input = self.build_local_inputs(src_rgb, tgt_rgb, observed_flow, observed_valid)
        local_tokens = self.local_embed(local_input) + self.local_pos_embed + self.local_type_embed
        context_tokens = self.build_context_sequence(src_dino, tgt_dino)
        encoded = self.encoder(torch.cat([context_tokens, local_tokens], dim=1))
        encoded_local_tokens = encoded[:, -self.num_patches :]

        outputs: dict[str, Any] = {
            "local_input": local_input,
            "flow_input": observed_flow,
            "observed_valid": observed_valid,
            "encoded_tokens": encoded_local_tokens,
            "context_tokens": context_tokens,
        }

        if return_latent:
            pair_latent = self.pool_local_tokens(encoded_local_tokens)
            outputs["pair_latent"] = pair_latent
            outputs["projected_latent"] = self.project_pair_latent(pair_latent)

        if decode:
            decoded = self.decoder(
                self.encoder_to_decoder(encoded_local_tokens)
                + self.decoder_local_pos_embed
                + self.decoder_local_type_embed
            )
            outputs["pred_flow"] = self.unpatchify_flow(self.flow_head(decoded))

        return outputs

    def forward(
        self,
        src_rgb: torch.Tensor,
        tgt_rgb: torch.Tensor,
        src_dino: torch.Tensor,
        tgt_dino: torch.Tensor,
        flow: torch.Tensor,
        valid: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        observed_valid_override: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if observed_valid_override is not None:
            observed_valid = observed_valid_override * valid
            masked_pixels = valid * (1.0 - observed_valid)
            patch_mask = (self.patchify_mask(masked_pixels) > 0).float()
            is_full_mask = (observed_valid.sum(dim=(1, 2)) <= 0).to(valid.dtype)
        elif patch_mask is not None:
            masked_pixels = self.expand_patch_mask(patch_mask) * valid
            observed_valid = valid * (1.0 - masked_pixels)
            is_full_mask = (observed_valid.sum(dim=(1, 2)) <= 0).to(valid.dtype)
        else:
            view = self.sample_student_observation(valid)
            observed_valid = view["observed_valid"]
            patch_mask = view["patch_mask"]
            masked_pixels = view["masked_pixels"]
            is_full_mask = view["is_full_mask"]

        observed_flow = flow * observed_valid.unsqueeze(1)
        outputs = self.forward_branch(
            src_rgb=src_rgb,
            tgt_rgb=tgt_rgb,
            src_dino=src_dino,
            tgt_dino=tgt_dino,
            observed_flow=observed_flow,
            observed_valid=observed_valid,
            return_latent=True,
            decode=True,
        )
        recon_mask = self.resolve_reconstruction_mask(valid, masked_pixels)
        outputs["loss"] = self.compute_reconstruction_loss(outputs["pred_flow"], flow, recon_mask)
        outputs["patch_mask"] = patch_mask
        outputs["masked_pixels"] = masked_pixels
        outputs["observed_pixels"] = observed_valid
        outputs["reconstruction_mask"] = recon_mask
        outputs["is_full_mask"] = is_full_mask
        return outputs
