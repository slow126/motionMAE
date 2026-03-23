from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowMAEModelConfig:
    image_size: int = 256
    patch_size: int = 16
    rgb_channels: int = 6
    flow_channels: int = 2
    valid_channels: int = 1
    encoder_dim: int = 384
    encoder_depth: int = 6
    encoder_heads: int = 6
    mlp_ratio: float = 4.0
    decoder_dim: int = 256
    decoder_depth: int = 2
    decoder_heads: int = 8
    dropout: float = 0.0
    attention_dropout: float = 0.0
    mask_ratio: float = 0.75
    observation_mask_mode: str = "patch"
    speckle_keep_ratio: float = 0.05
    speckle_dilation_kernel: int = 1
    loss: str = "smooth_l1"
    smooth_l1_beta: float = 1.0


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        self.drop_path = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerStack(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class FlowMaskedAutoencoderViT(nn.Module):
    def __init__(self, config: FlowMAEModelConfig) -> None:
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

        self.rgb_embed = PatchEmbed(int(config.rgb_channels), patch_size, int(config.encoder_dim))
        self.flow_embed = PatchEmbed(int(config.flow_channels + config.valid_channels), patch_size, int(config.encoder_dim))

        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, int(config.encoder_dim)))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, int(config.decoder_dim)))

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

        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
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

    def sample_patch_mask(self, valid: torch.Tensor) -> torch.Tensor:
        device = valid.device
        valid_patches = self.patchify_mask(valid)
        bsz, num_patches = valid_patches.shape
        masked = torch.zeros((bsz, num_patches), device=device, dtype=torch.float32)
        ratio = float(self.config.mask_ratio)
        for b in range(bsz):
            valid_idx = torch.nonzero(valid_patches[b] > 0, as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue
            num_mask = max(1, int(math.ceil(valid_idx.numel() * ratio)))
            if valid_idx.numel() > 1:
                num_mask = min(num_mask, valid_idx.numel() - 1)
            perm = torch.randperm(valid_idx.numel(), device=device)[:num_mask]
            masked[b, valid_idx[perm]] = 1.0
        return masked

    def sample_speckle_observation(self, valid: torch.Tensor) -> torch.Tensor:
        keep_ratio = float(self.config.speckle_keep_ratio)
        keep_ratio = max(0.0, min(1.0, keep_ratio))
        observed = ((torch.rand_like(valid) < keep_ratio) & (valid > 0)).float()

        kernel = max(1, int(self.config.speckle_dilation_kernel))
        if kernel > 1:
            observed = F.max_pool2d(
                observed.unsqueeze(1),
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            )[:, 0]
            observed = (observed > 0).float()

        observed = observed * valid

        # Ensure each sample with valid support exposes at least one observed pixel.
        for b in range(valid.shape[0]):
            if float(valid[b].sum()) <= 0.0 or float(observed[b].sum()) > 0.0:
                continue
            valid_idx = torch.nonzero(valid[b] > 0, as_tuple=False)
            if valid_idx.numel() == 0:
                continue
            choice = valid_idx[torch.randint(valid_idx.shape[0], (1,), device=valid.device).item()]
            observed[b, choice[0], choice[1]] = 1.0
        return observed

    def sample_observation_mask(self, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mode = str(self.config.observation_mask_mode).lower()
        bsz = valid.shape[0]

        if mode == "patch":
            patch_mask = self.sample_patch_mask(valid)
            masked_pixels = F.interpolate(
                patch_mask.view(bsz, 1, self.grid_size, self.grid_size),
                scale_factor=self.patch_size,
                mode="nearest",
            )[:, 0]
            observed_valid = valid * (1.0 - masked_pixels)
            return observed_valid, patch_mask, masked_pixels * valid

        if mode == "speckle":
            observed_valid = self.sample_speckle_observation(valid)
            masked_pixels = valid * (1.0 - observed_valid)
            patch_mask = (self.patchify_mask(masked_pixels) > 0).float()
            return observed_valid, patch_mask, masked_pixels

        if mode == "mixed":
            patch_mask = self.sample_patch_mask(valid)
            patch_visible = 1.0 - F.interpolate(
                patch_mask.view(bsz, 1, self.grid_size, self.grid_size),
                scale_factor=self.patch_size,
                mode="nearest",
            )[:, 0]
            speckle_visible = self.sample_speckle_observation(valid)
            observed_valid = valid * patch_visible * speckle_visible
            masked_pixels = valid * (1.0 - observed_valid)
            return observed_valid, patch_mask, masked_pixels

        raise ValueError(
            f"Unsupported observation_mask_mode={self.config.observation_mask_mode!r}. "
            "Expected one of {'patch', 'speckle', 'mixed'}."
        )

    def unpatchify_flow(self, flow_tokens: torch.Tensor) -> torch.Tensor:
        bsz = flow_tokens.shape[0]
        p = self.patch_size
        flow_channels = int(self.config.flow_channels)
        x = flow_tokens.view(bsz, self.grid_size, self.grid_size, flow_channels, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(bsz, flow_channels, self.image_size, self.image_size)

    def build_inputs(
        self,
        src_rgb: torch.Tensor,
        tgt_rgb: torch.Tensor,
        flow: torch.Tensor,
        observed_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, height, width = flow.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected {self.image_size}x{self.image_size} inputs, got {height}x{width}"
            )

        rgb = torch.cat([src_rgb, tgt_rgb], dim=1)
        observed_flow = flow * observed_valid.unsqueeze(1)
        flow_input = torch.cat([observed_flow, observed_valid.unsqueeze(1)], dim=1)
        return rgb, flow_input

    def compute_loss(
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

    def forward(
        self,
        src_rgb: torch.Tensor,
        tgt_rgb: torch.Tensor,
        flow: torch.Tensor,
        valid: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        observed_valid_override: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if observed_valid_override is not None:
            observed_valid = observed_valid_override * valid
            masked_pixels = valid * (1.0 - observed_valid)
            patch_mask = (self.patchify_mask(masked_pixels) > 0).float()
        elif patch_mask is None and str(self.config.observation_mask_mode).lower() == "patch":
            patch_mask = self.sample_patch_mask(valid)
            masked_pixels = F.interpolate(
                patch_mask.view(flow.shape[0], 1, self.grid_size, self.grid_size),
                scale_factor=self.patch_size,
                mode="nearest",
            )[:, 0] * valid
            observed_valid = valid * (1.0 - masked_pixels)
        else:
            observed_valid, patch_mask, masked_pixels = self.sample_observation_mask(valid)

        rgb, flow_input = self.build_inputs(src_rgb, tgt_rgb, flow, observed_valid)
        tokens = self.rgb_embed(rgb) + self.flow_embed(flow_input) + self.encoder_pos_embed
        encoded = self.encoder(tokens)
        decoded = self.decoder(self.encoder_to_decoder(encoded) + self.decoder_pos_embed)
        pred_flow = self.unpatchify_flow(self.flow_head(decoded))

        loss = self.compute_loss(pred_flow, flow, masked_pixels)
        observed_pixels = observed_valid

        return {
            "loss": loss,
            "pred_flow": pred_flow,
            "patch_mask": patch_mask,
            "masked_pixels": masked_pixels,
            "observed_pixels": observed_pixels,
            "observed_valid": observed_valid[:, 0],
            "flow_input": flow_input[:, :2],
            "encoded_tokens": encoded,
        }
