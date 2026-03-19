"""Minimal models for Point Odyssey flow smoke test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class _UNetEncoder(nn.Module):
    """Simple 5-level conv encoder used by both models."""

    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.enc5 = ConvBlock(c * 8, c * 8)

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2, stride=2))
        e3 = self.enc3(F.max_pool2d(e2, kernel_size=2, stride=2))
        e4 = self.enc4(F.max_pool2d(e3, kernel_size=2, stride=2))
        e5 = self.enc5(F.max_pool2d(e4, kernel_size=2, stride=2))
        return e1, e2, e3, e4, e5


def _broadcast_dt(dt: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if dt.dim() == 2:
        dt = dt[:, 0]
    return dt.view(-1, 1, 1, 1).to(dtype=target.dtype, device=target.device).expand(
        -1, 1, target.shape[2], target.shape[3]
    )


@dataclass
class VAEOutput:
    flow: torch.Tensor
    mu: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None


class DeterministicUNet(nn.Module):
    """Deterministic conditional U-Net with image + dt conditioning."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.base_channels = base_channels
        self.encoder = _UNetEncoder(in_channels + 1, base_channels=base_channels)
        c = base_channels
        self.up4 = UpBlock(c * (8 + 8), c * 8)
        self.up3 = UpBlock(c * (8 + 4), c * 4)
        self.up2 = UpBlock(c * (4 + 2), c * 2)
        self.up1 = UpBlock(c * (2 + 1), c)
        self.out = nn.Conv2d(c, 2, kernel_size=1)

    def forward(self, src_img: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        dt_map = _broadcast_dt(dt, src_img)
        x = torch.cat([src_img, dt_map], dim=1)
        e1, e2, e3, e4, e5 = self.encoder(x)
        x = self.up4(e5, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        return self.out(x)


class ConditionalFlowVAE(nn.Module):
    """Conditional flow VAE that uses a small image-conditioned latent."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, z_dim: int = 32):
        super().__init__()
        if z_dim < 1:
            raise ValueError(f"z_dim must be >=1, got {z_dim}")

        self.base_channels = base_channels
        self.z_dim = int(z_dim)

        self.image_encoder = _UNetEncoder(in_channels + 1, base_channels=base_channels)
        self.flow_encoder = _UNetEncoder(in_channels=3, base_channels=max(16, base_channels // 2))
        self.flow_encoder_factor = max(1, base_channels // 2)

        flow_ch = self.flow_encoder_factor * 8
        img_ch = base_channels * 8
        fused_ch = img_ch + flow_ch + 1

        self.to_latent = ConvBlock(fused_ch, 2 * img_ch)
        latent_flat = 2 * img_ch
        self.mu = nn.Linear(latent_flat, z_dim)
        self.logvar = nn.Linear(latent_flat, z_dim)
        self.z_to_feat = nn.Sequential(
            nn.Linear(z_dim, img_ch // 2),
            nn.ReLU(inplace=True),
            nn.Linear(img_ch // 2, base_channels * 2),
        )

        self.up4 = UpBlock(img_ch + base_channels * 2 + 1 + img_ch, img_ch)
        self.up3 = UpBlock(img_ch + base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2 + base_channels, base_channels)
        self.out = nn.Conv2d(base_channels, 2, kernel_size=1)

    def encode(
        self,
        src_img: torch.Tensor,
        dt: torch.Tensor,
        flow_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt_img = _broadcast_dt(dt, src_img)
        img_in = torch.cat([src_img, dt_img], dim=1)
        _, _, _, _, img_e5 = self.image_encoder(img_in)
        dt_img = _broadcast_dt(dt, img_e5)

        dt_flow = _broadcast_dt(dt, flow_gt)
        flow_in = torch.cat([flow_gt, dt_flow], dim=1)
        _, _, _, _, flow_e5 = self.flow_encoder(flow_in)
        dt_flow = _broadcast_dt(dt, flow_e5)

        fused = torch.cat([img_e5, flow_e5, dt_img], dim=1)
        fused = self.to_latent(fused)
        fused = fused.mean(dim=(2, 3))
        mu = self.mu(fused)
        logvar = self.logvar(fused)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        src_img: torch.Tensor,
        dt: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        # dt in shape [B] or [B,1]
        dt_map = _broadcast_dt(dt, src_img)
        x = torch.cat([src_img, dt_map], dim=1)
        e1, e2, e3, e4, e5 = self.image_encoder(x)
        dt_map = _broadcast_dt(dt, e5)

        z_vec = self.z_to_feat(z).view(z.shape[0], -1, 1, 1)
        z_map = z_vec.expand(-1, -1, e5.shape[2], e5.shape[3])
        x = torch.cat([e5, z_map, dt_map], dim=1)
        x = self.up4(x, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        return self.out(x)

    def sample(self, src_img: torch.Tensor, dt: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decode(src_img, dt, z)

    def forward(
        self,
        src_img: torch.Tensor,
        dt: torch.Tensor,
        flow_gt: Optional[torch.Tensor] = None,
        n_samples: int = 1,
    ) -> VAEOutput:
        if flow_gt is None:
            mu = torch.zeros(src_img.shape[0], self.z_dim, device=src_img.device, dtype=src_img.dtype)
            logvar = torch.zeros_like(mu)
            z = self.reparameterize(mu, logvar)
            flow = self.decode(src_img, dt, z)
            return VAEOutput(flow=flow, mu=mu, logvar=logvar)

        mu, logvar = self.encode(src_img, dt, flow_gt)
        if n_samples <= 1:
            z = self.reparameterize(mu, logvar)
            flow = self.decode(src_img, dt, z)
            return VAEOutput(flow=flow, mu=mu, logvar=logvar)

        # Multiple samples for eval-time diversity.
        flows = []
        for _ in range(int(n_samples)):
            z = self.reparameterize(mu, logvar)
            flows.append(self.decode(src_img, dt, z))
        return VAEOutput(flow=torch.stack(flows, dim=0), mu=mu, logvar=logvar)

