from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino_context_model import PatchEmbed, TransformerStack


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CorrespondenceReprModelConfig:
    # Spatial layout
    image_size: int = 256
    patch_size: int = 16

    # Input dims
    context_feature_dim: int = 768  # DINO spatial token dim

    # Encoder
    encoder_dim: int = 768
    encoder_depth: int = 8
    encoder_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Decoder
    decoder_dim: int = 512
    decoder_depth: int = 4
    decoder_heads: int = 8

    # Readout / projector
    num_readout_tokens: int = 256
    projector_hidden_dim: int = 768
    projector_output_dim: int = 256
    normalize_projector_output: bool = True

    # Side-factor projectors (pre-encoder, diagnostic)
    rgb_projector_dim: int = 128
    sem_projector_dim: int = 128

    # RGB reconstruction (pre-encoder grounding)
    rgb_recon_enabled: bool = True

    # Supervision-density token
    sup_token_enabled: bool = True

    # Flow reconstruction
    flow_channels: int = 2
    loss: str = "smooth_l1"
    smooth_l1_beta: float = 1.0

    # Evidence embedding
    evidence_flow_dim: int = 512
    evidence_mask_dim: int = 192
    evidence_density_dim: int = 64
    evidence_norm_eps: float = 0.05

    # Evidence sampler — anchor
    anchor_dense_prob: float = 0.50
    anchor_light_mask_prob: float = 0.25
    anchor_speckle_prob: float = 0.15
    anchor_keypoint_prob: float = 0.10
    anchor_light_mask_ratio_min: float = 0.0
    anchor_light_mask_ratio_max: float = 0.10
    anchor_speckle_keep_ratio: float = 0.80
    anchor_keypoint_count_min: int = 200
    anchor_keypoint_count_max: int = 500

    # Evidence sampler — student
    student_moderate_mask_prob: float = 0.15
    student_heavy_mask_prob: float = 0.25
    student_speckle_prob: float = 0.20
    student_patch_speckle_prob: float = 0.15
    student_keypoint_prob: float = 0.10
    student_full_mask_prob: float = 0.15
    student_moderate_mask_ratio_min: float = 0.30
    student_moderate_mask_ratio_max: float = 0.60
    student_heavy_mask_ratio_min: float = 0.70
    student_heavy_mask_ratio_max: float = 0.95
    student_speckle_keep_ratio: float = 0.10
    student_patch_speckle_mask_ratio_min: float = 0.30
    student_patch_speckle_mask_ratio_max: float = 0.60
    student_patch_speckle_keep_ratio: float = 0.20
    student_keypoint_count_min: int = 30
    student_keypoint_count_max: int = 200


# ---------------------------------------------------------------------------
# Small helper modules
# ---------------------------------------------------------------------------

class ProjectorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


# ---------------------------------------------------------------------------
# Evidence Sampler
# ---------------------------------------------------------------------------

class EvidenceSampler:
    """Samples observation masks from dense GT valid masks.

    All operations are vectorized over the batch dimension — no per-sample
    Python for-loops.
    """

    def __init__(self, cfg: CorrespondenceReprModelConfig, patch_size: int,
                 grid_size: int) -> None:
        self.cfg = cfg
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size

    # -- low-level helpers (fully batched) -----------------------------------

    @staticmethod
    def _patchify_mask(valid: torch.Tensor, patch_size: int,
                       num_patches: int) -> torch.Tensor:
        patch_valid = F.avg_pool2d(valid.unsqueeze(1),
                                   kernel_size=patch_size, stride=patch_size)
        return (patch_valid[:, 0] > 0).float().reshape(valid.shape[0],
                                                        num_patches)

    @staticmethod
    def _expand_patch_mask(patch_mask: torch.Tensor, grid_size: int,
                           patch_size: int) -> torch.Tensor:
        return F.interpolate(
            patch_mask.view(patch_mask.shape[0], 1, grid_size, grid_size),
            scale_factor=patch_size, mode="nearest",
        )[:, 0]

    # -- regime implementations (batched) ------------------------------------

    def _dense(self, valid: torch.Tensor) -> torch.Tensor:
        return valid.clone()

    def _patch_mask(self, valid: torch.Tensor, ratio_min: float,
                    ratio_max: float) -> torch.Tensor:
        """Mask random patches — fully batched via random noise sorting."""
        ratio_min = max(0.0, min(1.0, ratio_min))
        ratio_max = max(0.0, min(1.0, ratio_max))
        if ratio_max < ratio_min:
            ratio_min, ratio_max = ratio_max, ratio_min
        # Per-sample random ratio
        bsz = valid.shape[0]
        device = valid.device
        ratios = torch.empty(bsz, device=device).uniform_(ratio_min, ratio_max)

        patch_valid = self._patchify_mask(valid, self.patch_size,
                                          self.num_patches)
        # Random noise, but invalid patches get -inf so they sort last
        noise = torch.rand_like(patch_valid)
        noise = noise * patch_valid + (-1.0) * (1.0 - patch_valid)
        # Sort descending — highest noise values first (these get masked)
        _, indices = noise.sort(dim=1, descending=True)

        # Number of valid patches per sample
        n_valid = patch_valid.sum(dim=1)  # [B]
        # Number to mask (at most n_valid - 1 to keep at least one)
        n_mask = (ratios * n_valid).ceil().long()
        n_mask = torch.clamp(n_mask, max=(n_valid - 1).long().clamp_min(0))

        # Build mask: position in sorted order < n_mask means masked
        positions = torch.arange(self.num_patches, device=device).unsqueeze(0)
        sorted_mask = (positions < n_mask.unsqueeze(1)).float()

        # Scatter back to original patch positions
        patch_masked = torch.zeros_like(patch_valid)
        patch_masked.scatter_(1, indices, sorted_mask)

        pixel_masked = self._expand_patch_mask(patch_masked, self.grid_size,
                                               self.patch_size)
        return valid * (1.0 - pixel_masked)

    def _speckle(self, valid: torch.Tensor,
                 keep_ratio: float) -> torch.Tensor:
        """IID per-pixel subsampling — fully batched."""
        keep_ratio = max(0.0, min(1.0, keep_ratio))
        if keep_ratio >= 1.0:
            return valid.clone()
        observed = ((torch.rand_like(valid) < keep_ratio) &
                    (valid > 0)).float()
        # Ensure at least one pixel per sample (batched)
        empty = (observed.sum(dim=(1, 2)) == 0) & (valid.sum(dim=(1, 2)) > 0)
        if empty.any():
            # For empty samples, pick a random valid pixel
            flat_valid = valid[empty].view(empty.sum(), -1)
            # Replace zeros with large values so argmin of (rand * invalid)
            # doesn't pick them
            noise = torch.rand_like(flat_valid)
            noise = noise + (1.0 - flat_valid) * 2.0  # invalid → large
            min_idx = noise.argmin(dim=1)
            flat_obs = observed[empty].view(empty.sum(), -1)
            flat_obs.scatter_(1, min_idx.unsqueeze(1), 1.0)
            observed[empty] = flat_obs.view_as(valid[empty])
        return observed

    def _patch_speckle(self, valid: torch.Tensor, mask_ratio_min: float,
                       mask_ratio_max: float,
                       speckle_keep: float) -> torch.Tensor:
        after_patch = self._patch_mask(valid, mask_ratio_min, mask_ratio_max)
        return self._speckle(after_patch, speckle_keep)

    def _keypoint(self, valid: torch.Tensor, count_min: int,
                  count_max: int) -> torch.Tensor:
        """Sample scattered valid pixels — batched via noise ranking."""
        bsz = valid.shape[0]
        device = valid.device
        count = torch.randint(count_min, count_max + 1, (1,),
                              device=device).item()

        flat_valid = valid.view(bsz, -1)  # [B, H*W]
        # Random noise; invalid pixels get -1 so they rank last
        noise = torch.rand_like(flat_valid)
        noise = noise * flat_valid + (-1.0) * (1.0 - flat_valid)
        # Top-k by noise value = random selection of valid pixels
        n_valid = flat_valid.sum(dim=1).long()  # [B]
        k = min(int(count), int(flat_valid.shape[1]))
        if k <= 0:
            return torch.zeros_like(valid)

        # Only keep positions < min(count, n_valid_per_sample)
        keep_limit = n_valid.clamp_max(k).unsqueeze(1)
        positions = torch.arange(k, device=device).unsqueeze(0)
        keep_mask = (positions < keep_limit).float()

        flat_observed = torch.zeros_like(flat_valid)
        _, indices = noise.topk(k=k, dim=1, largest=True, sorted=True)
        flat_observed.scatter_(1, indices, keep_mask)
        # Zero out invalid pixels (safety)
        flat_observed = flat_observed * flat_valid
        return flat_observed.view_as(valid)

    def _full_mask(self, valid: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(valid)

    # -- branch-level samplers (batched regime assignment) -------------------

    @staticmethod
    def _sample_regimes_batched(probs: list[float],
                                bsz: int,
                                device: torch.device) -> torch.Tensor:
        """Sample regime index per sample — returns [B] long tensor."""
        weights = torch.tensor(probs, device=device, dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-8)
        return torch.multinomial(weights.unsqueeze(0).expand(bsz, -1),
                                 num_samples=1).squeeze(1)

    def sample_anchor(self, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        c = self.cfg
        bsz = valid.shape[0]
        device = valid.device
        probs = [c.anchor_dense_prob, c.anchor_light_mask_prob,
                 c.anchor_speckle_prob, c.anchor_keypoint_prob]
        regimes = self._sample_regimes_batched(probs, bsz, device)
        observed = torch.zeros_like(valid)

        dense_mask = regimes == 0
        if dense_mask.any():
            observed[dense_mask] = self._dense(valid[dense_mask])

        patch_mask = regimes == 1
        if patch_mask.any():
            observed[patch_mask] = self._patch_mask(
                valid[patch_mask],
                c.anchor_light_mask_ratio_min,
                c.anchor_light_mask_ratio_max,
            )

        speckle_mask = regimes == 2
        if speckle_mask.any():
            observed[speckle_mask] = self._speckle(
                valid[speckle_mask],
                c.anchor_speckle_keep_ratio,
            )

        keypoint_mask = regimes == 3
        if keypoint_mask.any():
            observed[keypoint_mask] = self._keypoint(
                valid[keypoint_mask],
                c.anchor_keypoint_count_min,
                c.anchor_keypoint_count_max,
            )
        return self._build_result(valid, observed)

    def sample_student(self, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        c = self.cfg
        bsz = valid.shape[0]
        device = valid.device
        probs = [c.student_moderate_mask_prob, c.student_heavy_mask_prob,
                 c.student_speckle_prob, c.student_patch_speckle_prob,
                 c.student_keypoint_prob, c.student_full_mask_prob]
        regimes = self._sample_regimes_batched(probs, bsz, device)
        observed = torch.zeros_like(valid)

        moderate_mask = regimes == 0
        if moderate_mask.any():
            observed[moderate_mask] = self._patch_mask(
                valid[moderate_mask],
                c.student_moderate_mask_ratio_min,
                c.student_moderate_mask_ratio_max,
            )

        heavy_mask = regimes == 1
        if heavy_mask.any():
            observed[heavy_mask] = self._patch_mask(
                valid[heavy_mask],
                c.student_heavy_mask_ratio_min,
                c.student_heavy_mask_ratio_max,
            )

        speckle_mask = regimes == 2
        if speckle_mask.any():
            observed[speckle_mask] = self._speckle(
                valid[speckle_mask],
                c.student_speckle_keep_ratio,
            )

        patch_speckle_mask = regimes == 3
        if patch_speckle_mask.any():
            observed[patch_speckle_mask] = self._patch_speckle(
                valid[patch_speckle_mask],
                c.student_patch_speckle_mask_ratio_min,
                c.student_patch_speckle_mask_ratio_max,
                c.student_patch_speckle_keep_ratio,
            )

        keypoint_mask = regimes == 4
        if keypoint_mask.any():
            observed[keypoint_mask] = self._keypoint(
                valid[keypoint_mask],
                c.student_keypoint_count_min,
                c.student_keypoint_count_max,
            )

        full_mask = regimes == 5
        if full_mask.any():
            observed[full_mask] = self._full_mask(valid[full_mask])
        return self._build_result(valid, observed)

    def _build_result(self, valid: torch.Tensor,
                      observed: torch.Tensor) -> dict[str, torch.Tensor]:
        valid_pixels = valid.sum(dim=(1, 2))
        observed_pixels = observed.sum(dim=(1, 2))
        visible_ratio = torch.where(
            valid_pixels > 0,
            observed_pixels / valid_pixels.clamp_min(1.0),
            torch.zeros_like(valid_pixels),
        )
        return {
            "observed_valid": observed,
            "visible_ratio": visible_ratio,
        }


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CorrespondenceReprModel(nn.Module):
    def __init__(self, config: CorrespondenceReprModelConfig) -> None:
        super().__init__()
        self.config = config
        image_size = int(config.image_size)
        patch_size = int(config.patch_size)
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by "
                f"patch_size={patch_size}")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.flow_patch_dim = int(config.flow_channels) * patch_size * patch_size
        D = int(config.encoder_dim)

        # -- Embedding layers ------------------------------------------------

        # RGB: shared weights for src and tgt
        self.rgb_embed = PatchEmbed(in_channels=3, patch_size=patch_size,
                                    embed_dim=D)
        # DINO: shared projection for src and tgt
        self.context_norm = nn.LayerNorm(int(config.context_feature_dim))
        self.context_proj = nn.Linear(int(config.context_feature_dim), D)

        # Evidence: split flow, mask, and density before fusion.
        evidence_flow_dim = int(config.evidence_flow_dim)
        evidence_mask_dim = int(config.evidence_mask_dim)
        evidence_density_dim = int(config.evidence_density_dim)
        self.evidence_flow_embed = PatchEmbed(
            in_channels=int(config.flow_channels),
            patch_size=patch_size,
            embed_dim=evidence_flow_dim,
        )
        self.evidence_mask_embed = PatchEmbed(
            in_channels=1,
            patch_size=patch_size,
            embed_dim=evidence_mask_dim,
        )
        self.evidence_density_proj = nn.Linear(1, evidence_density_dim)
        evidence_fused_dim = (evidence_flow_dim + evidence_mask_dim
                              + evidence_density_dim)
        self.evidence_fuse_norm = nn.LayerNorm(evidence_fused_dim)
        self.evidence_fuse = nn.Linear(evidence_fused_dim, D)
        self.empty_flow_token = nn.Parameter(
            torch.zeros(1, 1, evidence_flow_dim))

        # -- Positional & type embeddings ------------------------------------

        # One shared spatial pos embed for all 256-token groups
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, D))

        # Source / target distinction
        self.src_type_embed = nn.Parameter(torch.zeros(1, 1, D))
        self.tgt_type_embed = nn.Parameter(torch.zeros(1, 1, D))

        # Modality distinction
        self.rgb_type_embed = nn.Parameter(torch.zeros(1, 1, D))
        self.dino_type_embed = nn.Parameter(torch.zeros(1, 1, D))
        self.evidence_type_embed = nn.Parameter(torch.zeros(1, 1, D))
        self.readout_type_embed = nn.Parameter(torch.zeros(1, 1, D))

        # Readout tokens (spatially aligned)
        N_readout = int(config.num_readout_tokens)
        self.readout_embed = nn.Parameter(torch.zeros(1, N_readout, D))

        # Supervision density token
        if config.sup_token_enabled:
            self.sup_type_embed = nn.Parameter(torch.zeros(1, 1, D))
            self.sup_base_embed = nn.Parameter(torch.zeros(1, 1, D))
            self.sup_ratio_proj = nn.Linear(1, D)
            self.sup_head = nn.Sequential(
                nn.Linear(D, 1),
                nn.Sigmoid(),
            )

        # -- Encoder ---------------------------------------------------------

        self.encoder = TransformerStack(
            dim=D,
            depth=int(config.encoder_depth),
            num_heads=int(config.encoder_heads),
            mlp_ratio=float(config.mlp_ratio),
            dropout=float(config.dropout),
            attention_dropout=float(config.attention_dropout),
        )

        # -- Decoder ---------------------------------------------------------

        Dd = int(config.decoder_dim)
        self.enc2dec_rgb = nn.Linear(D, Dd)
        self.enc2dec_dino = nn.Linear(D, Dd)
        self.enc2dec_evidence = nn.Linear(D, Dd)

        self.decoder_spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, Dd))
        self.decoder_src_type_embed = nn.Parameter(torch.zeros(1, 1, Dd))
        self.decoder_tgt_type_embed = nn.Parameter(torch.zeros(1, 1, Dd))
        self.decoder_rgb_type_embed = nn.Parameter(torch.zeros(1, 1, Dd))
        self.decoder_dino_type_embed = nn.Parameter(torch.zeros(1, 1, Dd))
        self.decoder_evidence_type_embed = nn.Parameter(torch.zeros(1, 1, Dd))

        self.decoder = TransformerStack(
            dim=Dd,
            depth=int(config.decoder_depth),
            num_heads=int(config.decoder_heads),
            mlp_ratio=float(config.mlp_ratio),
            dropout=float(config.dropout),
            attention_dropout=float(config.attention_dropout),
        )
        self.flow_head = nn.Linear(Dd, self.flow_patch_dim)

        # -- Latent projectors -----------------------------------------------

        # z_corr: post-encoder, from readout tokens
        self.corr_projector = ProjectorMLP(
            D, int(config.projector_hidden_dim),
            int(config.projector_output_dim),
            dropout=float(config.dropout))

        # z_rgb: pre-encoder, from RGB patch embeddings
        self.rgb_projector = ProjectorMLP(D, D // 2,
                                          int(config.rgb_projector_dim))

        # z_sem: pre-encoder, from DINO projections
        self.sem_projector = ProjectorMLP(D, D // 2,
                                          int(config.sem_projector_dim))

        # RGB reconstruction head (pre-encoder grounding)
        if config.rgb_recon_enabled:
            rgb_patch_dim = 3 * patch_size * patch_size  # per-image, not pair
            self.rgb_recon_head = nn.Linear(D, rgb_patch_dim)

        # -- Evidence sampler ------------------------------------------------
        self.evidence_sampler = EvidenceSampler(config, patch_size,
                                                self.grid_size)

        self.initialize_weights()

    # -------------------------------------------------------------------
    # Init
    # -------------------------------------------------------------------

    def initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.readout_embed, std=0.02)
        nn.init.trunc_normal_(self.src_type_embed, std=0.02)
        nn.init.trunc_normal_(self.tgt_type_embed, std=0.02)
        nn.init.trunc_normal_(self.rgb_type_embed, std=0.02)
        nn.init.trunc_normal_(self.dino_type_embed, std=0.02)
        nn.init.trunc_normal_(self.evidence_type_embed, std=0.02)
        nn.init.trunc_normal_(self.readout_type_embed, std=0.02)
        nn.init.trunc_normal_(self.empty_flow_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_src_type_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_tgt_type_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_rgb_type_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_dino_type_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_evidence_type_embed, std=0.02)
        if self.config.sup_token_enabled:
            nn.init.trunc_normal_(self.sup_type_embed, std=0.02)
            nn.init.trunc_normal_(self.sup_base_embed, std=0.02)
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

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def flatten_context(self, ctx: torch.Tensor) -> torch.Tensor:
        """[B, gh, gw, dim] or [B, N, dim] → [B, num_patches, dim]."""
        if ctx.dim() == 4:
            bsz, gh, gw, dim = ctx.shape
            ctx = ctx.reshape(bsz, gh * gw, dim)
        if ctx.shape[1] != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} DINO tokens, got {ctx.shape[1]}")
        return ctx

    def unpatchify_flow(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz = tokens.shape[0]
        p = self.patch_size
        fc = int(self.config.flow_channels)
        x = tokens.view(bsz, self.grid_size, self.grid_size, fc, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(bsz, fc, self.image_size, self.image_size)

    def unpatchify_rgb(self, tokens: torch.Tensor) -> torch.Tensor:
        """Unpatchify single-image RGB from [B, N, 3*p*p]."""
        bsz = tokens.shape[0]
        p = self.patch_size
        x = tokens.view(bsz, self.grid_size, self.grid_size, 3, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(bsz, 3, self.image_size, self.image_size)

    # -------------------------------------------------------------------
    # Encoder input assembly
    # -------------------------------------------------------------------

    def embed_rgb(self, src_rgb: torch.Tensor,
                  tgt_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed src and tgt RGB into token sequences."""
        src_tok = (self.rgb_embed(src_rgb)
                   + self.spatial_pos_embed
                   + self.src_type_embed + self.rgb_type_embed)
        tgt_tok = (self.rgb_embed(tgt_rgb)
                   + self.spatial_pos_embed
                   + self.tgt_type_embed + self.rgb_type_embed)
        return src_tok, tgt_tok

    def embed_dino(self, src_dino: torch.Tensor,
                   tgt_dino: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed src and tgt DINO spatial tokens."""
        src_flat = self.flatten_context(src_dino)
        tgt_flat = self.flatten_context(tgt_dino)
        src_tok = (self.context_proj(self.context_norm(src_flat))
                   + self.spatial_pos_embed
                   + self.src_type_embed + self.dino_type_embed)
        tgt_tok = (self.context_proj(self.context_norm(tgt_flat))
                   + self.spatial_pos_embed
                   + self.tgt_type_embed + self.dino_type_embed)
        return src_tok, tgt_tok

    def evidence_patch_valid_fraction(
        self, observed_valid: torch.Tensor
    ) -> torch.Tensor:
        patch_valid = F.avg_pool2d(
            observed_valid.unsqueeze(1),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return patch_valid[:, 0].reshape(observed_valid.shape[0],
                                         self.num_patches, 1)

    def embed_evidence_flow(
        self,
        observed_flow: torch.Tensor,
        observed_valid: torch.Tensor,
        patch_valid_fraction: torch.Tensor,
    ) -> torch.Tensor:
        masked_flow = observed_flow * observed_valid.unsqueeze(1)
        flow_tok = self.evidence_flow_embed(masked_flow)
        flow_tok = flow_tok / patch_valid_fraction.clamp_min(
            float(self.config.evidence_norm_eps))
        empty_mask = patch_valid_fraction <= 0
        if empty_mask.any():
            flow_tok = torch.where(
                empty_mask,
                self.empty_flow_token.expand(flow_tok.shape[0],
                                             self.num_patches, -1),
                flow_tok,
            )
        return flow_tok

    def embed_evidence_mask(self, observed_valid: torch.Tensor) -> torch.Tensor:
        return self.evidence_mask_embed(observed_valid.unsqueeze(1))

    def embed_evidence_density(
        self, patch_valid_fraction: torch.Tensor
    ) -> torch.Tensor:
        return self.evidence_density_proj(patch_valid_fraction)

    def fuse_evidence(
        self,
        flow_tok: torch.Tensor,
        mask_tok: torch.Tensor,
        density_tok: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([flow_tok, mask_tok, density_tok], dim=-1)
        fused = self.evidence_fuse_norm(fused)
        fused = self.evidence_fuse(fused)
        return (fused + self.spatial_pos_embed + self.evidence_type_embed)

    def embed_evidence(self, observed_flow: torch.Tensor,
                       observed_valid: torch.Tensor
                       ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Embed sparse flow evidence with explicit mask/density structure."""
        patch_valid_fraction = self.evidence_patch_valid_fraction(
            observed_valid)
        flow_tok = self.embed_evidence_flow(
            observed_flow, observed_valid, patch_valid_fraction)
        mask_tok = self.embed_evidence_mask(observed_valid)
        density_tok = self.embed_evidence_density(patch_valid_fraction)
        evidence_tok = self.fuse_evidence(flow_tok, mask_tok, density_tok)
        evidence_stats = {
            "evidence_patch_visible_mean": patch_valid_fraction.mean(),
            "evidence_patch_empty_frac": (patch_valid_fraction <= 0)
                .float().mean(),
            "evidence_flow_token_norm": flow_tok.norm(dim=-1).mean(),
            "evidence_mask_token_norm": mask_tok.norm(dim=-1).mean(),
            "evidence_density_token_norm": density_tok.norm(dim=-1).mean(),
            "evidence_fused_token_norm": evidence_tok.norm(dim=-1).mean(),
        }
        return evidence_tok, evidence_stats

    def embed_sup_token(self, visible_ratio: torch.Tensor) -> torch.Tensor:
        """Build supervision density token from scalar visible_ratio [B]."""
        ratio_input = visible_ratio.unsqueeze(-1).unsqueeze(-1)  # [B,1,1]
        return (self.sup_base_embed
                + self.sup_ratio_proj(ratio_input.squeeze(-1))
                    .unsqueeze(1)  # [B,1,D]
                + self.sup_type_embed)

    def embed_readout(self, bsz: int,
                      device: torch.device) -> torch.Tensor:
        """Expand readout tokens for batch."""
        N = int(self.config.num_readout_tokens)
        tok = self.readout_embed.expand(bsz, N, -1)
        # Use spatial pos embed if readout count matches num_patches,
        # otherwise use a slice or skip
        if N == self.num_patches:
            tok = tok + self.spatial_pos_embed
        tok = tok + self.readout_type_embed
        return tok

    # -------------------------------------------------------------------
    # Forward: branch
    # -------------------------------------------------------------------

    def forward_branch(
        self,
        src_rgb: torch.Tensor,
        tgt_rgb: torch.Tensor,
        src_dino: torch.Tensor,
        tgt_dino: torch.Tensor,
        observed_flow: torch.Tensor,
        observed_valid: torch.Tensor,
        visible_ratio: torch.Tensor,
        *,
        return_latent: bool = True,
        decode: bool = True,
        return_pre_encoder: bool = False,
    ) -> dict[str, Any]:
        bsz = src_rgb.shape[0]
        device = src_rgb.device

        # -- Pre-encoder embeddings ------------------------------------------

        src_rgb_tok, tgt_rgb_tok = self.embed_rgb(src_rgb, tgt_rgb)
        src_dino_tok, tgt_dino_tok = self.embed_dino(src_dino, tgt_dino)
        evidence_tok, evidence_stats = self.embed_evidence(
            observed_flow, observed_valid)
        readout_tok = self.embed_readout(bsz, device)

        # Pre-encoder side-factor readouts (before any mixing)
        pre_encoder_outputs: dict[str, Any] = {}
        if return_pre_encoder:
            # z_rgb: pool src+tgt RGB embeddings (before type/pos for purity,
            # but we use the post-embed tokens for practical reasons — the
            # pos/type embeds are small relative to content)
            rgb_pooled = torch.cat([src_rgb_tok, tgt_rgb_tok],
                                    dim=1).mean(dim=1)
            pre_encoder_outputs["z_rgb"] = self.rgb_projector(rgb_pooled)

            dino_pooled = torch.cat([src_dino_tok, tgt_dino_tok],
                                     dim=1).mean(dim=1)
            pre_encoder_outputs["z_sem"] = self.sem_projector(dino_pooled)

            # RGB reconstruction targets (pre-encoder token content)
            if self.config.rgb_recon_enabled:
                # Store raw embeddings (without pos/type) for reconstruction
                src_rgb_raw = self.rgb_embed(src_rgb)
                tgt_rgb_raw = self.rgb_embed(tgt_rgb)
                pre_encoder_outputs["src_rgb_tokens_raw"] = src_rgb_raw
                pre_encoder_outputs["tgt_rgb_tokens_raw"] = tgt_rgb_raw

        # -- Build encoder sequence ------------------------------------------

        encoder_parts = [readout_tok]

        if self.config.sup_token_enabled:
            sup_tok = self.embed_sup_token(visible_ratio)
            encoder_parts.append(sup_tok)

        encoder_parts.extend([
            src_rgb_tok, tgt_rgb_tok,
            src_dino_tok, tgt_dino_tok,
            evidence_tok,
        ])

        encoder_input = torch.cat(encoder_parts, dim=1)
        encoded = self.encoder(encoder_input)

        # -- Slice encoded output --------------------------------------------

        N_readout = int(self.config.num_readout_tokens)
        idx = N_readout
        encoded_readout = encoded[:, :idx]

        if self.config.sup_token_enabled:
            encoded_sup = encoded[:, idx:idx+1]
            idx += 1
        else:
            encoded_sup = None

        N = self.num_patches
        encoded_src_rgb = encoded[:, idx:idx+N]; idx += N
        encoded_tgt_rgb = encoded[:, idx:idx+N]; idx += N
        encoded_src_dino = encoded[:, idx:idx+N]; idx += N
        encoded_tgt_dino = encoded[:, idx:idx+N]; idx += N
        encoded_evidence = encoded[:, idx:idx+N]; idx += N

        outputs: dict[str, Any] = {
            "encoded_readout": encoded_readout,
            "encoded_evidence": encoded_evidence,
            **evidence_stats,
            **pre_encoder_outputs,
        }

        # -- Latent readout --------------------------------------------------

        if return_latent:
            pair_latent = encoded_readout.mean(dim=1)  # [B, D]
            projected = self.corr_projector(pair_latent)
            if self.config.normalize_projector_output:
                projected = F.normalize(projected, dim=-1)
            outputs["pair_latent"] = pair_latent
            outputs["projected_latent"] = projected

            # Supervision density prediction
            if self.config.sup_token_enabled and encoded_sup is not None:
                outputs["pred_visible_ratio"] = self.sup_head(
                    encoded_sup[:, 0]).squeeze(-1)

        # -- Decoder ---------------------------------------------------------

        if decode:
            dec_src_rgb = (self.enc2dec_rgb(encoded_src_rgb)
                           + self.decoder_spatial_pos_embed
                           + self.decoder_src_type_embed
                           + self.decoder_rgb_type_embed)
            dec_tgt_rgb = (self.enc2dec_rgb(encoded_tgt_rgb)
                           + self.decoder_spatial_pos_embed
                           + self.decoder_tgt_type_embed
                           + self.decoder_rgb_type_embed)
            dec_src_dino = (self.enc2dec_dino(encoded_src_dino)
                            + self.decoder_spatial_pos_embed
                            + self.decoder_src_type_embed
                            + self.decoder_dino_type_embed)
            dec_tgt_dino = (self.enc2dec_dino(encoded_tgt_dino)
                            + self.decoder_spatial_pos_embed
                            + self.decoder_tgt_type_embed
                            + self.decoder_dino_type_embed)
            dec_evidence = (self.enc2dec_evidence(encoded_evidence)
                            + self.decoder_spatial_pos_embed
                            + self.decoder_evidence_type_embed)

            decoder_input = torch.cat([
                dec_src_rgb, dec_tgt_rgb,
                dec_src_dino, dec_tgt_dino,
                dec_evidence,
            ], dim=1)  # [B, 1280, Dd]

            decoded = self.decoder(decoder_input)
            # Flow prediction from evidence positions (last 256 tokens)
            pred_flow = self.flow_head(decoded[:, -N:])
            outputs["pred_flow"] = self.unpatchify_flow(pred_flow)

        return outputs

    # -------------------------------------------------------------------
    # Loss helpers
    # -------------------------------------------------------------------

    def compute_reconstruction_loss(
        self,
        pred_flow: torch.Tensor,
        target_flow: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruction loss over all valid pixels."""
        mask = valid.unsqueeze(1)
        if self.config.loss == "l1":
            per_pixel = (pred_flow - target_flow).abs()
        else:
            per_pixel = F.smooth_l1_loss(
                pred_flow, target_flow,
                beta=float(self.config.smooth_l1_beta),
                reduction="none")
        denom = mask.sum().clamp_min(1.0) * pred_flow.shape[1]
        return (per_pixel * mask).sum() / denom

    def compute_rgb_reconstruction_loss(
        self,
        src_rgb_tokens_raw: torch.Tensor,
        tgt_rgb_tokens_raw: torch.Tensor,
        src_rgb: torch.Tensor,
        tgt_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """L2 reconstruction of RGB patches from pre-encoder embeddings."""
        p = self.patch_size
        # Patchify targets: [B, 3, H, W] → [B, N, 3*p*p]
        B = src_rgb.shape[0]
        src_target = src_rgb.unfold(2, p, p).unfold(3, p, p)
        src_target = src_target.contiguous().view(B, 3, -1, p, p)
        src_target = src_target.permute(0, 2, 1, 3, 4).reshape(
            B, self.num_patches, 3 * p * p)
        tgt_target = tgt_rgb.unfold(2, p, p).unfold(3, p, p)
        tgt_target = tgt_target.contiguous().view(B, 3, -1, p, p)
        tgt_target = tgt_target.permute(0, 2, 1, 3, 4).reshape(
            B, self.num_patches, 3 * p * p)

        src_pred = self.rgb_recon_head(src_rgb_tokens_raw)
        tgt_pred = self.rgb_recon_head(tgt_rgb_tokens_raw)

        loss_src = F.mse_loss(src_pred, src_target)
        loss_tgt = F.mse_loss(tgt_pred, tgt_target)
        return (loss_src + loss_tgt) * 0.5

    @staticmethod
    def endpoint_error(pred: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
        safe_target = torch.where(mask.unsqueeze(1) > 0, target, pred.detach())
        epe = torch.linalg.vector_norm(pred - safe_target, dim=1)
        denom = mask.sum().clamp_min(1.0)
        return (epe * mask).sum() / denom
