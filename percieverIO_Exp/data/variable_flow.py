from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import FlyingThings3D
from torchvision.transforms.functional import normalize


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class VariableObservationFlowDataConfig:
    root: str
    phase: int
    train_split: str = "train"
    val_split: str = "test"
    pass_name: str = "clean"
    camera: str = "left"
    image_size: Sequence[int] = (256, 256)
    query_stride: int = 4
    rgb_patch_size: int = 3
    normalize_rgb: bool = True
    batch_size: int = 4
    val_batch_size: int = 4
    num_workers: int = 8
    prefetch_factor: Optional[int] = None
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True
    train_subset_size: Optional[int] = None
    val_subset_size: Optional[int] = None
    random_seed: int = 2021
    phase2_observed_fraction_min: float = 0.05
    phase2_observed_fraction_max: float = 0.50
    phase34_view_a_fraction_min: float = 0.25
    phase34_view_a_fraction_max: float = 1.0
    phase34_view_b_fraction_min: float = 0.005
    phase34_view_b_fraction_max: float = 0.20
    mask_mode: str = "mixed"
    smoke_overfit_batches: Optional[int] = None
    fixed_observed_fraction: Optional[float] = None
    fixed_mask_mode: Optional[str] = None


def _resize_rgb(image: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    if tuple(image.shape[-2:]) == image_size:
        return image
    return F.interpolate(image.unsqueeze(0), size=image_size, mode="bilinear", align_corners=False).squeeze(0)


def _resize_flow_and_valid(
    flow: torch.Tensor,
    valid: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = flow.shape[-2:]
    target_h, target_w = image_size
    if (h, w) == image_size:
        return flow, valid
    scale_x = target_w / float(w)
    scale_y = target_h / float(h)
    flow = F.interpolate(flow.unsqueeze(0), size=image_size, mode="bilinear", align_corners=False).squeeze(0)
    flow[0] *= scale_x
    flow[1] *= scale_y
    valid = (
        F.interpolate(valid.float().unsqueeze(0).unsqueeze(0), size=image_size, mode="nearest")
        .squeeze(0)
        .squeeze(0)
        > 0.5
    )
    flow = torch.where(valid.unsqueeze(0), flow, torch.zeros_like(flow))
    return flow, valid


def _normalize_flow(flow: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    h, w = image_size
    scale = torch.tensor([float(w), float(h)], device=flow.device, dtype=flow.dtype).view(2, 1, 1)
    return flow / scale


def _make_query_grid(image_size: tuple[int, int], stride: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = image_size
    ys = torch.arange(0, h, stride, device=device)
    xs = torch.arange(0, w, stride, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid_hw = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
    xy = torch.stack(
        [
            (xx.reshape(-1).float() / max(w - 1, 1)) * 2.0 - 1.0,
            (yy.reshape(-1).float() / max(h - 1, 1)) * 2.0 - 1.0,
        ],
        dim=-1,
    )
    return grid_hw, xy


def _extract_patch_bank(image1: torch.Tensor, image2: torch.Tensor, patch_size: int) -> torch.Tensor:
    pair = torch.cat([image1, image2], dim=0).unsqueeze(0)
    unfolded = F.unfold(pair, kernel_size=patch_size, padding=patch_size // 2)
    return unfolded.squeeze(0).transpose(0, 1)


def _sample_indices(mask: torch.Tensor, num_keep: int, generator: torch.Generator) -> torch.Tensor:
    flat = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)
    if flat.numel() == 0 or num_keep <= 0:
        return flat[:0]
    perm = torch.randperm(flat.numel(), generator=generator, device=flat.device)
    return flat[perm[: min(num_keep, flat.numel())]]


def _sample_block_indices(mask: torch.Tensor, num_keep: int, generator: torch.Generator) -> torch.Tensor:
    valid_coords = torch.nonzero(mask, as_tuple=False)
    if valid_coords.numel() == 0 or num_keep <= 0:
        return valid_coords[:0, 0]
    choice = int(torch.randint(0, valid_coords.shape[0], (1,), generator=generator, device=mask.device).item())
    cy, cx = valid_coords[choice].tolist()
    side = max(1, int(np.sqrt(max(num_keep, 1))))
    half = side // 2
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(mask.shape[0], y0 + side)
    x1 = min(mask.shape[1], x0 + side)
    block_mask = torch.zeros_like(mask)
    block_mask[y0:y1, x0:x1] = True
    flat = torch.nonzero((mask & block_mask).reshape(-1), as_tuple=False).squeeze(-1)
    if flat.numel() >= num_keep:
        perm = torch.randperm(flat.numel(), generator=generator, device=mask.device)
        return flat[perm[:num_keep]]
    return flat


def _sample_observed_indices(
    valid: torch.Tensor,
    fraction: float,
    mask_mode: str,
    generator: torch.Generator,
) -> torch.Tensor:
    total_valid = int(valid.sum().item())
    num_keep = total_valid if fraction >= 1.0 else max(1, int(round(total_valid * fraction)))
    if total_valid == 0:
        return torch.zeros(0, dtype=torch.long, device=valid.device)
    if mask_mode == "random":
        return _sample_indices(valid, num_keep, generator)
    if mask_mode == "block":
        return _sample_block_indices(valid, num_keep, generator)
    if mask_mode == "mixed":
        block = _sample_block_indices(valid, max(1, num_keep // 2), generator)
        remaining_mask = valid.reshape(-1).clone()
        if block.numel() > 0:
            remaining_mask[block] = False
        random_extra = _sample_indices(remaining_mask.reshape_as(valid), num_keep - block.numel(), generator)
        return torch.unique(torch.cat([block, random_extra], dim=0), sorted=False)
    raise ValueError(f"Unsupported mask_mode={mask_mode}")


def _build_view_tokens(
    image1: torch.Tensor,
    image2: torch.Tensor,
    flow_norm: torch.Tensor,
    valid: torch.Tensor,
    query_hw: torch.Tensor,
    query_xy: torch.Tensor,
    patch_bank: torch.Tensor,
    phase: int,
    fraction: float,
    mask_mode: str,
    generator: torch.Generator,
) -> dict[str, Any]:
    h, w = valid.shape
    flat_valid = valid.reshape(-1)
    flat_flow = flow_norm.permute(1, 2, 0).reshape(-1, 2)
    flat_rgb = torch.cat([image1, image2], dim=0).permute(1, 2, 0).reshape(-1, 6)
    flat_xy = torch.stack(
        [
            (
                torch.arange(w, device=valid.device)
                .view(1, w)
                .expand(h, w)
                .reshape(-1)
                .float()
                / max(w - 1, 1)
            )
            * 2.0
            - 1.0,
            (
                torch.arange(h, device=valid.device)
                .view(h, 1)
                .expand(h, w)
                .reshape(-1)
                .float()
                / max(h - 1, 1)
            )
            * 2.0
            - 1.0,
        ],
        dim=-1,
    )

    flow_indices = torch.zeros(0, dtype=torch.long, device=valid.device)
    if phase == 0:
        flow_indices = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
        mask_mode = "dense"
        fraction = 1.0
    elif phase in (2, 3, 4):
        flow_indices = _sample_observed_indices(valid, fraction, mask_mode, generator)

    rgb_indices = (query_hw[:, 0] * w + query_hw[:, 1]).to(torch.long)
    if phase == 0:
        rgb_indices = rgb_indices[:0]
    rgb_tokens = torch.cat(
        [
            torch.zeros((rgb_indices.numel(), 2), device=valid.device, dtype=image1.dtype),
            flat_xy[rgb_indices],
            torch.zeros((rgb_indices.numel(), 2), device=valid.device, dtype=image1.dtype),
            flat_rgb[rgb_indices],
        ],
        dim=-1,
    )
    if rgb_tokens.numel() > 0:
        rgb_tokens[:, 1] = 1.0

    flow_tokens = torch.cat(
        [
            torch.zeros((flow_indices.numel(), 2), device=valid.device, dtype=image1.dtype),
            flat_xy[flow_indices],
            flat_flow[flow_indices],
            torch.zeros((flow_indices.numel(), 6), device=valid.device, dtype=image1.dtype),
        ],
        dim=-1,
    )
    if flow_tokens.numel() > 0:
        flow_tokens[:, 0] = 1.0

    tokens = torch.cat([flow_tokens, rgb_tokens], dim=0)
    if tokens.numel() == 0:
        tokens = torch.zeros((1, 12), device=valid.device, dtype=image1.dtype)

    observed_mask = torch.zeros((h * w,), device=valid.device, dtype=torch.float32)
    if flow_indices.numel() > 0:
        observed_mask[flow_indices] = 1.0
    observed_mask = observed_mask.view(h, w)

    return {
        "tokens": tokens,
        "observed_mask": observed_mask,
        "obs_fraction": float(fraction),
        "mask_type": mask_mode,
    }


def build_variable_flow_sample(
    *,
    sample_id: str,
    image1: torch.Tensor,
    image2: torch.Tensor,
    flow: torch.Tensor,
    valid: torch.Tensor,
    config: VariableObservationFlowDataConfig,
    rng_seed: int,
) -> dict[str, Any]:
    image_size = (int(config.image_size[0]), int(config.image_size[1]))
    image1 = _resize_rgb(image1, image_size)
    image2 = _resize_rgb(image2, image_size)
    flow, valid = _resize_flow_and_valid(flow, valid.bool(), image_size)
    flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    flow_norm = _normalize_flow(flow, image_size)

    if config.normalize_rgb:
        image1 = normalize(image1, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        image2 = normalize(image2, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    query_hw, query_xy = _make_query_grid(image_size, int(config.query_stride), image1.device)
    q_flat = (query_hw[:, 0] * image_size[1] + query_hw[:, 1]).to(torch.long)
    flow_flat = flow_norm.permute(1, 2, 0).reshape(-1, 2)
    valid_flat = valid.reshape(-1).float()
    patch_bank = _extract_patch_bank(image1, image2, int(config.rgb_patch_size))
    target_flow_q = flow_flat[q_flat]
    target_valid_q = valid_flat[q_flat]
    query_rgb = patch_bank[q_flat]
    query_inputs = torch.cat([query_xy, query_rgb], dim=-1)

    generator = torch.Generator(device=image1.device)
    generator.manual_seed(int(rng_seed))
    mask_mode = config.fixed_mask_mode or config.mask_mode

    if config.fixed_observed_fraction is not None:
        phase2_fraction = float(config.fixed_observed_fraction)
        view_a_fraction = float(config.fixed_observed_fraction)
        view_b_fraction = float(config.fixed_observed_fraction)
    else:
        phase2_fraction = float(
            torch.empty(1).uniform_(
                float(config.phase2_observed_fraction_min),
                float(config.phase2_observed_fraction_max),
                generator=generator,
            ).item()
        )
        view_a_fraction = float(
            torch.empty(1).uniform_(
                float(config.phase34_view_a_fraction_min),
                float(config.phase34_view_a_fraction_max),
                generator=generator,
            ).item()
        )
        view_b_fraction = float(
            torch.empty(1).uniform_(
                float(config.phase34_view_b_fraction_min),
                float(config.phase34_view_b_fraction_max),
                generator=generator,
            ).item()
        )

    if config.phase == 1:
        phase2_fraction = 0.0

    if config.phase == 0:
        view_a = _build_view_tokens(
            image1, image2, flow_norm, valid, query_hw, query_xy, patch_bank, 0, 1.0, "dense", generator
        )
        view_b = None
    elif config.phase == 1:
        view_a = _build_view_tokens(
            image1, image2, flow_norm, valid, query_hw, query_xy, patch_bank, 1, 0.0, "rgb_only", generator
        )
        view_b = None
    elif config.phase == 2:
        view_a = _build_view_tokens(
            image1, image2, flow_norm, valid, query_hw, query_xy, patch_bank, 2, phase2_fraction, mask_mode, generator
        )
        view_b = None
    else:
        view_a = _build_view_tokens(
            image1, image2, flow_norm, valid, query_hw, query_xy, patch_bank, config.phase, view_a_fraction, mask_mode, generator
        )
        alt_mode = "random" if mask_mode == "mixed" else mask_mode
        view_b = _build_view_tokens(
            image1, image2, flow_norm, valid, query_hw, query_xy, patch_bank, config.phase, view_b_fraction, alt_mode, generator
        )

    return {
        "sample_id": sample_id,
        "phase": int(config.phase),
        "image1": image1,
        "image2": image2,
        "flow": flow_norm,
        "valid": valid.float(),
        "query_xy": query_xy,
        "query_inputs": query_inputs,
        "target_flow_q": target_flow_q,
        "target_valid_q": target_valid_q,
        "view_a": view_a,
        "view_b": view_b,
        "flow_scale_hw": torch.tensor([float(image_size[1]), float(image_size[0])], dtype=torch.float32),
    }


class VariableObservationFlowDataset(Dataset):
    def __init__(self, root: str, split: str, config: VariableObservationFlowDataConfig) -> None:
        super().__init__()
        self.dataset = FlyingThings3D(root=root, split=split, pass_name=config.pass_name, camera=config.camera)
        self.config = config

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image1 = torch.from_numpy(np.array(item[0])).permute(2, 0, 1).float() / 255.0
        image2 = torch.from_numpy(np.array(item[1])).permute(2, 0, 1).float() / 255.0
        flow = torch.from_numpy(np.array(item[2])).float()
        valid = torch.isfinite(flow).all(dim=0)
        flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
        return build_variable_flow_sample(
            sample_id=f"{self.dataset.split}:{idx}",
            image1=image1,
            image2=image2,
            flow=flow,
            valid=valid,
            config=self.config,
            rng_seed=int(self.config.random_seed) + int(idx),
        )


def _pad_view_tokens(view_items: list[Optional[dict[str, Any]]]) -> dict[str, Any]:
    template = next((item for item in view_items if item is not None), None)
    if template is None:
        raise RuntimeError("Expected at least one non-empty view item.")
    token_dim = int(template["tokens"].shape[-1])
    batch_size = len(view_items)
    lengths = [0 if item is None else int(item["tokens"].shape[0]) for item in view_items]
    max_len = max(1, max(lengths))
    tokens = torch.zeros((batch_size, max_len, token_dim), dtype=template["tokens"].dtype)
    pad_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    obs_mask = torch.stack(
        [
            template["observed_mask"].new_zeros(template["observed_mask"].shape)
            if item is None
            else item["observed_mask"]
            for item in view_items
        ],
        dim=0,
    )
    obs_fraction = torch.tensor(
        [0.0 if item is None else float(item["obs_fraction"]) for item in view_items], dtype=torch.float32
    )
    mask_type = ["none" if item is None else str(item["mask_type"]) for item in view_items]
    for idx, item in enumerate(view_items):
        if item is None:
            continue
        n = int(item["tokens"].shape[0])
        tokens[idx, :n] = item["tokens"]
        pad_mask[idx, :n] = False
    return {
        "tokens": tokens,
        "pad_mask": pad_mask,
        "observed_mask": obs_mask,
        "obs_fraction": obs_fraction,
        "mask_type": mask_type,
    }


def variable_observation_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "phase": int(batch[0]["phase"]),
        "image1": torch.stack([item["image1"] for item in batch], dim=0),
        "image2": torch.stack([item["image2"] for item in batch], dim=0),
        "flow": torch.stack([item["flow"] for item in batch], dim=0),
        "valid": torch.stack([item["valid"] for item in batch], dim=0),
        "query_xy": torch.stack([item["query_xy"] for item in batch], dim=0),
        "query_inputs": torch.stack([item["query_inputs"] for item in batch], dim=0),
        "target_flow_q": torch.stack([item["target_flow_q"] for item in batch], dim=0),
        "target_valid_q": torch.stack([item["target_valid_q"] for item in batch], dim=0),
        "view_a": _pad_view_tokens([item["view_a"] for item in batch]),
        "view_b": _pad_view_tokens([item["view_b"] for item in batch]) if batch[0]["view_b"] is not None else None,
        "flow_scale_hw": torch.stack([item["flow_scale_hw"] for item in batch], dim=0),
    }


class VariableObservationFlowDataModule(pl.LightningDataModule):
    def __init__(self, config: VariableObservationFlowDataConfig) -> None:
        super().__init__()
        self.config = config
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit", "validate"):
            return
        train_dataset: Dataset = VariableObservationFlowDataset(self.config.root, self.config.train_split, self.config)
        val_dataset: Dataset = VariableObservationFlowDataset(self.config.root, self.config.val_split, self.config)
        if self.config.train_subset_size is not None:
            train_dataset = Subset(train_dataset, list(range(min(len(train_dataset), int(self.config.train_subset_size)))))
        if self.config.val_subset_size is not None:
            val_dataset = Subset(val_dataset, list(range(min(len(val_dataset), int(self.config.val_subset_size)))))
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def _loader(self, dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
        num_workers = int(self.config.num_workers)
        kwargs: dict[str, Any] = {}
        if num_workers > 0 and self.config.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.config.prefetch_factor)
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=bool(self.config.pin_memory),
            persistent_workers=bool(num_workers > 0 and self.config.persistent_workers),
            drop_last=drop_last,
            collate_fn=variable_observation_collate,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train dataset is not initialized")
        return self._loader(self.train_dataset, int(self.config.batch_size), True, bool(self.config.drop_last))

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val dataset is not initialized")
        return self._loader(self.val_dataset, int(self.config.val_batch_size), False, False)
