from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FlyingThings3D
from torchvision.transforms.functional import normalize

from src.flow_smoke.dataset import PointOdysseyFlowSmokeDataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def materialize_probe_manifest(
    source_manifest_path: str,
    output_manifest_path: str,
    num_samples: int,
    subset_indices_path: Optional[str] = None,
) -> str:
    source_manifest = Path(source_manifest_path)
    output_manifest = Path(output_manifest_path)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    num_samples = max(1, int(num_samples))
    if subset_indices_path is not None:
        with Path(subset_indices_path).open("r", encoding="utf-8") as handle:
            subset_data = json.load(handle)
        if isinstance(subset_data, dict):
            subset_data = subset_data.get("indices", subset_data.get("subset", []))
        selected_indices = [int(v) for v in subset_data[:num_samples]]
    else:
        selected_indices = list(range(num_samples))

    selected_indices = sorted(set(idx for idx in selected_indices if idx >= 0))
    if not selected_indices:
        raise RuntimeError("No valid probe manifest indices selected.")

    selected_set = set(selected_indices)
    written = 0
    with source_manifest.open("r", encoding="utf-8") as src, output_manifest.open("w", encoding="utf-8") as dst:
        for line_idx, line in enumerate(src):
            if line_idx not in selected_set:
                continue
            dst.write(line)
            written += 1
            if written >= len(selected_indices):
                break

    if written == 0:
        raise RuntimeError(
            f"Failed to materialize probe manifest from {source_manifest}; no selected rows were found."
        )
    return str(output_manifest)


@dataclass
class FlyingThingsFlowMAEConfig:
    root: str
    train_split: str = "train"
    val_split: str = "test"
    pass_name: str = "clean"
    camera: str = "left"
    image_size: Sequence[int] = (256, 256)
    reverse_flow: bool = False
    normalize_rgb: bool = True
    batch_size: int = 16
    val_batch_size: int = 16
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True
    normalize_flow: bool = True
    flow_scale: Optional[float] = None
    max_flow_magnitude: Optional[float] = None
    max_flow_magnitude_multiplier: float = 2.0


@dataclass
class PointOdysseyProbeConfig:
    manifest_path: str
    pointodyssey_root: str
    num_samples: int = 8
    batch_size: int = 4
    num_workers: int = 0
    image_size: Sequence[int] = (256, 256)
    reverse_flow: bool = False
    normalize_rgb: bool = True
    normalize_flow: bool = True
    flow_scale: Optional[float] = None
    max_flow_magnitude: Optional[float] = None
    max_flow_magnitude_multiplier: float = 2.0
    min_valid_points: int = 8
    trust_manifest: bool = True


class FlyingThingsFlowMAEDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: Sequence[int] = (256, 256),
        pass_name: str = "clean",
        camera: str = "left",
        reverse_flow: bool = False,
        normalize_rgb: bool = True,
        normalize_flow: bool = True,
        flow_scale: Optional[float] = None,
        max_flow_magnitude: Optional[float] = None,
        max_flow_magnitude_multiplier: float = 2.0,
    ) -> None:
        super().__init__()
        self.dataset = FlyingThings3D(
            root=root,
            split=split,
            pass_name=pass_name,
            camera=camera,
        )
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.reverse_flow = bool(reverse_flow)
        self.normalize_rgb = bool(normalize_rgb)
        self.normalize_flow = bool(normalize_flow)
        scale_value = flow_scale
        self.flow_scale = float(scale_value) if scale_value is not None else float(max(self.image_size))
        if max_flow_magnitude is not None:
            self.max_flow_magnitude = float(max_flow_magnitude)
        else:
            self.max_flow_magnitude = float(max(self.image_size)) * float(max_flow_magnitude_multiplier)

    def __len__(self) -> int:
        return len(self.dataset)

    def _resize_rgb(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] == self.image_size:
            return image
        image = F.interpolate(
            image.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return image

    def _resize_flow_and_valid(self, flow: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = flow.shape[-2:]
        target_h, target_w = self.image_size
        if (h, w) == (target_h, target_w):
            return flow, valid

        scale_x = target_w / float(w)
        scale_y = target_h / float(h)

        flow = F.interpolate(
            flow.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        flow[0] *= scale_x
        flow[1] *= scale_y

        valid = F.interpolate(
            valid.float().unsqueeze(0).unsqueeze(0),
            size=self.image_size,
            mode="nearest",
        ).squeeze(0).squeeze(0) > 0.5
        flow = torch.where(valid.unsqueeze(0), flow, torch.zeros_like(flow))
        return flow, valid

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.dataset[idx]
        src_index, tgt_index = (1, 0) if self.reverse_flow else (0, 1)

        src_rgb = torch.from_numpy(np.array(item[src_index])).permute(2, 0, 1).float() / 255.0
        tgt_rgb = torch.from_numpy(np.array(item[tgt_index])).permute(2, 0, 1).float() / 255.0
        flow = torch.from_numpy(np.array(item[2])).float()

        valid = torch.isfinite(flow).all(dim=0)
        flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)

        src_rgb = self._resize_rgb(src_rgb)
        tgt_rgb = self._resize_rgb(tgt_rgb)
        flow, valid = self._resize_flow_and_valid(flow, valid)

        if self.max_flow_magnitude is not None and self.max_flow_magnitude > 0:
            flow_mag = torch.linalg.vector_norm(flow, dim=0)
            valid = valid & (flow_mag <= self.max_flow_magnitude)
            flow = torch.where(valid.unsqueeze(0), flow, torch.zeros_like(flow))

        if self.normalize_flow and self.flow_scale > 0:
            flow = flow / self.flow_scale

        if self.normalize_rgb:
            src_rgb = normalize(src_rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            tgt_rgb = normalize(tgt_rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        return {
            "src_rgb": src_rgb,
            "tgt_rgb": tgt_rgb,
            "flow": flow,
            "valid": valid.float(),
            "flow_scale": torch.tensor(float(self.flow_scale), dtype=torch.float32),
            "observed_valid_override": valid.float(),
        }


class FlyingThingsFlowMAEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: FlyingThingsFlowMAEConfig,
        pointodyssey_probe_config: Optional[PointOdysseyProbeConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.pointodyssey_probe_config = pointodyssey_probe_config
        self.train_dataset: Optional[FlyingThingsFlowMAEDataset] = None
        self.val_dataset: Optional[FlyingThingsFlowMAEDataset] = None
        self.pointodyssey_probe_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = FlyingThingsFlowMAEDataset(
                root=self.config.root,
                split=self.config.train_split,
                image_size=self.config.image_size,
                pass_name=self.config.pass_name,
                camera=self.config.camera,
                reverse_flow=self.config.reverse_flow,
                normalize_rgb=self.config.normalize_rgb,
                normalize_flow=self.config.normalize_flow,
                flow_scale=self.config.flow_scale,
                max_flow_magnitude=self.config.max_flow_magnitude,
                max_flow_magnitude_multiplier=self.config.max_flow_magnitude_multiplier,
            )
            self.val_dataset = FlyingThingsFlowMAEDataset(
                root=self.config.root,
                split=self.config.val_split,
                image_size=self.config.image_size,
                pass_name=self.config.pass_name,
                camera=self.config.camera,
                reverse_flow=self.config.reverse_flow,
                normalize_rgb=self.config.normalize_rgb,
                normalize_flow=self.config.normalize_flow,
                flow_scale=self.config.flow_scale,
                max_flow_magnitude=self.config.max_flow_magnitude,
                max_flow_magnitude_multiplier=self.config.max_flow_magnitude_multiplier,
            )
            print(
                "[FlowMAEDataModule] "
                f"train_samples={len(self.train_dataset)} "
                f"val_samples={len(self.val_dataset)} "
                f"image_size={tuple(self.config.image_size)} "
                f"normalize_flow={self.config.normalize_flow} "
                f"flow_scale={self.train_dataset.flow_scale:.2f} "
                f"max_flow_magnitude={self.train_dataset.max_flow_magnitude:.2f}"
            )
            if self.pointodyssey_probe_config is not None:
                self.pointodyssey_probe_dataset = PointOdysseyProbeDataset(self.pointodyssey_probe_config)
                print(
                    "[FlowMAEDataModule] "
                    f"pointodyssey_probe_samples={len(self.pointodyssey_probe_dataset)} "
                    f"manifest={self.pointodyssey_probe_config.manifest_path}"
                )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train dataset is not initialized")
        num_workers = int(self.config.num_workers)
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.config.batch_size),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=bool(self.config.pin_memory),
            persistent_workers=bool(num_workers > 0 and self.config.persistent_workers),
            drop_last=bool(self.config.drop_last),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val dataset is not initialized")
        num_workers = int(self.config.num_workers)
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.config.val_batch_size),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(self.config.pin_memory),
            persistent_workers=bool(num_workers > 0 and self.config.persistent_workers),
            drop_last=False,
        )

    def get_pointodyssey_probe_dataloader(self) -> Optional[DataLoader]:
        if self.pointodyssey_probe_dataset is None or self.pointodyssey_probe_config is None:
            return None
        num_workers = int(self.pointodyssey_probe_config.num_workers)
        return DataLoader(
            self.pointodyssey_probe_dataset,
            batch_size=int(self.pointodyssey_probe_config.batch_size),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(self.config.pin_memory),
            persistent_workers=bool(num_workers > 0 and self.config.persistent_workers),
            drop_last=False,
        )


class PointOdysseyProbeDataset(Dataset):
    def __init__(self, config: PointOdysseyProbeConfig) -> None:
        super().__init__()
        self.config = config
        self.image_size = (int(config.image_size[0]), int(config.image_size[1]))
        self.normalize_rgb = bool(config.normalize_rgb)
        self.normalize_flow = bool(config.normalize_flow)
        scale_value = config.flow_scale
        self.flow_scale = float(scale_value) if scale_value is not None else float(max(self.image_size))
        if config.max_flow_magnitude is not None:
            self.max_flow_magnitude = float(config.max_flow_magnitude)
        else:
            self.max_flow_magnitude = float(max(self.image_size)) * float(config.max_flow_magnitude_multiplier)
        self.base_dataset = PointOdysseyFlowSmokeDataset(
            manifest_path=config.manifest_path,
            indices=list(range(max(1, int(config.num_samples)))),
            pointodyssey_root=config.pointodyssey_root,
            reverse_flow=config.reverse_flow,
            size=self.image_size,
            min_valid_points=int(config.min_valid_points),
            normalize_flow=False,
            trust_manifest=bool(config.trust_manifest),
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.base_dataset[idx]
        src_rgb = sample["src_img"]
        tgt_rgb = sample["trg_img"]
        flow = sample["flow"]
        valid = sample["valid_flow_mask"].to(torch.bool)
        flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)

        if self.max_flow_magnitude is not None and self.max_flow_magnitude > 0:
            flow_mag = torch.linalg.vector_norm(flow, dim=0)
            valid = valid & (flow_mag <= self.max_flow_magnitude)
            flow = torch.where(valid.unsqueeze(0), flow, torch.zeros_like(flow))

        if self.normalize_flow and self.flow_scale > 0:
            flow = flow / self.flow_scale

        if self.normalize_rgb:
            src_rgb = normalize(src_rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            tgt_rgb = normalize(tgt_rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        return {
            "src_rgb": src_rgb,
            "tgt_rgb": tgt_rgb,
            "flow": flow,
            "valid": valid.float(),
            "flow_scale": torch.tensor(float(self.flow_scale), dtype=torch.float32),
            "observed_valid_override": valid.float(),
        }
