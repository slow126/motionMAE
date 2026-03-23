from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FlyingThings3D
from torchvision.transforms.functional import normalize


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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

        if self.normalize_rgb:
            src_rgb = normalize(src_rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            tgt_rgb = normalize(tgt_rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        return {
            "src_rgb": src_rgb,
            "tgt_rgb": tgt_rgb,
            "flow": flow,
            "valid": valid.float(),
        }


class FlyingThingsFlowMAEDataModule(pl.LightningDataModule):
    def __init__(self, config: FlyingThingsFlowMAEConfig) -> None:
        super().__init__()
        self.config = config
        self.train_dataset: Optional[FlyingThingsFlowMAEDataset] = None
        self.val_dataset: Optional[FlyingThingsFlowMAEDataset] = None

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
            )
            self.val_dataset = FlyingThingsFlowMAEDataset(
                root=self.config.root,
                split=self.config.val_split,
                image_size=self.config.image_size,
                pass_name=self.config.pass_name,
                camera=self.config.camera,
                reverse_flow=self.config.reverse_flow,
                normalize_rgb=self.config.normalize_rgb,
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
