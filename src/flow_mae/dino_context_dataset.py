from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .dataset import FlyingThingsFlowMAEConfig, FlyingThingsFlowMAEDataset


def _dataset_root_from_argument(root: str | Path) -> Path:
    root_path = Path(root)
    return root_path if root_path.name == "FlyingThings3D" else root_path / "FlyingThings3D"


@dataclass
class FlyingThingsDINOFlowMAEConfig(FlyingThingsFlowMAEConfig):
    dino_features_root: str = ""
    dino_feature_subdir: str = "features"


class FlyingThingsDINOFlowMAEDataset(FlyingThingsFlowMAEDataset):
    def __init__(
        self,
        root: str,
        split: str,
        dino_features_root: str,
        dino_feature_subdir: str = "features",
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
        super().__init__(
            root=root,
            split=split,
            image_size=image_size,
            pass_name=pass_name,
            camera=camera,
            reverse_flow=reverse_flow,
            normalize_rgb=normalize_rgb,
            normalize_flow=normalize_flow,
            flow_scale=flow_scale,
            max_flow_magnitude=max_flow_magnitude,
            max_flow_magnitude_multiplier=max_flow_magnitude_multiplier,
        )
        self.dataset_root = _dataset_root_from_argument(root)
        self.dino_features_root = Path(dino_features_root)
        self.dino_feature_subdir = str(dino_feature_subdir)
        self.dino_metadata = self._load_dino_metadata()
        self.dino_grid_hw, self.dino_feature_dim = self._infer_dino_layout()
        self._validate_dino_layout()

    def _load_dino_metadata(self) -> dict[str, object]:
        metadata_path = self.dino_features_root / "run_metadata.json"
        if not metadata_path.exists():
            return {}
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return metadata if isinstance(metadata, dict) else {}

    def _feature_path_from_image_path(self, image_path: str | Path) -> Path:
        image_rel = Path(image_path).relative_to(self.dataset_root)
        return self.dino_features_root / self.dino_feature_subdir / image_rel.with_suffix(".pt")

    def _load_feature_tensor(self, feature_path: Path) -> torch.Tensor:
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Missing DINO feature file {feature_path}. "
                "Make sure the precompute script has been run for this FlyingThings split."
            )
        feature = torch.load(feature_path, map_location="cpu")
        if isinstance(feature, dict):
            if "patch_features" in feature:
                feature = feature["patch_features"]
            elif "features" in feature:
                feature = feature["features"]
            else:
                raise ValueError(f"Unsupported feature dict format in {feature_path}")
        if not isinstance(feature, torch.Tensor):
            raise TypeError(f"Expected a tensor in {feature_path}, got {type(feature)!r}")
        if feature.dim() == 2:
            grid_h, grid_w = self.dino_grid_hw
            feature = feature.view(grid_h, grid_w, feature.shape[-1])
        if feature.dim() != 3:
            raise ValueError(f"Expected [grid_h, grid_w, dim] features in {feature_path}, got {tuple(feature.shape)}")
        return feature.float()

    def _infer_dino_layout(self) -> tuple[tuple[int, int], int]:
        if len(self.dataset._image_list) == 0:
            raise RuntimeError("FlyingThings dataset is empty.")
        feature_path = self._feature_path_from_image_path(self.dataset._image_list[0][0])
        feature = self._load_feature_tensor(feature_path)
        grid_h, grid_w, feature_dim = feature.shape
        return (int(grid_h), int(grid_w)), int(feature_dim)

    def _validate_dino_layout(self) -> None:
        metadata_image_size = self.dino_metadata.get("image_size")
        if metadata_image_size is not None:
            metadata_image_size = tuple(int(v) for v in metadata_image_size)
            if metadata_image_size != tuple(self.image_size):
                raise ValueError(
                    f"DINO feature image_size {metadata_image_size} does not match dataset image_size {self.image_size}."
                )

        patch_size = int(self.dino_metadata.get("patch_size", 16))
        expected_grid_hw = (
            self.image_size[0] // patch_size,
            self.image_size[1] // patch_size,
        )
        if tuple(expected_grid_hw) != tuple(self.dino_grid_hw):
            raise ValueError(
                f"DINO patch grid {self.dino_grid_hw} is incompatible with image_size {self.image_size}. "
                f"Expected {expected_grid_hw}."
            )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        src_index, tgt_index = (1, 0) if self.reverse_flow else (0, 1)
        image_pair = self.dataset._image_list[idx]
        src_dino = self._load_feature_tensor(self._feature_path_from_image_path(image_pair[src_index]))
        tgt_dino = self._load_feature_tensor(self._feature_path_from_image_path(image_pair[tgt_index]))
        sample["src_dino"] = src_dino
        sample["tgt_dino"] = tgt_dino
        return sample


class FlyingThingsDINOFlowMAEDataModule(pl.LightningDataModule):
    def __init__(self, config: FlyingThingsDINOFlowMAEConfig) -> None:
        super().__init__()
        self.config = config
        self.train_dataset: Optional[FlyingThingsDINOFlowMAEDataset] = None
        self.val_dataset: Optional[FlyingThingsDINOFlowMAEDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = FlyingThingsDINOFlowMAEDataset(
                root=self.config.root,
                split=self.config.train_split,
                dino_features_root=self.config.dino_features_root,
                dino_feature_subdir=self.config.dino_feature_subdir,
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
            self.val_dataset = FlyingThingsDINOFlowMAEDataset(
                root=self.config.root,
                split=self.config.val_split,
                dino_features_root=self.config.dino_features_root,
                dino_feature_subdir=self.config.dino_feature_subdir,
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
                "[FlowMAEDINODataModule] "
                f"train_samples={len(self.train_dataset)} "
                f"val_samples={len(self.val_dataset)} "
                f"image_size={tuple(self.config.image_size)} "
                f"dino_grid={self.train_dataset.dino_grid_hw} "
                f"dino_dim={self.train_dataset.dino_feature_dim}"
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train dataset is not initialized")
        num_workers = int(self.config.num_workers)
        kwargs = {}
        if num_workers > 0 and self.config.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.config.prefetch_factor)
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.config.batch_size),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=bool(self.config.pin_memory),
            persistent_workers=bool(num_workers > 0 and self.config.persistent_workers),
            drop_last=bool(self.config.drop_last),
            **kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val dataset is not initialized")
        num_workers = int(self.config.num_workers)
        kwargs = {}
        if num_workers > 0 and self.config.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.config.prefetch_factor)
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.config.val_batch_size),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(self.config.pin_memory),
            persistent_workers=bool(num_workers > 0 and self.config.persistent_workers),
            drop_last=False,
            **kwargs,
        )
