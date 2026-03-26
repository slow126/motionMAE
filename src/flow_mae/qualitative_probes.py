from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize

from src.data.real.datasets.semantic_pairs.pfpascal import PFPascalDataset
from src.data.synth.datasets.KittiDataset import KittiSimpleDataset
from src.data.synth.datasets.TSSDataset import TSSSimpleDataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class QualitativeProbeConfig:
    name: str
    dataset_type: str
    num_samples: int = 8
    batch_size: int = 4
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    image_size: Sequence[int] = (256, 256)
    reverse_flow: bool = False
    normalize_rgb: bool = True
    normalize_flow: bool = True
    flow_scale: Optional[float] = None
    max_flow_magnitude: Optional[float] = None
    max_flow_magnitude_multiplier: float = 2.0
    dino_features_root: Optional[str] = None
    dino_feature_subdir: str = "features"
    dino_model_dir: Optional[str] = None
    tss_root: Optional[str] = None
    kitti_root: Optional[str] = None
    kitti_split: str = "training"
    kitti_version: str = "2015"
    kitti_occ_type: str = "occ"
    pfpascal_datapath: Optional[str] = None
    pfpascal_split: str = "val"
    pfpascal_thres: str = "img"


class BaseQualitativeProbeDataset(Dataset):
    def __init__(self, config: QualitativeProbeConfig) -> None:
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
        self.dino_features_root = None if config.dino_features_root is None else Path(config.dino_features_root)
        self.dino_feature_subdir = str(config.dino_feature_subdir)
        self._indices: list[int] = []

    def __len__(self) -> int:
        return len(self._indices)

    def _set_indices(self, dataset_length: int) -> None:
        count = min(max(1, int(self.config.num_samples)), int(dataset_length))
        self._indices = list(range(count))

    def _resize_rgb(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] == self.image_size:
            return image
        return F.interpolate(
            image.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

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

    def _feature_path_from_image_path(self, feature_root_base: Path, image_path: str | Path) -> Path:
        if self.dino_features_root is None:
            raise RuntimeError("DINO feature root is not configured for this qualitative probe.")
        image_rel = Path(image_path).relative_to(feature_root_base)
        return self.dino_features_root / self.dino_feature_subdir / image_rel.with_suffix(".pt")

    def _load_feature_tensor(self, feature_path: Path) -> torch.Tensor:
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Missing DINO feature file {feature_path}. "
                f"Precompute qualitative probe features for {self.config.name} first."
            )
        feature = torch.load(feature_path, map_location="cpu")
        if isinstance(feature, dict):
            if "patch_features" in feature:
                feature = feature["patch_features"]
            elif "features" in feature:
                feature = feature["features"]
            else:
                raise ValueError(f"Unsupported DINO feature format in {feature_path}")
        if not isinstance(feature, torch.Tensor):
            raise TypeError(f"Expected tensor DINO features in {feature_path}, got {type(feature)!r}")
        return feature.float()

    def _finalize_sample(
        self,
        src_rgb: torch.Tensor,
        tgt_rgb: torch.Tensor,
        flow: torch.Tensor,
        valid: torch.Tensor,
        feature_root_base: Path,
        src_image_path: str | Path,
        tgt_image_path: str | Path,
    ) -> dict[str, torch.Tensor]:
        src_rgb = self._resize_rgb(src_rgb.float())
        tgt_rgb = self._resize_rgb(tgt_rgb.float())
        flow, valid = self._resize_flow_and_valid(flow.float(), valid.to(torch.bool))
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

        sample = {
            "src_rgb": src_rgb,
            "tgt_rgb": tgt_rgb,
            "flow": flow,
            "valid": valid.float(),
            "flow_scale": torch.tensor(float(self.flow_scale), dtype=torch.float32),
            "observed_valid_override": valid.float(),
        }

        if self.dino_features_root is not None:
            sample["src_dino"] = self._load_feature_tensor(
                self._feature_path_from_image_path(feature_root_base, src_image_path)
            )
            sample["tgt_dino"] = self._load_feature_tensor(
                self._feature_path_from_image_path(feature_root_base, tgt_image_path)
            )
        return sample


class TSSQualitativeProbeDataset(BaseQualitativeProbeDataset):
    def __init__(self, config: QualitativeProbeConfig) -> None:
        super().__init__(config)
        if not config.tss_root:
            raise ValueError("TSS qualitative probe requires tss_root.")
        self.root = Path(config.tss_root)
        self.base_dataset = TSSSimpleDataset(root=self.root, reverse_flow=bool(config.reverse_flow))
        self._set_indices(len(self.base_dataset))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base_idx = self._indices[idx]
        pair_dir = self.base_dataset.pairs[base_idx]
        item = self.base_dataset[base_idx]
        valid = torch.isfinite(item["flow"]).all(dim=0)
        return self._finalize_sample(
            src_rgb=item["src_img"],
            tgt_rgb=item["trg_img"],
            flow=item["flow"],
            valid=valid,
            feature_root_base=self.root,
            src_image_path=pair_dir / "image1.png",
            tgt_image_path=pair_dir / "image2.png",
        )


class KittiQualitativeProbeDataset(BaseQualitativeProbeDataset):
    def __init__(self, config: QualitativeProbeConfig) -> None:
        super().__init__(config)
        if not config.kitti_root:
            raise ValueError("KITTI qualitative probe requires kitti_root.")
        self.root = Path(config.kitti_root)
        self.base_dataset = KittiSimpleDataset(
            root=str(self.root),
            split=str(config.kitti_split),
            version=str(config.kitti_version),
            occ_type=str(config.kitti_occ_type),
            reverse_flow=bool(config.reverse_flow),
        )
        self._set_indices(len(self.base_dataset))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base_idx = self._indices[idx]
        item = self.base_dataset[base_idx]
        img_paths, _ = self.base_dataset.file_list[base_idx]
        src_index, tgt_index = (1, 0) if self.base_dataset.reverse_flow else (0, 1)
        src_image_path = self.root / self.base_dataset.data_dir_name / img_paths[src_index]
        tgt_image_path = self.root / self.base_dataset.data_dir_name / img_paths[tgt_index]
        valid = torch.isfinite(item["flow"]).all(dim=0)
        return self._finalize_sample(
            src_rgb=item["src_img"],
            tgt_rgb=item["trg_img"],
            flow=item["flow"],
            valid=valid,
            feature_root_base=self.root,
            src_image_path=src_image_path,
            tgt_image_path=tgt_image_path,
        )


class PFPascalQualitativeProbeDataset(BaseQualitativeProbeDataset):
    def __init__(self, config: QualitativeProbeConfig) -> None:
        super().__init__(config)
        if not config.pfpascal_datapath:
            raise ValueError("PFPascal qualitative probe requires pfpascal_datapath.")
        self.root = Path(config.pfpascal_datapath)
        self.base_dataset = PFPascalDataset(
            benchmark="pfpascal",
            datapath=str(self.root),
            thres=str(config.pfpascal_thres),
            split=str(config.pfpascal_split),
            augmentation=False,
            feature_size=64,
            receptive_field_size=11,
            bidirectional_flows=False,
            normalize=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        self._set_indices(len(self.base_dataset))

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255.0

    @staticmethod
    def _build_sparse_target_to_source_flow(
        src_kps: torch.Tensor,
        tgt_kps: torch.Tensor,
        n_pts: int,
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = image_hw
        flow = torch.zeros((2, height, width), dtype=torch.float32)
        valid = torch.zeros((height, width), dtype=torch.bool)
        counts = torch.zeros((height, width), dtype=torch.float32)

        count = max(0, int(n_pts))
        for kp_idx in range(count):
            tx = int(torch.round(tgt_kps[0, kp_idx]).item())
            ty = int(torch.round(tgt_kps[1, kp_idx]).item())
            if tx < 0 or tx >= width or ty < 0 or ty >= height:
                continue
            displacement = src_kps[:, kp_idx] - tgt_kps[:, kp_idx]
            flow[:, ty, tx] += displacement
            counts[ty, tx] += 1.0
            valid[ty, tx] = True

        counts = counts.clamp_min(1.0)
        flow = flow / counts.unsqueeze(0)
        return flow, valid

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base_idx = self._indices[idx]
        src_image_path = Path(self.base_dataset.img_path) / self.base_dataset.src_imnames[base_idx]
        tgt_image_path = Path(self.base_dataset.img_path) / self.base_dataset.trg_imnames[base_idx]

        with Image.open(src_image_path) as src_image_pil:
            src_image_pil = src_image_pil.convert("RGB")
            src_imsize = src_image_pil.size
            src_rgb = self._pil_to_tensor(src_image_pil.resize((256, 256), Image.Resampling.BILINEAR))
        with Image.open(tgt_image_path) as tgt_image_pil:
            tgt_image_pil = tgt_image_pil.convert("RGB")
            tgt_imsize = tgt_image_pil.size
            tgt_rgb = self._pil_to_tensor(tgt_image_pil.resize((256, 256), Image.Resampling.BILINEAR))

        src_kps, n_pts = self.base_dataset.get_points(self.base_dataset.src_kps, base_idx, src_imsize)
        tgt_kps, _ = self.base_dataset.get_points(self.base_dataset.trg_kps, base_idx, tgt_imsize)
        flow, valid = self._build_sparse_target_to_source_flow(src_kps, tgt_kps, n_pts, (256, 256))

        return self._finalize_sample(
            src_rgb=src_rgb,
            tgt_rgb=tgt_rgb,
            flow=flow,
            valid=valid,
            feature_root_base=self.root,
            src_image_path=src_image_path,
            tgt_image_path=tgt_image_path,
        )


def build_qualitative_probe_dataset(config: QualitativeProbeConfig) -> BaseQualitativeProbeDataset:
    dataset_type = str(config.dataset_type).lower()
    if dataset_type == "tss":
        return TSSQualitativeProbeDataset(config)
    if dataset_type == "kitti":
        return KittiQualitativeProbeDataset(config)
    if dataset_type == "pfpascal":
        return PFPascalQualitativeProbeDataset(config)
    raise ValueError(f"Unsupported qualitative probe dataset_type={config.dataset_type!r}")
