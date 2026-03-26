#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.real.datasets.semantic_pairs.pfpascal import PFPascalDataset
from src.data.synth.datasets.KittiDataset import KittiSimpleDataset


DEFAULT_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_IMAGE_SIZE = (256, 256)


@dataclass(frozen=True)
class UniqueImageRecord:
    image_rel_path: str
    feature_rel_path: str
    image_abs_path: str
    num_references: int


class UniqueImageDataset(Dataset):
    def __init__(self, records: list[UniqueImageRecord], image_size_hw: tuple[int, int]) -> None:
        self.records = records
        self.image_size_hw = (int(image_size_hw[0]), int(image_size_hw[1]))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image_path = Path(record.image_abs_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.size != (self.image_size_hw[1], self.image_size_hw[0]):
                image = image.resize((self.image_size_hw[1], self.image_size_hw[0]), Image.Resampling.BILINEAR)
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return {
            "image": tensor,
            "image_rel_path": record.image_rel_path,
            "feature_rel_path": record.feature_rel_path,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute per-image DINOv3 patch embeddings for qualitative probe datasets.")
    parser.add_argument("--dataset", type=str, required=True, choices=["tss", "kitti", "pfpascal"])
    parser.add_argument("--tss-root", type=str, default=None)
    parser.add_argument("--kitti-root", type=str, default=None)
    parser.add_argument("--kitti-split", type=str, default="training", choices=["training", "train", "val"])
    parser.add_argument("--kitti-version", type=str, default="2015", choices=["2012", "2015", "auto"])
    parser.add_argument("--kitti-occ-type", type=str, default="occ", choices=["occ", "noc", "only_occ"])
    parser.add_argument("--pfpascal-datapath", type=str, default=None)
    parser.add_argument("--pfpascal-split", type=str, default="val", choices=["trn", "val", "test"])
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=DEFAULT_IMAGE_SIZE,
        help="Resize used before DINO inference.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-dir", type=str, default="pretrained_models/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--model-dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    return parser.parse_args()


def dtype_from_string(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def default_output_dir(dataset_name: str, model_name: str, image_size: tuple[int, int]) -> Path:
    model_slug = model_name.split("/")[-1].replace("-", "_")
    return Path("precomputed") / "probes" / f"{dataset_name}_{model_slug}_{image_size[0]}x{image_size[1]}"


def collate_unique_images(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "images": torch.stack([sample["image"] for sample in batch], dim=0),
        "image_rel_paths": [sample["image_rel_path"] for sample in batch],
        "feature_rel_paths": [sample["feature_rel_path"] for sample in batch],
    }


def ensure_model_snapshot(model_name: str, model_dir: Path, force_download: bool) -> Path:
    from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=str(model_dir),
        force_download=force_download,
        local_dir_use_symlinks=False,
        resume_download=not force_download,
    )
    return model_dir


def feature_rel_path_for(image_rel_path: str) -> str:
    return str(Path("features") / Path(image_rel_path).with_suffix(".pt"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def make_loader(
    records: list[UniqueImageRecord],
    image_size_hw: tuple[int, int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        UniqueImageDataset(records, image_size_hw=image_size_hw),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_unique_images,
    )


def infer_patch_grid(num_tokens: int, grid_hint_hw: tuple[int, int], patch_size: int) -> tuple[int, int]:
    hinted_h = grid_hint_hw[0] // patch_size
    hinted_w = grid_hint_hw[1] // patch_size
    if hinted_h * hinted_w == num_tokens:
        return hinted_h, hinted_w
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Cannot infer a 2D patch grid from {num_tokens} tokens.")
    return side, side


def atomic_torch_save(tensor: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(tensor, tmp_path)
    tmp_path.replace(output_path)


def build_image_manifest_rows(output_dir: Path, records: list[UniqueImageRecord]) -> list[dict[str, object]]:
    rows = []
    for record in records:
        feature_path = output_dir / record.feature_rel_path
        rows.append(
            {
                "image_rel_path": record.image_rel_path,
                "feature_rel_path": record.feature_rel_path,
                "num_references": record.num_references,
                "feature_exists": feature_path.exists(),
            }
        )
    return rows


def add_unique_record(
    unique_records: "OrderedDict[str, dict[str, object]]",
    dataset_root: Path,
    image_path: Path,
) -> tuple[str, str]:
    image_rel = image_path.relative_to(dataset_root).as_posix()
    feature_rel = feature_rel_path_for(image_rel)
    record = unique_records.get(image_rel)
    if record is None:
        unique_records[image_rel] = {
            "image_rel_path": image_rel,
            "feature_rel_path": feature_rel,
            "image_abs_path": str(image_path),
            "num_references": 1,
        }
    else:
        record["num_references"] = int(record["num_references"]) + 1
    return image_rel, feature_rel


def collect_tss_records(dataset_root: Path) -> tuple[list[UniqueImageRecord], list[dict[str, object]]]:
    unique_records: "OrderedDict[str, dict[str, object]]" = OrderedDict()
    pair_manifest: list[dict[str, object]] = []
    pair_index = 0
    for subdir in sorted(dataset_root.iterdir()):
        if not subdir.is_dir():
            continue
        for pair_dir in sorted(subdir.iterdir()):
            if not pair_dir.is_dir():
                continue
            src_path = pair_dir / "image1.png"
            tgt_path = pair_dir / "image2.png"
            if not src_path.exists() or not tgt_path.exists():
                continue
            src_rel, src_feature_rel = add_unique_record(unique_records, dataset_root, src_path)
            tgt_rel, tgt_feature_rel = add_unique_record(unique_records, dataset_root, tgt_path)
            pair_manifest.append(
                {
                    "pair_index": pair_index,
                    "src_image_rel_path": src_rel,
                    "tgt_image_rel_path": tgt_rel,
                    "src_feature_rel_path": src_feature_rel,
                    "tgt_feature_rel_path": tgt_feature_rel,
                }
            )
            pair_index += 1
    return [UniqueImageRecord(**record) for record in unique_records.values()], pair_manifest


def collect_kitti_records(
    dataset_root: Path,
    split: str,
    version: str,
    occ_type: str,
) -> tuple[list[UniqueImageRecord], list[dict[str, object]]]:
    dataset = KittiSimpleDataset(
        root=str(dataset_root),
        split=split,
        version=version,
        occ_type=occ_type,
        reverse_flow=False,
    )
    unique_records: "OrderedDict[str, dict[str, object]]" = OrderedDict()
    pair_manifest: list[dict[str, object]] = []
    for pair_index, (img_paths, _) in enumerate(dataset.file_list):
        src_path = dataset_root / dataset.data_dir_name / img_paths[0]
        tgt_path = dataset_root / dataset.data_dir_name / img_paths[1]
        src_rel, src_feature_rel = add_unique_record(unique_records, dataset_root, src_path)
        tgt_rel, tgt_feature_rel = add_unique_record(unique_records, dataset_root, tgt_path)
        pair_manifest.append(
            {
                "pair_index": pair_index,
                "src_image_rel_path": src_rel,
                "tgt_image_rel_path": tgt_rel,
                "src_feature_rel_path": src_feature_rel,
                "tgt_feature_rel_path": tgt_feature_rel,
            }
        )
    return [UniqueImageRecord(**record) for record in unique_records.values()], pair_manifest


def collect_pfpascal_records(dataset_root: Path, split: str) -> tuple[list[UniqueImageRecord], list[dict[str, object]]]:
    dataset = PFPascalDataset(
        benchmark="pfpascal",
        datapath=str(dataset_root),
        thres="img",
        split=split,
        augmentation=False,
        feature_size=64,
        receptive_field_size=11,
        bidirectional_flows=False,
        normalize=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )
    unique_records: "OrderedDict[str, dict[str, object]]" = OrderedDict()
    pair_manifest: list[dict[str, object]] = []
    image_root = Path(dataset.img_path)
    for pair_index, (src_name, tgt_name) in enumerate(zip(dataset.src_imnames, dataset.trg_imnames)):
        src_path = image_root / src_name
        tgt_path = image_root / tgt_name
        src_rel, src_feature_rel = add_unique_record(unique_records, dataset_root, src_path)
        tgt_rel, tgt_feature_rel = add_unique_record(unique_records, dataset_root, tgt_path)
        pair_manifest.append(
            {
                "pair_index": pair_index,
                "src_image_rel_path": src_rel,
                "tgt_image_rel_path": tgt_rel,
                "src_feature_rel_path": src_feature_rel,
                "tgt_feature_rel_path": tgt_feature_rel,
            }
        )
    return [UniqueImageRecord(**record) for record in unique_records.values()], pair_manifest


def collect_records(args: argparse.Namespace) -> tuple[Path, list[UniqueImageRecord], list[dict[str, object]]]:
    if args.dataset == "tss":
        if not args.tss_root:
            raise ValueError("--tss-root is required for --dataset tss")
        dataset_root = Path(args.tss_root)
        unique_images, pair_manifest = collect_tss_records(dataset_root)
        return dataset_root, unique_images, pair_manifest
    if args.dataset == "kitti":
        if not args.kitti_root:
            raise ValueError("--kitti-root is required for --dataset kitti")
        dataset_root = Path(args.kitti_root)
        unique_images, pair_manifest = collect_kitti_records(
            dataset_root=dataset_root,
            split=args.kitti_split,
            version=args.kitti_version,
            occ_type=args.kitti_occ_type,
        )
        return dataset_root, unique_images, pair_manifest
    if args.dataset == "pfpascal":
        if not args.pfpascal_datapath:
            raise ValueError("--pfpascal-datapath is required for --dataset pfpascal")
        dataset_root = Path(args.pfpascal_datapath)
        unique_images, pair_manifest = collect_pfpascal_records(dataset_root, split=args.pfpascal_split)
        return dataset_root, unique_images, pair_manifest
    raise ValueError(f"Unsupported dataset {args.dataset!r}")


def precompute_features(
    dataset_root: Path,
    unique_images: list[UniqueImageRecord],
    pair_manifest: list[dict[str, object]],
    output_dir: Path,
    model_dir: Path,
    args: argparse.Namespace,
) -> None:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model_dtype = dtype_from_string(args.model_dtype)
    save_dtype = dtype_from_string(args.save_dtype)
    image_size_hw = (int(args.image_size[0]), int(args.image_size[1]))

    if device.type == "cpu":
        model_dtype = torch.float32

    if args.max_images is not None:
        unique_images = unique_images[: max(0, int(args.max_images))]

    pending_records = []
    for record in unique_images:
        if args.overwrite or not (output_dir / record.feature_rel_path).exists():
            pending_records.append(record)

    run_metadata = {
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "image_size": list(image_size_hw),
        "model_name": args.model_name,
        "model_dir": str(model_dir),
        "save_dtype": args.save_dtype,
        "model_dtype": str(model_dtype).replace("torch.", ""),
        "num_unique_images": len(unique_images),
        "num_preexisting_images": len(unique_images) - len(pending_records),
        "num_pending_images": len(pending_records),
        "num_pair_samples": len(pair_manifest),
    }
    if args.dataset == "kitti":
        run_metadata.update(
            {
                "kitti_split": args.kitti_split,
                "kitti_version": args.kitti_version,
                "kitti_occ_type": args.kitti_occ_type,
            }
        )
    elif args.dataset == "pfpascal":
        run_metadata["pfpascal_split"] = args.pfpascal_split

    write_json(output_dir / "run_metadata.json", run_metadata)
    write_jsonl(output_dir / "pair_manifest.jsonl", pair_manifest)
    write_jsonl(output_dir / "image_manifest.jsonl", build_image_manifest_rows(output_dir, unique_images))

    print(f"Collected {len(pair_manifest)} pair samples and {len(unique_images)} unique images.")
    print(f"Need to process {len(pending_records)} images into {output_dir}.")
    if not pending_records:
        print("All feature files already exist. Nothing to do.")
        return

    from transformers import AutoImageProcessor, AutoModel

    print(f"Loading processor from {model_dir}...")
    processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    print(f"Loading model from {model_dir} on {device}...")
    model = AutoModel.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    patch_size = int(getattr(model.config, "patch_size", 16))
    num_register_tokens = int(getattr(model.config, "num_register_tokens", 4))
    num_prefix_tokens = 1 + max(0, num_register_tokens)
    if image_size_hw[0] % patch_size != 0 or image_size_hw[1] % patch_size != 0:
        raise ValueError(f"image-size {image_size_hw} must be divisible by the model patch size {patch_size}.")
    image_mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
    image_std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
    feature_dim = int(getattr(model.config, "hidden_size", 0))

    loader = make_loader(
        records=pending_records,
        image_size_hw=image_size_hw,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False if args.no_pin_memory else bool(args.pin_memory or device.type == "cuda"),
    )

    use_autocast = device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16)
    progress = tqdm(loader, desc="Precomputing DINO features", unit="batch")
    for batch in progress:
        images = batch["images"]
        image_rel_paths = batch["image_rel_paths"]
        feature_rel_paths = batch["feature_rel_paths"]

        images = F.interpolate(
            images,
            size=image_size_hw,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        images = images.clamp_(0.0, 1.0)
        images = (images - image_mean) / image_std
        images = images.to(device, non_blocking=True)

        with torch.inference_mode():
            if use_autocast:
                with torch.autocast(device_type=device.type, dtype=model_dtype):
                    outputs = model(pixel_values=images)
            else:
                outputs = model(pixel_values=images)

        patch_tokens = outputs.last_hidden_state[:, num_prefix_tokens:, :]
        grid_h, grid_w = infer_patch_grid(
            num_tokens=patch_tokens.shape[1],
            grid_hint_hw=image_size_hw,
            patch_size=patch_size,
        )
        patch_tokens = patch_tokens.reshape(patch_tokens.shape[0], grid_h, grid_w, patch_tokens.shape[-1])
        patch_tokens = patch_tokens.detach().to(dtype=save_dtype).cpu().contiguous()

        for feature_tensor, image_rel_path, feature_rel_path in zip(patch_tokens, image_rel_paths, feature_rel_paths):
            output_path = output_dir / str(feature_rel_path)
            atomic_torch_save(feature_tensor, output_path)
            progress.set_postfix_str(str(image_rel_path))

    run_metadata.update(
        {
            "patch_size": patch_size,
            "num_prefix_tokens": num_prefix_tokens,
            "feature_dim": feature_dim,
            "patch_grid_hw": [grid_h, grid_w],
            "num_processed_this_run": len(pending_records),
            "num_saved_feature_files": sum((output_dir / record.feature_rel_path).exists() for record in unique_images),
        }
    )
    write_json(output_dir / "run_metadata.json", run_metadata)
    write_jsonl(output_dir / "image_manifest.jsonl", build_image_manifest_rows(output_dir, unique_images))


def main() -> None:
    args = parse_args()
    image_size_hw = (int(args.image_size[0]), int(args.image_size[1]))
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.dataset, args.model_name, image_size_hw)
    model_dir = Path(args.model_dir)

    if args.setup_only:
        print(f"Downloading {args.model_name} into {model_dir}...")
        ensure_model_snapshot(args.model_name, model_dir, force_download=args.force_download)
        print("Setup complete.")
        print(f"Offline model directory: {model_dir}")
        return

    dataset_root, unique_images, pair_manifest = collect_records(args)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Expected dataset root at {dataset_root}.")
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory {model_dir} does not exist. Run this script once with --setup-only on an internet-connected node."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    precompute_features(
        dataset_root=dataset_root,
        unique_images=unique_images,
        pair_manifest=pair_manifest,
        output_dir=output_dir,
        model_dir=model_dir,
        args=args,
    )


if __name__ == "__main__":
    main()
