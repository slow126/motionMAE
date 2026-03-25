#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FlyingThings3D
from tqdm import tqdm


DEFAULT_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_IMAGE_SIZE = (256, 256)


@dataclass(frozen=True)
class UniqueImageRecord:
    image_rel_path: str
    feature_rel_path: str
    image_abs_path: str
    num_references: int


class FlyingThingsUniqueImageDataset(Dataset):
    def __init__(self, records: list[UniqueImageRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image_path = Path(record.image_abs_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return {
            "image": tensor,
            "image_rel_path": record.image_rel_path,
            "feature_rel_path": record.feature_rel_path,
            "original_size_hw": (height, width),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute per-image DINOv3 patch embeddings for FlyingThings3D."
    )
    parser.add_argument(
        "--flyingthings-root",
        type=str,
        required=True,
        help="Root passed to torchvision.datasets.FlyingThings3D (the parent directory that contains FlyingThings3D/).",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--pass-name", type=str, default="clean", choices=["clean", "final", "both"])
    parser.add_argument("--camera", type=str, default="left", choices=["left", "right", "both"])
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=DEFAULT_IMAGE_SIZE,
        help="Resize used before DINO inference. Use the same setting later in the MAE dataloader.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Run output directory. Defaults to precomputed/flyingthings_dinov3_vitb16_<HxW>.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model repo to download during setup and load offline during precompute.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="pretrained_models/dinov3-vitb16-pretrain-lvd1689m",
        help="Local directory where setup mode stores the HF snapshot and inference mode loads it from.",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Download the model snapshot into --model-dir and exit. Run this on the internet-connected login node.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the Hugging Face snapshot even if files already exist in --model-dir.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing feature files.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional debug limit after deduplication.")
    parser.add_argument(
        "--save-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Feature tensor dtype on disk.",
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model inference dtype when running on GPU. CPU inference always falls back to float32.",
    )
    return parser.parse_args()


def dtype_from_string(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def default_output_dir(model_name: str, image_size: tuple[int, int]) -> Path:
    model_slug = model_name.split("/")[-1].replace("-", "_")
    return Path("precomputed") / f"flyingthings_{model_slug}_{image_size[0]}x{image_size[1]}"


def collate_unique_images(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "images": torch.stack([sample["image"] for sample in batch], dim=0),
        "image_rel_paths": [sample["image_rel_path"] for sample in batch],
        "feature_rel_paths": [sample["feature_rel_path"] for sample in batch],
        "original_sizes_hw": [sample["original_size_hw"] for sample in batch],
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


def dataset_root_from_argument(root: Path) -> Path:
    return root if root.name == "FlyingThings3D" else root / "FlyingThings3D"


def feature_rel_path_for(image_rel_path: str) -> str:
    return str(Path("features") / Path(image_rel_path).with_suffix(".pt"))


def collect_flyingthings_records(
    dataset: FlyingThings3D,
    dataset_root: Path,
) -> tuple[list[UniqueImageRecord], list[dict[str, object]]]:
    unique_records: "OrderedDict[str, dict[str, object]]" = OrderedDict()
    pair_manifest: list[dict[str, object]] = []

    flow_list: Iterable[str | None]
    if getattr(dataset, "_flow_list", None):
        flow_list = dataset._flow_list
    else:
        flow_list = [None] * len(dataset)

    for pair_index, (image_pair, flow_path) in enumerate(zip(dataset._image_list, flow_list)):
        src_abs = Path(image_pair[0])
        tgt_abs = Path(image_pair[1])
        src_rel = src_abs.relative_to(dataset_root).as_posix()
        tgt_rel = tgt_abs.relative_to(dataset_root).as_posix()
        src_feature_rel = feature_rel_path_for(src_rel)
        tgt_feature_rel = feature_rel_path_for(tgt_rel)

        for abs_path, rel_path, feature_rel_path in (
            (src_abs, src_rel, src_feature_rel),
            (tgt_abs, tgt_rel, tgt_feature_rel),
        ):
            record = unique_records.get(rel_path)
            if record is None:
                unique_records[rel_path] = {
                    "image_rel_path": rel_path,
                    "feature_rel_path": feature_rel_path,
                    "image_abs_path": str(abs_path),
                    "num_references": 1,
                }
            else:
                record["num_references"] = int(record["num_references"]) + 1

        pair_manifest.append(
            {
                "pair_index": pair_index,
                "src_image_rel_path": src_rel,
                "tgt_image_rel_path": tgt_rel,
                "src_feature_rel_path": src_feature_rel,
                "tgt_feature_rel_path": tgt_feature_rel,
                "flow_rel_path": None if flow_path is None else Path(flow_path).relative_to(dataset_root).as_posix(),
            }
        )

    unique_images = [UniqueImageRecord(**record) for record in unique_records.values()]
    return unique_images, pair_manifest


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
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        FlyingThingsUniqueImageDataset(records),
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


def build_image_manifest_rows(
    output_dir: Path,
    records: list[UniqueImageRecord],
) -> list[dict[str, object]]:
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


def precompute_features(
    dataset_root: Path,
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

    dataset = FlyingThings3D(
        root=str(dataset_root.parent),
        split=args.split,
        pass_name=args.pass_name,
        camera=args.camera,
    )
    unique_images, pair_manifest = collect_flyingthings_records(dataset, dataset_root)
    if args.max_images is not None:
        unique_images = unique_images[: max(0, int(args.max_images))]

    pending_records: list[UniqueImageRecord] = []
    for record in unique_images:
        if args.overwrite or not (output_dir / record.feature_rel_path).exists():
            pending_records.append(record)

    run_metadata = {
        "flyingthings_root": str(dataset_root.parent),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "pass_name": args.pass_name,
        "camera": args.camera,
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
        raise ValueError(
            f"image-size {image_size_hw} must be divisible by the model patch size {patch_size}."
        )
    image_mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
    image_std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
    feature_dim = int(getattr(model.config, "hidden_size", 0))

    loader = make_loader(
        records=pending_records,
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

        for feature_tensor, image_rel_path, feature_rel_path in zip(
            patch_tokens,
            image_rel_paths,
            feature_rel_paths,
        ):
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
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.model_name, image_size_hw)
    model_dir = Path(args.model_dir)
    flyingthings_root = Path(args.flyingthings_root)
    dataset_root = dataset_root_from_argument(flyingthings_root)

    if args.setup_only:
        print(f"Downloading {args.model_name} into {model_dir}...")
        ensure_model_snapshot(args.model_name, model_dir, force_download=args.force_download)
        print("Setup complete.")
        print(f"Offline model directory: {model_dir}")
        return

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Expected dataset root at {dataset_root}. Pass the parent directory that contains FlyingThings3D/."
        )
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory {model_dir} does not exist. Run this script once with --setup-only on an internet-connected node."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    precompute_features(dataset_root=dataset_root, output_dir=output_dir, model_dir=model_dir, args=args)


if __name__ == "__main__":
    main()
