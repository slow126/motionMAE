#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import FlyingThings3D


def dataset_root_from_argument(root: Path) -> Path:
    return root if root.name == "FlyingThings3D" else root / "FlyingThings3D"


def feature_path_from_image(dataset_root: Path, dino_root: Path, feature_subdir: str, image_path: str) -> Path:
    rel = Path(image_path).relative_to(dataset_root)
    return dino_root / feature_subdir / rel.with_suffix(".pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check FlyingThings DINO cache coverage for a split.")
    parser.add_argument("--flyingthings-root", required=True, help="Parent dir containing FlyingThings3D/ or FlyingThings3D itself")
    parser.add_argument("--dino-root", required=True, help="Root output dir from precompute_flyingthings_dinov3.py")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--pass-name", default="clean", choices=["clean", "final", "both"])
    parser.add_argument("--camera", default="left", choices=["left", "right", "both"])
    parser.add_argument("--feature-subdir", default="features")
    parser.add_argument("--show-missing", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.flyingthings_root)
    dataset_root = dataset_root_from_argument(root)
    dino_root = Path(args.dino_root)

    dataset = FlyingThings3D(
        root=str(dataset_root.parent),
        split=args.split,
        pass_name=args.pass_name,
        camera=args.camera,
    )

    unique_images: dict[str, Path] = {}
    for image_pair in dataset._image_list:
        for image_path in image_pair:
            unique_images.setdefault(image_path, feature_path_from_image(dataset_root, dino_root, args.feature_subdir, image_path))

    missing = [(img, feat) for img, feat in unique_images.items() if not feat.exists()]
    print(f"split={args.split} pairs={len(dataset)} unique_images={len(unique_images)} missing_features={len(missing)}")
    for image_path, feature_path in missing[: max(0, int(args.show_missing))]:
        print(f"MISSING image={image_path}")
        print(f"        feature={feature_path}")


if __name__ == "__main__":
    main()
