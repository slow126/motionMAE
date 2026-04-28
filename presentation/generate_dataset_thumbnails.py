#!/usr/bin/env python3
"""
Generate simple representative dataset thumbnails for the train/eval gallery slide.
"""

from __future__ import annotations

import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "presentation" / "figures"
DATA_ROOT = Path("/home/spencer/Data")


def read_rgb(path: Path) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[-1] > 3:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def fit_image(img: np.ndarray, width: int, height: int) -> Image.Image:
    pil = Image.fromarray(img)
    return ImageOps.fit(pil, (width, height), method=Image.Resampling.BILINEAR)


def make_pair_thumbnail(img_a: np.ndarray, img_b: np.ndarray, out_path: Path, width: int = 640, height: int = 250) -> None:
    gutter = 10
    cell_w = (width - gutter) // 2
    canvas = Image.new("RGB", (width, height), "#0F172A")
    canvas.paste(fit_image(img_a, cell_w, height), (0, 0))
    canvas.paste(fit_image(img_b, width - cell_w - gutter, height), (cell_w + gutter, 0))
    canvas.save(out_path)


def make_triplet_thumbnail(imgs: list[np.ndarray], out_path: Path, width: int = 640, height: int = 250) -> None:
    gutter = 8
    cell_w = (width - 2 * gutter) // 3
    canvas = Image.new("RGB", (width, height), "#0F172A")
    x = 0
    for img in imgs:
        canvas.paste(fit_image(img, cell_w, height), (x, 0))
        x += cell_w + gutter
    canvas.save(out_path)


def write_spair() -> None:
    root = DATA_ROOT / "correspondence" / "SPair-71k"
    annos = pd.read_json(root / "PairAnnotation" / "trn" / "pair_annotations.json", typ="series")
    key = next(iter(sorted(annos.index)))
    anno = annos[key]
    parts = key.split("-")
    category = anno["category"]
    src = read_rgb(root / "JPEGImages" / category / f"{parts[1]}.jpg")
    trg = read_rgb(root / "JPEGImages" / category / f"{parts[2].split(':')[0]}.jpg")
    make_pair_thumbnail(src, trg, FIG_DIR / "thumb_train_spair.png")


def write_pointodyssey() -> None:
    seq_dir = sorted((DATA_ROOT / "PointOdyssey" / "train").iterdir())[0]
    frame_dir = seq_dir / "rgbs"
    paths = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])[:3]
    imgs = [read_rgb(p) for p in paths]
    make_triplet_thumbnail(imgs, FIG_DIR / "thumb_train_pointodyssey.png")


def write_sdf() -> None:
    img = read_rgb(FIG_DIR / "ch2_sdf_julia.png")
    make_triplet_thumbnail([img, img, img], FIG_DIR / "thumb_train_sdf_fractal3d.png")


def write_flyingthings() -> None:
    base = DATA_ROOT / "FlyingThings3D_tiny" / "FlyingThings3D" / "frames_cleanpass" / "TRAIN" / "A" / "0003" / "left"
    src = read_rgb(base / "0006.png")
    trg = read_rgb(base / "0007.png")
    make_pair_thumbnail(src, trg, FIG_DIR / "thumb_train_flyingthings3d.png")


def write_kitti() -> None:
    base = DATA_ROOT / "correspondence" / "kitti" / "kitti-2015" / "training" / "image_2"
    src = read_rgb(base / "000000_10.png")
    trg = read_rgb(base / "000000_11.png")
    make_pair_thumbnail(src, trg, FIG_DIR / "thumb_eval_kitti2015.png")


def write_pfpascal() -> None:
    root = DATA_ROOT / "correspondence" / "PF-PASCAL"
    with (root / "trn_pairs.csv").open() as f:
        reader = csv.reader(f)
        next(reader)
        row = next(reader)
    src = read_rgb(root / "JPEGImages" / Path(row[0]).name)
    trg = read_rgb(root / "JPEGImages" / Path(row[1]).name)
    make_pair_thumbnail(src, trg, FIG_DIR / "thumb_eval_pfpascal.png")


def write_pfwillow() -> None:
    root = REPO_ROOT / "models" / "Datasets_CATs" / "PF-WILLOW"
    df = pd.read_csv(root / "test_pairs.csv")
    row = df.iloc[0]
    src = read_rgb(root / Path(str(row["imageA"]).replace("PF-dataset/", "")))
    trg = read_rgb(root / Path(str(row["imageB"]).replace("PF-dataset/", "")))
    make_pair_thumbnail(src, trg, FIG_DIR / "thumb_eval_pfwillow.png")


def write_tss() -> None:
    pair_dir = sorted((DATA_ROOT / "correspondence" / "TSS_CVPR2016").glob("*/*"))[0]
    src = read_rgb(pair_dir / "image1.png")
    trg = read_rgb(pair_dir / "image2.png")
    make_pair_thumbnail(src, trg, FIG_DIR / "thumb_eval_tss.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    write_spair()
    write_pointodyssey()
    write_sdf()
    write_flyingthings()
    write_kitti()
    write_pfpascal()
    write_pfwillow()
    write_tss()
    for name in [
        "thumb_train_spair.png",
        "thumb_train_pointodyssey.png",
        "thumb_train_sdf_fractal3d.png",
        "thumb_train_flyingthings3d.png",
        "thumb_eval_kitti2015.png",
        "thumb_eval_pfpascal.png",
        "thumb_eval_pfwillow.png",
        "thumb_eval_tss.png",
    ]:
        print(f"wrote {FIG_DIR / name}")


if __name__ == "__main__":
    main()
