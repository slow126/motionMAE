#!/usr/bin/env python3
"""
Warp a source image into a target frame using a dense flow field and compare it to the target.

Supports two flow conventions:
- `src_to_trg`: flow is defined on the source image and points to target coordinates.
- `trg_to_src`: flow is defined on the target image and points back to source coordinates.

Outputs:
- warped_source.png
- valid_mask.png
- abs_diff.png
- comparison_strip.png
- meta.json

Examples:
  python scripts/warp_image_diagnostic.py \
    --src-image a.png \
    --trg-image b.png \
    --flow flow.npy \
    --flow-direction src_to_trg \
    --out-dir tmp/warp_diag

  python scripts/warp_image_diagnostic.py \
    --src-image image1.png \
    --trg-image image2.png \
    --flow flow2.flo \
    --flow-direction trg_to_src \
    --out-dir tmp/warp_diag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.io import read_flo_file

try:
    from models.RAFT.core.utils.flow_viz import flow_to_image
except Exception:
    flow_to_image = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-image", type=Path, required=True)
    parser.add_argument("--trg-image", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument(
        "--flow-encoding",
        choices=("auto", "array", "rg_centered"),
        default="auto",
        help="How to interpret --flow. 'rg_centered' decodes dx/dy from image R/G channels centered at --flow-zero.",
    )
    parser.add_argument(
        "--flow-direction",
        choices=("src_to_trg", "trg_to_src"),
        required=True,
        help="Whether flow is defined on source pixels or target pixels.",
    )
    parser.add_argument(
        "--flow-zero",
        type=float,
        default=0.5,
        help="Zero-displacement channel value when using --flow-encoding rg_centered.",
    )
    parser.add_argument(
        "--flow-scale-x",
        type=float,
        default=None,
        help="Positive x displacement corresponding to channel value 1.0 when using rg_centered. Defaults to target/source width depending on flow direction.",
    )
    parser.add_argument(
        "--flow-scale-y",
        type=float,
        default=None,
        help="Positive y displacement corresponding to channel value 1.0 when using rg_centered. Defaults to target/source height depending on flow direction.",
    )
    parser.add_argument(
        "--flow-valid-threshold",
        type=float,
        default=0.02,
        help="Pixels with RGB sum at or below this are treated as invalid when decoding rg_centered flow images.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/warp_diag"))
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_flow_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".flo":
        flow = read_flo_file(path)
    elif suffix == ".npy":
        flow = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        if "flow" in data:
            flow = data["flow"]
        else:
            first_key = next(iter(data.keys()))
            flow = data[first_key]
    else:
        raise ValueError(f"Unsupported flow format: {path}")

    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3:
        raise ValueError(f"Expected flow with shape (H, W, 2) or (2, H, W), got {flow.shape}")
    if flow.shape[-1] == 2:
        pass
    elif flow.shape[0] == 2:
        flow = np.moveaxis(flow, 0, -1)
    else:
        raise ValueError(f"Unsupported flow shape: {flow.shape}")

    invalid = np.abs(flow) > 1e9
    flow[invalid] = np.inf
    return flow.astype(np.float32)


def decode_rg_centered_flow(
    path: Path,
    out_hw: tuple[int, int],
    zero: float,
    scale_x: float,
    scale_y: float,
    valid_threshold: float,
) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((out_hw[1], out_hw[0]), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    flow = np.empty((out_hw[0], out_hw[1], 2), dtype=np.float32)
    flow[..., 0] = (arr[..., 0] - zero) * (2.0 * scale_x)
    flow[..., 1] = (arr[..., 1] - zero) * (2.0 * scale_y)

    valid = arr.sum(axis=2) > valid_threshold
    flow[~valid] = np.inf
    return flow


def resize_flow(flow: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    if flow.shape[:2] == out_hw:
        return flow

    in_h, in_w = flow.shape[:2]
    out_h, out_w = out_hw
    valid = np.isfinite(flow).all(axis=2)
    safe = flow.copy()
    safe[~valid] = 0.0

    tensor = torch.from_numpy(safe).permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(tensor, size=out_hw, mode="bilinear", align_corners=True)
    tensor[:, 0] *= float(out_w) / float(in_w)
    tensor[:, 1] *= float(out_h) / float(in_h)
    out = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    valid_tensor = torch.from_numpy(valid.astype(np.float32))[None, None]
    valid_resized = F.interpolate(valid_tensor, size=out_hw, mode="nearest").squeeze().cpu().numpy() > 0.5
    out[~valid_resized] = np.inf
    return out.astype(np.float32)


def load_flow(
    path: Path,
    encoding: str,
    out_hw: tuple[int, int],
    zero: float,
    scale_x: float | None,
    scale_y: float | None,
    valid_threshold: float,
) -> np.ndarray:
    suffix = path.suffix.lower()
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    if encoding == "auto":
        encoding = "rg_centered" if suffix in image_suffixes else "array"

    if encoding == "array":
        return resize_flow(load_flow_array(path), out_hw)

    if encoding == "rg_centered":
        sx = float(scale_x if scale_x is not None else out_hw[1])
        sy = float(scale_y if scale_y is not None else out_hw[0])
        return decode_rg_centered_flow(path, out_hw, zero, sx, sy, valid_threshold)

    raise ValueError(f"Unsupported flow encoding: {encoding}")


def backward_warp(src_img: np.ndarray, flow_trg_to_src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_h, src_w = src_img.shape[:2]
    trg_h, trg_w = flow_trg_to_src.shape[:2]

    yy, xx = np.meshgrid(
        np.arange(trg_h, dtype=np.float32),
        np.arange(trg_w, dtype=np.float32),
        indexing="ij",
    )
    source_xy = np.stack([xx + flow_trg_to_src[..., 0], yy + flow_trg_to_src[..., 1]], axis=-1)

    finite = np.isfinite(source_xy).all(axis=2)
    grid = torch.from_numpy(source_xy.copy())
    grid_x = (grid[..., 0] / max(src_w - 1, 1)) * 2.0 - 1.0
    grid_y = (grid[..., 1] / max(src_h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    src = torch.from_numpy(src_img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    warped = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    warped = warped.squeeze(0).permute(1, 2, 0).cpu().numpy()

    in_bounds = (
        finite
        & (source_xy[..., 0] >= 0.0)
        & (source_xy[..., 0] <= float(src_w - 1))
        & (source_xy[..., 1] >= 0.0)
        & (source_xy[..., 1] <= float(src_h - 1))
    )
    warped[~in_bounds] = 0.0
    return (warped * 255.0).clip(0.0, 255.0).astype(np.uint8), in_bounds


def forward_splat(src_img: np.ndarray, flow_src_to_trg: np.ndarray, trg_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    trg_h, trg_w = trg_hw
    src_h, src_w = src_img.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(src_h, dtype=np.float32),
        np.arange(src_w, dtype=np.float32),
        indexing="ij",
    )

    pos_x = xx + flow_src_to_trg[..., 0]
    pos_y = yy + flow_src_to_trg[..., 1]
    finite = np.isfinite(pos_x) & np.isfinite(pos_y)

    src_rgb = src_img.astype(np.float32) / 255.0
    pos_x = pos_x[finite]
    pos_y = pos_y[finite]
    rgb = src_rgb[finite]

    if pos_x.size == 0:
        return np.zeros((trg_h, trg_w, 3), dtype=np.uint8), np.zeros((trg_h, trg_w), dtype=bool)

    x0 = np.floor(pos_x).astype(np.int64)
    y0 = np.floor(pos_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = pos_x - x0
    wy = pos_y - y0

    acc_rgb = np.zeros((trg_h, trg_w, 3), dtype=np.float32)
    acc_w = np.zeros((trg_h, trg_w), dtype=np.float32)

    splats = (
        (x0, y0, (1.0 - wx) * (1.0 - wy)),
        (x1, y0, wx * (1.0 - wy)),
        (x0, y1, (1.0 - wx) * wy),
        (x1, y1, wx * wy),
    )
    for xi, yi, wgt in splats:
        valid = (xi >= 0) & (xi < trg_w) & (yi >= 0) & (yi < trg_h) & (wgt > 0)
        if not np.any(valid):
            continue
        np.add.at(acc_w, (yi[valid], xi[valid]), wgt[valid])
        for ch in range(3):
            np.add.at(acc_rgb[..., ch], (yi[valid], xi[valid]), rgb[valid, ch] * wgt[valid])

    warped = np.zeros_like(acc_rgb)
    valid = acc_w > 1e-6
    warped[valid] = acc_rgb[valid] / acc_w[valid, None]
    return (warped * 255.0).clip(0.0, 255.0).astype(np.uint8), valid


def make_abs_diff(warped: np.ndarray, target: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    diff = np.abs(warped.astype(np.float32) - target.astype(np.float32))
    diff_vis = diff.clip(0.0, 255.0).astype(np.uint8)

    if np.any(valid):
        valid_diff = diff[valid]
        mae = float(valid_diff.mean())
        mse = float(np.square(valid_diff).mean())
        rmse = float(np.sqrt(mse))
        psnr = float(20.0 * np.log10(255.0 / max(rmse, 1e-8)))
    else:
        mae = mse = rmse = psnr = float("nan")

    diff_vis[~valid] = 0
    metrics = {
        "mae_rgb_0_255": mae,
        "mse_rgb_0_255": mse,
        "rmse_rgb_0_255": rmse,
        "psnr_db": psnr,
        "valid_pixel_count": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
    }
    return diff_vis, metrics


def make_strip(src: np.ndarray, warped: np.ndarray, target: np.ndarray, diff: np.ndarray, valid: np.ndarray) -> np.ndarray:
    mask_rgb = np.repeat((valid.astype(np.uint8) * 255)[..., None], 3, axis=2)
    return np.concatenate([src, warped, target, diff, mask_rgb], axis=1)


def maybe_save_flow_vis(flow: np.ndarray, out_path: Path) -> None:
    if flow_to_image is None:
        return
    safe = flow.copy()
    valid = np.isfinite(safe).all(axis=2)
    safe[~valid] = 0.0
    vis = flow_to_image(safe.astype(np.float32))
    vis[~valid] = 0
    Image.fromarray(vis).save(out_path)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    src_img = load_rgb(args.src_image.expanduser().resolve())
    trg_img = load_rgb(args.trg_image.expanduser().resolve())
    if args.flow_direction == "src_to_trg":
        flow = load_flow(
            args.flow.expanduser().resolve(),
            args.flow_encoding,
            src_img.shape[:2],
            args.flow_zero,
            args.flow_scale_x,
            args.flow_scale_y,
            args.flow_valid_threshold,
        )
        warped, valid = forward_splat(src_img, flow, trg_img.shape[:2])
    else:
        flow = load_flow(
            args.flow.expanduser().resolve(),
            args.flow_encoding,
            trg_img.shape[:2],
            args.flow_zero,
            args.flow_scale_x,
            args.flow_scale_y,
            args.flow_valid_threshold,
        )
        warped, valid = backward_warp(src_img, flow)

    diff, metrics = make_abs_diff(warped, trg_img, valid)
    strip = make_strip(src_img, warped, trg_img, diff, valid)

    Image.fromarray(src_img).save(out_dir / "source.png")
    Image.fromarray(trg_img).save(out_dir / "target.png")
    Image.fromarray(warped).save(out_dir / "warped_source.png")
    Image.fromarray((valid.astype(np.uint8) * 255)).save(out_dir / "valid_mask.png")
    Image.fromarray(diff).save(out_dir / "abs_diff.png")
    Image.fromarray(strip).save(out_dir / "comparison_strip.png")
    maybe_save_flow_vis(flow, out_dir / "flow_vis.png")

    meta = {
        "src_image": str(args.src_image),
        "trg_image": str(args.trg_image),
        "flow": str(args.flow),
        "flow_encoding": args.flow_encoding,
        "flow_direction": args.flow_direction,
        "source_shape_hwc": list(src_img.shape),
        "target_shape_hwc": list(trg_img.shape),
        "flow_shape_hw2": list(flow.shape),
    }
    meta.update(metrics)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"saved outputs to {out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
