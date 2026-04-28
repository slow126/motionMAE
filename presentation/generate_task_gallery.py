#!/usr/bin/env python3
"""
Generate representative correspondence task figures for the slide-2 gallery.

The gallery uses a single visual grammar:
  - source image on the left
  - target image on the right
  - colored correspondence points on both images
  - thin connecting lines across the gutter

Dense tasks use a sampled subset of valid correspondences. Sparse tasks may
optionally show small numeric labels.

Output files are written into presentation/figures/ by default:
  - task_optical_flow_kitti.png
  - task_semantic_matching_spair.png
  - task_dense_tracking_pointodyssey.png
  - task_cross_modal_magic.png
  - task_scene_flow_flyingthings.png
  - task_template_matching_tss.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image
from matplotlib.collections import LineCollection


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "presentation" / "figures"
DATA_ROOT = Path("/home/spencer/Data")


@dataclass
class FigureSpec:
    name: str
    src_img: np.ndarray
    trg_img: np.ndarray
    src_pts: np.ndarray
    trg_pts: np.ndarray
    dataset_name: str
    subtitle: str
    label_points: bool = False


@dataclass
class TrackingSequenceSpec:
    name: str
    frames: list[np.ndarray]
    tracks: list[np.ndarray]
    dataset_name: str
    subtitle: str


@dataclass
class CrossModalSpec:
    name: str
    src_img: np.ndarray
    trg_img: np.ndarray
    dataset_name: str
    subtitle: str
    src_label: str
    trg_label: str


def read_rgb(path: Path) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[-1] > 3:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_pair(
    src_img: np.ndarray,
    trg_img: np.ndarray,
    src_pts: np.ndarray,
    trg_pts: np.ndarray,
    target_h: int = 420,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def _resize(img: np.ndarray, pts: np.ndarray):
        h, w = img.shape[:2]
        scale = target_h / float(h)
        new_w = max(1, int(round(w * scale)))
        resized = np.array(Image.fromarray(img).resize((new_w, target_h), Image.Resampling.BILINEAR))
        pts_resized = pts.astype(np.float32).copy()
        pts_resized[:, 0] *= scale
        pts_resized[:, 1] *= scale
        return resized, pts_resized

    src_img, src_pts = _resize(src_img, src_pts)
    trg_img, trg_pts = _resize(trg_img, trg_pts)
    return src_img, trg_img, src_pts, trg_pts


def filter_valid_correspondences(
    src_pts: np.ndarray,
    trg_pts: np.ndarray,
    src_shape: tuple[int, int, int],
    trg_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h0, w0 = src_shape[:2]
    h1, w1 = trg_shape[:2]
    src_pts = np.asarray(src_pts, dtype=np.float32)
    trg_pts = np.asarray(trg_pts, dtype=np.float32)
    finite = np.isfinite(src_pts).all(axis=1) & np.isfinite(trg_pts).all(axis=1)
    in_bounds = (
        (src_pts[:, 0] >= 0) & (src_pts[:, 0] < w0) &
        (src_pts[:, 1] >= 0) & (src_pts[:, 1] < h0) &
        (trg_pts[:, 0] >= 0) & (trg_pts[:, 0] < w1) &
        (trg_pts[:, 1] >= 0) & (trg_pts[:, 1] < h1)
    )
    keep = finite & in_bounds
    return src_pts[keep], trg_pts[keep]


def sample_correspondences(
    src_pts: np.ndarray,
    trg_pts: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(src_pts)
    if n <= max_points:
        return src_pts, trg_pts

    rng = np.random.default_rng(seed)
    motion = np.linalg.norm(trg_pts - src_pts, axis=1)
    weights = motion + 1.0
    weights = weights / weights.sum()
    indices = rng.choice(n, size=max_points, replace=False, p=weights)
    indices = np.sort(indices)
    return src_pts[indices], trg_pts[indices]


def render_correspondence_figure(spec: FigureSpec, out_path: Path) -> None:
    src_img, trg_img, src_pts, trg_pts = resize_pair(
        spec.src_img, spec.trg_img, spec.src_pts, spec.trg_pts
    )
    gutter = 40
    left_x0 = 0
    right_x0 = src_img.shape[1] + gutter
    canvas_w = src_img.shape[1] + trg_img.shape[1] + gutter
    canvas_h = max(src_img.shape[0], trg_img.shape[0])

    fig = plt.figure(figsize=(10.0, 4.4), facecolor="#0F172A")
    ax_left = 0.03
    ax_bottom = 0.16
    ax_width = 0.94
    ax_height = 0.72
    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.imshow(src_img, extent=[left_x0, left_x0 + src_img.shape[1], src_img.shape[0], 0], zorder=1)
    ax.imshow(trg_img, extent=[right_x0, right_x0 + trg_img.shape[1], trg_img.shape[0], 0], zorder=1)

    border_color = "#38BDF8"
    ax.add_patch(plt.Rectangle((left_x0, 0), src_img.shape[1], src_img.shape[0], fill=False, ec=border_color, lw=2.0, zorder=4))
    ax.add_patch(plt.Rectangle((right_x0, 0), trg_img.shape[1], trg_img.shape[0], fill=False, ec=border_color, lw=2.0, zorder=4))

    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(src_pts)))
    src_plot = src_pts.copy()
    trg_plot = trg_pts.copy()
    src_plot[:, 0] += left_x0
    trg_plot[:, 0] += right_x0

    segments = np.stack([src_plot, trg_plot], axis=1)
    lc = LineCollection(segments, colors=colors, linewidths=1.3, alpha=0.9, zorder=2)
    ax.add_collection(lc)
    ax.scatter(src_plot[:, 0], src_plot[:, 1], s=28, c=colors, edgecolors="white", linewidths=0.7, zorder=3)
    ax.scatter(trg_plot[:, 0], trg_plot[:, 1], s=28, c=colors, edgecolors="white", linewidths=0.7, zorder=3)

    if spec.label_points:
        for idx, (p0, p1, color) in enumerate(zip(src_plot, trg_plot, colors), start=1):
            for pt in (p0, p1):
                ax.text(
                    pt[0] + 6,
                    pt[1] - 6,
                    str(idx),
                    color="white",
                    fontsize=7,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=color, edgecolor="white", linewidth=0.5),
                    zorder=5,
                )

    fig.text(0.02, 0.955, spec.dataset_name, color="#E2E8F0", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.915, spec.subtitle, color="#94A3B8", fontsize=9)
    src_center = (left_x0 + src_img.shape[1] / 2.0) / canvas_w
    trg_center = (right_x0 + trg_img.shape[1] / 2.0) / canvas_w
    fig.text(ax_left + ax_width * src_center, 0.04, "source", color="#CBD5E1", fontsize=9, ha="center")
    fig.text(ax_left + ax_width * trg_center, 0.04, "target", color="#CBD5E1", fontsize=9, ha="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_tracking_sequence_figure(spec: TrackingSequenceSpec, out_path: Path) -> None:
    target_h = 320
    resized_frames = []
    resized_tracks = []
    for frame, pts in zip(spec.frames, spec.tracks):
        h, w = frame.shape[:2]
        scale = target_h / float(h)
        new_w = max(1, int(round(w * scale)))
        resized = np.array(Image.fromarray(frame).resize((new_w, target_h), Image.Resampling.BILINEAR))
        pts_resized = pts.astype(np.float32).copy()
        pts_resized[:, 0] *= scale
        pts_resized[:, 1] *= scale
        resized_frames.append(resized)
        resized_tracks.append(pts_resized)

    gap = 26
    x_offsets = []
    x = 0
    for frame in resized_frames:
        x_offsets.append(x)
        x += frame.shape[1] + gap
    canvas_w = x - gap
    canvas_h = target_h

    fig = plt.figure(figsize=(10.0, 4.4), facecolor="#0F172A")
    ax_left = 0.03
    ax_bottom = 0.16
    ax_width = 0.94
    ax_height = 0.72
    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    border_color = "#38BDF8"
    for frame, x0 in zip(resized_frames, x_offsets):
        ax.imshow(frame, extent=[x0, x0 + frame.shape[1], frame.shape[0], 0], zorder=1)
        ax.add_patch(plt.Rectangle((x0, 0), frame.shape[1], frame.shape[0], fill=False, ec=border_color, lw=2.0, zorder=4))

    n_tracks = resized_tracks[0].shape[0]
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, n_tracks))
    for idx in range(n_tracks):
        pts = []
        for x0, track_pts in zip(x_offsets, resized_tracks):
            pt = track_pts[idx].copy()
            pt[0] += x0
            pts.append(pt)
        pts = np.asarray(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=colors[idx], lw=1.5, alpha=0.9, zorder=2)
        ax.scatter(pts[:, 0], pts[:, 1], s=24, color=colors[idx], edgecolors="white", linewidths=0.7, zorder=3)

    fig.text(0.02, 0.955, spec.dataset_name, color="#E2E8F0", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.915, spec.subtitle, color="#94A3B8", fontsize=9)
    for i in range(len(resized_frames)):
        fig_x = ax_left + ax_width * ((x_offsets[i] + resized_frames[i].shape[1] / 2.0) / canvas_w)
        fig.text(fig_x, 0.04, f"t{i}", color="#CBD5E1", fontsize=9, ha="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_cross_modal_figure(spec: CrossModalSpec, out_path: Path) -> None:
    src_img, trg_img, _, _ = resize_pair(
        spec.src_img,
        spec.trg_img,
        np.zeros((0, 2), dtype=np.float32),
        np.zeros((0, 2), dtype=np.float32),
    )
    gutter = 40
    left_x0 = 0
    right_x0 = src_img.shape[1] + gutter
    canvas_w = src_img.shape[1] + trg_img.shape[1] + gutter
    canvas_h = max(src_img.shape[0], trg_img.shape[0])

    fig = plt.figure(figsize=(10.0, 4.4), facecolor="#0F172A")
    ax_left = 0.03
    ax_bottom = 0.16
    ax_width = 0.94
    ax_height = 0.72
    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.imshow(src_img, extent=[left_x0, left_x0 + src_img.shape[1], src_img.shape[0], 0], zorder=1)
    ax.imshow(trg_img, extent=[right_x0, right_x0 + trg_img.shape[1], trg_img.shape[0], 0], zorder=1)

    border_color = "#38BDF8"
    ax.add_patch(plt.Rectangle((left_x0, 0), src_img.shape[1], src_img.shape[0], fill=False, ec=border_color, lw=2.0, zorder=4))
    ax.add_patch(plt.Rectangle((right_x0, 0), trg_img.shape[1], trg_img.shape[0], fill=False, ec=border_color, lw=2.0, zorder=4))

    badge_style = dict(boxstyle="round,pad=0.28", facecolor="#0F172A", edgecolor="white", linewidth=1.1)
    ax.text(
        left_x0 + 12,
        24,
        spec.src_label,
        color="white",
        fontsize=9,
        fontweight="bold",
        va="top",
        bbox=badge_style,
        zorder=5,
    )
    ax.text(
        right_x0 + 12,
        24,
        spec.trg_label,
        color="white",
        fontsize=9,
        fontweight="bold",
        va="top",
        bbox=badge_style,
        zorder=5,
    )

    fig.text(0.02, 0.955, spec.dataset_name, color="#E2E8F0", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.915, spec.subtitle, color="#94A3B8", fontsize=9)
    src_center = (left_x0 + src_img.shape[1] / 2.0) / canvas_w
    trg_center = (right_x0 + trg_img.shape[1] / 2.0) / canvas_w
    fig.text(ax_left + ax_width * src_center, 0.04, "source", color="#CBD5E1", fontsize=9, ha="center")
    fig.text(ax_left + ax_width * trg_center, 0.04, "target", color="#CBD5E1", fontsize=9, ha="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def read_kitti_flow(path: Path) -> tuple[np.ndarray, np.ndarray]:
    flo = cv2.imread(str(path), -1)
    flow = flo[:, :, 2:0:-1].astype(np.float32)
    valid = flo[:, :, 0] == 1
    flow = (flow - 32768.0) / 64.0
    flow[~valid] = np.nan
    return flow, valid


def kitti_spec(seed: int) -> FigureSpec:
    root = DATA_ROOT / "correspondence" / "kitti" / "kitti-2015" / "training"
    flow_path = sorted((root / "flow_occ").glob("*.png"))[0]
    stem = flow_path.stem.replace("_10", "")
    src_path = root / "image_2" / f"{stem}_10.png"
    trg_path = root / "image_2" / f"{stem}_11.png"

    src_img = read_rgb(src_path)
    trg_img = read_rgb(trg_path)
    flow, valid = read_kitti_flow(flow_path)

    ys, xs = np.where(valid)
    src_pts = np.stack([xs, ys], axis=1).astype(np.float32)
    trg_pts = src_pts + flow[ys, xs]
    src_pts, trg_pts = sample_correspondences(src_pts, trg_pts, max_points=18, seed=seed)
    return FigureSpec(
        name="task_optical_flow_kitti.png",
        src_img=src_img,
        trg_img=trg_img,
        src_pts=src_pts,
        trg_pts=trg_pts,
        dataset_name="KITTI-2015",
        subtitle="Dense optical flow sampled from valid motion vectors",
    )


def spair_spec(seed: int) -> FigureSpec:
    root = DATA_ROOT / "correspondence" / "SPair-71k"
    pairs_path = root / "PairAnnotation" / "trn" / "pair_annotations.json"
    pair_annos = json.loads(pairs_path.read_text())
    keys = sorted(pair_annos.keys())
    key = None
    anno = None
    for cand in keys:
        candidate = pair_annos[cand]
        src_kps = np.asarray(candidate["src_kps"], dtype=np.float32)
        trg_kps = np.asarray(candidate["trg_kps"], dtype=np.float32)
        valid = np.isfinite(src_kps).all(axis=1) & np.isfinite(trg_kps).all(axis=1)
        if valid.sum() >= 8:
            key = cand
            anno = candidate
            break
    if anno is None:
        raise RuntimeError("Could not find a usable SPair sample")

    parts = key.split("-")
    category = anno["category"]
    src_name = parts[1] + ".jpg"
    trg_name = parts[2].split(":")[0] + ".jpg"

    src_img = read_rgb(root / "JPEGImages" / category / src_name)
    trg_img = read_rgb(root / "JPEGImages" / category / trg_name)
    src_pts = np.asarray(anno["src_kps"], dtype=np.float32)
    trg_pts = np.asarray(anno["trg_kps"], dtype=np.float32)
    src_pts, trg_pts = filter_valid_correspondences(src_pts, trg_pts, src_img.shape, trg_img.shape)
    src_pts, trg_pts = sample_correspondences(src_pts, trg_pts, max_points=10, seed=seed)

    return FigureSpec(
        name="task_semantic_matching_spair.png",
        src_img=src_img,
        trg_img=trg_img,
        src_pts=src_pts,
        trg_pts=trg_pts,
        dataset_name=f"SPair-71k ({category})",
        subtitle="Sparse semantic keypoints across pose/viewpoint change",
        label_points=True,
    )


def choose_pointodyssey_pair(seq_dir: Path) -> tuple[int, int, np.ndarray, np.ndarray]:
    anno = np.load(seq_dir / "anno.npz")
    trajs = anno["trajs_2d"]
    valids = anno["valids"].astype(bool)
    visibs = anno["visibs"].astype(bool) if "visibs" in anno.files else np.ones_like(valids, dtype=bool)

    best = None
    for i in range(min(8, trajs.shape[0] - 1)):
        for j in range(i + 2, min(i + 12, trajs.shape[0])):
            mask = valids[i] & valids[j] & visibs[i] & visibs[j]
            if mask.sum() < 50:
                continue
            src_pts = trajs[i][mask]
            trg_pts = trajs[j][mask]
            motion = np.linalg.norm(trg_pts - src_pts, axis=1)
            score = motion.mean()
            if best is None or score > best[0]:
                best = (score, i, j, src_pts, trg_pts)
    if best is None:
        raise RuntimeError(f"Could not find a usable PointOdyssey pair in {seq_dir}")
    _, i, j, src_pts, trg_pts = best
    return i, j, src_pts, trg_pts


def choose_pointodyssey_sequence(seq_dir: Path, n_frames: int = 4) -> tuple[list[int], list[np.ndarray]]:
    anno = np.load(seq_dir / "anno.npz")
    trajs = anno["trajs_2d"]
    valids = anno["valids"].astype(bool)
    visibs = anno["visibs"].astype(bool) if "visibs" in anno.files else np.ones_like(valids, dtype=bool)

    best = None
    max_start = max(1, trajs.shape[0] - n_frames)
    for start in range(min(12, max_start)):
        frame_ids = list(range(start, start + n_frames))
        mask = np.ones(trajs.shape[1], dtype=bool)
        for f in frame_ids:
            mask &= valids[f] & visibs[f]
        if mask.sum() < 12:
            continue
        tracks = [trajs[f][mask] for f in frame_ids]
        motion = np.linalg.norm(tracks[-1] - tracks[0], axis=1)
        score = motion.mean()
        if best is None or score > best[0]:
            best = (score, frame_ids, tracks)
    if best is None:
        raise RuntimeError(f"Could not find a usable PointOdyssey track sequence in {seq_dir}")
    _, frame_ids, tracks = best
    return frame_ids, tracks


def find_pointodyssey_frame(frame_dir: Path, frame_idx: int) -> Path:
    stem = f"{frame_idx:06d}"
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        path = frame_dir / f"{stem}{ext}"
        if path.exists():
            return path
    candidates = sorted(frame_dir.iterdir())
    if frame_idx < len(candidates):
        return candidates[frame_idx]
    raise FileNotFoundError(f"Could not resolve PointOdyssey frame {frame_idx} in {frame_dir}")


def pointodyssey_spec(seed: int) -> FigureSpec:
    seq_dir = sorted((DATA_ROOT / "PointOdyssey" / "train").iterdir())[0]
    frame_i, frame_j, src_pts, trg_pts = choose_pointodyssey_pair(seq_dir)
    src_img = read_rgb(find_pointodyssey_frame(seq_dir / "rgbs", frame_i))
    trg_img = read_rgb(find_pointodyssey_frame(seq_dir / "rgbs", frame_j))
    src_pts, trg_pts = filter_valid_correspondences(src_pts, trg_pts, src_img.shape, trg_img.shape)
    src_pts, trg_pts = sample_correspondences(src_pts, trg_pts, max_points=22, seed=seed)

    return FigureSpec(
        name="task_dense_tracking_pointodyssey.png",
        src_img=src_img,
        trg_img=trg_img,
        src_pts=src_pts,
        trg_pts=trg_pts,
        dataset_name=f"PointOdyssey ({seq_dir.name})",
        subtitle=f"Long-range tracked points from frames {frame_i} → {frame_j}",
    )


def pointodyssey_tracking_sequence_spec(seed: int) -> TrackingSequenceSpec:
    seq_dir = sorted((DATA_ROOT / "PointOdyssey" / "train").iterdir())[0]
    frame_ids, tracks = choose_pointodyssey_sequence(seq_dir, n_frames=4)
    frames = [read_rgb(find_pointodyssey_frame(seq_dir / "rgbs", frame_idx)) for frame_idx in frame_ids]

    valid_tracks = tracks
    ref_src = valid_tracks[0]
    ref_trg = valid_tracks[-1]
    ref_src, ref_trg = filter_valid_correspondences(ref_src, ref_trg, frames[0].shape, frames[-1].shape)
    keep_src, keep_trg = sample_correspondences(ref_src, ref_trg, max_points=12, seed=seed)

    keep_indices = []
    for pt in keep_src:
        diffs = np.linalg.norm(valid_tracks[0] - pt[None, :], axis=1)
        keep_indices.append(int(np.argmin(diffs)))
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    sampled_tracks = [pts[keep_indices] for pts in valid_tracks]

    return TrackingSequenceSpec(
        name="task_dense_tracking_pointodyssey.png",
        frames=frames,
        tracks=sampled_tracks,
        dataset_name=f"PointOdyssey ({seq_dir.name})",
        subtitle=f"Short tracked sequence across frames {frame_ids[0]} → {frame_ids[-1]}",
    )


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header = f.readline().decode("latin-1").rstrip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"Invalid PFM header in {path}")

        dims = f.readline().decode("latin-1")
        while dims.startswith("#"):
            dims = f.readline().decode("latin-1")
        width, height = map(int, dims.strip().split())

        scale = float(f.readline().decode("latin-1").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")

        shape = (height, width, 3) if header == "PF" else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data


def flyingthings_spec(seed: int) -> FigureSpec:
    base = DATA_ROOT / "FlyingThings3D_tiny" / "FlyingThings3D"
    src_path = base / "frames_cleanpass" / "TRAIN" / "A" / "0003" / "left" / "0006.png"
    trg_path = base / "frames_cleanpass" / "TRAIN" / "A" / "0003" / "left" / "0007.png"
    flow_path = base / "optical_flow" / "TRAIN" / "A" / "0003" / "into_future" / "left" / "OpticalFlowIntoFuture_0006_L.pfm"

    src_img = read_rgb(src_path)
    trg_img = read_rgb(trg_path)
    flow = read_pfm(flow_path)[..., :2].astype(np.float32)
    valid = np.isfinite(flow).all(axis=2) & (np.abs(flow).sum(axis=2) > 1e-5)
    ys, xs = np.where(valid)
    src_pts = np.stack([xs, ys], axis=1).astype(np.float32)
    trg_pts = src_pts + flow[ys, xs]
    src_pts, trg_pts = sample_correspondences(src_pts, trg_pts, max_points=18, seed=seed)

    return FigureSpec(
        name="task_scene_flow_flyingthings.png",
        src_img=src_img,
        trg_img=trg_img,
        src_pts=src_pts,
        trg_pts=trg_pts,
        dataset_name="FlyingThings3D",
        subtitle="Synthetic dense correspondences for 3D scene motion",
    )


def tss_spec(seed: int) -> FigureSpec:
    pair_dir = sorted((DATA_ROOT / "correspondence" / "TSS_CVPR2016").glob("*/*"))[0]
    src_img = read_rgb(pair_dir / "image1.png")
    trg_img = read_rgb(pair_dir / "image2.png")

    flow_path = pair_dir / "flow1.flo"
    with flow_path.open("rb") as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if not math.isclose(float(magic), 202021.25, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError(f"Invalid .flo file header in {flow_path}")
        w = int(np.fromfile(f, np.int32, count=1)[0])
        h = int(np.fromfile(f, np.int32, count=1)[0])
        flow = np.fromfile(f, np.float32, count=2 * w * h).reshape(h, w, 2)

    valid = np.isfinite(flow).all(axis=2) & (np.abs(flow).sum(axis=2) < 1e9)
    ys, xs = np.where(valid)
    src_pts = np.stack([xs, ys], axis=1).astype(np.float32)
    trg_pts = src_pts + flow[ys, xs]
    src_pts, trg_pts = sample_correspondences(src_pts, trg_pts, max_points=14, seed=seed)

    return FigureSpec(
        name="task_template_matching_tss.png",
        src_img=src_img,
        trg_img=trg_img,
        src_pts=src_pts,
        trg_pts=trg_pts,
        dataset_name=f"TSS ({pair_dir.parent.name})",
        subtitle="Template-style matching across object instances",
    )


def cross_modal_spec() -> CrossModalSpec:
    fig_dir = REPO_ROOT / "presentation" / "figures"
    return CrossModalSpec(
        name="task_cross_modal_magic.png",
        src_img=read_rgb(fig_dir / "EO.png"),
        trg_img=read_rgb(fig_dir / "Sar.png"),
        dataset_name="MAGIC",
        subtitle="Cross-modal SAR and EO pair without explicit point annotations",
        src_label="EO",
        trg_label="SAR",
    )


def pfpascal_probe() -> tuple[str, int]:
    root = DATA_ROOT / "correspondence" / "PF-PASCAL"
    csv_path = root / "trn_pairs.csv"
    with csv_path.open() as f:
        reader = csv.reader(f)
        first = None
        for row in reader:
            if row and row[0] != "source_image":
                first = row
                break
    if first is None:
        raise RuntimeError("Could not find a PF-PASCAL sample row")
    cls_id = int(first[2]) - 1
    classes = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor",
    ]
    category = classes[cls_id]
    mat_path = root / "Annotations" / category / Path(first[0]).name.replace("jpg", "mat")
    mat = sio.loadmat(mat_path)
    kps = np.asarray(mat["kps"], dtype=np.float32)
    valid = np.isfinite(kps).all(axis=0)
    return category, int(valid.sum())


def build_specs(seed: int) -> list[FigureSpec]:
    specs = [
        kitti_spec(seed + 1),
        spair_spec(seed + 2),
        flyingthings_spec(seed + 4),
        tss_spec(seed + 5),
    ]
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    specs = build_specs(args.seed)
    for spec in specs:
        render_correspondence_figure(spec, args.out_dir / spec.name)
        print(f"wrote {args.out_dir / spec.name}")

    tracking_spec = pointodyssey_tracking_sequence_spec(args.seed + 3)
    render_tracking_sequence_figure(tracking_spec, args.out_dir / tracking_spec.name)
    print(f"wrote {args.out_dir / tracking_spec.name}")

    cross_modal = cross_modal_spec()
    render_cross_modal_figure(cross_modal, args.out_dir / cross_modal.name)
    print(f"wrote {args.out_dir / cross_modal.name}")

    category, n_pts = pfpascal_probe()
    print(f"pf-pascal probe: found usable category '{category}' with {n_pts} visible keypoints")


if __name__ == "__main__":
    main()
