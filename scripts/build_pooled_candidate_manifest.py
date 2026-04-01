#!/usr/bin/env python3
"""Build a pooled candidate manifest from PointOdyssey and SPAIR sources.

The output manifest is designed for:
1. raw-joint nearest-neighbor scoring against target benchmark vectors
2. manifest-backed pooled training via dataset_name='pooled_pairs'

PointOdyssey rows keep their original frame-pair fields.
SPAIR rows store direct image paths and sparse keypoints in original image space.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pooled candidate manifest")
    p.add_argument("--output", type=Path, required=True, help="Output pooled manifest JSONL path.")

    p.add_argument(
        "--pointodyssey-manifest",
        type=Path,
        default=None,
        help="Existing PointOdyssey manifest JSONL to include.",
    )
    p.add_argument(
        "--max-pointodyssey-rows",
        type=int,
        default=0,
        help="Optional cap on PointOdyssey rows (0 = all).",
    )

    p.add_argument(
        "--spair-root",
        type=Path,
        default=Path("./models/Datasets_CATs/SPair-71k"),
        help="Root of SPair-71k dataset.",
    )
    p.add_argument(
        "--include-spair",
        action="store_true",
        help="Include SPAIR trn pairs in the pooled manifest.",
    )
    p.add_argument(
        "--spair-split",
        type=str,
        default="trn",
        choices=["trn", "val", "test"],
        help="SPAIR split to include (default: trn).",
    )
    p.add_argument(
        "--max-spair-rows",
        type=int,
        default=0,
        help="Optional cap on SPAIR rows (0 = all).",
    )

    p.add_argument(
        "--pfpascal-datapath",
        type=Path,
        default=Path("./models/Datasets_CATs"),
        help="Parent datapath passed to PFPascalDataset; should contain PF-PASCAL/.",
    )
    p.add_argument(
        "--include-pfpascal",
        action="store_true",
        help="Include PF-PASCAL pairs in the pooled manifest.",
    )
    p.add_argument(
        "--pfpascal-split",
        type=str,
        default="trn",
        choices=["trn", "val", "test"],
        help="PF-PASCAL split to include (default: trn).",
    )
    p.add_argument(
        "--max-pfpascal-rows",
        type=int,
        default=0,
        help="Optional cap on PF-PASCAL rows (0 = all).",
    )
    return p.parse_args()


def _iter_pointodyssey_rows(path: Path, max_rows: int) -> Iterable[Dict]:
    with path.open("r") as f:
        for idx, line in enumerate(f):
            if max_rows > 0 and idx >= int(max_rows):
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out = dict(row)
            out["source_dataset"] = "pointodyssey"
            out["source_split"] = str(row.get("source_split", "train"))
            out["source_sample_id"] = int(row.get("pair_id", idx))
            out["source_pair_id"] = int(row.get("pair_id", idx))
            yield out


def _iter_spair_rows(root: Path, split: str, max_rows: int) -> Iterable[Dict]:
    split_name = "test" if split == "test" else split
    layout_path = root / "Layout" / "large" / f"{split_name}.txt"
    ann_dir = root / "PairAnnotation" / split_name
    img_root = root / "JPEGImages"

    if not layout_path.exists():
        raise FileNotFoundError(f"SPAIR split file not found: {layout_path}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"SPAIR annotation directory not found: {ann_dir}")
    if not img_root.exists():
        raise FileNotFoundError(f"SPAIR image directory not found: {img_root}")

    with layout_path.open("r") as f:
        pair_names = [line.strip() for line in f if line.strip()]
    if max_rows > 0:
        pair_names = pair_names[: int(max_rows)]

    for local_idx, pair_name in enumerate(pair_names):
        ann_path = ann_dir / f"{pair_name}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"SPAIR annotation file not found: {ann_path}")
        ann = json.loads(ann_path.read_text())
        category = str(ann["category"])
        src_img_path = img_root / category / str(ann["src_imname"])
        trg_img_path = img_root / category / str(ann["trg_imname"])

        src_kps = [[float(x), float(y)] for x, y in ann.get("src_kps", [])]
        trg_kps = [[float(x), float(y)] for x, y in ann.get("trg_kps", [])]
        if len(src_kps) != len(trg_kps):
            raise ValueError(f"SPAIR pair has mismatched keypoint counts: {ann_path}")

        yield {
            "source_dataset": "spair",
            "source_split": split_name,
            "source_sample_id": int(local_idx),
            "source_pair_id": int(ann.get("pair_id", local_idx)),
            "filename": str(ann.get("filename", pair_name)),
            "category": category,
            "category_id": category,
            "src_img_path": str(src_img_path),
            "trg_img_path": str(trg_img_path),
            "src_img_rel_path": str(Path(category) / str(ann["src_imname"])),
            "trg_img_rel_path": str(Path(category) / str(ann["trg_imname"])),
            "source_root": str(img_root),
            "src_imsize": ann.get("src_imsize"),
            "trg_imsize": ann.get("trg_imsize"),
            "src_bbox": ann.get("src_bndbox"),
            "trg_bbox": ann.get("trg_bndbox"),
            "src_kps": src_kps,
            "trg_kps": trg_kps,
            "n_pts": int(len(src_kps)),
            "valid_points": int(len(src_kps)),
        }


def _iter_pfpascal_rows(datapath: Path, split: str, max_rows: int) -> Iterable[Dict]:
    import pandas as pd
    import scipy.io as sio

    cls_names = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]

    base_path = Path(datapath) / "PF-PASCAL"
    split_csv = base_path / f"{split}_pairs.csv"
    image_root = base_path / "JPEGImages"
    ann_root = base_path / "Annotations"

    if not split_csv.exists():
        raise FileNotFoundError(f"PF-PASCAL split CSV not found: {split_csv}")
    if not image_root.exists():
        raise FileNotFoundError(f"PF-PASCAL image directory not found: {image_root}")
    if not ann_root.exists():
        raise FileNotFoundError(f"PF-PASCAL annotation directory not found: {ann_root}")

    table = pd.read_csv(split_csv)
    if max_rows > 0:
        table = table.iloc[: int(max_rows)]

    for local_idx, row in table.reset_index(drop=True).iterrows():
        cls_id = int(row["class"]) - 1
        if cls_id < 0 or cls_id >= len(cls_names):
            raise ValueError(f"Invalid PF-PASCAL class id {row['class']} at row {local_idx}")
        category = cls_names[cls_id]

        src_name = Path(str(row["source_image"])).name
        trg_name = Path(str(row["target_image"])).name
        src_ann_path = ann_root / category / src_name.replace(".jpg", ".mat")
        trg_ann_path = ann_root / category / trg_name.replace(".jpg", ".mat")
        if not src_ann_path.exists():
            raise FileNotFoundError(f"PF-PASCAL source annotation not found: {src_ann_path}")
        if not trg_ann_path.exists():
            raise FileNotFoundError(f"PF-PASCAL target annotation not found: {trg_ann_path}")

        src_ann = sio.loadmat(src_ann_path)
        trg_ann = sio.loadmat(trg_ann_path)
        src_pts = src_ann["kps"]
        trg_pts = trg_ann["kps"]
        if src_pts.shape != trg_pts.shape:
            raise ValueError(
                f"PF-PASCAL pair {local_idx} has mismatched keypoint arrays: "
                f"{src_pts.shape} vs {trg_pts.shape}"
            )

        src_keep = []
        trg_keep = []
        for src_xy, trg_xy in zip(src_pts, trg_pts):
            if not np.isfinite(src_xy).all() or not np.isfinite(trg_xy).all():
                continue
            src_keep.append([float(src_xy[0]), float(src_xy[1])])
            trg_keep.append([float(trg_xy[0]), float(trg_xy[1])])

        yield {
            "source_dataset": "pfpascal",
            "source_split": split,
            "source_sample_id": int(local_idx),
            "source_pair_id": int(local_idx),
            "filename": f"pfpascal_{split}_{local_idx:06d}",
            "category": category,
            "category_id": int(cls_id),
            "src_img_path": str(image_root / src_name),
            "trg_img_path": str(image_root / trg_name),
            "src_img_rel_path": src_name,
            "trg_img_rel_path": trg_name,
            "source_root": str(image_root),
            "src_bbox": [float(x) for x in np.asarray(src_ann["bbox"]).reshape(-1).tolist()],
            "trg_bbox": [float(x) for x in np.asarray(trg_ann["bbox"]).reshape(-1).tolist()],
            "src_kps": src_keep,
            "trg_kps": trg_keep,
            "n_pts": int(len(src_keep)),
            "valid_points": int(len(src_keep)),
        }


def main() -> None:
    args = parse_args()

    rows: List[Dict] = []
    counts: Dict[str, int] = {}

    if args.pointodyssey_manifest is not None:
        for row in _iter_pointodyssey_rows(args.pointodyssey_manifest, args.max_pointodyssey_rows):
            rows.append(row)
            counts["pointodyssey"] = counts.get("pointodyssey", 0) + 1

    if args.include_spair:
        for row in _iter_spair_rows(args.spair_root, args.spair_split, args.max_spair_rows):
            rows.append(row)
            counts["spair"] = counts.get("spair", 0) + 1

    if args.include_pfpascal:
        for row in _iter_pfpascal_rows(args.pfpascal_datapath, args.pfpascal_split, args.max_pfpascal_rows):
            rows.append(row)
            counts["pfpascal"] = counts.get("pfpascal", 0) + 1

    if not rows:
        raise RuntimeError("No rows selected. Provide at least one source dataset.")

    # Reassign pair_id globally so the pooled manifest has stable unique ids.
    for global_idx, row in enumerate(rows):
        row["pair_id"] = int(global_idx)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")

    summary = {
        "output": str(args.output),
        "num_rows": len(rows),
        "counts": counts,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
