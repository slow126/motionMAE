import argparse
import json
from pathlib import Path
from typing import Union

from PIL import Image
import numpy as np
import torch
from torch.nn.functional import interpolate
from torchvision.transforms.functional import normalize

from models.RAFT.core.utils.flow_viz import flow_to_image
from src.io import read_flo_file


class TSSDataset(object):
    def __init__(
        self,
        root,
        size: Union[tuple, int] = 256,
        normalize: Union[bool, str, tuple, list] = 'imagenet',
    ):
        self.root = Path(root)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

        self.labels = {}

        self.pairs = []
        idx = 0
        for sub in sorted(self.root.iterdir()):
            if not sub.is_dir(): continue
            self.labels[sub.name] = idx
            idx += 1
            self.pairs.extend(sorted(sub.iterdir()))

        self.flipped = []
        for p in self.pairs:
            self.flipped.append(int(p.joinpath('flip_gt.txt').open().read()))

        if normalize == 'imagenet':
            from src import imagenet_stats
            normalize = imagenet_stats
        elif normalize == True:
            normalize = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.normalize = normalize

    def __len__(self):
        return len(self.pairs)

    def _read_image(self, path, name):
        img = Image.open(path.joinpath(name)).convert('RGB')
        if self.size is not None:
            img = img.resize(self.size, Image.Resampling.BILINEAR)
        img = torch.from_numpy(np.array(img, dtype=np.float32)).div_(255).moveaxis(-1, 0)
        if self.normalize:
            img = normalize(img, *self.normalize)
        return img

    def _read_flow(self, path, name):
        flow = read_flo_file(path.joinpath(name))
        h, w = flow.shape[:2]

        flow = torch.from_numpy(flow).moveaxis(-1, 0)
        if self.size is not None:
            flow = interpolate(flow.unsqueeze(0), self.size, mode='nearest-exact').squeeze(0)
            flow[flow > 1e9] = torch.inf
            flow[0] *= (self.size[1] / w)
            flow[1] *= (self.size[0] / h)
        else:
            flow[flow > 1e9] = torch.inf

        return flow
    
    def __getitem__(self, idx):
        pair_dir = self.pairs[idx]

        label = self.labels[pair_dir.parent.name]

        img1 = self._read_image(pair_dir, 'image1.png')
        img2 = self._read_image(pair_dir, 'image2.png')
        flow1 = self._read_flow(pair_dir, 'flow1.flo')
        flow2 = self._read_flow(pair_dir, 'flow2.flo')
        flipped = self.flipped[idx]

        return {
            'src_img': img1,
            'trg_img': img2,
            'src_flow': flow1,
            'trg_flow': flow2,
            'flipped': flipped,
            'label': label,
        }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Save a few TSS samples and labels without creating a separate script."
    )
    parser.add_argument(
        "--root",
        default="/home/spencer/Data/correspondence/TSS_CVPR2016",
        help="Path to the TSS dataset root.",
    )
    parser.add_argument(
        "--out-dir",
        default="tmp/tss_samples",
        help="Directory where dumped samples and manifest will be written.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of samples to save.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Dataset index to start from.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Resize images/flow to this square size before saving.",
    )
    parser.add_argument(
        "--normalize",
        choices=("imagenet", "true", "false"),
        default="imagenet",
        help="Normalization mode for saved tensors.",
    )
    parser.add_argument(
        "--save-format",
        choices=("pt", "png", "npz", "both", "all"),
        default="pt",
        help="Whether to save torch tensors, PNGs, NPZ files, or combinations.",
    )
    parser.add_argument(
        "--npz-contents",
        choices=("full", "images_only"),
        default="full",
        help="When saving NPZ files, include either the full sample or only src/trg images.",
    )
    return parser.parse_args()


def _normalize_arg(value: str):
    if value == "true":
        return True
    if value == "false":
        return False
    return value


def _to_display_image(img: torch.Tensor, normalize_mode) -> np.ndarray:
    img = img.detach().cpu().clone()
    if normalize_mode == "imagenet":
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=img.dtype).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=img.dtype).view(3, 1, 1)
        img = img * std + mean
    elif normalize_mode is True:
        img = img * 0.5 + 0.5
    img = img.clamp(0.0, 1.0)
    return (img.mul(255).byte().permute(1, 2, 0).numpy())


def _flow_to_png(flow: torch.Tensor) -> np.ndarray:
    flow_hw2 = flow.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    valid_mask = np.isfinite(flow_hw2).all(axis=2)
    safe_flow = flow_hw2.copy()
    safe_flow[~valid_mask] = 0.0
    flow_png = flow_to_image(safe_flow)
    flow_png[~valid_mask] = 0
    return flow_png


def _save_png_sample(sample_dir: Path, sample: dict, normalize_mode) -> dict:
    sample_dir.mkdir(parents=True, exist_ok=True)

    src_img_path = sample_dir / "src_img.png"
    trg_img_path = sample_dir / "trg_img.png"
    src_flow_path = sample_dir / "src_flow.png"
    trg_flow_path = sample_dir / "trg_flow.png"
    label_path = sample_dir / "label.json"

    Image.fromarray(_to_display_image(sample["src_img"], normalize_mode)).save(src_img_path)
    Image.fromarray(_to_display_image(sample["trg_img"], normalize_mode)).save(trg_img_path)
    Image.fromarray(_flow_to_png(sample["src_flow"])).save(src_flow_path)
    Image.fromarray(_flow_to_png(sample["trg_flow"])).save(trg_flow_path)

    return {
        "src_img_png": str(src_img_path),
        "trg_img_png": str(trg_img_path),
        "src_flow_png": str(src_flow_path),
        "trg_flow_png": str(trg_flow_path),
        "label_json": str(label_path),
    }


def _save_npz_sample(sample_dir: Path, sample: dict, npz_contents: str) -> dict:
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_path = sample_dir / "sample.npz"
    arrays = {
        "src_img": sample["src_img"].detach().cpu().numpy().astype(np.float32),
        "trg_img": sample["trg_img"].detach().cpu().numpy().astype(np.float32),
    }
    if npz_contents == "full":
        arrays["src_flow"] = sample["src_flow"].detach().cpu().numpy().astype(np.float32)
        arrays["trg_flow"] = sample["trg_flow"].detach().cpu().numpy().astype(np.float32)
        arrays["label"] = np.int64(sample["label"])
        arrays["flipped"] = np.int64(sample["flipped"])
    np.savez_compressed(sample_path, **arrays)

    return {
        "sample_npz_path": str(sample_path),
        "sample_npz_contents": npz_contents,
    }


def _dump_samples(args) -> None:
    normalize_mode = _normalize_arg(args.normalize)
    dataset = TSSDataset(
        root=args.root,
        size=args.size,
        normalize=normalize_mode,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    end_index = min(args.start_index + args.num_samples, len(dataset))
    label_names = {idx: name for name, idx in dataset.labels.items()}

    for idx in range(args.start_index, end_index):
        sample = dataset[idx]
        pair_dir = dataset.pairs[idx]
        sample_dir = out_dir / f"sample_{idx:05d}"
        item = {
            "dataset_index": idx,
            "pair_dir": str(pair_dir),
            "label_id": int(sample["label"]),
            "label_name": label_names[int(sample["label"])],
            "flipped": int(sample["flipped"]),
            "src_img_shape": list(sample["src_img"].shape),
            "trg_img_shape": list(sample["trg_img"].shape),
            "src_flow_shape": list(sample["src_flow"].shape),
            "trg_flow_shape": list(sample["trg_flow"].shape),
        }

        if args.save_format in ("pt", "both", "all"):
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_path = sample_dir / "sample.pt"
            torch.save(sample, sample_path)
            item["sample_path"] = str(sample_path)

        if args.save_format in ("npz", "all"):
            item.update(_save_npz_sample(sample_dir, sample, args.npz_contents))

        if args.save_format in ("png", "both", "all"):
            label_path = sample_dir / "label.json"
            label_payload = {
                "dataset_index": idx,
                "pair_dir": str(pair_dir),
                "label_id": int(sample["label"]),
                "label_name": label_names[int(sample["label"])],
                "flipped": int(sample["flipped"]),
            }
            sample_dir.mkdir(parents=True, exist_ok=True)
            label_path.write_text(json.dumps(label_payload, indent=2))
            item.update(_save_png_sample(sample_dir, sample, normalize_mode))

        items.append(item)

    manifest = {
        "root": str(Path(args.root).resolve()),
        "out_dir": str(out_dir.resolve()),
        "num_saved": len(items),
        "labels": dataset.labels,
        "samples": items,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"saved {len(items)} sample(s) to {out_dir}")
    print(f"manifest: {manifest_path}")
    for item in items:
        sample_dir = out_dir / f"sample_{item['dataset_index']:05d}"
        print(
            f"idx={item['dataset_index']} label={item['label_name']} "
            f"dir={sample_dir}"
        )


if __name__ == "__main__":
    _dump_samples(_parse_args())
