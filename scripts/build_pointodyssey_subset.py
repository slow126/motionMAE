#!/usr/bin/env python3
"""Build deterministic PointOdyssey subset index files for training.

Use cases:
 - random subset: create a reproducible random index subset from cached valid indices
 - heuristic subset passthrough: normalize and filter an offline heuristic index list

The output file contains a JSON list under the `indices` key for compatibility with
`pointodyssey_subset_indices_path` in dataset configs.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Optional

from src.data.synth.datasets.PointOdysseyCorrespondence import PointOdysseySimpleDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Build PointOdyssey subset index list")
    parser.add_argument("--pointodyssey_root", required=True, type=str,
                        help="Path to PointOdyssey dataset root")
    parser.add_argument("--dset", type=str, default="train", choices=["train", "val", "test"],
                        help="Split to use")
    parser.add_argument("--S", type=int, default=2, help="Sequence length")
    parser.add_argument("--N", type=int, default=32, help="Track points")
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4], help="Stride list")
    parser.add_argument("--size", type=int, default=512, help="Crop size (square)")
    parser.add_argument("--max-sequences", type=int, default=3,
                        help="If set, keep only first N sampled sequence pool (deterministic)")
    parser.add_argument("--subset-mode", type=str, default="random", choices=["random", "passthrough"],
                        help="How to build subset")
    parser.add_argument("--subset-size", type=int, default=800,
                        help="Number of indices for random mode")
    parser.add_argument("--subset-fraction", type=float, default=None,
                        help="Alternative to --subset-size: fraction of available pool")
    parser.add_argument("--seed", type=int, default=2021, help="RNG seed for random subset")
    parser.add_argument("--source-indices", type=str, default=None,
                        help="Source index file for passthrough mode")
    parser.add_argument("--subset-output", type=str, required=True,
                        help="Output JSON path")
    return parser.parse_args()


def _load_source_indices(path: str) -> List[int]:
    import torch

    suffix = Path(path).suffix.lower()
    if suffix in [".json", ".js", ".jsn"]:
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            if 'indices' in data:
                data = data['indices']
            elif 'valid' in data:
                data = data['valid']
            elif 'subset' in data:
                data = data['subset']
        return [int(x) for x in data]

    if suffix in [".npy", ".npz"]:
        data = np.load(path)
        if isinstance(data, dict):
            if 'indices' in data:
                data = data['indices']
        return [int(x) for x in data.tolist()]

    if suffix in [".pt", ".pth", ".ckpt"]:
        data = torch.load(path, map_location='cpu')
        if isinstance(data, dict) and 'indices' in data:
            data = data['indices']
        return [int(x) for x in data]

    with open(path, 'r') as f:
        return [int(line.strip().split(',')[0]) for line in f if line.strip()]


def main():
    args = parse_args()

    dataset = PointOdysseySimpleDataset(
        dataset_location=args.pointodyssey_root,
        dset=args.dset,
        S=args.S,
        N=args.N,
        strides=args.strides,
        clip_step=2,
        quick=False,
        resize_size=(args.size + 64, args.size + 64),
        crop_size=(args.size, args.size),
        reverse_flow=True,
        max_sequences=args.max_sequences,
    )

    if hasattr(dataset, '_valid_indices_list') and dataset._valid_indices_list:
        candidate_indices = list(dataset._valid_indices_list)
    else:
        candidate_indices = list(range(len(dataset.base_dataset)))

    if not candidate_indices:
        raise RuntimeError("No candidate indices found. Run precompute_pointodyssey_cache first.")

    if args.subset_mode == 'random':
        if args.subset_fraction is not None:
            target = int(len(candidate_indices) * args.subset_fraction)
        else:
            target = args.subset_size
        target = max(0, min(int(target), len(candidate_indices)))

        if target == 0:
            indices = []
        elif target == len(candidate_indices):
            indices = candidate_indices
        else:
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(candidate_indices, size=target, replace=False).tolist()

    else:
        if args.source_indices is None:
            raise ValueError("--source-indices is required for passthrough mode")
        source = _load_source_indices(args.source_indices)
        candidate_set = set(candidate_indices)
        filtered = [int(idx) for idx in source if 0 <= idx < len(dataset.base_dataset)]
        if candidate_set:
            indices = [idx for idx in filtered if idx in candidate_set]
        else:
            indices = filtered

    output = {
        'indices': [int(i) for i in indices],
        'mode': args.subset_mode,
        'source': {
            'pointodyssey_root': args.pointodyssey_root,
            'dset': args.dset,
            'S': args.S,
            'N': args.N,
            'strides': args.strides,
            'max_sequences': args.max_sequences,
        },
        'count': len(indices),
    }

    output_path = Path(args.subset_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(indices)} indices to {output_path}")


if __name__ == '__main__':
    main()
