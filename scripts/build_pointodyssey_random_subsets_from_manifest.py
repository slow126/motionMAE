#!/usr/bin/env python3
"""Build deterministic random subset index files from an existing manifest.jsonl."""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Build random subset files from an existing manifest")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest.jsonl")
    parser.add_argument(
        "--fractions",
        type=str,
        default="0.50,0.30,0.10",
        help="Comma-separated fractions in [0,1], e.g. '0.50,0.30,0.10'",
    )
    parser.add_argument("--seed", type=int, default=2021, help="Base seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: manifest parent directory)",
    )
    return parser.parse_args()


def parse_fractions(raw: str) -> List[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        frac = float(part)
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"Invalid fraction {frac}; expected in (0,1]")
        values.append(frac)
    if not values:
        raise ValueError("No valid fractions provided")
    return values


def count_manifest_rows(manifest_path: Path) -> int:
    n = 0
    with manifest_path.open("r") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def build_random_indices(total: int, fraction: float, seed: int) -> List[int]:
    k = int(round(total * fraction))
    k = max(0, min(k, total))
    if k == 0:
        return []
    if k == total:
        return list(range(total))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.arange(total), size=k, replace=False).tolist()
    chosen = [int(x) for x in chosen]
    chosen.sort()
    return chosen


def main():
    args = parse_args()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fractions = parse_fractions(args.fractions)
    total = count_manifest_rows(manifest_path)
    if total <= 0:
        raise RuntimeError(f"Manifest appears empty: {manifest_path}")

    print(f"Manifest rows: {total}", flush=True)
    for frac in fractions:
        seed = int(args.seed)
        pct = int(round(frac * 100))
        indices = build_random_indices(total=total, fraction=frac, seed=seed)
        out_path = output_dir / f"subset_random_{pct}_seed{args.seed}.json"
        with out_path.open("w") as f:
            json.dump(indices, f, indent=2)
        print(
            f"  wrote {out_path} ({len(indices)}/{total}, fraction={frac:.2f}, rng_seed={seed})",
            flush=True,
        )


if __name__ == "__main__":
    main()
