#!/usr/bin/env python3
"""Merge scored CSVs by manifest_idx into a new combined CSV.

Later CSVs win on duplicate manifest_idx so you can append or overwrite
rows from a smaller delta score pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge score CSVs on manifest_idx")
    p.add_argument("--inputs", nargs="+", type=Path, required=True, help="Input CSVs in merge order.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sort-by-manifest-idx", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for path in args.inputs:
        if not path.exists():
            raise FileNotFoundError(f"Missing input CSV: {path}")
        frames.append(pd.read_csv(path))
    if not frames:
        raise RuntimeError("No input CSVs provided")

    df = pd.concat(frames, ignore_index=True)
    if "manifest_idx" not in df.columns:
        raise KeyError("Expected manifest_idx column in merged CSVs")
    df = df.drop_duplicates(subset=["manifest_idx"], keep="last")
    if args.sort_by_manifest_idx:
        df = df.sort_values("manifest_idx", ascending=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} merged rows -> {args.output}")


if __name__ == "__main__":
    main()
