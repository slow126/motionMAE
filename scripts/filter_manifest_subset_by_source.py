#!/usr/bin/env python3
"""Filter an existing manifest subset file by source dataset/split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter a manifest subset by source dataset/split")
    p.add_argument("--manifest-path", type=Path, required=True, help="Manifest JSONL path.")
    p.add_argument("--subset-path", type=Path, required=True, help="Input subset JSON path.")
    p.add_argument("--source-dataset", type=str, required=True, help="Source dataset to keep.")
    p.add_argument("--source-split", type=str, default=None, help="Optional source split to keep.")
    p.add_argument("--output", type=Path, required=True, help="Output subset JSON path.")
    return p.parse_args()


def coerce_indices(raw: object) -> list[int]:
    if isinstance(raw, dict):
        if "indices" in raw:
            raw = raw["indices"]
        elif "subset" in raw:
            raw = raw["subset"]
        else:
            raise TypeError(f"Unsupported subset dict format: {list(raw.keys())}")
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        raise TypeError(f"Unsupported subset payload type: {type(raw)!r}")
    return [int(x) for x in raw]


def main() -> None:
    args = parse_args()
    if not args.manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest_path}")
    if not args.subset_path.exists():
        raise FileNotFoundError(f"Subset not found: {args.subset_path}")

    want_dataset = str(args.source_dataset).strip().lower()
    want_split = None if args.source_split is None else str(args.source_split).strip().lower()

    with args.subset_path.open("r") as f:
        subset_payload = json.load(f)
    subset_indices = sorted(set(coerce_indices(subset_payload)))
    wanted = set(subset_indices)

    kept: list[int] = []
    remaining = len(wanted)
    with args.manifest_path.open("r") as f:
        for idx, line in enumerate(f):
            if idx not in wanted:
                continue
            remaining -= 1
            row = json.loads(line)
            source_dataset = str(row.get("source_dataset", "pointodyssey")).strip().lower()
            source_split = str(row.get("source_split", "")).strip().lower()
            if source_dataset != want_dataset:
                if remaining == 0:
                    break
                continue
            if want_split is not None and source_split != want_split:
                if remaining == 0:
                    break
                continue
            kept.append(int(idx))
            if remaining == 0:
                break

    payload = {
        "manifest_path": str(args.manifest_path),
        "input_subset_path": str(args.subset_path),
        "source_dataset": want_dataset,
        "source_split": want_split,
        "indices": kept,
        "count": len(kept),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(args.output), "count": len(kept)}, indent=2))


if __name__ == "__main__":
    main()
