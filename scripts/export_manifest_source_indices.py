#!/usr/bin/env python3
"""Export manifest indices for selected source datasets/splits.

Example:
  python3 scripts/export_manifest_source_indices.py \
    --manifest-path analysis/pooled_candidates_smoke_plus_pfpascal/manifest.jsonl \
    --source-dataset pfpascal \
    --source-split trn \
    --output analysis/pfpascal_trn_manifest_indices.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export manifest indices matching source dataset filters")
    p.add_argument("--manifest-path", type=Path, required=True)
    p.add_argument("--source-dataset", type=str, required=True)
    p.add_argument("--source-split", type=str, default=None)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    want_dataset = str(args.source_dataset).strip().lower()
    want_split = None if args.source_split is None else str(args.source_split).strip().lower()

    indices: list[int] = []
    with args.manifest_path.open("r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_dataset = str(row.get("source_dataset", "pointodyssey")).strip().lower()
            source_split = str(row.get("source_split", "")).strip().lower()
            if source_dataset != want_dataset:
                continue
            if want_split is not None and source_split != want_split:
                continue
            indices.append(int(idx))

    payload = {
        "manifest_path": str(args.manifest_path),
        "source_dataset": want_dataset,
        "source_split": want_split,
        "indices": indices,
        "count": len(indices),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(args.output), "count": len(indices)}, indent=2))


if __name__ == "__main__":
    main()
