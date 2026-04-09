#!/usr/bin/env python3
"""Show which source datasets each benchmark selected from."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-benchmark", type=Path, required=True, help="The _per_benchmark.json file")
    p.add_argument("--manifest", type=Path, required=True, help="The pooled manifest JSONL")
    args = p.parse_args()

    # Build manifest_idx -> source_dataset lookup
    source_map = {}
    with args.manifest.open() as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                row = json.loads(line)
                source_map[idx] = str(row.get("source_dataset", "unknown"))

    bench_selections = json.loads(args.per_benchmark.read_text())

    for bench_name, idxs in bench_selections.items():
        counts = {}
        for i in idxs:
            src = source_map.get(int(i), "unknown")
            counts[src] = counts.get(src, 0) + 1
        print(f"\n{bench_name} ({len(idxs)} selected):")
        for src in sorted(counts, key=counts.get, reverse=True):
            print(f"  {src}: {counts[src]}")


if __name__ == "__main__":
    main()
