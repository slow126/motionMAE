#!/usr/bin/env python3
"""Build source ablation subsets from a pooled manifest.

Creates:
- PointOdyssey-only random subset at a requested fraction of PointOdyssey rows
- SPAIR-only random subset matching the PointOdyssey-only selected count when possible
- PF-PASCAL-only subset using all PF-PASCAL rows (optionally filtered by split)
- Mixed balanced random subset using all PF-PASCAL rows, then roughly equal
  SPAIR and PointOdyssey counts up to the target total budget
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pooled single-source ablation subsets")
    p.add_argument("--manifest-path", type=Path, required=True, help="Pooled manifest JSONL path.")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write subset JSONs and summary JSON.",
    )
    p.add_argument(
        "--pointodyssey-fraction",
        type=float,
        default=0.005,
        help="Fraction of PointOdyssey rows to sample randomly (default: 0.005 = 0.5%%).",
    )
    p.add_argument(
        "--spair-count",
        type=int,
        default=0,
        help="Optional explicit SPAIR sample count. Default 0 means match the PointOdyssey-only selected count.",
    )
    p.add_argument(
        "--pfpascal-split",
        type=str,
        default="trn",
        help="PF-PASCAL split filter for the all-inclusive subset (default: trn).",
    )
    p.add_argument("--seed", type=int, default=2021, help="Base RNG seed.")
    return p.parse_args()


def fraction_label(fraction: float) -> str:
    pct = float(fraction) * 100.0
    rounded = round(pct)
    if abs(pct - rounded) < 1e-9:
        return str(int(rounded))
    return f"{pct:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def choose_random(sorted_indices: List[int], k: int, seed: int) -> List[int]:
    total = len(sorted_indices)
    k = max(0, min(int(k), total))
    if k == 0:
        return []
    if k == total:
        return list(sorted_indices)
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(np.asarray(sorted_indices, dtype=np.int64), size=k, replace=False)
    out = [int(x) for x in chosen.tolist()]
    out.sort()
    return out


def main() -> None:
    args = parse_args()
    if not args.manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest_path}")
    if not (0.0 < float(args.pointodyssey_fraction) <= 1.0):
        raise ValueError("--pointodyssey-fraction must be in (0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_source: Dict[str, List[int]] = defaultdict(list)
    by_source_split: Dict[str, Counter] = defaultdict(Counter)
    total = 0

    with args.manifest_path.open("r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source = str(row.get("source_dataset", "pointodyssey")).strip().lower()
            split = str(row.get("source_split", "")).strip().lower()
            by_source[source].append(int(idx))
            by_source_split[source][split] += 1
            total += 1

    pointodyssey_indices = by_source.get("pointodyssey", [])
    spair_indices = by_source.get("spair", [])
    pfpascal_all_indices = by_source.get("pfpascal", [])
    pfpascal_split = str(args.pfpascal_split).strip().lower()

    pfpascal_indices: List[int] = []
    if pfpascal_split:
        with args.manifest_path.open("r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                source = str(row.get("source_dataset", "pointodyssey")).strip().lower()
                split = str(row.get("source_split", "")).strip().lower()
                if source == "pfpascal" and split == pfpascal_split:
                    pfpascal_indices.append(int(idx))
    else:
        pfpascal_indices = list(pfpascal_all_indices)

    pointodyssey_k = int(round(len(pointodyssey_indices) * float(args.pointodyssey_fraction)))
    pointodyssey_k = max(0, min(pointodyssey_k, len(pointodyssey_indices)))
    pointodyssey_subset = choose_random(pointodyssey_indices, pointodyssey_k, seed=int(args.seed))

    spair_target = int(args.spair_count) if int(args.spair_count) > 0 else int(len(pointodyssey_subset))
    spair_subset = choose_random(spair_indices, spair_target, seed=int(args.seed) + 1)

    pfpascal_subset = sorted(int(x) for x in pfpascal_indices)

    mixed_total_budget = int(len(pointodyssey_subset))
    pfpascal_keep = list(pfpascal_subset)
    remaining_budget = max(0, mixed_total_budget - len(pfpascal_keep))
    spair_target_balanced = remaining_budget // 2
    pointodyssey_target_balanced = remaining_budget - spair_target_balanced

    spair_balanced_subset = choose_random(spair_indices, spair_target_balanced, seed=int(args.seed) + 11)
    pointodyssey_balanced_subset = choose_random(
        pointodyssey_indices,
        pointodyssey_target_balanced,
        seed=int(args.seed) + 12,
    )
    mixed_balanced_subset = sorted(
        int(x) for x in (pfpascal_keep + spair_balanced_subset + pointodyssey_balanced_subset)
    )

    pointodyssey_label = fraction_label(float(args.pointodyssey_fraction))
    pointodyssey_out = args.output_dir / f"subset_pointodyssey_random_{pointodyssey_label}_seed{args.seed}.json"
    spair_out = args.output_dir / f"subset_spair_random_match_pointodyssey_seed{args.seed}.json"
    pfpascal_out = args.output_dir / f"subset_pfpascal_all_{pfpascal_split or 'all'}.json"
    mixed_balanced_out = args.output_dir / f"subset_mixed_balanced_match_pointodyssey_seed{args.seed}.json"
    summary_out = args.output_dir / "subset_source_ablation_summary.json"

    pointodyssey_out.write_text(json.dumps(pointodyssey_subset, indent=2))
    spair_out.write_text(json.dumps(spair_subset, indent=2))
    pfpascal_out.write_text(json.dumps(pfpascal_subset, indent=2))
    mixed_balanced_out.write_text(json.dumps(mixed_balanced_subset, indent=2))

    summary = {
        "manifest_path": str(args.manifest_path),
        "manifest_total": int(total),
        "source_counts": {k: int(len(v)) for k, v in sorted(by_source.items())},
        "source_split_counts": {k: dict(v) for k, v in sorted(by_source_split.items())},
        "outputs": {
            "pointodyssey_random": {
                "path": str(pointodyssey_out),
                "count": int(len(pointodyssey_subset)),
                "fraction_of_pointodyssey": float(args.pointodyssey_fraction),
            },
            "spair_random": {
                "path": str(spair_out),
                "count": int(len(spair_subset)),
                "requested_count": int(spair_target),
                "matched_pointodyssey_count": int(len(pointodyssey_subset)),
            },
            "pfpascal_all": {
                "path": str(pfpascal_out),
                "count": int(len(pfpascal_subset)),
                "split": pfpascal_split,
            },
            "mixed_balanced": {
                "path": str(mixed_balanced_out),
                "count": int(len(mixed_balanced_subset)),
                "target_total_budget": int(mixed_total_budget),
                "remaining_budget_after_pfpascal": int(remaining_budget),
                "pointodyssey_count": int(len(pointodyssey_balanced_subset)),
                "spair_count": int(len(spair_balanced_subset)),
                "pfpascal_count": int(len(pfpascal_keep)),
            },
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
