#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pointodyssey_pairs.selectors import select_random, select_stratified_bins, select_top


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select a scored PointOdyssey subset from score table.")
    p.add_argument("--scores", type=Path, required=True, help="CSV or parquet table from score_source_samples.py")
    p.add_argument("--output", type=Path, required=True, help="Output subset indices JSON")
    p.add_argument("--budget", type=int, default=None, help="Absolute number of rows to keep")
    p.add_argument("--fraction", type=float, default=None, help="Alternative budget as fraction in (0,1]")
    p.add_argument("--policy", type=str, required=True, choices=["random", "top", "stratified"])
    p.add_argument("--column", type=str, default="p90_mag", help="Metric column for top/stratified")
    p.add_argument("--ascending", action="store_true", help="Sort ascending for top policy")
    p.add_argument("--num-bins", type=int, default=4, help="Number of bins for stratified policy")
    p.add_argument("--seed", type=int, default=2021)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if (args.budget is None) == (args.fraction is None):
        raise ValueError("Specify exactly one of --budget or --fraction")

    usecols = None
    if args.policy in {"top", "stratified"}:
        usecols = ["manifest_idx", args.column]
    elif args.policy == "random":
        usecols = ["manifest_idx"]

    if args.scores.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(args.scores, columns=usecols)
    else:
        df = pd.read_csv(args.scores, usecols=usecols)

    budget = args.budget
    if args.fraction is not None:
        frac = float(args.fraction)
        if not (0.0 < frac <= 1.0):
            raise ValueError("--fraction must be in (0,1]")
        budget = int(len(df) * frac)
        budget = max(1, budget)

    assert budget is not None
    budget = int(budget)
    if budget <= 0:
        raise ValueError("--budget must be > 0")

    if args.policy in {"top", "stratified"}:
        if args.column not in df.columns:
            raise KeyError(f"Missing column for selection: {args.column}")
        df[args.column] = pd.to_numeric(df[args.column], errors="coerce")
        # Treat infinities as invalid scores so they are excluded from ranking.
        df.loc[~df[args.column].map(pd.notna), args.column] = float("nan")
        df.loc[df[args.column] == float("inf"), args.column] = float("nan")
        df.loc[df[args.column] == float("-inf"), args.column] = float("nan")

    if args.policy == "random":
        chosen = select_random(df, budget=budget, seed=args.seed)
    elif args.policy == "top":
        chosen = select_top(df, column=args.column, budget=budget, ascending=args.ascending)
    else:
        chosen = select_stratified_bins(
            df,
            column=args.column,
            num_bins=args.num_bins,
            budget=budget,
            seed=args.seed,
        )

    if "manifest_idx" not in chosen.columns:
        raise KeyError("scores table must contain 'manifest_idx' for subset export")

    idx = [int(x) for x in chosen["manifest_idx"].tolist()]
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "indices": idx,
                "metadata": {
                    "scores": str(args.scores),
                    "policy": args.policy,
                    "column": args.column,
                    "ascending": bool(args.ascending),
                    "budget": int(budget),
                    "fraction": (None if args.fraction is None else float(args.fraction)),
                    "seed": int(args.seed),
                },
            },
            indent=2,
        )
    )
    print(f"[select_source_subset] wrote {len(idx)} indices -> {out}")


if __name__ == "__main__":
    main()
