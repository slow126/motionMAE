#!/usr/bin/env python3
"""Append a step-0 validation row to a snapshot CSV using the first logged block."""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append initial validation rows (step=0) to validation_results.csv "
            "using values printed in a training log."
        )
    )
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Snapshot directory that contains validation_results.csv.",
    )
    parser.add_argument(
        "--log",
        required=True,
        help="Training log that contains printed validation blocks.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help=(
            "Replace existing step=0/validation_scope=initial rows for matching benchmarks "
            "instead of skipping."
        ),
    )
    parser.add_argument(
        "--scope",
        default="initial",
        choices=["initial", "step"],
        help="Scope to write into the CSV (default: initial).",
    )
    return parser.parse_args()


SECTION_STEP_RE = re.compile(r"Validation Results \(step=(\d+)\):")
SECTION_INITIAL_RE = re.compile(r"Initial validation results:")
BENCH_RE = re.compile(r"^\s{2}([a-zA-Z0-9_\-]+): PCK=([0-9.+-eE]+)%, Loss=([0-9.+-eE]+)")


def parse_first_validation_block(log_text: str) -> Dict[str, Dict[str, float]]:
    sections: List[Dict[str, Dict[str, float]]] = []

    current_scope = None
    current_step = None
    current_rows: Dict[str, Dict[str, float]] = {}

    def flush_section():
        nonlocal current_rows, current_scope, current_step
        if current_scope is not None and current_rows:
            sections.append({
                "scope": current_scope,
                "step": current_step,
                "rows": current_rows,
            })

    for line in log_text.splitlines():
        line = line.rstrip()
        if SECTION_INITIAL_RE.search(line):
            flush_section()
            current_scope = "initial"
            current_step = 0
            current_rows = {}
            continue

        match_step = SECTION_STEP_RE.search(line)
        if match_step:
            flush_section()
            current_scope = "step"
            current_step = int(match_step.group(1))
            current_rows = {}
            continue

        match_bench = BENCH_RE.match(line)
        if match_bench and current_scope is not None:
            bench = match_bench.group(1)
            pck = float(match_bench.group(2))
            loss = float(match_bench.group(3))
            current_rows[bench] = {"pck": pck, "loss": loss}

    flush_section()

    # Prefer initial if present, else first available step section.
    for section in sections:
        if section["scope"] == "initial":
            return section["rows"]

    if not sections:
        return {}
    return sections[0]["rows"]


def load_rows(csv_path: pathlib.Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def append_step_zero_rows(
    csv_path: pathlib.Path,
    bench_rows: Dict[str, Dict[str, float]],
    scope: str,
    overwrite: bool,
) -> int:
    rows = load_rows(csv_path)
    header = [
        "epoch",
        "training_steps",
        "validation_scope",
        "validation_target",
        "benchmark",
        "pck",
        "loss",
        "pck_motion_aware",
        "pck_motion_small",
        "pck_motion_medium",
        "pck_motion_large",
        "zero_flow_precision",
        "zero_flow_recall",
        "zero_flow_f1",
        "static_bias_ratio",
        "mmd2_pred_corr_vs_pred_miss",
        "mmd2_pred_corr_vs_gt",
        "mmd2_pred_miss_vs_gt",
    ]

    existing_rows = rows
    added = 0

    output_rows = existing_rows
    for bench, values in sorted(bench_rows.items()):
        exists = any(
            (r.get("benchmark") == bench and r.get("training_steps") in {"0", 0})
            for r in existing_rows
        )
        if exists and not overwrite:
            print(f"Skipping {bench}: existing step-0 entry found in {csv_path}")
            continue

        if exists and overwrite:
            output_rows = [
                r for r in output_rows
                if not (r.get("benchmark") == bench and r.get("training_steps") == "0")
            ]

        new_row = {
            "epoch": "0",
            "training_steps": "0",
            "validation_scope": scope,
            "validation_target": "0",
            "benchmark": bench,
            "pck": f"{values['pck']:.4f}",
            "loss": f"{values['loss']:.6f}",
            "pck_motion_aware": "",
            "pck_motion_small": "",
            "pck_motion_medium": "",
            "pck_motion_large": "",
            "zero_flow_precision": "",
            "zero_flow_recall": "",
            "zero_flow_f1": "",
            "static_bias_ratio": "",
            "mmd2_pred_corr_vs_pred_miss": "",
            "mmd2_pred_corr_vs_gt": "",
            "mmd2_pred_miss_vs_gt": "",
        }

        output_rows.append(new_row)
        added += 1

    if added == 0:
        return 0

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({k: row.get(k, "") for k in header})

    return added


def main() -> None:
    args = parse_args()
    snapshot = pathlib.Path(args.snapshot).expanduser().resolve()
    log_path = pathlib.Path(args.log).expanduser().resolve()

    if not log_path.exists():
        raise FileNotFoundError(f"Missing log file: {log_path}")

    csv_path = snapshot / "validation_results.csv"
    if not csv_path.parent.is_dir():
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot}")

    log_text = log_path.read_text(errors="replace")
    bench_rows = parse_first_validation_block(log_text)
    if not bench_rows:
        raise ValueError(
            f"No validation block found in log file: {log_path}. "
            f"Make sure the log includes 'Initial validation results:' or 'Validation Results (step=...)'."
        )

    added = append_step_zero_rows(
        csv_path=csv_path,
        bench_rows=bench_rows,
        scope=args.scope,
        overwrite=args.overwrite_existing,
    )

    if args.overwrite_existing:
        print(f"Appended and/or replaced {added} row(s) in {csv_path} with scope={args.scope}, step=0")
    else:
        print(f"Appended {added} row(s) in {csv_path} with scope={args.scope}, step=0")
    if added == 0:
        print("No new rows added (already present).")


if __name__ == "__main__":
    main()
