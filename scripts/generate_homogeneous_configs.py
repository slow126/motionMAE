#!/usr/bin/env python3
"""Generate training YAML configs for the homogeneous source ablation.

Reads the summary JSON produced by build_homogeneous_source_ablation_subsets.py
and stamps out one config per (source, fraction, method) using the pooled
clustercov config as a template.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate training configs for homogeneous source ablation")
    p.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to homogeneous_ablation_summary.json from build_homogeneous_source_ablation_subsets.py.",
    )
    p.add_argument(
        "--template",
        type=Path,
        required=True,
        help="Base YAML config to use as template.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write generated YAML configs.",
    )
    p.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Optional comma-separated source filter, e.g. 'spair,pfpascal'.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (default: use template).",
    )
    p.add_argument(
        "--lr-backbone",
        type=float,
        default=None,
        help="Override backbone learning rate (default: use template).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs (default: use template).",
    )
    p.add_argument(
        "--step-milestones",
        type=str,
        default=None,
        help="Override step scheduler milestones, e.g. '[350, 400, 450]' or '350,400,450'.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Set max_steps to stop training after N optimizer steps (default: use template).",
    )
    p.add_argument(
        "--validation-step-interval",
        type=int,
        default=None,
        help="Override validation_step_interval (default: use template).",
    )
    p.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=None,
        help="Override check_val_every_n_epoch (default: use template).",
    )
    p.add_argument(
        "--checkpoint-every-n-epochs",
        type=int,
        default=None,
        help="Override regular epoch checkpoint interval (default: use template).",
    )
    p.add_argument(
        "--disable-epoch-checkpoints",
        action="store_true",
        default=False,
        help="Disable growing epoch_N.pth checkpoints while keeping fixed best-model checkpoints.",
    )
    p.add_argument(
        "--rc",
        action="store_true",
        default=False,
        help="Rewrite data paths for the RC cluster (/home/slow1/Data/...).",
    )
    return p.parse_args()


RC_PATH_REWRITES = {
    "/home/spencer/Data/PointOdyssey": "/home/slow1/Data/PointOdyssey",
    "/home/spencer/Data/correspondence/TSS_CVPR2016": "/home/slow1/Data/correspondence/TSS_CVPR2016",
    "/home/spencer/Data/correspondence/kitti": "/home/slow1/Data/correspondence/kitti",
    "/home/spencer/Data/FlyingThings3D_tiny": "/home/slow1/Data/FlyingThings3D_Pytorch",
    "./models/Datasets_CATs": "/home/slow1/Projects/OnlineSyntheticCorrespondence/models/Datasets_CATs",
}


def _rewrite_paths_rc(cfg: Dict[str, Any]) -> None:
    """Rewrite local data paths to RC cluster paths in-place."""
    for section in cfg.values():
        if not isinstance(section, dict):
            continue
        for key, val in section.items():
            if isinstance(val, str):
                for local, remote in RC_PATH_REWRITES.items():
                    if val == local:
                        section[key] = remote
            elif isinstance(val, dict):
                for subkey, subval in val.items():
                    if isinstance(subval, str):
                        for local, remote in RC_PATH_REWRITES.items():
                            if subval == local:
                                val[subkey] = remote


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Path, comment: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        if comment:
            for line in comment.strip().splitlines():
                f.write(f"# {line}\n")
            f.write("\n")
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    template = load_yaml(args.template)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_filter = None
    if args.sources:
        source_filter = {src.strip() for src in args.sources.split(",") if src.strip()}

    generated = []
    for run in summary["runs"]:
        source = run["source"]
        if source_filter is not None and source not in source_filter:
            continue
        frac_label = run["fraction_label"]

        for method in ("clustercov", "random"):
            subset_path = run[f"{method}_path"]
            config_name = f"homogeneous_{source}_{method}_{frac_label}"

            cfg = json.loads(json.dumps(template))  # deep copy

            # Update dataset section
            cfg["dataset"]["pooled_pairs_subset_mode"] = "heuristic"
            cfg["dataset"]["pooled_pairs_subset_indices_path"] = subset_path

            # Override training params if requested
            if args.lr is not None:
                cfg["training"]["lr"] = args.lr
            if args.lr_backbone is not None:
                cfg["training"]["lr_backbone"] = args.lr_backbone
            if args.epochs is not None:
                cfg["training"]["epochs"] = args.epochs
            if args.step_milestones is not None:
                cfg["training"]["step"] = args.step_milestones
            if args.max_steps is not None:
                cfg["training"]["max_steps"] = args.max_steps
            if args.validation_step_interval is not None:
                cfg["training"]["validation_step_interval"] = args.validation_step_interval
            if args.check_val_every_n_epoch is not None:
                cfg["training"]["check_val_every_n_epoch"] = args.check_val_every_n_epoch
            if args.checkpoint_every_n_epochs is not None:
                cfg["training"]["checkpoint_every_n_epochs"] = args.checkpoint_every_n_epochs
            if args.disable_epoch_checkpoints:
                cfg["training"]["save_epoch_checkpoints"] = False

            if args.rc:
                _rewrite_paths_rc(cfg)

            comment = (
                f"Homogeneous source ablation: {source} only, {method} selection, {frac_label}\n"
                f"Pool size: {run['pool_size']}, Budget: {run['budget']}, "
                f"Selected: {run[f'{method}_count']}\n"
                f"Generated from template: {args.template.name}"
            )

            out_path = args.output_dir / f"{config_name}.yaml"
            save_yaml(cfg, out_path, comment=comment)
            generated.append({"config": str(out_path), "source": source, "method": method, "fraction": frac_label})
            print(f"  {out_path.name}")

    manifest_path = args.output_dir / "generated_configs_manifest.json"
    manifest_path.write_text(json.dumps(generated, indent=2))
    print(f"\n{len(generated)} configs generated -> {args.output_dir}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
