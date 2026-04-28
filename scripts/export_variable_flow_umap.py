#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO_ROOT / ".cache" / "numba"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from percieverIO_Exp.data import VariableObservationFlowDataConfig, VariableObservationFlowDataModule
from percieverIO_Exp.lightning import VariableFlowLightningModule
from percieverIO_Exp.model import VariableFlowConfig
from scripts.train_variable_flow_perceiver import load_config


OBS_FRACTIONS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]


def pick_checkpoint(snapshot: Path) -> Path:
    if snapshot.is_file():
        return snapshot
    ckpt_dir = snapshot / "checkpoints"
    if not ckpt_dir.exists():
        raise RuntimeError(f"No checkpoints directory under {snapshot}")
    best = ckpt_dir / "best.ckpt"
    if best.exists():
        return best
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    epoch_re = re.compile(r"epoch(\d+)")
    best_epoch = -1
    best_path: Path | None = None
    for ckpt in ckpt_dir.glob("*.ckpt"):
        match = epoch_re.search(ckpt.name)
        if match and int(match.group(1)) > best_epoch:
            best_epoch = int(match.group(1))
            best_path = ckpt
    if best_path is None:
        raise RuntimeError(f"No ckpt files found under {ckpt_dir}")
    return best_path


def load_module(ckpt_path: Path, config: dict[str, Any]) -> VariableFlowLightningModule:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model_config = VariableFlowConfig.from_config_dict(config["model"])
    training_config = dict(config["training"])
    training_config.setdefault("image_size", list(model_config.encoder.image_shape))
    training_config.setdefault("query_stride", int(model_config.decoder.query_stride))
    module = VariableFlowLightningModule(model_config=model_config, training_config=training_config)
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    module.model.load_state_dict(state, strict=True)
    module.eval().requires_grad_(False)
    return module


def fit_projection(latents: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        import umap as umap_lib

        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(latents), "umap"
    except Exception:
        centered = latents - latents.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        return u[:, :2] * s[:2], "pca"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export latent embeddings and UMAP for variable flow experiment.")
    parser.add_argument("--snapshot", required=True, help="Run directory or checkpoint path")
    parser.add_argument("--output", default=None, help="Optional output directory")
    parser.add_argument("--num-samples", type=int, default=32, help="Validation samples to export")
    args = parser.parse_args()

    snapshot = Path(args.snapshot).resolve()
    ckpt_path = pick_checkpoint(snapshot)
    run_dir = snapshot if snapshot.is_dir() else snapshot.parent.parent
    config = load_config(str(run_dir / "config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = load_module(ckpt_path, config).to(device)

    output_dir = Path(args.output).resolve() if args.output else run_dir / "analysis" / "variable_flow_umap"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []

    for obs_fraction in OBS_FRACTIONS:
        data_cfg = deepcopy(config["data"])
        data_cfg["fixed_observed_fraction"] = float(obs_fraction)
        data_cfg["fixed_mask_mode"] = str(config.get("analysis", {}).get("mask_type", data_cfg.get("mask_mode", "random")))
        datamodule = VariableObservationFlowDataModule(VariableObservationFlowDataConfig(**data_cfg))
        datamodule.setup("fit")
        if datamodule.val_dataset is None:
            raise RuntimeError("Validation dataset is not available.")
        val_subset = Subset(datamodule.val_dataset, list(range(min(len(datamodule.val_dataset), int(args.num_samples)))))
        loader = DataLoader(
            val_subset,
            batch_size=int(config["analysis"].get("batch_size", config["data"].get("val_batch_size", 4))),
            shuffle=False,
            collate_fn=datamodule.val_dataloader().collate_fn,  # type: ignore[attr-defined]
        )
        with torch.no_grad():
            for batch in loader:
                batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                    if key not in {"view_a", "view_b"}
                } | {
                    "view_a": {
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch["view_a"].items()
                    },
                    "view_b": None
                    if batch["view_b"] is None
                    else {
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch["view_b"].items()
                    },
                }
                out = module.model(batch["view_a"]["tokens"], batch["query_inputs"], pad_mask=batch["view_a"]["pad_mask"])
                epe = module.endpoint_error(out["pred_flow"], batch["target_flow_q"], batch["target_valid_q"])
                z = out["z_content"].detach().cpu().numpy()
                embeddings.append(z)
                for idx, sample_id in enumerate(batch["sample_id"]):
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "obs_fraction": float(obs_fraction),
                            "mask_type": str(data_cfg["fixed_mask_mode"]),
                            "epe": float(epe.detach().cpu().item()),
                        }
                    )

    if not embeddings:
        raise RuntimeError("No embeddings were generated.")

    z_all = np.concatenate(embeddings, axis=0)
    points_2d, method = fit_projection(z_all)
    np.savez(
        output_dir / "embeddings.npz",
        z_content=z_all,
        projection=points_2d,
        sample_id=np.array([row["sample_id"] for row in rows], dtype=object),
        obs_fraction=np.array([row["obs_fraction"] for row in rows], dtype=np.float32),
        mask_type=np.array([row["mask_type"] for row in rows], dtype=object),
        epe=np.array([row["epe"] for row in rows], dtype=np.float32),
    )
    metrics = {
        "num_points": int(z_all.shape[0]),
        "embedding_dim": int(z_all.shape[1]),
        "projection_method": method,
        "mean_epe": float(np.mean([row["epe"] for row in rows])),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    fractions = np.array([row["obs_fraction"] for row in rows], dtype=np.float32)
    scatter = ax.scatter(points_2d[:, 0], points_2d[:, 1], c=fractions, cmap="viridis", s=12, alpha=0.9)
    ax.set_title("Variable Flow Latent Projection")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    fig.colorbar(scatter, ax=ax, label="Observation Fraction")
    fig.tight_layout()
    fig.savefig(output_dir / "umap.png")
    plt.close(fig)

    with (output_dir / "export_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"checkpoint": str(ckpt_path), "output_dir": str(output_dir)}, handle, indent=2)


if __name__ == "__main__":
    main()
