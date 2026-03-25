#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.flow_mae.dino_context_dataset import (
    FlyingThingsDINOFlowMAEConfig,
    FlyingThingsDINOFlowMAEDataModule,
)
from src.flow_mae.dino_context_lightning import FlowMAEDINOContextLightningModule
from src.flow_mae.dino_context_model import FlowMAEDINOContextModelConfig


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def make_run_dir(config: dict[str, Any], config_path: str) -> Path:
    snapshots_root = Path(config["paths"]["snapshots"])
    snapshots_root.mkdir(parents=True, exist_ok=True)

    name = config["experiment"].get("name")
    if not name:
        name = Path(config_path).stem
    timestamp = time.strftime("%Y_%m_%d_%H_%M")
    run_dir = snapshots_root / f"{name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DINO-context masked flow MAE on FlyingThings with Lightning")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    training_config = config["training"]
    model_config = dict(config["model"])
    data_config = config["data"]

    seed = int(training_config.get("seed", 2021))
    set_seed(seed)

    if torch.cuda.is_available():
        allow_tf32 = bool(training_config.get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = bool(training_config.get("cudnn_benchmark", True))
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(training_config.get("float32_matmul_precision", "high")))

    datamodule = FlyingThingsDINOFlowMAEDataModule(FlyingThingsDINOFlowMAEConfig(**data_config))
    datamodule.setup("fit")
    if datamodule.train_dataset is None:
        raise RuntimeError("Failed to initialize training dataset.")

    dino_grid_hw = datamodule.train_dataset.dino_grid_hw
    dino_feature_dim = datamodule.train_dataset.dino_feature_dim
    model_grid = int(model_config["image_size"]) // int(model_config["patch_size"])
    if tuple(dino_grid_hw) != (model_grid, model_grid):
        raise ValueError(
            f"Model patch grid {(model_grid, model_grid)} does not match DINO feature grid {dino_grid_hw}. "
            "Use matching image_size / patch_size and DINO precompute settings."
        )
    configured_context_dim = int(model_config.get("context_feature_dim", dino_feature_dim))
    if configured_context_dim != dino_feature_dim:
        raise ValueError(
            f"Configured context_feature_dim={configured_context_dim} does not match dataset DINO dim={dino_feature_dim}."
        )
    model_config["context_feature_dim"] = dino_feature_dim

    run_dir = make_run_dir(config, args.config)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(f"[train_flow_mae_dino_context] run_dir={run_dir}")
    print(
        "[train_flow_mae_dino_context] "
        f"precision={training_config.get('precision')} "
        f"lr={training_config.get('lr')} "
        f"batch_size={data_config.get('batch_size')} "
        f"observation_mask_mode={model_config.get('observation_mask_mode', 'scheduled')} "
        f"dino_dim={dino_feature_dim} "
        f"dino_grid={dino_grid_hw}"
    )

    lightning_module = FlowMAEDINOContextLightningModule(
        model_config=FlowMAEDINOContextModelConfig(**model_config),
        training_config=training_config,
    )

    logger = TensorBoardLogger(save_dir=str(run_dir), name="tensorboard", version="")
    callbacks = [LearningRateMonitor(logging_interval="epoch")]
    if bool(training_config.get("save_checkpoints", True)):
        checkpoint_dir = run_dir / "checkpoints"
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="epoch{epoch:02d}-val_loss{val_loss:.4f}",
                monitor=str(training_config.get("checkpoint_monitor", "val_loss")),
                mode=str(training_config.get("checkpoint_mode", "min")),
                save_top_k=int(training_config.get("checkpoint_save_top_k", 2)),
                save_last=bool(training_config.get("checkpoint_save_last", True)),
                every_n_epochs=int(training_config.get("checkpoint_every_n_epochs", 1)),
                auto_insert_metric_name=False,
            ),
        )

    requested_devices = training_config.get("devices", 1)
    if requested_devices == "auto":
        requested_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    devices = max(1, min(int(requested_devices), torch.cuda.device_count() if torch.cuda.is_available() else 1))
    if not torch.cuda.is_available():
        devices = 1

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        max_epochs=int(training_config.get("max_epochs", 1)),
        precision=training_config.get("precision", "16-mixed" if torch.cuda.is_available() else "32-true"),
        gradient_clip_val=float(training_config.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(training_config.get("log_every_n_steps", 25)),
        num_sanity_val_steps=int(training_config.get("num_sanity_val_steps", 2)),
        overfit_batches=training_config.get("overfit_batches", 0.0),
        limit_train_batches=training_config.get("limit_train_batches", 1.0),
        limit_val_batches=training_config.get("limit_val_batches", 1.0),
        check_val_every_n_epoch=int(training_config.get("check_val_every_n_epoch", 1)),
        benchmark=bool(training_config.get("cudnn_benchmark", True)) if torch.cuda.is_available() else False,
    )

    resume_ckpt = training_config.get("resume_ckpt", None)
    if resume_ckpt:
        resume_ckpt = str(Path(resume_ckpt).expanduser().resolve())
        print(f"[train_flow_mae_dino_context] warm-starting model weights from {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        missing, unexpected = lightning_module.model.load_state_dict(state, strict=True)
        if missing:
            print(f"  [warn] missing keys: {missing}")
        if unexpected:
            print(f"  [warn] unexpected keys: {unexpected}")
        print(f"  loaded {len(state)} tensors — optimizer/LR schedule starts fresh")

    with open(run_dir / "launch.txt", "w", encoding="utf-8") as handle:
        handle.write(f"config={Path(args.config).resolve()}\n")
        handle.write(f"cwd={Path(os.getcwd()).resolve()}\n")
        handle.write(f"devices={devices}\n")
        if resume_ckpt:
            handle.write(f"resume_ckpt={resume_ckpt}\n")

    trainer.fit(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
