#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from copy import deepcopy
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

from percieverIO_Exp.data import VariableObservationFlowDataConfig, VariableObservationFlowDataModule
from percieverIO_Exp.lightning import VariableFlowLightningModule
from percieverIO_Exp.model import VariableFlowConfig


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    extends = data.pop("extends", None)
    if extends is None:
        return data
    parent_path = (config_path.parent / extends).resolve()
    return _deep_update(load_config(str(parent_path)), data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def make_run_dir(config: dict[str, Any], config_path: str) -> Path:
    snapshots_root = Path(config["paths"]["snapshots"])
    snapshots_root.mkdir(parents=True, exist_ok=True)
    name = config["experiment"].get("name") or Path(config_path).stem
    timestamp = time.strftime("%Y_%m_%d_%H_%M")
    run_dir = snapshots_root / f"{name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train variable-observation Perceiver IO flow model.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    training_config = config["training"]
    model_config_dict = config["model"]
    data_config_dict = config["data"]

    seed = int(training_config.get("seed", 2021))
    set_seed(seed)

    if torch.cuda.is_available():
        allow_tf32 = bool(training_config.get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = bool(training_config.get("cudnn_benchmark", True))
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(training_config.get("float32_matmul_precision", "high")))

    run_dir = make_run_dir(config, args.config)
    resolved_config = deepcopy(config)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)

    model_config = VariableFlowConfig.from_config_dict(model_config_dict)
    training_config = dict(training_config)
    training_config.setdefault("image_size", list(model_config.encoder.image_shape))
    training_config.setdefault("query_stride", int(model_config.decoder.query_stride))
    datamodule = VariableObservationFlowDataModule(VariableObservationFlowDataConfig(**data_config_dict))
    lightning_module = VariableFlowLightningModule(model_config=model_config, training_config=training_config)

    logger = TensorBoardLogger(save_dir=str(run_dir), name="tensorboard", version="")
    checkpoint_dir = run_dir / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
            every_n_epochs=int(training_config.get("checkpoint_every_n_epochs", 1)),
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

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
        max_steps=int(training_config["max_steps"]) if training_config.get("max_steps") is not None else -1,
        precision=training_config.get("precision", "16-mixed" if torch.cuda.is_available() else "32-true"),
        gradient_clip_val=float(training_config.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(training_config.get("log_every_n_steps", 25)),
        num_sanity_val_steps=int(training_config.get("num_sanity_val_steps", 2)),
        benchmark=bool(training_config.get("cudnn_benchmark", True)) if torch.cuda.is_available() else False,
        overfit_batches=float(training_config.get("overfit_batches", 0.0)),
        limit_train_batches=training_config.get("limit_train_batches", 1.0),
        limit_val_batches=training_config.get("limit_val_batches", 1.0),
        val_check_interval=training_config.get("val_check_interval", None),
        check_val_every_n_epoch=training_config.get("check_val_every_n_epoch", 1),
    )

    print(f"[train_variable_flow_perceiver] run_dir={run_dir}")
    print(
        "[train_variable_flow_perceiver] "
        f"phase={data_config_dict['phase']} "
        f"batch_size={data_config_dict.get('batch_size')} "
        f"precision={training_config.get('precision')} "
        f"lr={training_config.get('lr')}"
    )

    with (run_dir / "launch.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"config={Path(args.config).resolve()}\n")
        handle.write(f"cwd={Path.cwd().resolve()}\n")
        handle.write(f"devices={devices}\n")

    trainer.fit(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
