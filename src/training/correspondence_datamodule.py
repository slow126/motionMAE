"""
PyTorch Lightning DataModule for correspondence datasets.

Manages train/val datasets using CorrespondenceDataset.
"""

import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from train_cats_unified import create_training_dataset, create_validation_datasets


class CorrespondenceDataModule(pl.LightningDataModule):
    """
    DataModule for correspondence training.
    
    Handles training dataset and multiple validation datasets (one per benchmark).
    """
    
    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        """
        Initialize DataModule.
        
        Args:
            config: Full training config dict (with dataset, training, evaluation sections)
            device: Optional device (for dataset creation)
        """
        super().__init__()
        self.config = config
        self.device = device
        
        # Extract config sections
        self.dataset_config = config['dataset']
        self.training_config = config['training']
        self.eval_config = config['evaluation']
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_datasets = {}
        self.val_dataloaders = {}
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets and dataloaders.
        
        Called by Lightning before training starts.
        """
        if stage == 'fit' or stage is None:
            # Create training dataset
            self.train_dataset = create_training_dataset(self.config, device=self.device)
            
            # Create validation datasets
            self.val_datasets, self.val_dataloaders = create_validation_datasets(
                self.config, device=self.device
            )
            
            print(f"Train dataset size: {len(self.train_dataset)}")
            for benchmark, dataloader in self.val_dataloaders.items():
                print(f"  Val dataloader for benchmark '{benchmark}' size: {len(dataloader)}")

    def on_train_epoch_start(self) -> None:
        """Keep pointodyssey pair-manifest training windows contiguous across epochs."""
        steps_per_epoch = self.training_config.get("steps_per_epoch", None)
        if not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
            return

        if self.train_dataset is None:
            return

        epoch = int(getattr(self.trainer, "current_epoch", 0)) if hasattr(self, "trainer") else 0
        batch_size = int(self.training_config.get("batch_size", 1))
        world_size = max(1, self._distributed_world_size())
        chunk_samples = steps_per_epoch * batch_size * world_size
        if chunk_samples <= 0:
            return

        start_idx = epoch * chunk_samples

        dataset = self.train_dataset
        window_set = False
        seen = set()
        while dataset is not None:
            did_set = hasattr(dataset, "set_epoch_window")
            if did_set:
                dataset.set_epoch_window(start_idx, chunk_samples)
                window_set = True
                break
            dataset_id = id(dataset)
            if dataset_id in seen:
                break
            seen.add(dataset_id)
            dataset = getattr(dataset, "adapter", None)
            if dataset is not None:
                dataset = getattr(dataset, "dataset", None)

        if window_set and bool(self.training_config.get("enable_debug", False)):
            print(
                f"[CorrespondenceDataModule] train epoch {epoch + 1}: "
                f"manifest window start={start_idx}, length={chunk_samples}"
            )

    @staticmethod
    def _distributed_rank() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
        return 0

    @staticmethod
    def _distributed_world_size() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
        return 1

    @classmethod
    def _resolve_worker_count(cls, requested) -> int:
        if isinstance(requested, str):
            value = requested.strip().lower()
            if value in {"auto", "max", "all", "-1"}:
                requested = -1
            else:
                requested = int(value)
        requested = int(requested)
        if requested >= 0:
            return requested
        cpu_total = int(os.cpu_count() or 1)
        world_size = max(1, cls._distributed_world_size())
        # Allocate all available CPU workers evenly across ranks.
        return max(1, cpu_total // world_size)

    @classmethod
    def _worker_init_fn(cls, base_seed: int):
        rank = cls._distributed_rank()

        def _init(worker_id: int):
            seed = int(base_seed) + (rank * 100000) + int(worker_id)
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            torch.manual_seed(seed)

        return _init
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset not initialized. Call setup() first.")
        
        batch_size = self.training_config['batch_size']
        train_shuffle = bool(self.training_config.get('train_shuffle', True))
        n_threads = self._resolve_worker_count(self.training_config.get('n_threads', 0))
        pointodyssey_train_workers = self._resolve_worker_count(
            self.training_config.get('pointodyssey_train_workers', 0)
        )
        train_prefetch_factor = int(self.training_config.get('train_prefetch_factor', batch_size))

        dataset_overrides = self.dataset_config.get('dataset_overrides', {})
        def is_synthetic_name(name: str) -> bool:
            return isinstance(name, str) and name.startswith("synthetic")
        def is_pointodyssey_name(name: str) -> bool:
            return isinstance(name, str) and name in {"pointodyssey", "pointodyssey_pairs"}
        def dataset_has_pointodyssey():
            datasets_list = self.dataset_config.get('datasets', [])
            train_dataset_name = self.dataset_config.get('dataset_name', '')
            return (
                any(is_pointodyssey_name(name) for name in datasets_list)
                or is_pointodyssey_name(train_dataset_name)
                or any(is_pointodyssey_name(name) for name in dataset_overrides.keys())
            )

        # Check if dataset is mixed or single
        is_mixed = self.dataset_config.get('mixed', False) or 'datasets' in self.dataset_config
        if is_mixed:
            # For mixed datasets, check if any sub-dataset is synthetic
            datasets_list = self.dataset_config.get('datasets', [])
            has_synthetic = any(is_synthetic_name(name) for name in datasets_list)
            has_pointodyssey = any(is_pointodyssey_name(name) for name in datasets_list)
            if has_synthetic:
                train_num_workers = 0
            elif has_pointodyssey:
                train_num_workers = pointodyssey_train_workers
            else:
                train_num_workers = n_threads
        else:
            train_dataset_name = self.dataset_config.get('dataset_name', '')
            # Use num_workers=0 for synthetic dataset (GPU-bound rendering)
            has_pointodyssey = dataset_has_pointodyssey()
            if is_synthetic_name(train_dataset_name):
                train_num_workers = 0
            elif has_pointodyssey:
                train_num_workers = pointodyssey_train_workers
            else:
                train_num_workers = n_threads
        
        train_num_workers = max(0, int(train_num_workers))
        persistent_workers = bool(
            train_num_workers > 0 and self.training_config.get('train_persistent_workers', True)
        )
        train_seed = int(self.training_config.get('seed', 2021))
        rank = self._distributed_rank()
        generator = torch.Generator()
        generator.manual_seed(train_seed + (rank * 1000))
        worker_init_fn = self._worker_init_fn(train_seed) if train_num_workers > 0 else None
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=train_num_workers,
            shuffle=train_shuffle,
            collate_fn=self.train_dataset.collate_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=train_prefetch_factor if train_num_workers > 0 else None,
            pin_memory=True if train_num_workers > 0 else False,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
    
    def val_dataloader(self) -> DataLoader:
        """
        Return a single validation dataloader for Lightning's validation loop.
        
        Note: Actual multi-benchmark validation is handled by MMDValidationCallback
        which calls validate_epoch_multi_benchmark with all validation dataloaders.
        This returns the primary benchmark's dataloader as a placeholder.
        """
        if not self.val_dataloaders:
            raise RuntimeError("val_dataloaders not initialized. Call setup() first.")
        
        # Return primary benchmark dataloader (Lightning needs a single dataloader)
        # The callback will handle multi-benchmark validation
        primary_benchmark = self.eval_config['eval_benchmarks'][0]
        return self.val_dataloaders[primary_benchmark]
    
    def get_val_dataloaders(self) -> Dict[str, DataLoader]:
        """Get validation dataloaders dict."""
        return self.val_dataloaders
    
    def get_train_dataset_name(self) -> str:
        """Get training dataset name."""
        # Handle mixed datasets
        is_mixed = self.dataset_config.get('mixed', False) or 'datasets' in self.dataset_config
        if is_mixed:
            # For mixed datasets, return a string representation
            datasets_list = self.dataset_config.get('datasets', [])
            if datasets_list:
                return '+'.join(datasets_list)  # e.g., "spair+synthetic"
            return "mixed"
        else:
            # Single dataset
            return self.dataset_config.get('dataset_name', 'unknown')
