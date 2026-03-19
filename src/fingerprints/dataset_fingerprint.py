"""
dataset_fingerprint.py
======================
Process PyTorch datasets to generate flow fingerprints.

Main functions:
- compute_dataset_fingerprint(dataset, config, ...) -> dict
- process_all_datasets(dataset_configs, output_dir, ...)

Handles flow format conversion, valid masks, and temporal tracking.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

from src.fingerprints.flow_fingerprint import (
    FlowFingerprint,
    FlowFingerprintConfig,
    save_stats_json,
    load_stats_json,
)
from src.fingerprints.plot_flow_fingerprint import (
    plot_histograms,
    plot_spatial_maps,
    plot_overlay,
)


# ------------------------------
# Flow Format Conversion
# ------------------------------

def convert_flow_to_numpy(flow: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert flow tensor/array to numpy array with shape [H, W, 2].
    
    Handles:
    - torch.Tensor [2, H, W] -> numpy [H, W, 2]
    - torch.Tensor [H, W, 2] -> numpy [H, W, 2]
    - numpy array [2, H, W] -> numpy [H, W, 2]
    - numpy array [H, W, 2] -> numpy [H, W, 2] (no-op)
    
    Note: Preserves inf/nan values for later detection. Invalid flow (inf/nan) 
    will be automatically excluded in add_frame() if no valid_mask is provided.
    
    Args:
        flow: Flow tensor/array in any supported format
        
    Returns:
        numpy array with shape [H, W, 2] where flow[..., 0] = u (x-component),
        flow[..., 1] = v (y-component). May contain inf/nan for invalid pixels.
    """
    # Convert torch tensor to numpy
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
    
    # Handle different shapes
    if flow.ndim == 3:
        if flow.shape[0] == 2 and flow.shape[1] != 2:
            # [2, H, W] -> [H, W, 2]
            flow = np.transpose(flow, (1, 2, 0))
        elif flow.shape[-1] == 2:
            # Already [H, W, 2]
            pass
        else:
            raise ValueError(f"Unexpected flow shape: {flow.shape}. Expected [2, H, W] or [H, W, 2]")
    else:
        raise ValueError(f"Flow must be 3D, got shape: {flow.shape}")
    
    return flow.astype(np.float64)


def extract_valid_mask(sample: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Extract valid flow mask from dataset sample.
    
    Handles different mask formats:
    - 'valid_flow_mask' (bool tensor/array)
    - 'valid_mask' (bool tensor/array)
    - None (no mask available)
    
    Args:
        sample: Dataset sample dictionary
        
    Returns:
        numpy boolean array [H, W] or None if no mask available
    """
    # Try different key names
    for key in ['valid_flow_mask', 'valid_mask', 'mask']:
        if key in sample:
            mask = sample[key]
            # Convert torch tensor to numpy (handles GPU tensors)
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            # Ensure boolean
            if mask.dtype != bool:
                mask = mask > 0
            return mask.astype(bool)
    
    return None


# ------------------------------
# Dataset Processing
# ------------------------------

def compute_dataset_fingerprint(
    dataset: Union[Dataset, DataLoader],
    config: Optional[FlowFingerprintConfig] = None,
    dataset_name: str = "unknown",
    max_samples: Optional[int] = None,
    use_dataloader: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[callable] = None,
    progress: bool = True,
    track_temporal: bool = True,
    flow_filter: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compute flow fingerprint for a PyTorch dataset or DataLoader.
    
    Args:
        dataset: PyTorch Dataset or DataLoader
        config: FlowFingerprintConfig (uses default if None)
        dataset_name: Name of dataset for metadata
        max_samples: Maximum number of samples to process (None = all)
        use_dataloader: If True and dataset is Dataset, wrap in DataLoader
        batch_size: Batch size for DataLoader (only used if use_dataloader=True)
        num_workers: Number of workers for DataLoader
        collate_fn: Custom collate function for DataLoader (e.g., for synthetic datasets with CUDA kernels)
        progress: Show progress bar
        track_temporal: Track previous flow for temporal delta calculation
        flow_filter: Optional FlowLengthFilter instance to filter flow vectors by length
        
    Returns:
        Dictionary with fingerprint stats and metadata
    """
    if config is None:
        config = FlowFingerprintConfig()
    
    # Initialize fingerprint accumulator
    fingerprint = FlowFingerprint(config)
    
    # Determine if we need to create a DataLoader
    if isinstance(dataset, DataLoader):
        dataloader = dataset
        is_dataloader = True
        # Get batch_size from DataLoader
        actual_batch_size = dataloader.batch_size or 1
    elif use_dataloader:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=collate_fn,  # Use custom collate_fn if provided (e.g., for synthetic datasets)
        )
        is_dataloader = True
        actual_batch_size = batch_size
    else:
        dataloader = None
        is_dataloader = False
        actual_batch_size = 1
    
    # Track previous flow for temporal delta
    prev_flow = None
    
    # Determine total number of samples
    # For DataLoader, we need to estimate (will update as we process)
    if is_dataloader:
        # Estimate: assume actual_batch_size samples per batch
        estimated_samples = len(dataloader) * actual_batch_size
        if max_samples is not None:
            estimated_samples = min(estimated_samples, max_samples)
        total_samples = estimated_samples
    else:
        total_samples = len(dataset)
        if max_samples is not None:
            total_samples = min(total_samples, max_samples)
    
    # Progress bar
    if progress:
        pbar = tqdm(total=total_samples, desc=f"Processing {dataset_name}")
    
    samples_processed = 0
    errors = []
    
    try:
        if is_dataloader:
            # Process batches from DataLoader
            for batch_idx, batch in enumerate(dataloader):
                if max_samples is not None and samples_processed >= max_samples:
                    break
                
                # Handle batched data
                if isinstance(batch, dict):
                    if 'flow' in batch:
                        flow_batch = batch['flow']
                        
                        # Apply flow filtering if specified
                        if flow_filter is not None and isinstance(flow_batch, torch.Tensor):
                            if flow_batch.ndim == 4:  # [B, 2, H, W]
                                batch['flow'] = flow_filter.filter_batch_flow(flow_batch)
                            elif flow_batch.ndim == 3:  # [2, H, W] - single sample
                                # Add batch dimension, filter, then remove
                                flow_filtered = flow_filter.filter_batch_flow(flow_batch.unsqueeze(0))
                                batch['flow'] = flow_filtered.squeeze(0)
                        
                        # Get flow_batch again after potential filtering
                        flow_batch = batch['flow']
                        if flow_batch.ndim == 4:  # [B, 2, H, W] or [B, H, W, 2]
                            # Process each item in batch
                            batch_size_actual = flow_batch.shape[0]
                            for i in range(batch_size_actual):
                                if max_samples is not None and samples_processed >= max_samples:
                                    break
                                
                                sample = {k: v[i] if isinstance(v, torch.Tensor) else v 
                                         for k, v in batch.items()}
                                _process_single_sample(
                                    sample, fingerprint, prev_flow, track_temporal
                                )
                                if track_temporal:
                                    prev_flow = _extract_flow_from_sample(sample)
                                samples_processed += 1
                                if progress:
                                    pbar.update(1)
                        else:
                            # Single sample (3D tensor)
                            _process_single_sample(
                                batch, fingerprint, prev_flow, track_temporal
                            )
                            if track_temporal:
                                prev_flow = _extract_flow_from_sample(batch)
                            samples_processed += 1
                            if progress:
                                pbar.update(1)
                    else:
                        # No flow in batch, skip
                        if progress:
                            pbar.update(1)
                        samples_processed += 1
                else:
                    # Unexpected batch format, skip
                    if progress:
                        pbar.update(1)
                    samples_processed += 1
        else:
            # Process dataset directly
            for idx in range(total_samples):
                try:
                    sample = dataset[idx]
                    
                    # Apply flow filtering if specified
                    if flow_filter is not None and 'flow' in sample and isinstance(sample['flow'], torch.Tensor):
                        flow_tensor = sample['flow']
                        if flow_tensor.ndim == 3:  # [2, H, W]
                            # Add batch dimension, filter, then remove
                            flow_filtered = flow_filter.filter_batch_flow(flow_tensor.unsqueeze(0))
                            sample['flow'] = flow_filtered.squeeze(0)
                        elif flow_tensor.ndim == 4:  # [B, 2, H, W] - shouldn't happen for direct dataset access, but handle it
                            sample['flow'] = flow_filter.filter_batch_flow(flow_tensor)
                    
                    _process_single_sample(
                        sample, fingerprint, prev_flow, track_temporal
                    )
                    if track_temporal:
                        prev_flow = _extract_flow_from_sample(sample)
                    samples_processed += 1
                    if progress:
                        pbar.update(1)
                except Exception as e:
                    errors.append(f"Sample {idx}: {str(e)}")
                    if progress:
                        pbar.update(1)
                    continue
    
    finally:
        if progress:
            pbar.close()
    
    # Finalize fingerprint
    stats = fingerprint.finalize()
    
    # Add metadata
    stats['metadata'] = {
        'dataset_name': dataset_name,
        'samples_processed': samples_processed,
        'total_samples_available': total_samples,
        'errors': errors,
        'track_temporal': track_temporal,
    }
    
    return stats


def _process_single_sample(
    sample: Dict[str, Any],
    fingerprint: FlowFingerprint,
    prev_flow: Optional[np.ndarray],
    track_temporal: bool,
) -> None:
    """Process a single sample and add to fingerprint."""
    if 'flow' not in sample:
        return
    
    # Convert flow to numpy [H, W, 2]
    flow = convert_flow_to_numpy(sample['flow'])
    
    # Extract valid mask
    valid_mask = extract_valid_mask(sample)
    
    # Add frame to fingerprint
    fingerprint.add_frame(
        flow=flow,
        prev_flow=prev_flow if track_temporal else None,
        valid_mask=valid_mask,
    )


def _extract_flow_from_sample(sample: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extract flow from sample for temporal tracking."""
    if 'flow' not in sample:
        return None
    return convert_flow_to_numpy(sample['flow'])


# ------------------------------
# Batch Processing
# ------------------------------

def process_all_datasets(
    dataset_configs: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    config: Optional[FlowFingerprintConfig] = None,
    generate_plots: bool = True,
    generate_comparison: bool = True,
    max_samples_per_dataset: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Process multiple datasets and generate fingerprints.
    
    Args:
        dataset_configs: List of dicts, each with:
            - 'name': str, dataset name
            - 'dataset': Dataset or DataLoader instance
            - Optional: 'max_samples', 'use_dataloader', 'batch_size', etc.
        output_dir: Directory to save fingerprints and plots
        config: FlowFingerprintConfig (uses default if None)
        generate_plots: Generate individual plots for each dataset
        generate_comparison: Generate comparison plots across datasets
        max_samples_per_dataset: Default max samples (can be overridden per dataset)
        **kwargs: Additional kwargs passed to compute_dataset_fingerprint
    
    Returns:
        Dictionary with paths to saved files and summary statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config is None:
        config = FlowFingerprintConfig()
    
    all_stats = []
    all_labels = []
    results = {
        'fingerprints': {},
        'plots': {},
        'summary': {},
    }
    
    print(f"Processing {len(dataset_configs)} datasets...")
    print(f"Output directory: {output_dir}\n")
    
    for ds_config in dataset_configs:
        name = ds_config['name']
        dataset = ds_config['dataset']
        
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")
        
        # Extract dataset-specific parameters
        ds_kwargs = {
            'dataset_name': name,
            'config': config,
            'flow_filter': ds_config.get('flow_filter'),  # Pass flow filter if provided
            'max_samples': ds_config.get('max_samples', max_samples_per_dataset),
            'use_dataloader': ds_config.get('use_dataloader', False),
            'batch_size': ds_config.get('batch_size', 1),
            'num_workers': ds_config.get('num_workers', 0),
            'collate_fn': ds_config.get('collate_fn', None),  # Custom collate_fn for synthetic datasets
            'track_temporal': ds_config.get('track_temporal', True),
            **kwargs,
        }
        
        # Compute fingerprint
        stats = compute_dataset_fingerprint(dataset, **ds_kwargs)
        
        # Save JSON
        json_path = output_dir / f"{name}_fingerprint.json"
        save_stats_json(str(json_path), stats)
        results['fingerprints'][name] = str(json_path)
        print(f"Saved fingerprint: {json_path}")
        
        # Generate plots
        if generate_plots:
            plot_dir = output_dir / f"{name}_plots"
            plot_dir.mkdir(exist_ok=True)
            
            plot_histograms(stats, str(plot_dir))
            plot_spatial_maps(stats, str(plot_dir))
            results['plots'][name] = str(plot_dir)
            print(f"Saved plots: {plot_dir}")
        
        # Store for comparison
        all_stats.append(stats)
        all_labels.append(name)
    
    # Generate comparison plots
    if generate_comparison and len(all_stats) > 1:
        comparison_dir = output_dir / "comparison"
        comparison_dir.mkdir(exist_ok=True)
        
        plot_overlay(all_stats, all_labels, str(comparison_dir))
        results['comparison_plots'] = str(comparison_dir)
        print(f"\nSaved comparison plots: {comparison_dir}")
    
    # Summary
    results['summary'] = {
        'num_datasets': len(dataset_configs),
        'dataset_names': all_labels,
        'output_dir': str(output_dir),
    }
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")
    
    return results


# ------------------------------
# Convenience Functions
# ------------------------------

def load_and_compare_fingerprints(
    fingerprint_paths: List[Union[str, Path]],
    labels: Optional[List[str]] = None,
    output_dir: Union[str, Path] = "comparison",
) -> None:
    """
    Load multiple fingerprint JSON files and generate comparison plots.
    
    Args:
        fingerprint_paths: List of paths to fingerprint JSON files
        labels: Optional list of labels (uses filenames if None)
        output_dir: Directory to save comparison plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_stats = []
    if labels is None:
        labels = [Path(p).stem.replace('_fingerprint', '') for p in fingerprint_paths]
    
    for path in fingerprint_paths:
        stats = load_stats_json(str(path))
        all_stats.append(stats)
    
    plot_overlay(all_stats, labels, str(output_dir))
    print(f"Comparison plots saved to: {output_dir}")


# ------------------------------
# CLI / Example Usage
# ------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate flow fingerprints for datasets")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (flyingthings, pointodyssey, etc.)")
    parser.add_argument("--output_dir", type=str, default="./fingerprints",
                       help="Output directory for fingerprints and plots")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process")
    parser.add_argument("--no_plots", action="store_true",
                       help="Skip generating plots")
    parser.add_argument("--no_comparison", action="store_true",
                       help="Skip generating comparison plots")
    
    args = parser.parse_args()
    
    # This is a template - users should customize based on their dataset setup
    print("Note: This is a template script. Customize dataset loading based on your setup.")
    print("See the function documentation for usage examples.")

