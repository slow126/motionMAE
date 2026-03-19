# Flow Fingerprint Module

This module generates compact "flow fingerprints" for optical flow datasets, providing statistical characterizations of flow patterns.

## Overview

The fingerprint module consists of three main components:

1. **`flow_fingerprint.py`**: Core fingerprint computation engine
   - `FlowFingerprint` class: Accumulates statistics from flow fields
   - `FlowFingerprintConfig`: Configuration for fingerprint parameters
   - Produces histograms, spatial maps, and statistical moments

2. **`plot_flow_fingerprint.py`**: Visualization utilities
   - Plot histograms (magnitude, angle, delta, divergence, curl)
   - Plot spatial maps (motion probability, mean magnitude)
   - Overlay comparisons across multiple datasets

3. **`dataset_fingerprint.py`**: Dataset processing pipeline
   - Converts flow formats from different datasets
   - Processes PyTorch datasets/dataloaders
   - Generates fingerprints and saves results

## Quick Start

### Basic Usage

```python
from src.fingerprints.dataset_fingerprint import (
    compute_dataset_fingerprint,
    FlowFingerprintConfig,
)
from src.data.synth.datasets.FlyingThingsDataset import FlyingThingsDataset

# Create dataset
dataset = FlyingThingsDataset(
    root='/path/to/FlyingThings3D',
    split='train',
    size=(512, 512),
)

# Generate fingerprint
stats = compute_dataset_fingerprint(
    dataset=dataset,
    dataset_name='flyingthings_train',
    max_samples=1000,  # Optional: limit samples
)

# Save results
from src.fingerprints.flow_fingerprint import save_stats_json
save_stats_json('fingerprint.json', stats)
```

### Process Multiple Datasets

```python
from src.fingerprints.dataset_fingerprint import process_all_datasets

dataset_configs = [
    {
        'name': 'flyingthings_train',
        'dataset': flyingthings_dataset,
        'max_samples': 1000,
    },
    {
        'name': 'pointodyssey_train',
        'dataset': pointodyssey_dataset,
        'max_samples': 1000,
    },
]

results = process_all_datasets(
    dataset_configs=dataset_configs,
    output_dir='./fingerprints',
    generate_plots=True,
    generate_comparison=True,
)
```

### Using the Example Script

```bash
# Process all datasets
python src/fingerprints/example_generate_fingerprints.py \
    --output_dir ./fingerprints \
    --max_samples 1000

# Process specific dataset
python src/fingerprints/example_generate_fingerprints.py \
    --dataset flyingthings \
    --split train \
    --output_dir ./fingerprints

# Process without plots (faster)
python src/fingerprints/example_generate_fingerprints.py \
    --dataset all \
    --no_plots \
    --output_dir ./fingerprints
```

## Flow Format Handling

The module automatically handles different flow formats:

- **`[H, W, 2]`** (FlyingThings): Directly compatible
- **`[2, H, W]`** (PointOdyssey, TSS): Automatically converted
- **Torch tensors**: Converted to numpy arrays
- **Valid masks**: Extracted from `valid_flow_mask` or `valid_mask` keys

## Fingerprint Output

The fingerprint JSON contains:

```json
{
  "config": { ... },           // Configuration used
  "bins": { ... },             // Bin edges for histograms
  "hists": {                   // Probability histograms
    "mag": [...],              // Magnitude distribution
    "angle": [...],            // Angle distribution
    "joint_mag_angle": [[...]], // Joint magnitude×angle
    "delta": [...],            // Temporal delta
    "div": [...],              // Divergence
    "curl": [...]              // Curl
  },
  "moments": {                 // Statistical moments
    "mag_mean": ...,
    "mag_median": ...,
    "mag_p95": ...,
    "sparsity_motion_frac": ...
  },
  "spatial": {                 // Spatial maps
    "grid_hw": [32, 32],
    "motion_prob": [[...]],    // P[motion > threshold]
    "mean_magnitude": [[...]]  // E[magnitude]
  },
  "metadata": {                 // Processing metadata
    "dataset_name": ...,
    "samples_processed": ...,
    ...
  }
}
```

## Customization

### Custom Configuration

```python
from src.fingerprints.flow_fingerprint import FlowFingerprintConfig

config = FlowFingerprintConfig(
    mag_bins=50,                    # More bins for magnitude
    spatial_downsample_hw=(64, 64), # Higher resolution spatial maps
    motion_thresh=0.5,              # Higher motion threshold
)

stats = compute_dataset_fingerprint(
    dataset=dataset,
    config=config,
    dataset_name='custom',
)
```

### Processing Options

```python
stats = compute_dataset_fingerprint(
    dataset=dataset,
    dataset_name='mydataset',
    max_samples=5000,           # Limit samples
    use_dataloader=True,        # Use DataLoader (for batching)
    batch_size=4,               # Batch size
    num_workers=2,               # DataLoader workers
    track_temporal=True,        # Enable temporal delta
    progress=True,              # Show progress bar
)
```

## Visualization

### Generate Individual Plots

```python
from src.fingerprints.plot_flow_fingerprint import (
    plot_histograms,
    plot_spatial_maps,
)

stats = load_stats_json('fingerprint.json')
plot_histograms(stats, './plots')
plot_spatial_maps(stats, './plots')
```

### Compare Multiple Datasets

```python
from src.fingerprints.plot_flow_fingerprint import plot_overlay
from src.fingerprints.flow_fingerprint import load_stats_json

stats_list = [
    load_stats_json('flyingthings_fingerprint.json'),
    load_stats_json('pointodyssey_fingerprint.json'),
]

plot_overlay(
    stats_list,
    labels=['FlyingThings', 'PointOdyssey'],
    out_dir='./comparison',
)
```

## Integration with Your Datasets

The module works with any PyTorch dataset that returns samples with a `'flow'` key:

```python
sample = {
    'flow': torch.Tensor,  # Shape: [2, H, W] or [H, W, 2]
    'valid_flow_mask': torch.Tensor,  # Optional: [H, W] boolean
    # ... other keys ...
}
```

If your dataset uses a different format, you can create a wrapper:

```python
class MyDatasetWrapper:
    def __init__(self, original_dataset):
        self.dataset = original_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # Convert to expected format
        return {
            'flow': sample['my_flow_key'],  # Convert format if needed
            'valid_flow_mask': sample.get('my_mask_key'),
        }
```

## Troubleshooting

### Flow Format Issues

If you see errors about flow shapes, check:
- Flow should be `[H, W, 2]` or `[2, H, W]`
- Flow values should be in pixels (not normalized)
- Use `convert_flow_to_numpy()` to debug format conversion

### Memory Issues

For large datasets:
- Use `max_samples` to limit processing
- Set `use_dataloader=False` to avoid batching overhead
- Process datasets separately instead of all at once

### Missing Valid Masks

The module handles missing valid masks gracefully. If your dataset doesn't provide masks, all pixels are treated as valid.

## API Reference

See docstrings in:
- `flow_fingerprint.py`: Core fingerprint computation
- `plot_flow_fingerprint.py`: Visualization functions
- `dataset_fingerprint.py`: Dataset processing functions

