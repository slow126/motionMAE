"""
generate_sweep_fingerprints.py
===============================
Generate flow fingerprints for synthetic dataset parameter sweeps.
Loads SLURM experiment config and generates fingerprints for each parameter combination.

Usage:
    python src/fingerprints/generate_sweep_fingerprints.py \
        --experiment_config slurm/experiment_configs/synthetic_view_sweep.yaml \
        --machine_config slurm/machine_configs/local.yaml \
        --output_dir ./fingerprints
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.fingerprints.dataset_fingerprint import (
    process_all_datasets,
    FlowFingerprintConfig,
)
from src.data.synth.datasets.OnlineCorrespondenceDataset import OnlineCorrespondenceDataset

# Import SLURM generator functions
from slurm.generate_jobs import load_experiments, load_machine_config, load_experiment_config


def build_geometry_config_overrides(exp_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build geometry_config_overrides dict from experiment config parameters."""
    geometry_config_overrides = {}
    
    # Angle sampler overrides
    if any(key.startswith('angle_sampler_x_') for key in exp_config.keys()):
        geometry_config_overrides.setdefault('angle_sampler', {}).setdefault('x_components', {})
        if 'angle_sampler_x_loc' in exp_config and exp_config['angle_sampler_x_loc'] is not None:
            geometry_config_overrides['angle_sampler']['x_components']['loc'] = exp_config['angle_sampler_x_loc']
        if 'angle_sampler_x_scale' in exp_config and exp_config['angle_sampler_x_scale'] is not None:
            geometry_config_overrides['angle_sampler']['x_components']['scale'] = exp_config['angle_sampler_x_scale']
        if 'angle_sampler_x_distribution' in exp_config and exp_config['angle_sampler_x_distribution'] is not None:
            geometry_config_overrides['angle_sampler']['x_components']['distribution'] = exp_config['angle_sampler_x_distribution']
    
    if any(key.startswith('angle_sampler_y_') for key in exp_config.keys()):
        geometry_config_overrides.setdefault('angle_sampler', {}).setdefault('y_components', {})
        if 'angle_sampler_y_loc' in exp_config and exp_config['angle_sampler_y_loc'] is not None:
            geometry_config_overrides['angle_sampler']['y_components']['loc'] = exp_config['angle_sampler_y_loc']
        if 'angle_sampler_y_scale' in exp_config and exp_config['angle_sampler_y_scale'] is not None:
            geometry_config_overrides['angle_sampler']['y_components']['scale'] = exp_config['angle_sampler_y_scale']
        if 'angle_sampler_y_distribution' in exp_config and exp_config['angle_sampler_y_distribution'] is not None:
            geometry_config_overrides['angle_sampler']['y_components']['distribution'] = exp_config['angle_sampler_y_distribution']
    
    # Scale sampler overrides
    if any(key.startswith('scale_sampler_abs_') for key in exp_config.keys()):
        geometry_config_overrides.setdefault('scale_sampler', {}).setdefault('abs_components', {})
        if 'scale_sampler_abs_loc' in exp_config and exp_config['scale_sampler_abs_loc'] is not None:
            geometry_config_overrides['scale_sampler']['abs_components']['loc'] = exp_config['scale_sampler_abs_loc']
        if 'scale_sampler_abs_scale' in exp_config and exp_config['scale_sampler_abs_scale'] is not None:
            geometry_config_overrides['scale_sampler']['abs_components']['scale'] = exp_config['scale_sampler_abs_scale']
        if 'scale_sampler_abs_distribution' in exp_config and exp_config['scale_sampler_abs_distribution'] is not None:
            geometry_config_overrides['scale_sampler']['abs_components']['distribution'] = exp_config['scale_sampler_abs_distribution']
    
    if any(key.startswith('scale_sampler_rel_') for key in exp_config.keys()):
        geometry_config_overrides.setdefault('scale_sampler', {}).setdefault('rel_components', {})
        if 'scale_sampler_rel_loc' in exp_config and exp_config['scale_sampler_rel_loc'] is not None:
            geometry_config_overrides['scale_sampler']['rel_components']['loc'] = exp_config['scale_sampler_rel_loc']
        if 'scale_sampler_rel_scale' in exp_config and exp_config['scale_sampler_rel_scale'] is not None:
            geometry_config_overrides['scale_sampler']['rel_components']['scale'] = exp_config['scale_sampler_rel_scale']
        if 'scale_sampler_rel_distribution' in exp_config and exp_config['scale_sampler_rel_distribution'] is not None:
            geometry_config_overrides['scale_sampler']['rel_components']['distribution'] = exp_config['scale_sampler_rel_distribution']
    
    # Return None if no overrides
    if not geometry_config_overrides:
        return None
    
    return geometry_config_overrides


def generate_fingerprint_name(exp_config: Dict[str, Any]) -> str:
    """Generate a unique fingerprint name from experiment config parameters."""
    parts = ['synthetic']
    
    # Add angle sampler parameters
    if 'angle_sampler_x_loc' in exp_config and exp_config['angle_sampler_x_loc'] is not None:
        parts.append(f"xloc{exp_config['angle_sampler_x_loc']}")
    if 'angle_sampler_x_scale' in exp_config and exp_config['angle_sampler_x_scale'] is not None:
        parts.append(f"xscale{exp_config['angle_sampler_x_scale']}")
    if 'angle_sampler_x_distribution' in exp_config and exp_config['angle_sampler_x_distribution'] is not None:
        parts.append(f"xdist{exp_config['angle_sampler_x_distribution']}")
    
    if 'angle_sampler_y_loc' in exp_config and exp_config['angle_sampler_y_loc'] is not None:
        parts.append(f"yloc{exp_config['angle_sampler_y_loc']}")
    if 'angle_sampler_y_scale' in exp_config and exp_config['angle_sampler_y_scale'] is not None:
        parts.append(f"yscale{exp_config['angle_sampler_y_scale']}")
    if 'angle_sampler_y_distribution' in exp_config and exp_config['angle_sampler_y_distribution'] is not None:
        parts.append(f"ydist{exp_config['angle_sampler_y_distribution']}")
    
    # Add scale sampler parameters
    if 'scale_sampler_abs_loc' in exp_config and exp_config['scale_sampler_abs_loc'] is not None:
        parts.append(f"absloc{exp_config['scale_sampler_abs_loc']}")
    if 'scale_sampler_abs_scale' in exp_config and exp_config['scale_sampler_abs_scale'] is not None:
        parts.append(f"absscale{exp_config['scale_sampler_abs_scale']}")
    if 'scale_sampler_abs_distribution' in exp_config and exp_config['scale_sampler_abs_distribution'] is not None:
        parts.append(f"absdist{exp_config['scale_sampler_abs_distribution']}")
    
    if 'scale_sampler_rel_loc' in exp_config and exp_config['scale_sampler_rel_loc'] is not None:
        parts.append(f"relloc{exp_config['scale_sampler_rel_loc']}")
    if 'scale_sampler_rel_scale' in exp_config and exp_config['scale_sampler_rel_scale'] is not None:
        parts.append(f"relscale{exp_config['scale_sampler_rel_scale']}")
    if 'scale_sampler_rel_distribution' in exp_config and exp_config['scale_sampler_rel_distribution'] is not None:
        parts.append(f"reldist{exp_config['scale_sampler_rel_distribution']}")
    
    # Use name_exp if available, otherwise join parts
    if 'name_exp' in exp_config:
        return exp_config['name_exp']
    
    return '_'.join(parts)


def load_fingerprint_config(config_path: Optional[str] = None) -> FlowFingerprintConfig:
    """Load fingerprint configuration from YAML file."""
    if config_path is None:
        config_path = project_root / 'src/configs/fingerprints/fingerprint_config.yaml'
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    fp_config = config_dict.get('fingerprint_config', {})
    
    # Convert spatial_downsample_hw list to tuple
    if 'spatial_downsample_hw' in fp_config:
        fp_config['spatial_downsample_hw'] = tuple(fp_config['spatial_downsample_hw'])
    
    # Ensure numeric values are converted from strings
    numeric_fields = [
        'mag_min', 'mag_max', 'ang_weight_clip',
        'delta_min', 'delta_max',
        'div_min', 'div_max', 'curl_min', 'curl_max',
        'motion_thresh'
    ]
    for field in numeric_fields:
        if field in fp_config:
            fp_config[field] = float(fp_config[field])
    
    return FlowFingerprintConfig(**fp_config)


def main():
    parser = argparse.ArgumentParser(
        description='Generate flow fingerprints for synthetic dataset parameter sweeps'
    )
    
    parser.add_argument(
        '--experiment_config',
        type=str,
        required=True,
        help='Path to SLURM experiment config YAML file'
    )
    parser.add_argument(
        '--machine_config',
        type=str,
        default='slurm/machine_configs/local.yaml',
        help='Path to machine config YAML file (default: slurm/machine_configs/local.yaml)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./fingerprints',
        help='Output directory for fingerprints and plots (default: ./fingerprints)'
    )
    parser.add_argument(
        '--fingerprint_config',
        type=str,
        default=None,
        help='Path to fingerprint config YAML (default: src/configs/fingerprints/fingerprint_config.yaml)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples per dataset (None = all)'
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    args = parser.parse_args()
    
    # Load experiment and machine configs
    experiment_config = load_experiment_config(args.experiment_config)
    machine_config = load_machine_config(args.machine_config)
    
    # Expand all experiments from grids
    experiments = load_experiments(experiment_config, machine_config)
    
    print(f"Found {len(experiments)} experiment configurations")
    print(f"Output directory: {args.output_dir}\n")
    
    # Load fingerprint config
    fingerprint_config = load_fingerprint_config(args.fingerprint_config)
    
    # Process each experiment
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, exp_config in enumerate(experiments, 1):
        # Only process synthetic datasets
        if exp_config.get('train_dataset') != 'synthetic':
            print(f"[{i}/{len(experiments)}] Skipping non-synthetic dataset: {exp_config.get('train_dataset')}")
            continue
        
        # Generate fingerprint name
        fingerprint_name = generate_fingerprint_name(exp_config)
        print(f"[{i}/{len(experiments)}] Processing: {fingerprint_name}")
        
        # Build geometry config overrides
        geometry_config_overrides = build_geometry_config_overrides(exp_config)
        
        # Get config paths
        geometry_config_path = exp_config.get('geometry_config_path', 
                                             'src/configs/online_synth_configs/OnlineGeometryConfig.yaml')
        processor_config_path = exp_config.get('processor_config_path',
                                              'src/configs/online_synth_configs/OnlineProcessorConfig.yaml')
        
        try:
            # Create dataset with overrides
            dataset = OnlineCorrespondenceDataset(
                geometry_config_path=geometry_config_path,
                processor_config_path=processor_config_path,
                split='train',
                opengl_device_index=None,
                geometry_config_overrides=geometry_config_overrides
            )
            
            # Move dataset to CUDA if available
            if torch.cuda.is_available():
                dataset.cuda()
            
            # Build dataset config for process_all_datasets
            dataset_config = {
                'name': fingerprint_name,
                'dataset': dataset,
                'max_samples': args.max_samples,
                'use_dataloader': True,
                'batch_size': 1,
                'num_workers': 0,  # Required for synthetic dataset
                'collate_fn': dataset.collate_fn,
            }
            
            # Generate fingerprint using process_all_datasets
            print(f"  Generating fingerprint...")
            results = process_all_datasets(
                dataset_configs=[dataset_config],
                output_dir=str(output_dir),
                config=fingerprint_config,
                generate_plots=not args.no_plots,
                generate_comparison=False,  # Don't generate comparison plots for sweeps
            )
            
            if results.get('fingerprints'):
                print(f"  ✓ Saved: {list(results['fingerprints'].values())[0]}")
            if not args.no_plots and results.get('plots'):
                print(f"  ✓ Plots: {list(results['plots'].values())[0]}")
        
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Fingerprint generation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

