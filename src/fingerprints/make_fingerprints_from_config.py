"""
make_fingerprints_from_config.py
=================================
Generate flow fingerprints from SLURM experiment configs.
Reads in a list of experiment config files, expands them into individual experiments,
creates datasets for both training and evaluation benchmarks, and generates fingerprints.

Usage:
    python src/fingerprints/make_fingerprints_from_config.py
"""

import os
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
from src.data.synth.datasets.FlyingThingsDataset import FlyingThingsDataset
from src.data.synth.datasets.PointOdysseyCorrespondence import PointOdysseyFlowDataset
from src.data.synth.datasets.OnlineCorrespondenceDataset import OnlineCorrespondenceDataset
from src.data.synth.datasets.KittiDataset import KittiDataset
import models.CATs_PlusPlus.data.download as download

# Import SLURM generator functions
from slurm.generate_jobs import load_experiments, load_machine_config, load_experiment_config

# ============================================================================
# CONFIGURATION - Set your experiment configs here
# ============================================================================
CONFIGS = [
    # 'slurm/experiment_configs/synthetic.yaml',
    # 'slurm/experiment_configs/synthetic_views.yaml',
    'slurm/experiment_configs/fingerprints.yaml',
    # Add more config paths as needed
]

MACHINE_CONFIG = 'slurm/machine_configs/local.yaml'
OUTPUT_DIR = './fingerprints'
FINGERPRINT_CONFIG = None  # None = use default: src/configs/fingerprints/fingerprint_config.yaml
MAX_SAMPLES = 100  # None = process all samples
GENERATE_PLOTS = True
GENERATE_COMPARISON = False  # Set to True if you want comparison plots across all experiments


# ============================================================================
# Helper Functions
# ============================================================================

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


def load_fingerprint_config(config_path: Optional[str] = None) -> FlowFingerprintConfig:
    """Load fingerprint configuration from YAML file."""
    if config_path is None:
        config_path = project_root / 'src/configs/fingerprints/fingerprint_config.yaml'
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = project_root / config_path
    
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


# ============================================================================
# Training Dataset Creation
# ============================================================================

def create_training_dataset_from_exp_config(exp_config: Dict[str, Any], split: str = 'train') -> Any:
    """Create a training dataset from experiment config based on train_dataset type."""
    train_dataset = exp_config.get('train_dataset', 'synthetic')
    
    if train_dataset == 'synthetic':
        # Build geometry config overrides
        geometry_config_overrides = build_geometry_config_overrides(exp_config)
        
        # Get config paths
        geometry_config_path = exp_config.get('geometry_config_path', 
                                             'src/configs/online_synth_configs/OnlineGeometryConfig.yaml')
        processor_config_path = exp_config.get('processor_config_path',
                                              'src/configs/online_synth_configs/OnlineProcessorConfig.yaml')
        
        # Use val geometry config for val split
        if split == 'val':
            geometry_config_path = exp_config.get('geometry_config_val_path', 
                                                 exp_config.get('geometry_config_path',
                                                              'src/configs/online_synth_configs/OnlineGeometryConfig_Val.yaml'))
        
        opengl_device_index = exp_config.get('opengl_device_index')
        if opengl_device_index == 'null' or opengl_device_index == '':
            opengl_device_index = None
        
        dataset = OnlineCorrespondenceDataset(
            geometry_config_path=geometry_config_path,
            processor_config_path=processor_config_path,
            split=split,
            opengl_device_index=opengl_device_index,
            geometry_config_overrides=geometry_config_overrides
        )
        
        # Move dataset to CUDA if available
        if torch.cuda.is_available():
            dataset.cuda()
        
        return dataset
    
    elif train_dataset == 'flyingthings':
        root = exp_config.get('flyingthings_root', '')
        size = exp_config.get('size', 512)
        downsample_flow = exp_config.get('feature_size', 32)
        
        dataset = FlyingThingsDataset(
            root=root,
            split=split,
            transforms=None,
            size=(size, size),
            downsample_flow=downsample_flow,
            subsample_flow=0.6,
            use_valid_mask=True,
            reverse_flow=True,
            filter_out_of_bounds=True,
        )
        return dataset
    
    elif train_dataset == 'pointodyssey':
        root = exp_config.get('pointodyssey_root', '')
        size = exp_config.get('size', 512)
        feature_size = exp_config.get('feature_size', 32)
        num_pts = exp_config.get('num_pts_to_track_pointodyssey', 32)
        
        # Map split: train -> train, val -> val, test -> test
        dset = split
        
        # Handle resize_size and crop_size
        resize_size = (size + 64, size + 64)
        crop_size = (size, size)
        
        dataset = PointOdysseyFlowDataset(
            dataset_location=root,
            dset=dset,
            use_augs=False,  # No augmentation for fingerprinting
            S=8,
            N=num_pts,
            strides=[1, 2, 4],
            quick=False,
            verbose=False,
            resize_size=resize_size,
            crop_size=crop_size,
            filter_instances=True,
            downsample_for_cats=True,
            cats_feat_size=feature_size,
            all_points=True,
            max_pts=40,
            normalize=True,
        )
        return dataset
    
    elif train_dataset in ['kitti2012', 'kitti2015']:
        version = '2012' if '2012' in train_dataset else '2015'
        
        # Determine which root to use based on kitti_val_use_full_training flag
        if split == 'val' and exp_config.get('kitti_val_use_full_training', False):
            # Use unsplit root with full training set
            kitti_root = exp_config.get('kitti_unsplit_root', '/home/spencer/Data/correspondence/kitti')
            split = 'training'  # Use full training set for validation
        else:
            # Use split root
            kitti_root = exp_config.get('kitti_root', '/home/spencer/Data/correspondence/kitti-split')
        
        # Construct full path: kitti_root/kitti-{version}
        root = os.path.join(kitti_root, f'kitti-{version}')
        
        size = exp_config.get('size', 512)
        if isinstance(size, int):
            size = (size, size)
        
        feature_size = exp_config.get('feature_size', 32)
        
        dataset = KittiDataset(
            root=root,
            split=split,
            version=version,
            occ_type='occ',
            size=size,
            downsample_flow=feature_size,
            normalize=True,
            normalize_images=True,
            thres='img',
            max_pts=200,
        )
        return dataset
    
    # Real datasets (SPair, PFPascal, PFWillow)
    elif train_dataset in ['spair', 'pfpascal', 'pfwillow']:
        datapath = exp_config.get('datapath', './models/Datasets_CATs')
        thres = exp_config.get('thres', 'img')
        feature_size = exp_config.get('feature_size', 32)
        device = torch.device('cpu')
        
        # Map split names for real datasets
        # For training, use 'trn' split
        if split == 'train':
            dataset_split = 'trn'
        elif split == 'val':
            dataset_split = 'val'
        elif split == 'test':
            dataset_split = 'test'
        else:
            dataset_split = 'trn'  # Default to training split
        
        # Download if needed
        try:
            download.download_dataset(datapath, train_dataset)
        except:
            pass  # May already exist
        
        dataset = download.load_dataset(
            train_dataset, datapath, thres, device, dataset_split,
            False,  # No augmentation for fingerprinting
            feature_size
        )
        return dataset
    
    else:
        raise ValueError(f"Unknown train_dataset: {train_dataset}")


# ============================================================================
# Evaluation Dataset Creation
# ============================================================================

def create_eval_dataset_from_exp_config(exp_config: Dict[str, Any], benchmark_name: str, split: str = 'val') -> Any:
    """Create an evaluation dataset from experiment config for a specific benchmark."""
    benchmark_lower = benchmark_name.lower()
    
    # TSS dataset (no splits)
    if benchmark_lower == 'tss':
        from src.data.synth.datasets.TSSDataset import TSSDataset
        
        tss_root = exp_config.get('tss_root', '/home/spencer/Data/correspondence/TSS_CVPR2016')
        size = exp_config.get('size', 512)
        feature_size = exp_config.get('feature_size', 32)
        
        dataset = TSSDataset(
            root=tss_root,
            device='cpu',
            size=size,
            feature_size=feature_size,
            max_pts=40,
            thres='img',
            augmentation=False,
            sample_keypoints=True,
        )
        return dataset
    
    # Real datasets (SPair, PFPascal, PFWillow)
    elif benchmark_lower in ['spair', 'pfpascal', 'pfwillow']:
        datapath = exp_config.get('datapath', './models/Datasets_CATs')
        thres = exp_config.get('thres', 'img')
        feature_size = exp_config.get('feature_size', 32)
        device = torch.device('cpu')
        
        # Map split names for real datasets
        if split == 'train':
            split = 'trn'
        elif split == 'val':
            split = 'val'
        elif split == 'test':
            split = 'test'
        
        # Download if needed
        try:
            download.download_dataset(datapath, benchmark_lower)
        except:
            pass  # May already exist
        
        dataset = download.load_dataset(
            benchmark_lower, datapath, thres, device, split,
            False,  # No augmentation for fingerprinting
            feature_size
        )
        return dataset
    
    # KITTI datasets
    elif benchmark_lower in ['kitti2012', 'kitti2015']:
        version = '2012' if '2012' in benchmark_lower else '2015'
        
        # For evaluation benchmarks, check if we should use full training set
        if exp_config.get('kitti_val_use_full_training', False):
            # Use unsplit root with full training set
            kitti_root = exp_config.get('kitti_unsplit_root', '/home/spencer/Data/correspondence/kitti')
            split = 'training'  # Use full training set
        else:
            # Use split root with val split
            kitti_root = exp_config.get('kitti_root', '/home/spencer/Data/correspondence/kitti-split')
            split = 'val'  # Use val split from split root
        
        # Construct full path: kitti_root/kitti-{version}
        root = os.path.join(kitti_root, f'kitti-{version}')
        
        size = exp_config.get('size', 512)
        if isinstance(size, int):
            size = (size, size)
        
        feature_size = exp_config.get('feature_size', 32)
        
        dataset = KittiDataset(
            root=root,
            split=split,
            version=version,
            occ_type='occ',
            size=size,
            downsample_flow=feature_size,
            normalize=True,
            normalize_images=True,
            thres='img',
            max_pts=200,
        )
        return dataset
    
    # PointOdyssey
    elif benchmark_lower == 'pointodyssey':
        root = exp_config.get('pointodyssey_root', '')
        size = exp_config.get('size', 512)
        feature_size = exp_config.get('feature_size', 32)
        num_pts = exp_config.get('num_pts_to_track_pointodyssey', 32)
        
        # Use 'val' split for evaluation
        dset = 'val'
        
        # Handle resize_size and crop_size
        resize_size = (size + 64, size + 64)
        crop_size = (size, size)
        
        dataset = PointOdysseyFlowDataset(
            dataset_location=root,
            dset=dset,
            use_augs=False,
            S=8,
            N=num_pts,
            strides=[1, 2, 4],
            quick=False,
            verbose=False,
            resize_size=resize_size,
            crop_size=crop_size,
            filter_instances=True,
            downsample_for_cats=True,
            cats_feat_size=feature_size,
            all_points=True,
            max_pts=40,
            normalize=True,
        )
        return dataset
    
    # FlyingThings
    elif benchmark_lower == 'flyingthings':
        root = exp_config.get('flyingthings_root', '')
        size = exp_config.get('size', 512)
        downsample_flow = exp_config.get('feature_size', 32)
        
        # Use 'test' split for evaluation (FlyingThings uses test, not val)
        dataset = FlyingThingsDataset(
            root=root,
            split='test',
            transforms=None,
            size=(size, size),
            downsample_flow=downsample_flow,
            subsample_flow=0.6,
            use_valid_mask=True,
            reverse_flow=True,
            filter_out_of_bounds=True,
        )
        return dataset
    
    else:
        raise ValueError(f"Unknown evaluation benchmark: {benchmark_name}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to generate fingerprints from experiment configs."""
    print("="*60)
    print("Generating Fingerprints from Experiment Configs")
    print("="*60)
    print(f"Experiment configs: {len(CONFIGS)}")
    print(f"Machine config: {MACHINE_CONFIG}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'all'}")
    print()
    
    # Load machine config
    machine_config_path = Path(MACHINE_CONFIG)
    if not machine_config_path.is_absolute():
        machine_config_path = project_root / machine_config_path
    
    if not machine_config_path.exists():
        print(f"Error: Machine config file not found: {machine_config_path}")
        return
    
    machine_config = load_machine_config(str(machine_config_path))
    
    # Load fingerprint config
    fingerprint_config = load_fingerprint_config(FINGERPRINT_CONFIG)
    
    # Collect all experiments from all config files
    all_experiments = []
    for config_path in CONFIGS:
        config_path_obj = Path(config_path)
        if not config_path_obj.is_absolute():
            config_path_obj = project_root / config_path_obj
        
        if not config_path_obj.exists():
            print(f"Warning: Experiment config not found: {config_path_obj}, skipping...")
            continue
        
        print(f"Loading experiments from: {config_path_obj}")
        experiment_config = load_experiment_config(str(config_path_obj))
        experiments = load_experiments(experiment_config, machine_config)
        all_experiments.extend(experiments)
        print(f"  Found {len(experiments)} experiments")
    
    print(f"\nTotal experiments to process: {len(all_experiments)}\n")
    
    # Prepare dataset configs for fingerprint generation
    dataset_configs = []
    
    # Cache for evaluation datasets - only create each eval dataset once
    eval_dataset_cache = {}
    
    for i, exp_config in enumerate(all_experiments, 1):
        train_dataset = exp_config.get('train_dataset', 'synthetic')
        exp_name = exp_config.get('name_exp', f'exp_{i}')
        
        print(f"[{i}/{len(all_experiments)}] Processing: {exp_name}")
        print(f"  Training dataset: {train_dataset}")
        
        # 1. Create training dataset and generate fingerprint
        try:
            dataset = create_training_dataset_from_exp_config(exp_config, split='train')
            
            # Build dataset config for process_all_datasets
            dataset_config = {
                'name': exp_name,
                'dataset': dataset,
                'max_samples': MAX_SAMPLES,
            }
            
            # For synthetic datasets, always use DataLoader with custom collate_fn
            if train_dataset == 'synthetic':
                dataset_config['use_dataloader'] = True
                dataset_config['batch_size'] = 1
                dataset_config['num_workers'] = 0  # Required for synthetic dataset
                dataset_config['collate_fn'] = dataset.collate_fn
            
            dataset_configs.append(dataset_config)
            print(f"  ✓ Training dataset created")
        
        except Exception as e:
            print(f"  ✗ Failed to create training dataset: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 2. Create evaluation datasets
        eval_benchmarks = exp_config.get('eval_benchmarks', [])
        if not eval_benchmarks:
            print(f"  No evaluation benchmarks specified")
            continue
        
        if not isinstance(eval_benchmarks, list):
            eval_benchmarks = [eval_benchmarks]
        
        print(f"  Evaluation benchmarks: {len(eval_benchmarks)}")
        for benchmark in eval_benchmarks:
            benchmark_str = str(benchmark)
            benchmark_lower = benchmark_str.lower()
            
            try:
                # Check if we've already created this evaluation dataset
                cache_key = benchmark_lower
                if cache_key in eval_dataset_cache:
                    # Reuse cached dataset
                    eval_dataset = eval_dataset_cache[cache_key]
                    print(f"    ✓ {benchmark_str} dataset (reused from cache)")
                else:
                    # Determine split for evaluation dataset
                    # TSS has no splits, FlyingThings uses 'test', others use 'val' by default
                    if benchmark_lower == 'tss':
                        eval_split = None  # TSS doesn't use splits
                    elif benchmark_lower == 'flyingthings':
                        eval_split = 'test'  # FlyingThings uses test split for evaluation
                    else:
                        eval_split = 'val'
                    
                    # Create new dataset and cache it
                    eval_dataset = create_eval_dataset_from_exp_config(exp_config, benchmark_str, split=eval_split)
                    eval_dataset_cache[cache_key] = eval_dataset
                    print(f"    ✓ {benchmark_str} dataset created (cached)")
                
                # Build dataset config for process_all_datasets
                eval_dataset_config = {
                    'name': f"{exp_name}_eval_{benchmark_lower}",
                    'dataset': eval_dataset,
                    'max_samples': MAX_SAMPLES,
                }
                
                # For synthetic datasets (if benchmark is synthetic), use DataLoader
                if benchmark_lower == 'synthetic':
                    eval_dataset_config['use_dataloader'] = True
                    eval_dataset_config['batch_size'] = 1
                    eval_dataset_config['num_workers'] = 0
                    if hasattr(eval_dataset, 'collate_fn'):
                        eval_dataset_config['collate_fn'] = eval_dataset.collate_fn
                
                dataset_configs.append(eval_dataset_config)
            
            except Exception as e:
                print(f"    ✗ Failed to create {benchmark_str} dataset: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if not dataset_configs:
        print("No datasets to process!")
        return
    
    print(f"\n{'='*60}")
    print(f"Generating fingerprints for {len(dataset_configs)} datasets...")
    print(f"{'='*60}\n")
    
    # Generate fingerprints
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    
    results = process_all_datasets(
        dataset_configs=dataset_configs,
        output_dir=str(output_dir),
        config=fingerprint_config,
        generate_plots=GENERATE_PLOTS,
        generate_comparison=GENERATE_COMPARISON,
    )
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Processed {len(dataset_configs)} datasets")
    print(f"Output directory: {results['summary']['output_dir']}")
    print("\nFingerprints saved:")
    for name, path in results['fingerprints'].items():
        print(f"  - {name}: {path}")
    
    if results.get('plots'):
        print("\nPlots saved:")
        for name, path in results['plots'].items():
            print(f"  - {name}: {path}")
    
    if results.get('comparison_plots'):
        print(f"\nComparison plots: {results['comparison_plots']}")


if __name__ == "__main__":
    main()

