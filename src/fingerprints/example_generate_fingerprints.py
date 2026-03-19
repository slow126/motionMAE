"""
example_generate_fingerprints.py
================================
Example script showing how to generate flow fingerprints for all datasets.
Loads configuration from YAML files in src/configs/fingerprints/

Usage:
    python src/fingerprints/example_generate_fingerprints.py --output_dir ./fingerprints
    python src/fingerprints/example_generate_fingerprints.py --dataset flyingthings --split train
    python src/fingerprints/example_generate_fingerprints.py --config src/configs/fingerprints/flyingthings.yaml
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.fingerprints.dataset_fingerprint import (
    process_all_datasets,
    compute_dataset_fingerprint,
    FlowFingerprintConfig,
)
from src.data.synth.datasets.FlyingThingsDataset import FlyingThingsDataset
from src.data.synth.datasets.PointOdysseyCorrespondence import PointOdysseyFlowDataset
from src.data.synth.datasets.OnlineCorrespondenceDataset import OnlineCorrespondenceDataset
from src.data.synth.datasets.KittiDataset import KittiDataset
import models.CATs_PlusPlus.data.download as download


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_fingerprint_config(config_path: Optional[str] = None) -> FlowFingerprintConfig:
    """Load fingerprint configuration from YAML file."""
    if config_path is None:
        config_path = project_root / 'src/configs/fingerprints/fingerprint_config.yaml'
    
    config_dict = load_config(str(config_path))
    fp_config = config_dict.get('fingerprint_config', {})
    
    # Convert spatial_downsample_hw list to tuple
    if 'spatial_downsample_hw' in fp_config:
        fp_config['spatial_downsample_hw'] = tuple(fp_config['spatial_downsample_hw'])
    
    # Ensure numeric values are converted from strings (YAML may parse 1e-3 as string)
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


def load_dataset_config(dataset_name: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load dataset-specific configuration."""
    if config_dir is None:
        config_dir = project_root / 'src/configs/fingerprints'
    
    config_path = config_dir / f"{dataset_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return load_config(str(config_path))


def create_flyingthings_dataset(config: Dict[str, Any], split_override: Optional[str] = None) -> FlyingThingsDataset:
    """Create FlyingThings dataset from config."""
    paths = config.get('paths', {})
    params = config.get('dataset_params', {})
    
    # Override split if provided
    if split_override:
        params['split'] = split_override
    
    # Map split names (train/val/test -> train/val/test)
    split = params.get('split', 'train')
    
    dataset = FlyingThingsDataset(
        root=paths.get('root'),
        split=split,
        transforms=None,
        size=(params.get('size', 512), params.get('size', 512)),
        downsample_flow=params.get('downsample_flow', 32),
        subsample_flow=params.get('subsample_flow', 0.6),
        use_valid_mask=params.get('use_valid_mask', True),
        reverse_flow=params.get('reverse_flow', True),
        filter_out_of_bounds=params.get('filter_out_of_bounds', True),
    )
    return dataset


def create_pointodyssey_dataset(config: Dict[str, Any], split_override: Optional[str] = None) -> PointOdysseyFlowDataset:
    """Create PointOdyssey dataset from config."""
    paths = config.get('paths', {})
    params = config.get('dataset_params', {})
    
    # Override split if provided
    if split_override:
        params['dset'] = split_override
    
    # Map split names
    dset = params.get('dset', 'train')
    
    # Handle resize_size (can be list or computed from crop_size)
    resize_size = params.get('resize_size')
    if isinstance(resize_size, list):
        resize_size = tuple(resize_size)
    elif resize_size is None:
        crop_size = params.get('crop_size', 512)
        if isinstance(crop_size, list):
            crop_size = crop_size[0]
        resize_size = (crop_size + 64, crop_size + 64)
    
    crop_size = params.get('crop_size', 512)
    if isinstance(crop_size, list):
        crop_size = tuple(crop_size)
    elif isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    
    dataset = PointOdysseyFlowDataset(
        dataset_location=paths.get('root'),
        dset=dset,
        use_augs=params.get('use_augs', False),
        S=params.get('S', 8),
        N=params.get('N', 32),
        strides=params.get('strides', [1, 2, 4]),
        quick=params.get('quick', False),
        verbose=params.get('verbose', False),
        resize_size=resize_size,
        crop_size=crop_size,
        filter_instances=params.get('filter_instances', True),
        downsample_for_cats=params.get('downsample_for_cats', True),
        cats_feat_size=params.get('cats_feat_size', 32),
        all_points=params.get('all_points', True),
        max_pts=params.get('max_pts', 40),
        normalize=params.get('normalize', True),
    )
    return dataset


def create_synthetic_dataset(config: Dict[str, Any], split_override: Optional[str] = None) -> OnlineCorrespondenceDataset:
    """Create synthetic dataset from config."""
    paths = config.get('paths', {})
    params = config.get('dataset_params', {})
    
    # Override split if provided
    if split_override:
        params['split'] = split_override
    
    split = params.get('split', 'train')
    
    # Use val geometry config for val split
    if split == 'val':
        geometry_config_path = paths.get('geometry_config_val_path', 
                                         paths.get('geometry_config_path', 
                                                  'src/configs/online_synth_configs/OnlineGeometryConfig_Val.yaml'))
    else:
        geometry_config_path = paths.get('geometry_config_path', 
                                         'src/configs/online_synth_configs/OnlineGeometryConfig.yaml')
    
    processor_config_path = paths.get('processor_config_path', 
                                      'src/configs/online_synth_configs/OnlineProcessorConfig.yaml')
    
    opengl_device_index = params.get('opengl_device_index')
    if opengl_device_index == 'null' or opengl_device_index == '':
        opengl_device_index = None
    
    dataset = OnlineCorrespondenceDataset(
        geometry_config_path=geometry_config_path,
        processor_config_path=processor_config_path,
        split=split,
        opengl_device_index=opengl_device_index,
    )
    
    # Move dataset to CUDA if available (required for collate_fn CUDA kernels)
    if torch.cuda.is_available():
        dataset.cuda()
    
    return dataset


def create_tss_dataset(config: Dict[str, Any], split_override: Optional[str] = None) -> Any:
    """Create TSS dataset from config. TSS has no splits - it's just a benchmark."""
    from src.data.synth.datasets.TSSDataset import TSSDataset
    
    paths = config.get('paths', {})
    params = config.get('dataset_params', {})
    
    root = paths.get('root', '/home/spencer/Data/correspondence/TSS_CVPR2016')
    size = params.get('size', 512)
    feature_size = params.get('feature_size', 32)
    max_pts = params.get('max_pts', 40)
    thres = params.get('thres', 'img')
    augmentation = params.get('augmentation', False)
    sample_keypoints = params.get('sample_keypoints', True)
    
    device = 'cpu'  # Use CPU for fingerprinting
    
    dataset = TSSDataset(
        root=root,
        device=device,
        size=size,
        feature_size=feature_size,
        max_pts=max_pts,
        thres=thres,
        augmentation=augmentation,
        sample_keypoints=sample_keypoints,
    )
    return dataset


def create_kitti_dataset(dataset_name: str, config: Dict[str, Any], split_override: Optional[str] = None) -> KittiDataset:
    """Create KITTI dataset from config."""
    paths = config.get('paths', {})
    params = config.get('dataset_params', {})
    
    # Determine version from dataset name
    version = '2012' if '2012' in dataset_name else '2015'
    
    # Get root path - can be full path or relative to kitti_root
    root = paths.get('root')
    if root is None:
        # Try to construct from kitti_root if available
        kitti_root = paths.get('kitti_root', '/home/spencer/Data/correspondence/kitti')
        root = os.path.join(kitti_root, f'kitti-{version}')
    
    # Determine split
    if split_override:
        split = split_override
    else:
        split = params.get('split', 'val')
    
    # Handle kitti_val_use_full_training option
    if params.get('kitti_val_use_full_training', False) and split == 'val':
        split = 'training'  # Use full training set for validation
    
    # Get parameters
    size = params.get('size', 512)
    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, list):
        size = tuple(size)
    
    feature_size = params.get('feature_size', 32)
    occ_type = params.get('occ_type', 'occ')
    max_pts = params.get('max_pts', 200)
    thres = params.get('thres', 'img')
    normalize = params.get('normalize', True)
    normalize_images = params.get('normalize_images', True)  # Use validation format for fingerprinting
    
    dataset = KittiDataset(
        root=root,
        split=split,
        version=version,
        occ_type=occ_type,
        size=size,
        downsample_flow=feature_size,
        normalize=normalize,
        normalize_images=normalize_images,
        thres=thres,
        max_pts=max_pts,
    )
    return dataset


def create_real_dataset(dataset_name: str, config: Dict[str, Any], split_override: Optional[str] = None) -> Any:
    """Create real dataset (SPair, PFPascal, etc.) from config."""
    paths = config.get('paths', {})
    params = config.get('dataset_params', {})
    
    # Override split if provided
    if split_override:
        # Map train -> trn for real datasets
        if split_override == 'train':
            split = 'trn'
        elif split_override == 'val':
            split = 'val'
        elif split_override == 'test':
            split = 'test'
        else:
            split = split_override
    else:
        split = params.get('split', 'val')
        # Map train -> trn
        if split == 'train':
            split = 'trn'
    
    datapath = paths.get('datapath', './models/Datasets_CATs')
    thres = params.get('thres', 'img')
    feature_size = params.get('feature_size', 32)
    device = torch.device('cpu')  # Use CPU for fingerprinting
    
    # Download if needed
    try:
        download.download_dataset(datapath, dataset_name)
    except:
        pass  # May already exist
    
    dataset = download.load_dataset(
        dataset_name, datapath, thres, device, split, 
        params.get('augmentation', False), feature_size
    )
    return dataset


def get_available_datasets(config_dir: Optional[Path] = None) -> list:
    """Get list of available datasets from datasets.yaml."""
    if config_dir is None:
        config_dir = project_root / 'src/configs/fingerprints'
    
    datasets_config_path = config_dir / 'datasets.yaml'
    if datasets_config_path.exists():
        config = load_config(str(datasets_config_path))
        return config.get('available_datasets', [])
    else:
        # Fallback to default list
        return ['flyingthings', 'pointodyssey', 'spair', 'pfpascal', 'pfwillow']


def main():
    parser = argparse.ArgumentParser(
        description='Generate flow fingerprints for datasets using YAML configs'
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to fingerprint config YAML (default: src/configs/fingerprints/fingerprint_config.yaml)'
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        default=None,
        help='Directory containing dataset configs (default: src/configs/fingerprints)'
    )
    
    # Dataset selection
    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        help='Dataset to process (or "all" for all datasets). Options: flyingthings, pointodyssey, spair, pfpascal, pfwillow, tss, kitti2012, kitti2015'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        choices=['train', 'val', 'test'],
        help='Dataset split (overrides config file)'
    )
    
    # Path overrides (override config file values)
    parser.add_argument(
        '--flyingthings_root',
        type=str,
        default=None,
        help='Path to FlyingThings3D dataset (overrides config)'
    )
    parser.add_argument(
        '--pointodyssey_root',
        type=str,
        default=None,
        help='Path to PointOdyssey dataset (overrides config)'
    )
    parser.add_argument(
        '--datapath',
        type=str,
        default=None,
        help='Path to real datasets (overrides config)'
    )
    
    # Processing options (override config file values)
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for fingerprints and plots (overrides config)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples per dataset (None = all, overrides config)'
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Skip generating plots (overrides config)'
    )
    parser.add_argument(
        '--no_comparison',
        action='store_true',
        help='Skip generating comparison plots (overrides config)'
    )
    parser.add_argument(
        '--use_dataloader',
        action='store_true',
        help='Use DataLoader (overrides config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size if using DataLoader (overrides config)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of DataLoader workers (overrides config)'
    )
    parser.add_argument(
        '--min_flow_length',
        type=float,
        default=None,
        help='Minimum flow vector length for flow filtering (overrides config)'
    )
    parser.add_argument(
        '--max_flow_length',
        type=float,
        default=None,
        help='Maximum flow vector length for flow filtering (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Determine config directory
    if args.config_dir:
        config_dir = Path(args.config_dir)
    else:
        config_dir = project_root / 'src/configs/fingerprints'
    
    # Load fingerprint config
    fingerprint_config = load_fingerprint_config(args.config)
    
    # Load datasets config for defaults
    datasets_config_path = config_dir / 'datasets.yaml'
    if datasets_config_path.exists():
        datasets_config = load_config(str(datasets_config_path))
        default_paths = datasets_config.get('default_paths', {})
        default_processing = datasets_config.get('default_processing', {})
        default_output = datasets_config.get('default_output', {})
    else:
        default_paths = {}
        default_processing = {}
        default_output = {'output_dir': './fingerprints', 'generate_plots': True, 'generate_comparison': True}
    
    # Determine which datasets to process
    if args.dataset == 'all':
        datasets_to_process = get_available_datasets(config_dir)
    else:
        datasets_to_process = [args.dataset]
    
    # Build dataset configs
    dataset_configs = []
    
    for ds_name in datasets_to_process:
        try:
            # Load dataset-specific config
            ds_config = load_dataset_config(ds_name, config_dir)
            dataset_type = ds_config.get('dataset_type')
            
            # Override paths from command line
            if 'paths' in ds_config:
                if args.flyingthings_root and ds_name == 'flyingthings':
                    ds_config['paths']['root'] = args.flyingthings_root
                if args.pointodyssey_root and ds_name == 'pointodyssey':
                    ds_config['paths']['root'] = args.pointodyssey_root
                if args.datapath and dataset_type == 'real':
                    ds_config['paths']['datapath'] = args.datapath
            
            # Determine split to use (before creating dataset)
            # TSS and KITTI are special cases
            if ds_name == 'tss':
                # TSS doesn't use splits - ignore split parameter
                actual_split = None
                print(f"TSS dataset (no splits - benchmark dataset)")
            elif ds_name in ['kitti2012', 'kitti2015']:
                # KITTI can use 'training' split if kitti_val_use_full_training is True
                # Priority: command line arg > config file > default
                if args.split:
                    actual_split = args.split
                else:
                    actual_split = ds_config.get('dataset_params', {}).get('split', 'val')
                print(f"Using split: '{actual_split}' (from {'command line' if args.split else 'config file'})")
            else:
                # Priority: command line arg > config file > default
                if args.split:
                    actual_split = args.split
                else:
                    actual_split = ds_config.get('dataset_params', {}).get('split', 'train')
                print(f"Using split: '{actual_split}' (from {'command line' if args.split else 'config file'})")
            
            # Create dataset with the determined split
            if dataset_type == 'flyingthings':
                dataset = create_flyingthings_dataset(ds_config, actual_split)
            elif dataset_type == 'pointodyssey':
                dataset = create_pointodyssey_dataset(ds_config, actual_split)
            elif dataset_type == 'synthetic':
                dataset = create_synthetic_dataset(ds_config, actual_split)
            elif dataset_type == 'real':
                # TSS and KITTI are special cases
                if ds_name == 'tss':
                    dataset = create_tss_dataset(ds_config, actual_split)
                elif ds_name in ['kitti2012', 'kitti2015']:
                    dataset = create_kitti_dataset(ds_name, ds_config, actual_split)
                else:
                    dataset = create_real_dataset(ds_name, ds_config, actual_split)
            else:
                print(f"Skipping unknown dataset type: {dataset_type} for {ds_name}")
                continue
            
            # Get processing options (merge config with command line overrides)
            processing = ds_config.get('processing', default_processing.copy())
            if args.max_samples is not None:
                processing['max_samples'] = args.max_samples
            if args.batch_size is not None:
                processing['batch_size'] = args.batch_size
            if args.num_workers is not None:
                processing['num_workers'] = args.num_workers
            
            # Create flow filter if parameters are provided
            flow_filter = None
            if args.min_flow_length is not None or args.max_flow_length is not None:
                from src.data.synth.datasets.flow_filter import FlowLengthFilter
                flow_filter = FlowLengthFilter(
                    min_flow_length=args.min_flow_length,
                    max_flow_length=args.max_flow_length
                )
                print(f"  Flow filtering enabled: min={args.min_flow_length}, max={args.max_flow_length}")
            elif processing.get('min_flow_length') is not None or processing.get('max_flow_length') is not None:
                from src.data.synth.datasets.flow_filter import FlowLengthFilter
                flow_filter = FlowLengthFilter(
                    min_flow_length=processing.get('min_flow_length'),
                    max_flow_length=processing.get('max_flow_length')
                )
                print(f"  Flow filtering enabled (from config): min={processing.get('min_flow_length')}, max={processing.get('max_flow_length')}")
            
            if flow_filter is not None:
                processing['flow_filter'] = flow_filter
                        
            # For synthetic datasets, always use DataLoader with custom collate_fn
            if dataset_type == 'synthetic':
                processing['use_dataloader'] = True
                # Synthetic datasets require num_workers=0 (GPU-bound rendering)
                processing['num_workers'] = 0
                # Use batch_size from config or default to 1
                if processing.get('batch_size') is None:
                    processing['batch_size'] = 1
                # Store the collate_fn for DataLoader creation
                processing['collate_fn'] = dataset.collate_fn
            elif args.use_dataloader:
                processing['use_dataloader'] = True
            
            # Determine split name for output (use the actual split that was used)
            # TSS has no splits - use 'val' for benchmark naming consistency
            if ds_name == 'tss':
                split_name = 'val'  # TSS is a benchmark, use 'val' for consistency
            elif ds_name in ['kitti2012', 'kitti2015']:
                # KITTI: if using 'training' split, use 'val' for output naming (it's validation data)
                if actual_split == 'training':
                    split_name = 'val'
                else:
                    split_name = actual_split
            elif dataset_type == 'real' and actual_split == 'train':
                # Keep as 'train' for output naming (even though dataset uses 'trn' internally)
                split_name = 'train'
            else:
                split_name = actual_split
            
            # Build config dict for process_all_datasets
            dataset_config_dict = {
                'name': f"{ds_name}_{split_name}",
                'dataset': dataset,
                **processing,
            }
            
            dataset_configs.append(dataset_config_dict)
            
            print(f"✓ Added dataset: {ds_name} ({split_name})")
        
        except Exception as e:
            print(f"✗ Failed to create dataset {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not dataset_configs:
        print("No datasets to process!")
        return
    
    # Get output options
    output_dir = args.output_dir or default_output.get('output_dir', './fingerprints')
    generate_plots = not args.no_plots if args.no_plots else default_output.get('generate_plots', True)
    generate_comparison = not args.no_comparison if args.no_comparison else default_output.get('generate_comparison', True)
    
    # Process all datasets
    results = process_all_datasets(
        dataset_configs=dataset_configs,
        output_dir=output_dir,
        config=fingerprint_config,
        generate_plots=generate_plots,
        generate_comparison=generate_comparison,
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
