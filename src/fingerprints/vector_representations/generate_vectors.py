"""
generate_vectors.py
===================
Generate flow vectors from SLURM experiment configs.

Reads in a list of experiment config files, expands them into individual experiments,
creates datasets for both training and evaluation benchmarks, and generates vectors.

Usage:
    python src/fingerprints/vector_representations/generate_vectors.py
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.fingerprints.vector_representations.flow_vector import FlowVector, FlowVectorConfig
from src.fingerprints.vector_representations.vector_utils import save_vector_coverage
from src.fingerprints.dataset_fingerprint import (
    convert_flow_to_numpy,
    extract_valid_mask,
)
from src.data.synth.datasets.FlyingThingsDataset import FlyingThingsDataset
from src.data.synth.datasets.PointOdysseyCorrespondence import PointOdysseyFlowDataset
from src.data.synth.datasets.OnlineCorrespondenceDataset import OnlineCorrespondenceDataset
from src.data.synth.datasets.KittiDataset import KittiDataset
import models.CATs_PlusPlus.data.download as download

# Import SLURM generator functions
from slurm.generate_jobs import load_experiments, load_machine_config, load_experiment_config

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ============================================================================
# CONFIGURATION - Set your experiment configs here
# ============================================================================
CONFIGS = [
    'slurm/experiment_configs/fingerprints.yaml',
    # Add more config paths as needed
]

MACHINE_CONFIG = 'slurm/machine_configs/local.yaml'
OUTPUT_DIR = './vector_coverage'
VECTOR_CONFIG = None  # None = use default config
MAX_SAMPLES = 1000  # None = process all samples
TRAIN_SAMPLE_FRACTION = 0.1  # Fraction of training frames to sample (for large datasets)


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


def load_vector_config(config_path: Optional[str] = None) -> FlowVectorConfig:
    """Load vector coverage configuration from YAML file."""
    if config_path is None:
        config_path = project_root / 'src/configs/fingerprints/vector_config.yaml'
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = project_root / config_path
    
    if not config_path.exists():
        # Use default config if file doesn't exist
        return FlowVectorConfig()
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    vc_config = config_dict.get('vector_config', {})
    
    # Convert cell_size list to tuple
    if 'cell_size' in vc_config and vc_config['cell_size'] is not None:
        vc_config['cell_size'] = tuple(vc_config['cell_size'])
    
    # Ensure numeric values are converted from strings
    numeric_fields = [
        'random_fraction',
        'max_vectors_per_frame',
        'max_vectors_total',
        'random_seed',
    ]
    for field in numeric_fields:
        if field in vc_config:
            vc_config[field] = float(vc_config[field]) if vc_config[field] is not None else None
            if field == 'random_seed':
                vc_config[field] = int(vc_config[field])
    
    # Validate normalize_flow value
    if 'normalize_flow' in vc_config:
        valid_modes = ['global', 'local', 'none']
        if vc_config['normalize_flow'] not in valid_modes:
            raise ValueError(f"normalize_flow must be one of {valid_modes}, got {vc_config['normalize_flow']}")
    
    return FlowVectorConfig(**vc_config)


# ============================================================================
# Dataset Creation (reuse from make_fingerprints_from_config.py)
# ============================================================================

def create_training_dataset_from_exp_config(exp_config: Dict[str, Any], split: str = 'train') -> Any:
    """Create a training dataset from experiment config based on train_dataset type."""
    train_dataset = exp_config.get('train_dataset', 'synthetic')
    
    if train_dataset == 'synthetic':
        geometry_config_overrides = build_geometry_config_overrides(exp_config)
        
        geometry_config_path = exp_config.get('geometry_config_path', 
                                             'src/configs/online_synth_configs/OnlineGeometryConfig.yaml')
        processor_config_path = exp_config.get('processor_config_path',
                                              'src/configs/online_synth_configs/OnlineProcessorConfig.yaml')
        
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
        
        if torch.cuda.is_available():
            dataset.cuda()
        
        return dataset
    
    elif train_dataset == 'flyingthings':
        root = exp_config.get('flyingthings_root', '')
        size = exp_config.get('size', 512)
        downsample_flow = exp_config.get('feature_size', 32)
        normalize = exp_config.get('normalize', True)
        
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
            normalize=True,
        )
        return dataset
    
    elif train_dataset == 'pointodyssey':
        root = exp_config.get('pointodyssey_root', '')
        size = exp_config.get('size', 512)
        feature_size = exp_config.get('feature_size', 32)
        num_pts = exp_config.get('num_pts_to_track_pointodyssey', 32)
        normalize = exp_config.get('normalize', True)
        
        dset = split
        
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
    
    elif train_dataset in ['kitti2012', 'kitti2015']:
        version = '2012' if '2012' in train_dataset else '2015'
        
        if split == 'val' and exp_config.get('kitti_val_use_full_training', False):
            kitti_root = exp_config.get('kitti_unsplit_root', '/home/spencer/Data/correspondence/kitti')
            split = 'training'
        else:
            kitti_root = exp_config.get('kitti_root', '/home/spencer/Data/correspondence/kitti-split')
        
        root = os.path.join(kitti_root, f'kitti-{version}')
        
        size = exp_config.get('size', 512)
        if isinstance(size, int):
            size = (size, size)
        
        feature_size = exp_config.get('feature_size', 32)
        normalize = exp_config.get('normalize', True)
        
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
    
    elif train_dataset in ['spair', 'pfpascal', 'pfwillow']:
        datapath = exp_config.get('datapath', './models/Datasets_CATs')
        thres = exp_config.get('thres', 'img')
        feature_size = exp_config.get('feature_size', 32)
        normalize = exp_config.get('normalize', 'imagenet')
        if normalize is False:
            normalize = None
        device = torch.device('cpu')
        
        if split == 'train':
            dataset_split = 'trn'
        elif split == 'val':
            dataset_split = 'val'
        elif split == 'test':
            dataset_split = 'test'
        else:
            dataset_split = 'trn'
        
        try:
            download.download_dataset(datapath, train_dataset)
        except:
            pass
        
        dataset = download.load_dataset(
            train_dataset, datapath, thres, device, dataset_split,
            False,
            feature_size
        )
        return dataset
    
    else:
        raise ValueError(f"Unknown train_dataset: {train_dataset}")


def create_eval_dataset_from_exp_config(exp_config: Dict[str, Any], benchmark_name: str, split: str = 'val') -> Any:
    """Create an evaluation dataset from experiment config for a specific benchmark."""
    benchmark_lower = benchmark_name.lower()
    
    if benchmark_lower == 'tss':
        from src.data.synth.datasets.TSSDataset import TSSDataset
        
        tss_root = exp_config.get('tss_root', '/home/spencer/Data/correspondence/TSS_CVPR2016')
        size = exp_config.get('size', 512)
        feature_size = exp_config.get('feature_size', 32)
        normalize = exp_config.get('normalize', True)
        
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
        # TSSDataset doesn't have normalize parameter, so modify transform after creation
        if not normalize:
            from torchvision import transforms
            dataset.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ])
        return dataset
    
    elif benchmark_lower in ['spair', 'pfpascal', 'pfwillow']:
        datapath = exp_config.get('datapath', './models/Datasets_CATs')
        thres = exp_config.get('thres', 'img')
        feature_size = exp_config.get('feature_size', 32)
        normalize = exp_config.get('normalize', 'imagenet')
        if normalize is False:
            normalize = None
        device = torch.device('cpu')
        
        if split == 'train':
            split = 'trn'
        elif split == 'val':
            split = 'val'
        elif split == 'test':
            split = 'test'
        
        try:
            download.download_dataset(datapath, benchmark_lower)
        except:
            pass
        
        dataset = download.load_dataset(
            benchmark_lower, datapath, thres, device, split,
            False,
            feature_size
        )
        return dataset
    
    elif benchmark_lower in ['kitti2012', 'kitti2015']:
        version = '2012' if '2012' in benchmark_lower else '2015'
        
        if exp_config.get('kitti_val_use_full_training', False):
            kitti_root = exp_config.get('kitti_unsplit_root', '/home/spencer/Data/correspondence/kitti')
            split = 'training'
        else:
            kitti_root = exp_config.get('kitti_root', '/home/spencer/Data/correspondence/kitti-split')
            split = 'val'
        
        root = os.path.join(kitti_root, f'kitti-{version}')
        
        size = exp_config.get('size', 512)
        if isinstance(size, int):
            size = (size, size)
        
        feature_size = exp_config.get('feature_size', 32)
        normalize = exp_config.get('normalize', True)
        
        dataset = KittiDataset(
            root=root,
            split=split,
            version=version,
            occ_type='occ',
            size=size,
            downsample_flow=feature_size,
            normalize=normalize,
            normalize_images=normalize,
            thres='img',
            max_pts=200,
        )
        return dataset
    
    elif benchmark_lower == 'pointodyssey':
        root = exp_config.get('pointodyssey_root', '')
        size = exp_config.get('size', 512)
        feature_size = exp_config.get('feature_size', 32)
        num_pts = exp_config.get('num_pts_to_track_pointodyssey', 32)
        
        dset = 'val'
        
        resize_size = (size + 64, size + 64)
        crop_size = (size, size)
        normalize = exp_config.get('normalize', True)
        
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
    
    elif benchmark_lower == 'flyingthings':
        root = exp_config.get('flyingthings_root', '')
        size = exp_config.get('size', 512)
        downsample_flow = exp_config.get('feature_size', 32)
        normalize = exp_config.get('normalize', True)
        
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
            normalize=True,
        )
        return dataset
    
    else:
        raise ValueError(f"Unknown evaluation benchmark: {benchmark_name}")


# ============================================================================
# Dataset Processing Functions
# ============================================================================

def compute_vector_coverage(
    dataset,
    config: FlowVectorConfig,
    dataset_name: str = "unknown",
    max_samples: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    progress: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compute vector coverage for a dataset.
    
    Automatically detects synthetic datasets (those with collate_fn) and calls collate_fn directly.
    
    Args:
        dataset: PyTorch Dataset
        config: FlowVectorConfig
        dataset_name: Name of dataset
        max_samples: Maximum number of samples to process
        sample_fraction: Fraction of frames to randomly sample (for large datasets)
        progress: Show progress bar
        **kwargs: Additional kwargs
    
    Returns:
        Dictionary with vector coverage results
    """
    # Initialize accumulator
    flow_vector = FlowVector(config)
    
    # Auto-detect synthetic datasets (those with collate_fn method)
    has_collate_fn = hasattr(dataset, 'collate_fn') and callable(getattr(dataset, 'collate_fn', None))
    
    # Determine dataset length
    if hasattr(dataset, '__len__'):
        total_samples = len(dataset)
    else:
        total_samples = max_samples or 1000  # Estimate
    
    # Frame-level random sampling for large datasets
    if sample_fraction is not None and sample_fraction < 1.0:
        n_samples = int(total_samples * sample_fraction)
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)
        frame_indices = np.random.choice(total_samples, size=n_samples, replace=False)
        frame_indices = sorted(frame_indices)
        print(f"  Randomly sampling {n_samples} frames from {total_samples} total")
    else:
        if max_samples is not None:
            frame_indices = list(range(min(total_samples, max_samples)))
        else:
            frame_indices = list(range(total_samples))
    
    # Progress bar
    if progress:
        pbar = tqdm(total=len(frame_indices), desc=f"Processing {dataset_name}")
    
    samples_processed = 0
    errors = []
    
    try:
        for idx in frame_indices:
            try:
                # Get raw sample from dataset
                raw_sample = dataset[idx]
                
                # If dataset has process_sample method, use it (for single sample processing)
                if hasattr(dataset, 'process_sample'):
                    sample = dataset.process_sample(raw_sample)
                elif has_collate_fn:
                    # Fallback: collate_fn expects a list of samples (adds batch dim)
                    batch_result = dataset.collate_fn([raw_sample])
                    # Remove batch dimension from all tensors
                    sample = {}
                    for key, value in batch_result.items():
                        if isinstance(value, torch.Tensor):
                            if value.dim() > 0:
                                sample[key] = value.squeeze(0)
                            else:
                                sample[key] = value
                        else:
                            sample[key] = value
                else:
                    # Use sample directly
                    sample = raw_sample
                
                if 'flow' not in sample:
                    if progress:
                        pbar.update(1)
                    continue
                
                # Convert flow to numpy
                flow = convert_flow_to_numpy(sample['flow'])
                valid_mask = extract_valid_mask(sample)
                
                # Add frame
                flow_vector.add_frame(
                    flow=flow,
                    valid_mask=valid_mask,
                    dataset_label=dataset_name,
                    frame_idx=idx,
                )
                
                samples_processed += 1
                if progress:
                    pbar.update(1)
                    
            except Exception as e:
                errors.append(f"Frame {idx}: {str(e)}")
                if progress:
                    pbar.update(1)
                continue
    
    finally:
        if progress:
            pbar.close()
    
    # Finalize
    result = flow_vector.finalize()
    result['metadata'] = {
        'dataset_name': dataset_name,
        'samples_processed': samples_processed,
        'total_samples_available': total_samples,
        'errors': errors,
    }
    
    return result


def compute_vector_coverage_all_datasets(
    dataset_configs: List[Dict[str, Any]],
    config: FlowVectorConfig,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compute vector coverage for multiple datasets.
    
    Args:
        dataset_configs: List of dicts with 'name' and 'dataset' keys
        config: FlowVectorConfig
        output_dir: Optional directory to save results
        **kwargs: Additional kwargs
    
    Returns:
        Dictionary with results for all datasets
    """
    # Single accumulator for all datasets (for global normalization)
    if config.normalize_flow == 'global':
        # Create a temporary FlowVector for collecting flows and computing stats
        stats_flow_vector = FlowVector(config)
        # Store FlowVector instances per dataset for vector extraction
        dataset_flow_vectors = {}
        all_results = {}
        
        # Single pass: extract vectors (with placeholder stats) and collect flows for global stats
        print("Processing datasets (extracting vectors and collecting flows for global statistics)...")
        for ds_config in dataset_configs:
            name = ds_config['name']
            dataset = ds_config['dataset']
            max_samples = ds_config.get('max_samples', kwargs.get('max_samples'))
            sample_fraction = ds_config.get('sample_fraction', kwargs.get('sample_fraction'))
            
            # Create FlowVector for this dataset (will extract vectors with placeholder stats)
            flow_vector = FlowVector(config)
            # Use placeholder stats (1.0, 1.0) - we'll re-normalize after computing global stats
            flow_vector.s_u = 1.0
            flow_vector.s_v = 1.0
            flow_vector.flow_stats_computed = True  # Prevent it from computing its own stats
            dataset_flow_vectors[name] = flow_vector
            
            # Auto-detect synthetic datasets (those with collate_fn method)
            has_collate_fn = hasattr(dataset, 'collate_fn') and callable(getattr(dataset, 'collate_fn', None))
            
            # Determine frame indices
            if hasattr(dataset, '__len__'):
                total_samples = len(dataset)
            else:
                total_samples = max_samples or 1000
            
            if sample_fraction is not None and sample_fraction < 1.0:
                n_samples = int(total_samples * sample_fraction)
                if max_samples is not None:
                    n_samples = min(n_samples, max_samples)
                frame_indices = np.random.choice(total_samples, size=n_samples, replace=False)
                frame_indices = sorted(frame_indices)
            else:
                if max_samples is not None:
                    frame_indices = list(range(min(total_samples, max_samples)))
                else:
                    frame_indices = list(range(total_samples))
            
            samples_processed = 0
            errors = []
            
            print(f"Processing {name}...")
            pbar = tqdm(total=len(frame_indices), desc=f"  {name}")
            try:
                for idx in frame_indices:
                    try:
                        # Get raw sample from dataset
                        raw_sample = dataset[idx]
                        
                        # If dataset has process_sample method, use it (for single sample processing)
                        if hasattr(dataset, 'process_sample'):
                            sample = dataset.process_sample(raw_sample)
                        elif has_collate_fn:
                            # Fallback: collate_fn expects a list of samples (adds batch dim)
                            batch_result = dataset.collate_fn([raw_sample])
                            # Remove batch dimension from all tensors
                            sample = {}
                            for key, value in batch_result.items():
                                if isinstance(value, torch.Tensor):
                                    if value.dim() > 0:
                                        sample[key] = value.squeeze(0)
                                    else:
                                        sample[key] = value
                                else:
                                    sample[key] = value
                        else:
                            sample = raw_sample
                        
                        if 'flow' not in sample:
                            pbar.update(1)
                            continue
                        
                        flow = convert_flow_to_numpy(sample['flow'])
                        valid_mask = extract_valid_mask(sample)
                        
                        # Extract vectors (with placeholder normalization)
                        flow_vector.add_frame(
                            flow=flow,
                            valid_mask=valid_mask,
                            dataset_label=name,
                            frame_idx=idx,
                        )
                        
                        # Also collect flows for global stats computation
                        stats_flow_vector._temp_flows.append(flow)
                        stats_flow_vector._temp_masks.append(valid_mask)
                        
                        samples_processed += 1
                        pbar.update(1)
                    except Exception as e:
                        errors.append(f"Frame {idx}: {str(e)}")
                        pbar.update(1)
                        continue
            finally:
                pbar.close()
            
            # Store metadata for this dataset
            dataset_flow_vectors[name]._metadata = {
                'dataset_name': name,
                'samples_processed': samples_processed,
                'total_samples_available': total_samples,
                'errors': errors,
            }
        
        # Compute global stats from all collected flows
        print("\nComputing global flow statistics...")
        stats_flow_vector.compute_flow_stats()
        s_u_global = stats_flow_vector.s_u
        s_v_global = stats_flow_vector.s_v
        print(f"Global flow stats: s_u={s_u_global:.2f}, s_v={s_v_global:.2f}")
        
        # Re-normalize all vectors with the computed global stats
        print("\nRe-normalizing vectors with global statistics...")
        for name, flow_vector in dataset_flow_vectors.items():
            # Re-normalize vectors: divide u and v components by global stats
            # Vectors were extracted with s_u=1.0, s_v=1.0, so u_norm = u, v_norm = v
            # Now we need: u_norm_new = u / s_u_global, v_norm_new = v / s_v_global
            if flow_vector.vectors:
                vectors_array = np.array(flow_vector.vectors, dtype=np.float32)
                # vectors_array shape: [N, 4] where columns are [x_norm, y_norm, u_norm, v_norm]
                # Re-normalize u and v components (indices 2 and 3)
                vectors_array[:, 2] = vectors_array[:, 2] / s_u_global
                vectors_array[:, 3] = vectors_array[:, 3] / s_v_global
                flow_vector.vectors = vectors_array.tolist()
            
            # Update stats in flow_vector for finalize()
            flow_vector.s_u = s_u_global
            flow_vector.s_v = s_v_global
            
            # Finalize and save results
            result = flow_vector.finalize()
            result['metadata'] = flow_vector._metadata
            all_results[name] = result
        
    else:
        # Per-dataset normalization
        all_results = {}
        for ds_config in dataset_configs:
            name = ds_config['name']
            dataset = ds_config['dataset']
            result = compute_vector_coverage(
                dataset=dataset,
                config=config,
                dataset_name=name,
                max_samples=ds_config.get('max_samples', kwargs.get('max_samples')),
                sample_fraction=ds_config.get('sample_fraction', kwargs.get('sample_fraction')),
                **kwargs,
            )
            all_results[name] = result
    
    # Save results
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, result in all_results.items():
            output_path = output_dir / f"{name}_vector_coverage.json"
            save_vector_coverage(output_path, result)
            print(f"Saved: {output_path}")
    
    return all_results


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to generate vectors from experiment configs."""
    print("="*60)
    print("Generating Flow Vectors from Experiment Configs")
    print("="*60)
    print(f"Experiment configs: {len(CONFIGS)}")
    print(f"Machine config: {MACHINE_CONFIG}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'all'}")
    print(f"Train sample fraction: {TRAIN_SAMPLE_FRACTION}")
    print()
    
    # Load machine config
    machine_config_path = Path(MACHINE_CONFIG)
    if not machine_config_path.is_absolute():
        machine_config_path = project_root / machine_config_path
    
    if not machine_config_path.exists():
        print(f"Error: Machine config file not found: {machine_config_path}")
        return
    
    machine_config = load_machine_config(str(machine_config_path))
    
    # Load vector config
    vector_config = load_vector_config(VECTOR_CONFIG)
    
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
    
    # Prepare dataset configs for vector generation
    dataset_configs = []
    eval_dataset_cache = {}
    
    for i, exp_config in enumerate(all_experiments, 1):
        train_dataset = exp_config.get('train_dataset', 'synthetic')
        exp_name = exp_config.get('name_exp', f'exp_{i}')
        
        print(f"[{i}/{len(all_experiments)}] Processing: {exp_name}")
        print(f"  Training dataset: {train_dataset}")
        
        # 1. Create training dataset
        try:
            dataset = create_training_dataset_from_exp_config(exp_config, split='train')
            
            dataset_config = {
                'name': exp_name,
                'dataset': dataset,
                'max_samples': MAX_SAMPLES,
                'sample_fraction': TRAIN_SAMPLE_FRACTION,  # Random frame sampling for training
            }
            
            if train_dataset == 'synthetic':
                dataset_config['use_dataloader'] = True
                dataset_config['batch_size'] = 1
                dataset_config['num_workers'] = 0
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
                cache_key = benchmark_lower
                if cache_key in eval_dataset_cache:
                    eval_dataset = eval_dataset_cache[cache_key]
                    print(f"    ✓ {benchmark_str} dataset (reused from cache)")
                else:
                    if benchmark_lower == 'tss':
                        eval_split = None
                    elif benchmark_lower == 'flyingthings':
                        eval_split = 'test'
                    else:
                        eval_split = 'val'
                    
                    eval_dataset = create_eval_dataset_from_exp_config(exp_config, benchmark_str, split=eval_split)
                    eval_dataset_cache[cache_key] = eval_dataset
                    print(f"    ✓ {benchmark_str} dataset created (cached)")
                
                eval_dataset_config = {
                    'name': f"{exp_name}_eval_{benchmark_lower}",
                    'dataset': eval_dataset,
                    'max_samples': MAX_SAMPLES,
                    'sample_fraction': None,  # Process all eval frames (they're small)
                }
                
                # No need to set use_dataloader - auto-detected in compute_vector_coverage
                
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
    print(f"Generating vectors for {len(dataset_configs)} datasets...")
    print(f"{'='*60}\n")
    
    # Generate vectors using the vector coverage module
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    
    # Process datasets
    results = compute_vector_coverage_all_datasets(
        dataset_configs=dataset_configs,
        config=vector_config,
        output_dir=output_dir,
    )
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Processed {len(dataset_configs)} datasets")
    print(f"Output directory: {output_dir}")
    print("\nVector files saved:")
    for json_file in output_dir.glob("*_vector_coverage.json"):
        print(f"  - {json_file.name}")


if __name__ == "__main__":
    main()

