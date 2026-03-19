"""
vector_utils.py
===============
Reusable utility functions and I/O for flow vector representations.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import numpy as np
import json


# ------------------------------
# Helper Functions
# ------------------------------

def _magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute flow magnitude."""
    return np.hypot(u, v)


def _angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute flow angle."""
    return np.arctan2(v, u)


def _compute_flow_stats(flows: List[np.ndarray], valid_masks: List[np.ndarray], 
                        method: str = 'p95') -> Tuple[float, float]:
    """
    Compute normalization statistics for flow components.
    
    Args:
        flows: List of flow arrays [H, W, 2]
        valid_masks: List of valid masks [H, W]
        method: 'std', 'p95', 'p99', 'max'
    
    Returns:
        (s_u, s_v): Normalization scales for u and v components
    """
    all_u = []
    all_v = []
    
    for flow, mask in zip(flows, valid_masks):
        if mask is None:
            mask = np.ones(flow.shape[:2], dtype=bool)
        
        u = flow[..., 0][mask]
        v = flow[..., 1][mask]
        
        # Filter out inf/nan - invalid vector handling
        u_valid = u[np.isfinite(u)]
        v_valid = v[np.isfinite(v)]
        
        all_u.append(u_valid)
        all_v.append(v_valid)
    
    if not all_u:
        return 1.0, 1.0
    
    all_u = np.concatenate(all_u)
    all_v = np.concatenate(all_v)
    
    if method == 'std':
        s_u = np.std(all_u) if len(all_u) > 0 else 1.0
        s_v = np.std(all_v) if len(all_v) > 0 else 1.0
    elif method == 'p95':
        s_u = np.percentile(np.abs(all_u), 95) if len(all_u) > 0 else 1.0
        s_v = np.percentile(np.abs(all_v), 95) if len(all_v) > 0 else 1.0
    elif method == 'p99':
        s_u = np.percentile(np.abs(all_u), 99) if len(all_u) > 0 else 1.0
        s_v = np.percentile(np.abs(all_v), 99) if len(all_v) > 0 else 1.0
    elif method == 'max':
        s_u = np.max(np.abs(all_u)) if len(all_u) > 0 else 1.0
        s_v = np.max(np.abs(all_v)) if len(all_v) > 0 else 1.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Avoid division by zero
    s_u = max(s_u, 1e-6)
    s_v = max(s_v, 1e-6)
    
    return s_u, s_v


def _subsample_pixels(
    H: int, 
    W: int, 
    strategy: str,
    cell_size: Optional[Tuple[int, int]] = None,
    random_fraction: float = 0.1,
    max_vectors: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate pixel coordinates for subsampling.
    
    Args:
        H, W: Image dimensions
        strategy: 'all', 'cell_centers', 'random', 'sparse'
        cell_size: (h_cell, w_cell) for cell_centers
        random_fraction: Fraction to keep for random
        max_vectors: Max vectors for sparse
        rng: Random number generator
    
    Returns:
        Array of shape [N, 2] with (y, x) coordinates
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if strategy == 'all':
        y, x = np.mgrid[0:H, 0:W]
        coords = np.stack([y.ravel(), x.ravel()], axis=1)
        return coords
    
    elif strategy == 'cell_centers':
        if cell_size is None:
            # Default to ~32x32 cells
            h_cell = max(1, H // 32)
            w_cell = max(1, W // 32)
        else:
            h_cell, w_cell = cell_size
        
        y_centers = np.arange(h_cell // 2, H, h_cell)
        x_centers = np.arange(w_cell // 2, W, w_cell)
        y, x = np.meshgrid(y_centers, x_centers, indexing='ij')
        coords = np.stack([y.ravel(), x.ravel()], axis=1)
        return coords
    
    elif strategy == 'random':
        total_pixels = H * W
        n_keep = int(total_pixels * random_fraction)
        if max_vectors is not None:
            n_keep = min(n_keep, max_vectors)
        
        all_coords = np.stack([
            np.repeat(np.arange(H), W),
            np.tile(np.arange(W), H)
        ], axis=1)
        
        idx = rng.choice(len(all_coords), size=n_keep, replace=False)
        return all_coords[idx]
    
    elif strategy == 'sparse':
        if max_vectors is None:
            max_vectors = min(10000, H * W)
        
        all_coords = np.stack([
            np.repeat(np.arange(H), W),
            np.tile(np.arange(W), H)
        ], axis=1)
        
        idx = rng.choice(len(all_coords), size=min(max_vectors, len(all_coords)), replace=False)
        return all_coords[idx]
    
    else:
        raise ValueError(f"Unknown subsample strategy: {strategy}")


# ------------------------------
# I/O Functions (Reusable for Visualizations)
# ------------------------------

def load_vector_coverage(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load vector coverage JSON file, return dict with vectors as numpy array.
    
    Args:
        path: Path to JSON file
    
    Returns:
        Dictionary with keys:
        - 'vectors': numpy array [N, 4] of feature vectors
        - 'config': configuration dict
        - 'metadata': metadata dict
        - 'statistics': statistics dict (if available)
        - 'dataset_labels': list of dataset labels (if available)
        - 'frame_indices': list of frame indices (if available)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vector coverage file not found: {path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Convert vectors list to numpy array
    if 'vectors' in data and data['vectors']:
        data['vectors'] = np.array(data['vectors'], dtype=np.float32)
    else:
        data['vectors'] = np.array([], dtype=np.float32).reshape(0, 4)
    
    return data


def load_multiple_vector_coverage(
    paths: List[Union[str, Path]],
    names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load multiple vector coverage JSON files.
    
    Args:
        paths: List of paths to JSON files
        names: Optional list of names (uses filenames if None)
    
    Returns:
        Dictionary mapping dataset names to loaded data
    """
    if names is None:
        names = [Path(p).stem.replace('_vector_coverage', '') for p in paths]
    
    results = {}
    for path, name in zip(paths, names):
        results[name] = load_vector_coverage(path)
    
    return results


def get_vectors_from_json(path: Union[str, Path]) -> np.ndarray:
    """
    Convenience function to extract just the vectors array from JSON path.
    
    Args:
        path: Path to JSON file
    
    Returns:
        numpy array [N, 4] of feature vectors
    """
    data = load_vector_coverage(path)
    return data['vectors']


def save_vector_coverage(path: Union[str, Path], result: Dict[str, Any]) -> None:
    """
    Save vector coverage results to JSON.
    
    Args:
        path: Path to save JSON file
        result: Dictionary with results (vectors will be converted to list)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types recursively
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8, np.uint16,
                           np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert the entire result
    result_converted = convert_numpy_types(result)
    
    with open(path, 'w') as f:
        json.dump(result_converted, f, indent=2)

