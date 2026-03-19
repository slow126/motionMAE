"""
flow_vector.py
==============
Flow vector representation in joint space (Cartesian).

Creates feature vectors z = [x/H, y/W, u/s_u, v/s_v] for each valid flow vector.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple, Literal
import numpy as np

from src.fingerprints.vector_representations.vector_utils import (
    _magnitude,
    _compute_flow_stats,
    _subsample_pixels,
)


# ------------------------------
# Configuration
# ------------------------------

@dataclass
class FlowVectorConfig:
    """Configuration for flow vector representation."""
    # Subsampling options
    subsample_strategy: Literal['all', 'cell_centers', 'random', 'sparse'] = 'all'
    cell_size: Optional[Tuple[int, int]] = None  # (h_cell, w_cell) for cell_centers
    random_fraction: float = 0.1  # Fraction to keep for random subsampling
    max_vectors_per_frame: Optional[int] = None  # Max vectors per frame (for sparse)
    
    # Normalization
    normalize_flow_by: Literal['std', 'p95', 'p99', 'max'] = 'p95'
    normalize_flow: Literal['global', 'local', 'none'] = 'global'  # 'global': use stats from all datasets, 'local': per-dataset, 'none': no normalization
    
    # Storage
    max_vectors_total: Optional[int] = None  # Limit total vectors across all datasets
    save_vectors: bool = True  # Whether to save raw vectors
    save_stats: bool = True  # Whether to save statistics
    
    # Reproducibility
    random_seed: int = 42


# ------------------------------
# Flow Vector Accumulator
# ------------------------------

class FlowVector:
    """
    Accumulate flow vectors in joint space representation.
    
    Processes flow fields and converts valid vectors to joint space features:
    z = [x/H, y/W, u/s_u, v/s_v]
    """
    
    def __init__(self, config: FlowVectorConfig):
        self.config = config
        self.vectors = []  # List of feature vectors
        self.dataset_labels = []  # Track which dataset each vector came from
        self.frame_indices = []  # Track which frame each vector came from
        
        # Statistics for normalization
        self.flow_stats_computed = False
        self.s_u = 1.0
        self.s_v = 1.0
        
        # Temporary storage for computing global stats
        self._temp_flows = []
        self._temp_masks = []
        
        # Random number generator
        self.rng = np.random.default_rng(config.random_seed)
    
    def add_frame(
        self,
        flow: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        dataset_label: Optional[str] = None,
        frame_idx: Optional[int] = None,
    ) -> int:
        """
        Add a flow frame and extract valid vectors.
        
        Args:
            flow: Flow array [H, W, 2]
            valid_mask: Optional valid mask [H, W]
            dataset_label: Optional label for this dataset
            frame_idx: Optional frame index
        
        Returns:
            Number of vectors added
        """
        assert flow.ndim == 3 and flow.shape[-1] == 2, f"flow must be [H, W, 2], got {flow.shape}"
        H, W, _ = flow.shape
        
        u = flow[..., 0]
        v = flow[..., 1]
        
        # Create valid mask - check for invalid vectors (inf/nan)
        if valid_mask is None:
            valid_mask = np.isfinite(u) & np.isfinite(v)
        else:
            # Combine provided mask with finite check
            valid_mask = (valid_mask > 0) & np.isfinite(u) & np.isfinite(v)
        
        # Store for later normalization if needed
        if self.config.normalize_flow != 'none' and (self.config.normalize_flow == 'local' or not self.flow_stats_computed):
            self._temp_flows.append(flow)
            self._temp_masks.append(valid_mask)
        
        # Subsample pixel coordinates
        coords = _subsample_pixels(
            H, W,
            strategy=self.config.subsample_strategy,
            cell_size=self.config.cell_size,
            random_fraction=self.config.random_fraction,
            max_vectors=self.config.max_vectors_per_frame,
            rng=self.rng,
        )
        
        # Extract vectors at subsampled coordinates
        vectors_added = 0
        for y, x in coords:
            y, x = int(y), int(x)
            
            # Check if valid - multiple validation points
            if not valid_mask[y, x]:
                continue
            
            # Get flow components
            u_val = u[y, x]
            v_val = v[y, x]
            
            # Final check for invalid vectors
            if not (np.isfinite(u_val) and np.isfinite(v_val)):
                continue
            
            # Filter out exactly (0,0) flows (ambiguous/static pixels)
            if u_val == 0.0 and v_val == 0.0:
                continue
            
            # Create feature vector
            z = self._create_cartesian_vector(y, x, H, W, u_val, v_val)
            
            self.vectors.append(z)
            self.dataset_labels.append(dataset_label)
            self.frame_indices.append(frame_idx)
            vectors_added += 1
            
            # Check total limit
            if (self.config.max_vectors_total is not None and 
                len(self.vectors) >= self.config.max_vectors_total):
                return vectors_added
        
        return vectors_added
    
    def _create_cartesian_vector(
        self, y: int, x: int, H: int, W: int, u: float, v: float
    ) -> np.ndarray:
        """
        Create Cartesian representation: [x/H, y/W, u/s_u, v/s_v].
        
        Args:
            y, x: Pixel coordinates
            H, W: Image dimensions
            u, v: Flow components
        
        Returns:
            Feature vector [x_norm, y_norm, u_norm, v_norm]
        """
        # Normalize spatial coordinates
        x_norm = x / W
        y_norm = y / H
        
        # Normalize flow components only if normalization is enabled
        if self.config.normalize_flow != 'none':
            u_norm = u / self.s_u
            v_norm = v / self.s_v
        else:
            # No normalization - preserve raw scale
            u_norm = u
            v_norm = v
        
        return np.array([x_norm, y_norm, u_norm, v_norm], dtype=np.float32)
    
    def compute_flow_stats(self) -> None:
        """Compute flow normalization statistics (always computed for reference, even if not normalizing)."""
        if self.flow_stats_computed:
            return
        
        if not self._temp_flows:
            self.s_u, self.s_v = 1.0, 1.0
            self.flow_stats_computed = True
            return
        
        # Always compute stats (even for 'none' mode) - useful for reference
        self.s_u, self.s_v = _compute_flow_stats(
            self._temp_flows,
            self._temp_masks,
            method=self.config.normalize_flow_by,
        )
        self.flow_stats_computed = True
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize and return statistics.
        
        Returns:
            Dictionary with vectors, statistics, and metadata
        """
        # Compute flow stats if needed
        if self.config.normalize_flow == 'global':
            self.compute_flow_stats()
        elif self.config.normalize_flow == 'local':
            self.compute_flow_stats()
        else:  # 'none'
            # Still compute stats for reference, but don't use for normalization
            self.compute_flow_stats()
        
        # Convert to numpy array
        if self.vectors:
            vectors_array = np.array(self.vectors, dtype=np.float32)
        else:
            vectors_array = np.array([], dtype=np.float32).reshape(0, 4)
        
        result = {
            'config': asdict(self.config),
            'num_vectors': len(self.vectors),
            'flow_stats': {
                's_u': float(self.s_u),
                's_v': float(self.s_v),
                'normalize_by': self.config.normalize_flow_by,
            },
        }
        
        if self.config.save_vectors:
            result['vectors'] = vectors_array.tolist()
            result['dataset_labels'] = self.dataset_labels
            result['frame_indices'] = self.frame_indices
        
        if self.config.save_stats and len(vectors_array) > 0:
            result['statistics'] = {
                'mean': vectors_array.mean(axis=0).tolist(),
                'std': vectors_array.std(axis=0).tolist(),
                'min': vectors_array.min(axis=0).tolist(),
                'max': vectors_array.max(axis=0).tolist(),
            }
            
            # Per-dimension statistics
            dim_names = ['x_norm', 'y_norm', 'u_norm', 'v_norm']
            result['statistics']['per_dimension'] = {
                name: {
                    'mean': float(vectors_array[:, i].mean()),
                    'std': float(vectors_array[:, i].std()),
                    'min': float(vectors_array[:, i].min()),
                    'max': float(vectors_array[:, i].max()),
                }
                for i, name in enumerate(dim_names)
            }
        
        return result

