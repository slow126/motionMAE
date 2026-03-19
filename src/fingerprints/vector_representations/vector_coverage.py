"""
vector_coverage.py
==================
Coverage analysis and distribution comparison for flow vectors.

Implements:
- MMD (Maximum Mean Discrepancy) with RBF kernel
- Asymmetric containment/coverage metrics
- K-means clustering comparison
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import numpy as np

try:
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist
    from scipy.spatial import cKDTree
    from scipy.stats import wasserstein_distance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn/scipy not available. Some comparison methods will be disabled.")

from src.fingerprints.vector_representations.vector_utils import load_vector_coverage


# ------------------------------
# Comparison Functions
# ------------------------------

def compute_mmd(
    Z_T: np.ndarray,
    Z_E: np.ndarray,
    sigma: float = 1.0,
    use_multiple_bandwidths: bool = True,
    bandwidths: Optional[List[float]] = None,
    max_vectors: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Compute Maximum Mean Discrepancy (MMD²) between two sets of vectors.
    
    Args:
        Z_T: Training vectors [n_T, d]
        Z_E: Eval vectors [n_E, d]
        sigma: RBF kernel bandwidth
        use_multiple_bandwidths: If True, use multiple bandwidths and sum
        bandwidths: List of bandwidths to use (if None, auto-generate)
        max_vectors: Maximum number of vectors to use (subsample if needed). None = use all.
        random_state: Random seed for subsampling
    
    Returns:
        Dictionary with MMD² values
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scipy is required for MMD computation")
    
    # Subsample vectors if needed (before computing kernels to save memory)
    if max_vectors is not None:
        rng = np.random.default_rng(random_state)
        original_n_T = len(Z_T)
        original_n_E = len(Z_E)
        
        if len(Z_T) > max_vectors:
            idx_T = rng.choice(len(Z_T), size=max_vectors, replace=False)
            Z_T = Z_T[idx_T]
            print(f"  Subsampled training vectors: {original_n_T} -> {len(Z_T)}")
        
        if len(Z_E) > max_vectors:
            idx_E = rng.choice(len(Z_E), size=max_vectors, replace=False)
            Z_E = Z_E[idx_E]
            print(f"  Subsampled eval vectors: {original_n_E} -> {len(Z_E)}")
    
    n_T = len(Z_T)
    n_E = len(Z_E)
    
    if n_T == 0 or n_E == 0:
        return {'mmd2': float('nan'), 'mmd2_per_bandwidth': {}}
    
    if bandwidths is None and use_multiple_bandwidths:
        # Use fixed bandwidths for consistent comparisons across datasets
        # 5 regularly sampled bandwidths from 1 to 5
        bandwidths = np.linspace(1.0, 5.0, 5).tolist()
    elif not use_multiple_bandwidths:
        bandwidths = [sigma]
    
    mmd2_per_bandwidth = {}
    total_mmd2 = 0.0
    
    # Process each bandwidth separately to minimize memory usage
    for bw_idx, bw in enumerate(bandwidths):
        # RBF kernel: k(z, z') = exp(-||z - z'||² / (2σ²))
        # Compute distance matrices and kernels one at a time, process, then free
        
        # Term 1: E[k(z_T, z_T')] excluding diagonal
        dists_TT = cdist(Z_T, Z_T, metric='sqeuclidean')
        K_TT = np.exp(-dists_TT / (2 * bw**2))
        term1 = (K_TT.sum() - np.trace(K_TT)) / (n_T * (n_T - 1))
        # Free memory immediately
        del dists_TT, K_TT
        
        # Term 2: E[k(z_E, z_E')] excluding diagonal
        dists_EE = cdist(Z_E, Z_E, metric='sqeuclidean')
        K_EE = np.exp(-dists_EE / (2 * bw**2))
        term2 = (K_EE.sum() - np.trace(K_EE)) / (n_E * (n_E - 1))
        # Free memory immediately
        del dists_EE, K_EE
        
        # Term 3: E[k(z_T, z_E)]
        dists_TE = cdist(Z_T, Z_E, metric='sqeuclidean')
        K_TE = np.exp(-dists_TE / (2 * bw**2))
        term3 = 2 * K_TE.mean()
        # Free memory immediately
        del dists_TE, K_TE
        
        # Compute MMD² for this bandwidth
        mmd2 = term1 + term2 - term3
        mmd2_per_bandwidth[f'sigma_{bw:.3f}'] = float(mmd2)
        total_mmd2 += mmd2
        
        # Force garbage collection between bandwidths to free memory
        if bw_idx < len(bandwidths) - 1:  # Don't GC on last iteration
            import gc
            gc.collect()
    
    return {
        'mmd2': float(total_mmd2),
        'mmd2_per_bandwidth': mmd2_per_bandwidth,
    }


def compute_containment_metrics(
    Z_T: np.ndarray,
    Z_E: np.ndarray,
    epsilon: float = 1e-6,
) -> Dict[str, Any]:
    """
    Compute asymmetric containment/coverage metrics.
    
    For each eval sample, compute distance to nearest train sample:
    d_E→T(j) = min_i ||z_j^(E) - z_i^(T)||
    
    For each train sample, compute distance to nearest eval sample:
    d_T→E(i) = min_j ||z_i^(T) - z_j^(E)||
    
    Coverage score: C = d_T→E / (d_E→T + ε)
    - C > 1: train covers more than eval (good) - train extends beyond eval
    - C ≈ 1: distributions overlap comparably
    - C < 1: eval is broader or off-support (bad) - eval extends beyond train
    
    Args:
        Z_T: Training vectors [n_T, d]
        Z_E: Eval vectors [n_E, d]
        epsilon: Small value to avoid division by zero
    
    Returns:
        Dictionary with containment metrics
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scipy is required for containment computation")
    
    if len(Z_T) == 0 or len(Z_E) == 0:
        return {
            'd_E_to_T': {'mean': float('nan'), 'median': float('nan'), 'p95': float('nan')},
            'd_T_to_E': {'mean': float('nan'), 'median': float('nan'), 'p95': float('nan')},
            'coverage_score': float('nan'),
        }
    
    # Build KD-trees for efficient nearest neighbor search
    tree_T = cKDTree(Z_T)
    tree_E = cKDTree(Z_E)
    
    # For each eval sample, distance to nearest train sample
    d_E_to_T, _ = tree_T.query(Z_E, k=1)
    d_E_to_T = d_E_to_T.flatten()
    
    # For each train sample, distance to nearest eval sample
    d_T_to_E, _ = tree_E.query(Z_T, k=1)
    d_T_to_E = d_T_to_E.flatten()
    
    # Summarize
    d_E_to_T_mean = float(np.mean(d_E_to_T))
    d_E_to_T_median = float(np.median(d_E_to_T))
    d_E_to_T_p95 = float(np.percentile(d_E_to_T, 95))
    
    d_T_to_E_mean = float(np.mean(d_T_to_E))
    d_T_to_E_median = float(np.median(d_T_to_E))
    d_T_to_E_p95 = float(np.percentile(d_T_to_E, 95))
    
    # Coverage score: C = d_T_to_E / (d_E_to_T + ε)
    # Higher is better: C > 1 means train covers eval well (train extends beyond eval)
    coverage_score = d_T_to_E_mean / (d_E_to_T_mean + epsilon)
    
    return {
        'd_E_to_T': {
            'mean': d_E_to_T_mean,
            'median': d_E_to_T_median,
            'p95': d_E_to_T_p95,
        },
        'd_T_to_E': {
            'mean': d_T_to_E_mean,
            'median': d_T_to_E_median,
            'p95': d_T_to_E_p95,
        },
        'coverage_score': float(coverage_score),
    }


def compute_kmeans_comparison(
    Z_T: np.ndarray,
    Z_E: np.ndarray,
    n_clusters: int = 50,
    max_samples: Optional[int] = 100000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compare distributions using K-means clustering.
    
    Fit K-means on training data, then compute cluster assignment histograms
    for both training and eval datasets. Compare using JS divergence and
    Wasserstein distance.
    
    Args:
        Z_T: Training vectors [n_T, d]
        Z_E: Eval vectors [n_E, d]
        n_clusters: Number of clusters
        max_samples: Max samples to use for K-means (subsample if needed)
        random_state: Random seed
    
    Returns:
        Dictionary with cluster comparison metrics
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for K-means comparison")
    
    if len(Z_T) == 0 or len(Z_E) == 0:
        return {
            'n_clusters': n_clusters,
            'cluster_hist_T': None,
            'cluster_hist_E': None,
            'js_divergence': float('nan'),
            'wasserstein_distance': float('nan'),
        }
    
    # Subsample if needed
    if max_samples is not None and len(Z_T) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(Z_T), size=max_samples, replace=False)
        Z_T_sample = Z_T[idx]
    else:
        Z_T_sample = Z_T
    
    # Fit K-means on training data
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(Z_T_sample)
    
    # Get cluster assignments
    labels_T = kmeans.predict(Z_T)
    labels_E = kmeans.predict(Z_E)
    
    # Compute cluster histograms
    hist_T = np.bincount(labels_T, minlength=n_clusters).astype(float)
    hist_E = np.bincount(labels_E, minlength=n_clusters).astype(float)
    
    # Normalize to probabilities
    p_T = hist_T / hist_T.sum() if hist_T.sum() > 0 else hist_T
    p_E = hist_E / hist_E.sum() if hist_E.sum() > 0 else hist_E
    
    # Compute JS divergence
    # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)
    M = 0.5 * (p_T + p_E)
    M = np.maximum(M, 1e-10)  # Avoid log(0)
    
    p_T_safe = np.maximum(p_T, 1e-10)
    p_E_safe = np.maximum(p_E, 1e-10)
    
    kl_PM = np.sum(p_T_safe * np.log(p_T_safe / M))
    kl_QM = np.sum(p_E_safe * np.log(p_E_safe / M))
    js_div = 0.5 * kl_PM + 0.5 * kl_QM
    
    # Compute Wasserstein distance (1D, between histograms)
    wasserstein_dist = wasserstein_distance(
        np.arange(n_clusters),
        np.arange(n_clusters),
        p_T,
        p_E,
    )
    
    return {
        'n_clusters': n_clusters,
        'cluster_hist_T': p_T.tolist(),
        'cluster_hist_E': p_E.tolist(),
        'cluster_counts_T': hist_T.tolist(),
        'cluster_counts_E': hist_E.tolist(),
        'js_divergence': float(js_div),
        'wasserstein_distance': float(wasserstein_dist),
        'cluster_centers': kmeans.cluster_centers_.tolist(),
    }


def compare_vector_coverage(
    train_path: Union[str, Path],
    eval_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    mmd_sigma: float = 1.0,
    mmd_use_multiple_bandwidths: bool = True,
    mmd_max_vectors: Optional[int] = None,
    kmeans_n_clusters: int = 50,
) -> Dict[str, Any]:
    """
    Compare two vector coverage files using all metrics.
    
    Args:
        train_path: Path to training dataset vector coverage JSON
        eval_path: Path to eval dataset vector coverage JSON
        output_path: Optional path to save comparison results
        mmd_sigma: MMD kernel bandwidth
        mmd_use_multiple_bandwidths: Use multiple bandwidths for MMD
        mmd_max_vectors: Maximum vectors to use for MMD (subsample if needed). None = use all.
        kmeans_n_clusters: Number of clusters for K-means
    
    Returns:
        Dictionary with all comparison metrics
    """
    # Load vector coverage files
    train_data = load_vector_coverage(train_path)
    eval_data = load_vector_coverage(eval_path)
    
    # Extract vectors
    Z_T = train_data['vectors']
    Z_E = eval_data['vectors']
    
    print(f"Training vectors: {len(Z_T)}")
    print(f"Eval vectors: {len(Z_E)}")
    
    # Compute all metrics
    results = {
        'train_dataset': train_data.get('metadata', {}).get('dataset_name', 'unknown'),
        'eval_dataset': eval_data.get('metadata', {}).get('dataset_name', 'unknown'),
        'num_vectors_T': len(Z_T),
        'num_vectors_E': len(Z_E),
    }
    
    print("\nComputing MMD...")
    results['mmd'] = compute_mmd(
        Z_T, Z_E,
        sigma=mmd_sigma,
        use_multiple_bandwidths=mmd_use_multiple_bandwidths,
        max_vectors=mmd_max_vectors,
    )
    
    print("Computing containment metrics...")
    results['containment'] = compute_containment_metrics(Z_T, Z_E)
    
    print("Computing K-means comparison...")
    results['kmeans'] = compute_kmeans_comparison(
        Z_T, Z_E,
        n_clusters=kmeans_n_clusters,
    )
    
    # Save results
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved comparison results to: {output_path}")
    
    return results

