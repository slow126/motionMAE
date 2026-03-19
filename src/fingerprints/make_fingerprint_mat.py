"""
make_fingerprint_mat.py
========================
Compute distances between flow fingerprints and create confusion matrix visualizations.
Loads fingerprints based on fingerprints.yaml config and computes all distance metrics.
"""

import os
import sys
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from slurm.generate_jobs import load_experiments, load_machine_config, load_experiment_config
from src.fingerprints.flow_fingerprint import load_stats_json

# ============================================================================
# CONFIGURATION
# ============================================================================
EXPERIMENT_CONFIG = 'slurm/experiment_configs/fingerprints.yaml'
MACHINE_CONFIG = 'slurm/machine_configs/local.yaml'
FINGERPRINT_DIR = './fingerprints'  # Directory containing fingerprint JSON files
OUTPUT_DIR = './fingerprint_matrices'  # Output directory for distance matrices

# Distance metric weights (for combined distance - optional)
DISTANCE_WEIGHTS = {
    'alpha_js_hist': 1.0,
    'beta_w1_hist': 1.0,
    'gamma_occ_L2': 1.0,
    'delta_occ_JS': 1.0,
    'eta_mag_map': 1.0,
}

# ============================================================================
# Distance Computation Functions (from pseudocode)
# ============================================================================

def to_prob_dist(x, eps=1e-12):
    """Normalize a non-negative vector x into a probability distribution."""
    x = np.asarray(x, dtype=np.float64)
    total = x.sum()
    if total <= 0:
        return np.ones_like(x) / len(x)
    return x / (total + eps)

def kl_divergence(p, q, eps=1e-12):
    """Kullback–Leibler divergence KL(p || q) for discrete distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))

def jensen_shannon_divergence(p, q, eps=1e-12):
    """Jensen–Shannon divergence between two discrete distributions p and q."""
    p = to_prob_dist(p, eps=eps)
    q = to_prob_dist(q, eps=eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)

def wasserstein_1d(p, q, x, eps=1e-12):
    """1D Wasserstein-1 (Earth Mover's Distance) between two discrete distributions."""
    p = to_prob_dist(p, eps=eps)
    q = to_prob_dist(q, eps=eps)
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure x is sorted in ascending order along with p and q
    order = np.argsort(x)
    x = x[order]
    p = p[order]
    q = q[order]
    
    # CDFs
    Fp = np.cumsum(p)
    Fq = np.cumsum(q)
    
    # Bin spacings Δx_i = x_i - x_{i-1}, with Δx_0 = 0 by convention
    dx = np.diff(x, prepend=x[0])
    
    # Approximate W1
    return np.sum(np.abs(Fp - Fq) * dx)

def spatial_occ_distance_L2(P_train, P_eval):
    """L2 distance between spatial occupancy probability maps."""
    P_train = np.asarray(P_train, dtype=np.float64)
    P_eval = np.asarray(P_eval, dtype=np.float64)
    return np.sqrt(np.mean((P_train - P_eval) ** 2))

def spatial_occ_distance_JS(P_train, P_eval, eps=1e-12):
    """Jensen–Shannon divergence between normalized spatial occupancy distributions."""
    P_train_flat = np.asarray(P_train, dtype=np.float64).reshape(-1)
    P_eval_flat = np.asarray(P_eval, dtype=np.float64).reshape(-1)
    return jensen_shannon_divergence(P_train_flat, P_eval_flat, eps=eps)

def spatial_mag_map_distance(P_train, M_train, P_eval, M_eval, eps=1e-12, use_log=True):
    """Distance between spatial maps of expected flow magnitude."""
    P_train = np.asarray(P_train, dtype=np.float64)
    M_train = np.asarray(M_train, dtype=np.float64)
    P_eval = np.asarray(P_eval, dtype=np.float64)
    M_eval = np.asarray(M_eval, dtype=np.float64)
    
    E_train = P_train * M_train
    E_eval = P_eval * M_eval
    
    if use_log:
        L_train = np.log(E_train + eps)
        L_eval = np.log(E_eval + eps)
        return np.sqrt(np.mean((L_train - L_eval) ** 2))
    else:
        return np.sqrt(np.mean((E_train - E_eval) ** 2))

def magnitude_hist_distances(hist_train, hist_eval, bin_centers=None, eps=1e-12):
    """Compute both JS divergence and 1D Wasserstein distance between flow magnitude histograms."""
    hist_train = np.asarray(hist_train, dtype=np.float64)
    hist_eval = np.asarray(hist_eval, dtype=np.float64)
    
    js = jensen_shannon_divergence(hist_train, hist_eval, eps=eps)
    
    if bin_centers is None:
        K = hist_train.shape[0]
        x = np.arange(K, dtype=np.float64)
    else:
        x = np.asarray(bin_centers, dtype=np.float64)
    
    w1 = wasserstein_1d(hist_train, hist_eval, x, eps=eps)
    
    return js, w1

def dataset_distance(f_train, f_eval, alpha_js_hist=1.0, beta_w1_hist=1.0, 
                     gamma_occ_L2=1.0, delta_occ_JS=1.0, eta_mag_map=1.0, eps=1e-12):
    """Combined distance between two dataset fingerprints."""
    P_t = f_train["P"]
    M_t = f_train["M"]
    h_t = f_train["hist"]
    P_e = f_eval["P"]
    M_e = f_eval["M"]
    h_e = f_eval["hist"]
    
    # Optional bin centers for Wasserstein on magnitude hist
    bin_centers_t = f_train.get("bin_centers", None)
    bin_centers_e = f_eval.get("bin_centers", None)
    if bin_centers_t is not None:
        bin_centers = bin_centers_t
    else:
        bin_centers = bin_centers_e
    
    # Magnitude histogram distances
    d_js_hist, d_w1_hist = magnitude_hist_distances(h_t, h_e, bin_centers=bin_centers, eps=eps)
    
    # Spatial occupancy distances
    d_occ_L2 = spatial_occ_distance_L2(P_t, P_e)
    d_occ_JS = spatial_occ_distance_JS(P_t, P_e, eps=eps)
    
    # Spatial magnitude map distance (log-RMSE by default)
    d_mag_map = spatial_mag_map_distance(P_t, M_t, P_e, M_e, eps=eps, use_log=True)
    
    # Weighted sum as a single scalar
    D_total = (
        alpha_js_hist * d_js_hist +
        beta_w1_hist * d_w1_hist +
        gamma_occ_L2 * d_occ_L2 +
        delta_occ_JS * d_occ_JS +
        eta_mag_map * d_mag_map
    )
    
    components = {
        "js_hist": d_js_hist,
        "w1_hist": d_w1_hist,
        "occ_L2": d_occ_L2,
        "occ_JS": d_occ_JS,
        "mag_map": d_mag_map,
    }
    
    return D_total, components

# ============================================================================
# Fingerprint Format Conversion
# ============================================================================

def convert_fingerprint_format(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert fingerprint JSON format to distance computation format.
    
    Input format (from JSON):
        stats["spatial"]["motion_prob"] -> (H, W) list
        stats["spatial"]["mean_magnitude"] -> (H, W) list
        stats["hists"]["mag"] -> (K,) list
        stats["bins"]["mag_edges"] -> (K+1,) list
    
    Output format:
        f["P"] -> (H, W) numpy array
        f["M"] -> (H, W) numpy array
        f["hist"] -> (K,) numpy array
        f["bin_centers"] -> (K,) numpy array (computed from edges)
    """
    # Convert spatial maps
    P = np.array(stats["spatial"]["motion_prob"], dtype=np.float64)
    M = np.array(stats["spatial"]["mean_magnitude"], dtype=np.float64)
    
    # Convert histogram
    hist = np.array(stats["hists"]["mag"], dtype=np.float64)
    
    # Compute bin centers from edges
    mag_edges = np.array(stats["bins"]["mag_edges"], dtype=np.float64)
    bin_centers = (mag_edges[:-1] + mag_edges[1:]) / 2.0
    
    return {
        "P": P,
        "M": M,
        "hist": hist,
        "bin_centers": bin_centers,
    }

# ============================================================================
# Label Parsing and Alignment (reused from make_confusion_mat.py)
# ============================================================================

def extract_base_dataset_name(training_label):
    """Extract base dataset name from training label."""
    base = training_label.split('_')[0]
    if base.lower() in ['ptod', 'ptodyssey']:
        return 'pointodyssey'
    return base.lower()

def order_for_diagonal_alignment(training_labels, benchmark_labels):
    """Order training labels and benchmark labels so matching ones align on diagonal."""
    training_base_map = {label: extract_base_dataset_name(label) for label in training_labels}
    
    # Group training labels by base dataset name
    training_groups = defaultdict(list)
    for label in training_labels:
        base = training_base_map[label]
        training_groups[base].append(label)
    
    # Sort each group for consistent ordering
    for base in training_groups:
        training_groups[base].sort()
    
    # Find matches
    matched_training = []
    matched_benchmarks = []
    unmatched_training = []
    unmatched_benchmarks = []
    
    matched_benchmark_set = set()
    matched_training_base_set = set()
    
    # First pass: find exact matches
    for benchmark in benchmark_labels:
        benchmark_lower = benchmark.lower()
        if benchmark_lower in training_groups:
            matched_training.extend(training_groups[benchmark_lower])
            matched_benchmarks.append(benchmark)
            matched_benchmark_set.add(benchmark)
            matched_training_base_set.add(benchmark_lower)
    
    # Collect unmatched
    for label in training_labels:
        base = training_base_map[label]
        if base not in matched_training_base_set:
            unmatched_training.append(label)
    
    for benchmark in benchmark_labels:
        if benchmark not in matched_benchmark_set:
            unmatched_benchmarks.append(benchmark)
    
    ordered_training = matched_training + sorted(unmatched_training)
    ordered_benchmarks = matched_benchmarks + sorted(unmatched_benchmarks)
    
    return ordered_training, ordered_benchmarks

# ============================================================================
# Fingerprint Loading (with caching)
# ============================================================================

def load_all_fingerprints(experiment_config_path: str, machine_config_path: str, 
                          fingerprint_dir: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Load all fingerprints based on experiment config.
    Avoids redundant loading by caching each fingerprint once.
    
    Returns:
        Tuple of (training_fingerprints_dict, eval_fingerprints_dict)
    """
    # Load configs once
    experiment_config_path_obj = Path(experiment_config_path)
    if not experiment_config_path_obj.is_absolute():
        experiment_config_path_obj = project_root / experiment_config_path_obj
    
    machine_config_path_obj = Path(machine_config_path)
    if not machine_config_path_obj.is_absolute():
        machine_config_path_obj = project_root / machine_config_path_obj
    
    experiment_config = load_experiment_config(str(experiment_config_path_obj))
    machine_config = load_machine_config(str(machine_config_path_obj))
    
    # Expand all experiments once
    all_experiments = load_experiments(experiment_config, machine_config)
    
    # Extract eval_benchmarks list once from base config
    eval_benchmarks = experiment_config.get('base', {}).get('eval_benchmarks', [])
    if not eval_benchmarks and all_experiments:
        # Fallback to first experiment's eval_benchmarks
        eval_benchmarks = all_experiments[0].get('eval_benchmarks', [])
    
    # Load fingerprint directory
    fingerprint_dir_obj = Path(fingerprint_dir)
    if not fingerprint_dir_obj.is_absolute():
        fingerprint_dir_obj = project_root / fingerprint_dir_obj
    
    # Cache for fingerprints - load each JSON file exactly once
    fingerprint_cache = {}
    training_fingerprints = {}
    eval_fingerprints = {}
    
    # Load training dataset fingerprints
    print("Loading training fingerprints...")
    for exp_config in all_experiments:
        exp_name = exp_config.get('name_exp', 'unknown')
        
        # Check cache first
        if exp_name in fingerprint_cache:
            training_fingerprints[exp_name] = fingerprint_cache[exp_name]
            continue
        
        fingerprint_path = fingerprint_dir_obj / f"{exp_name}_fingerprint.json"
        
        if fingerprint_path.exists():
            print(f"  Loading: {exp_name}")
            try:
                stats = load_stats_json(str(fingerprint_path))
                converted_fp = convert_fingerprint_format(stats)
                fingerprint_cache[exp_name] = converted_fp
                training_fingerprints[exp_name] = converted_fp
            except Exception as e:
                print(f"  Warning: Failed to load {exp_name}: {e}")
        else:
            print(f"  Warning: Not found: {exp_name}_fingerprint.json")
    
    # Load evaluation benchmark fingerprints
    # Since eval fingerprints are identical across experiments, load once per benchmark
    # Try loading from any experiment that has them
    print("\nLoading evaluation fingerprints...")
    if all_experiments and eval_benchmarks:
        for benchmark in eval_benchmarks:
            benchmark_lower = str(benchmark).lower()
            cache_key = benchmark_lower
            
            # Check cache first
            if cache_key in fingerprint_cache:
                eval_fingerprints[benchmark_lower] = fingerprint_cache[cache_key]
                continue
            
            # Try loading from any experiment (they're all the same)
            loaded = False
            for exp_config in all_experiments:
                exp_name = exp_config.get('name_exp', 'unknown')
                eval_fingerprint_name = f"{exp_name}_eval_{benchmark_lower}"
                fingerprint_path = fingerprint_dir_obj / f"{eval_fingerprint_name}_fingerprint.json"
                
                if fingerprint_path.exists():
                    print(f"  Loading: {benchmark_lower} (from {exp_name})")
                    try:
                        stats = load_stats_json(str(fingerprint_path))
                        converted_fp = convert_fingerprint_format(stats)
                        fingerprint_cache[cache_key] = converted_fp
                        eval_fingerprints[benchmark_lower] = converted_fp
                        loaded = True
                        break  # Found it, no need to check other experiments
                    except Exception as e:
                        print(f"  Warning: Failed to load {benchmark_lower} from {exp_name}: {e}")
                        continue
            
            if not loaded:
                print(f"  Warning: Not found: *_eval_{benchmark_lower}_fingerprint.json (tried all experiments)")
    
    return training_fingerprints, eval_fingerprints

# ============================================================================
# Matrix Computation
# ============================================================================

def compute_distance_matrix(training_fps: Dict[str, Dict], 
                            eval_fps: Dict[str, Dict],
                            distance_name: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute distance matrix between training and evaluation fingerprints.
    
    Args:
        training_fps: Dictionary mapping training dataset name -> fingerprint dict
        eval_fps: Dictionary mapping eval benchmark name -> fingerprint dict
        distance_name: Name of distance metric to compute
    
    Returns:
        Tuple of (matrix, training_labels, eval_labels) with diagonal alignment
    """
    training_labels = sorted(training_fps.keys())
    eval_labels = sorted(eval_fps.keys())
    
    # Order for diagonal alignment
    ordered_training, ordered_eval = order_for_diagonal_alignment(training_labels, eval_labels)
    
    # Pre-allocate matrix
    matrix = np.full((len(ordered_training), len(ordered_eval)), np.nan)
    
    # Compute distances for each pair
    for i, train_label in enumerate(ordered_training):
        for j, eval_label in enumerate(ordered_eval):
            train_fp = training_fps.get(train_label)
            eval_fp = eval_fps.get(eval_label)
            
            if train_fp is None or eval_fp is None:
                matrix[i, j] = np.nan
                continue
            
            try:
                if distance_name == "combined":
                    dist, _ = dataset_distance(train_fp, eval_fp, **DISTANCE_WEIGHTS)
                elif distance_name == "js_hist":
                    dist, _ = magnitude_hist_distances(
                        train_fp["hist"], eval_fp["hist"], 
                        bin_centers=train_fp.get("bin_centers")
                    )
                elif distance_name == "w1_hist":
                    _, dist = magnitude_hist_distances(
                        train_fp["hist"], eval_fp["hist"],
                        bin_centers=train_fp.get("bin_centers")
                    )
                elif distance_name == "occ_L2":
                    dist = spatial_occ_distance_L2(train_fp["P"], eval_fp["P"])
                elif distance_name == "occ_JS":
                    dist = spatial_occ_distance_JS(train_fp["P"], eval_fp["P"])
                elif distance_name == "mag_map":
                    dist = spatial_mag_map_distance(
                        train_fp["P"], train_fp["M"], 
                        eval_fp["P"], eval_fp["M"]
                    )
                else:
                    dist = np.nan
                
                matrix[i, j] = dist
            except Exception as e:
                print(f"  Warning: Failed to compute {distance_name} for {train_label} vs {eval_label}: {e}")
                matrix[i, j] = np.nan
    
    return matrix, ordered_training, ordered_eval

# ============================================================================
# Visualization
# ============================================================================

def save_distance_matrix_data(matrix: np.ndarray, training_labels: List[str],
                              eval_labels: List[str], output_path: Path,
                              distance_name: str):
    """
    Save distance matrix data to JSON and numpy files.
    
    Args:
        matrix: Distance matrix
        training_labels: List of training dataset labels
        eval_labels: List of evaluation benchmark labels
        output_path: Base path to save the data (will create .json and .npy files)
        distance_name: Name of the distance metric
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save numpy array separately
    npy_path = output_path.with_suffix('.npy')
    np.save(npy_path, matrix)
    
    # Save metadata as JSON
    json_data = {
        'distance_name': distance_name,
        'training_labels': training_labels,
        'eval_labels': eval_labels,
        'matrix_shape': list(matrix.shape),
        'matrix_file': npy_path.name
    }
    
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"  Saved matrix data: {json_path} (matrix: {npy_path})")


def load_distance_matrix_data(input_path):
    """
    Load distance matrix data from JSON and numpy files.
    
    Args:
        input_path: Path to the JSON file (or .json/.npy base path)
        
    Returns:
        Tuple of (matrix, training_labels, eval_labels, distance_name)
    """
    input_path = Path(input_path)
    
    # If .npy was provided, find the .json
    if input_path.suffix == '.npy':
        json_path = input_path.with_suffix('.json')
        npy_path = input_path
    else:
        json_path = input_path.with_suffix('.json')
        npy_path = input_path.with_suffix('.npy')
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not npy_path.exists():
        raise FileNotFoundError(f"NumPy file not found: {npy_path}")
    
    # Load JSON metadata
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Load numpy matrix
    matrix = np.load(npy_path)
    
    return (
        matrix,
        json_data['training_labels'],
        json_data['eval_labels'],
        json_data.get('distance_name', 'unknown')
    )


def create_distance_matrix_plot(matrix: np.ndarray, training_labels: List[str], 
                                eval_labels: List[str], output_path: Path, 
                                distance_name: str):
    """Create and save a distance matrix heatmap visualization."""
    fig, ax = plt.subplots(figsize=(max(12, len(eval_labels) * 1.2), 
                                    max(8, len(training_labels) * 0.5)))
    
    # Create heatmap - for distances, lower is better, so use reversed colormap
    # Use RdYlGn_r: Green (low/good) -> Yellow (medium) -> Red (high/bad)
    sns.heatmap(matrix, 
                annot=True, 
                fmt='.4f',
                cmap='RdYlGn_r',  # Reversed: green for low distance (good), red for high (bad)
                cbar_kws={'label': f'{distance_name} Distance'},
                xticklabels=eval_labels,
                yticklabels=training_labels,
                ax=ax,
                linewidths=0.5,
                linecolor='gray',
                mask=np.isnan(matrix))
    
    # Highlight diagonal cells where training dataset base matches benchmark
    training_base_map = {label: extract_base_dataset_name(label) for label in training_labels}
    eval_base_map = {label: label for label in eval_labels}  # Eval labels are already benchmark names
    
    benchmark_positions = {base.lower(): j for j, base in enumerate(eval_base_map.values())}
    
    highlighted_benchmarks = set()
    for i, training_label in enumerate(training_labels):
        base = training_base_map[training_label]
        if base in benchmark_positions:
            j = benchmark_positions[base]
            if base not in highlighted_benchmarks and not np.isnan(matrix[i, j]):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                         edgecolor='blue', lw=2, zorder=10))
                highlighted_benchmarks.add(base)
    
    distance_display = distance_name.replace('_', ' ').title()
    ax.set_title(f'Fingerprint Distance Matrix - {distance_display}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Evaluation Benchmark', fontsize=12)
    ax.set_ylabel('Training Dataset', fontsize=12)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to compute fingerprint distances and create matrices."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compute fingerprint distance matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--save_data',
        action='store_true',
        help='Save matrix data to JSON/numpy files for faster reloading'
    )
    
    parser.add_argument(
        '--load_data',
        type=str,
        default=None,
        help='Load matrix data from saved files (provide base directory, e.g., ./fingerprint_matrices/data)'
    )
    
    parser.add_argument(
        '--distance',
        type=str,
        default=None,
        help='Compute only specific distance metric (default: all metrics)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Computing Fingerprint Distance Matrices")
    print("="*60)
    print(f"Experiment config: {EXPERIMENT_CONFIG}")
    print(f"Fingerprint directory: {FINGERPRINT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define all distance metrics
    distance_metrics = [
        ("js_hist", "JS Divergence (Magnitude Histogram)"),
        ("w1_hist", "Wasserstein-1 (Magnitude Histogram)"),
        ("occ_L2", "L2 Distance (Spatial Occupancy)"),
        ("occ_JS", "JS Divergence (Spatial Occupancy)"),
        ("mag_map", "Log-RMSE (Spatial Magnitude Map)"),
        ("combined", "Combined Distance"),
    ]
    
    # Filter to specific distance if requested
    if args.distance:
        distance_metrics = [dm for dm in distance_metrics if dm[0] == args.distance]
        if not distance_metrics:
            print(f"Error: Unknown distance metric: {args.distance}")
            return
    
    # Load or compute matrices
    if args.load_data:
        print(f"Loading saved data from: {args.load_data}")
        data_dir = Path(args.load_data)
        if not data_dir.is_absolute():
            data_dir = project_root / data_dir
        
        for distance_name, distance_display in distance_metrics:
            try:
                matrix, training_labels, eval_labels, _ = load_distance_matrix_data(
                    data_dir / f"{distance_name}_distance_matrix_data"
                )
                print(f"  Loaded {distance_display}")
                
                # Create plot
                output_path = output_dir / f"{distance_name}_distance_matrix.png"
                create_distance_matrix_plot(matrix, training_labels, eval_labels, 
                                           output_path, distance_display)
            except FileNotFoundError as e:
                print(f"  Warning: Could not load {distance_name}: {e}")
    else:
        # Load fingerprints (once, with caching)
        training_fingerprints, eval_fingerprints = load_all_fingerprints(
            EXPERIMENT_CONFIG, MACHINE_CONFIG, FINGERPRINT_DIR
        )
        
        print(f"\nLoaded {len(training_fingerprints)} training fingerprints")
        print(f"Loaded {len(eval_fingerprints)} evaluation fingerprints")
        
        if not training_fingerprints or not eval_fingerprints:
            print("Error: Need both training and evaluation fingerprints!")
            return
        
        # Compute all distance matrices
        print(f"\nComputing distance matrices...")
        for distance_name, distance_display in distance_metrics:
            print(f"  Computing {distance_display}...")
            matrix, training_labels, eval_labels = compute_distance_matrix(
                training_fingerprints, eval_fingerprints, distance_name
            )
            
            # Save data if requested
            if args.save_data:
                data_path = output_dir / "data" / f"{distance_name}_distance_matrix_data"
                save_distance_matrix_data(matrix, training_labels, eval_labels,
                                         data_path, distance_name)
            
            # Create plot
            output_path = output_dir / f"{distance_name}_distance_matrix.png"
            create_distance_matrix_plot(matrix, training_labels, eval_labels, 
                                       output_path, distance_display)
    
    print("\nDone!")

if __name__ == "__main__":
    main()

