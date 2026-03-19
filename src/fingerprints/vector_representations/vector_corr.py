#!/usr/bin/env python3
"""
Plot correlation between PCK performance and vector metrics (MMD, Coverage, etc.).
X-axis: Vector metric value (MMD², Coverage Score, JS Divergence, etc.)
Y-axis: PCK performance (from validation_results.csv or training_summary.txt)
"""

import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import functions from existing scripts
from make_confusion_mat import (
    collect_snapshots,
    parse_snapshot_for_confusion_matrix,
    build_confusion_matrix_data,
    format_training_dataset_label,
    parse_average_performance_from_csv,
    parse_best_performance_from_summary,
)


def load_vector_metrics(csv_path):
    """
    Load vector metrics from summary_table.csv.
    
    Args:
        csv_path: Path to summary_table.csv
        
    Returns:
        pandas DataFrame with vector metrics
    """
    df = pd.read_csv(csv_path)
    return df


def normalize_eval_dataset_name(eval_name):
    """
    Normalize evaluation dataset name to match benchmark names.
    
    Examples:
        "spair_freezeTrue_eval_flyingthings" -> "flyingthings"
        "spair_freezeTrue_eval_kitti2012" -> "kitti2012"
        "spair_freezeTrue_eval_spair" -> "spair"
    """
    # Remove "eval_" prefix and everything before it
    if '_eval_' in eval_name:
        parts = eval_name.split('_eval_')
        if len(parts) > 1:
            return parts[-1]
    
    # If no eval prefix, try to extract last part
    parts = eval_name.split('_')
    # Common benchmark names
    benchmark_names = ['flyingthings', 'kitti2012', 'kitti2015', 'pfpascal', 
                      'pfwillow', 'pointodyssey', 'spair', 'tss']
    
    for part in reversed(parts):
        if part.lower() in benchmark_names:
            return part.lower()
    
    # Fallback: return last part
    return parts[-1].lower() if parts else eval_name.lower()


def normalize_training_label(label):
    """
    Normalize training label to match between confusion matrix and fingerprint formats.
    Both scripts use similar formatting, but we need to ensure consistency.
    """
    # Both use similar formats, but fingerprint labels might use full names
    # while confusion matrix might abbreviate (e.g., PtOd vs pointodyssey)
    label_lower = label.lower()
    
    # Normalize pointodyssey abbreviations
    if label_lower.startswith('ptod'):
        label_lower = label_lower.replace('ptod', 'pointodyssey', 1)
    
    return label_lower


def normalize_train_dataset_name(train_name):
    """
    Normalize training dataset name to match format used in confusion matrix.
    
    The confusion matrix uses format_training_dataset_label which:
    - Converts freezeTrue -> freezeT, freezeFalse -> freezeF
    - Preserves synthetic variant names (e.g., synthetic_large_flow_centered)
    - Abbreviates pointodyssey to PtOd
    
    Examples:
        "spair_freezeTrue" -> "spair_freezeT"
        "synthetic_large_flow_centered" -> "synthetic_large_flow_centered"
    """
    label_lower = train_name.lower()
    
    # Normalize freezeTrue/False to freezeT/F to match confusion matrix format
    label_lower = label_lower.replace('freezetrue', 'freezet')
    label_lower = label_lower.replace('freezefalse', 'freezef')
    
    # Normalize pointodyssey abbreviations
    if label_lower.startswith('ptod'):
        label_lower = label_lower.replace('ptod', 'pointodyssey', 1)
    
    return label_lower


def match_labels(vector_train_labels, vector_eval_labels, pck_train_labels, pck_bench_labels):
    """
    Create mappings between vector metrics labels and PCK performance labels.
    
    Returns:
        Tuple of (train_mapping, eval_mapping) where:
        - train_mapping: dict mapping vector_train_label -> pck_train_label
        - eval_mapping: dict mapping vector_eval_label -> pck_bench_label
    """
    train_mapping = {}
    eval_mapping = {}
    
    # Normalize all labels for matching
    vector_train_normalized = {normalize_train_dataset_name(l): l for l in vector_train_labels}
    pck_train_normalized = {normalize_training_label(l): l for l in pck_train_labels}
    
    # Debug: print what we're trying to match
    print(f"\n   Debug - Vector training labels (normalized): {sorted(vector_train_normalized.keys())}")
    print(f"   Debug - PCK training labels (normalized): {sorted(pck_train_normalized.keys())}")
    
    # Match training datasets
    for norm_label, orig_label in vector_train_normalized.items():
        # Try exact match first
        if norm_label in pck_train_normalized:
            train_mapping[orig_label] = pck_train_normalized[norm_label]
            print(f"   Matched training: {orig_label} -> {pck_train_normalized[norm_label]} (exact)")
        else:
            # Try partial match - check if normalized labels share a common prefix
            # For synthetic datasets, we want to match the full variant name
            best_match = None
            best_match_score = 0
            
            for pck_norm, pck_orig in pck_train_normalized.items():
                # Check if one contains the other (for synthetic variants)
                if norm_label in pck_norm or pck_norm in norm_label:
                    # Prefer longer matches (more specific)
                    match_score = min(len(norm_label), len(pck_norm))
                    if match_score > best_match_score:
                        best_match = pck_orig
                        best_match_score = match_score
            
            if best_match:
                train_mapping[orig_label] = best_match
                print(f"   Matched training: {orig_label} -> {best_match} (partial)")
            else:
                print(f"   Warning: Could not match training label: {orig_label} (normalized: {norm_label})")
    
    # Match evaluation datasets/benchmarks
    # Create mapping: original -> normalized for all vector eval labels
    vector_eval_to_normalized = {orig: normalize_eval_dataset_name(orig) for orig in vector_eval_labels}
    unique_normalized = set(vector_eval_to_normalized.values())
    
    pck_bench_normalized = {b.lower(): b for b in pck_bench_labels}
    
    print(f"\n   Debug - Vector eval labels (normalized, unique): {sorted(unique_normalized)}")
    print(f"   Debug - PCK benchmark labels (normalized): {sorted(pck_bench_normalized.keys())}")
    print(f"   Debug - Sample vector eval labels (original -> normalized):")
    for orig in list(vector_eval_labels)[:3]:
        print(f"      {orig} -> {vector_eval_to_normalized[orig]}")
    
    # Create a mapping from normalized labels to PCK benchmark labels
    norm_to_pck = {}
    for norm_label in unique_normalized:
        if norm_label in pck_bench_normalized:
            norm_to_pck[norm_label] = pck_bench_normalized[norm_label]
        else:
            # Try partial match
            for pck_norm, pck_orig in pck_bench_normalized.items():
                if norm_label in pck_norm or pck_norm in norm_label:
                    norm_to_pck[norm_label] = pck_orig
                    print(f"   Matched normalized eval: {norm_label} -> {pck_orig} (partial, {norm_label} <-> {pck_norm})")
                    break
    
    # Now map all original vector eval labels to PCK benchmark labels
    for orig_label, norm_label in vector_eval_to_normalized.items():
        if norm_label in norm_to_pck:
            eval_mapping[orig_label] = norm_to_pck[norm_label]
            print(f"   Matched eval: {orig_label} -> {norm_to_pck[norm_label]} (via {norm_label})")
        else:
            print(f"   Warning: Could not match eval label: {orig_label} (normalized: {norm_label})")
    
    return train_mapping, eval_mapping


def extract_correlation_data(vector_df, pck_matrix_data, train_mapping, eval_mapping, vector_metric_col):
    """
    Extract (vector_metric_value, pck_performance) pairs for all matching pairs.
    
    Args:
        vector_df: DataFrame with vector metrics
        pck_matrix_data: Dictionary with 'matrix', 'training_labels', 'benchmark_labels'
        train_mapping: Dictionary mapping vector train labels -> pck train labels
        eval_mapping: Dictionary mapping vector eval labels -> pck bench labels
        vector_metric_col: Name of the vector metric column to extract
        
    Returns:
        Tuple of (vector_values, pck_values, labels) where labels are (train_label, eval_label) tuples
    """
    vector_values = []
    pck_values = []
    labels = []
    
    pck_matrix = pck_matrix_data['matrix']
    pck_train_labels = pck_matrix_data['training_labels']
    pck_bench_labels = pck_matrix_data['benchmark_labels']
    
    # Create index maps for quick lookup
    pck_train_idx = {label: i for i, label in enumerate(pck_train_labels)}
    pck_bench_idx = {label: j for j, label in enumerate(pck_bench_labels)}
    
    # Iterate through vector metrics DataFrame
    unmatched_train = set()
    unmatched_eval = set()
    matched_pairs = []
    
    for _, row in vector_df.iterrows():
        vector_train = row['Train Dataset']
        vector_eval = row['Eval Dataset']
        
        # Map to PCK labels
        pck_train = train_mapping.get(vector_train)
        pck_eval = eval_mapping.get(vector_eval)
        
        if pck_train is None:
            unmatched_train.add(vector_train)
        if pck_eval is None:
            unmatched_eval.add(vector_eval)
        if pck_train is None or pck_eval is None:
            continue
        
        matched_pairs.append((vector_train, vector_eval, pck_train, pck_eval))
        
        # Get vector metric value
        if vector_metric_col not in row or pd.isna(row[vector_metric_col]):
            continue
        
        try:
            vector_value = float(row[vector_metric_col])
        except (ValueError, TypeError):
            continue
        
        # Get PCK performance
        pck_train_idx_val = pck_train_idx.get(pck_train)
        pck_eval_idx_val = pck_bench_idx.get(pck_eval)
        
        if pck_train_idx_val is None or pck_eval_idx_val is None:
            continue
        
        pck_value = pck_matrix[pck_train_idx_val, pck_eval_idx_val]
        if np.isnan(pck_value):
            continue
        
        # Both values are valid
        vector_values.append(vector_value)
        pck_values.append(pck_value)
        labels.append((vector_train, vector_eval))
    
    if unmatched_train:
        print(f"   Warning: {len(unmatched_train)} training datasets could not be matched: {unmatched_train}")
    if unmatched_eval:
        print(f"   Warning: {len(unmatched_eval)} eval datasets could not be matched: {unmatched_eval}")
    
    print(f"   Successfully matched {len(matched_pairs)} pairs")
    if len(matched_pairs) > 0 and len(matched_pairs) <= 5:
        print(f"   Sample matched pairs:")
        for v_train, v_eval, p_train, p_eval in matched_pairs[:3]:
            print(f"      {v_train} + {v_eval} -> {p_train} + {p_eval}")
    
    return np.array(vector_values), np.array(pck_values), labels


def plot_correlation(vector_values, pck_values, labels, output_path, vector_metric_name, 
                     metric='pck', normalized=False):
    """
    Create scatter plot of PCK performance vs vector metric.
    
    Args:
        vector_values: Array of vector metric values
        pck_values: Array of PCK performances
        labels: List of (train_label, eval_label) tuples
        output_path: Path to save the plot
        vector_metric_name: Name of the vector metric for axis label
        metric: Metric name for y-axis label
        normalized: Whether performances are normalized (0-1 range)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(vector_values, pck_values, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    # Compute correlations
    if len(vector_values) > 1:
        pearson_corr, pearson_p = pearsonr(vector_values, pck_values)
        spearman_corr, spearman_p = spearmanr(vector_values, pck_values)
        
        # Add text box with correlation statistics
        textstr = f'Pearson r = {pearson_corr:.3f}\np-value = {pearson_p:.4f}\n\n'
        textstr += f'Spearman ρ = {spearman_corr:.3f}\np-value = {spearman_p:.4f}'
        
        ax.text(0.05, 0.95, textstr,
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Format metric name for display
    metric_display = vector_metric_name.replace('_', ' ').title()
    if 'Mmd' in metric_display:
        metric_display = metric_display.replace('Mmd', 'MMD')
    if 'Js' in metric_display:
        metric_display = metric_display.replace('Js', 'JS')
    
    # Labels and title
    ax.set_xlabel(metric_display, fontsize=12)
    if normalized:
        ax.set_ylabel(f'{metric.upper()} Performance (Normalized, 0-1)', fontsize=12)
        title_suffix = " [Normalized]"
    else:
        ax.set_ylabel(f'{metric.upper()} Performance (%)', fontsize=12)
        title_suffix = ""
    ax.set_title(f'PCK Performance vs {metric_display}{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot correlation between PCK performance and vector metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--snapshots_dir',
        type=str,
        required=True,
        help='Directory containing snapshot subdirectories (for PCK performance)'
    )
    
    parser.add_argument(
        '--vector_table',
        type=str,
        default='vector_visualizations/summary_table.csv',
        help='Path to vector metrics summary table CSV (default: vector_visualizations/summary_table.csv)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='vector_visualizations/correlations/',
        help='Output directory for correlation plots (default: vector_visualizations/correlations/)'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        default='pck',
        help='Metric to use from validation results (default: pck)'
    )
    
    parser.add_argument(
        '--use_best',
        action='store_true',
        help='Use best performance instead of average (default: False, uses average)'
    )
    
    parser.add_argument(
        '--vector_metrics',
        nargs='+',
        default=None,
        help='Specific vector metrics to plot (default: all numeric metrics in table)'
    )
    
    parser.add_argument(
        '--normalized',
        action='store_true',
        help='Use column-normalized PCK performance (normalized within each benchmark)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Computing Correlation: PCK Performance vs Vector Metrics")
    print("="*60)
    
    # 1. Load vector metrics
    print(f"\n1. Loading vector metrics from: {args.vector_table}")
    vector_df = load_vector_metrics(args.vector_table)
    print(f"   Loaded {len(vector_df)} rows")
    print(f"   Columns: {list(vector_df.columns)}")
    
    # 2. Build PCK performance matrix
    print(f"\n2. Building PCK performance matrix...")
    print(f"   Snapshots directory: {args.snapshots_dir}")
    snapshot_dirs = collect_snapshots(args.snapshots_dir)
    
    if not snapshot_dirs:
        print("Error: No snapshot directories found!")
        return
    
    print(f"   Found {len(snapshot_dirs)} snapshot directory(ies)")
    
    snapshots_data = []
    for snapshot_dir in snapshot_dirs:
        data = parse_snapshot_for_confusion_matrix(snapshot_dir)
        if data and (data['best_performance'] or data.get('snapshot_path')):
            snapshots_data.append(data)
    
    if not snapshots_data:
        print("Error: No valid snapshot data found!")
        return
    
    pck_matrix_data = build_confusion_matrix_data(
        snapshots_data,
        metric=args.metric,
        use_best=args.use_best
    )
    
    print(f"   PCK matrix shape: {pck_matrix_data['matrix'].shape}")
    print(f"   Training datasets: {len(pck_matrix_data['training_labels'])}")
    print(f"   Benchmarks: {len(pck_matrix_data['benchmark_labels'])}")
    
    # Normalize PCK matrix if requested
    if args.normalized:
        print("\n   Normalizing PCK matrix (column-normalized)...")
        from make_confusion_mat import normalize_matrix_columns
        pck_matrix_data = {
            'matrix': normalize_matrix_columns(pck_matrix_data['matrix']),
            'training_labels': pck_matrix_data['training_labels'],
            'benchmark_labels': pck_matrix_data['benchmark_labels']
        }
    
    # 3. Match labels
    print("\n3. Matching labels between vector metrics and PCK performance...")
    vector_train_labels = vector_df['Train Dataset'].unique().tolist()
    vector_eval_labels = vector_df['Eval Dataset'].unique().tolist()
    
    train_mapping, eval_mapping = match_labels(
        vector_train_labels,
        vector_eval_labels,
        pck_matrix_data['training_labels'],
        pck_matrix_data['benchmark_labels']
    )
    
    print(f"   Matched {len(train_mapping)}/{len(vector_train_labels)} training datasets")
    print(f"   Matched {len(eval_mapping)}/{len(vector_eval_labels)} evaluation datasets")
    
    if len(train_mapping) == 0 or len(eval_mapping) == 0:
        print("Error: Could not match any labels!")
        return
    
    # 4. Identify vector metrics to plot
    print("\n4. Identifying vector metrics to plot...")
    # Exclude non-metric columns
    exclude_cols = {'Train Dataset', 'Eval Dataset', 'Num Vectors (Train)', 'Num Vectors (Eval)'}
    numeric_cols = [col for col in vector_df.columns 
                   if col not in exclude_cols and vector_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    if args.vector_metrics:
        # Filter to requested metrics
        vector_metrics_to_plot = [m for m in args.vector_metrics if m in numeric_cols]
        if not vector_metrics_to_plot:
            print(f"Warning: None of the requested metrics found. Available: {numeric_cols}")
            return
    else:
        vector_metrics_to_plot = numeric_cols
    
    print(f"   Will plot {len(vector_metrics_to_plot)} vector metrics: {vector_metrics_to_plot}")
    
    # 5. Create correlation plots for each vector metric
    print(f"\n5. Creating correlation plots...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store summary statistics
    summary_stats = []
    norm_suffix = "_normalized" if args.normalized else ""
    
    for vector_metric in vector_metrics_to_plot:
        print(f"\n   Processing: {vector_metric}")
        
        # Extract correlation data
        vector_values, pck_values, labels = extract_correlation_data(
            vector_df,
            pck_matrix_data,
            train_mapping,
            eval_mapping,
            vector_metric
        )
        
        if len(vector_values) == 0:
            print(f"      Warning: No matching data points found for {vector_metric}")
            continue
        
        print(f"      Found {len(vector_values)} valid (vector_metric, pck) pairs")
        
        # Compute correlations
        if len(vector_values) > 1:
            pearson_corr, pearson_p = pearsonr(vector_values, pck_values)
            spearman_corr, spearman_p = spearmanr(vector_values, pck_values)
            
            summary_stats.append({
                'vector_metric': vector_metric,
                'n_points': len(vector_values),
                'pearson_r': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_rho': spearman_corr,
                'spearman_p': spearman_p
            })
            
            print(f"      Pearson r = {pearson_corr:.3f} (p={pearson_p:.4f})")
            print(f"      Spearman ρ = {spearman_corr:.3f} (p={spearman_p:.4f})")
        
        # Create plot
        output_file = output_dir / f'pck_vs_{vector_metric}{norm_suffix}.png'
        
        plot_correlation(
            vector_values,
            pck_values,
            labels,
            output_file,
            vector_metric,
            metric=args.metric,
            normalized=args.normalized
        )
    
    # 6. Save summary statistics
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_file = output_dir / f'correlation_summary{norm_suffix}.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n6. Saved correlation summary: {summary_file}")
        print("\nSummary Statistics:")
        print(summary_df.to_string(index=False))
    
    print("\nDone!")
    print(f"All plots saved to: {output_dir}")


if __name__ == '__main__':
    main()

