"""
vector_visual.py
================
Visualize vector coverage comparison results.

Generates:
- MMD summary plots
- Containment/coverage metrics
- K-means cluster comparisons
- Vector distribution visualizations (PCA/UMAP/t-SNE)
- Summary tables
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150


def load_comparison_results(comparison_path: Path) -> Dict:
    """Load comparison JSON file."""
    with open(comparison_path, 'r') as f:
        return json.load(f)


def load_vector_coverage(vector_path: Path) -> Dict:
    """Load vector coverage JSON file."""
    with open(vector_path, 'r') as f:
        return json.load(f)


def plot_mmd_summary(comparisons: List[Dict], output_path: Path):
    """Plot MMD values across different comparisons."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Extract data
    train_names = [c['train_dataset'] for c in comparisons]
    eval_names = [c['eval_dataset'] for c in comparisons]
    mmd_values = [c['mmd']['mmd2'] for c in comparisons]
    
    # Create shorter labels
    labels_short = []
    for train, eval_name in zip(train_names, eval_names):
        train_parts = train.split('_')
        eval_parts = eval_name.split('_')
        train_short = train_parts[0] if len(train_parts) > 0 else train[:8]
        eval_short = eval_parts[-1] if len(eval_parts) > 1 else eval_name[:8]
        labels_short.append(f"{train_short}\nvs\n{eval_short}")
    
    # Bar plot of MMD² values
    ax = axes[0]
    bars = ax.bar(range(len(comparisons)), mmd_values, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('MMD²')
    ax.set_title('MMD² Comparison Across Datasets')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mmd_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=7)
    
    # Heatmap of MMD per bandwidth
    ax = axes[1]
    # Collect all unique bandwidths across all comparisons
    all_bandwidths = set()
    for comp in comparisons:
        for key in comp['mmd']['mmd2_per_bandwidth'].keys():
            # Extract bandwidth value from key like 'sigma_0.364' or 'sigma_0.267'
            try:
                bandwidth = float(key.split('_')[1])
                all_bandwidths.add(bandwidth)
            except (IndexError, ValueError):
                continue
    
    bandwidths = sorted(all_bandwidths)
    if not bandwidths:
        # Fallback: try to get from first comparison
        bandwidths = sorted([float(k.split('_')[1]) for k in comparisons[0]['mmd']['mmd2_per_bandwidth'].keys()])
    
    mmd_matrix = []
    labels_short = []
    for comp in comparisons:
        row = []
        comp_bandwidths = comp['mmd']['mmd2_per_bandwidth']
        for b in bandwidths:
            # Try to find matching key - check exact match first, then closest match
            key_found = False
            best_match = None
            best_diff = float('inf')
            
            for key in comp_bandwidths.keys():
                try:
                    key_b = float(key.split('_')[1])
                    diff = abs(key_b - b)
                    if diff < 1e-6:  # Exact match
                        row.append(comp_bandwidths[key])
                        key_found = True
                        break
                    elif diff < best_diff:
                        best_diff = diff
                        best_match = key
                except (IndexError, ValueError):
                    continue
            
            if not key_found and best_match is not None and best_diff < 0.01:
                # Close enough match (within 0.01)
                row.append(comp_bandwidths[best_match])
            elif not key_found:
                # Missing bandwidth - use NaN
                row.append(np.nan)
        
        mmd_matrix.append(row)
        # Short labels for heatmap
        train_short = comp['train_dataset'].split('_')[0] if '_' in comp['train_dataset'] else comp['train_dataset'][:10]
        eval_short = comp['eval_dataset'].split('_')[-1] if '_' in comp['eval_dataset'] else comp['eval_dataset'][:10]
        labels_short.append(f"{train_short}\nvs\n{eval_short}")
    
    mmd_matrix = np.array(mmd_matrix)
    # Handle NaN values in heatmap (use masked array or set to 0)
    mmd_matrix_plot = np.ma.masked_invalid(mmd_matrix)
    if mmd_matrix_plot.mask.any():
        # If there are NaNs, use a colormap that handles them
        im = ax.imshow(mmd_matrix, aspect='auto', cmap='viridis', interpolation='nearest',
                      vmin=np.nanmin(mmd_matrix), vmax=np.nanmax(mmd_matrix))
    else:
        im = ax.imshow(mmd_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_yticks(range(len(comparisons)))
    ax.set_yticklabels(labels_short, fontsize=8)
    ax.set_xticks(range(len(bandwidths)))
    ax.set_xticklabels([f'σ={b:.2f}' for b in bandwidths], rotation=45, ha='right')
    ax.set_xlabel('Bandwidth (σ)')
    ax.set_title('MMD² per Bandwidth')
    plt.colorbar(im, ax=ax, label='MMD²')
    
    plt.tight_layout()
    plt.savefig(output_path / 'mmd_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: mmd_summary.png")


def plot_containment_metrics(comparisons: List[Dict], output_path: Path):
    """Plot containment/coverage metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Extract data - create better short labels
    labels_short = []
    full_labels = []
    for c in comparisons:
        train_name = c['train_dataset']
        eval_name = c['eval_dataset']
        # Better label shortening
        train_parts = train_name.split('_')
        eval_parts = eval_name.split('_')
        # Take first meaningful part and last meaningful part
        train_short = train_parts[0] if len(train_parts) > 0 else train_name[:8]
        eval_short = eval_parts[-1] if len(eval_parts) > 1 else eval_name[:8]
        labels_short.append(f"{train_short}\nvs\n{eval_short}")
        full_labels.append(f"{train_name} vs {eval_name}")
    
    # Coverage scores
    ax = axes[0, 0]
    coverage_scores = [c['containment']['coverage_score'] for c in comparisons]
    bars = ax.bar(range(len(comparisons)), coverage_scores, color='steelblue', alpha=0.8)
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Equal coverage (C=1)')
    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Coverage Score (C)')
    ax.set_title('Coverage Score: C = d_T→E / d_E→T')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, coverage_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    
    # Mean distances
    ax = axes[0, 1]
    d_E_to_T = [c['containment']['d_E_to_T']['mean'] for c in comparisons]
    d_T_to_E = [c['containment']['d_T_to_E']['mean'] for c in comparisons]
    x = np.arange(len(comparisons))
    width = 0.35
    bars1 = ax.bar(x - width/2, d_E_to_T, width, label='d_E→T (eval to train)', 
                   alpha=0.8, color='coral')
    bars2 = ax.bar(x + width/2, d_T_to_E, width, label='d_T→E (train to eval)', 
                   alpha=0.8, color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Mean Distance')
    ax.set_title('Asymmetric Nearest-Neighbor Distances')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Distance statistics
    ax = axes[1, 0]
    medians_E_to_T = [c['containment']['d_E_to_T']['median'] for c in comparisons]
    p95_E_to_T = [c['containment']['d_E_to_T']['p95'] for c in comparisons]
    medians_T_to_E = [c['containment']['d_T_to_E']['median'] for c in comparisons]
    p95_T_to_E = [c['containment']['d_T_to_E']['p95'] for c in comparisons]
    
    x = np.arange(len(comparisons))
    ax.plot(x, medians_E_to_T, 'o-', label='Median d_E→T', alpha=0.7, linewidth=2, markersize=6)
    ax.plot(x, p95_E_to_T, 's--', label='P95 d_E→T', alpha=0.7, linewidth=2, markersize=6)
    ax.plot(x, medians_T_to_E, 'o-', label='Median d_T→E', alpha=0.7, linewidth=2, markersize=6)
    ax.plot(x, p95_T_to_E, 's--', label='P95 d_T→E', alpha=0.7, linewidth=2, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Distance')
    ax.set_title('Distance Statistics (Median and P95)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Scatter: coverage score vs MMD
    ax = axes[1, 1]
    mmd_values = [c['mmd']['mmd2'] for c in comparisons]
    scatter = ax.scatter(coverage_scores, mmd_values, s=100, alpha=0.6, c=range(len(comparisons)), 
                        cmap='viridis', edgecolors='black', linewidth=1)
    for i, (cs, mmd) in enumerate(zip(coverage_scores, mmd_values)):
        ax.annotate(f"{i+1}", (cs, mmd), fontsize=8, alpha=0.7, 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Coverage Score (C)')
    ax.set_ylabel('MMD²')
    ax.set_title('Coverage vs Distribution Distance')
    ax.grid(alpha=0.3)
    
    # Add legend for scatter plot numbers
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='gray', markersize=8, 
                                  label=f"{i+1}: {full_labels[i]}", 
                                  linestyle='None') 
                      for i in range(len(comparisons))]
    ax.legend(handles=legend_elements, loc='best', fontsize=7, 
             bbox_to_anchor=(1.05, 1), borderaxespad=0)
    
    plt.tight_layout()
    plt.savefig(output_path / 'containment_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: containment_metrics.png")


def plot_kmeans_comparison(comparison: Dict, output_path: Path):
    """Plot K-means cluster histogram comparison for a single comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    cluster_hist_T = np.array(comparison['kmeans']['cluster_hist_T'])
    cluster_hist_E = np.array(comparison['kmeans']['cluster_hist_E'])
    n_clusters = len(cluster_hist_T)
    
    # Side-by-side bar chart
    ax = axes[0]
    x = np.arange(n_clusters)
    width = 0.35
    bars1 = ax.bar(x - width/2, cluster_hist_T, width, label='Training', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, cluster_hist_E, width, label='Eval', alpha=0.8, color='coral')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Probability')
    ax.set_title(f'Cluster Histogram Comparison: {comparison["train_dataset"]} vs {comparison["eval_dataset"]}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(x[::5])  # Show every 5th cluster label
    ax.set_xticklabels(x[::5])
    
    # Difference heatmap
    ax = axes[1]
    diff = cluster_hist_T - cluster_hist_E
    vmax = max(abs(diff))
    im = ax.imshow(diff.reshape(1, -1), aspect='auto', cmap='RdBu_r', 
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_yticks([0])
    ax.set_yticklabels(['T - E'])
    ax.set_xticks(range(0, n_clusters, 5))
    ax.set_xticklabels(range(0, n_clusters, 5))
    ax.set_xlabel('Cluster ID')
    ax.set_title('Cluster Probability Difference (Training - Eval)')
    cbar = plt.colorbar(im, ax=ax, label='Probability Difference')
    cbar.ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Add JS divergence and Wasserstein if available
    js_div = comparison['kmeans'].get('js_divergence', None)
    wasserstein = comparison['kmeans'].get('wasserstein', None)
    if js_div is not None or wasserstein is not None:
        info_text = []
        if js_div is not None:
            info_text.append(f'JS Divergence: {js_div:.4f}')
        if wasserstein is not None:
            info_text.append(f'Wasserstein: {wasserstein:.4f}')
        ax.text(0.02, 0.98, '\n'.join(info_text), transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10)
    
    plt.tight_layout()
    train_name = comparison['train_dataset']
    eval_name = comparison['eval_dataset']
    filename = f'kmeans_{train_name}_vs_{eval_name}.png'
    # Sanitize filename
    filename = filename.replace('/', '_').replace('\\', '_')
    plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_vector_distributions(
    vector_path: Path, 
    output_path: Path, 
    n_samples: int = 5000, 
    use_umap: bool = True
):
    """Visualize actual vector distributions using PCA/UMAP or t-SNE."""
    from sklearn.decomposition import PCA
    
    # Load vectors
    print(f"  Loading vectors from {vector_path.name}...")
    data = load_vector_coverage(vector_path)
    vectors = np.array(data['vectors'])
    
    # Subsample for visualization (UMAP/t-SNE are slow on large datasets)
    if len(vectors) > n_samples:
        indices = np.random.choice(len(vectors), n_samples, replace=False)
        vectors = vectors[indices]
        print(f"  Subsampled to {n_samples} vectors for visualization")
    else:
        print(f"  Using all {len(vectors)} vectors")
    
    # PCA projection (always fast)
    print("  Computing PCA...")
    pca = PCA(n_components=2)
    vectors_2d_pca = pca.fit_transform(vectors)
    
    # UMAP or t-SNE projection
    if use_umap:
        try:
            import umap
            print("  Computing UMAP (this may take a minute)...")
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            vectors_2d_manifold = reducer.fit_transform(vectors)
            manifold_name = 'UMAP'
        except ImportError:
            print("  UMAP not available, falling back to t-SNE...")
            use_umap = False
    
    if not use_umap:
        from sklearn.manifold import TSNE
        print("  Computing t-SNE (this may take a few minutes)...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        vectors_2d_manifold = reducer.fit_transform(vectors)
        manifold_name = 't-SNE'
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA plot - colored by spatial location (x, y)
    ax = axes[0]
    scatter = ax.scatter(vectors_2d_pca[:, 0], vectors_2d_pca[:, 1], 
                        c=vectors[:, 0], cmap='viridis', alpha=0.5, s=1)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title('PCA Projection (colored by normalized x position)')
    plt.colorbar(scatter, ax=ax, label='x/H (normalized)')
    
    # UMAP/t-SNE plot - colored by flow magnitude
    ax = axes[1]
    # Compute flow magnitude from u and v components
    flow_magnitude = np.sqrt(vectors[:, 2]**2 + vectors[:, 3]**2)
    scatter = ax.scatter(vectors_2d_manifold[:, 0], vectors_2d_manifold[:, 1],
                        c=flow_magnitude, cmap='plasma', alpha=0.5, s=1)
    ax.set_xlabel(f'{manifold_name} 1')
    ax.set_ylabel(f'{manifold_name} 2')
    ax.set_title(f'{manifold_name} Projection (colored by flow magnitude)')
    plt.colorbar(scatter, ax=ax, label='|u,v| magnitude')
    
    plt.tight_layout()
    dataset_name = data.get('metadata', {}).get('dataset_name', 'unknown')
    filename = f'vector_distribution_{dataset_name}.png'
    filename = filename.replace('/', '_').replace('\\', '_')
    plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_vector_distribution_comparison(
    train_vector_path: Path, 
    eval_vector_path: Path, 
    output_path: Path, 
    n_samples: int = 5000,
    use_umap: bool = True
):
    """Compare vector distributions from two datasets with overlay visualization."""
    # Load both datasets
    print(f"  Loading training vectors from {train_vector_path.name}...")
    train_data = load_vector_coverage(train_vector_path)
    train_vectors = np.array(train_data['vectors'])
    
    print(f"  Loading eval vectors from {eval_vector_path.name}...")
    eval_data = load_vector_coverage(eval_vector_path)
    eval_vectors = np.array(eval_data['vectors'])
    
    # Subsample
    if len(train_vectors) > n_samples:
        indices = np.random.choice(len(train_vectors), n_samples, replace=False)
        train_vectors = train_vectors[indices]
        print(f"  Subsampled training to {n_samples} vectors")
    if len(eval_vectors) > n_samples:
        indices = np.random.choice(len(eval_vectors), n_samples, replace=False)
        eval_vectors = eval_vectors[indices]
        print(f"  Subsampled eval to {n_samples} vectors")
    
    # Combine for joint embedding (important for comparison!)
    print("  Combining vectors for joint embedding...")
    all_vectors = np.vstack([train_vectors, eval_vectors])
    
    # Create binary labels: 0 for train, 1 for eval
    dataset_labels = np.concatenate([
        np.zeros(len(train_vectors), dtype=int),  # 0 = train
        np.ones(len(eval_vectors), dtype=int)     # 1 = eval
    ])
    
    # UMAP or t-SNE on combined vectors (skip PCA)
    if use_umap:
        try:
            import umap
            print("  Computing UMAP on combined vectors (this may take a minute)...")
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            all_vectors_2d_manifold = reducer.fit_transform(all_vectors)
            manifold_name = 'UMAP'
        except ImportError:
            use_umap = False
    
    if not use_umap:
        from sklearn.manifold import TSNE
        print("  Computing t-SNE on combined vectors (this may take a few minutes)...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        all_vectors_2d_manifold = reducer.fit_transform(all_vectors)
        manifold_name = 't-SNE'
    
    # Create comparison plot - overlay both datasets
    train_mask = dataset_labels == 0
    eval_mask = dataset_labels == 1
    
    train_name = train_data.get('metadata', {}).get('dataset_name', 'train')
    eval_name = eval_data.get('metadata', {}).get('dataset_name', 'eval')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # UMAP/t-SNE - Overlay plot
    ax.scatter(all_vectors_2d_manifold[train_mask, 0], all_vectors_2d_manifold[train_mask, 1],
              c='steelblue', alpha=0.4, s=1, label=f'Training (n={len(train_vectors)})')
    ax.scatter(all_vectors_2d_manifold[eval_mask, 0], all_vectors_2d_manifold[eval_mask, 1],
              c='coral', alpha=0.4, s=1, label=f'Eval (n={len(eval_vectors)})')
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    # Add title showing what's being compared
    ax.set_title(f'{train_name} vs {eval_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = f'vector_comparison_{train_name}_vs_{eval_name}.png'
    filename = filename.replace('/', '_').replace('\\', '_')
    plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def create_summary_table(comparisons: List[Dict], output_path: Path):
    """Create a summary table of all metrics."""
    rows = []
    for comp in comparisons:
        row = {
            'Train Dataset': comp['train_dataset'],
            'Eval Dataset': comp['eval_dataset'],
            'MMD²': comp['mmd']['mmd2'],
            'Coverage Score': comp['containment']['coverage_score'],
            'd_E→T (mean)': comp['containment']['d_E_to_T']['mean'],
            'd_E→T (median)': comp['containment']['d_E_to_T']['median'],
            'd_E→T (p95)': comp['containment']['d_E_to_T']['p95'],
            'd_T→E (mean)': comp['containment']['d_T_to_E']['mean'],
            'd_T→E (median)': comp['containment']['d_T_to_E']['median'],
            'd_T→E (p95)': comp['containment']['d_T_to_E']['p95'],
            'JS Divergence': comp['kmeans'].get('js_divergence', 'N/A'),
            'Wasserstein': comp['kmeans'].get('wasserstein', 'N/A'),
            'Num Vectors (Train)': comp['num_vectors_T'],
            'Num Vectors (Eval)': comp['num_vectors_E'],
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = output_path / 'summary_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved: summary_table.csv")
    
    # Also create a nicely formatted markdown table
    md_path = output_path / 'summary_table.md'
    with open(md_path, 'w') as f:
        f.write("# Vector Coverage Comparison Summary\n\n")
        f.write(df.to_markdown(index=False))
    print(f"  Saved: summary_table.md")


def main(
    comparison_dir: Path,
    output_dir: Path,
    vector_dir: Optional[Path] = None,
    plot_distributions: bool = True,
    plot_comparisons: bool = True,
    n_samples: int = 5000,
    use_umap: bool = True,
):
    """Generate all visualizations.
    
    Args:
        comparison_dir: Directory containing comparison JSON files
        output_dir: Directory to save visualizations
        vector_dir: Directory containing vector coverage JSON files (for distribution plots)
        plot_distributions: Whether to plot individual vector distributions
        plot_comparisons: Whether to plot side-by-side distribution comparisons
        n_samples: Number of samples to use for distribution plots
        use_umap: Use UMAP instead of t-SNE (faster, better global structure)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all comparison files
    comparison_files = sorted(list(comparison_dir.glob('*_comparison.json')))
    if not comparison_files:
        print(f"No comparison files found in {comparison_dir}")
        return
    
    comparisons = [load_comparison_results(f) for f in comparison_files]
    print(f"Loaded {len(comparisons)} comparisons")
    
    # Generate summary visualizations
    print("\n=== Generating Summary Visualizations ===")
    print("Generating MMD summary...")
    plot_mmd_summary(comparisons, output_dir)
    
    print("Generating containment metrics...")
    plot_containment_metrics(comparisons, output_dir)
    
    print("Generating K-means comparisons...")
    for comp in comparisons:
        plot_kmeans_comparison(comp, output_dir)
    
    print("Creating summary table...")
    create_summary_table(comparisons, output_dir)
    
    # Generate distribution visualizations if vector files are available
    # Skip individual distributions - only generate comparison plots
    if plot_comparisons and vector_dir is not None:
        print("\n=== Generating Distribution Comparison Visualizations ===")
        vector_dir = Path(vector_dir)
        vector_files = {f.stem.replace('_vector_coverage', ''): f 
                       for f in vector_dir.glob('*_vector_coverage.json')}
        
        # Plot side-by-side comparisons only
        print("Plotting distribution comparisons...")
        for comp in comparisons:
            train_name = comp['train_dataset']
            eval_name = comp['eval_dataset']
            
            train_path = vector_files.get(train_name)
            eval_path = vector_files.get(eval_name)
            
            if train_path and eval_path:
                try:
                    plot_vector_distribution_comparison(
                        train_path, eval_path, output_dir, 
                        n_samples=n_samples, use_umap=use_umap
                    )
                except Exception as e:
                    print(f"  Error plotting comparison {train_name} vs {eval_name}: {e}")
            else:
                print(f"  Skipping {train_name} vs {eval_name} (vector files not found)")
    
    print(f"\n✓ All visualizations saved to {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize vector coverage comparison results')
    parser.add_argument('--comparison_dir', type=str, default='vector_comparisons',
                       help='Directory containing comparison JSON files')
    parser.add_argument('--output_dir', type=str, default='vector_visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--vector_dir', type=str, default='vector_coverage',
                       help='Directory containing vector coverage JSON files (for distribution plots)')
    parser.add_argument('--no_distributions', action='store_true',
                       help='Skip individual distribution plots')
    parser.add_argument('--no_comparisons', action='store_true',
                       help='Skip side-by-side distribution comparison plots')
    parser.add_argument('--n_samples', type=int, default=5000,
                       help='Number of samples to use for distribution plots')
    parser.add_argument('--use_tsne', action='store_true',
                       help='Use t-SNE instead of UMAP (slower)')
    
    args = parser.parse_args()
    
    main(
        comparison_dir=Path(args.comparison_dir),
        output_dir=Path(args.output_dir),
        vector_dir=Path(args.vector_dir) if args.vector_dir else None,
        plot_distributions=not args.no_distributions,
        plot_comparisons=not args.no_comparisons,
        n_samples=args.n_samples,
        use_umap=not args.use_tsne,
    )

