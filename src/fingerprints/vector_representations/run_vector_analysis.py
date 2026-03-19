"""
run_vector_analysis.py
======================
Unified script to generate flow vectors and run comparisons.

Reads configuration from YAML file and runs the complete pipeline:
1. Generate vectors for all datasets
2. Run comparisons (MMD, containment, K-means)
3. Save all results

Usage:
    python src/fingerprints/vector_representations/run_vector_analysis.py \
        --config src/configs/fingerprints/vector_analysis_config.yaml
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.fingerprints.vector_representations.generate_vectors import (
    load_vector_config,
    compute_vector_coverage_all_datasets,
    create_training_dataset_from_exp_config,
    create_eval_dataset_from_exp_config,
    compute_vector_coverage,
)
from src.fingerprints.vector_representations.vector_coverage import compare_vector_coverage
from src.fingerprints.vector_representations.vector_utils import load_vector_coverage
from slurm.generate_jobs import load_experiments, load_machine_config, load_experiment_config


# ============================================================================
# Configuration Loading
# ============================================================================

def load_analysis_config(config_path: str) -> Dict[str, Any]:
    """Load vector analysis configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


# ============================================================================
# Vector Generation
# ============================================================================

def generate_vectors_from_config(config: Dict[str, Any]) -> Path:
    """
    Generate vectors using configuration.
    
    Returns:
        Path to output directory containing vector JSON files
    """
    print("="*60)
    print("Step 1: Generating Vectors")
    print("="*60)
    
    # Reuse the dataset creation and processing logic
    # (Functions already imported at top of file)
    
    # Load machine config
    machine_config_path = Path(config['machine_config'])
    if not machine_config_path.is_absolute():
        machine_config_path = project_root / machine_config_path
    
    if not machine_config_path.exists():
        raise FileNotFoundError(f"Machine config not found: {machine_config_path}")
    
    machine_config = load_machine_config(str(machine_config_path))
    
    # Load vector config
    vector_config_path = config.get('vector_config')
    vector_config = load_vector_config(vector_config_path)
    
    # Load experiment configs
    all_experiments = []
    for exp_config_path in config['experiment_configs']:
        exp_config_path_obj = Path(exp_config_path)
        if not exp_config_path_obj.is_absolute():
            exp_config_path_obj = project_root / exp_config_path_obj
        
        if not exp_config_path_obj.exists():
            print(f"Warning: Experiment config not found: {exp_config_path_obj}, skipping...")
            continue
        
        print(f"Loading experiments from: {exp_config_path_obj}")
        experiment_config = load_experiment_config(str(exp_config_path_obj))
        experiments = load_experiments(experiment_config, machine_config)
        all_experiments.extend(experiments)
        print(f"  Found {len(experiments)} experiments")
    
    print(f"\nTotal experiments to process: {len(all_experiments)}\n")
    
    # Prepare dataset configs
    dataset_configs = []
    eval_dataset_cache = {}
    processing_opts = config.get('processing', {})
    max_samples = processing_opts.get('max_samples')
    train_sample_fraction = processing_opts.get('train_sample_fraction', 0.1)
    
    for i, exp_config in enumerate(all_experiments, 1):
        train_dataset = exp_config.get('train_dataset', 'synthetic')
        exp_name = exp_config.get('name_exp', f'exp_{i}')
        
        print(f"[{i}/{len(all_experiments)}] Processing: {exp_name}")
        print(f"  Training dataset: {train_dataset}")
        
        # Create training dataset
        try:
            dataset = create_training_dataset_from_exp_config(exp_config, split='train')
            
            dataset_config = {
                'name': exp_name,
                'dataset': dataset,
                'max_samples': max_samples,
                'sample_fraction': train_sample_fraction,
            }
            
            # No need to set use_dataloader - auto-detected in compute_vector_coverage
            
            dataset_configs.append(dataset_config)
            print(f"  ✓ Training dataset created")
        
        except Exception as e:
            print(f"  ✗ Failed to create training dataset: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Create evaluation datasets
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
                    'max_samples': max_samples,
                    'sample_fraction': None,  # Process all eval frames
                }
                
                # Check if vector file already exists (skip re-extraction for eval datasets)
                output_dir = Path(config['output']['vector_dir'])
                if not output_dir.is_absolute():
                    output_dir = project_root / output_dir
                
                eval_vector_file = output_dir / f"{eval_dataset_config['name']}_vector_coverage.json"
                if eval_vector_file.exists():
                    print(f"    ⊘ {benchmark_str} vectors already exist, skipping: {eval_vector_file.name}")
                    continue  # Skip this eval dataset
                
                # No need to set use_dataloader - auto-detected in compute_vector_coverage
                
                dataset_configs.append(eval_dataset_config)
            
            except Exception as e:
                print(f"    ✗ Failed to create {benchmark_str} dataset: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if not dataset_configs:
        print("No datasets to process!")
        return None
    
    # Generate vectors
    output_dir = Path(config['output']['vector_dir'])
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    
    print(f"\n{'='*60}")
    print(f"Generating vectors for {len(dataset_configs)} datasets...")
    print(f"{'='*60}\n")
    
    results = compute_vector_coverage_all_datasets(
        dataset_configs=dataset_configs,
        config=vector_config,
        output_dir=output_dir,
    )
    
    print("\n" + "="*60)
    print("Vector Generation Summary:")
    print("="*60)
    print(f"Processed {len(dataset_configs)} datasets")
    print(f"Output directory: {output_dir}")
    print("\nVector files saved:")
    for json_file in output_dir.glob("*_vector_coverage.json"):
        print(f"  - {json_file.name}")
    
    return output_dir


# ============================================================================
# Comparison
# ============================================================================

def find_train_eval_pairs(vector_dir: Path) -> List[Tuple[str, str]]:
    """
    Find training/evaluation pairs from vector files.
    
    Returns:
        List of (train_name, eval_name) tuples
    """
    vector_files = list(vector_dir.glob("*_vector_coverage.json"))
    
    # Extract training dataset names (those without "_eval_" in name)
    train_names = set()
    eval_to_train = {}
    
    for vec_file in vector_files:
        name = vec_file.stem.replace('_vector_coverage', '')
        if '_eval_' in name:
            # This is an eval dataset
            # Extract train name: "exp1_eval_kitti2012" -> "exp1"
            train_name = name.split('_eval_')[0]
            eval_name = name
            eval_to_train[eval_name] = train_name
        else:
            # This is a training dataset
            train_names.add(name)
    
    # Build pairs
    pairs = []
    for eval_name, train_name in eval_to_train.items():
        if train_name in train_names:
            pairs.append((train_name, eval_name))
    
    return pairs


def run_comparisons(
    vector_dir: Path,
    comparison_dir: Path,
    config: Dict[str, Any],
) -> None:
    """
    Run comparisons between training and evaluation datasets.
    
    Args:
        vector_dir: Directory containing vector JSON files
        comparison_dir: Directory to save comparison results
        config: Analysis configuration dict
    """
    print("\n" + "="*60)
    print("Step 2: Running Comparisons")
    print("="*60)
    
    comparison_opts = config.get('comparison', {})
    compare_train_to_eval = comparison_opts.get('compare_train_to_eval', True)
    compare_all_pairs = comparison_opts.get('compare_all_pairs', False)
    
    if not compare_train_to_eval and not compare_all_pairs:
        print("No comparisons requested. Skipping.")
        return
    
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Find pairs
    if compare_train_to_eval:
        pairs = find_train_eval_pairs(vector_dir)
        print(f"\nFound {len(pairs)} train/eval pairs to compare")
    else:
        pairs = []
    
    if compare_all_pairs:
        # Get all vector files
        vector_files = list(vector_dir.glob("*_vector_coverage.json"))
        all_names = [f.stem.replace('_vector_coverage', '') for f in vector_files]
        
        # Create all pairs
        all_pairs = []
        for i, name1 in enumerate(all_names):
            for name2 in all_names[i+1:]:
                all_pairs.append((name1, name2))
        
        pairs.extend(all_pairs)
        print(f"Added {len(all_pairs)} additional pairs for all-pairs comparison")
    
    if not pairs:
        print("No pairs found for comparison.")
        return
    
    # Run comparisons
    comparison_results = {}
    name_template = comparison_opts.get('comparison_name_template', '{train_name}_vs_{eval_name}')
    
    for train_name, eval_name in pairs:
        train_path = vector_dir / f"{train_name}_vector_coverage.json"
        eval_path = vector_dir / f"{eval_name}_vector_coverage.json"
        
        if not train_path.exists():
            print(f"Warning: Training vector file not found: {train_path}")
            continue
        
        if not eval_path.exists():
            print(f"Warning: Eval vector file not found: {eval_path}")
            continue
        
        print(f"\nComparing: {train_name} vs {eval_name}")
        
        # Generate comparison name
        comp_name = name_template.format(
            train_name=train_name,
            eval_name=eval_name,
        )
        output_path = comparison_dir / f"{comp_name}_comparison.json"
        
        try:
            results = compare_vector_coverage(
                train_path=train_path,
                eval_path=eval_path,
                output_path=output_path,
                mmd_sigma=comparison_opts.get('mmd_sigma', 1.0),
                mmd_use_multiple_bandwidths=comparison_opts.get('mmd_use_multiple_bandwidths', True),
                mmd_max_vectors=comparison_opts.get('mmd_max_vectors', None),
                kmeans_n_clusters=comparison_opts.get('kmeans_n_clusters', 50),
            )
            comparison_results[comp_name] = results
            print(f"  ✓ Saved: {output_path}")
        
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("Comparison Summary:")
    print("="*60)
    print(f"Completed {len(comparison_results)} comparisons")
    print(f"Output directory: {comparison_dir}")
    print("\nComparison files saved:")
    for json_file in comparison_dir.glob("*_comparison.json"):
        print(f"  - {json_file.name}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to run complete vector analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Run complete vector analysis pipeline (generate vectors and run comparisons)"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='src/configs/fingerprints/vector_analysis_config.yaml',
        help='Path to analysis configuration YAML file',
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_analysis_config(args.config)
    
    # Step 1: Generate vectors (if requested)
    vector_dir = None
    if config.get('processing', {}).get('generate_vectors', True):
        vector_dir = generate_vectors_from_config(config)
        if vector_dir is None:
            print("Vector generation failed or skipped.")
            return
    else:
        # Use existing vectors
        vector_dir = Path(config['output']['vector_dir'])
        if not vector_dir.is_absolute():
            vector_dir = project_root / vector_dir
        
        if not vector_dir.exists():
            raise FileNotFoundError(f"Vector directory not found: {vector_dir}")
        
        print(f"Using existing vectors from: {vector_dir}")
    
    # Step 2: Run comparisons
    comparison_dir = Path(config['output']['comparison_dir'])
    if not comparison_dir.is_absolute():
        comparison_dir = project_root / comparison_dir
    
    run_comparisons(vector_dir, comparison_dir, config)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Vectors: {vector_dir}")
    print(f"Comparisons: {comparison_dir}")


if __name__ == "__main__":
    main()

