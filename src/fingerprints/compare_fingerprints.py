"""
compare_fingerprints.py
======================
Compare validation benchmarks against training datasets.

Generates comparison plots for:
- spair_val vs [spair_train, pointodyssey_train, flyingthings_train, synthetic_train]
- tss_val vs [spair_train, pointodyssey_train, flyingthings_train, synthetic_train]
- pfpascal_val vs [spair_train, pointodyssey_train, flyingthings_train, synthetic_train]

Usage:
    python src/fingerprints/compare_fingerprints.py --fingerprints_dir ./fingerprints
    python src/fingerprints/compare_fingerprints.py --fingerprints_dir ./fingerprints --val_benchmark spair
"""

import argparse
from pathlib import Path
from typing import List, Optional

from src.fingerprints.flow_fingerprint import load_stats_json
from src.fingerprints.plot_flow_fingerprint import plot_overlay


def find_fingerprint_file(fingerprints_dir: Path, dataset_name: str, split: str) -> Optional[Path]:
    """Find fingerprint JSON file for a dataset and split."""
    # Try common naming patterns
    patterns = [
        f"{dataset_name}_{split}_fingerprint.json",  # spair_val_fingerprint.json
        f"{dataset_name}_{split}.json",              # spair_val.json
    ]
    
    for pattern in patterns:
        path = fingerprints_dir / pattern
        if path.exists():
            return path
    
    # If not found, list available files for debugging
    available = list(fingerprints_dir.glob("*_fingerprint.json"))
    if available:
        print(f"  Available fingerprint files: {[f.name for f in available]}")
    
    return None


def compare_benchmark_to_training(
    fingerprints_dir: Path,
    val_benchmark: str,
    val_split: str,
    train_datasets: List[str],
    train_split: str,
    output_dir: Optional[Path] = None,
):
    """
    Compare a validation benchmark against multiple training datasets.
    
    Args:
        fingerprints_dir: Directory containing fingerprint JSON files
        val_benchmark: Validation benchmark name (e.g., 'spair', 'tss', 'pfpascal')
        val_split: Validation split (typically 'val')
        train_datasets: List of training dataset names to compare against
        train_split: Training split (typically 'train')
        output_dir: Output directory for comparison plots (default: fingerprints_dir/comparisons/{val_benchmark}_val)
    """
    # Find validation fingerprint
    val_fp_path = find_fingerprint_file(fingerprints_dir, val_benchmark, val_split)
    if val_fp_path is None:
        print(f"⚠ Warning: Could not find fingerprint for {val_benchmark} {val_split}")
        print(f"  Looked in: {fingerprints_dir}")
        return False
    
    # Load validation stats
    try:
        val_stats = load_stats_json(str(val_fp_path))
        print(f"✓ Loaded {val_benchmark} {val_split}: {val_fp_path}")
    except Exception as e:
        print(f"✗ Failed to load {val_benchmark} {val_split}: {e}")
        return False
    
    # Find and load training fingerprints
    train_stats_list = []
    train_labels = []
    
    for train_ds in train_datasets:
        train_fp_path = find_fingerprint_file(fingerprints_dir, train_ds, train_split)
        if train_fp_path is None:
            print(f"⚠ Warning: Could not find fingerprint for {train_ds} {train_split}, skipping")
            continue
        
        try:
            train_stats = load_stats_json(str(train_fp_path))
            train_stats_list.append(train_stats)
            train_labels.append(f"{train_ds}_{train_split}")
            print(f"✓ Loaded {train_ds} {train_split}: {train_fp_path}")
        except Exception as e:
            print(f"✗ Failed to load {train_ds} {train_split}: {e}")
            continue
    
    if not train_stats_list:
        print(f"✗ No training datasets found for comparison")
        return False
    
    # Prepare stats and labels for overlay plot
    all_stats = [val_stats] + train_stats_list
    all_labels = [f"{val_benchmark}_{val_split}"] + train_labels
    
    # Create output directory
    if output_dir is None:
        output_dir = fingerprints_dir / "comparisons" / f"{val_benchmark}_{val_split}_vs_training"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison plots
    print(f"\nGenerating comparison plots: {val_benchmark} {val_split} vs training datasets")
    print(f"  Training datasets: {', '.join(train_labels)}")
    print(f"  Output directory: {output_dir}")
    
    try:
        plot_overlay(all_stats, all_labels, str(output_dir))
        print(f"✓ Comparison plots saved to: {output_dir}")
        return True
    except Exception as e:
        print(f"✗ Failed to generate comparison plots: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Compare validation benchmarks against training datasets'
    )
    
    parser.add_argument(
        '--fingerprints_dir',
        type=str,
        default='./fingerprints',
        help='Directory containing fingerprint JSON files'
    )
    
    parser.add_argument(
        '--val_benchmark',
        type=str,
        default='all',
        choices=['all', 'spair', 'tss', 'pfpascal', 'pointodyssey'],
        help='Validation benchmark to compare (or "all" for all benchmarks)'
    )
    
    parser.add_argument(
        '--val_split',
        type=str,
        default='val',
        help='Validation split (default: val)'
    )
    
    parser.add_argument(
        '--train_split',
        type=str,
        default='train',
        help='Training split (default: train)'
    )
    
    parser.add_argument(
        '--train_datasets',
        type=str,
        nargs='+',
        default=None,
        help='Training datasets to compare against (default: spair pointodyssey flyingthings synthetic)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Base output directory for comparisons (default: fingerprints_dir/comparisons)'
    )
    
    args = parser.parse_args()
    
    fingerprints_dir = Path(args.fingerprints_dir)
    if not fingerprints_dir.exists():
        print(f"Error: Fingerprints directory does not exist: {fingerprints_dir}")
        return
    
    # Default training datasets
    if args.train_datasets is None:
        train_datasets = ['spair', 'pointodyssey', 'flyingthings', 'synthetic']
    else:
        train_datasets = args.train_datasets
    
    # Determine which validation benchmarks to process
    if args.val_benchmark == 'all':
        val_benchmarks = ['spair', 'tss', 'pfpascal', 'pointodyssey']
    else:
        val_benchmarks = [args.val_benchmark]
    
    print("="*60)
    print("Flow Fingerprint Comparisons")
    print("="*60)
    print(f"Fingerprints directory: {fingerprints_dir}")
    print(f"Validation benchmarks: {val_benchmarks}")
    print(f"Training datasets: {train_datasets}")
    print(f"Validation split: {args.val_split}")
    print(f"Training split: {args.train_split}")
    print("="*60 + "\n")
    
    # Process each validation benchmark
    results = {}
    for val_benchmark in val_benchmarks:
        print(f"\n{'='*60}")
        print(f"Comparing: {val_benchmark} {args.val_split}")
        print(f"{'='*60}")
        
        output_dir = None
        if args.output_dir:
            output_dir = Path(args.output_dir) / f"{val_benchmark}_{args.val_split}_vs_training"
        
        success = compare_benchmark_to_training(
            fingerprints_dir=fingerprints_dir,
            val_benchmark=val_benchmark,
            val_split=args.val_split,
            train_datasets=train_datasets,
            train_split=args.train_split,
            output_dir=output_dir,
        )
        
        results[val_benchmark] = success
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for val_benchmark, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {val_benchmark} {args.val_split}")
    
    successful = sum(1 for s in results.values() if s)
    print(f"\nCompleted {successful}/{len(results)} comparisons")


if __name__ == "__main__":
    main()

