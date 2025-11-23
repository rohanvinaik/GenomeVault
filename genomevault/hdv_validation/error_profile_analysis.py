#!/usr/bin/env python3
"""
Detailed error profile analysis for HDV quantization modes.

Analyzes:
- Error patterns by nucleotide
- Error patterns by genomic context
- Confidence distribution for correct vs incorrect predictions
- Lens-specific error patterns
- Systematic biases

Usage:
    python error_profile_analysis.py --quantization float32
    python error_profile_analysis.py --quantization int8 --sample-size 5000
"""

import argparse
import logging
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from scipy import stats

# Add parent directory to path

from genomevault.hdv_validation.query_engine import PreEncodedMultiLensHDV
from genomevault.hdv_validation.validation_utils import (
    load_validated_n_positions,
    load_gdiff,
    sample_test_positions,
    get_ground_truth,
    predict_multi_lens_voting,
    check_lens_property,
    save_results,
    LENS_DEFINITIONS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_error_profile(
    quantization='float32',
    sample_size=1000,
    seed=42,
    output_dir=None
):
    """
    Perform detailed error profile analysis for a quantization mode.
    """
    if output_dir is None:
        output_dir = Path("HDV_VALIDATION_PACKAGE/architecture_testing/error_profiles")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info(f"ERROR PROFILE ANALYSIS: {quantization.upper()}")
    logger.info("=" * 80)
    logger.info("")
    
    # Paths - use correct quantized 3D files
    base_dir = Path("data/experimental_strands/ERR3239334/hdv_encoding")
    if quantization == 'float32':
        hdf5_path = base_dir / "encoded_genome_5lenses_3d.h5"
    elif quantization == 'int8':
        hdf5_path = base_dir / "encoded_genome_5lenses_3d_int8.h5"
    elif quantization == 'int4':
        hdf5_path = base_dir / "encoded_genome_5lenses_3d_int4.h5"
    elif quantization == 'binary':
        hdf5_path = base_dir / "encoded_genome_5lenses_3d_binary.h5"
    else:
        raise ValueError(f"Unknown quantization mode: {quantization}")

    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    guide_fasta_dir = Path("/Volumes/1TBStorage/guide_strands")
    validated_n_path = Path("HDV_VALIDATION_PACKAGE/validated_n_positions.json")
    exp_bam_path = Path("data/experimental_strands/ERR3239334/alignment/k11_bams/experimental_vs_ref1.sorted.bam")

    # Load system
    logger.info(f"Loading {quantization} HDV system...")
    logger.info(f"  H5 file: {hdf5_path}")
    hdv = PreEncodedMultiLensHDV(hdf5_path, guide_fasta_dir=guide_fasta_dir, quantization=quantization)
    logger.info("  ✓ System loaded")
    logger.info("")
    
    # Load data
    gdiff, variant_index = load_gdiff(gdiff_path)
    validated_n_positions = load_validated_n_positions(validated_n_path)
    
    # Sample positions
    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        chunk_keys_bytes = f['chunk_keys'][:]
        chunk_keys = [k.decode('utf-8') for k in chunk_keys_bytes]
    
    test_positions, high_n_set = sample_test_positions(
        chunk_keys,
        validated_n_positions,
        sample_size,
        seed=seed
    )
    
    logger.info(f"Testing {len(test_positions):,} positions")
    logger.info("")
    
    # Get ground truth
    import pysam
    exp_bam = pysam.AlignmentFile(str(exp_bam_path), 'rb') if exp_bam_path.exists() else None
    region_map = gdiff.get("region_guide_map", {})
    
    ground_truths = []
    for chrom, pos in test_positions:
        gt, guide_idx, has_n = get_ground_truth(chrom, pos, variant_index, exp_bam, region_map)
        if gt and gt in 'ATGC':
            ground_truths.append({
                'chrom': chrom,
                'pos': pos,
                'nucleotide': gt,
                'has_n': has_n,
                'is_high_n': (chrom, pos) in high_n_set
            })
    
    if exp_bam:
        exp_bam.close()
    
    logger.info(f"Valid ground truth: {len(ground_truths):,} positions")
    logger.info("")
    
    # Test all positions and collect detailed data
    logger.info("Collecting prediction data...")
    
    predictions = []
    correct_by_nuc = {nuc: [] for nuc in 'ATGC'}
    incorrect_by_nuc = {nuc: [] for nuc in 'ATGC'}
    
    confidence_correct = []
    confidence_incorrect = []
    
    lens_similarities_correct = {lens: [] for lens in LENS_DEFINITIONS}
    lens_similarities_incorrect = {lens: [] for lens in LENS_DEFINITIONS}
    
    confusion_matrix = np.zeros((4, 4), dtype=int)
    nuc_to_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    idx_to_nuc = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
    
    for i, gt in enumerate(ground_truths):
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{len(ground_truths)}")
        
        chrom = gt['chrom']
        pos = gt['pos']
        truth = gt['nucleotide']
        
        # Query
        lens_results = hdv.query_position_all_lenses(chrom, pos)
        pred, conf, votes = predict_multi_lens_voting(lens_results)
        
        is_correct = (pred == truth)
        
        # Store prediction
        pred_data = {
            'position': f"{chrom}:{pos}",
            'ground_truth': truth,
            'predicted': pred,
            'confidence': conf,
            'correct': is_correct,
            'has_n': gt['has_n'],
            'lens_results': lens_results,
            'votes': votes
        }
        predictions.append(pred_data)
        
        # Update confusion matrix
        confusion_matrix[nuc_to_idx[truth], nuc_to_idx[pred]] += 1
        
        # Collect data by correctness
        if is_correct:
            correct_by_nuc[truth].append(pred_data)
            confidence_correct.append(conf)
            for lens, sim in lens_results.items():
                lens_similarities_correct[lens].append(sim)
        else:
            incorrect_by_nuc[truth].append(pred_data)
            confidence_incorrect.append(conf)
            for lens, sim in lens_results.items():
                lens_similarities_incorrect[lens].append(sim)
    
    logger.info("")
    logger.info("✓ Data collection complete")
    logger.info("")
    
    # Analyze error patterns
    logger.info("=" * 80)
    logger.info("ERROR PATTERN ANALYSIS")
    logger.info("=" * 80)
    logger.info("")
    
    total = len(predictions)
    correct = sum(1 for p in predictions if p['correct'])
    accuracy = correct / total
    
    logger.info(f"Overall Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    logger.info("")
    
    # Per-nucleotide error analysis
    logger.info("Per-Nucleotide Error Rates:")
    per_nuc_stats = {}
    for nuc in 'ATGC':
        total_nuc = len(correct_by_nuc[nuc]) + len(incorrect_by_nuc[nuc])
        correct_nuc = len(correct_by_nuc[nuc])
        error_rate = 1 - (correct_nuc / total_nuc) if total_nuc > 0 else 0
        
        per_nuc_stats[nuc] = {
            'total': total_nuc,
            'correct': correct_nuc,
            'incorrect': len(incorrect_by_nuc[nuc]),
            'accuracy': correct_nuc / total_nuc if total_nuc > 0 else 0,
            'error_rate': error_rate
        }
        
        logger.info(f"  {nuc}: {error_rate*100:5.2f}% error rate ({len(incorrect_by_nuc[nuc])}/{total_nuc} errors)")
    
    logger.info("")
    
    # Confusion matrix analysis
    logger.info("Confusion Matrix:")
    logger.info("(rows=truth, cols=predicted)")
    logger.info("")
    logger.info("       " + "".join(f"{idx_to_nuc[i]:>6s}" for i in range(4)))
    for i in range(4):
        row = f"  {idx_to_nuc[i]:>3s}  " + "".join(f"{confusion_matrix[i,j]:>6d}" for j in range(4))
        logger.info(row)
    logger.info("")
    
    # Most common misclassifications
    logger.info("Most Common Misclassifications:")
    misclassifications = []
    for i in range(4):
        for j in range(4):
            if i != j and confusion_matrix[i, j] > 0:
                misclassifications.append((
                    idx_to_nuc[i],
                    idx_to_nuc[j],
                    confusion_matrix[i, j]
                ))
    
    misclassifications.sort(key=lambda x: x[2], reverse=True)
    for truth, pred, count in misclassifications[:10]:
        pct = (count / confusion_matrix[nuc_to_idx[truth], :].sum()) * 100
        logger.info(f"  {truth} → {pred}: {count:4d} ({pct:5.2f}% of {truth} errors)")
    
    logger.info("")
    
    # Confidence analysis
    logger.info("=" * 80)
    logger.info("CONFIDENCE ANALYSIS")
    logger.info("=" * 80)
    logger.info("")
    
    conf_correct_arr = np.array(confidence_correct)
    conf_incorrect_arr = np.array(confidence_incorrect)
    
    logger.info("Confidence Distribution:")
    logger.info(f"  Correct predictions:")
    logger.info(f"    Mean:   {np.mean(conf_correct_arr):.4f}")
    logger.info(f"    Median: {np.median(conf_correct_arr):.4f}")
    logger.info(f"    Std:    {np.std(conf_correct_arr):.4f}")
    logger.info("")
    logger.info(f"  Incorrect predictions:")
    logger.info(f"    Mean:   {np.mean(conf_incorrect_arr):.4f}")
    logger.info(f"    Median: {np.median(conf_incorrect_arr):.4f}")
    logger.info(f"    Std:    {np.std(conf_incorrect_arr):.4f}")
    logger.info("")
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(conf_correct_arr, conf_incorrect_arr)
    logger.info(f"T-test (correct vs incorrect confidence):")
    logger.info(f"  t-statistic: {t_stat:.4f}")
    logger.info(f"  p-value: {p_value:.4e}")
    logger.info(f"  Result: {'Significantly different' if p_value < 0.05 else 'Not significantly different'}")
    logger.info("")
    
    # Lens-specific error patterns
    logger.info("=" * 80)
    logger.info("LENS-SPECIFIC ERROR PATTERNS")
    logger.info("=" * 80)
    logger.info("")
    
    lens_error_stats = {}
    for lens in LENS_DEFINITIONS:
        correct_sims = np.array(lens_similarities_correct[lens])
        incorrect_sims = np.array(lens_similarities_incorrect[lens])
        
        lens_error_stats[lens] = {
            'correct_mean': float(np.mean(correct_sims)),
            'correct_std': float(np.std(correct_sims)),
            'incorrect_mean': float(np.mean(incorrect_sims)),
            'incorrect_std': float(np.std(incorrect_sims))
        }
        
        logger.info(f"{lens} lens:")
        logger.info(f"  Correct:   mean={np.mean(correct_sims):>7.4f}, std={np.std(correct_sims):.4f}")
        logger.info(f"  Incorrect: mean={np.mean(incorrect_sims):>7.4f}, std={np.std(incorrect_sims):.4f}")
        
        if len(correct_sims) > 0 and len(incorrect_sims) > 0:
            t_stat, p_value = stats.ttest_ind(correct_sims, incorrect_sims)
            logger.info(f"  T-test: t={t_stat:.4f}, p={p_value:.4e}")
        logger.info("")
    
    # Generate plots
    logger.info("Generating visualization plots...")
    generate_error_plots(
        confusion_matrix,
        idx_to_nuc,
        conf_correct_arr,
        conf_incorrect_arr,
        lens_similarities_correct,
        lens_similarities_incorrect,
        output_dir,
        quantization
    )
    logger.info("  ✓ Plots saved")
    logger.info("")
    
    # Compile comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'quantization': quantization,
        'configuration': {
            'sample_size': sample_size,
            'seed': seed,
            'test_positions': len(ground_truths)
        },
        'overall': {
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'incorrect': total - correct
        },
        'per_nucleotide': per_nuc_stats,
        'confusion_matrix': {
            'matrix': confusion_matrix.tolist(),
            'nucleotides': ['A', 'T', 'G', 'C'],
            'top_misclassifications': [
                {'truth': m[0], 'predicted': m[1], 'count': m[2]}
                for m in misclassifications[:10]
            ]
        },
        'confidence_analysis': {
            'correct': {
                'mean': float(np.mean(conf_correct_arr)),
                'median': float(np.median(conf_correct_arr)),
                'std': float(np.std(conf_correct_arr)),
                'min': float(np.min(conf_correct_arr)),
                'max': float(np.max(conf_correct_arr))
            },
            'incorrect': {
                'mean': float(np.mean(conf_incorrect_arr)),
                'median': float(np.median(conf_incorrect_arr)),
                'std': float(np.std(conf_incorrect_arr)),
                'min': float(np.min(conf_incorrect_arr)),
                'max': float(np.max(conf_incorrect_arr))
            },
            'statistical_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significantly_different': bool(p_value < 0.05)
            }
        },
        'lens_error_patterns': lens_error_stats,
        'example_errors': [
            p for p in predictions if not p['correct']
        ][:50]  # First 50 errors
    }
    
    # Save report
    report_file = output_dir / f"{quantization}_error_profile.json"
    save_results(report, report_file)
    
    # Close
    hdv.close()
    
    logger.info("=" * 80)
    logger.info("ERROR PROFILE ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    
    return report


def generate_error_plots(
    confusion_matrix,
    idx_to_nuc,
    conf_correct,
    conf_incorrect,
    lens_sims_correct,
    lens_sims_incorrect,
    output_dir,
    quantization
):
    """Generate visualization plots for error analysis."""
    
    # 1. Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
    
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([idx_to_nuc[i] for i in range(4)])
    ax.set_yticklabels([idx_to_nuc[i] for i in range(4)])
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title(f'Confusion Matrix - {quantization.upper()}')
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, str(confusion_matrix[i, j]),
                          ha="center", va="center", color="black" if confusion_matrix[i, j] < confusion_matrix.max()/2 else "white")
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / f"{quantization}_confusion_matrix.png", dpi=300)
    plt.close()
    
    # 2. Confidence distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 21)
    ax.hist(conf_correct, bins=bins, alpha=0.6, label='Correct', density=True)
    ax.hist(conf_incorrect, bins=bins, alpha=0.6, label='Incorrect', density=True)
    
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Density')
    ax.set_title(f'Confidence Distribution - {quantization.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{quantization}_confidence_distribution.png", dpi=300)
    plt.close()
    
    # 3. Lens similarity comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, lens in enumerate(LENS_DEFINITIONS.keys()):
        ax = axes[i]
        
        correct_sims = lens_sims_correct[lens]
        incorrect_sims = lens_sims_incorrect[lens]
        
        bins = np.linspace(-1, 1, 41)
        ax.hist(correct_sims, bins=bins, alpha=0.6, label='Correct', density=True)
        ax.hist(incorrect_sims, bins=bins, alpha=0.6, label='Incorrect', density=True)
        
        ax.set_xlabel('Similarity')
        ax.set_ylabel('Density')
        ax.set_title(f'{lens} Lens')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplot
    axes[5].axis('off')
    
    fig.suptitle(f'Lens Similarity Distributions - {quantization.upper()}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"{quantization}_lens_distributions.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Detailed error profile analysis for HDV quantization'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        required=True,
        choices=['float32', 'int8', 'int4', 'binary'],
        help='Quantization mode to analyze'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,
        help='Number of test positions'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    analyze_error_profile(
        quantization=args.quantization,
        sample_size=args.sample_size,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
