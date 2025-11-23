#!/usr/bin/env python3
"""
Empirically determine optimal thresholds for each biophysical lens.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent directory to path

from genomevault.hdv_validation.query_engine importmulti_lens_with_theoretical import PreEncodedMultiLensHDV
from genomevault.hdv_validation.validation_utils import (
    NUCLEOTIDE_SIGNATURES,
    load_validated_n_positions,
    load_gdiff,
    sample_test_positions,
    get_ground_truth
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def predict_with_per_lens_thresholds(
    lens_results: Dict[str, float],
    thresholds: Dict[str, float]
) -> Tuple[str, float, Dict[str, int]]:
    """
    Multi-lens voting with per-lens thresholds.

    Args:
        lens_results: Dict of lens similarities
        thresholds: Dict of per-lens thresholds
    """
    votes = {nuc: 0 for nuc in 'ATGC'}

    for nuc, signature in NUCLEOTIDE_SIGNATURES.items():
        score = 0
        for lens_name, expected_sign in signature.items():
            observed_similarity = lens_results.get(lens_name, 0.0)
            threshold = thresholds.get(lens_name, 0.0)

            if expected_sign == 0:
                continue
            elif expected_sign > 0 and observed_similarity > threshold:
                score += 1
            elif expected_sign < 0 and observed_similarity < -threshold:
                score += 1
        votes[nuc] = score

    best_nuc = max(votes, key=votes.get)
    confidence = votes[best_nuc] / 5.0

    return best_nuc, confidence, votes


def test_threshold_config(
    hdv: PreEncodedMultiLensHDV,
    ground_truths: List[Dict],
    thresholds: Dict[str, float]
) -> float:
    """Test a specific threshold configuration and return accuracy."""
    correct = 0
    total = 0

    for gt in ground_truths:
        chrom = gt['chrom']
        pos = gt['pos']
        truth = gt['nucleotide']

        if truth is None:
            continue

        # Query all 5 lenses
        lens_results = hdv.query_position_all_lenses(chrom, pos)

        # Predict with per-lens thresholds
        pred, conf, votes = predict_with_per_lens_thresholds(lens_results, thresholds)

        if pred == truth:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def tune_lens_thresholds(
    quantization: str = 'float32',
    sample_size: int = 1000,
    seed: int = 42
):
    """
    Empirically tune thresholds for each lens.

    Strategy:
    1. Start with AT and GC at low thresholds (0.05)
    2. Sweep through threshold values for each lens
    3. Find optimal configuration
    """
    logger.info("=" * 80)
    logger.info("LENS THRESHOLD TUNING")
    logger.info("=" * 80)
    logger.info(f"Quantization: {quantization}")
    logger.info(f"Sample size: {sample_size}")
    logger.info("")

    # Paths
    base_dir = Path("/Users/rohanvinaik/genomevault")
    hdf5_dir = base_dir / "data/experimental_strands/ERR3239334/hdv_encoding"
    gdiff_path = base_dir / "data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz"
    guide_fasta_dir = base_dir / "data/guide_strands"
    validated_n_path = base_dir / "data/experimental_strands/ERR3239334/validated_n_positions.json"

    # Load H5 file
    if quantization == 'float32':
        hdf5_path = hdf5_dir / "encoded_genome_5lenses_3d.h5"
    else:
        hdf5_path = hdf5_dir / f"encoded_genome_5lenses_3d_{quantization}.h5"

    logger.info("Loading shared data...")
    gdiff, variant_index = load_gdiff(gdiff_path)
    validated_n_positions = load_validated_n_positions(validated_n_path)
    logger.info(f"  ✓ Loaded {len(variant_index):,} variants from GDiff")
    logger.info(f"  ✓ Loaded {len(validated_n_positions)} validated N positions")
    logger.info("")

    # Load HDV validator
    logger.info(f"Loading {quantization} HDV encoder...")
    hdv = PreEncodedMultiLensHDV(
        hdf5_path,
        guide_fasta_dir=guide_fasta_dir,
        quantization=quantization
    )
    logger.info("  ✓ Loaded")
    logger.info("")

    # Sample test positions
    logger.info("Sampling test positions...")
    chunk_keys = [k.decode('utf-8') if isinstance(k, bytes) else k
                  for k in hdv.h5_file['chunk_keys'][()]]

    test_positions, high_n_set = sample_test_positions(
        chunk_keys,
        validated_n_positions,
        sample_size,
        seed=seed
    )
    logger.info(f"  ✓ Sampled {len(test_positions)} positions")
    logger.info("")

    # Compute ground truth
    logger.info("Computing ground truth...")
    import pysam
    exp_bam_path = base_dir / "data/experimental_strands/ERR3239334/alignment/k11_bams/experimental_vs_ref1.sorted.bam"
    exp_bam = pysam.AlignmentFile(str(exp_bam_path), 'rb') if exp_bam_path.exists() else None

    if exp_bam is None:
        logger.error(f"Experimental BAM not found: {exp_bam_path}")
        logger.error("Cannot compute ground truth without BAM file")
        return

    ground_truths = []
    for chrom, pos in test_positions:
        truth, guide_idx, has_n = get_ground_truth(
            chrom, pos, variant_index, exp_bam,
            gdiff.get("region_guide_map", {})
        )
        if truth and truth in 'ATGC':
            ground_truths.append({
                'chrom': chrom,
                'pos': pos,
                'nucleotide': truth
            })

    logger.info(f"  ✓ {len(ground_truths)} positions with valid ground truth")
    logger.info("")

    # Define threshold sweep ranges
    # For float32/int8: test 0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2
    # For int4: scale down by ~18× (127/7), use finer granularity
    # For binary: scale down by ~127× (127/1), use very fine granularity

    if quantization == 'float32' or quantization == 'int8':
        threshold_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    elif quantization == 'int4':
        # Finer granularity for int4: ~18× smaller range
        # Map float32 [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2] → int4 scale
        threshold_values = [
            0.0,
            0.0005,  # ~0.01/18
            0.001,   # ~0.02/18
            0.0028,  # ~0.05/18
            0.0055,  # ~0.1/18
            0.0083,  # ~0.15/18
            0.011,   # ~0.2/18
            0.015,   # Extra: higher threshold
            0.020,   # Extra: even higher
        ]
    elif quantization == 'binary':
        # Very fine granularity for binary: ~127× smaller range
        # Map float32 [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2] → binary scale
        threshold_values = [
            0.0,
            0.00008,  # ~0.01/127
            0.00016,  # ~0.02/127
            0.0004,   # ~0.05/127
            0.0008,   # ~0.1/127
            0.0012,   # ~0.15/127
            0.0016,   # ~0.2/127
            0.0020,   # Extra: higher threshold
            0.0025,   # Extra: even higher
        ]
    else:
        threshold_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

    logger.info(f"Threshold sweep values: {threshold_values}")
    logger.info("")

    # Test each lens independently first
    logger.info("=" * 80)
    logger.info("PHASE 1: Individual Lens Threshold Tuning")
    logger.info("=" * 80)
    logger.info("")

    optimal_thresholds = {}
    lens_names = ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']

    for lens_name in lens_names:
        logger.info(f"Tuning {lens_name} lens...")

        best_threshold = 0.0
        best_accuracy = 0.0
        results = []

        for threshold_val in threshold_values:
            # Set this lens to threshold_val, others to 0.0
            test_thresholds = {ln: 0.0 for ln in lens_names}
            test_thresholds[lens_name] = threshold_val

            accuracy = test_threshold_config(hdv, ground_truths, test_thresholds)
            results.append({
                'threshold': threshold_val,
                'accuracy': accuracy
            })

            logger.info(f"  {lens_name}={threshold_val:.4f}: {accuracy:.2%}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold_val

        optimal_thresholds[lens_name] = best_threshold
        logger.info(f"  ✓ Best threshold for {lens_name}: {best_threshold:.4f} (accuracy: {best_accuracy:.2%})")
        logger.info("")

    logger.info("=" * 80)
    logger.info("PHASE 2: Test Optimal Configuration")
    logger.info("=" * 80)
    logger.info("")

    logger.info("Optimal per-lens thresholds:")
    for lens_name, threshold in optimal_thresholds.items():
        logger.info(f"  {lens_name}: {threshold:.4f}")
    logger.info("")

    final_accuracy = test_threshold_config(hdv, ground_truths, optimal_thresholds)
    logger.info(f"Final accuracy with optimal thresholds: {final_accuracy:.2%}")
    logger.info("")

    # Compare to baseline strategies
    logger.info("=" * 80)
    logger.info("COMPARISON TO BASELINE STRATEGIES")
    logger.info("=" * 80)
    logger.info("")

    baselines = {
        'threshold_free': {ln: 0.0 for ln in lens_names},
        'uniform_0.1': {ln: 0.1 for ln in lens_names},
        'hybrid_at_gc_free': {
            'AT': 0.0,
            'GC': 0.0,
            'PuPy': 0.1,
            'AmKe': 0.1,
            'StWk': 0.1
        }
    }

    for strategy_name, thresholds in baselines.items():
        accuracy = test_threshold_config(hdv, ground_truths, thresholds)
        logger.info(f"{strategy_name:20s}: {accuracy:.2%}")

    logger.info(f"{'optimal_per_lens':20s}: {final_accuracy:.2%}")
    logger.info("")

    # Save results
    output_path = Path(f"HDV_VALIDATION_PACKAGE/architecture_testing/threshold_tuning_{quantization}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'quantization': quantization,
        'sample_size': len(ground_truths),
        'optimal_thresholds': optimal_thresholds,
        'final_accuracy': final_accuracy,
        'baselines': {
            name: {
                'thresholds': thresholds,
                'accuracy': test_threshold_config(hdv, ground_truths, thresholds)
            }
            for name, thresholds in baselines.items()
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Results saved to: {output_path}")

    exp_bam.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Tune per-lens thresholds empirically')
    parser.add_argument('--quantization', default='float32', choices=['float32', 'int8', 'int4', 'binary'])
    parser.add_argument('--samples', type=int, default=1000, help='Number of test positions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    tune_lens_thresholds(
        quantization=args.quantization,
        sample_size=args.samples,
        seed=args.seed
    )
