#!/usr/bin/env python3
"""
Analyze overlap between correction signatures and calculate cumulative accuracy.

For each quantization level:
1. Load all safe signatures
2. Apply each signature individually to see which errors it fixes
3. Measure overlap (do signatures fix the same errors or different ones?)
4. Apply all signatures together to measure cumulative improvement
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_signature_overlap(
    predictions_path: Path,
    signatures_path: Path,
    quantization: str
):
    """Analyze which errors each signature fixes and measure overlap."""

    logger.info(f"=" * 80)
    logger.info(f"SIGNATURE OVERLAP ANALYSIS ({quantization.upper()})")
    logger.info(f"=" * 80)
    logger.info("")

    # Load predictions
    with open(predictions_path, 'r') as f:
        results = json.load(f)

    errors = [r for r in results if not r['correct']]
    correct = [r for r in results if r['correct']]

    logger.info(f"Total errors: {len(errors)}")
    logger.info(f"Total correct: {len(correct)}")
    logger.info("")

    # Load signatures
    with open(signatures_path, 'r') as f:
        signatures = json.load(f)

    logger.info(f"Loaded {len(signatures)} safe signatures")
    logger.info("")

    # For each signature, determine which errors it fixes
    signature_fixes: Dict[int, Set[int]] = {}

    for sig_idx, sig in enumerate(signatures):
        transform_name = sig['transform']
        constraints = sig['constraints']

        # Create transform function
        transform_fn = create_transform_fn(transform_name)

        # Test which errors this signature fixes
        fixed_error_indices = set()

        for err_idx, err in enumerate(errors):
            lens = err['lens_results']
            gt = err['ground_truth']

            # Check magnitude constraints
            passes_constraints = True
            for lens_name, thresh in constraints.items():
                if abs(lens[lens_name]) < thresh:
                    passes_constraints = False
                    break

            if not passes_constraints:
                continue

            # Apply transformation
            modified_lens = transform_fn(lens)

            # Predict
            from genomevault.hdv_validation.validation_utils import (
                predict_multi_lens_voting
            )
            pred, _, _ = predict_multi_lens_voting(modified_lens, quantization=quantization)

            if pred == gt:
                fixed_error_indices.add(err_idx)

        signature_fixes[sig_idx] = fixed_error_indices

        logger.info(f"Signature {sig_idx}: {transform_name} {constraints}")
        logger.info(f"  Fixes {len(fixed_error_indices)} errors")

    logger.info("")
    logger.info(f"=" * 80)
    logger.info(f"OVERLAP ANALYSIS")
    logger.info(f"=" * 80)
    logger.info("")

    # Calculate pairwise overlaps
    total_overlaps = 0
    total_pairs = 0

    for i in range(len(signatures)):
        for j in range(i+1, len(signatures)):
            overlap = len(signature_fixes[i] & signature_fixes[j])
            if overlap > 0:
                logger.info(f"Signatures {i} ∩ {j}: {overlap} shared fixes")
                total_overlaps += overlap
                total_pairs += 1

    if total_pairs == 0:
        logger.info("No overlaps detected - all signatures fix different errors!")
    else:
        avg_overlap = total_overlaps / total_pairs
        logger.info(f"")
        logger.info(f"Average pairwise overlap: {avg_overlap:.1f} errors")

    logger.info("")
    logger.info(f"=" * 80)
    logger.info(f"CUMULATIVE ACCURACY")
    logger.info(f"=" * 80)
    logger.info("")

    # Apply all signatures together
    all_fixed_errors = set()
    for fixes in signature_fixes.values():
        all_fixed_errors |= fixes

    baseline_errors = len(errors)
    baseline_correct = len(correct)
    baseline_total = baseline_errors + baseline_correct
    baseline_accuracy = baseline_correct / baseline_total * 100

    cumulative_correct = baseline_correct + len(all_fixed_errors)
    cumulative_total = baseline_total
    cumulative_accuracy = cumulative_correct / cumulative_total * 100

    logger.info(f"Baseline accuracy: {baseline_accuracy:.2f}% ({baseline_correct}/{baseline_total})")
    logger.info(f"Cumulative accuracy: {cumulative_accuracy:.2f}% ({cumulative_correct}/{cumulative_total})")
    logger.info(f"Improvement: +{cumulative_accuracy - baseline_accuracy:.2f}%")
    logger.info(f"Total errors fixed: {len(all_fixed_errors)}/{baseline_errors}")
    logger.info(f"Remaining errors: {baseline_errors - len(all_fixed_errors)}")
    logger.info("")

    # Show top contributing signatures
    logger.info("Top contributing signatures:")
    sorted_sigs = sorted(
        signature_fixes.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:10]

    for sig_idx, fixes in sorted_sigs:
        sig = signatures[sig_idx]
        logger.info(f"  {sig['transform']} {sig['constraints']}: {len(fixes)} fixes")


def create_transform_fn(transform_name: str):
    """Create transformation function from name."""
    transforms = {
        'flip_AT': lambda l: {**l, 'AT': -l['AT']},
        'flip_GC': lambda l: {**l, 'GC': -l['GC']},
        'flip_PuPy': lambda l: {**l, 'PuPy': -l['PuPy']},
        'flip_AmKe': lambda l: {**l, 'AmKe': -l['AmKe']},
        'flip_StWk': lambda l: {**l, 'StWk': -l['StWk']},
        'drop_AT': lambda l: {**l, 'AT': 0.0},
        'drop_GC': lambda l: {**l, 'GC': 0.0},
        'drop_PuPy': lambda l: {**l, 'PuPy': 0.0},
        'drop_AmKe': lambda l: {**l, 'AmKe': 0.0},
        'drop_StWk': lambda l: {**l, 'StWk': 0.0},
        'dampen_AT_50%': lambda l: {**l, 'AT': l['AT'] * 0.5},
        'dampen_GC_50%': lambda l: {**l, 'GC': l['GC'] * 0.5},
        'dampen_PuPy_50%': lambda l: {**l, 'PuPy': l['PuPy'] * 0.5},
        'dampen_AmKe_50%': lambda l: {**l, 'AmKe': l['AmKe'] * 0.5},
        'dampen_StWk_50%': lambda l: {**l, 'StWk': l['StWk'] * 0.5},
        'boost_AT_2x': lambda l: {**l, 'AT': l['AT'] * 2.0},
        'boost_GC_2x': lambda l: {**l, 'GC': l['GC'] * 2.0},
        'boost_PuPy_2x': lambda l: {**l, 'PuPy': l['PuPy'] * 2.0},
        'boost_AmKe_2x': lambda l: {**l, 'AmKe': l['AmKe'] * 2.0},
        'boost_StWk_2x': lambda l: {**l, 'StWk': l['StWk'] * 2.0},
    }
    return transforms[transform_name]


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze signature overlap and cumulative accuracy')
    parser.add_argument('--predictions-dir', type=str,
                       default='HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/comparison_results/')
    parser.add_argument('--signatures-dir', type=str,
                       default='HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/exhaustive_search/')

    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    signatures_dir = Path(args.signatures_dir)

    for quant in ['float32', 'int8', 'int4', 'binary']:
        pred_file = predictions_dir / f"{quant}_predictions_detailed.json"
        sig_file = signatures_dir / f"{quant}_exhaustive_search_results.json"

        if not pred_file.exists() or not sig_file.exists():
            logger.warning(f"Skipping {quant} - missing files")
            continue

        analyze_signature_overlap(pred_file, sig_file, quant)
        logger.info("")
        logger.info("")


if __name__ == '__main__':
    main()
