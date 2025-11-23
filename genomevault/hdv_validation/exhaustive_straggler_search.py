#!/usr/bin/env python3
"""
Exhaustive search for correction signatures across ALL errors.

Searches the full space of:
- All 5 lenses (AT, GC, PuPy, AmKe, StWk)
- All quantization levels (float32, int8, int4, binary)
- All possible transformations (flip, drop, dampen, boost)
- All magnitude threshold combinations

Goal: Find any remaining safe corrections we might have missed.
"""

import json
import logging
import itertools
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def exhaustive_search(predictions_path: Path, quantization: str, output_dir: Path):
    """
    Exhaustive search across all possible single-lens transformations.
    """

    logger.info(f"=" * 80)
    logger.info(f"EXHAUSTIVE STRAGGLER SEARCH ({quantization.upper()})")
    logger.info(f"=" * 80)
    logger.info("")

    with open(predictions_path, 'r') as f:
        results = json.load(f)

    errors = [r for r in results if not r['correct']]
    correct = [r for r in results if r['correct']]

    logger.info(f"Searching across {len(errors)} errors")
    logger.info(f"Testing against {len(correct)} correct predictions")
    logger.info("")

    # Single-lens transformations to test
    transformations = [
        ('flip_AT', lambda l: {**l, 'AT': -l['AT']}),
        ('flip_GC', lambda l: {**l, 'GC': -l['GC']}),
        ('flip_PuPy', lambda l: {**l, 'PuPy': -l['PuPy']}),
        ('flip_AmKe', lambda l: {**l, 'AmKe': -l['AmKe']}),
        ('flip_StWk', lambda l: {**l, 'StWk': -l['StWk']}),
        ('drop_AT', lambda l: {**l, 'AT': 0.0}),
        ('drop_GC', lambda l: {**l, 'GC': 0.0}),
        ('drop_PuPy', lambda l: {**l, 'PuPy': 0.0}),
        ('drop_AmKe', lambda l: {**l, 'AmKe': 0.0}),
        ('drop_StWk', lambda l: {**l, 'StWk': 0.0}),
        ('dampen_AT_50%', lambda l: {**l, 'AT': l['AT'] * 0.5}),
        ('dampen_GC_50%', lambda l: {**l, 'GC': l['GC'] * 0.5}),
        ('dampen_PuPy_50%', lambda l: {**l, 'PuPy': l['PuPy'] * 0.5}),
        ('dampen_AmKe_50%', lambda l: {**l, 'AmKe': l['AmKe'] * 0.5}),
        ('dampen_StWk_50%', lambda l: {**l, 'StWk': l['StWk'] * 0.5}),
        ('boost_AT_2x', lambda l: {**l, 'AT': l['AT'] * 2.0}),
        ('boost_GC_2x', lambda l: {**l, 'GC': l['GC'] * 2.0}),
        ('boost_PuPy_2x', lambda l: {**l, 'PuPy': l['PuPy'] * 2.0}),
        ('boost_AmKe_2x', lambda l: {**l, 'AmKe': l['AmKe'] * 2.0}),
        ('boost_StWk_2x', lambda l: {**l, 'StWk': l['StWk'] * 2.0}),
    ]

    # Magnitude thresholds to test
    if quantization == 'float32':
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    elif quantization == 'int8':
        thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    elif quantization == 'int4':
        thresholds = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]
    else:  # binary
        thresholds = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]

    # Lenses to constrain
    lenses = ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']

    best_signatures = []

    logger.info(f"Testing {len(transformations)} transformations × {len(thresholds)}^5 magnitude combinations")
    logger.info(f"(Will focus on most promising combinations to keep runtime reasonable)")
    logger.info("")

    # Test each transformation
    for transform_name, transform_fn in transformations:
        logger.info(f"Testing: {transform_name}")

        # For each transformation, test different magnitude constraints
        # Start with no constraints, then add constraints on each lens

        # Test with no magnitude constraints first
        fixes, breaks = test_transform(errors, correct, transform_fn, {}, quantization)

        if fixes > 0:
            logger.info(f"  No constraints: Fixes={fixes}, Breaks={breaks}")

            if breaks == 0:
                best_signatures.append({
                    'transform': transform_name,
                    'constraints': {},
                    'fixes': fixes,
                    'breaks': breaks
                })
                logger.info(f"    ✓ SAFE SIGNATURE FOUND!")

        # Test with single-lens constraints
        for lens in lenses:
            best_for_this_lens = None
            max_fixes = 0

            for thresh in thresholds:
                constraints = {lens: thresh}
                fixes, breaks = test_transform(errors, correct, transform_fn, constraints, quantization)

                if breaks == 0 and fixes > max_fixes:
                    max_fixes = fixes
                    best_for_this_lens = {
                        'transform': transform_name,
                        'constraints': constraints,
                        'fixes': fixes,
                        'breaks': breaks
                    }

            if best_for_this_lens:
                logger.info(f"  Best with {lens} constraint: {best_for_this_lens['constraints']} → Fixes={best_for_this_lens['fixes']}")
                best_signatures.append(best_for_this_lens)

    logger.info("")
    logger.info(f"=" * 80)
    logger.info(f"SAFE SIGNATURES FOUND: {len(best_signatures)}")
    logger.info(f"=" * 80)
    logger.info("")

    for sig in sorted(best_signatures, key=lambda x: -x['fixes']):
        logger.info(f"{sig['transform']}: Fixes={sig['fixes']}, Constraints={sig['constraints']}")

    # Save results
    output_file = output_dir / f"{quantization}_exhaustive_search_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(best_signatures, f, indent=2)
    logger.info(f"")
    logger.info(f"Saved to: {output_file}")


def test_transform(errors, correct, transform_fn, constraints, quantization):
    """Test a transformation with magnitude constraints."""
    from genomevault.hdv_validation.validation_utils import (
        predict_multi_lens_voting
    )

    fixes = 0
    breaks = 0

    # Test on errors
    for err in errors:
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
        pred, _, _ = predict_multi_lens_voting(modified_lens, quantization=quantization)

        if pred == gt:
            fixes += 1

    # Test on ALL correct predictions to ensure no breaks
    for c in correct:  # Test on ALL, not just first 1000
        lens = c['lens_results']
        gt = c['ground_truth']

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
        pred, _, _ = predict_multi_lens_voting(modified_lens, quantization=quantization)

        if pred != gt:
            breaks += 1

    return fixes, breaks


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Exhaustive straggler search')
    parser.add_argument('--predictions-dir', type=str,
                       default='HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/comparison_results/')
    parser.add_argument('--output-dir', type=str,
                       default='HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/exhaustive_search/')

    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)

    for quant in ['float32', 'int8', 'int4', 'binary']:
        pred_file = predictions_dir / f"{quant}_predictions_detailed.json"
        exhaustive_search(pred_file, quant, output_dir)
        logger.info("")
        logger.info("")


if __name__ == '__main__':
    main()
