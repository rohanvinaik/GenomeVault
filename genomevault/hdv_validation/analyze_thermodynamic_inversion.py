#!/usr/bin/env python3
"""
Analyze and correct thermodynamic inversion errors.

Strong→Weak Confusion: Model confuses G/C (strong bonds) with A/T (weak bonds)
while maintaining purine/pyrimidine class.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_common_errors(common_errors_path: Path):
    """Analyze strong→weak confusion pattern in common errors."""

    with open(common_errors_path, 'r') as f:
        data = json.load(f)

    errors = data['errors']
    logger.info(f"Analyzing {len(errors)} common errors (all quantizations wrong)")
    logger.info("")

    # Group by ground_truth → predicted (for each quantization)
    for quant in ['float32', 'int8', 'int4', 'binary']:
        logger.info(f"=" * 80)
        logger.info(f"QUANTIZATION: {quant.upper()}")
        logger.info(f"=" * 80)
        logger.info("")

        confusion = defaultdict(list)
        for err in errors:
            gt = err['ground_truth']
            pred = err['predictions'][quant]
            key = f"{gt}→{pred}"
            confusion[key].append(err)

        # Analyze by nucleotide class
        strong_nts = ['G', 'C']  # Strong hydrogen bonds
        weak_nts = ['A', 'T']    # Weak hydrogen bonds

        logger.info("Confusion Matrix:")
        for gt in ['G', 'C', 'A', 'T']:
            gt_is_strong = gt in strong_nts
            logger.info(f"\n{gt} ({'Strong' if gt_is_strong else 'Weak'}):")

            pred_counts = {}
            for key, err_list in confusion.items():
                if key.startswith(f"{gt}→"):
                    pred = key.split('→')[1]
                    pred_counts[pred] = len(err_list)

            total = sum(pred_counts.values())
            if total > 0:
                for pred in sorted(pred_counts.keys()):
                    pred_is_strong = pred in strong_nts
                    count = pred_counts[pred]
                    pct = 100 * count / total

                    # Mark thermodynamic inversions
                    if gt_is_strong != pred_is_strong:
                        marker = "⚠️ INVERSION"
                    else:
                        marker = "✓ Same class"

                    logger.info(f"  →{pred}: {count:3d}/{total} ({pct:5.1f}%)  {marker}")

        # Analyze StWk lens behavior
        logger.info("")
        logger.info("StWk Lens Statistics:")
        for gt in ['G', 'C', 'A', 'T']:
            stWk_values = []
            for err in errors:
                if err['ground_truth'] == gt:
                    lens = err['lens_results'][quant]
                    stWk_values.append(lens.get('StWk', 0))

            if stWk_values:
                avg = sum(stWk_values) / len(stWk_values)
                gt_is_strong = gt in strong_nts
                expected_sign = "+" if gt_is_strong else "-"
                actual_sign = "+" if avg > 0 else "-"

                match = "✓" if expected_sign == actual_sign else "✗ INVERTED"
                logger.info(f"  {gt}: avg={avg:+.3f}  (expected {expected_sign})  {match}")

        logger.info("")


def discover_stWk_flip_signature(
    predictions_path: Path,
    quantization: str,
    output_dir: Path
):
    """
    Discover signature for StWk lens inversion correction.

    Hypothesis: Some positions have inverted StWk lens, causing strong→weak confusion.
    """

    logger.info("=" * 80)
    logger.info(f"DISCOVERING StWk FLIP SIGNATURE ({quantization.upper()})")
    logger.info("=" * 80)
    logger.info("")

    with open(predictions_path, 'r') as f:
        results = json.load(f)

    # Split into errors and correct predictions
    errors = [r for r in results if not r['correct']]
    correct = [r for r in results if r['correct']]

    logger.info(f"Loaded {len(errors)} errors, {len(correct)} correct predictions")
    logger.info("")

    # Test flip_StWk transform
    strong_nts = ['G', 'C']
    weak_nts = ['A', 'T']

    # Find errors where flipping StWk would help
    fixable = []
    for err in errors:
        gt = err['ground_truth']
        pred = err['predicted']
        lens = err['lens_results']

        # Check if this is a strong→weak or weak→strong confusion
        gt_is_strong = gt in strong_nts
        pred_is_strong = pred in strong_nts

        if gt_is_strong == pred_is_strong:
            continue  # Not a thermodynamic inversion

        # Simulate flipping StWk sign
        flipped_lens = lens.copy()
        flipped_lens['StWk'] = -lens['StWk']

        # Re-vote with flipped StWk
        from genomevault.hdv_validation.validation_utils import (
            predict_multi_lens_voting
        )

        new_pred, confidence, votes = predict_multi_lens_voting(
            flipped_lens,
            quantization=quantization
        )

        if new_pred == gt:
            fixable.append({
                'position': err['position'],
                'ground_truth': gt,
                'original_pred': pred,
                'corrected_pred': new_pred,
                'lens_results': lens,
                'flipped_lens': flipped_lens
            })

    logger.info(f"Found {len(fixable)} positions fixable by flipping StWk")

    if len(fixable) == 0:
        logger.info("No fixable errors found")
        return

    # Analyze lens magnitudes in fixable positions
    logger.info("")
    logger.info("Lens magnitudes in fixable positions:")
    for lens_name in ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']:
        values = [abs(f['lens_results'][lens_name]) for f in fixable]
        logger.info(f"  {lens_name}: min={min(values):.3f}, median={sorted(values)[len(values)//2]:.3f}, max={max(values):.3f}")

    # Test magnitude thresholds to avoid breaking correct predictions
    logger.info("")
    logger.info("Testing magnitude constraints...")

    best_signature = None
    max_fixes = 0

    for stWk_thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        # Count fixes and breaks
        fixes = 0
        breaks = 0

        # Test on fixable errors
        for f in fixable:
            if abs(f['lens_results']['StWk']) >= stWk_thresh:
                fixes += 1

        # Test on correct predictions
        for c in correct[:1000]:  # Sample for speed
            lens = c['lens_results']
            if abs(lens['StWk']) >= stWk_thresh:
                # Would we break this?
                flipped_lens = lens.copy()
                flipped_lens['StWk'] = -lens['StWk']

                from genomevault.hdv_validation.validation_utils import (
                    predict_multi_lens_voting
                )

                new_pred, _, _ = predict_multi_lens_voting(
                    flipped_lens,
                    quantization=quantization
                )

                if new_pred != c['ground_truth']:
                    breaks += 1

        logger.info(f"  StWk≥{stWk_thresh:.1f}: Fixes={fixes}/{len(fixable)}, Breaks={breaks}/1000")

        if breaks == 0 and fixes > max_fixes:
            best_signature = {
                'transform': 'flip_StWk',
                'constraints': {'StWk': stWk_thresh},
                'fixes': fixes,
                'breaks': breaks
            }
            max_fixes = fixes

    if best_signature:
        logger.info("")
        logger.info(f"✓ Best signature: flip_StWk with StWk≥{best_signature['constraints']['StWk']:.1f}")
        logger.info(f"  Fixes: {best_signature['fixes']}/{len(fixable)}")
        logger.info(f"  Breaks: {best_signature['breaks']}/1000")

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{quantization}_stWk_flip_signature.json"
        with open(output_file, 'w') as f:
            json.dump(best_signature, f, indent=2)
        logger.info(f"  Saved to: {output_file}")
    else:
        logger.info("")
        logger.info("✗ No safe signature found")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze thermodynamic inversion errors')
    parser.add_argument('--common-errors', type=str,
                       default='HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/comparison_results/common_errors.json')
    parser.add_argument('--predictions-dir', type=str,
                       default='HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/comparison_results/')
    parser.add_argument('--output-dir', type=str,
                       default='HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/thermodynamic_corrections/')

    args = parser.parse_args()

    # Analyze common errors
    common_errors_path = Path(args.common_errors)
    analyze_common_errors(common_errors_path)

    logger.info("")
    logger.info("=" * 80)
    logger.info("")

    # Discover StWk flip signatures for each quantization
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)

    for quant in ['float32', 'int8', 'int4', 'binary']:
        pred_file = predictions_dir / f"{quant}_predictions_detailed.json"
        discover_stWk_flip_signature(pred_file, quant, output_dir)
        logger.info("")


if __name__ == '__main__':
    main()
