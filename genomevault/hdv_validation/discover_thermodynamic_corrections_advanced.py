#!/usr/bin/env python3
"""
Advanced thermodynamic inversion correction discovery.

Creative approaches to fix Strong→Weak confusion without breaking correct predictions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def predict_with_transform(lens_results: Dict[str, float], transform: str, quantization: str):
    """Apply transformation and re-predict."""
    from genomevault.hdv_validation.validation_utils import (
        predict_multi_lens_voting
    )

    modified_lens = lens_results.copy()

    if transform == 'flip_StWk':
        modified_lens['StWk'] = -lens_results['StWk']
    elif transform == 'dampen_StWk_50%':
        modified_lens['StWk'] = lens_results['StWk'] * 0.5
    elif transform == 'dampen_StWk_25%':
        modified_lens['StWk'] = lens_results['StWk'] * 0.25
    elif transform == 'drop_StWk':
        modified_lens['StWk'] = 0.0
    elif transform == 'boost_other_lenses_2x':
        for lens in ['AT', 'GC', 'PuPy', 'AmKe']:
            modified_lens[lens] = lens_results[lens] * 2.0
    elif transform == 'flip_StWk_if_weak_consensus':
        # Only flip if other 4 lenses agree on purine/pyrimidine class
        # and suggest opposite strength from StWk
        pass  # Will be handled specially

    pred, conf, votes = predict_multi_lens_voting(modified_lens, quantization=quantization)
    return pred, conf


def discover_conditional_stWk_flip(
    predictions_path: Path,
    quantization: str,
    output_dir: Path
):
    """
    Discover CONDITIONAL StWk flip - only when other lenses strongly disagree.

    Hypothesis: StWk inversion is only a problem when it contradicts a strong consensus
    from the other 4 lenses.
    """

    logger.info("=" * 80)
    logger.info(f"CONDITIONAL StWk FLIP DISCOVERY ({quantization.upper()})")
    logger.info("=" * 80)
    logger.info("")

    with open(predictions_path, 'r') as f:
        results = json.load(f)

    errors = [r for r in results if not r['correct']]
    correct = [r for r in results if r['correct']]

    strong_nts = ['G', 'C']
    weak_nts = ['A', 'T']

    # Find thermodynamic inversion errors
    inversion_errors = []
    for err in errors:
        gt = err['ground_truth']
        pred = err['predicted']

        gt_is_strong = gt in strong_nts
        pred_is_strong = pred in strong_nts

        if gt_is_strong != pred_is_strong:
            inversion_errors.append(err)

    logger.info(f"Found {len(inversion_errors)} thermodynamic inversion errors")
    logger.info(f"  (out of {len(errors)} total errors)")
    logger.info("")

    # Strategy 1: Flip StWk only when OTHER 4 lenses agree strongly
    logger.info("Strategy 1: Conditional flip when other lenses show strong consensus")
    logger.info("")

    for min_other_agreement in [3, 4]:  # At least 3 or 4 of the other lenses
        for min_magnitude in [0.3, 0.5, 0.7, 1.0]:
            fixes = 0
            breaks = 0

            # Test on inversion errors
            for err in inversion_errors:
                lens = err['lens_results']
                gt = err['ground_truth']

                # Count how many non-StWk lenses have strong signal
                strong_signals = 0
                for ln in ['AT', 'GC', 'PuPy', 'AmKe']:
                    if abs(lens[ln]) >= min_magnitude:
                        strong_signals += 1

                if strong_signals >= min_other_agreement:
                    # Try flipping StWk
                    new_pred, _ = predict_with_transform(lens, 'flip_StWk', quantization)
                    if new_pred == gt:
                        fixes += 1

            # Test on correct predictions
            for c in correct[:1000]:
                lens = c['lens_results']
                gt = c['ground_truth']

                # Count strong signals
                strong_signals = 0
                for ln in ['AT', 'GC', 'PuPy', 'AmKe']:
                    if abs(lens[ln]) >= min_magnitude:
                        strong_signals += 1

                if strong_signals >= min_other_agreement:
                    new_pred, _ = predict_with_transform(lens, 'flip_StWk', quantization)
                    if new_pred != gt:
                        breaks += 1

            logger.info(f"  {min_other_agreement} lenses ≥{min_magnitude:.1f}: Fixes={fixes}/{len(inversion_errors)}, Breaks={breaks}/1000")

            if breaks == 0 and fixes > 0:
                logger.info(f"    ✓ SAFE SIGNATURE FOUND!")
                signature = {
                    'transform': 'conditional_flip_StWk',
                    'condition': {
                        'min_other_lenses_strong': min_other_agreement,
                        'min_magnitude': min_magnitude
                    },
                    'fixes': fixes,
                    'breaks': breaks
                }

                output_file = output_dir / f"{quantization}_conditional_stWk_flip.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(signature, f, indent=2)
                logger.info(f"    Saved to: {output_file}")
                logger.info("")
                return signature

    logger.info("")

    # Strategy 2: Dampen StWk instead of flipping
    logger.info("Strategy 2: Dampen StWk instead of flipping")
    logger.info("")

    for transform in ['dampen_StWk_50%', 'dampen_StWk_25%', 'drop_StWk']:
        for min_magnitude in [0.0, 0.2, 0.3, 0.5]:
            fixes = 0
            breaks = 0

            for err in inversion_errors:
                lens = err['lens_results']
                gt = err['ground_truth']

                if abs(lens['StWk']) >= min_magnitude:
                    new_pred, _ = predict_with_transform(lens, transform, quantization)
                    if new_pred == gt:
                        fixes += 1

            for c in correct[:1000]:
                lens = c['lens_results']
                gt = c['ground_truth']

                if abs(lens['StWk']) >= min_magnitude:
                    new_pred, _ = predict_with_transform(lens, transform, quantization)
                    if new_pred != gt:
                        breaks += 1

            logger.info(f"  {transform} (StWk≥{min_magnitude:.1f}): Fixes={fixes}/{len(inversion_errors)}, Breaks={breaks}/1000")

            if breaks == 0 and fixes > 0:
                logger.info(f"    ✓ SAFE SIGNATURE FOUND!")
                signature = {
                    'transform': transform,
                    'condition': {'min_StWk_magnitude': min_magnitude},
                    'fixes': fixes,
                    'breaks': breaks
                }

                output_file = output_dir / f"{quantization}_{transform}_signature.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(signature, f, indent=2)
                logger.info(f"    Saved to: {output_file}")
                logger.info("")
                return signature

    logger.info("")

    # Strategy 3: Selective flip based on prediction direction
    logger.info("Strategy 3: Flip StWk ONLY for Strong→Weak errors (not Weak→Strong)")
    logger.info("")

    for min_stWk in [0.0, 0.1, 0.2, 0.3]:
        fixes = 0
        breaks = 0

        # Only try to fix Strong→Weak errors
        strong_to_weak_errors = [e for e in inversion_errors
                                 if e['ground_truth'] in strong_nts
                                 and e['predicted'] in weak_nts]

        for err in strong_to_weak_errors:
            lens = err['lens_results']
            gt = err['ground_truth']

            if abs(lens['StWk']) >= min_stWk:
                new_pred, _ = predict_with_transform(lens, 'flip_StWk', quantization)
                if new_pred == gt:
                    fixes += 1

        # Test on ALL correct predictions (including both strong and weak)
        for c in correct[:1000]:
            lens = c['lens_results']
            gt = c['ground_truth']

            # Only apply if ground truth is strong (G/C)
            gt_is_strong = gt in strong_nts

            # Only flip if ground truth is strong
            if gt_is_strong and abs(lens['StWk']) >= min_stWk:
                new_pred, _ = predict_with_transform(lens, 'flip_StWk', quantization)
                if new_pred != gt:
                    breaks += 1

        logger.info(f"  Strong→Weak only (StWk≥{min_stWk:.1f}): Fixes={fixes}/{len(strong_to_weak_errors)}, Breaks={breaks}/1000")

        if breaks == 0 and fixes > 0:
            logger.info(f"    ✓ SAFE SIGNATURE FOUND!")
            signature = {
                'transform': 'selective_flip_StWk_strong_to_weak',
                'condition': {'min_StWk_magnitude': min_stWk},
                'fixes': fixes,
                'breaks': breaks
            }

            output_file = output_dir / f"{quantization}_selective_stWk_flip.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(signature, f, indent=2)
            logger.info(f"    Saved to: {output_file}")
            logger.info("")
            return signature

    logger.info("")
    logger.info("✗ No safe thermodynamic correction found")
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Advanced thermodynamic correction discovery')
    parser.add_argument('--predictions-dir', type=str,
                       default='HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/comparison_results/')
    parser.add_argument('--output-dir', type=str,
                       default='HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/thermodynamic_corrections/')

    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)

    for quant in ['float32', 'int8', 'int4', 'binary']:
        pred_file = predictions_dir / f"{quant}_predictions_detailed.json"
        discover_conditional_stWk_flip(pred_file, quant, output_dir)
        logger.info("")


if __name__ == '__main__':
    main()
