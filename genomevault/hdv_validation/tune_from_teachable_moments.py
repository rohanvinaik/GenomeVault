#!/usr/bin/env python3
"""
Empirical Correction Tuning from Teachable Moments

Uses the 22 real teachable moments (float32/int8 wrong, int4/binary right)
to tune correction parameters for maximum accuracy improvement.

Strategy:
1. Load teachable moments from aligned 10K validation
2. Try different binary threshold voting strategies
3. Find parameters that maximize corrections while minimizing breaks
"""

import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Binary thresholds (ultra-permissive)
BINARY_THRESHOLDS = {
    'AT': 0.0025,
    'GC': 0.00,
    'PuPy': 0.0020,
    'AmKe': 0.0012,
    'StWk': 0.0020
}

# Int4 thresholds (very permissive)
INT4_THRESHOLDS = {
    'AT': 0.0028,
    'GC': 0.00,
    'PuPy': 0.0083,
    'AmKe': 0.0055,
    'StWk': 0.0083
}


def vote_for_base_with_threshold(lens: str, similarity: float, threshold: float) -> str:
    """Apply threshold and vote for a base."""
    if abs(similarity) < threshold:
        return None

    # Lens voting logic
    if lens == 'AT':
        return 'A' if similarity > 0 else 'T'
    elif lens == 'GC':
        return 'G' if similarity > 0 else 'C'
    elif lens == 'PuPy':  # Purine vs Pyrimidine
        return 'A' if similarity > 0 else 'C'  # A/G are purines, T/C are pyrimidines
    elif lens == 'AmKe':  # Amino vs Keto
        return 'A' if similarity > 0 else 'G'  # A/C are amino, T/G are keto
    elif lens == 'StWk':  # Strong vs Weak
        return 'G' if similarity > 0 else 'A'  # G/C are strong, A/T are weak

    return None


@dataclass
class CorrectionParams:
    """Parameters for correction strategy."""
    use_binary_thresholds: bool = True  # Use binary vs int4
    min_votes_required: int = 3  # Minimum votes to override
    max_confidence_for_correction: float = 0.65  # Only correct low-confidence predictions


def apply_correction(
    prediction: Dict,
    params: CorrectionParams,
    thresholds: Dict[str, float]
) -> Tuple[str, bool]:
    """
    Apply correction strategy to a prediction.

    Returns:
        (corrected_base, was_corrected)
    """
    lens_results = prediction['lens_results']
    confidence = prediction['confidence']
    original_pred = prediction['predicted']

    # Only correct low-confidence predictions
    if confidence > params.max_confidence_for_correction:
        return original_pred, False

    # Apply alternative thresholds
    votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
    for lens, similarity in lens_results.items():
        threshold = thresholds[lens]
        voted_base = vote_for_base_with_threshold(lens, similarity, threshold)
        if voted_base:
            votes[voted_base] += 1

    # Get alternative prediction
    alt_pred = max(votes, key=votes.get)
    alt_votes = votes[alt_pred]

    # Override if alternative has enough votes and differs from original
    if alt_pred != original_pred and alt_votes >= params.min_votes_required:
        return alt_pred, True

    return original_pred, False


def evaluate_correction_strategy(
    teachable_moments: List[Dict],
    all_correct_predictions: List[Dict],
    params: CorrectionParams,
    thresholds: Dict[str, float]
) -> Dict:
    """Evaluate how well a correction strategy performs."""

    # Test on teachable moments (should fix these)
    errors_fixed = 0
    for tm in teachable_moments:
        # Create prediction dict for float32
        pred = {
            'lens_results': tm['lens_results']['float32'],
            'confidence': tm['confidences']['float32'],
            'predicted': tm['lens_results']['float32']  # This will be wrong by definition
        }

        # Reconstruct original float32 prediction
        original_votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for lens, sim in pred['lens_results'].items():
            # Use float32 thresholds for original prediction
            float32_thresholds = {
                'AT': 0.05, 'GC': 0.00, 'PuPy': 0.20, 'AmKe': 0.20, 'StWk': 0.20
            }
            voted_base = vote_for_base_with_threshold(lens, sim, float32_thresholds[lens])
            if voted_base:
                original_votes[voted_base] += 1
        pred['predicted'] = max(original_votes, key=original_votes.get)

        corrected, was_corrected = apply_correction(pred, params, thresholds)

        if was_corrected and corrected == tm['ground_truth']:
            errors_fixed += 1

    # Test on sample of correct predictions (should not break these)
    # Sample 1000 random correct predictions
    import random
    random.seed(42)
    sample_correct = random.sample(all_correct_predictions, min(1000, len(all_correct_predictions)))

    correct_broken = 0
    for pred in sample_correct:
        corrected, was_corrected = apply_correction(pred, params, thresholds)
        if was_corrected and corrected != pred['ground_truth']:
            correct_broken += 1

    return {
        'errors_fixed': errors_fixed,
        'total_errors': len(teachable_moments),
        'correct_broken': correct_broken,
        'total_correct_tested': len(sample_correct),
        'net_benefit': errors_fixed - correct_broken,
        'fix_rate': errors_fixed / len(teachable_moments) if teachable_moments else 0,
        'break_rate': correct_broken / len(sample_correct) if sample_correct else 0
    }


def main():
    logger.info("="*80)
    logger.info("EMPIRICAL CORRECTION TUNING FROM TEACHABLE MOMENTS")
    logger.info("="*80)
    logger.info("")

    # Load teachable moments
    tm_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/teachable_moments.json")
    with open(tm_file, 'r') as f:
        tm_data = json.load(f)

    teachable_moments = tm_data['positions']
    logger.info(f"Loaded {len(teachable_moments)} teachable moments")

    # Load all float32 predictions to get correct ones
    float32_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/comparison_results/float32_predictions_detailed.json")
    with open(float32_file, 'r') as f:
        all_float32 = json.load(f)

    correct_predictions = [p for p in all_float32 if p['correct']]
    logger.info(f"Loaded {len(correct_predictions)} correct float32 predictions")
    logger.info("")

    # Test different parameter combinations
    logger.info("Testing parameter combinations...")
    logger.info("")

    best_result = None
    best_params = None
    best_thresholds = None

    for use_binary in [True, False]:
        thresholds = BINARY_THRESHOLDS if use_binary else INT4_THRESHOLDS
        threshold_name = "binary" if use_binary else "int4"

        for min_votes in [2, 3, 4]:
            for max_conf in [0.50, 0.60, 0.65, 0.70]:
                params = CorrectionParams(
                    use_binary_thresholds=use_binary,
                    min_votes_required=min_votes,
                    max_confidence_for_correction=max_conf
                )

                result = evaluate_correction_strategy(
                    teachable_moments,
                    correct_predictions,
                    params,
                    thresholds
                )

                if best_result is None or result['net_benefit'] > best_result['net_benefit']:
                    best_result = result
                    best_params = params
                    best_thresholds = thresholds
                    threshold_name_best = threshold_name

                logger.info(
                    f"{threshold_name:6s} | votes≥{min_votes} | conf≤{max_conf:.2f} | "
                    f"Fixed: {result['errors_fixed']:2d}/22 | Broke: {result['correct_broken']:2d}/1000 | "
                    f"Net: {result['net_benefit']:+3d}"
                )

    logger.info("")
    logger.info("="*80)
    logger.info("BEST CORRECTION STRATEGY")
    logger.info("="*80)
    logger.info(f"Thresholds: {threshold_name_best}")
    logger.info(f"Min votes: {best_params.min_votes_required}")
    logger.info(f"Max confidence: {best_params.max_confidence_for_correction}")
    logger.info("")
    logger.info(f"Errors fixed: {best_result['errors_fixed']}/22 ({100*best_result['fix_rate']:.1f}%)")
    logger.info(f"Correct broken: {best_result['correct_broken']}/1000 ({100*best_result['break_rate']:.1f}%)")
    logger.info(f"Net benefit: {best_result['net_benefit']:+d}")
    logger.info("")
    logger.info(f"Projected improvement on full 9,555 ATGC positions:")
    logger.info(f"  Expected fixes: {best_result['fix_rate'] * 22:.1f}")
    logger.info(f"  Expected breaks: {best_result['break_rate'] * 9489:.1f}")  # 9555 - 66 errors
    logger.info(f"  Net accuracy change: {best_result['fix_rate'] * 22 - best_result['break_rate'] * 9489:.1f} positions")
    logger.info(f"  New accuracy: {100 * (9489 - best_result['break_rate'] * 9489 + best_result['fix_rate'] * 22) / 9555:.2f}%")
    logger.info(f"  Improvement: {100 * (best_result['fix_rate'] * 22 - best_result['break_rate'] * 9489) / 9555:.2f}%")

    # Save best parameters
    output = {
        'best_params': {
            'use_binary_thresholds': best_params.use_binary_thresholds,
            'min_votes_required': best_params.min_votes_required,
            'max_confidence_for_correction': best_params.max_confidence_for_correction
        },
        'thresholds': best_thresholds,
        'results': best_result
    }

    output_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/best_correction_params.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info("")
    logger.info(f"✓ Best parameters saved to: {output_file}")


if __name__ == '__main__':
    main()
