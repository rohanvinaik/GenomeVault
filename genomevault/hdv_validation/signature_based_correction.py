#!/usr/bin/env python3
"""
Signature-Based Correction

Key insight: Different error patterns have different biophysical signatures.
Detect the signature, apply the appropriate transformation.

Observed signatures from teachable moments:
1. "drop_AT + flip_PuPy" signature:
   - AT: positive
   - GC: negative
   - PuPy: positive (but will flip)
   - AmKe: positive
   - StWk: positive

Strategy: Learn signature boundaries from teachable moments, then apply
transformation only to positions matching that signature.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def vote_for_base_with_threshold(lens: str, similarity: float, threshold: float) -> str:
    """Apply threshold and vote for a base."""
    if abs(similarity) < threshold:
        return None
    if lens == 'AT':
        return 'A' if similarity > 0 else 'T'
    elif lens == 'GC':
        return 'G' if similarity > 0 else 'C'
    elif lens == 'PuPy':
        return 'A' if similarity > 0 else 'C'
    elif lens == 'AmKe':
        return 'A' if similarity > 0 else 'G'
    elif lens == 'StWk':
        return 'G' if similarity > 0 else 'A'
    return None


def matches_drop_at_flip_pupy_signature(lens_results: Dict[str, float],
                                        min_votes_required: int = 3) -> bool:
    """
    Detect if position matches the "drop_AT + flip_PuPy" error signature.

    Pattern from teachable moments:
    - AT: positive (but we'll drop it anyway)
    - GC: negative
    - PuPy: positive (moderate, 0.2-1.5 range)
    - AmKe: positive (often strong, 0.2-1.5)
    - StWk: positive (moderate to strong, 0.3-1.2)
    """
    at = lens_results['AT']
    gc = lens_results['GC']
    pupy = lens_results['PuPy']
    amke = lens_results['AmKe']
    stwk = lens_results['StWk']

    # Core pattern: GC negative, others positive
    pattern_match = (
        gc < 0 and           # GC negative
        pupy > 0 and         # PuPy positive
        amke > 0 and         # AmKe positive
        stwk > 0             # StWk positive
    )

    if not pattern_match:
        return False

    # Additional constraint: AmKe should be reasonably strong
    # (from teachable moments: AmKe ranges 0.24-1.45, usually > 0.8)
    if amke < 0.2:
        return False

    return True


def apply_signature_based_correction(
    prediction: Dict,
    max_confidence: float = 0.65,
    min_votes: int = 2
) -> Tuple[str, bool]:
    """
    Apply signature-based correction.

    If position matches error signature, apply appropriate transformation.
    """
    lens_results = prediction['lens_results']
    confidence = prediction['confidence']
    original_pred = prediction['predicted']

    # Only correct low-confidence predictions
    if confidence > max_confidence:
        return original_pred, False

    # Check for "drop_AT + flip_PuPy" signature
    if matches_drop_at_flip_pupy_signature(lens_results):
        # Apply transformation
        transformed_lens = {
            'GC': lens_results['GC'],
            'PuPy': -lens_results['PuPy'],  # Flip
            'AmKe': lens_results['AmKe'],
            'StWk': lens_results['StWk']
        }

        # Vote with ultra-low thresholds
        ultra_low = {'GC': 0.01, 'PuPy': 0.01, 'AmKe': 0.01, 'StWk': 0.01}
        votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}

        for lens, sim in transformed_lens.items():
            voted_base = vote_for_base_with_threshold(lens, sim, ultra_low[lens])
            if voted_base:
                votes[voted_base] += 1

        corrected_pred = max(votes, key=votes.get)
        vote_count = votes[corrected_pred]

        # Override if different and has enough votes
        if corrected_pred != original_pred and vote_count >= min_votes:
            return corrected_pred, True

    return original_pred, False


def evaluate_signature_correction(
    teachable_moments: List[Dict],
    correct_predictions: List[Dict]
) -> Dict:
    """Evaluate signature-based correction."""

    float32_thresholds = {'AT': 0.05, 'GC': 0.00, 'PuPy': 0.20, 'AmKe': 0.20, 'StWk': 0.20}

    # Test on teachable moments
    errors_fixed = 0
    signatures_detected = 0

    for tm in teachable_moments:
        ground_truth = tm['ground_truth']
        lens_results = tm['lens_results']['float32']
        confidence = tm['confidences']['float32']

        # Create prediction dict
        pred = {
            'lens_results': lens_results,
            'confidence': confidence,
            'predicted': None,
            'ground_truth': ground_truth
        }

        # Compute original float32 prediction
        orig_votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for lens, sim in lens_results.items():
            voted_base = vote_for_base_with_threshold(lens, sim, float32_thresholds[lens])
            if voted_base:
                orig_votes[voted_base] += 1
        pred['predicted'] = max(orig_votes, key=orig_votes.get)

        # Check if signature detected
        if matches_drop_at_flip_pupy_signature(lens_results):
            signatures_detected += 1

        # Apply correction
        corrected, was_corrected = apply_signature_based_correction(pred)

        if was_corrected and corrected == ground_truth:
            errors_fixed += 1

    # Test on correct predictions
    import random
    random.seed(42)
    sample_correct = random.sample(correct_predictions, min(1000, len(correct_predictions)))

    correct_broken = 0
    false_positives = 0

    for pred in sample_correct:
        # Check for false positive signature detection
        if matches_drop_at_flip_pupy_signature(pred['lens_results']):
            false_positives += 1

        corrected, was_corrected = apply_signature_based_correction(pred)

        if was_corrected and corrected != pred['ground_truth']:
            correct_broken += 1

    return {
        'errors_fixed': errors_fixed,
        'teachable_signatures_detected': signatures_detected,
        'total_teachable': len(teachable_moments),
        'correct_broken': correct_broken,
        'false_positives': false_positives,
        'total_correct_tested': len(sample_correct),
        'net_benefit': errors_fixed - correct_broken
    }


def main():
    logger.info("="*80)
    logger.info("SIGNATURE-BASED CORRECTION")
    logger.info("="*80)
    logger.info("")

    # Load data
    tm_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/teachable_moments.json")
    with open(tm_file, 'r') as f:
        tm_data = json.load(f)
    teachable_moments = tm_data['positions']

    float32_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/comparison_results/float32_predictions_detailed.json")
    with open(float32_file, 'r') as f:
        all_float32 = json.load(f)
    correct_predictions = [p for p in all_float32 if p['correct']]

    logger.info(f"Loaded {len(teachable_moments)} teachable moments")
    logger.info(f"Loaded {len(correct_predictions)} correct predictions")
    logger.info("")

    # Evaluate
    result = evaluate_signature_correction(teachable_moments, correct_predictions)

    logger.info("RESULTS:")
    logger.info(f"  Teachable moments with signature: {result['teachable_signatures_detected']}/{result['total_teachable']}")
    logger.info(f"  Errors fixed: {result['errors_fixed']}/{result['total_teachable']}")
    logger.info(f"  Correct predictions broken: {result['correct_broken']}/{result['total_correct_tested']}")
    logger.info(f"  False positive signatures in correct: {result['false_positives']}/{result['total_correct_tested']}")
    logger.info(f"  Net benefit: {result['net_benefit']:+d}")
    logger.info("")

    if result['net_benefit'] > 0:
        logger.info(f"✓ Signature-based correction provides net benefit of {result['net_benefit']} positions!")
    else:
        logger.info(f"✗ Signature too broad - need tighter constraints")

    # Save
    output_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/signature_correction_results.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
