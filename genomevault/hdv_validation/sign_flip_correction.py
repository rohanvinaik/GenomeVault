#!/usr/bin/env python3
"""
Sign-Flip Correction

Key insight: If the problem is SIGN not magnitude, try FLIPPING signs!

Strategy:
1. Drop AT lens entirely (often weak, 11.2% of signal in teachable moments)
2. Try flipping signs of other lenses (GC, PuPy, AmKe, StWk)
3. Find which transformation fixes the most teachable moments
"""

import json
import logging
import numpy as np
from pathlib import Path
from itertools import product
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


class SignFlipTransform:
    """Transformation that drops/flips lens signals."""

    def __init__(self, drop_at: bool = False, flip_gc: bool = False,
                 flip_pupy: bool = False, flip_amke: bool = False,
                 flip_stwk: bool = False):
        self.drop_at = drop_at
        self.flip_gc = flip_gc
        self.flip_pupy = flip_pupy
        self.flip_amke = flip_amke
        self.flip_stwk = flip_stwk

    def transform(self, lens_results: Dict[str, float]) -> Dict[str, float]:
        """Apply transformation to lens results."""
        transformed = {}

        # AT: drop or keep
        if not self.drop_at:
            transformed['AT'] = lens_results['AT']

        # Other lenses: flip sign or keep
        transformed['GC'] = -lens_results['GC'] if self.flip_gc else lens_results['GC']
        transformed['PuPy'] = -lens_results['PuPy'] if self.flip_pupy else lens_results['PuPy']
        transformed['AmKe'] = -lens_results['AmKe'] if self.flip_amke else lens_results['AmKe']
        transformed['StWk'] = -lens_results['StWk'] if self.flip_stwk else lens_results['StWk']

        return transformed

    def __repr__(self):
        parts = []
        if self.drop_at:
            parts.append("drop_AT")
        if self.flip_gc:
            parts.append("flip_GC")
        if self.flip_pupy:
            parts.append("flip_PuPy")
        if self.flip_amke:
            parts.append("flip_AmKe")
        if self.flip_stwk:
            parts.append("flip_StWk")
        return " + ".join(parts) if parts else "no_transform"


def evaluate_transform(
    teachable_moments: List[Dict],
    correct_predictions: List[Dict],
    transform: SignFlipTransform,
    use_ultra_low_thresholds: bool = True
) -> Dict:
    """Evaluate a sign-flip transformation."""

    # Choose thresholds
    if use_ultra_low_thresholds:
        thresholds = {'AT': 0.01, 'GC': 0.01, 'PuPy': 0.01, 'AmKe': 0.01, 'StWk': 0.01}
    else:
        thresholds = {'AT': 0.05, 'GC': 0.00, 'PuPy': 0.20, 'AmKe': 0.20, 'StWk': 0.20}

    float32_thresholds = {'AT': 0.05, 'GC': 0.00, 'PuPy': 0.20, 'AmKe': 0.20, 'StWk': 0.20}

    # Test on teachable moments
    errors_fixed = 0

    for tm in teachable_moments:
        ground_truth = tm['ground_truth']
        lens_results = tm['lens_results']['float32']
        confidence = tm['confidences']['float32']

        # Only apply to low-confidence predictions
        if confidence > 0.65:
            continue

        # Original float32 prediction
        orig_votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for lens, sim in lens_results.items():
            voted_base = vote_for_base_with_threshold(lens, sim, float32_thresholds[lens])
            if voted_base:
                orig_votes[voted_base] += 1
        orig_pred = max(orig_votes, key=orig_votes.get)

        # Apply transformation
        transformed_lens = transform.transform(lens_results)

        # Vote with transformed lens
        trans_votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for lens, sim in transformed_lens.items():
            threshold = thresholds.get(lens, 0.01)
            voted_base = vote_for_base_with_threshold(lens, sim, threshold)
            if voted_base:
                trans_votes[voted_base] += 1

        trans_pred = max(trans_votes, key=trans_votes.get)
        trans_vote_count = trans_votes[trans_pred]

        # Check if transformation fixes it
        if trans_pred != orig_pred and trans_vote_count >= 2 and trans_pred == ground_truth:
            errors_fixed += 1

    # Test on correct predictions (sample)
    import random
    random.seed(42)
    sample_correct = random.sample(correct_predictions, min(1000, len(correct_predictions)))

    correct_broken = 0
    for pred in sample_correct:
        ground_truth = pred['ground_truth']
        lens_results = pred['lens_results']
        confidence = pred['confidence']

        if confidence > 0.65:
            continue

        # Original prediction
        orig_votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for lens, sim in lens_results.items():
            voted_base = vote_for_base_with_threshold(lens, sim, float32_thresholds[lens])
            if voted_base:
                orig_votes[voted_base] += 1
        orig_pred = max(orig_votes, key=orig_votes.get)

        # Apply transformation
        transformed_lens = transform.transform(lens_results)

        # Vote with transformed lens
        trans_votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for lens, sim in transformed_lens.items():
            threshold = thresholds.get(lens, 0.01)
            voted_base = vote_for_base_with_threshold(lens, sim, threshold)
            if voted_base:
                trans_votes[voted_base] += 1

        trans_pred = max(trans_votes, key=trans_votes.get)
        trans_vote_count = trans_votes[trans_pred]

        # Check if transformation breaks it
        if trans_pred != orig_pred and trans_vote_count >= 2 and trans_pred != ground_truth:
            correct_broken += 1

    return {
        'errors_fixed': errors_fixed,
        'correct_broken': correct_broken,
        'net_benefit': errors_fixed - correct_broken
    }


def main():
    logger.info("="*80)
    logger.info("SIGN-FLIP CORRECTION")
    logger.info("="*80)
    logger.info("")
    logger.info("Strategy: If the problem is SIGN, flip signs!")
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

    # Test all combinations of transformations
    logger.info("Testing sign-flip transformations...")
    logger.info("")

    best_result = None
    best_transform = None

    # Try all combinations
    for drop_at, flip_gc, flip_pupy, flip_amke, flip_stwk in product([False, True], repeat=5):
        transform = SignFlipTransform(drop_at, flip_gc, flip_pupy, flip_amke, flip_stwk)

        # Skip no-op transformation
        if not any([drop_at, flip_gc, flip_pupy, flip_amke, flip_stwk]):
            continue

        result = evaluate_transform(teachable_moments, correct_predictions, transform, use_ultra_low_thresholds=True)

        if best_result is None or result['net_benefit'] > best_result['net_benefit']:
            best_result = result
            best_transform = transform

        # Only show results with some activity
        if result['errors_fixed'] > 0 or result['correct_broken'] > 0:
            logger.info(
                f"{str(transform):50s} | Fix:{result['errors_fixed']:2d} Break:{result['correct_broken']:2d} Net:{result['net_benefit']:+3d}"
            )

    logger.info("")
    logger.info("="*80)
    logger.info("BEST TRANSFORMATION")
    logger.info("="*80)
    logger.info(f"Transform: {best_transform}")
    logger.info(f"Errors fixed: {best_result['errors_fixed']}/22")
    logger.info(f"Correct broken: {best_result['correct_broken']}/1000")
    logger.info(f"Net benefit: {best_result['net_benefit']:+d}")
    logger.info("")

    # Save result
    output = {
        'best_transform': {
            'drop_at': best_transform.drop_at,
            'flip_gc': best_transform.flip_gc,
            'flip_pupy': best_transform.flip_pupy,
            'flip_amke': best_transform.flip_amke,
            'flip_stwk': best_transform.flip_stwk,
        },
        'result': best_result
    }

    output_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/best_sign_flip_transform.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"✓ Best transformation saved to: {output_file}")


if __name__ == '__main__':
    main()
