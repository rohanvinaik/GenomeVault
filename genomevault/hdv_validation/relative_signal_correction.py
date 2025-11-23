#!/usr/bin/env python3
"""
Relative Signal Correction

Simulates the effect of quantization flattening by re-weighting lens contributions
based on RELATIVE signal strength rather than absolute thresholds.

Key insight from teachable moments:
- Float32's strict thresholds (AT:0.05, StWk:0.20) favor strong AT/GC signals
- When quantization flattens to 0.01-0.05 range, relative signal matters more
- StWk contributes 37% of signal in teachable moments but gets suppressed
- By lowering ALL thresholds, we let relative strength determine votes
"""

import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple
from collections import Counter

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


@dataclass
class RelativeSignalParams:
    """Parameters for relative signal correction."""
    at_suppression: float = 1.0  # Multiplier for AT signal (< 1.0 = suppress)
    gc_suppression: float = 1.0  # Multiplier for GC signal
    use_ultra_low_thresholds: bool = True  # Use 0.01 threshold for all lenses
    min_votes_required: int = 3
    max_confidence_for_correction: float = 0.65


def apply_relative_signal_correction(
    prediction: Dict,
    params: RelativeSignalParams
) -> Tuple[str, bool]:
    """
    Apply correction by re-weighting lens signals.

    Strategy:
    1. Optionally suppress AT/GC signals
    2. Apply ultra-low thresholds to simulate quantization flattening
    3. Let relative signal strength determine votes
    """
    lens_results = prediction['lens_results']
    confidence = prediction['confidence']
    original_pred = prediction['predicted']

    # Only correct low-confidence predictions
    if confidence > params.max_confidence_for_correction:
        return original_pred, False

    # Apply signal modifications
    adjusted_lens = lens_results.copy()
    adjusted_lens['AT'] *= params.at_suppression
    adjusted_lens['GC'] *= params.gc_suppression

    # Apply ultra-low thresholds (simulates quantization flattening)
    if params.use_ultra_low_thresholds:
        thresholds = {'AT': 0.01, 'GC': 0.01, 'PuPy': 0.01, 'AmKe': 0.01, 'StWk': 0.01}
    else:
        # Use float32's original thresholds
        thresholds = {'AT': 0.05, 'GC': 0.00, 'PuPy': 0.20, 'AmKe': 0.20, 'StWk': 0.20}

    # Vote with adjusted signals
    votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
    for lens, similarity in adjusted_lens.items():
        threshold = thresholds[lens]
        voted_base = vote_for_base_with_threshold(lens, similarity, threshold)
        if voted_base:
            votes[voted_base] += 1

    alt_pred = max(votes, key=votes.get)
    alt_votes = votes[alt_pred]

    # Override if alternative differs and has enough votes
    if alt_pred != original_pred and alt_votes >= params.min_votes_required:
        return alt_pred, True

    return original_pred, False


def evaluate_strategy(
    teachable_moments: list,
    correct_predictions: list,
    params: RelativeSignalParams
) -> Dict:
    """Evaluate a correction strategy."""

    # Reconstruct float32 predictions for teachable moments
    errors_fixed = 0
    for tm in teachable_moments:
        # Float32 prediction dict
        pred = {
            'lens_results': tm['lens_results']['float32'],
            'confidence': tm['confidences']['float32'],
            'predicted': None,  # Will compute
            'ground_truth': tm['ground_truth']
        }

        # Reconstruct original float32 prediction
        float32_thresholds = {'AT': 0.05, 'GC': 0.00, 'PuPy': 0.20, 'AmKe': 0.20, 'StWk': 0.20}
        votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for lens, sim in pred['lens_results'].items():
            voted_base = vote_for_base_with_threshold(lens, sim, float32_thresholds[lens])
            if voted_base:
                votes[voted_base] += 1
        pred['predicted'] = max(votes, key=votes.get)

        # Apply correction
        corrected, was_corrected = apply_relative_signal_correction(pred, params)

        if was_corrected and corrected == tm['ground_truth']:
            errors_fixed += 1

    # Test on correct predictions (sample 1000)
    import random
    random.seed(42)
    sample_correct = random.sample(correct_predictions, min(1000, len(correct_predictions)))

    correct_broken = 0
    for pred in sample_correct:
        corrected, was_corrected = apply_relative_signal_correction(pred, params)
        if was_corrected and corrected != pred['ground_truth']:
            correct_broken += 1

    return {
        'errors_fixed': errors_fixed,
        'correct_broken': correct_broken,
        'net_benefit': errors_fixed - correct_broken
    }


def main():
    logger.info("="*80)
    logger.info("RELATIVE SIGNAL CORRECTION TUNING")
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

    # Test different parameter combinations
    logger.info("Testing parameter combinations...")
    logger.info("")

    best_result = None
    best_params = None

    # Grid search
    for at_supp in [1.0, 0.5, 0.3, 0.1, 0.0]:  # Suppress AT
        for gc_supp in [1.0, 0.5, 0.3]:  # Suppress GC
            for use_ultra_low in [True, False]:
                for min_votes in [2, 3, 4]:
                    for max_conf in [0.50, 0.60, 0.65, 0.70]:
                        params = RelativeSignalParams(
                            at_suppression=at_supp,
                            gc_suppression=gc_supp,
                            use_ultra_low_thresholds=use_ultra_low,
                            min_votes_required=min_votes,
                            max_confidence_for_correction=max_conf
                        )

                        result = evaluate_strategy(teachable_moments, correct_predictions, params)

                        if best_result is None or result['net_benefit'] > best_result['net_benefit']:
                            best_result = result
                            best_params = params

                        if result['errors_fixed'] > 0 or result['correct_broken'] > 0:
                            thresh_mode = "ultra-low" if use_ultra_low else "float32"
                            logger.info(
                                f"AT×{at_supp:.1f} GC×{gc_supp:.1f} | {thresh_mode:9s} | "
                                f"v≥{min_votes} c≤{max_conf:.2f} | "
                                f"Fix:{result['errors_fixed']:2d} Break:{result['correct_broken']:2d} Net:{result['net_benefit']:+3d}"
                            )

    logger.info("")
    logger.info("="*80)
    logger.info("BEST STRATEGY")
    logger.info("="*80)
    logger.info(f"AT suppression: {best_params.at_suppression}")
    logger.info(f"GC suppression: {best_params.gc_suppression}")
    logger.info(f"Ultra-low thresholds: {best_params.use_ultra_low_thresholds}")
    logger.info(f"Min votes: {best_params.min_votes_required}")
    logger.info(f"Max confidence: {best_params.max_confidence_for_correction}")
    logger.info("")
    logger.info(f"Errors fixed: {best_result['errors_fixed']}/22")
    logger.info(f"Correct broken: {best_result['correct_broken']}/1000")
    logger.info(f"Net benefit: {best_result['net_benefit']:+d}")


if __name__ == '__main__':
    main()
