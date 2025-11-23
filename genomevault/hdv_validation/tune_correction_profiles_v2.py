#!/usr/bin/env python3
"""
OPTIMIZED Correction Profile Tuning v2 (Evidence-Based)

Uses actual teachable moment characteristics to find optimal detection parameters.

Evidence from analysis:
- 65 teachable moments (float32 wrong, int4/int8/binary correct)
- Mean magnitude: 0.58 ± 0.27 (range 0.09 - 1.19)
- Variance: 0.13 ± 0.10 (range 0.002 - 0.37)
- Confidence: 0.55 ± 0.13 (range 0.2 - 0.8)
- Max votes: 1-4 (diverse, not just 3)

Key insight: The teachable moments are DIVERSE, not a single narrow signature.
We need flexible detection with scoring instead of rigid AND logic.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class FlexibleCorrectionParams:
    """Flexible parameters for detecting correction opportunities."""
    # Simply: try int4 thresholds on positions below this confidence
    max_confidence_for_correction: float
    min_int4_votes_required: int  # Minimum votes from int4 to accept correction


def apply_flexible_correction(
    prediction: Dict,
    int4_thresholds: Dict[str, float],
    params: FlexibleCorrectionParams
) -> Tuple[str, bool]:
    """
    Ultra-simple correction strategy:
    - If confidence < threshold, try int4 thresholds
    - If int4 gives different answer with ≥N votes, use it

    Returns:
        (corrected_base, was_corrected)
    """
    from genomevault.hdv_validation.adaptive_correction import vote_for_base_with_threshold

    lens_similarities = prediction['lens_results']
    confidence = prediction['confidence']
    predicted_base = prediction['predicted']

    # Only try correction if confidence below threshold
    if confidence >= params.max_confidence_for_correction:
        return predicted_base, False

    # Apply int4 thresholds
    int4_votes = {base: 0 for base in ['A', 'T', 'G', 'C']}

    for lens, similarity in lens_similarities.items():
        threshold = int4_thresholds[lens]
        voted_base = vote_for_base_with_threshold(lens, similarity, threshold)
        if voted_base:
            int4_votes[voted_base] += 1

    int4_prediction = max(int4_votes, key=int4_votes.get)
    int4_vote_count = int4_votes[int4_prediction]

    # Apply correction if int4 gives different answer with enough votes
    if int4_prediction != predicted_base and int4_vote_count >= params.min_int4_votes_required:
        return int4_prediction, True

    return predicted_base, False


def test_flexible_params(
    predictions: List[Dict],
    teachable_positions: Set[str],
    int4_thresholds: Dict[str, float],
    params: FlexibleCorrectionParams
) -> Dict:
    """Test flexible correction parameters."""
    stats = {
        'total': 0,
        'baseline_correct': 0,
        'corrected_correct': 0,
        'corrections_applied': 0,
        'errors_fixed': 0,
        'correct_broken': 0,
        'teachable_moments_detected': 0,
        'teachable_moments_fixed': 0,
    }

    for pred in predictions:
        if pred.get('has_n', False):
            continue

        stats['total'] += 1

        ground_truth = pred['ground_truth']
        baseline_pred = pred['predicted']
        baseline_correct = (baseline_pred == ground_truth)

        # Apply correction
        corrected_pred, was_corrected = apply_flexible_correction(
            pred, int4_thresholds, params
        )

        corrected_correct = (corrected_pred == ground_truth)

        if baseline_correct:
            stats['baseline_correct'] += 1
        if corrected_correct:
            stats['corrected_correct'] += 1

        if was_corrected:
            stats['corrections_applied'] += 1

            # Check if teachable moment
            if pred['position'] in teachable_positions:
                stats['teachable_moments_detected'] += 1
                if not baseline_correct and corrected_correct:
                    stats['teachable_moments_fixed'] += 1

            if not baseline_correct and corrected_correct:
                stats['errors_fixed'] += 1
            elif baseline_correct and not corrected_correct:
                stats['correct_broken'] += 1

    # Calculate metrics
    total = stats['total']
    stats['baseline_accuracy'] = stats['baseline_correct'] / total if total > 0 else 0
    stats['corrected_accuracy'] = stats['corrected_correct'] / total if total > 0 else 0
    stats['improvement'] = stats['corrected_accuracy'] - stats['baseline_accuracy']
    stats['net_benefit'] = stats['errors_fixed'] - stats['correct_broken']

    total_teachable = len(teachable_positions)
    stats['teachable_detection_rate'] = stats['teachable_moments_detected'] / total_teachable if total_teachable > 0 else 0
    stats['teachable_fix_rate'] = stats['teachable_moments_fixed'] / total_teachable if total_teachable > 0 else 0

    return stats


def optimize_flexible_correction(
    quantization: str = 'float32',
    validation_results_dir: str = None
):
    """
    Ultra-simple optimization:
    - Sweep confidence threshold (when to try int4 correction)
    - Sweep min votes required from int4
    - Find best combination
    """
    logger.info("=" * 80)
    logger.info("FLEXIBLE CORRECTION OPTIMIZATION v2")
    logger.info("Evidence-Based Ultra-Simple Strategy")
    logger.info("=" * 80)
    logger.info(f"Quantization: {quantization}")
    logger.info("")

    # Load data
    if validation_results_dir is None:
        validation_results_dir = Path("HDV_VALIDATION_PACKAGE/architecture_testing/comparison_results")
    else:
        validation_results_dir = Path(validation_results_dir)

    logger.info("Loading data...")

    # Load predictions
    pred_file = validation_results_dir / f"{quantization}_predictions_detailed.json"
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    logger.info(f"  ✓ Loaded {len(predictions)} predictions")

    # Load teachable moments
    analysis_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/cross_quantization_error_analysis.json")
    with open(analysis_file, 'r') as f:
        cross_quant = json.load(f)

    teachable_moments = cross_quant['categories']['teachable_moments']
    teachable_positions = {tm['position'] for tm in teachable_moments}
    logger.info(f"  ✓ Loaded {len(teachable_moments)} teachable moments")

    # Filter predictions
    predictions = [p for p in predictions if not p.get('has_n', False)]
    logger.info(f"  ✓ {len(predictions)} positions with ground truth")
    logger.info("")

    # Load int4 thresholds
    from genomevault.hdv_validation.adaptive_correction import OPTIMAL_THRESHOLDS
    int4_thresholds = OPTIMAL_THRESHOLDS['int4']

    logger.info("Int4 thresholds:")
    for lens, thresh in int4_thresholds.items():
        logger.info(f"  {lens}: {thresh:.4f}")
    logger.info("")

    # Define simple sweep
    logger.info("Parameter sweep:")

    # When to try correction (based on observed confidence range 0.2-0.8)
    confidence_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    # Min votes required from int4 (most teachable moments have 3-4 votes, some have 1-2)
    min_votes_vals = [1, 2, 3, 4]

    logger.info(f"  max_confidence: {confidence_thresholds}")
    logger.info(f"  min_int4_votes: {min_votes_vals}")
    logger.info("")

    total_combos = len(confidence_thresholds) * len(min_votes_vals)
    logger.info(f"Total combinations: {total_combos}")
    logger.info("")

    # Sweep
    best_params = None
    best_score = -float('inf')
    best_stats = None

    logger.info("Starting optimization...")
    logger.info("")

    results_table = []

    for max_conf in confidence_thresholds:
        for min_votes in min_votes_vals:
            params = FlexibleCorrectionParams(
                max_confidence_for_correction=max_conf,
                min_int4_votes_required=min_votes
            )

            stats = test_flexible_params(
                predictions, teachable_positions, int4_thresholds, params
            )

            # Score: prioritize fixing errors without breaking correct ones
            # Bonus for detecting teachable moments
            score = (
                stats['net_benefit'] * 10 +  # Net benefit is critical
                stats['teachable_moments_fixed'] * 5  # Bonus for teachable moments
            )

            results_table.append({
                'max_conf': max_conf,
                'min_votes': min_votes,
                'errors_fixed': stats['errors_fixed'],
                'correct_broken': stats['correct_broken'],
                'net_benefit': stats['net_benefit'],
                'teachable_fixed': stats['teachable_moments_fixed'],
                'teachable_detected': stats['teachable_moments_detected'],
                'corrections_applied': stats['corrections_applied'],
                'score': score
            })

            if score > best_score:
                best_score = score
                best_params = params
                best_stats = stats

    logger.info("")
    logger.info("=" * 80)
    logger.info("TOP 10 CONFIGURATIONS")
    logger.info("=" * 80)
    logger.info("")

    # Sort by score
    results_table.sort(key=lambda x: x['score'], reverse=True)

    logger.info(f"{'Conf':>6} {'Votes':>5} {'Fixed':>6} {'Broken':>7} {'Net':>5} {'TM_Fix':>7} {'TM_Det':>7} {'Applied':>8} {'Score':>7}")
    logger.info("-" * 80)

    for r in results_table[:10]:
        logger.info(
            f"{r['max_conf']:>6.2f} {r['min_votes']:>5} "
            f"{r['errors_fixed']:>6} {r['correct_broken']:>7} {r['net_benefit']:>5} "
            f"{r['teachable_fixed']:>7} {r['teachable_detected']:>7} "
            f"{r['corrections_applied']:>8} {r['score']:>7.0f}"
        )

    logger.info("")
    logger.info("=" * 80)
    logger.info("BEST CONFIGURATION")
    logger.info("=" * 80)
    logger.info("")

    logger.info("Parameters:")
    logger.info(f"  max_confidence_for_correction: {best_params.max_confidence_for_correction:.2f}")
    logger.info(f"  min_int4_votes_required: {best_params.min_int4_votes_required}")
    logger.info("")

    logger.info("Performance:")
    logger.info(f"  Baseline accuracy: {best_stats['baseline_accuracy']:.4%}")
    logger.info(f"  Corrected accuracy: {best_stats['corrected_accuracy']:.4%}")
    logger.info(f"  Improvement: {best_stats['improvement']*100:+.2f}%")
    logger.info("")

    logger.info("Corrections:")
    logger.info(f"  Errors fixed: {best_stats['errors_fixed']}")
    logger.info(f"  Correct broken: {best_stats['correct_broken']}")
    logger.info(f"  Net benefit: {best_stats['net_benefit']}")
    logger.info(f"  Total applied: {best_stats['corrections_applied']}")
    logger.info("")

    logger.info("Teachable moments:")
    logger.info(f"  Total: {len(teachable_moments)}")
    logger.info(f"  Detected: {best_stats['teachable_moments_detected']} ({best_stats['teachable_detection_rate']:.1%})")
    logger.info(f"  Fixed: {best_stats['teachable_moments_fixed']} ({best_stats['teachable_fix_rate']:.1%})")
    logger.info("")

    # Save results
    output_path = Path(f"HDV_VALIDATION_PACKAGE/architecture_testing/flexible_correction_params_{quantization}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'quantization': quantization,
        'sample_size': len(predictions),
        'teachable_moments_count': len(teachable_moments),
        'optimal_parameters': {
            'max_confidence_for_correction': best_params.max_confidence_for_correction,
            'min_int4_votes_required': best_params.min_int4_votes_required
        },
        'performance': best_stats,
        'all_configurations': results_table
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Results saved to: {output_path}")
    logger.info("")

    return best_params, best_stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Flexible correction optimization v2'
    )
    parser.add_argument(
        '--quantization',
        default='float32',
        choices=['float32', 'int8', 'int4', 'binary'],
        help='Quantization mode'
    )
    parser.add_argument(
        '--validation-results',
        type=str,
        default=None,
        help='Validation results directory'
    )

    args = parser.parse_args()

    optimize_flexible_correction(
        quantization=args.quantization,
        validation_results_dir=args.validation_results
    )
