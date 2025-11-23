#!/usr/bin/env python3
"""
OPTIMIZED Correction Profile Tuning (Cross-Quantization Aware)

Uses teachable moments (positions where float32 fails but int4/binary succeed)
to empirically determine optimal correction parameters.

Key Insight from Cross-Quantization Analysis:
- 65 teachable moments discovered
- Float32 signature: moderate magnitudes (0.4-0.7), low variance (~0.13),
  medium confidence (~0.55), mostly 3 votes
- Int4 succeeds with ultra-permissive thresholds detecting 28-31× smaller signals

Strategy:
1. Load teachable moments as ground truth training data
2. Optimize parameters to detect this specific signature
3. Apply int4-style permissive correction when detected
4. Reduce search space using empirical insights
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class OptimizedCorrectionParams:
    """Optimized parameters based on cross-quantization analysis."""
    # "Structural Flexibility Zone" detection (from teachable moments)
    float32_overfitting_signal_min: float  # Moderate range lower bound
    float32_overfitting_signal_max: float  # Moderate range upper bound
    low_variance_threshold: float          # Detect low variance (overfitting)
    medium_confidence_min: float           # Medium-low confidence range
    medium_confidence_max: float
    dominant_vote_count: int               # Typical vote pattern (3)

    # Correction strategy
    apply_int4_thresholds: bool = True     # Use int4's permissive thresholds


def load_teachable_moments(analysis_file: Path) -> List[Dict]:
    """Load the 65 teachable moment positions from cross-quantization analysis."""
    if not analysis_file.exists():
        raise FileNotFoundError(f"Cross-quantization analysis not found: {analysis_file}")

    with open(analysis_file, 'r') as f:
        data = json.load(f)

    teachable_moments = data['categories']['teachable_moments']
    logger.info(f"  ✓ Loaded {len(teachable_moments)} teachable moments")

    return teachable_moments


def detect_float32_overfitting(
    lens_similarities: Dict[str, float],
    votes: Dict[str, int],
    confidence: float,
    params: OptimizedCorrectionParams
) -> bool:
    """
    Detect the float32 overfitting signature discovered in cross-quantization analysis.

    Signature:
    - Moderate lens magnitudes (0.4-0.7 range)
    - LOW variance (~0.13) indicating structural breathing, not noise
    - Medium-low confidence (0.5-0.6)
    - Dominant vote pattern (typically 3 votes)
    """
    magnitudes = [abs(v) for v in lens_similarities.values()]
    mean_mag = np.mean(magnitudes)
    variance = np.var(magnitudes)

    max_votes = max(votes.values())

    # Check all signature components
    is_moderate_signal = (
        params.float32_overfitting_signal_min <= mean_mag <= params.float32_overfitting_signal_max
    )
    is_low_variance = (variance <= params.low_variance_threshold)
    is_medium_confidence = (
        params.medium_confidence_min <= confidence <= params.medium_confidence_max
    )
    has_dominant_vote = (max_votes == params.dominant_vote_count)

    return is_moderate_signal and is_low_variance and is_medium_confidence and has_dominant_vote


def apply_int4_correction(
    lens_similarities: Dict[str, float],
    int4_thresholds: Dict[str, float]
) -> Tuple[str, int]:
    """
    Apply int4's ultra-permissive thresholds to detect signals float32 missed.

    Returns:
        (predicted_base, vote_count)
    """
    from genomevault.hdv_validation.adaptive_correction import vote_for_base_with_threshold

    votes = {base: 0 for base in ['A', 'T', 'G', 'C']}

    for lens, similarity in lens_similarities.items():
        threshold = int4_thresholds[lens]
        voted_base = vote_for_base_with_threshold(lens, similarity, threshold)
        if voted_base:
            votes[voted_base] += 1

    predicted_base = max(votes, key=votes.get)
    vote_count = votes[predicted_base]

    return predicted_base, vote_count


def apply_optimized_correction(
    prediction: Dict,
    int4_thresholds: Dict[str, float],
    params: OptimizedCorrectionParams
) -> Tuple[str, bool]:
    """
    Apply optimized correction using cross-quantization insights.

    Returns:
        (corrected_base, was_corrected)
    """
    lens_similarities = prediction['lens_results']
    votes = prediction['votes']
    confidence = prediction['confidence']
    predicted_base = prediction['predicted']

    # Fast path: high confidence positions don't need correction
    if confidence >= 0.8:
        return predicted_base, False

    # Detect float32 overfitting signature
    is_overfitting = detect_float32_overfitting(
        lens_similarities, votes, confidence, params
    )

    if not is_overfitting:
        return predicted_base, False

    # Apply int4 correction
    if params.apply_int4_thresholds:
        corrected_base, vote_count = apply_int4_correction(
            lens_similarities, int4_thresholds
        )

        # Only apply if confident (≥3 votes) and different
        if corrected_base != predicted_base and vote_count >= 3:
            return corrected_base, True

    return predicted_base, False


def test_optimized_params(
    predictions: List[Dict],
    teachable_moments: List[Dict],
    int4_thresholds: Dict[str, float],
    params: OptimizedCorrectionParams
) -> Dict:
    """
    Test optimized parameters focusing on teachable moments.

    Returns comprehensive stats including teachable moment detection rate.
    """
    # Build position lookup for teachable moments
    teachable_positions = {tm['position'] for tm in teachable_moments}

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
        # Skip N positions
        if pred.get('has_n', False):
            continue

        stats['total'] += 1

        ground_truth = pred['ground_truth']
        baseline_pred = pred['predicted']
        baseline_correct = (baseline_pred == ground_truth)

        # Apply correction
        corrected_pred, was_corrected = apply_optimized_correction(
            pred, int4_thresholds, params
        )

        corrected_correct = (corrected_pred == ground_truth)

        if baseline_correct:
            stats['baseline_correct'] += 1
        if corrected_correct:
            stats['corrected_correct'] += 1

        if was_corrected:
            stats['corrections_applied'] += 1

            # Check if this is a teachable moment
            position = pred['position']
            if position in teachable_positions:
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

    # Teachable moment metrics
    total_teachable = len(teachable_positions)
    stats['teachable_moment_detection_rate'] = (
        stats['teachable_moments_detected'] / total_teachable if total_teachable > 0 else 0
    )
    stats['teachable_moment_fix_rate'] = (
        stats['teachable_moments_fixed'] / total_teachable if total_teachable > 0 else 0
    )

    return stats


def optimize_correction_profiles(
    quantization: str = 'float32',
    validation_results_dir: str = None
):
    """
    Optimized correction profile tuning using cross-quantization insights.

    Reduced search space based on empirical findings:
    - Signal range: 0.4-0.7 (from analysis: mean 0.47-0.68)
    - Variance: 0.10-0.15 (from analysis: 0.13±0.10)
    - Confidence: 0.5-0.6 (from analysis: 0.55±0.13)
    - Vote count: 3 (from analysis: 46/65 positions)
    """
    logger.info("=" * 80)
    logger.info("OPTIMIZED CORRECTION PROFILE TUNING")
    logger.info("Using Cross-Quantization Teachable Moments")
    logger.info("=" * 80)
    logger.info(f"Quantization: {quantization}")
    logger.info("")

    # Load validation results
    if validation_results_dir is None:
        validation_results_dir = Path("HDV_VALIDATION_PACKAGE/architecture_testing/comparison_results")
    else:
        validation_results_dir = Path(validation_results_dir)

    logger.info("Loading data...")

    # Load predictions
    pred_file = validation_results_dir / f"{quantization}_predictions_detailed.json"
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_file}")

    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    logger.info(f"  ✓ Loaded {len(predictions)} predictions")

    # Load teachable moments
    analysis_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/cross_quantization_error_analysis.json")
    teachable_moments = load_teachable_moments(analysis_file)

    # Filter predictions to ground truth only
    predictions = [p for p in predictions if not p.get('has_n', False)]
    logger.info(f"  ✓ {len(predictions)} positions with ground truth")
    logger.info("")

    # Load thresholds
    from genomevault.hdv_validation.adaptive_correction import OPTIMAL_THRESHOLDS
    int4_thresholds = OPTIMAL_THRESHOLDS['int4']

    logger.info("Int4 thresholds (ultra-permissive):")
    for lens, thresh in int4_thresholds.items():
        logger.info(f"  {lens}: {thresh:.4f}")
    logger.info("")

    # Define FOCUSED parameter sweep based on empirical findings
    logger.info("Parameter sweep ranges (empirically focused):")

    # Signal range: centered around observed 0.4-0.7
    signal_min_vals = [0.35, 0.40, 0.45]
    signal_max_vals = [0.65, 0.70, 0.75, 0.80]

    # Variance: centered around observed 0.13
    variance_vals = [0.10, 0.12, 0.14, 0.16]

    # Confidence: centered around observed 0.55
    conf_min_vals = [0.45, 0.50, 0.52]
    conf_max_vals = [0.58, 0.60, 0.65]

    # Vote count: observed dominant pattern is 3
    vote_count_vals = [3]  # Focus on dominant pattern

    logger.info(f"  signal_min: {signal_min_vals}")
    logger.info(f"  signal_max: {signal_max_vals}")
    logger.info(f"  variance: {variance_vals}")
    logger.info(f"  conf_min: {conf_min_vals}")
    logger.info(f"  conf_max: {conf_max_vals}")
    logger.info(f"  vote_count: {vote_count_vals}")
    logger.info("")

    total_combos = (
        len(signal_min_vals) * len(signal_max_vals) * len(variance_vals) *
        len(conf_min_vals) * len(conf_max_vals) * len(vote_count_vals)
    )
    logger.info(f"Total combinations to test: {total_combos}")
    logger.info(f"(vs. 832 in original unfocused approach)")
    logger.info("")

    # Parameter sweep
    best_params = None
    best_improvement = -float('inf')
    best_stats = None

    logger.info("Starting parameter sweep...")
    logger.info("")

    tested = 0
    for sig_min in signal_min_vals:
        for sig_max in signal_max_vals:
            if sig_min >= sig_max:
                continue

            for var in variance_vals:
                for conf_min in conf_min_vals:
                    for conf_max in conf_max_vals:
                        if conf_min >= conf_max:
                            continue

                        for vote_count in vote_count_vals:
                            params = OptimizedCorrectionParams(
                                float32_overfitting_signal_min=sig_min,
                                float32_overfitting_signal_max=sig_max,
                                low_variance_threshold=var,
                                medium_confidence_min=conf_min,
                                medium_confidence_max=conf_max,
                                dominant_vote_count=vote_count,
                                apply_int4_thresholds=True
                            )

                            stats = test_optimized_params(
                                predictions, teachable_moments, int4_thresholds, params
                            )

                            tested += 1
                            if tested % 10 == 0:
                                logger.info(f"  Progress: {tested}/{total_combos}")

                            # Optimize for net benefit (errors fixed - correct broken)
                            # But also consider teachable moment detection rate
                            score = stats['net_benefit'] + (stats['teachable_moment_fix_rate'] * 10)

                            if score > best_improvement:
                                best_improvement = score
                                best_params = params
                                best_stats = stats

    logger.info("")
    logger.info("=" * 80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 80)
    logger.info("")

    logger.info("Best parameters found:")
    logger.info(f"  signal_min: {best_params.float32_overfitting_signal_min:.4f}")
    logger.info(f"  signal_max: {best_params.float32_overfitting_signal_max:.4f}")
    logger.info(f"  variance_threshold: {best_params.low_variance_threshold:.4f}")
    logger.info(f"  confidence_min: {best_params.medium_confidence_min:.4f}")
    logger.info(f"  confidence_max: {best_params.medium_confidence_max:.4f}")
    logger.info(f"  dominant_vote_count: {best_params.dominant_vote_count}")
    logger.info("")

    logger.info("Performance metrics:")
    logger.info(f"  Baseline accuracy: {best_stats['baseline_accuracy']:.4%}")
    logger.info(f"  Corrected accuracy: {best_stats['corrected_accuracy']:.4%}")
    logger.info(f"  Improvement: {best_stats['improvement']*100:+.2f}%")
    logger.info("")

    logger.info("Correction statistics:")
    logger.info(f"  Errors fixed: {best_stats['errors_fixed']}")
    logger.info(f"  Correct broken: {best_stats['correct_broken']}")
    logger.info(f"  Net benefit: {best_stats['net_benefit']}")
    logger.info(f"  Total corrections applied: {best_stats['corrections_applied']}")
    logger.info("")

    logger.info("Teachable moment metrics:")
    logger.info(f"  Total teachable moments: {len(teachable_moments)}")
    logger.info(f"  Detected: {best_stats['teachable_moments_detected']} ({best_stats['teachable_moment_detection_rate']:.1%})")
    logger.info(f"  Fixed: {best_stats['teachable_moments_fixed']} ({best_stats['teachable_moment_fix_rate']:.1%})")
    logger.info("")

    # Save results
    output_path = Path(f"HDV_VALIDATION_PACKAGE/architecture_testing/optimized_correction_params_{quantization}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'quantization': quantization,
        'sample_size': len(predictions),
        'teachable_moments_count': len(teachable_moments),
        'optimal_parameters': {
            'float32_overfitting_signal_min': best_params.float32_overfitting_signal_min,
            'float32_overfitting_signal_max': best_params.float32_overfitting_signal_max,
            'low_variance_threshold': best_params.low_variance_threshold,
            'medium_confidence_min': best_params.medium_confidence_min,
            'medium_confidence_max': best_params.medium_confidence_max,
            'dominant_vote_count': best_params.dominant_vote_count,
            'apply_int4_thresholds': best_params.apply_int4_thresholds
        },
        'performance': best_stats
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Results saved to: {output_path}")
    logger.info("")

    return best_params, best_stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Optimized correction profile tuning using cross-quantization insights'
    )
    parser.add_argument(
        '--quantization',
        default='float32',
        choices=['float32', 'int8', 'int4', 'binary'],
        help='Quantization mode to tune'
    )
    parser.add_argument(
        '--validation-results',
        type=str,
        default=None,
        help='Path to validation results directory'
    )

    args = parser.parse_args()

    optimize_correction_profiles(
        quantization=args.quantization,
        validation_results_dir=args.validation_results
    )
