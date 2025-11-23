#!/usr/bin/env python3
"""
Empirically determine optimal parameters for adaptive correction profiles.

This tunes the detection thresholds for low-precision and high-precision error patterns,
similar to how we tune per-lens thresholds.

Strategy:
1. Load a large validation dataset with known errors
2. Sweep through correction profile parameters
3. Measure effectiveness: errors fixed vs. correct predictions changed
4. Find optimal configuration
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import itertools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class CorrectionProfileParams:
    """Parameters for correction profile detection."""
    # Low-precision profile
    weak_signal_min: float
    weak_signal_max: float
    agreement_ratio_threshold: float
    low_confidence_threshold: float

    # High-precision profile
    moderate_signal_min: float
    moderate_signal_max: float
    variance_threshold: float
    near_miss_percentage: float
    min_near_miss_count: int


def detect_low_precision_profile(
    lens_similarities: Dict[str, float],
    votes: Dict[str, int],
    confidence: float,
    params: CorrectionProfileParams
) -> bool:
    """Detect weak but consistent signals (parameterized)."""
    magnitudes = [abs(v) for v in lens_similarities.values()]
    mean_mag = np.mean(magnitudes)

    max_votes = max(votes.values())
    total_votes = sum(votes.values())
    agreement_ratio = max_votes / total_votes if total_votes > 0 else 0

    is_weak_signal = (params.weak_signal_min < mean_mag < params.weak_signal_max)
    is_high_agreement = (agreement_ratio > params.agreement_ratio_threshold)
    is_low_confidence = (confidence <= params.low_confidence_threshold)

    return is_weak_signal and is_high_agreement and is_low_confidence


def detect_high_precision_profile(
    lens_similarities: Dict[str, float],
    votes: Dict[str, int],
    base_thresholds: Dict[str, float],
    params: CorrectionProfileParams
) -> bool:
    """Detect threshold trap positions (parameterized)."""
    magnitudes = [abs(v) for v in lens_similarities.values()]
    mean_mag = np.mean(magnitudes)
    variance = np.var(magnitudes)

    # Count near misses
    near_miss_count = 0
    for lens, sim in lens_similarities.items():
        threshold = base_thresholds[lens]
        # Within near_miss_percentage of threshold but below it
        if 0 < (threshold - abs(sim)) < (threshold * params.near_miss_percentage):
            near_miss_count += 1

    is_moderate_signal = (params.moderate_signal_min < mean_mag < params.moderate_signal_max)
    is_high_variance = (variance > params.variance_threshold)
    has_near_misses = (near_miss_count >= params.min_near_miss_count)

    return is_moderate_signal and is_high_variance and has_near_misses


def apply_correction_with_params(
    prediction: Dict,
    quantization: str,
    base_thresholds: Dict[str, float],
    params: CorrectionProfileParams
) -> Tuple[str, bool]:
    """
    Apply correction with given parameters.

    Returns:
        (corrected_base, was_corrected)
    """
    lens_similarities = prediction['lens_results']
    votes = prediction['votes']
    confidence = prediction['confidence']
    predicted_base = prediction['predicted']

    # Fast path for high confidence
    if confidence >= 0.8:
        return predicted_base, False

    # Detect profiles
    is_low_precision = detect_low_precision_profile(
        lens_similarities, votes, confidence, params
    )
    is_high_precision = detect_high_precision_profile(
        lens_similarities, votes, base_thresholds, params
    )

    # Apply corrections
    if is_low_precision:
        # Low-precision correction: trust consensus if 4-5 lenses agree
        consensus_base = max(votes, key=votes.get)
        consensus_votes = votes[consensus_base]

        if consensus_votes >= 4:
            return consensus_base, (consensus_base != predicted_base)
        elif consensus_votes == 3:
            return consensus_base, (consensus_base != predicted_base)
        else:
            return predicted_base, False

    elif is_high_precision:
        # High-precision correction: re-vote with int4 thresholds (more permissive)
        from genomevault.hdv_validation.adaptive_correction import OPTIMAL_THRESHOLDS, vote_for_base_with_threshold

        INT4_THRESHOLDS = OPTIMAL_THRESHOLDS['int4']

        relaxed_votes = {base: 0 for base in ['A', 'T', 'G', 'C']}
        for lens, similarity in lens_similarities.items():
            threshold = INT4_THRESHOLDS[lens]
            voted_base = vote_for_base_with_threshold(lens, similarity, threshold)
            if voted_base:
                relaxed_votes[voted_base] += 1

        corrected_base = max(relaxed_votes, key=relaxed_votes.get)
        max_votes = relaxed_votes[corrected_base]

        # Only accept if confident (≥3 votes) and different
        if corrected_base != predicted_base and max_votes >= 3:
            return corrected_base, True
        else:
            return predicted_base, False

    else:
        return predicted_base, False


def test_correction_params(
    predictions: List[Dict],
    quantization: str,
    base_thresholds: Dict[str, float],
    params: CorrectionProfileParams
) -> Dict:
    """Test a specific correction parameter configuration."""
    stats = {
        'total': 0,
        'baseline_correct': 0,
        'corrected_correct': 0,
        'corrections_applied': 0,
        'errors_fixed': 0,
        'correct_broken': 0
    }

    for pred in predictions:
        # Skip N positions (no ground truth)
        if pred.get('has_n', False):
            continue

        stats['total'] += 1

        ground_truth = pred['ground_truth']
        baseline_pred = pred['predicted']
        baseline_correct = (baseline_pred == ground_truth)

        # Apply correction
        corrected_pred, was_corrected = apply_correction_with_params(
            pred, quantization, base_thresholds, params
        )

        corrected_correct = (corrected_pred == ground_truth)

        if baseline_correct:
            stats['baseline_correct'] += 1
        if corrected_correct:
            stats['corrected_correct'] += 1

        if was_corrected:
            stats['corrections_applied'] += 1

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

    return stats


def load_predictions_from_validation(
    validation_results_dir: Path,
    quantization: str = 'float32'
) -> List[Dict]:
    """Load predictions from a previous validation run."""
    pred_file = validation_results_dir / f"{quantization}_predictions_detailed.json"

    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")

    with open(pred_file, 'r') as f:
        predictions = json.load(f)

    logger.info(f"  ✓ Loaded {len(predictions)} predictions from {pred_file}")

    return predictions


def tune_correction_profiles(
    quantization: str = 'float32',
    validation_results_dir: str = None
):
    """
    Empirically tune correction profile parameters.

    Args:
        quantization: Quantization mode to tune
        validation_results_dir: Directory with validation results (predictions JSON)
    """
    logger.info("=" * 80)
    logger.info("CORRECTION PROFILE PARAMETER TUNING")
    logger.info("=" * 80)
    logger.info(f"Quantization: {quantization}")
    logger.info("")

    # Load validation results
    if validation_results_dir is None:
        validation_results_dir = Path("HDV_VALIDATION_PACKAGE/architecture_testing/comparison_results")
    else:
        validation_results_dir = Path(validation_results_dir)

    logger.info("Loading validation predictions...")
    predictions = load_predictions_from_validation(validation_results_dir, quantization)

    # Filter to only positions with ground truth
    predictions = [p for p in predictions if not p.get('has_n', False)]
    logger.info(f"  ✓ {len(predictions)} positions with ground truth")
    logger.info("")

    # Load base thresholds
    from genomevault.hdv_validation.adaptive_correction import OPTIMAL_THRESHOLDS
    base_thresholds = OPTIMAL_THRESHOLDS[quantization]

    # Define parameter sweep ranges
    logger.info("Defining parameter sweep ranges...")
    logger.info("")

    # Low-precision profile parameters
    weak_signal_min_vals = [0.1, 0.15, 0.2, 0.25]
    weak_signal_max_vals = [0.4, 0.5, 0.6, 0.7]
    agreement_ratio_vals = [0.5, 0.6, 0.7, 0.8]
    low_conf_vals = [0.4, 0.5, 0.6, 0.7]

    # High-precision profile parameters
    moderate_min_vals = [0.3, 0.4, 0.5]
    moderate_max_vals = [0.6, 0.7, 0.8, 0.9]
    variance_vals = [0.10, 0.15, 0.20, 0.25]
    near_miss_pct_vals = [0.15, 0.20, 0.25, 0.30]
    min_near_miss_vals = [1, 2, 3]

    logger.info("Low-precision profile sweep:")
    logger.info(f"  weak_signal_min: {weak_signal_min_vals}")
    logger.info(f"  weak_signal_max: {weak_signal_max_vals}")
    logger.info(f"  agreement_ratio: {agreement_ratio_vals}")
    logger.info(f"  low_confidence: {low_conf_vals}")
    logger.info("")
    logger.info("High-precision profile sweep:")
    logger.info(f"  moderate_min: {moderate_min_vals}")
    logger.info(f"  moderate_max: {moderate_max_vals}")
    logger.info(f"  variance: {variance_vals}")
    logger.info(f"  near_miss_pct: {near_miss_pct_vals}")
    logger.info(f"  min_near_miss: {min_near_miss_vals}")
    logger.info("")

    # Phase 1: Optimize low-precision profile first
    logger.info("=" * 80)
    logger.info("PHASE 1: Low-Precision Profile Optimization")
    logger.info("=" * 80)
    logger.info("")

    best_low_prec_params = None
    best_low_prec_improvement = -float('inf')

    # Use baseline high-precision params during low-precision tuning
    baseline_high_prec = {
        'moderate_signal_min': 0.4,
        'moderate_signal_max': 0.8,
        'variance_threshold': 0.15,
        'near_miss_percentage': 0.2,
        'min_near_miss_count': 2
    }

    low_prec_combos = list(itertools.product(
        weak_signal_min_vals, weak_signal_max_vals,
        agreement_ratio_vals, low_conf_vals
    ))

    logger.info(f"Testing {len(low_prec_combos)} low-precision parameter combinations...")
    logger.info("")

    for i, (ws_min, ws_max, agr, lc) in enumerate(low_prec_combos):
        if ws_min >= ws_max:
            continue  # Invalid range

        params = CorrectionProfileParams(
            weak_signal_min=ws_min,
            weak_signal_max=ws_max,
            agreement_ratio_threshold=agr,
            low_confidence_threshold=lc,
            **baseline_high_prec
        )

        stats = test_correction_params(predictions, quantization, base_thresholds, params)

        if (i + 1) % 20 == 0:
            logger.info(f"  Progress: {i+1}/{len(low_prec_combos)}")

        if stats['improvement'] > best_low_prec_improvement:
            best_low_prec_improvement = stats['improvement']
            best_low_prec_params = {
                'weak_signal_min': ws_min,
                'weak_signal_max': ws_max,
                'agreement_ratio_threshold': agr,
                'low_confidence_threshold': lc
            }
            best_low_prec_stats = stats

    logger.info("")
    logger.info("Best low-precision profile parameters:")
    for key, val in best_low_prec_params.items():
        logger.info(f"  {key}: {val:.4f}")
    logger.info("")
    logger.info(f"Baseline accuracy: {best_low_prec_stats['baseline_accuracy']:.4%}")
    logger.info(f"Corrected accuracy: {best_low_prec_stats['corrected_accuracy']:.4%}")
    logger.info(f"Improvement: {best_low_prec_stats['improvement']*100:+.2f}%")
    logger.info(f"Errors fixed: {best_low_prec_stats['errors_fixed']}")
    logger.info(f"Correct broken: {best_low_prec_stats['correct_broken']}")
    logger.info(f"Net benefit: {best_low_prec_stats['net_benefit']}")
    logger.info("")

    # Phase 2: Optimize high-precision profile with best low-precision params
    logger.info("=" * 80)
    logger.info("PHASE 2: High-Precision Profile Optimization")
    logger.info("=" * 80)
    logger.info("")

    best_high_prec_params = None
    best_high_prec_improvement = -float('inf')

    high_prec_combos = list(itertools.product(
        moderate_min_vals, moderate_max_vals,
        variance_vals, near_miss_pct_vals, min_near_miss_vals
    ))

    logger.info(f"Testing {len(high_prec_combos)} high-precision parameter combinations...")
    logger.info("")

    for i, (mod_min, mod_max, var, nm_pct, nm_count) in enumerate(high_prec_combos):
        if mod_min >= mod_max:
            continue  # Invalid range

        params = CorrectionProfileParams(
            **best_low_prec_params,
            moderate_signal_min=mod_min,
            moderate_signal_max=mod_max,
            variance_threshold=var,
            near_miss_percentage=nm_pct,
            min_near_miss_count=nm_count
        )

        stats = test_correction_params(predictions, quantization, base_thresholds, params)

        if (i + 1) % 20 == 0:
            logger.info(f"  Progress: {i+1}/{len(high_prec_combos)}")

        if stats['improvement'] > best_high_prec_improvement:
            best_high_prec_improvement = stats['improvement']
            best_high_prec_params = {
                'moderate_signal_min': mod_min,
                'moderate_signal_max': mod_max,
                'variance_threshold': var,
                'near_miss_percentage': nm_pct,
                'min_near_miss_count': nm_count
            }
            best_high_prec_stats = stats

    logger.info("")
    logger.info("Best high-precision profile parameters:")
    for key, val in best_high_prec_params.items():
        if isinstance(val, int):
            logger.info(f"  {key}: {val}")
        else:
            logger.info(f"  {key}: {val:.4f}")
    logger.info("")
    logger.info(f"Baseline accuracy: {best_high_prec_stats['baseline_accuracy']:.4%}")
    logger.info(f"Corrected accuracy: {best_high_prec_stats['corrected_accuracy']:.4%}")
    logger.info(f"Improvement: {best_high_prec_stats['improvement']*100:+.2f}%")
    logger.info(f"Errors fixed: {best_high_prec_stats['errors_fixed']}")
    logger.info(f"Correct broken: {best_high_prec_stats['correct_broken']}")
    logger.info(f"Net benefit: {best_high_prec_stats['net_benefit']}")
    logger.info("")

    # Phase 3: Final test with both optimized profiles
    logger.info("=" * 80)
    logger.info("PHASE 3: Final Validation")
    logger.info("=" * 80)
    logger.info("")

    final_params = CorrectionProfileParams(
        **best_low_prec_params,
        **best_high_prec_params
    )

    final_stats = test_correction_params(predictions, quantization, base_thresholds, final_params)

    logger.info("Final optimized parameters:")
    logger.info("  Low-precision profile:")
    for key, val in best_low_prec_params.items():
        logger.info(f"    {key}: {val:.4f}")
    logger.info("  High-precision profile:")
    for key, val in best_high_prec_params.items():
        if isinstance(val, int):
            logger.info(f"    {key}: {val}")
        else:
            logger.info(f"    {key}: {val:.4f}")
    logger.info("")
    logger.info(f"Baseline accuracy: {final_stats['baseline_accuracy']:.4%}")
    logger.info(f"Corrected accuracy: {final_stats['corrected_accuracy']:.4%}")
    logger.info(f"Improvement: {final_stats['improvement']*100:+.2f}%")
    logger.info(f"Errors fixed: {final_stats['errors_fixed']}")
    logger.info(f"Correct broken: {final_stats['correct_broken']}")
    logger.info(f"Net benefit: {final_stats['net_benefit']}")
    logger.info(f"Corrections applied: {final_stats['corrections_applied']}")
    logger.info("")

    # Save results
    output_path = Path(f"HDV_VALIDATION_PACKAGE/architecture_testing/correction_profile_tuning_{quantization}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'quantization': quantization,
        'sample_size': len(predictions),
        'optimal_low_precision_params': best_low_prec_params,
        'optimal_high_precision_params': best_high_prec_params,
        'final_stats': final_stats
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Results saved to: {output_path}")
    logger.info("")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Tune correction profile parameters empirically'
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
        help='Path to validation results directory with prediction JSONs'
    )

    args = parser.parse_args()

    tune_correction_profiles(
        quantization=args.quantization,
        validation_results_dir=args.validation_results
    )
