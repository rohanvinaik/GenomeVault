#!/usr/bin/env python3
"""
Cross-Quantization Error Pattern Analysis

Finds positions where float32 fails but lower-precision quantizations succeed,
then analyzes what characteristics allow int4/binary to generalize better.

This is the KEY to adaptive correction - we need to detect when float32 is
overfitting and apply int4-style decision making.
"""

import json
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_all_predictions(validation_results_dir: Path) -> Dict[str, List[Dict]]:
    """Load predictions from all quantizations."""
    quantizations = ['float32', 'int8', 'int4', 'binary']
    predictions = {}

    for quant in quantizations:
        pred_file = validation_results_dir / f"{quant}_predictions_detailed.json"
        if pred_file.exists():
            with open(pred_file, 'r') as f:
                predictions[quant] = json.load(f)
            logger.info(f"  ✓ Loaded {len(predictions[quant])} {quant} predictions")
        else:
            logger.warning(f"  ✗ {quant} predictions not found: {pred_file}")

    return predictions


def categorize_error_patterns(predictions: Dict[str, List[Dict]]) -> Dict:
    """
    Categorize errors by which quantizations fail/succeed.

    CRITICAL FIX: Build position-aligned comparison by finding overlapping positions.

    Categories:
    - universal_errors: ALL quantizations fail
    - high_precision_only: float32/int8 fail, int4/binary succeed
    - low_precision_only: int4/binary fail, float32/int8 succeed
    - float32_only: ONLY float32 fails
    """
    categories = {
        'universal_errors': [],
        'high_precision_only': [],
        'low_precision_only': [],
        'float32_only': [],
        'teachable_moments': []  # float32 wrong, ANY other right
    }

    quantizations = list(predictions.keys())

    # Build position-indexed lookup for each quantization
    logger.info("Building position-aligned comparison...")
    position_data = {}

    for quant in quantizations:
        for pred in predictions[quant]:
            pos = pred['position']
            if pos not in position_data:
                position_data[pos] = {}
            position_data[pos][quant] = pred

    # Filter to only positions present in ALL quantizations
    complete_positions = [
        pos for pos, data in position_data.items()
        if all(q in data for q in quantizations)
    ]

    logger.info(f"  Total unique positions across all quantizations: {len(position_data)}")
    logger.info(f"  Positions present in ALL quantizations: {len(complete_positions)}")
    logger.info("")

    if len(complete_positions) == 0:
        logger.error("  ⚠️  NO OVERLAPPING POSITIONS! Cannot perform cross-quantization analysis.")
        return categories

    # Analyze overlapping positions only
    for pos in complete_positions:
        data = position_data[pos]

        # Skip N positions
        if data[quantizations[0]].get('has_n', False):
            continue

        ground_truth = data[quantizations[0]]['ground_truth']

        # Verify ground truth is consistent across quantizations
        if not all(data[q]['ground_truth'] == ground_truth for q in quantizations):
            logger.warning(f"  ⚠️  Position {pos}: inconsistent ground truth across quantizations")
            continue

        # Get correctness for each quantization
        correctness = {
            q: data[q]['correct']
            for q in quantizations
        }

        # Build error entry
        error_entry = {
            'position': pos,
            'ground_truth': ground_truth,
            'predictions': {q: data[q]['predicted'] for q in quantizations},
            'correctness': correctness,
            'confidences': {q: data[q]['confidence'] for q in quantizations},
            'lens_results': {q: data[q]['lens_results'] for q in quantizations},
            'votes': {q: data[q]['votes'] for q in quantizations}
        }

        # Categorize
        float32_wrong = not correctness.get('float32', True)
        int8_wrong = not correctness.get('int8', True)
        int4_wrong = not correctness.get('int4', True)
        binary_wrong = not correctness.get('binary', True)

        all_wrong = all(not correctness[q] for q in quantizations)
        any_right = any(correctness[q] for q in quantizations)

        if all_wrong:
            categories['universal_errors'].append(error_entry)

        if float32_wrong and int8_wrong and not int4_wrong and not binary_wrong:
            categories['high_precision_only'].append(error_entry)

        if not float32_wrong and not int8_wrong and int4_wrong and binary_wrong:
            categories['low_precision_only'].append(error_entry)

        if float32_wrong and not int8_wrong:
            categories['float32_only'].append(error_entry)

        # Teachable moments: float32 wrong but ANY other quantization right
        if float32_wrong and any_right:
            categories['teachable_moments'].append(error_entry)

    return categories


def analyze_error_category(errors: List[Dict], category_name: str):
    """Analyze characteristics of an error category."""
    logger.info(f"\n{'='*80}")
    logger.info(f"{category_name.upper().replace('_', ' ')}")
    logger.info(f"{'='*80}")
    logger.info(f"Total errors: {len(errors)}")

    if len(errors) == 0:
        logger.info("  No errors in this category")
        return

    # Analyze prediction patterns
    prediction_matrix = defaultdict(lambda: defaultdict(int))

    for error in errors:
        gt = error['ground_truth']
        float32_pred = error['predictions']['float32']
        prediction_matrix[gt][float32_pred] += 1

    logger.info("\nPrediction confusion (Truth → float32 Prediction):")
    logger.info(f"{'':>8} {'→A':>6} {'→T':>6} {'→G':>6} {'→C':>6}")
    for truth in 'ATGC':
        row = f"{truth:>8} │"
        for pred in 'ATGC':
            count = prediction_matrix[truth][pred]
            total = sum(prediction_matrix[truth].values())
            if count > 0:
                row += f" {count}/{total:<4}"
            else:
                row += "   —   "
        logger.info(row)

    # Analyze lens characteristics for float32 errors
    logger.info("\nLens magnitude analysis for float32:")
    all_magnitudes = {lens: [] for lens in ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']}

    for error in errors:
        lens_results = error['lens_results']['float32']
        for lens, sim in lens_results.items():
            all_magnitudes[lens].append(abs(sim))

    logger.info(f"{'Lens':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    for lens in ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']:
        mags = all_magnitudes[lens]
        if mags:
            logger.info(
                f"{lens:<8} {np.mean(mags):>8.3f} {np.std(mags):>8.3f} "
                f"{np.min(mags):>8.3f} {np.max(mags):>8.3f}"
            )

    # Analyze variance
    variances = []
    for error in errors:
        lens_results = error['lens_results']['float32']
        mags = [abs(v) for v in lens_results.values()]
        variances.append(np.var(mags))

    logger.info(f"\nVariance across lenses:")
    logger.info(f"  Mean variance: {np.mean(variances):.4f}")
    logger.info(f"  Std variance: {np.std(variances):.4f}")

    # Analyze confidence
    confidences = [error['confidences']['float32'] for error in errors]
    logger.info(f"\nConfidence distribution:")
    logger.info(f"  Mean: {np.mean(confidences):.4f}")
    logger.info(f"  Std: {np.std(confidences):.4f}")
    logger.info(f"  Min: {np.min(confidences):.4f}")
    logger.info(f"  Max: {np.max(confidences):.4f}")

    # Analyze vote distribution
    vote_dist = defaultdict(int)
    for error in errors:
        votes = error['votes']['float32']
        max_votes = max(votes.values())
        vote_dist[max_votes] += 1

    logger.info(f"\nVote distribution (max votes per position):")
    for vote_count in sorted(vote_dist.keys(), reverse=True):
        logger.info(f"  {vote_count} votes: {vote_dist[vote_count]} positions")


def compare_teachable_moments(errors: List[Dict]):
    """
    For teachable moments, compare float32 characteristics to the
    quantization that got it right.
    """
    logger.info(f"\n{'='*80}")
    logger.info("TEACHABLE MOMENTS - DETAILED ANALYSIS")
    logger.info(f"{'='*80}")

    # For each error, find which quantization(s) got it right
    correction_sources = defaultdict(int)

    for error in errors:
        for quant in ['int8', 'int4', 'binary']:
            if error['correctness'].get(quant, False):
                correction_sources[quant] += 1

    logger.info(f"\nQuantizations that can correct float32 errors:")
    for quant, count in sorted(correction_sources.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {quant}: {count} positions")

    # Compare lens characteristics between float32 and successful quantizations
    logger.info(f"\nLens magnitude comparison (float32 vs. successful quantization):")

    for quant in ['int4', 'binary']:
        # Find errors where this quantization succeeded
        relevant_errors = [
            e for e in errors
            if e['correctness'].get(quant, False)
        ]

        if not relevant_errors:
            continue

        logger.info(f"\n{quant.upper()} successes ({len(relevant_errors)} positions):")

        for lens in ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']:
            float32_mags = [abs(e['lens_results']['float32'][lens]) for e in relevant_errors]
            quant_mags = [abs(e['lens_results'][quant][lens]) for e in relevant_errors]

            logger.info(
                f"  {lens}: float32={np.mean(float32_mags):.3f}±{np.std(float32_mags):.3f}, "
                f"{quant}={np.mean(quant_mags):.3f}±{np.std(quant_mags):.3f}"
            )


def main():
    """Main analysis."""
    logger.info("="*80)
    logger.info("CROSS-QUANTIZATION ERROR PATTERN ANALYSIS")
    logger.info("="*80)
    logger.info("")

    # Load predictions
    validation_results_dir = Path("HDV_VALIDATION_PACKAGE/architecture_testing/comparison_results")

    logger.info("Loading predictions from all quantizations...")
    predictions = load_all_predictions(validation_results_dir)

    if 'float32' not in predictions:
        logger.error("float32 predictions not found!")
        return

    logger.info("")

    # Categorize errors
    logger.info("Categorizing error patterns...")
    categories = categorize_error_patterns(predictions)
    logger.info("")

    # Print category sizes
    logger.info("Error category sizes:")
    for cat_name, errors in categories.items():
        logger.info(f"  {cat_name}: {len(errors)} positions")
    logger.info("")

    # Analyze each category
    for cat_name, errors in categories.items():
        if errors:
            analyze_error_category(errors, cat_name)

    # Special analysis for teachable moments
    if categories['teachable_moments']:
        compare_teachable_moments(categories['teachable_moments'])

    # Save detailed results
    output_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/cross_quantization_error_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'category_sizes': {k: len(v) for k, v in categories.items()},
        'categories': categories
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info(f"✓ Detailed results saved to: {output_file}")


if __name__ == '__main__':
    main()
