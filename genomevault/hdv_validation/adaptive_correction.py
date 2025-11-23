#!/usr/bin/env python3
"""
Adaptive Threshold Correction System

Post-processing error correction that detects and fixes common error patterns
without changing the underlying quantization or base predictions.

This runs AFTER standard validation to show potential accuracy gains.
"""

import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class CorrectionResult:
    """Result of adaptive correction."""
    base: str
    confidence: float
    method: str
    original_base: str
    original_confidence: float
    was_corrected: bool


# Optimal thresholds from validation (Nov 19, 2025)
OPTIMAL_THRESHOLDS = {
    'float32': {
        'AT': 0.05,
        'GC': 0.00,
        'PuPy': 0.20,
        'AmKe': 0.20,
        'StWk': 0.20,
    },
    'int8': {
        'AT': 0.05,
        'GC': 0.00,
        'PuPy': 0.10,
        'AmKe': 0.10,
        'StWk': 0.15,
    },
    'int4': {
        'AT': 0.0028,
        'GC': 0.00,
        'PuPy': 0.0083,
        'AmKe': 0.0055,
        'StWk': 0.0083,
    },
    'binary': {
        'AT': 0.0025,
        'GC': 0.00,
        'PuPy': 0.0020,
        'AmKe': 0.0012,
        'StWk': 0.0020,
    },
}


def detect_low_precision_profile(
    lens_similarities: Dict[str, float],
    votes: Dict[str, int],
    confidence: float
) -> bool:
    """
    Detects weak but consistent signals that quantization might lose.

    Signature:
    - Small lens magnitudes (0.2-0.5 range)
    - High lens agreement (4-5 lenses vote for same base)
    - Low absolute confidence
    """
    magnitudes = [abs(v) for v in lens_similarities.values()]
    mean_mag = np.mean(magnitudes)

    # Check lens agreement
    max_votes = max(votes.values())
    total_votes = sum(votes.values())
    agreement_ratio = max_votes / total_votes if total_votes > 0 else 0

    # Detection criteria (from empirical analysis)
    is_weak_signal = (0.2 < mean_mag < 0.5)
    is_high_agreement = (agreement_ratio > 0.6)  # 3+ out of 5 lenses agree
    is_low_confidence = (confidence <= 0.6)

    return is_weak_signal and is_high_agreement and is_low_confidence


def detect_high_precision_profile(
    lens_similarities: Dict[str, float],
    votes: Dict[str, int],
    base_thresholds: Dict[str, float]
) -> bool:
    """
    Detects threshold trap positions where float32 is too strict.

    Signature:
    - Moderate lens magnitudes
    - High variance across lenses
    - Multiple lens values just below threshold
    """
    magnitudes = [abs(v) for v in lens_similarities.values()]
    mean_mag = np.mean(magnitudes)
    variance = np.var(magnitudes)

    # Count how many lenses are "near miss" (just below threshold)
    near_miss_count = 0
    for lens, sim in lens_similarities.items():
        threshold = base_thresholds[lens]
        # Within 20% of threshold but below it
        if 0 < (threshold - abs(sim)) < (threshold * 0.2):
            near_miss_count += 1

    # Detection criteria
    is_moderate_signal = (0.4 < mean_mag < 0.8)
    is_high_variance = (variance > 0.15)  # From empirical data
    has_near_misses = (near_miss_count >= 2)

    return is_moderate_signal and is_high_variance and has_near_misses


def vote_for_base_with_threshold(
    lens_name: str,
    similarity: float,
    threshold: float
) -> str:
    """Vote for a base given lens similarity and threshold."""
    # Lens signatures (from validation_utils.py)
    NUCLEOTIDE_SIGNATURES = {
        'A': {'AT': +1, 'GC': 0, 'PuPy': +1, 'AmKe': +1, 'StWk': -1},
        'T': {'AT': -1, 'GC': 0, 'PuPy': -1, 'AmKe': -1, 'StWk': -1},
        'G': {'AT': 0, 'GC': +1, 'PuPy': +1, 'AmKe': -1, 'StWk': +1},
        'C': {'AT': 0, 'GC': -1, 'PuPy': -1, 'AmKe': +1, 'StWk': +1},
    }

    # Check which base this lens votes for
    for base, signature in NUCLEOTIDE_SIGNATURES.items():
        expected_sign = signature[lens_name]

        if expected_sign == 0:
            continue  # Neutral lens for this base
        elif expected_sign > 0 and similarity > threshold:
            return base
        elif expected_sign < 0 and similarity < -threshold:
            return base

    return None  # No vote


def correct_low_precision(
    lens_similarities: Dict[str, float],
    votes: Dict[str, int]
) -> CorrectionResult:
    """
    For weak signals: trust the consensus even if magnitude is small.

    Strategy: Use "soft voting" - if all lenses weakly agree,
    boost confidence even though magnitudes are small.
    """
    # Find the consensus base (what most lenses agree on)
    consensus_base = max(votes, key=votes.get)
    consensus_votes = votes[consensus_base]
    original_confidence = consensus_votes / 5.0

    # If 4-5 lenses agree, trust it even with weak signal
    if consensus_votes >= 4:
        boosted_confidence = 0.8  # Promote to high confidence
        return CorrectionResult(
            base=consensus_base,
            confidence=boosted_confidence,
            method="low_precision_corrected",
            original_base=consensus_base,
            original_confidence=original_confidence,
            was_corrected=True
        )

    elif consensus_votes == 3:
        # Check if the 3 agreeing lenses are CONSISTENT (same sign)
        # This is a simpler heuristic - if 3/5 agree, moderately boost confidence
        boosted_confidence = 0.6
        return CorrectionResult(
            base=consensus_base,
            confidence=boosted_confidence,
            method="low_precision_corrected",
            original_base=consensus_base,
            original_confidence=original_confidence,
            was_corrected=True
        )

    # Otherwise, keep original prediction
    return CorrectionResult(
        base=consensus_base,
        confidence=original_confidence,
        method="low_precision_uncertain",
        original_base=consensus_base,
        original_confidence=original_confidence,
        was_corrected=False
    )


def correct_high_precision(
    lens_similarities: Dict[str, float],
    votes: Dict[str, int],
    base_thresholds: Dict[str, float]
) -> CorrectionResult:
    """
    For threshold traps: relax thresholds using int4 values.

    Strategy: Re-vote using more permissive int4 thresholds,
    which empirically generalize better for these positions.
    """
    original_base = max(votes, key=votes.get)
    original_confidence = votes[original_base] / 5.0

    # Int4 thresholds (from validation report)
    INT4_THRESHOLDS = OPTIMAL_THRESHOLDS['int4']

    # Re-vote with relaxed thresholds
    relaxed_votes = {base: 0 for base in ['A', 'T', 'G', 'C']}

    for lens, similarity in lens_similarities.items():
        threshold = INT4_THRESHOLDS[lens]
        voted_base = vote_for_base_with_threshold(lens, similarity, threshold)
        if voted_base:
            relaxed_votes[voted_base] += 1

    # Find new winner
    corrected_base = max(relaxed_votes, key=relaxed_votes.get)
    max_votes = relaxed_votes[corrected_base]

    # Confidence based on vote strength
    confidence_map = {5: 1.0, 4: 0.8, 3: 0.6, 2: 0.4, 1: 0.2, 0: 0.0}
    corrected_confidence = confidence_map.get(max_votes, 0.2)

    # Only accept correction if it's actually different AND confident
    if corrected_base != original_base and corrected_confidence >= 0.6:
        return CorrectionResult(
            base=corrected_base,
            confidence=corrected_confidence,
            method="high_precision_corrected",
            original_base=original_base,
            original_confidence=original_confidence,
            was_corrected=True
        )
    else:
        return CorrectionResult(
            base=original_base,
            confidence=corrected_confidence,
            method="high_precision_reviewed",
            original_base=original_base,
            original_confidence=original_confidence,
            was_corrected=False
        )


def apply_adaptive_correction(
    prediction: Dict,
    quantization: str = 'float32'
) -> CorrectionResult:
    """
    Main prediction pipeline with adaptive threshold correction.
    Only activates for low-confidence positions.

    Args:
        prediction: Dict with 'lens_results', 'votes', 'confidence', 'predicted'
        quantization: Quantization mode to use for base thresholds

    Returns:
        CorrectionResult with potentially corrected prediction
    """
    lens_similarities = prediction['lens_results']
    votes = prediction['votes']
    confidence = prediction['confidence']
    predicted_base = prediction['predicted']

    # Get base thresholds for this quantization
    base_thresholds = OPTIMAL_THRESHOLDS.get(quantization, OPTIMAL_THRESHOLDS['float32'])

    # Step 1: Fast confidence check (>99% of queries exit here)
    if confidence >= 0.8:  # High confidence - trust it
        return CorrectionResult(
            base=predicted_base,
            confidence=confidence,
            method="standard",
            original_base=predicted_base,
            original_confidence=confidence,
            was_corrected=False
        )

    # Step 2: Low confidence - profile the position
    is_low_precision = detect_low_precision_profile(lens_similarities, votes, confidence)
    is_high_precision = detect_high_precision_profile(lens_similarities, votes, base_thresholds)

    # Step 3: Apply correction if profile matches
    if is_low_precision:
        return correct_low_precision(lens_similarities, votes)
    elif is_high_precision:
        return correct_high_precision(lens_similarities, votes, base_thresholds)
    else:
        # No profile match, return original
        return CorrectionResult(
            base=predicted_base,
            confidence=confidence,
            method="standard",
            original_base=predicted_base,
            original_confidence=confidence,
            was_corrected=False
        )


def analyze_corrections(
    predictions: List[Dict],
    quantization: str = 'float32'
) -> Dict:
    """
    Analyze all predictions and apply adaptive correction.

    Args:
        predictions: List of prediction dicts from validation
        quantization: Quantization mode

    Returns:
        Dict with correction statistics and results
    """
    stats = {
        'total_queries': 0,
        'fast_path': 0,  # High confidence, no correction needed
        'low_precision_detected': 0,
        'high_precision_detected': 0,
        'corrections_applied': 0,
        'corrections_that_fixed_errors': 0,
        'corrections_that_introduced_errors': 0,
        'baseline_correct': 0,
        'corrected_correct': 0
    }

    corrected_predictions = []

    for pred in predictions:
        # Skip N positions (no ground truth)
        if pred.get('has_n', False):
            continue

        stats['total_queries'] += 1

        # Apply adaptive correction
        result = apply_adaptive_correction(pred, quantization)

        # Track statistics
        if result.method == "standard" and result.confidence >= 0.8:
            stats['fast_path'] += 1
        elif "low_precision" in result.method:
            stats['low_precision_detected'] += 1
        elif "high_precision" in result.method:
            stats['high_precision_detected'] += 1

        if result.was_corrected:
            stats['corrections_applied'] += 1

        # Check accuracy
        ground_truth = pred['ground_truth']
        baseline_correct = (pred['predicted'] == ground_truth)
        corrected_correct = (result.base == ground_truth)

        if baseline_correct:
            stats['baseline_correct'] += 1
        if corrected_correct:
            stats['corrected_correct'] += 1

        # Track if correction helped or hurt
        if result.was_corrected:
            if not baseline_correct and corrected_correct:
                stats['corrections_that_fixed_errors'] += 1
            elif baseline_correct and not corrected_correct:
                stats['corrections_that_introduced_errors'] += 1

        # Store corrected prediction
        corrected_predictions.append({
            **pred,
            'corrected_base': result.base,
            'corrected_confidence': result.confidence,
            'correction_method': result.method,
            'was_corrected': result.was_corrected,
            'corrected_correct': corrected_correct
        })

    # Calculate accuracies
    total = stats['total_queries']
    stats['baseline_accuracy'] = stats['baseline_correct'] / total if total > 0 else 0
    stats['corrected_accuracy'] = stats['corrected_correct'] / total if total > 0 else 0
    stats['accuracy_improvement'] = stats['corrected_accuracy'] - stats['baseline_accuracy']

    return {
        'statistics': stats,
        'corrected_predictions': corrected_predictions
    }
