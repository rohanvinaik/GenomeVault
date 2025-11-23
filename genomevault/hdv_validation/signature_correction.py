#!/usr/bin/env python3
"""
Signature-Based Correction System

Uses exhaustive search results to apply safe transformations that fix errors
without introducing new ones.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from genomevault.hdv_validation.validation_utils import (
    predict_multi_lens_voting
)

logger = logging.getLogger(__name__)


def load_signatures(signatures_path: Path, include_relaxed: bool = False) -> List[Dict]:
    """Load correction signatures from exhaustive search results.

    Args:
        signatures_path: Path to safe signatures file
        include_relaxed: If True, also load relaxed 5:1 ratio signatures

    Returns:
        List of signature dictionaries
    """
    signatures = []

    # Load safe signatures (breaks == 0)
    if signatures_path.exists():
        with open(signatures_path, 'r') as f:
            safe_sigs = json.load(f)

        # Filter to only safe signatures (fixes > 0, breaks == 0)
        safe_sigs = [sig for sig in safe_sigs if sig['fixes'] > 0 and sig['breaks'] == 0]
        signatures.extend(safe_sigs)
        logger.info(f"Loaded {len(safe_sigs)} safe signatures from {signatures_path}")
    else:
        logger.warning(f"Safe signatures file not found: {signatures_path}")

    # Load relaxed 5:1 ratio signatures if requested
    if include_relaxed:
        relaxed_path = signatures_path.parent / f"{signatures_path.stem}_relaxed_5to1.json"
        if relaxed_path.exists():
            with open(relaxed_path, 'r') as f:
                relaxed_sigs = json.load(f)

            # Filter to 5:1 or better ratio
            relaxed_sigs = [sig for sig in relaxed_sigs
                          if sig.get('breaks', 0) > 0 and sig['fixes'] >= 5 * sig['breaks']]
            signatures.extend(relaxed_sigs)
            logger.info(f"Loaded {len(relaxed_sigs)} relaxed 5:1 signatures from {relaxed_path}")
        else:
            logger.info(f"No relaxed signatures found at {relaxed_path}")

    return signatures


def apply_transform(lens_results: Dict[str, float], transform_name: str) -> Dict[str, float]:
    """Apply a transformation to lens results."""
    transforms = {
        'flip_AT': lambda l: {**l, 'AT': -l['AT']},
        'flip_GC': lambda l: {**l, 'GC': -l['GC']},
        'flip_PuPy': lambda l: {**l, 'PuPy': -l['PuPy']},
        'flip_AmKe': lambda l: {**l, 'AmKe': -l['AmKe']},
        'flip_StWk': lambda l: {**l, 'StWk': -l['StWk']},
        'drop_AT': lambda l: {**l, 'AT': 0.0},
        'drop_GC': lambda l: {**l, 'GC': 0.0},
        'drop_PuPy': lambda l: {**l, 'PuPy': 0.0},
        'drop_AmKe': lambda l: {**l, 'AmKe': 0.0},
        'drop_StWk': lambda l: {**l, 'StWk': 0.0},
        'dampen_AT_50%': lambda l: {**l, 'AT': l['AT'] * 0.5},
        'dampen_GC_50%': lambda l: {**l, 'GC': l['GC'] * 0.5},
        'dampen_PuPy_50%': lambda l: {**l, 'PuPy': l['PuPy'] * 0.5},
        'dampen_AmKe_50%': lambda l: {**l, 'AmKe': l['AmKe'] * 0.5},
        'dampen_StWk_50%': lambda l: {**l, 'StWk': l['StWk'] * 0.5},
        'boost_AT_2x': lambda l: {**l, 'AT': l['AT'] * 2.0},
        'boost_GC_2x': lambda l: {**l, 'GC': l['GC'] * 2.0},
        'boost_PuPy_2x': lambda l: {**l, 'PuPy': l['PuPy'] * 2.0},
        'boost_AmKe_2x': lambda l: {**l, 'AmKe': l['AmKe'] * 2.0},
        'boost_StWk_2x': lambda l: {**l, 'StWk': l['StWk'] * 2.0},
    }

    if transform_name not in transforms:
        logger.warning(f"Unknown transform: {transform_name}")
        return lens_results

    return transforms[transform_name](lens_results)


def matches_constraints(lens_results: Dict[str, float], constraints: Dict[str, float]) -> bool:
    """Check if lens results match the signature constraints."""
    for lens_name, threshold in constraints.items():
        if abs(lens_results[lens_name]) < threshold:
            return False
    return True


def apply_signature_corrections(
    prediction: Dict,
    signatures: List[Dict],
    quantization: str
) -> Tuple[str, float, bool, str]:
    """
    Apply signature-based corrections to a prediction.

    Args:
        prediction: Prediction dict with 'lens_results', 'predicted', etc.
        signatures: List of safe correction signatures
        quantization: Quantization level

    Returns:
        (corrected_nucleotide, confidence, was_corrected, transform_applied)
    """
    lens_results = prediction['lens_results']
    original_prediction = prediction['predicted']
    original_confidence = prediction['confidence']

    # Try each signature in order (they're sorted by number of fixes)
    for signature in signatures:
        transform_name = signature['transform']
        constraints = signature['constraints']

        # Check if this signature's constraints match
        if not matches_constraints(lens_results, constraints):
            continue

        # Apply the transformation
        transformed_lens = apply_transform(lens_results, transform_name)

        # Re-predict with transformed lens values
        new_prediction, new_confidence, new_votes = predict_multi_lens_voting(
            transformed_lens,
            quantization=quantization
        )

        # If prediction changed, use it
        if new_prediction != original_prediction:
            return new_prediction, new_confidence, True, transform_name

    # No signature matched or changed the prediction
    return original_prediction, original_confidence, False, "none"


def analyze_with_signatures(
    predictions: List[Dict],
    signatures_path: Path,
    quantization: str,
    include_relaxed: bool = True
) -> Dict:
    """
    Analyze predictions with signature-based corrections.

    Args:
        predictions: List of prediction dicts
        signatures_path: Path to exhaustive search results JSON
        quantization: Quantization level
        include_relaxed: Include relaxed 5:1 ratio signatures

    Returns:
        Dict with correction statistics and corrected predictions
    """
    # Load signatures (safe + relaxed if requested)
    signatures = load_signatures(signatures_path, include_relaxed=include_relaxed)

    if not signatures:
        logger.warning("No signatures loaded - returning baseline predictions")
        return {
            'corrected_predictions': predictions,
            'statistics': {
                'total_queries': len([p for p in predictions if not p.get('has_n', False)]),
                'corrections_applied': 0,
                'corrections_that_fixed_errors': 0,
                'corrections_that_introduced_errors': 0,
                'baseline_correct': sum(1 for p in predictions if p['correct'] and not p.get('has_n', False)),
                'corrected_correct': sum(1 for p in predictions if p['correct'] and not p.get('has_n', False)),
                'signatures_loaded': 0
            }
        }

    # Apply corrections
    corrected_predictions = []
    stats = {
        'total_queries': 0,
        'corrections_applied': 0,
        'corrections_that_fixed_errors': 0,
        'corrections_that_introduced_errors': 0,
        'baseline_correct': 0,
        'corrected_correct': 0,
        'signatures_loaded': len(signatures),
        'transforms_used': {}
    }

    for pred in predictions:
        # Skip N positions
        if pred.get('has_n', False):
            corrected_predictions.append(pred)
            continue

        stats['total_queries'] += 1

        # Track baseline
        if pred['correct']:
            stats['baseline_correct'] += 1

        # Apply signature-based correction
        corrected_nuc, corrected_conf, was_corrected, transform = apply_signature_corrections(
            pred, signatures, quantization
        )

        # Create corrected prediction
        corrected_pred = pred.copy()
        corrected_pred['original_predicted'] = pred['predicted']
        corrected_pred['original_confidence'] = pred['confidence']
        corrected_pred['predicted'] = corrected_nuc
        corrected_pred['confidence'] = corrected_conf
        corrected_pred['was_corrected'] = was_corrected
        corrected_pred['transform_applied'] = transform
        corrected_pred['correct'] = (corrected_nuc == pred['ground_truth'])

        corrected_predictions.append(corrected_pred)

        # Track correction stats
        if was_corrected:
            stats['corrections_applied'] += 1
            stats['transforms_used'][transform] = stats['transforms_used'].get(transform, 0) + 1

            # Did the correction fix an error?
            if not pred['correct'] and corrected_pred['correct']:
                stats['corrections_that_fixed_errors'] += 1
            # Did the correction introduce an error?
            elif pred['correct'] and not corrected_pred['correct']:
                stats['corrections_that_introduced_errors'] += 1

        # Track corrected accuracy
        if corrected_pred['correct']:
            stats['corrected_correct'] += 1

    # Compute accuracies
    if stats['total_queries'] > 0:
        stats['baseline_accuracy'] = stats['baseline_correct'] / stats['total_queries']
        stats['corrected_accuracy'] = stats['corrected_correct'] / stats['total_queries']
        stats['improvement'] = stats['corrected_accuracy'] - stats['baseline_accuracy']
    else:
        stats['baseline_accuracy'] = 0
        stats['corrected_accuracy'] = 0
        stats['improvement'] = 0

    return {
        'corrected_predictions': corrected_predictions,
        'statistics': stats
    }
