#!/usr/bin/env python3
"""
Discover Safe Correction Signatures for int4 (LOW-PRECISION ERRORS)

Conservative approach using ONLY int4's own lens values.
We won't have float32 in production, so corrections must work from int4 signals alone.

Low-precision errors: float32 CORRECT, int4 WRONG (37 positions)
Expected simpler patterns than high-precision errors.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

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
class SignatureConstraints:
    """Constraints for detecting a correction signature."""
    name: str
    # Sign requirements (None = any, True = positive, False = negative)
    at_sign: Optional[bool] = None
    gc_sign: Optional[bool] = None
    pupy_sign: Optional[bool] = None
    amke_sign: Optional[bool] = None
    stwk_sign: Optional[bool] = None

    # Magnitude requirements
    at_min: Optional[float] = None
    gc_min: Optional[float] = None
    pupy_min: Optional[float] = None
    amke_min: Optional[float] = None
    stwk_min: Optional[float] = None

    # Transformation to apply
    transform_multipliers: Dict[str, float] = None

    def matches(self, lens_results: Dict[str, float]) -> bool:
        """Check if lens results match this signature."""
        # Check signs
        if self.at_sign is not None:
            if (lens_results['AT'] > 0) != self.at_sign:
                return False
        if self.gc_sign is not None:
            if (lens_results['GC'] > 0) != self.gc_sign:
                return False
        if self.pupy_sign is not None:
            if (lens_results['PuPy'] > 0) != self.pupy_sign:
                return False
        if self.amke_sign is not None:
            if (lens_results['AmKe'] > 0) != self.amke_sign:
                return False
        if self.stwk_sign is not None:
            if (lens_results['StWk'] > 0) != self.stwk_sign:
                return False

        # Check magnitudes
        if self.at_min is not None and abs(lens_results['AT']) < self.at_min:
            return False
        if self.gc_min is not None and abs(lens_results['GC']) < self.gc_min:
            return False
        if self.pupy_min is not None and abs(lens_results['PuPy']) < self.pupy_min:
            return False
        if self.amke_min is not None and abs(lens_results['AmKe']) < self.amke_min:
            return False
        if self.stwk_min is not None and abs(lens_results['StWk']) < self.stwk_min:
            return False

        return True


def test_signature(
    signature: SignatureConstraints,
    low_precision_errors: List[Dict],
    correct_predictions: List[Dict],
    max_confidence: float = 0.65
) -> Dict:
    """Test a signature's performance on int4 low-precision errors."""

    int4_thresholds = {'AT': 0.05, 'GC': 0.00, 'PuPy': 0.20, 'AmKe': 0.20, 'StWk': 0.20}
    ultra_low = {'AT': 0.01, 'GC': 0.01, 'PuPy': 0.01, 'AmKe': 0.01, 'StWk': 0.01}

    # Test on low-precision errors
    errors_fixed = 0
    signatures_detected = 0

    for err in low_precision_errors:
        lens_results = err['int4_lens']
        confidence = err['int4_conf']
        ground_truth = err['ground_truth']

        if confidence > max_confidence:
            continue

        if not signature.matches(lens_results):
            continue

        signatures_detected += 1

        # Original int4 prediction
        orig_votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for lens, sim in lens_results.items():
            voted_base = vote_for_base_with_threshold(lens, sim, int4_thresholds[lens])
            if voted_base:
                orig_votes[voted_base] += 1
        orig_pred = max(orig_votes, key=orig_votes.get)

        # Apply transformation
        trans_lens = {}
        for lens, sim in lens_results.items():
            if lens in signature.transform_multipliers:
                trans_lens[lens] = sim * signature.transform_multipliers[lens]

        # Vote with transformed lens
        trans_votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for lens, sim in trans_lens.items():
            threshold = ultra_low.get(lens, 0.01)
            voted_base = vote_for_base_with_threshold(lens, sim, threshold)
            if voted_base:
                trans_votes[voted_base] += 1

        trans_pred = max(trans_votes, key=trans_votes.get)
        trans_count = trans_votes[trans_pred]

        # Check if fixes
        if trans_pred != orig_pred and trans_count >= 2 and trans_pred == ground_truth:
            errors_fixed += 1

    # Test on correct predictions
    import random
    random.seed(42)
    sample_correct = random.sample(correct_predictions, min(1000, len(correct_predictions)))

    correct_broken = 0
    false_positives = 0

    for pred in sample_correct:
        lens_results = pred['lens_results']
        confidence = pred['confidence']
        ground_truth = pred['ground_truth']

        if confidence > max_confidence:
            continue

        if not signature.matches(lens_results):
            continue

        false_positives += 1

        # Original prediction
        orig_votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for lens, sim in lens_results.items():
            voted_base = vote_for_base_with_threshold(lens, sim, int4_thresholds[lens])
            if voted_base:
                orig_votes[voted_base] += 1
        orig_pred = max(orig_votes, key=orig_votes.get)

        # Apply transformation
        trans_lens = {}
        for lens, sim in lens_results.items():
            if lens in signature.transform_multipliers:
                trans_lens[lens] = sim * signature.transform_multipliers[lens]

        # Vote
        trans_votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for lens, sim in trans_lens.items():
            threshold = ultra_low.get(lens, 0.01)
            voted_base = vote_for_base_with_threshold(lens, sim, threshold)
            if voted_base:
                trans_votes[voted_base] += 1

        trans_pred = max(trans_votes, key=trans_votes.get)
        trans_count = trans_votes[trans_pred]

        # Check if breaks
        if trans_pred != orig_pred and trans_count >= 2 and trans_pred != ground_truth:
            correct_broken += 1

    return {
        'signatures_detected_in_errors': signatures_detected,
        'errors_fixed': errors_fixed,
        'correct_broken': correct_broken,
        'false_positives': false_positives,
        'net_benefit': errors_fixed - correct_broken
    }


def discover_safe_signature_for_transform(
    transform_name: str,
    transform_multipliers: Dict[str, float],
    base_sign_pattern: Dict[str, Optional[bool]],
    low_precision_errors: List[Dict],
    correct_predictions: List[Dict]
) -> Optional[SignatureConstraints]:
    """Discover safe signature for int4 low-precision errors."""

    logger.info(f"\nDiscovering safe signature for: {transform_name}")
    logger.info(f"  Base sign pattern: {base_sign_pattern}")

    # Find positions this transform successfully fixes (using int4 lens values only!)
    successful_fixes = []

    for err in low_precision_errors:
        lens_results = err['int4_lens']  # ONLY int4 values

        # Check if matches sign pattern
        matches_signs = True
        for lens, required_sign in base_sign_pattern.items():
            if required_sign is not None:
                actual_positive = lens_results[lens] > 0
                if actual_positive != required_sign:
                    matches_signs = False
                    break

        if not matches_signs:
            continue

        # Test if transform fixes it
        int4_thresholds = {'AT': 0.05, 'GC': 0.00, 'PuPy': 0.20, 'AmKe': 0.20, 'StWk': 0.20}
        ultra_low = {'AT': 0.01, 'GC': 0.01, 'PuPy': 0.01, 'AmKe': 0.01, 'StWk': 0.01}

        orig_votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for lens, sim in lens_results.items():
            voted_base = vote_for_base_with_threshold(lens, sim, int4_thresholds[lens])
            if voted_base:
                orig_votes[voted_base] += 1
        orig_pred = max(orig_votes, key=orig_votes.get)

        trans_lens = {lens: sim * transform_multipliers.get(lens, 1.0)
                      for lens, sim in lens_results.items() if lens in transform_multipliers}
        trans_votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for lens, sim in trans_lens.items():
            voted_base = vote_for_base_with_threshold(lens, sim, ultra_low.get(lens, 0.01))
            if voted_base:
                trans_votes[voted_base] += 1
        trans_pred = max(trans_votes, key=trans_votes.get)

        if trans_pred != orig_pred and trans_votes[trans_pred] >= 2 and trans_pred == err['ground_truth']:
            successful_fixes.append(err)

    if not successful_fixes:
        logger.info(f"  ✗ No successful fixes found with this sign pattern")
        return None

    logger.info(f"  Found {len(successful_fixes)} positions this transform can fix")

    # Analyze int4 lens magnitudes in successful fixes
    logger.info(f"  Analyzing int4 lens magnitudes in successful fixes:")
    for lens in ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']:
        mags = [abs(fix['int4_lens'][lens]) for fix in successful_fixes]
        logger.info(f"    {lens}: min={min(mags):.3f}, median={np.median(mags):.3f}, max={max(mags):.3f}")

    # Try progressively stricter magnitude thresholds
    best_signature = None
    best_fixes = 0

    # For int4, use much smaller thresholds (signals are ~4× smaller than float32)
    for amke_min in [0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01, None]:
        for stwk_min in [0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01, None]:
            for gc_min in [0.1, 0.05, 0.03, 0.02, 0.01, None]:
                signature = SignatureConstraints(
                    name=transform_name,
                    at_sign=base_sign_pattern.get('AT'),
                    gc_sign=base_sign_pattern.get('GC'),
                    pupy_sign=base_sign_pattern.get('PuPy'),
                    amke_sign=base_sign_pattern.get('AmKe'),
                    stwk_sign=base_sign_pattern.get('StWk'),
                    amke_min=amke_min,
                    stwk_min=stwk_min,
                    gc_min=gc_min,
                    transform_multipliers=transform_multipliers
                )

                result = test_signature(signature, low_precision_errors, correct_predictions)

                # Only accept if breaks nothing
                if result['correct_broken'] == 0:
                    if result['errors_fixed'] > best_fixes:
                        best_fixes = result['errors_fixed']
                        best_signature = signature
                        logger.info(
                            f"  ✓ Safe signature found: AmKe≥{amke_min if amke_min else 'any'}, "
                            f"StWk≥{stwk_min if stwk_min else 'any'}, GC≥{gc_min if gc_min else 'any'} | "
                            f"Fixes:{result['errors_fixed']} FP:{result['false_positives']} Breaks:{result['correct_broken']}"
                        )

    return best_signature


def main():
    logger.info("="*80)
    logger.info("DISCOVER SAFE CORRECTION SIGNATURES (INT4 - LOW PRECISION)")
    logger.info("="*80)
    logger.info("")
    logger.info("Conservative approach: Only accept signatures with 0 broken correct predictions")
    logger.info("Using ONLY int4 lens values (no float32 oracle)")
    logger.info("")

    # Load low-precision errors
    lp_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/low_precision_errors.json")
    with open(lp_file, 'r') as f:
        lp_data = json.load(f)
    low_precision_errors = lp_data['positions']

    # Load int4 correct predictions
    int4_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/comparison_results/int4_predictions_detailed.json")
    with open(int4_file, 'r') as f:
        all_int4 = json.load(f)
    correct_predictions = [p for p in all_int4 if p['correct']]

    logger.info(f"Loaded {len(low_precision_errors)} low-precision errors")
    logger.info(f"Loaded {len(correct_predictions)} correct predictions")

    # Define transformations to test
    transforms = [
        # Based on float32/int8 success
        {
            'name': 'drop_AT + flip_PuPy',
            'multipliers': {'GC': 1.0, 'PuPy': -1.0, 'AmKe': 1.0, 'StWk': 1.0},
            'sign_pattern': {'AT': None, 'GC': False, 'PuPy': True, 'AmKe': True, 'StWk': True}
        },
        {
            'name': 'flip_AmKe',
            'multipliers': {'AT': 1.0, 'GC': 1.0, 'PuPy': 1.0, 'AmKe': -1.0, 'StWk': 1.0},
            'sign_pattern': {'AT': None, 'GC': None, 'PuPy': None, 'AmKe': True, 'StWk': None}
        },
        {
            'name': 'flip_StWk',
            'multipliers': {'AT': 1.0, 'GC': 1.0, 'PuPy': 1.0, 'AmKe': 1.0, 'StWk': -1.0},
            'sign_pattern': {'AT': None, 'GC': None, 'PuPy': None, 'AmKe': None, 'StWk': True}
        },
        {
            'name': 'flip_GC',
            'multipliers': {'AT': 1.0, 'GC': -1.0, 'PuPy': 1.0, 'AmKe': 1.0, 'StWk': 1.0},
            'sign_pattern': {'AT': None, 'GC': True, 'PuPy': None, 'AmKe': None, 'StWk': None}
        },
        {
            'name': 'flip_PuPy',
            'multipliers': {'AT': 1.0, 'GC': 1.0, 'PuPy': -1.0, 'AmKe': 1.0, 'StWk': 1.0},
            'sign_pattern': {'AT': None, 'GC': None, 'PuPy': True, 'AmKe': None, 'StWk': None}
        },
        # Boosting strategies (multiply by factor > 1)
        {
            'name': 'boost_GC_2x',
            'multipliers': {'AT': 1.0, 'GC': 2.0, 'PuPy': 1.0, 'AmKe': 1.0, 'StWk': 1.0},
            'sign_pattern': {'AT': None, 'GC': False, 'PuPy': False, 'AmKe': None, 'StWk': None}
        },
        {
            'name': 'boost_PuPy_2x',
            'multipliers': {'AT': 1.0, 'GC': 1.0, 'PuPy': 2.0, 'AmKe': 1.0, 'StWk': 1.0},
            'sign_pattern': {'AT': None, 'GC': None, 'PuPy': False, 'AmKe': None, 'StWk': None}
        },
    ]

    # Discover safe signatures
    safe_signatures = []

    for transform in transforms:
        signature = discover_safe_signature_for_transform(
            transform['name'],
            transform['multipliers'],
            transform['sign_pattern'],
            low_precision_errors,
            correct_predictions
        )

        if signature:
            safe_signatures.append(signature)

    # Report results
    logger.info("")
    logger.info("="*80)
    logger.info("SAFE SIGNATURES DISCOVERED (INT4)")
    logger.info("="*80)

    total_fixes = 0
    for sig in safe_signatures:
        result = test_signature(sig, low_precision_errors, correct_predictions)
        total_fixes += result['errors_fixed']
        logger.info(f"\n{sig.name}:")
        logger.info(f"  Sign pattern: GC={sig.gc_sign}, PuPy={sig.pupy_sign}, AmKe={sig.amke_sign}, StWk={sig.stwk_sign}")
        logger.info(f"  Magnitude constraints: GC≥{sig.gc_min}, AmKe≥{sig.amke_min}, StWk≥{sig.stwk_min}")
        logger.info(f"  Errors fixed: {result['errors_fixed']}/37")
        logger.info(f"  Correct broken: {result['correct_broken']}/1000")
        logger.info(f"  False positives: {result['false_positives']}/1000")

    logger.info(f"\nTOTAL ERRORS FIXED (if all signatures applied): {total_fixes}/37")
    logger.info(f"Accuracy improvement: {total_fixes}/9555 = +{100*total_fixes/9555:.3f}%")


if __name__ == '__main__':
    main()
