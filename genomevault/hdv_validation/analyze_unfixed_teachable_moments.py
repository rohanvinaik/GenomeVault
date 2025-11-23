#!/usr/bin/env python3
"""
Analyze Unfixed Teachable Moments

Identify which teachable moments are being fixed by which signatures,
and analyze the biophysical patterns of the remaining unfixed positions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set
from discover_safe_signatures import (
    SignatureConstraints,
    test_signature,
    vote_for_base_with_threshold
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_position_key(tm: Dict) -> str:
    """Create unique key for a teachable moment position."""
    return tm['position']


def identify_fixed_positions(
    teachable_moments: List[Dict],
    signatures: List[SignatureConstraints]
) -> Dict[str, List[str]]:
    """Identify which signatures fix which positions."""

    float32_thresholds = {'AT': 0.05, 'GC': 0.00, 'PuPy': 0.20, 'AmKe': 0.20, 'StWk': 0.20}
    ultra_low = {'AT': 0.01, 'GC': 0.01, 'PuPy': 0.01, 'AmKe': 0.01, 'StWk': 0.01}

    fixed_by_signature = {sig.name: [] for sig in signatures}

    for tm in teachable_moments:
        pos_key = get_position_key(tm)
        lens_results = tm['lens_results']['float32']
        ground_truth = tm['ground_truth']
        confidence = tm['confidences']['float32']

        if confidence > 0.65:
            continue

        # Original prediction
        orig_votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for lens, sim in lens_results.items():
            voted_base = vote_for_base_with_threshold(lens, sim, float32_thresholds[lens])
            if voted_base:
                orig_votes[voted_base] += 1
        orig_pred = max(orig_votes, key=orig_votes.get)

        # Test each signature
        for sig in signatures:
            if not sig.matches(lens_results):
                continue

            # Apply transformation
            trans_lens = {}
            for lens, sim in lens_results.items():
                if lens in sig.transform_multipliers:
                    trans_lens[lens] = sim * sig.transform_multipliers[lens]

            # Vote
            trans_votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
            for lens, sim in trans_lens.items():
                voted_base = vote_for_base_with_threshold(lens, sim, ultra_low.get(lens, 0.01))
                if voted_base:
                    trans_votes[voted_base] += 1

            trans_pred = max(trans_votes, key=trans_votes.get)

            # Check if fixes
            if trans_pred != orig_pred and trans_votes[trans_pred] >= 2 and trans_pred == ground_truth:
                fixed_by_signature[sig.name].append(pos_key)

    return fixed_by_signature


def analyze_unfixed_patterns(
    teachable_moments: List[Dict],
    fixed_positions: Set[str]
) -> None:
    """Analyze biophysical patterns in unfixed teachable moments."""

    float32_thresholds = {'AT': 0.05, 'GC': 0.00, 'PuPy': 0.20, 'AmKe': 0.20, 'StWk': 0.20}

    unfixed = [tm for tm in teachable_moments if get_position_key(tm) not in fixed_positions]

    logger.info(f"\n{'='*80}")
    logger.info(f"UNFIXED TEACHABLE MOMENTS: {len(unfixed)}/{len(teachable_moments)}")
    logger.info(f"{'='*80}\n")

    for i, tm in enumerate(unfixed, 1):
        pos_key = get_position_key(tm)
        lens_results = tm['lens_results']['float32']
        ground_truth = tm['ground_truth']

        # Get original prediction
        orig_votes = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for lens, sim in lens_results.items():
            voted_base = vote_for_base_with_threshold(lens, sim, float32_thresholds[lens])
            if voted_base:
                orig_votes[voted_base] += 1
        orig_pred = max(orig_votes, key=orig_votes.get)

        logger.info(f"{i}. {pos_key}")
        logger.info(f"   Ground truth: {ground_truth} | Predicted: {orig_pred}")
        logger.info(f"   Lens values:")
        logger.info(f"     AT:   {lens_results['AT']:+.3f} ({'>' if lens_results['AT'] > 0 else '<'}0)")
        logger.info(f"     GC:   {lens_results['GC']:+.3f} ({'>' if lens_results['GC'] > 0 else '<'}0)")
        logger.info(f"     PuPy: {lens_results['PuPy']:+.3f} ({'>' if lens_results['PuPy'] > 0 else '<'}0)")
        logger.info(f"     AmKe: {lens_results['AmKe']:+.3f} ({'>' if lens_results['AmKe'] > 0 else '<'}0)")
        logger.info(f"     StWk: {lens_results['StWk']:+.3f} ({'>' if lens_results['StWk'] > 0 else '<'}0)")
        logger.info(f"   Magnitudes: AT={abs(lens_results['AT']):.3f}, GC={abs(lens_results['GC']):.3f}, "
                   f"PuPy={abs(lens_results['PuPy']):.3f}, AmKe={abs(lens_results['AmKe']):.3f}, "
                   f"StWk={abs(lens_results['StWk']):.3f}")

        # Suggest potential transformations
        logger.info(f"   Votes: {orig_votes}")
        logger.info("")


def main():
    # Load data
    tm_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/teachable_moments.json")
    with open(tm_file, 'r') as f:
        tm_data = json.load(f)
    teachable_moments = tm_data['positions']

    # Define discovered safe signatures
    signatures = [
        SignatureConstraints(
            name='drop_AT + flip_PuPy',
            gc_sign=False,
            pupy_sign=True,
            amke_sign=True,
            stwk_sign=True,
            gc_min=0.2,
            amke_min=0.8,
            stwk_min=0.4,
            transform_multipliers={'GC': 1.0, 'PuPy': -1.0, 'AmKe': 1.0, 'StWk': 1.0}
        ),
        SignatureConstraints(
            name='flip_AmKe',
            amke_sign=True,
            gc_min=0.3,
            amke_min=0.2,
            stwk_min=0.8,
            transform_multipliers={'AT': 1.0, 'GC': 1.0, 'PuPy': 1.0, 'AmKe': -1.0, 'StWk': 1.0}
        ),
        SignatureConstraints(
            name='flip_PuPy',
            pupy_sign=True,
            gc_min=0.3,
            amke_min=0.8,
            stwk_min=0.3,
            transform_multipliers={'AT': 1.0, 'GC': 1.0, 'PuPy': -1.0, 'AmKe': 1.0, 'StWk': 1.0}
        ),
    ]

    # Identify which positions are fixed
    logger.info(f"{'='*80}")
    logger.info(f"SIGNATURE COVERAGE ANALYSIS")
    logger.info(f"{'='*80}\n")

    fixed_by_signature = identify_fixed_positions(teachable_moments, signatures)

    all_fixed = set()
    for sig_name, positions in fixed_by_signature.items():
        logger.info(f"{sig_name}:")
        for pos in positions:
            logger.info(f"  - {pos}")
        logger.info(f"  Total: {len(positions)}\n")
        all_fixed.update(positions)

    logger.info(f"Total unique positions fixed: {len(all_fixed)}/22")

    # Analyze unfixed
    analyze_unfixed_patterns(teachable_moments, all_fixed)


if __name__ == '__main__':
    main()
