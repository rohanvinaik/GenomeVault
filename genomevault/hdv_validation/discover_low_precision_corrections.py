#!/usr/bin/env python3
"""
Discover Corrections for Low-Precision Errors (int4)

Low-precision errors: float32 CORRECT, int4 WRONG
Expected patterns:
- C→T transitions (deamination signature)
- Purine stability (A↔G, T→A preference)
- Simpler fixes than high-precision errors

Strategy:
1. Load positions where float32 correct, int4 wrong
2. Analyze biophysical signatures
3. Discover simple correction rules
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
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


def analyze_low_precision_errors():
    """Analyze positions where float32 correct but int4 wrong."""

    logger.info("="*80)
    logger.info("LOW-PRECISION ERROR ANALYSIS (float32 CORRECT, int4 WRONG)")
    logger.info("="*80)
    logger.info("")

    # Load validation data
    float32_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/comparison_results/float32_predictions_detailed.json")
    int4_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/comparison_results/int4_predictions_detailed.json")

    with open(float32_file, 'r') as f:
        float32_preds = json.load(f)
    with open(int4_file, 'r') as f:
        int4_preds = json.load(f)

    # Create position lookup
    float32_by_pos = {p['position']: p for p in float32_preds}
    int4_by_pos = {p['position']: p for p in int4_preds}

    # Find low-precision errors
    low_precision_errors = []

    for pos, f32_pred in float32_by_pos.items():
        if pos not in int4_by_pos:
            continue

        i4_pred = int4_by_pos[pos]

        # float32 correct, int4 wrong
        if f32_pred['correct'] and not i4_pred['correct']:
            low_precision_errors.append({
                'position': pos,
                'ground_truth': f32_pred['ground_truth'],
                'float32_pred': f32_pred['predicted'],
                'int4_pred': i4_pred['predicted'],
                'float32_lens': f32_pred['lens_results'],
                'int4_lens': i4_pred['lens_results'],
                'float32_conf': f32_pred['confidence'],
                'int4_conf': i4_pred['confidence']
            })

    logger.info(f"Found {len(low_precision_errors)} low-precision errors")
    logger.info("")

    # Analyze transition patterns
    logger.info("TRANSITION PATTERNS:")
    logger.info("")

    transitions = Counter()
    for err in low_precision_errors:
        key = f"{err['ground_truth']}→{err['int4_pred']}"
        transitions[key] += 1

    # Group by ground truth
    for base in ['A', 'T', 'G', 'C']:
        base_errors = [k for k in transitions.keys() if k.startswith(f"{base}→")]
        if base_errors:
            total = sum(transitions[k] for k in base_errors)
            logger.info(f"{base} errors (n={total}):")
            for trans in sorted(base_errors, key=lambda x: transitions[x], reverse=True):
                count = transitions[trans]
                pct = 100 * count / total
                logger.info(f"  {trans}: {count}/{total} ({pct:.1f}%)")
            logger.info("")

    # Analyze biophysical signatures
    logger.info("="*80)
    logger.info("BIOPHYSICAL SIGNATURE ANALYSIS")
    logger.info("="*80)
    logger.info("")

    # Group by transition type
    c_to_t = [e for e in low_precision_errors if e['ground_truth'] == 'C' and e['int4_pred'] == 'T']
    g_to_t = [e for e in low_precision_errors if e['ground_truth'] == 'G' and e['int4_pred'] == 'T']
    a_to_t = [e for e in low_precision_errors if e['ground_truth'] == 'A' and e['int4_pred'] == 'T']

    logger.info(f"C→T transitions (deamination signature): {len(c_to_t)} positions")
    if c_to_t:
        analyze_transition_signature("C→T", c_to_t)

    logger.info(f"\nG→T transitions: {len(g_to_t)} positions")
    if g_to_t:
        analyze_transition_signature("G→T", g_to_t)

    logger.info(f"\nA→T transitions: {len(a_to_t)} positions")
    if a_to_t:
        analyze_transition_signature("A→T", a_to_t)

    # Save low-precision errors for correction discovery
    output_file = Path("HDV_VALIDATION_PACKAGE/architecture_testing/aligned_10k/low_precision_errors.json")
    with open(output_file, 'w') as f:
        json.dump({
            'total_errors': len(low_precision_errors),
            'transition_counts': dict(transitions),
            'positions': low_precision_errors
        }, f, indent=2)

    logger.info(f"\n✓ Low-precision errors saved to: {output_file}")

    return low_precision_errors


def analyze_transition_signature(transition_name: str, errors: List[Dict]):
    """Analyze lens signatures for a specific transition type."""

    logger.info(f"\n{transition_name} Biophysical Signature:")
    logger.info("-" * 60)

    # Analyze float32 lens values (what int4 is missing)
    logger.info("\nfloat32 lens characteristics (correct prediction):")
    for lens in ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']:
        values = [e['float32_lens'][lens] for e in errors]
        signs = ['+' if v > 0 else '-' for v in values]
        sign_counter = Counter(signs)

        logger.info(f"  {lens}:")
        logger.info(f"    Range: [{min(values):.3f}, {max(values):.3f}]")
        logger.info(f"    Mean: {np.mean(values):.3f}, Median: {np.median(values):.3f}")
        logger.info(f"    Signs: {dict(sign_counter)} ({100*sign_counter.get('+',0)/len(values):.1f}% positive)")

    # Analyze int4 lens values (incorrect prediction)
    logger.info("\nint4 lens characteristics (incorrect prediction):")
    for lens in ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']:
        values = [e['int4_lens'][lens] for e in errors]
        signs = ['+' if v > 0 else '-' for v in values]
        sign_counter = Counter(signs)

        logger.info(f"  {lens}:")
        logger.info(f"    Range: [{min(values):.3f}, {max(values):.3f}]")
        logger.info(f"    Mean: {np.mean(values):.3f}, Median: {np.median(values):.3f}")
        logger.info(f"    Signs: {dict(sign_counter)} ({100*sign_counter.get('+',0)/len(values):.1f}% positive)")

    # Compare: what's different?
    logger.info("\nKey differences (float32 - int4):")
    for lens in ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']:
        f32_vals = [e['float32_lens'][lens] for e in errors]
        i4_vals = [e['int4_lens'][lens] for e in errors]
        diff_vals = [f32 - i4 for f32, i4 in zip(f32_vals, i4_vals)]

        logger.info(f"  {lens} difference:")
        logger.info(f"    Mean: {np.mean(diff_vals):.3f}, Median: {np.median(diff_vals):.3f}")
        logger.info(f"    Magnitude: {np.mean([abs(d) for d in diff_vals]):.3f}")


def discover_simple_boosting_rules(low_precision_errors: List[Dict]):
    """
    Discover simple boosting rules for int4.

    Strategy: Boost specific lenses that float32 uses but int4 misses.
    """

    logger.info("\n" + "="*80)
    logger.info("SIMPLE BOOSTING RULE DISCOVERY")
    logger.info("="*80)
    logger.info("")
    logger.info("Strategy: Boost lens signals that float32 uses to make correct predictions")
    logger.info("")

    # For C→T errors, what lens does float32 use to get C correct?
    c_to_t = [e for e in low_precision_errors if e['ground_truth'] == 'C' and e['int4_pred'] == 'T']

    if c_to_t:
        logger.info(f"Analyzing C→T transitions ({len(c_to_t)} positions):")
        logger.info("")

        # Check which lens votes for C in float32
        int4_thresholds = {'AT': 0.05, 'GC': 0.00, 'PuPy': 0.20, 'AmKe': 0.20, 'StWk': 0.20}

        lens_votes_for_C = {lens: 0 for lens in ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']}
        lens_magnitudes = {lens: [] for lens in ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']}

        for err in c_to_t:
            for lens in ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']:
                # Does this lens vote for C?
                voted = vote_for_base_with_threshold(lens, err['float32_lens'][lens], int4_thresholds[lens])
                if voted == 'C':
                    lens_votes_for_C[lens] += 1
                    lens_magnitudes[lens].append(abs(err['float32_lens'][lens]))

        logger.info("Lens voting patterns (float32, for ground truth C):")
        for lens in sorted(lens_votes_for_C.keys(), key=lambda x: lens_votes_for_C[x], reverse=True):
            count = lens_votes_for_C[lens]
            pct = 100 * count / len(c_to_t)
            if count > 0:
                avg_mag = np.mean(lens_magnitudes[lens])
                logger.info(f"  {lens}: {count}/{len(c_to_t)} ({pct:.1f}%) | Avg magnitude: {avg_mag:.3f}")

        logger.info("")
        logger.info("Proposed boosting rule for C→T corrections:")
        logger.info("  Boost GC lens by 1.5× when:")
        logger.info("    - GC < 0 (votes for C)")
        logger.info("    - |GC| > 0.1 in int4")
        logger.info("    - PuPy < 0 (agrees with C)")


def main():
    # Analyze low-precision errors
    low_precision_errors = analyze_low_precision_errors()

    # Discover simple boosting rules
    if low_precision_errors:
        discover_simple_boosting_rules(low_precision_errors)


if __name__ == '__main__':
    main()
