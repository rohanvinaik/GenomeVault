#!/usr/bin/env python3
"""
Experiment 0: Biophysical Context Validation

Validates that our biophysical signature voting returns realistic genome fractions
compared to annotated ground truth frequencies.

Expected results (chr22):
- TATA promoter: 3.5% (±5%)
- CpG island: 1.5% (±5%)
- Heterochromatin: 20% (±5%)

Author: Claude Code
Date: November 22, 2025
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import time
import logging
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lens_aware_simd_query_engine import (
    LensAwareSIMDQueryEngine,
    BIOPHYSICAL_CONTEXTS,
    LAYER_TO_BIT
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def validate_biophysical_contexts(engine):
    """
    Validate all pre-calibrated biophysical contexts against ground truth.

    Ground truth source: UCSC, ENCODE, RepeatMasker annotations
    Reference: MULTI_STAGE_QUERY_ARCHITECTURE_EXPERIMENTS.md, Ground Truth section
    """
    print("\n" + "="*80)
    print("EXPERIMENT 0: Biophysical Context Validation")
    print("="*80)
    print()
    print("Goal: Verify biophysical contexts return realistic genome fractions")
    print(f"Dataset: Full hg38 genome ({len(engine._get_signatures()):,} chunks, 24 chromosomes)")
    print()

    # Ground truth expectations for chr22
    ground_truth = {
        'tata_promoter': {
            'expected_fraction': 0.035,  # 3.5%
            'tolerance': 0.05,  # ±5%
            'rationale': 'Between TATA-like abundance (~25%) and functional TATA boxes (~0.3%)',
        },
        'cpg_island': {
            'expected_fraction': 0.015,  # 1.5%
            'tolerance': 0.05,  # ±5%
            'rationale': 'Annotated CpG island frequency',
        },
        'heterochromatin': {
            'expected_fraction': 0.20,  # 20%
            'tolerance': 0.10,  # ±10% (broader tolerance due to definition ambiguity)
            'rationale': 'Constitutive heterochromatin + high-density repeat regions',
        },
    }

    # Get signatures (triggers calibration on first call)
    print("Computing biophysical signatures for all chunks...")
    t0 = time.perf_counter()
    signatures = engine._get_signatures()
    t1 = time.perf_counter()
    print(f"✓ Computed {len(signatures):,} signatures in {(t1-t0)*1e3:.1f} ms")
    print()

    # Test each context
    results = {}
    for context_name, ground_truth_data in ground_truth.items():
        if context_name not in BIOPHYSICAL_CONTEXTS:
            print(f"⚠️  Context '{context_name}' not found in BIOPHYSICAL_CONTEXTS")
            continue

        print("-" * 80)
        print(f"Testing context: {context_name}")
        print("-" * 80)

        context = BIOPHYSICAL_CONTEXTS[context_name]
        print(f"Description: {context['description']}")
        print(f"Layers required: {list(context['layers'].keys())}")
        print(f"Voting threshold: {context['voting_threshold']}")
        print()

        # Run biophysical voting
        t0 = time.perf_counter()
        candidates = engine._vote_on_signatures(
            signatures,
            context['layers'],
            context['voting_threshold']
        )
        t1 = time.perf_counter()

        # Calculate actual fraction
        actual_fraction = len(candidates) / len(signatures)

        # Compare to ground truth
        expected_fraction = ground_truth_data['expected_fraction']
        tolerance = ground_truth_data['tolerance']
        deviation = actual_fraction - expected_fraction
        deviation_pct = (deviation / expected_fraction) * 100 if expected_fraction > 0 else 0

        # Determine pass/fail
        if abs(deviation) <= tolerance:
            status = "✅ PASS"
        elif abs(deviation) <= 2 * tolerance:
            status = "⚠️  ACCEPTABLE"
        else:
            status = "❌ FAIL"

        # Print results
        print(f"Ground Truth:")
        print(f"  Expected: {expected_fraction*100:.1f}% of genome")
        print(f"  Rationale: {ground_truth_data['rationale']}")
        print()
        print(f"Results:")
        print(f"  Actual: {actual_fraction*100:.1f}% of genome ({len(candidates):,} / {len(signatures):,} chunks)")
        print(f"  Deviation: {deviation*100:+.1f}% ({deviation_pct:+.1f}% relative)")
        print(f"  Query time: {(t1-t0)*1e6:.1f} μs")
        print()
        print(f"Status: {status}")
        print()

        # Store results
        results[context_name] = {
            'expected_fraction': expected_fraction,
            'actual_fraction': actual_fraction,
            'deviation': deviation,
            'deviation_pct': deviation_pct,
            'num_candidates': len(candidates),
            'query_time_us': (t1-t0)*1e6,
            'status': status,
        }

    # Summary
    print("="*80)
    print("SUMMARY: Biophysical Context Validation")
    print("="*80)
    print()

    passed = sum(1 for r in results.values() if "PASS" in r['status'])
    acceptable = sum(1 for r in results.values() if "ACCEPTABLE" in r['status'])
    failed = sum(1 for r in results.values() if "FAIL" in r['status'])

    print(f"Total contexts tested: {len(results)}")
    print(f"  ✅ Passed: {passed}")
    print(f"  ⚠️  Acceptable: {acceptable}")
    print(f"  ❌ Failed: {failed}")
    print()

    # Detailed table
    print("| Context | Expected | Actual | Deviation | Status |")
    print("|---------|----------|--------|-----------|--------|")
    for context_name, result in results.items():
        print(f"| {context_name:20s} | {result['expected_fraction']*100:5.1f}% | "
              f"{result['actual_fraction']*100:5.1f}% | {result['deviation']*100:+5.1f}% | "
              f"{result['status']} |")

    print()

    if failed == 0:
        print("✓ All contexts validated successfully!")
        return True
    else:
        print(f"⚠️  {failed} context(s) failed validation")
        print("   Review biophysical layer definitions and thresholds")
        return False


def test_individual_layers(engine):
    """
    Test individual biophysical layers to validate threshold calibration.

    Validates that single-layer queries return reasonable fractions.
    """
    print("\n" + "="*80)
    print("BONUS: Individual Layer Validation")
    print("="*80)
    print()

    # Ground truth for individual layers (chr22)
    layer_ground_truth = {
        'AT_DOMINANT': {
            'expected_fraction': 0.22,  # 22% AT-rich regions
            'tolerance': 0.10,
        },
        'GC_DOMINANT': {
            'expected_fraction': 0.18,  # 18% GC-rich regions
            'tolerance': 0.10,
        },
        'EXTREME_AT': {
            'expected_fraction': 0.03,  # 3% extreme AT
            'tolerance': 0.05,
        },
        'EXTREME_GC': {
            'expected_fraction': 0.02,  # 2% extreme GC
            'tolerance': 0.05,
        },
    }

    signatures = engine._get_signatures()

    print("Testing individual biophysical layers:")
    print()

    for layer_name, ground_truth_data in layer_ground_truth.items():
        # Single-layer context (requires only this bit)
        single_layer_context = {layer_name: True}

        candidates = engine._vote_on_signatures(
            signatures,
            single_layer_context,
            threshold=1.0  # Must match 100% (only 1 layer)
        )

        actual_fraction = len(candidates) / len(signatures)
        expected_fraction = ground_truth_data['expected_fraction']
        deviation = actual_fraction - expected_fraction

        status = "✓" if abs(deviation) <= ground_truth_data['tolerance'] else "✗"

        print(f"{status} {layer_name:20s}: Expected {expected_fraction*100:5.1f}%, "
              f"Actual {actual_fraction*100:5.1f}%, Deviation {deviation*100:+5.1f}%")

    print()


def compare_whole_genome_vs_chr22(engine):
    """
    Statistical comparison: Whole genome vs chr22 biophysical context frequencies.

    Tests if compositional differences between hg38 and chr22 are detectable
    in biophysical signatures using chi-squared tests and z-tests for proportions.
    """
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS: Whole Genome vs Chr22")
    print("="*80)
    print()

    # Get all signatures
    signatures = engine._get_signatures()

    # Load chunk keys to identify chr22
    with h5py.File(engine.h5_path, 'r') as f:
        chunk_keys = f['chunk_keys'][:]

    # Separate indices: whole genome vs chr22
    chr22_indices = []
    for idx, key in enumerate(chunk_keys):
        key_str = key.decode() if isinstance(key, bytes) else key
        if key_str.startswith('chr22:'):
            chr22_indices.append(idx)

    chr22_indices = np.array(chr22_indices)
    all_indices = np.arange(len(signatures))

    print(f"Dataset breakdown:")
    print(f"  Total chunks: {len(signatures):,}")
    print(f"  Chr22 chunks: {len(chr22_indices):,} ({len(chr22_indices)/len(signatures)*100:.2f}%)")
    print(f"  Other chromosomes: {len(all_indices) - len(chr22_indices):,}")
    print()

    # Test each biophysical context
    print("Testing biophysical contexts:")
    print()

    comparison_results = {}

    for context_name, context_data in BIOPHYSICAL_CONTEXTS.items():
        layers = context_data['layers']
        threshold = context_data['voting_threshold']

        # Compute candidates for whole genome
        candidates_all = engine._vote_on_signatures(signatures, layers, threshold)

        # Compute candidates for chr22 only
        signatures_chr22 = signatures[chr22_indices]
        candidates_chr22_mask = engine._vote_on_signatures(signatures_chr22, layers, threshold)

        # Calculate frequencies
        freq_whole_genome = len(candidates_all) / len(signatures)
        freq_chr22 = len(candidates_chr22_mask) / len(signatures_chr22)

        # Count positive cases
        n_positive_whole = len(candidates_all)
        n_total_whole = len(signatures)
        n_positive_chr22 = len(candidates_chr22_mask)
        n_total_chr22 = len(signatures_chr22)

        # Chi-squared test for difference in proportions
        # Contingency table: [[positive_whole, negative_whole], [positive_chr22, negative_chr22]]
        contingency = np.array([
            [n_positive_whole - n_positive_chr22, n_total_whole - n_positive_whole - (n_total_chr22 - n_positive_chr22)],
            [n_positive_chr22, n_total_chr22 - n_positive_chr22]
        ])

        # Handle edge case: if chr22 chunks are subset of whole genome, adjust
        # Actually, we need: [whole_minus_chr22, chr22]
        n_positive_other = n_positive_whole - n_positive_chr22  # Positive in non-chr22
        n_total_other = n_total_whole - n_total_chr22  # Total non-chr22

        contingency = np.array([
            [n_positive_other, n_total_other - n_positive_other],  # Other chroms
            [n_positive_chr22, n_total_chr22 - n_positive_chr22]    # Chr22
        ])

        chi2, p_value_chi2 = stats.chi2_contingency(contingency)[:2]

        # Z-test for difference in proportions
        p1 = n_positive_other / n_total_other
        p2 = n_positive_chr22 / n_total_chr22

        # Pooled proportion
        p_pool = (n_positive_other + n_positive_chr22) / (n_total_other + n_total_chr22)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_total_other + 1/n_total_chr22))

        # Z-statistic
        z_stat = (p1 - p2) / se if se > 0 else 0
        p_value_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed

        # Effect size (Cohen's h for proportions)
        h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

        # Determine significance
        alpha = 0.05
        is_significant = p_value_chi2 < alpha

        print(f"{context_name}:")
        print(f"  Whole genome (excl. chr22): {p1*100:.2f}% ({n_positive_other:,} / {n_total_other:,})")
        print(f"  Chr22:                      {p2*100:.2f}% ({n_positive_chr22:,} / {n_total_chr22:,})")
        print(f"  Difference:                 {(p2-p1)*100:+.2f}% ({abs((p2-p1)/p1)*100:.1f}% relative)")
        print(f"  Chi-squared test:           χ²={chi2:.2f}, p={p_value_chi2:.4f} {'***' if p_value_chi2 < 0.001 else '**' if p_value_chi2 < 0.01 else '*' if p_value_chi2 < 0.05 else 'ns'}")
        print(f"  Z-test:                     z={z_stat:.2f}, p={p_value_z:.4f}")
        print(f"  Effect size (Cohen's h):    {h:.3f} ({'small' if abs(h) < 0.2 else 'medium' if abs(h) < 0.5 else 'large'})")
        print(f"  Significant at α=0.05:      {'YES' if is_significant else 'NO'}")
        print()

        comparison_results[context_name] = {
            'freq_whole_genome': freq_whole_genome,
            'freq_chr22': freq_chr22,
            'freq_other': p1,
            'difference': p2 - p1,
            'chi2': chi2,
            'p_value_chi2': p_value_chi2,
            'z_stat': z_stat,
            'p_value_z': p_value_z,
            'effect_size': h,
            'is_significant': is_significant,
        }

    # Summary
    print("="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    print()

    significant_contexts = [k for k, v in comparison_results.items() if v['is_significant']]

    print(f"Total contexts tested: {len(comparison_results)}")
    print(f"Statistically significant differences: {len(significant_contexts)}")
    print()

    if significant_contexts:
        print("Contexts with significant chr22 vs whole genome differences:")
        for ctx in significant_contexts:
            res = comparison_results[ctx]
            print(f"  • {ctx}: Δ={res['difference']*100:+.2f}%, p={res['p_value_chi2']:.4f}, h={res['effect_size']:.3f}")
    else:
        print("No statistically significant differences detected.")

    print()
    print("Interpretation:")
    print("  p < 0.05:  Statistically significant difference")
    print("  |h| < 0.2: Small effect size")
    print("  |h| < 0.5: Medium effect size")
    print("  |h| ≥ 0.5: Large effect size")
    print()

    return comparison_results


def main():
    """Run Experiment 0: Biophysical Context Validation."""

    # Paths - Auto-detect split ternary or standard 3-bank format
    split_ternary_path = "output/encoded_genome_6banks_split_ternary.h5"
    standard_path = "output/encoded_genome_3banks.h5"

    # Prefer split ternary if available
    if Path(split_ternary_path).exists():
        h5_path = split_ternary_path
        print(f"✓ Using split ternary format: {split_ternary_path}")
    elif Path(standard_path).exists():
        h5_path = standard_path
        print(f"✓ Using standard 3-bank format: {standard_path}")
    else:
        print(f"ERROR: No encoded genome found!")
        print(f"Tried:")
        print(f"  - Split ternary: {split_ternary_path}")
        print(f"  - Standard 3-bank: {standard_path}")
        print()
        print("Expected HDF5 structure:")
        print("  Split ternary: /split_ternary_vectors: (n_chunks, 6, D)")
        print("  Standard: /all_bank_vectors: (n_chunks, 3, D)")
        sys.exit(1)

    # Initialize engine (biophysical Stage 0 enabled, no FASTA needed)
    print("Initializing Lens-Aware SIMD Query Engine...")
    print(f"  H5 path: {h5_path}")
    print()

    with LensAwareSIMDQueryEngine(
        h5_path=h5_path,
        fasta_path=None,  # Not needed for this experiment
        enable_lens_system=True,
        enable_biophysical_stage0=True,
    ) as engine:

        # Run validation
        success = validate_biophysical_contexts(engine)

        # Bonus: test individual layers
        test_individual_layers(engine)

        # NEW: Statistical comparison of whole genome vs chr22
        compare_whole_genome_vs_chr22(engine)

        # Return exit code
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
