"""
Bank3 Variance as Genomic Complexity Metric

Hypothesis: Bank3 variance (σ) can distinguish between:
  - Low variance (σ < 1.0) = Structurally constrained regions (functional elements)
  - High variance (σ > 3.0) = Compositional noise (low-complexity, repetitive)

Complexity Levels:
1. HIGH COMPLEXITY (σ < 1.0): CpG islands, promoters, regulatory motifs
   - Tight dinucleotide transition patterns
   - Functional constraint → structural consistency

2. MEDIUM COMPLEXITY (1.0 < σ < 3.0): Normal coding/non-coding regions
   - Moderate structural variation
   - Balanced composition

3. LOW COMPLEXITY (σ > 3.0): Repetitive elements, homopolymers
   - Wide variance due to composition extremes
   - Includes zero-signal regions (assembly gaps, low coverage)

Author: Phase 1 Week 3 - Complexity Metric Discovery
Date: November 22, 2025
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, mannwhitneyu
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_bank3_complexity(data_file: str, output_file: str):
    """
    Analyze Bank3 variance as a complexity metric across composition bins.
    """
    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)

    logger.info("\n" + "="*80)
    logger.info("BANK3 VARIANCE AS COMPLEXITY METRIC")
    logger.info("="*80)

    results = {
        'complexity_levels': {},
        'pathway_analysis': {},
        'variance_vs_composition': {},
        'outlier_detection': {},
    }

    # Collect all chunks with Bank3 variance
    all_chunks_with_variance = []

    for pathway_name in ['AT_pathway', 'GC_pathway']:
        for bin_name, bin_data in data[pathway_name].items():
            # Compute Bank3 variance for this bin
            bank3_pos = np.array([c['signals']['bank3_pos_mag'] for c in bin_data])
            bank3_neg = np.array([c['signals']['bank3_neg_mag'] for c in bin_data])

            mask = (bank3_pos > 0) & (bank3_neg > 0)

            if mask.sum() > 0:
                variance_pos = np.std(bank3_pos[mask])
                variance_neg = np.std(bank3_neg[mask])
                median_pos = np.median(bank3_pos[mask])
                median_neg = np.median(bank3_neg[mask])

                # Store for global analysis
                for c in bin_data:
                    if c['signals']['bank3_pos_mag'] > 0:
                        all_chunks_with_variance.append({
                            'pathway': pathway_name,
                            'bin': bin_name,
                            'bank3_pos': c['signals']['bank3_pos_mag'],
                            'bank3_neg': c['signals']['bank3_neg_mag'],
                            'composition': c['composition'],
                        })

    # 1. Define complexity levels based on variance
    logger.info("\n=== COMPLEXITY LEVEL CLASSIFICATION ===")

    # Collect variance per bin
    bin_variances = []
    for pathway_name in ['AT_pathway', 'GC_pathway']:
        for bin_name in sorted(data[pathway_name].keys()):
            bin_data = data[pathway_name][bin_name]

            bank3_pos = np.array([c['signals']['bank3_pos_mag'] for c in bin_data])
            bank3_neg = np.array([c['signals']['bank3_neg_mag'] for c in bin_data])
            mask = (bank3_pos > 0) & (bank3_neg > 0)

            if mask.sum() > 10:
                var_pos = np.std(bank3_pos[mask])
                var_neg = np.std(bank3_neg[mask])
                median_pos = np.median(bank3_pos[mask])
                median_neg = np.median(bank3_neg[mask])

                # Compute AT% and GC%
                at_pct = np.mean([(c['composition']['A_pct'] + c['composition']['T_pct'])
                                   for c in bin_data])
                gc_pct = np.mean([(c['composition']['G_pct'] + c['composition']['C_pct'])
                                   for c in bin_data])

                bin_variances.append({
                    'pathway': pathway_name,
                    'bin': bin_name,
                    'var_pos': var_pos,
                    'var_neg': var_neg,
                    'median_pos': median_pos,
                    'median_neg': median_neg,
                    'at_pct': at_pct,
                    'gc_pct': gc_pct,
                    'n': int(mask.sum()),
                })

                # Classify complexity
                avg_var = (var_pos + var_neg) / 2
                if avg_var < 1.0:
                    complexity = 'HIGH (σ < 1.0)'
                elif avg_var < 3.0:
                    complexity = 'MEDIUM (1.0 < σ < 3.0)'
                else:
                    complexity = 'LOW (σ > 3.0)'

                logger.info(f"\n{pathway_name} - {bin_name}:")
                logger.info(f"  Composition: AT={at_pct:.1f}%, GC={gc_pct:.1f}%")
                logger.info(f"  Bank3_pos: median={median_pos:.2f}, σ={var_pos:.2f}")
                logger.info(f"  Bank3_neg: median={median_neg:.2f}, σ={var_neg:.2f}")
                logger.info(f"  Complexity: {complexity}")

    # 2. Variance vs Composition Analysis
    logger.info("\n=== VARIANCE vs COMPOSITION CORRELATION ===")

    # Test if variance correlates with compositional extremity
    at_pcts = np.array([b['at_pct'] for b in bin_variances])
    gc_pcts = np.array([b['gc_pct'] for b in bin_variances])
    vars_pos = np.array([b['var_pos'] for b in bin_variances])
    vars_neg = np.array([b['var_neg'] for b in bin_variances])

    # Distance from balanced composition (50% AT, 50% GC)
    compositional_distance = np.abs(at_pcts - 50.0) + np.abs(gc_pcts - 50.0)

    r_var_comp, p_var_comp = pearsonr(compositional_distance, vars_pos)
    logger.info(f"\nCompositional Distance vs Bank3_pos Variance:")
    logger.info(f"  r={r_var_comp:.3f}, p={p_var_comp:.6f}")
    logger.info(f"  Interpretation: {'Variance INCREASES with compositional extremity' if r_var_comp > 0.5 else 'Weak or NO correlation'}")

    results['variance_vs_composition'] = {
        'correlation_r': float(r_var_comp),
        'correlation_p': float(p_var_comp),
        'interpretation': 'Variance increases with compositional extremity' if r_var_comp > 0.5 else 'Weak or no correlation',
    }

    # 3. Complexity Level Summary
    high_complexity_bins = [b for b in bin_variances if (b['var_pos'] + b['var_neg'])/2 < 1.0]
    medium_complexity_bins = [b for b in bin_variances if 1.0 <= (b['var_pos'] + b['var_neg'])/2 < 3.0]
    low_complexity_bins = [b for b in bin_variances if (b['var_pos'] + b['var_neg'])/2 >= 3.0]

    logger.info("\n=== COMPLEXITY LEVEL SUMMARY ===")
    logger.info(f"\nHIGH COMPLEXITY (σ < 1.0): {len(high_complexity_bins)} bins")
    for b in high_complexity_bins:
        logger.info(f"  {b['pathway']} - {b['bin']}: σ={b['var_pos']:.2f}, AT={b['at_pct']:.1f}%, GC={b['gc_pct']:.1f}%")

    logger.info(f"\nMEDIUM COMPLEXITY (1.0 < σ < 3.0): {len(medium_complexity_bins)} bins")
    for b in medium_complexity_bins:
        logger.info(f"  {b['pathway']} - {b['bin']}: σ={b['var_pos']:.2f}, AT={b['at_pct']:.1f}%, GC={b['gc_pct']:.1f}%")

    logger.info(f"\nLOW COMPLEXITY (σ > 3.0): {len(low_complexity_bins)} bins")
    for b in low_complexity_bins:
        logger.info(f"  {b['pathway']} - {b['bin']}: σ={b['var_pos']:.2f}, AT={b['at_pct']:.1f}%, GC={b['gc_pct']:.1f}%")

    results['complexity_levels'] = {
        'high_complexity': {
            'threshold': 'σ < 1.0',
            'n_bins': len(high_complexity_bins),
            'bins': [{'pathway': b['pathway'], 'bin': b['bin'], 'variance': float(b['var_pos']),
                      'at_pct': float(b['at_pct']), 'gc_pct': float(b['gc_pct'])}
                     for b in high_complexity_bins],
        },
        'medium_complexity': {
            'threshold': '1.0 < σ < 3.0',
            'n_bins': len(medium_complexity_bins),
            'bins': [{'pathway': b['pathway'], 'bin': b['bin'], 'variance': float(b['var_pos']),
                      'at_pct': float(b['at_pct']), 'gc_pct': float(b['gc_pct'])}
                     for b in medium_complexity_bins],
        },
        'low_complexity': {
            'threshold': 'σ > 3.0',
            'n_bins': len(low_complexity_bins),
            'bins': [{'pathway': b['pathway'], 'bin': b['bin'], 'variance': float(b['var_pos']),
                      'at_pct': float(b['at_pct']), 'gc_pct': float(b['gc_pct'])}
                     for b in low_complexity_bins],
        },
    }

    # 4. Outlier Detection (High variance chunks within high-complexity bins)
    logger.info("\n=== OUTLIER DETECTION (Bank3 Variance Spikes) ===")

    for b in high_complexity_bins:
        # Get all chunks in this bin
        bin_data = data[b['pathway']][b['bin']]
        bank3_pos_all = np.array([c['signals']['bank3_pos_mag'] for c in bin_data])
        mask = bank3_pos_all > 0

        if mask.sum() > 0:
            median = np.median(bank3_pos_all[mask])
            mad = np.median(np.abs(bank3_pos_all[mask] - median))  # Median Absolute Deviation

            # Outliers: |x - median| > 3 × MAD
            outlier_threshold = median + 3 * mad
            outliers = bank3_pos_all[mask] > outlier_threshold

            if outliers.sum() > 0:
                logger.info(f"\n{b['pathway']} - {b['bin']}:")
                logger.info(f"  Median: {median:.2f}, MAD: {mad:.2f}")
                logger.info(f"  Outliers: {outliers.sum()} / {mask.sum()} ({outliers.sum()/mask.sum()*100:.1f}%)")
                logger.info(f"  Outlier threshold: > {outlier_threshold:.2f}")

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nBank3 complexity analysis saved to {output_path}")

    # Summary
    print("\n" + "="*80)
    print("BANK3 VARIANCE AS COMPLEXITY METRIC - SUMMARY")
    print("="*80)
    print("\nComplexity Classification:")
    print(f"  HIGH COMPLEXITY (σ < 1.0): {len(high_complexity_bins)} bins - Structurally constrained")
    print(f"  MEDIUM COMPLEXITY (1.0 < σ < 3.0): {len(medium_complexity_bins)} bins - Normal variation")
    print(f"  LOW COMPLEXITY (σ > 3.0): {len(low_complexity_bins)} bins - Compositional noise")
    print("\nVariance vs Composition:")
    print(f"  Correlation: r={r_var_comp:.3f}, p={p_var_comp:.6f}")
    print(f"  Interpretation: {'Variance INCREASES at compositional extremes' if r_var_comp > 0.5 else 'Weak correlation - variance is composition-independent'}")
    print("\nProduction Insight:")
    print("  - Use σ < 1.0 bins for HIGH-PRECISION motif queries")
    print("  - Use σ > 3.0 bins to FILTER OUT low-complexity/repetitive regions")
    print("  - Bank3 variance is a quality metric for structural constraint")
    print("="*80)

    return results


if __name__ == '__main__':
    data_file = "genomevault/hdv_validation/hdc_experimentation/output/AT_GC_pathway_analysis.json"
    output_file = "genomevault/hdv_validation/hdc_experimentation/output/bank3_complexity_metric.json"

    results = analyze_bank3_complexity(data_file, output_file)
