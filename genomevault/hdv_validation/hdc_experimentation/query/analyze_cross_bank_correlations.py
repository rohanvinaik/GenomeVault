"""
Cross-Bank Correlation Analysis: Bank1 vs Bank2 Anti-Correlation at Extremes

Hypothesis: Since AT% + GC% = 100%, Bank1 and Bank2 should show anti-correlation
at compositional extremes (high AT → low GC, and vice versa).

This analysis tests whether:
1. Bank1_pos and Bank2_pos are anti-correlated
2. Bank1_neg and Bank2_neg are anti-correlated
3. The anti-correlation is stronger at compositional extremes
4. Bank3 remains independent of Bank1/Bank2 (composition-independent)

Author: Phase 1 Week 3 - Cross-Bank Signal Independence
Date: November 22, 2025
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_cross_bank_correlations(data_file: str, output_file: str):
    """
    Analyze correlations between Bank1, Bank2, and Bank3 signals.
    """
    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)

    results = {
        'global_correlations': {},
        'bin_specific_correlations': {},
        'compositional_extremes': {},
        'bank3_independence': {},
    }

    # Collect all chunks across both pathways
    all_chunks = []
    for pathway_name in ['AT_pathway', 'GC_pathway']:
        for bin_name, bin_data in data[pathway_name].items():
            all_chunks.extend(bin_data)

    logger.info("\n" + "="*80)
    logger.info("GLOBAL CROSS-BANK CORRELATIONS (n=800 chunks)")
    logger.info("="*80)

    # Extract all bank signals
    bank1_pos = np.array([c['signals']['bank1_pos_mag'] for c in all_chunks])
    bank1_neg = np.array([c['signals']['bank1_neg_mag'] for c in all_chunks])
    bank2_pos = np.array([c['signals']['bank2_pos_mag'] for c in all_chunks])
    bank2_neg = np.array([c['signals']['bank2_neg_mag'] for c in all_chunks])
    bank3_pos = np.array([c['signals']['bank3_pos_mag'] for c in all_chunks])
    bank3_neg = np.array([c['signals']['bank3_neg_mag'] for c in all_chunks])

    # Also get composition
    at_pct = np.array([(c['composition']['A_pct'] + c['composition']['T_pct']) for c in all_chunks])
    gc_pct = np.array([(c['composition']['G_pct'] + c['composition']['C_pct']) for c in all_chunks])

    # Filter out zero signals
    mask = (bank1_pos > 0) & (bank2_pos > 0) & (bank3_pos > 0)

    # 1. Bank1 vs Bank2 correlations (HYPOTHESIS: Anti-correlation)
    logger.info("\n=== BANK1 vs BANK2 (Anti-Correlation Hypothesis) ===")

    r_b1pos_b2pos, p_b1pos_b2pos = pearsonr(bank1_pos[mask], bank2_pos[mask])
    logger.info(f"Bank1_pos vs Bank2_pos: r={r_b1pos_b2pos:.3f}, p={p_b1pos_b2pos:.6f}")

    r_b1neg_b2neg, p_b1neg_b2neg = pearsonr(bank1_neg[mask], bank2_neg[mask])
    logger.info(f"Bank1_neg vs Bank2_neg: r={r_b1neg_b2neg:.3f}, p={p_b1neg_b2neg:.6f}")

    # Spearman for non-linear anti-correlation
    rho_b1pos_b2pos, p_rho_b1pos_b2pos = spearmanr(bank1_pos[mask], bank2_pos[mask])
    logger.info(f"Bank1_pos vs Bank2_pos (Spearman): ρ={rho_b1pos_b2pos:.3f}, p={p_rho_b1pos_b2pos:.6f}")

    results['global_correlations']['bank1_vs_bank2'] = {
        'bank1_pos_vs_bank2_pos': {
            'pearson_r': float(r_b1pos_b2pos),
            'pearson_p': float(p_b1pos_b2pos),
            'spearman_rho': float(rho_b1pos_b2pos),
            'spearman_p': float(p_rho_b1pos_b2pos),
        },
        'bank1_neg_vs_bank2_neg': {
            'pearson_r': float(r_b1neg_b2neg),
            'pearson_p': float(p_b1neg_b2neg),
        },
    }

    # 2. Bank3 independence (HYPOTHESIS: No correlation with Bank1/Bank2)
    logger.info("\n=== BANK3 INDEPENDENCE (Composition-Independent Hypothesis) ===")

    r_b1pos_b3pos, p_b1pos_b3pos = pearsonr(bank1_pos[mask], bank3_pos[mask])
    logger.info(f"Bank1_pos vs Bank3_pos: r={r_b1pos_b3pos:.3f}, p={p_b1pos_b3pos:.6f}")

    r_b2pos_b3pos, p_b2pos_b3pos = pearsonr(bank2_pos[mask], bank3_pos[mask])
    logger.info(f"Bank2_pos vs Bank3_pos: r={r_b2pos_b3pos:.3f}, p={p_b2pos_b3pos:.6f}")

    r_b1neg_b3neg, p_b1neg_b3neg = pearsonr(bank1_neg[mask], bank3_neg[mask])
    logger.info(f"Bank1_neg vs Bank3_neg: r={r_b1neg_b3neg:.3f}, p={p_b1neg_b3neg:.6f}")

    r_b2neg_b3neg, p_b2neg_b3neg = pearsonr(bank2_neg[mask], bank3_neg[mask])
    logger.info(f"Bank2_neg vs Bank3_neg: r={r_b2neg_b3neg:.3f}, p={p_b2neg_b3neg:.6f}")

    results['bank3_independence'] = {
        'bank1_pos_vs_bank3_pos': {'r': float(r_b1pos_b3pos), 'p': float(p_b1pos_b3pos)},
        'bank2_pos_vs_bank3_pos': {'r': float(r_b2pos_b3pos), 'p': float(p_b2pos_b3pos)},
        'bank1_neg_vs_bank3_neg': {'r': float(r_b1neg_b3neg), 'p': float(p_b1neg_b3neg)},
        'bank2_neg_vs_bank3_neg': {'r': float(r_b2neg_b3neg), 'p': float(p_b2neg_b3neg)},
    }

    # 3. Anti-correlation at compositional extremes
    logger.info("\n=== ANTI-CORRELATION AT COMPOSITIONAL EXTREMES ===")

    # Define extremes: AT% > 60% (high AT, low GC) and GC% > 45% (high GC, low AT)
    high_at_mask = (at_pct > 60) & mask
    high_gc_mask = (gc_pct > 45) & mask
    normal_mask = (at_pct >= 50) & (at_pct <= 60) & (gc_pct >= 40) & (gc_pct <= 50) & mask

    logger.info(f"\nHigh AT regions (AT% > 60%, n={high_at_mask.sum()}):")
    if high_at_mask.sum() > 10:
        r_extreme_at, p_extreme_at = pearsonr(bank1_pos[high_at_mask], bank2_pos[high_at_mask])
        logger.info(f"  Bank1_pos vs Bank2_pos: r={r_extreme_at:.3f}, p={p_extreme_at:.6f}")
        results['compositional_extremes']['high_AT'] = {
            'n': int(high_at_mask.sum()),
            'bank1_vs_bank2_r': float(r_extreme_at),
            'bank1_vs_bank2_p': float(p_extreme_at),
        }

    logger.info(f"\nHigh GC regions (GC% > 45%, n={high_gc_mask.sum()}):")
    if high_gc_mask.sum() > 10:
        r_extreme_gc, p_extreme_gc = pearsonr(bank1_pos[high_gc_mask], bank2_pos[high_gc_mask])
        logger.info(f"  Bank1_pos vs Bank2_pos: r={r_extreme_gc:.3f}, p={p_extreme_gc:.6f}")
        results['compositional_extremes']['high_GC'] = {
            'n': int(high_gc_mask.sum()),
            'bank1_vs_bank2_r': float(r_extreme_gc),
            'bank1_vs_bank2_p': float(p_extreme_gc),
        }

    logger.info(f"\nNormal composition (AT% 50-60%, GC% 40-50%, n={normal_mask.sum()}):")
    if normal_mask.sum() > 10:
        r_normal, p_normal = pearsonr(bank1_pos[normal_mask], bank2_pos[normal_mask])
        logger.info(f"  Bank1_pos vs Bank2_pos: r={r_normal:.3f}, p={p_normal:.6f}")
        results['compositional_extremes']['normal'] = {
            'n': int(normal_mask.sum()),
            'bank1_vs_bank2_r': float(r_normal),
            'bank1_vs_bank2_p': float(p_normal),
        }

    # 4. Bin-specific correlations
    logger.info("\n=== BIN-SPECIFIC CROSS-BANK CORRELATIONS ===")

    for pathway_name in ['AT_pathway', 'GC_pathway']:
        logger.info(f"\n{pathway_name.upper()}:")
        results['bin_specific_correlations'][pathway_name] = {}

        for bin_name in sorted(data[pathway_name].keys()):
            bin_data = data[pathway_name][bin_name]
            logger.info(f"\n  {bin_name} (n={len(bin_data)}):")

            # Extract bin signals
            b1_pos = np.array([c['signals']['bank1_pos_mag'] for c in bin_data])
            b2_pos = np.array([c['signals']['bank2_pos_mag'] for c in bin_data])
            b3_pos = np.array([c['signals']['bank3_pos_mag'] for c in bin_data])

            bin_mask = (b1_pos > 0) & (b2_pos > 0) & (b3_pos > 0)

            if bin_mask.sum() > 10:
                r_b1_b2, p_b1_b2 = pearsonr(b1_pos[bin_mask], b2_pos[bin_mask])
                r_b1_b3, p_b1_b3 = pearsonr(b1_pos[bin_mask], b3_pos[bin_mask])
                r_b2_b3, p_b2_b3 = pearsonr(b2_pos[bin_mask], b3_pos[bin_mask])

                logger.info(f"    Bank1 vs Bank2: r={r_b1_b2:.3f}, p={p_b1_b2:.6f}")
                logger.info(f"    Bank1 vs Bank3: r={r_b1_b3:.3f}, p={p_b1_b3:.6f}")
                logger.info(f"    Bank2 vs Bank3: r={r_b2_b3:.3f}, p={p_b2_b3:.6f}")

                results['bin_specific_correlations'][pathway_name][bin_name] = {
                    'n': int(bin_mask.sum()),
                    'bank1_vs_bank2': {'r': float(r_b1_b2), 'p': float(p_b1_b2)},
                    'bank1_vs_bank3': {'r': float(r_b1_b3), 'p': float(p_b1_b3)},
                    'bank2_vs_bank3': {'r': float(r_b2_b3), 'p': float(p_b2_b3)},
                }

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nCross-bank correlation analysis saved to {output_path}")

    # Summary
    print("\n" + "="*80)
    print("CROSS-BANK CORRELATION SUMMARY")
    print("="*80)
    print("\nHypothesis 1: Bank1 vs Bank2 Anti-Correlation")
    print(f"  Global: r={r_b1pos_b2pos:.3f} (p={p_b1pos_b2pos:.6f})")
    print(f"  Interpretation: {'NEGATIVE correlation (anti-correlated)' if r_b1pos_b2pos < -0.1 else 'WEAK or NO anti-correlation'}")

    print("\nHypothesis 2: Bank3 Independence")
    print(f"  Bank1 vs Bank3: r={r_b1pos_b3pos:.3f} (p={p_b1pos_b3pos:.6f})")
    print(f"  Bank2 vs Bank3: r={r_b2pos_b3pos:.3f} (p={p_b2pos_b3pos:.6f})")
    print(f"  Interpretation: {'INDEPENDENT' if abs(r_b1pos_b3pos) < 0.2 and abs(r_b2pos_b3pos) < 0.2 else 'SOME DEPENDENCY'}")

    print("\nHypothesis 3: Stronger Anti-Correlation at Extremes")
    if 'high_AT' in results['compositional_extremes'] and 'normal' in results['compositional_extremes']:
        r_extreme = results['compositional_extremes']['high_AT']['bank1_vs_bank2_r']
        r_normal = results['compositional_extremes']['normal']['bank1_vs_bank2_r']
        print(f"  High AT: r={r_extreme:.3f}")
        print(f"  Normal: r={r_normal:.3f}")
        print(f"  Interpretation: {'STRONGER anti-correlation at extremes' if r_extreme < r_normal else 'NO stronger anti-correlation'}")

    print("="*80)

    return results


if __name__ == '__main__':
    data_file = "genomevault/hdv_validation/hdc_experimentation/output/AT_GC_pathway_analysis.json"
    output_file = "genomevault/hdv_validation/hdc_experimentation/output/cross_bank_correlations.json"

    results = analyze_cross_bank_correlations(data_file, output_file)
