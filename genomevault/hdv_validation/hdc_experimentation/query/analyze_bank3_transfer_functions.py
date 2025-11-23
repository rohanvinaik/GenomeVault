"""
Bank3 Transfer Function Analysis: The Clean Structure Signal

Bank3 captures dinucleotide transition patterns (Y→R and R→Y) that are MORE
predictable than raw nucleotide composition. This is the primary "radio tuning"
mechanism for structural motif queries.

Hypothesis: Bank3 signals are dominated by purine/pyrimidine balance (Y% vs R%),
which is structurally constrained by Chargaff's second parity rule and functional
genomic architecture.

Author: Phase 1 Week 3 - Clean Signal Extraction
Date: November 22, 2025
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def linear(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def logarithmic(x, a, b, c):
    return a * np.log(x + 1) + b * x + c


def polynomial_4(x, a, b, c, d, e):
    """4th degree polynomial for complex compositional interactions"""
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def analyze_bank3_signals(data_file: str, output_file: str):
    """
    Analyze Bank3 (dinucleotide transition) signals across composition spectrum.

    Bank3_pos: Y→R transitions (pyrimidine to purine)
    Bank3_neg: R→Y transitions (purine to pyrimidine)

    Where Y = C + T (pyrimidines), R = A + G (purines)
    """
    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)

    results = {
        'global_models': {},
        'AT_pathway_bins': {},
        'GC_pathway_bins': {},
        'summary': {},
    }

    # First, build GLOBAL transfer functions across all chunks
    logger.info("\n" + "="*80)
    logger.info("GLOBAL BANK3 TRANSFER FUNCTIONS (n=800 chunks)")
    logger.info("="*80)

    all_chunks = []
    for pathway_name in ['AT_pathway', 'GC_pathway']:
        for bin_name, bin_data in data[pathway_name].items():
            all_chunks.extend(bin_data)

    # Extract Y% (pyrimidines) and R% (purines)
    y_pcts = np.array([(c['composition']['C_pct'] + c['composition']['T_pct']) for c in all_chunks])
    r_pcts = np.array([(c['composition']['A_pct'] + c['composition']['G_pct']) for c in all_chunks])

    bank3_pos = np.array([c['signals']['bank3_pos_mag'] for c in all_chunks])
    bank3_neg = np.array([c['signals']['bank3_neg_mag'] for c in all_chunks])

    # Y% → Bank3_pos (Y→R transitions)
    logger.info("\nY% (pyrimidines) → Bank3_pos (Y→R transitions):")
    mask = bank3_pos > 0
    y_filtered = y_pcts[mask]
    bank3_pos_filtered = bank3_pos[mask]

    r_y, p_y = pearsonr(y_filtered, bank3_pos_filtered)
    logger.info(f"  Correlation: r={r_y:.3f}, p={p_y:.10f}, n={len(y_filtered)}")

    # Fit models
    try:
        # Linear
        p_lin, _ = curve_fit(linear, y_filtered, bank3_pos_filtered)
        r2_lin = 1 - np.sum((bank3_pos_filtered - linear(y_filtered, *p_lin))**2) / \
                  np.sum((bank3_pos_filtered - bank3_pos_filtered.mean())**2)

        # Quadratic
        p_quad, _ = curve_fit(quadratic, y_filtered, bank3_pos_filtered, maxfev=5000)
        r2_quad = 1 - np.sum((bank3_pos_filtered - quadratic(y_filtered, *p_quad))**2) / \
                   np.sum((bank3_pos_filtered - bank3_pos_filtered.mean())**2)

        # Logarithmic
        p_log, _ = curve_fit(logarithmic, y_filtered, bank3_pos_filtered, maxfev=5000)
        r2_log = 1 - np.sum((bank3_pos_filtered - logarithmic(y_filtered, *p_log))**2) / \
                  np.sum((bank3_pos_filtered - bank3_pos_filtered.mean())**2)

        # Polynomial (4th degree) - for complex compositional interactions
        p_poly4, _ = curve_fit(polynomial_4, y_filtered, bank3_pos_filtered, maxfev=10000)
        r2_poly4 = 1 - np.sum((bank3_pos_filtered - polynomial_4(y_filtered, *p_poly4))**2) / \
                    np.sum((bank3_pos_filtered - bank3_pos_filtered.mean())**2)

        best_model = 'polynomial_4' if r2_poly4 == max(r2_lin, r2_quad, r2_log, r2_poly4) else \
                     ('logarithmic' if r2_log == max(r2_lin, r2_quad, r2_log) else \
                     ('quadratic' if r2_quad > r2_lin else 'linear'))

        results['global_models']['Y_to_Bank3_pos'] = {
            'correlation': {'r': float(r_y), 'p': float(p_y)},
            'n_samples': int(len(y_filtered)),
            'linear': {'coeffs': p_lin.tolist(), 'r2': float(r2_lin)},
            'quadratic': {'coeffs': p_quad.tolist(), 'r2': float(r2_quad)},
            'logarithmic': {'coeffs': p_log.tolist(), 'r2': float(r2_log)},
            'polynomial_4': {'coeffs': p_poly4.tolist(), 'r2': float(r2_poly4)},
            'best_model': best_model,
        }

        logger.info(f"  Linear:      R²={r2_lin:.4f}")
        logger.info(f"  Quadratic:   R²={r2_quad:.4f}")
        logger.info(f"  Logarithmic: R²={r2_log:.4f}")
        logger.info(f"  Polynomial4: R²={r2_poly4:.4f}")
        logger.info(f"  Best: {best_model}")

    except Exception as e:
        logger.warning(f"  Could not fit Y → Bank3_pos: {e}")

    # R% → Bank3_neg (R→Y transitions)
    logger.info("\nR% (purines) → Bank3_neg (R→Y transitions):")
    mask = bank3_neg > 0
    r_filtered = r_pcts[mask]
    bank3_neg_filtered = bank3_neg[mask]

    r_r, p_r = pearsonr(r_filtered, bank3_neg_filtered)
    logger.info(f"  Correlation: r={r_r:.3f}, p={p_r:.10f}, n={len(r_filtered)}")

    try:
        # Linear
        p_lin, _ = curve_fit(linear, r_filtered, bank3_neg_filtered)
        r2_lin = 1 - np.sum((bank3_neg_filtered - linear(r_filtered, *p_lin))**2) / \
                  np.sum((bank3_neg_filtered - bank3_neg_filtered.mean())**2)

        # Quadratic
        p_quad, _ = curve_fit(quadratic, r_filtered, bank3_neg_filtered, maxfev=5000)
        r2_quad = 1 - np.sum((bank3_neg_filtered - quadratic(r_filtered, *p_quad))**2) / \
                   np.sum((bank3_neg_filtered - bank3_neg_filtered.mean())**2)

        # Logarithmic
        p_log, _ = curve_fit(logarithmic, r_filtered, bank3_neg_filtered, maxfev=5000)
        r2_log = 1 - np.sum((bank3_neg_filtered - logarithmic(r_filtered, *p_log))**2) / \
                  np.sum((bank3_neg_filtered - bank3_neg_filtered.mean())**2)

        # Polynomial (4th degree)
        p_poly4, _ = curve_fit(polynomial_4, r_filtered, bank3_neg_filtered, maxfev=10000)
        r2_poly4 = 1 - np.sum((bank3_neg_filtered - polynomial_4(r_filtered, *p_poly4))**2) / \
                    np.sum((bank3_neg_filtered - bank3_neg_filtered.mean())**2)

        best_model = 'polynomial_4' if r2_poly4 == max(r2_lin, r2_quad, r2_log, r2_poly4) else \
                     ('logarithmic' if r2_log == max(r2_lin, r2_quad, r2_log) else \
                     ('quadratic' if r2_quad > r2_lin else 'linear'))

        results['global_models']['R_to_Bank3_neg'] = {
            'correlation': {'r': float(r_r), 'p': float(p_r)},
            'n_samples': int(len(r_filtered)),
            'linear': {'coeffs': p_lin.tolist(), 'r2': float(r2_lin)},
            'quadratic': {'coeffs': p_quad.tolist(), 'r2': float(r2_quad)},
            'logarithmic': {'coeffs': p_log.tolist(), 'r2': float(r2_log)},
            'polynomial_4': {'coeffs': p_poly4.tolist(), 'r2': float(r2_poly4)},
            'best_model': best_model,
        }

        logger.info(f"  Linear:      R²={r2_lin:.4f}")
        logger.info(f"  Quadratic:   R²={r2_quad:.4f}")
        logger.info(f"  Logarithmic: R²={r2_log:.4f}")
        logger.info(f"  Polynomial4: R²={r2_poly4:.4f}")
        logger.info(f"  Best: {best_model}")

    except Exception as e:
        logger.warning(f"  Could not fit R → Bank3_neg: {e}")

    # Now build BIN-SPECIFIC models for each composition cohort
    logger.info("\n" + "="*80)
    logger.info("BIN-SPECIFIC BANK3 TRANSFER FUNCTIONS")
    logger.info("="*80)

    for pathway_name in ['AT_pathway', 'GC_pathway']:
        logger.info(f"\n{pathway_name.upper()}:")

        for bin_name in sorted(data[pathway_name].keys()):
            bin_data = data[pathway_name][bin_name]
            logger.info(f"\n  {bin_name} (n={len(bin_data)}):")

            # Extract Y% and R%
            y_bin = np.array([(c['composition']['C_pct'] + c['composition']['T_pct']) for c in bin_data])
            r_bin = np.array([(c['composition']['A_pct'] + c['composition']['G_pct']) for c in bin_data])

            bank3_pos_bin = np.array([c['signals']['bank3_pos_mag'] for c in bin_data])
            bank3_neg_bin = np.array([c['signals']['bank3_neg_mag'] for c in bin_data])

            bin_results = {
                'composition_ranges': {
                    'Y_pct_min': float(y_bin.min()),
                    'Y_pct_max': float(y_bin.max()),
                    'Y_pct_median': float(np.median(y_bin)),
                    'R_pct_min': float(r_bin.min()),
                    'R_pct_max': float(r_bin.max()),
                    'R_pct_median': float(np.median(r_bin)),
                },
                'signal_ranges': {
                    'bank3_pos_min': float(bank3_pos_bin.min()),
                    'bank3_pos_max': float(bank3_pos_bin.max()),
                    'bank3_pos_median': float(np.median(bank3_pos_bin)),
                    'bank3_pos_std': float(np.std(bank3_pos_bin)),
                    'bank3_neg_min': float(bank3_neg_bin.min()),
                    'bank3_neg_max': float(bank3_neg_bin.max()),
                    'bank3_neg_median': float(np.median(bank3_neg_bin)),
                    'bank3_neg_std': float(np.std(bank3_neg_bin)),
                },
            }

            logger.info(f"    Y%: {y_bin.min():.1f}% - {y_bin.max():.1f}% (median {np.median(y_bin):.1f}%)")
            logger.info(f"    R%: {r_bin.min():.1f}% - {r_bin.max():.1f}% (median {np.median(r_bin):.1f}%)")
            logger.info(f"    Bank3_pos: {bank3_pos_bin.min():.2f} - {bank3_pos_bin.max():.2f} (median {np.median(bank3_pos_bin):.2f}, σ={np.std(bank3_pos_bin):.2f})")
            logger.info(f"    Bank3_neg: {bank3_neg_bin.min():.2f} - {bank3_neg_bin.max():.2f} (median {np.median(bank3_neg_bin):.2f}, σ={np.std(bank3_neg_bin):.2f})")

            # Y% → Bank3_pos correlation
            mask = bank3_pos_bin > 0
            if mask.sum() > 10 and len(np.unique(y_bin[mask])) > 1:
                r_y_bin, p_y_bin = pearsonr(y_bin[mask], bank3_pos_bin[mask])
                logger.info(f"    Y% → Bank3_pos: r={r_y_bin:.3f}, p={p_y_bin:.6f}")
                bin_results['Y_to_Bank3_pos_correlation'] = {'r': float(r_y_bin), 'p': float(p_y_bin)}

            # R% → Bank3_neg correlation
            mask = bank3_neg_bin > 0
            if mask.sum() > 10 and len(np.unique(r_bin[mask])) > 1:
                r_r_bin, p_r_bin = pearsonr(r_bin[mask], bank3_neg_bin[mask])
                logger.info(f"    R% → Bank3_neg: r={r_r_bin:.3f}, p={p_r_bin:.6f}")
                bin_results['R_to_Bank3_neg_correlation'] = {'r': float(r_r_bin), 'p': float(p_r_bin)}

            if pathway_name == 'AT_pathway':
                results['AT_pathway_bins'][bin_name] = bin_results
            else:
                results['GC_pathway_bins'][bin_name] = bin_results

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nBank3 transfer functions saved to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("BANK3 TRANSFER FUNCTION SUMMARY")
    print("="*80)
    print("\nGLOBAL MODELS (n=800 chunks):")
    print("\nY% → Bank3_pos (Y→R transitions):")
    y_model = results['global_models']['Y_to_Bank3_pos']
    print(f"  Correlation: r={y_model['correlation']['r']:.3f}, p={y_model['correlation']['p']:.10f}")
    print(f"  Best model: {y_model['best_model']} (R²={y_model[y_model['best_model']]['r2']:.4f})")
    print(f"  Coefficients: {y_model[y_model['best_model']]['coeffs']}")

    print("\nR% → Bank3_neg (R→Y transitions):")
    r_model = results['global_models']['R_to_Bank3_neg']
    print(f"  Correlation: r={r_model['correlation']['r']:.3f}, p={r_model['correlation']['p']:.10f}")
    print(f"  Best model: {r_model['best_model']} (R²={r_model[r_model['best_model']]['r2']:.4f})")
    print(f"  Coefficients: {r_model[r_model['best_model']]['coeffs']}")

    print("\n" + "="*80)
    print("RADIO TUNING STRATEGY:")
    print("="*80)
    print("Bank3 is the PRIMARY structure signal - use it for motif-specific queries.")
    print("Bank1/Bank2 are composition filters - use for coarse filtering only.")
    print("="*80)

    return results


if __name__ == '__main__':
    data_file = "genomevault/hdv_validation/hdc_experimentation/output/AT_GC_pathway_analysis.json"
    output_file = "genomevault/hdv_validation/hdc_experimentation/output/bank3_transfer_functions.json"

    results = analyze_bank3_signals(data_file, output_file)
