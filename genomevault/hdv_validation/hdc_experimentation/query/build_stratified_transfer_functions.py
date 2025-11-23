"""
Build Stratified Transfer Functions for Split Binary Architecture

Hypothesis: Global transfer functions have low R² (0.045-0.108) because
different composition ranges have different signal dynamics.

Solution: Build separate regression models for each composition percentile bin
to achieve higher local R² values.

Author: Phase 1 Week 3 - Composition-Specific Tuning
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


def build_stratified_functions(data_file: str, output_file: str):
    """
    Build transfer functions separately for each composition percentile bin.

    This should yield much higher R² values than global regressions.

    Args:
        data_file: Path to AT_GC_pathway_analysis.json
        output_file: Where to save stratified transfer functions
    """
    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)

    results = {
        'AT_pathway': {},
        'GC_pathway': {},
    }

    # AT Pathway: Build bin-specific transfer functions
    logger.info("\n=== AT PATHWAY: Stratified Transfer Functions ===")

    for bin_name, bin_data in data['AT_pathway'].items():
        logger.info(f"\n{bin_name}:")
        bin_results = {}

        # Extract A% and Bank1_pos signal
        a_pcts = np.array([chunk['composition']['A_pct'] for chunk in bin_data])
        bank1_pos_signals = np.array([chunk['signals']['bank1_pos_mag'] for chunk in bin_data])

        # Filter out zero signals for regression
        mask = bank1_pos_signals > 0
        if mask.sum() < 10:
            logger.warning(f"  A → Bank1_pos: Too few non-zero samples ({mask.sum()})")
        else:
            a_pcts_filtered = a_pcts[mask]
            bank1_pos_filtered = bank1_pos_signals[mask]

            # Compute correlation
            if len(np.unique(a_pcts_filtered)) > 1:
                r_a, p_a = pearsonr(a_pcts_filtered, bank1_pos_filtered)
                logger.info(f"  A% → Bank1_pos: r={r_a:.3f}, p={p_a:.6f}, n={len(a_pcts_filtered)}")

                # Fit models
                try:
                    p_lin, _ = curve_fit(linear, a_pcts_filtered, bank1_pos_filtered)
                    r2_lin = 1 - np.sum((bank1_pos_filtered - linear(a_pcts_filtered, *p_lin))**2) / np.sum((bank1_pos_filtered - bank1_pos_filtered.mean())**2)

                    p_quad, _ = curve_fit(quadratic, a_pcts_filtered, bank1_pos_filtered, maxfev=5000)
                    r2_quad = 1 - np.sum((bank1_pos_filtered - quadratic(a_pcts_filtered, *p_quad))**2) / np.sum((bank1_pos_filtered - bank1_pos_filtered.mean())**2)

                    p_log, _ = curve_fit(logarithmic, a_pcts_filtered, bank1_pos_filtered, maxfev=5000)
                    r2_log = 1 - np.sum((bank1_pos_filtered - logarithmic(a_pcts_filtered, *p_log))**2) / np.sum((bank1_pos_filtered - bank1_pos_filtered.mean())**2)

                    best_model = 'logarithmic' if r2_log == max(r2_lin, r2_quad, r2_log) else ('quadratic' if r2_quad > r2_lin else 'linear')

                    bin_results['A_to_Bank1_pos'] = {
                        'correlation': {'r': float(r_a), 'p': float(p_a)},
                        'n_samples': int(len(a_pcts_filtered)),
                        'linear': {'coeffs': p_lin.tolist(), 'r2': float(r2_lin)},
                        'quadratic': {'coeffs': p_quad.tolist(), 'r2': float(r2_quad)},
                        'logarithmic': {'coeffs': p_log.tolist(), 'r2': float(r2_log)},
                        'best_model': best_model,
                    }

                    logger.info(f"    Linear:      R²={r2_lin:.4f}")
                    logger.info(f"    Quadratic:   R²={r2_quad:.4f}")
                    logger.info(f"    Logarithmic: R²={r2_log:.4f}")
                    logger.info(f"    Best: {best_model}")
                except Exception as e:
                    logger.warning(f"    Could not fit: {e}")

        # Extract T% and Bank1_neg signal
        t_pcts = np.array([chunk['composition']['T_pct'] for chunk in bin_data])
        bank1_neg_signals = np.array([chunk['signals']['bank1_neg_mag'] for chunk in bin_data])

        mask = bank1_neg_signals > 0
        if mask.sum() < 10:
            logger.warning(f"  T → Bank1_neg: Too few non-zero samples ({mask.sum()})")
        else:
            t_pcts_filtered = t_pcts[mask]
            bank1_neg_filtered = bank1_neg_signals[mask]

            if len(np.unique(t_pcts_filtered)) > 1:
                r_t, p_t = pearsonr(t_pcts_filtered, bank1_neg_filtered)
                logger.info(f"  T% → Bank1_neg: r={r_t:.3f}, p={p_t:.6f}, n={len(t_pcts_filtered)}")

                try:
                    p_lin, _ = curve_fit(linear, t_pcts_filtered, bank1_neg_filtered)
                    r2_lin = 1 - np.sum((bank1_neg_filtered - linear(t_pcts_filtered, *p_lin))**2) / np.sum((bank1_neg_filtered - bank1_neg_filtered.mean())**2)

                    p_quad, _ = curve_fit(quadratic, t_pcts_filtered, bank1_neg_filtered, maxfev=5000)
                    r2_quad = 1 - np.sum((bank1_neg_filtered - quadratic(t_pcts_filtered, *p_quad))**2) / np.sum((bank1_neg_filtered - bank1_neg_filtered.mean())**2)

                    p_log, _ = curve_fit(logarithmic, t_pcts_filtered, bank1_neg_filtered, maxfev=5000)
                    r2_log = 1 - np.sum((bank1_neg_filtered - logarithmic(t_pcts_filtered, *p_log))**2) / np.sum((bank1_neg_filtered - bank1_neg_filtered.mean())**2)

                    best_model = 'logarithmic' if r2_log == max(r2_lin, r2_quad, r2_log) else ('quadratic' if r2_quad > r2_lin else 'linear')

                    bin_results['T_to_Bank1_neg'] = {
                        'correlation': {'r': float(r_t), 'p': float(p_t)},
                        'n_samples': int(len(t_pcts_filtered)),
                        'linear': {'coeffs': p_lin.tolist(), 'r2': float(r2_lin)},
                        'quadratic': {'coeffs': p_quad.tolist(), 'r2': float(r2_quad)},
                        'logarithmic': {'coeffs': p_log.tolist(), 'r2': float(r2_log)},
                        'best_model': best_model,
                    }

                    logger.info(f"    Linear:      R²={r2_lin:.4f}")
                    logger.info(f"    Quadratic:   R²={r2_quad:.4f}")
                    logger.info(f"    Logarithmic: R²={r2_log:.4f}")
                    logger.info(f"    Best: {best_model}")
                except Exception as e:
                    logger.warning(f"    Could not fit: {e}")

        results['AT_pathway'][bin_name] = bin_results

    # GC Pathway: Build bin-specific transfer functions
    logger.info("\n=== GC PATHWAY: Stratified Transfer Functions ===")

    for bin_name, bin_data in data['GC_pathway'].items():
        logger.info(f"\n{bin_name}:")
        bin_results = {}

        # Extract G% and Bank2_pos signal
        g_pcts = np.array([chunk['composition']['G_pct'] for chunk in bin_data])
        bank2_pos_signals = np.array([chunk['signals']['bank2_pos_mag'] for chunk in bin_data])

        mask = bank2_pos_signals > 0
        if mask.sum() < 10:
            logger.warning(f"  G → Bank2_pos: Too few non-zero samples ({mask.sum()})")
        else:
            g_pcts_filtered = g_pcts[mask]
            bank2_pos_filtered = bank2_pos_signals[mask]

            if len(np.unique(g_pcts_filtered)) > 1:
                r_g, p_g = pearsonr(g_pcts_filtered, bank2_pos_filtered)
                logger.info(f"  G% → Bank2_pos: r={r_g:.3f}, p={p_g:.6f}, n={len(g_pcts_filtered)}")

                try:
                    p_lin, _ = curve_fit(linear, g_pcts_filtered, bank2_pos_filtered)
                    r2_lin = 1 - np.sum((bank2_pos_filtered - linear(g_pcts_filtered, *p_lin))**2) / np.sum((bank2_pos_filtered - bank2_pos_filtered.mean())**2)

                    p_quad, _ = curve_fit(quadratic, g_pcts_filtered, bank2_pos_filtered, maxfev=5000)
                    r2_quad = 1 - np.sum((bank2_pos_filtered - quadratic(g_pcts_filtered, *p_quad))**2) / np.sum((bank2_pos_filtered - bank2_pos_filtered.mean())**2)

                    p_log, _ = curve_fit(logarithmic, g_pcts_filtered, bank2_pos_filtered, maxfev=5000)
                    r2_log = 1 - np.sum((bank2_pos_filtered - logarithmic(g_pcts_filtered, *p_log))**2) / np.sum((bank2_pos_filtered - bank2_pos_filtered.mean())**2)

                    best_model = 'logarithmic' if r2_log == max(r2_lin, r2_quad, r2_log) else ('quadratic' if r2_quad > r2_lin else 'linear')

                    bin_results['G_to_Bank2_pos'] = {
                        'correlation': {'r': float(r_g), 'p': float(p_g)},
                        'n_samples': int(len(g_pcts_filtered)),
                        'linear': {'coeffs': p_lin.tolist(), 'r2': float(r2_lin)},
                        'quadratic': {'coeffs': p_quad.tolist(), 'r2': float(r2_quad)},
                        'logarithmic': {'coeffs': p_log.tolist(), 'r2': float(r2_log)},
                        'best_model': best_model,
                    }

                    logger.info(f"    Linear:      R²={r2_lin:.4f}")
                    logger.info(f"    Quadratic:   R²={r2_quad:.4f}")
                    logger.info(f"    Logarithmic: R²={r2_log:.4f}")
                    logger.info(f"    Best: {best_model}")
                except Exception as e:
                    logger.warning(f"    Could not fit: {e}")

        # Extract C% and Bank2_neg signal
        c_pcts = np.array([chunk['composition']['C_pct'] for chunk in bin_data])
        bank2_neg_signals = np.array([chunk['signals']['bank2_neg_mag'] for chunk in bin_data])

        mask = bank2_neg_signals > 0
        if mask.sum() < 10:
            logger.warning(f"  C → Bank2_neg: Too few non-zero samples ({mask.sum()})")
        else:
            c_pcts_filtered = c_pcts[mask]
            bank2_neg_filtered = bank2_neg_signals[mask]

            if len(np.unique(c_pcts_filtered)) > 1:
                r_c, p_c = pearsonr(c_pcts_filtered, bank2_neg_filtered)
                logger.info(f"  C% → Bank2_neg: r={r_c:.3f}, p={p_c:.6f}, n={len(c_pcts_filtered)}")

                try:
                    p_lin, _ = curve_fit(linear, c_pcts_filtered, bank2_neg_filtered)
                    r2_lin = 1 - np.sum((bank2_neg_filtered - linear(c_pcts_filtered, *p_lin))**2) / np.sum((bank2_neg_filtered - bank2_neg_filtered.mean())**2)

                    p_quad, _ = curve_fit(quadratic, c_pcts_filtered, bank2_neg_filtered, maxfev=5000)
                    r2_quad = 1 - np.sum((bank2_neg_filtered - quadratic(c_pcts_filtered, *p_quad))**2) / np.sum((bank2_neg_filtered - bank2_neg_filtered.mean())**2)

                    p_log, _ = curve_fit(logarithmic, c_pcts_filtered, bank2_neg_filtered, maxfev=5000)
                    r2_log = 1 - np.sum((bank2_neg_filtered - logarithmic(c_pcts_filtered, *p_log))**2) / np.sum((bank2_neg_filtered - bank2_neg_filtered.mean())**2)

                    best_model = 'logarithmic' if r2_log == max(r2_lin, r2_quad, r2_log) else ('quadratic' if r2_quad > r2_lin else 'linear')

                    bin_results['C_to_Bank2_neg'] = {
                        'correlation': {'r': float(r_c), 'p': float(p_c)},
                        'n_samples': int(len(c_pcts_filtered)),
                        'linear': {'coeffs': p_lin.tolist(), 'r2': float(r2_lin)},
                        'quadratic': {'coeffs': p_quad.tolist(), 'r2': float(r2_quad)},
                        'logarithmic': {'coeffs': p_log.tolist(), 'r2': float(r2_log)},
                        'best_model': best_model,
                    }

                    logger.info(f"    Linear:      R²={r2_lin:.4f}")
                    logger.info(f"    Quadratic:   R²={r2_quad:.4f}")
                    logger.info(f"    Logarithmic: R²={r2_log:.4f}")
                    logger.info(f"    Best: {best_model}")
                except Exception as e:
                    logger.warning(f"    Could not fit: {e}")

        results['GC_pathway'][bin_name] = bin_results

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nStratified transfer functions saved to {output_path}")

    # Print comparison summary
    print("\n" + "="*80)
    print("STRATIFIED TRANSFER FUNCTIONS SUMMARY")
    print("="*80)
    print("\nCompare these R² values to global regressions:")
    print("  Global A → Bank1_pos: R² = 0.0453")
    print("  Global T → Bank1_neg: R² = 0.0996")
    print("  Global G → Bank2_pos: R² = 0.1056")
    print("  Global C → Bank2_neg: R² = 0.1082")
    print("\n" + "="*80)

    for pathway in ['AT_pathway', 'GC_pathway']:
        print(f"\n{pathway.upper()}:")
        for bin_name in sorted(results[pathway].keys()):
            print(f"\n  {bin_name}:")
            for tf_name, tf_data in results[pathway][bin_name].items():
                best = tf_data['best_model']
                r2 = tf_data[best]['r2']
                r = tf_data['correlation']['r']
                p = tf_data['correlation']['p']
                n = tf_data['n_samples']
                print(f"    {tf_name}:")
                print(f"      r={r:.3f}, p={p:.6f}, n={n}")
                print(f"      Best: {best} (R²={r2:.4f})")

    print("="*80)

    return results


if __name__ == '__main__':
    data_file = "genomevault/hdv_validation/hdc_experimentation/output/AT_GC_pathway_analysis.json"
    output_file = "genomevault/hdv_validation/hdc_experimentation/output/stratified_transfer_functions.json"

    results = build_stratified_functions(data_file, output_file)
