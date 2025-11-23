"""
Build Transfer Functions for Split Binary Architecture

Analyzes the AT/GC pathway data to build complete signal transfer functions.

Goal: Given a target composition (A%, T%, G%, C%), predict the required
bank threshold parameters for selective queries ("radio tuning").

Author: Phase 1 Week 3 - Complete Degradation Curves
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


def build_transfer_functions(data_file: str, output_file: str):
    """
    Build transfer functions for each pathway and nucleotide.

    Args:
        data_file: Path to AT_GC_pathway_analysis.json
        output_file: Where to save transfer functions
    """
    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)

    results = {
        'AT_pathway': {},
        'GC_pathway': {},
    }

    # AT Pathway: Build transfer functions for A% → Bank1_pos and T% → Bank1_neg
    logger.info("\n=== AT Pathway Transfer Functions ===")

    at_chunks = []
    for bin_name, bin_data in data['AT_pathway'].items():
        at_chunks.extend(bin_data)

    # Extract A% and Bank1_pos signal
    a_pcts = np.array([chunk['composition']['A_pct'] for chunk in at_chunks])
    bank1_pos_signals = np.array([chunk['signals']['bank1_pos_mag'] for chunk in at_chunks])

    # Filter out zero signals for regression
    mask = bank1_pos_signals > 0
    a_pcts_filtered = a_pcts[mask]
    bank1_pos_filtered = bank1_pos_signals[mask]

    # Compute correlation
    r_a, p_a = pearsonr(a_pcts_filtered, bank1_pos_filtered)
    logger.info(f"A% → Bank1_pos: r={r_a:.3f}, p={p_a:.6f}")

    # Fit models
    try:
        p_lin, _ = curve_fit(linear, a_pcts_filtered, bank1_pos_filtered)
        r2_lin = 1 - np.sum((bank1_pos_filtered - linear(a_pcts_filtered, *p_lin))**2) / np.sum((bank1_pos_filtered - bank1_pos_filtered.mean())**2)

        p_quad, _ = curve_fit(quadratic, a_pcts_filtered, bank1_pos_filtered)
        r2_quad = 1 - np.sum((bank1_pos_filtered - quadratic(a_pcts_filtered, *p_quad))**2) / np.sum((bank1_pos_filtered - bank1_pos_filtered.mean())**2)

        p_log, _ = curve_fit(logarithmic, a_pcts_filtered, bank1_pos_filtered)
        r2_log = 1 - np.sum((bank1_pos_filtered - logarithmic(a_pcts_filtered, *p_log))**2) / np.sum((bank1_pos_filtered - bank1_pos_filtered.mean())**2)

        results['AT_pathway']['A_to_Bank1_pos'] = {
            'correlation': {'r': float(r_a), 'p': float(p_a)},
            'linear': {'coeffs': p_lin.tolist(), 'r2': float(r2_lin)},
            'quadratic': {'coeffs': p_quad.tolist(), 'r2': float(r2_quad)},
            'logarithmic': {'coeffs': p_log.tolist(), 'r2': float(r2_log)},
            'best_model': 'quadratic' if r2_quad == max(r2_lin, r2_quad, r2_log) else ('logarithmic' if r2_log > r2_lin else 'linear'),
        }

        logger.info(f"  Linear:      R²={r2_lin:.4f}")
        logger.info(f"  Quadratic:   R²={r2_quad:.4f}")
        logger.info(f"  Logarithmic: R²={r2_log:.4f}")
    except Exception as e:
        logger.warning(f"Could not fit A% → Bank1_pos: {e}")

    # Extract T% and Bank1_neg signal
    t_pcts = np.array([chunk['composition']['T_pct'] for chunk in at_chunks])
    bank1_neg_signals = np.array([chunk['signals']['bank1_neg_mag'] for chunk in at_chunks])

    mask = bank1_neg_signals > 0
    t_pcts_filtered = t_pcts[mask]
    bank1_neg_filtered = bank1_neg_signals[mask]

    r_t, p_t = pearsonr(t_pcts_filtered, bank1_neg_filtered)
    logger.info(f"\nT% → Bank1_neg: r={r_t:.3f}, p={p_t:.6f}")

    try:
        p_lin, _ = curve_fit(linear, t_pcts_filtered, bank1_neg_filtered)
        r2_lin = 1 - np.sum((bank1_neg_filtered - linear(t_pcts_filtered, *p_lin))**2) / np.sum((bank1_neg_filtered - bank1_neg_filtered.mean())**2)

        p_quad, _ = curve_fit(quadratic, t_pcts_filtered, bank1_neg_filtered)
        r2_quad = 1 - np.sum((bank1_neg_filtered - quadratic(t_pcts_filtered, *p_quad))**2) / np.sum((bank1_neg_filtered - bank1_neg_filtered.mean())**2)

        p_log, _ = curve_fit(logarithmic, t_pcts_filtered, bank1_neg_filtered)
        r2_log = 1 - np.sum((bank1_neg_filtered - logarithmic(t_pcts_filtered, *p_log))**2) / np.sum((bank1_neg_filtered - bank1_neg_filtered.mean())**2)

        results['AT_pathway']['T_to_Bank1_neg'] = {
            'correlation': {'r': float(r_t), 'p': float(p_t)},
            'linear': {'coeffs': p_lin.tolist(), 'r2': float(r2_lin)},
            'quadratic': {'coeffs': p_quad.tolist(), 'r2': float(r2_quad)},
            'logarithmic': {'coeffs': p_log.tolist(), 'r2': float(r2_log)},
            'best_model': 'quadratic' if r2_quad == max(r2_lin, r2_quad, r2_log) else ('logarithmic' if r2_log > r2_lin else 'linear'),
        }

        logger.info(f"  Linear:      R²={r2_lin:.4f}")
        logger.info(f"  Quadratic:   R²={r2_quad:.4f}")
        logger.info(f"  Logarithmic: R²={r2_log:.4f}")
    except Exception as e:
        logger.warning(f"Could not fit T% → Bank1_neg: {e}")

    # GC Pathway: Build transfer functions for G% → Bank2_pos and C% → Bank2_neg
    logger.info("\n=== GC Pathway Transfer Functions ===")

    gc_chunks = []
    for bin_name, bin_data in data['GC_pathway'].items():
        gc_chunks.extend(bin_data)

    # Extract G% and Bank2_pos signal
    g_pcts = np.array([chunk['composition']['G_pct'] for chunk in gc_chunks])
    bank2_pos_signals = np.array([chunk['signals']['bank2_pos_mag'] for chunk in gc_chunks])

    mask = bank2_pos_signals > 0
    g_pcts_filtered = g_pcts[mask]
    bank2_pos_filtered = bank2_pos_signals[mask]

    r_g, p_g = pearsonr(g_pcts_filtered, bank2_pos_filtered)
    logger.info(f"G% → Bank2_pos: r={r_g:.3f}, p={p_g:.6f}")

    try:
        p_lin, _ = curve_fit(linear, g_pcts_filtered, bank2_pos_filtered)
        r2_lin = 1 - np.sum((bank2_pos_filtered - linear(g_pcts_filtered, *p_lin))**2) / np.sum((bank2_pos_filtered - bank2_pos_filtered.mean())**2)

        p_quad, _ = curve_fit(quadratic, g_pcts_filtered, bank2_pos_filtered)
        r2_quad = 1 - np.sum((bank2_pos_filtered - quadratic(g_pcts_filtered, *p_quad))**2) / np.sum((bank2_pos_filtered - bank2_pos_filtered.mean())**2)

        p_log, _ = curve_fit(logarithmic, g_pcts_filtered, bank2_pos_filtered)
        r2_log = 1 - np.sum((bank2_pos_filtered - logarithmic(g_pcts_filtered, *p_log))**2) / np.sum((bank2_pos_filtered - bank2_pos_filtered.mean())**2)

        results['GC_pathway']['G_to_Bank2_pos'] = {
            'correlation': {'r': float(r_g), 'p': float(p_g)},
            'linear': {'coeffs': p_lin.tolist(), 'r2': float(r2_lin)},
            'quadratic': {'coeffs': p_quad.tolist(), 'r2': float(r2_quad)},
            'logarithmic': {'coeffs': p_log.tolist(), 'r2': float(r2_log)},
            'best_model': 'quadratic' if r2_quad == max(r2_lin, r2_quad, r2_log) else ('logarithmic' if r2_log > r2_lin else 'linear'),
        }

        logger.info(f"  Linear:      R²={r2_lin:.4f}")
        logger.info(f"  Quadratic:   R²={r2_quad:.4f}")
        logger.info(f"  Logarithmic: R²={r2_log:.4f}")
    except Exception as e:
        logger.warning(f"Could not fit G% → Bank2_pos: {e}")

    # Extract C% and Bank2_neg signal
    c_pcts = np.array([chunk['composition']['C_pct'] for chunk in gc_chunks])
    bank2_neg_signals = np.array([chunk['signals']['bank2_neg_mag'] for chunk in gc_chunks])

    mask = bank2_neg_signals > 0
    c_pcts_filtered = c_pcts[mask]
    bank2_neg_filtered = bank2_neg_signals[mask]

    r_c, p_c = pearsonr(c_pcts_filtered, bank2_neg_filtered)
    logger.info(f"\nC% → Bank2_neg: r={r_c:.3f}, p={p_c:.6f}")

    try:
        p_lin, _ = curve_fit(linear, c_pcts_filtered, bank2_neg_filtered)
        r2_lin = 1 - np.sum((bank2_neg_filtered - linear(c_pcts_filtered, *p_lin))**2) / np.sum((bank2_neg_filtered - bank2_neg_filtered.mean())**2)

        p_quad, _ = curve_fit(quadratic, c_pcts_filtered, bank2_neg_filtered)
        r2_quad = 1 - np.sum((bank2_neg_filtered - quadratic(c_pcts_filtered, *p_quad))**2) / np.sum((bank2_neg_filtered - bank2_neg_filtered.mean())**2)

        p_log, _ = curve_fit(logarithmic, c_pcts_filtered, bank2_neg_filtered)
        r2_log = 1 - np.sum((bank2_neg_filtered - logarithmic(c_pcts_filtered, *p_log))**2) / np.sum((bank2_neg_filtered - bank2_neg_filtered.mean())**2)

        results['GC_pathway']['C_to_Bank2_neg'] = {
            'correlation': {'r': float(r_c), 'p': float(p_c)},
            'linear': {'coeffs': p_lin.tolist(), 'r2': float(r2_lin)},
            'quadratic': {'coeffs': p_quad.tolist(), 'r2': float(r2_quad)},
            'logarithmic': {'coeffs': p_log.tolist(), 'r2': float(r2_log)},
            'best_model': 'quadratic' if r2_quad == max(r2_lin, r2_quad, r2_log) else ('logarithmic' if r2_log > r2_lin else 'linear'),
        }

        logger.info(f"  Linear:      R²={r2_lin:.4f}")
        logger.info(f"  Quadratic:   R²={r2_quad:.4f}")
        logger.info(f"  Logarithmic: R²={r2_log:.4f}")
    except Exception as e:
        logger.warning(f"Could not fit C% → Bank2_neg: {e}")

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nTransfer functions saved to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("TRANSFER FUNCTION SUMMARY")
    print("="*80)

    for pathway in ['AT_pathway', 'GC_pathway']:
        print(f"\n{pathway.upper()}:")
        for tf_name, tf_data in results[pathway].items():
            print(f"\n  {tf_name}:")
            print(f"    Correlation: r={tf_data['correlation']['r']:.3f}, p={tf_data['correlation']['p']:.6f}")
            print(f"    Best model: {tf_data['best_model']} (R²={tf_data[tf_data['best_model']]['r2']:.4f})")

    print("="*80)

    return results


if __name__ == '__main__':
    data_file = "genomevault/hdv_validation/hdc_experimentation/output/AT_GC_pathway_analysis.json"
    output_file = "genomevault/hdv_validation/hdc_experimentation/output/transfer_functions.json"

    results = build_transfer_functions(data_file, output_file)
