#!/usr/bin/env python3
"""
Analyze correlations between density, bank magnitudes, and nucleotide composition
for the extreme chunks found by find_extreme_motifs.py.

This analyzes the METRICS directly without needing sequence data.

Author: Phase 1 Week 3 - Experiment 6 Completion
Date: November 22, 2025
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

def analyze_correlations(extreme_chunks_file: str, output_dir: str):
    """
    Analyze correlations between metrics for extreme chunks.

    Args:
        extreme_chunks_file: Path to JSON with extreme chunk data from find_extreme_motifs.py
        output_dir: Where to save analysis results
    """
    # Load extreme chunks data
    with open(extreme_chunks_file, 'r') as f:
        extreme_data = json.load(f)

    results = {}

    for category in ['GC_RICH', 'AT_RICH', 'BALANCED']:
        print(f"\n{'='*80}")
        print(f"Analyzing {category} chunks (n={len(extreme_data[category]['chunk_indices'])})")
        print('='*80)

        # Extract metrics
        metrics = extreme_data[category]['metrics']

        if len(metrics) == 0:
            print(f"No chunks in {category} category")
            continue

        # Convert to arrays
        densities = np.array([m['density'] for m in metrics])
        bank1_mags = np.array([m['bank1_mag'] for m in metrics])
        bank2_mags = np.array([m['bank2_mag'] for m in metrics])
        bank3_mags = np.array([m['bank3_mag'] for m in metrics])
        ratios = np.array([m['ratio'] for m in metrics])

        # Calculate statistics
        print(f"\nDensity statistics:")
        print(f"  Mean: {np.mean(densities):.4f} ± {np.std(densities):.4f}")
        print(f"  Range: [{np.min(densities):.4f}, {np.max(densities):.4f}]")

        print(f"\nBank magnitudes:")
        print(f"  Bank1 (AT): {np.mean(bank1_mags):.2f} ± {np.std(bank1_mags):.2f}")
        print(f"  Bank2 (GC): {np.mean(bank2_mags):.2f} ± {np.std(bank2_mags):.2f}")
        print(f"  Bank3 (Hinge): {np.mean(bank3_mags):.2f} ± {np.std(bank3_mags):.2f}")
        print(f"  Ratio (Bank2/Bank1): {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")

        # Correlation analysis
        print(f"\nCorrelations (Pearson r, p-value):")

        # Density vs bank magnitudes
        r_dens_b1, p_dens_b1 = pearsonr(densities, bank1_mags)
        r_dens_b2, p_dens_b2 = pearsonr(densities, bank2_mags)
        r_dens_b3, p_dens_b3 = pearsonr(densities, bank3_mags)

        print(f"  Density vs Bank1: r={r_dens_b1:.3f}, p={p_dens_b1:.4f}")
        print(f"  Density vs Bank2: r={r_dens_b2:.3f}, p={p_dens_b2:.4f}")
        print(f"  Density vs Bank3: r={r_dens_b3:.3f}, p={p_dens_b3:.4f}")

        # Bank ratios
        r_b1_b2, p_b1_b2 = pearsonr(bank1_mags, bank2_mags)
        r_b2_b3, p_b2_b3 = pearsonr(bank2_mags, bank3_mags)
        r_b1_b3, p_b1_b3 = pearsonr(bank1_mags, bank3_mags)

        print(f"  Bank1 vs Bank2: r={r_b1_b2:.3f}, p={p_b1_b2:.4f}")
        print(f"  Bank2 vs Bank3: r={r_b2_b3:.3f}, p={p_b2_b3:.4f}")
        print(f"  Bank1 vs Bank3: r={r_b1_b3:.3f}, p={p_b1_b3:.4f}")

        # Fit regression models
        print(f"\nRegression models (density vs signal strength):")

        # Linear model: bank_mag = a * density + b
        from scipy.optimize import curve_fit

        def linear(x, a, b):
            return a * x + b

        def exponential(x, a, b, c):
            return a * np.exp(b * x) + c

        def power_law(x, a, b, c):
            return a * np.power(x + 1e-10, b) + c

        # Fit Bank2 (GC signal) vs density
        try:
            popt_lin, _ = curve_fit(linear, densities, bank2_mags)
            popt_exp, _ = curve_fit(exponential, densities, bank2_mags, p0=[1, 1, 20], maxfev=10000)
            popt_pow, _ = curve_fit(power_law, densities, bank2_mags, p0=[1, 0.5, 20], maxfev=10000)

            # Calculate R²
            residuals_lin = bank2_mags - linear(densities, *popt_lin)
            residuals_exp = bank2_mags - exponential(densities, *popt_exp)
            residuals_pow = bank2_mags - power_law(densities, *popt_pow)

            ss_tot = np.sum((bank2_mags - np.mean(bank2_mags))**2)
            r2_lin = 1 - np.sum(residuals_lin**2) / ss_tot
            r2_exp = 1 - np.sum(residuals_exp**2) / ss_tot
            r2_pow = 1 - np.sum(residuals_pow**2) / ss_tot

            print(f"  Linear: bank2 = {popt_lin[0]:.2f} * density + {popt_lin[1]:.2f} (R²={r2_lin:.4f})")
            print(f"  Exponential: bank2 = {popt_exp[0]:.2f} * exp({popt_exp[1]:.2f} * density) + {popt_exp[2]:.2f} (R²={r2_exp:.4f})")
            print(f"  Power law: bank2 = {popt_exp[0]:.2f} * density^{popt_pow[1]:.2f} + {popt_pow[2]:.2f} (R²={r2_pow:.4f})")
        except Exception as e:
            print(f"  Could not fit models: {e}")

        # Store results
        results[category] = {
            'n_chunks': len(metrics),
            'density': {
                'mean': float(np.mean(densities)),
                'std': float(np.std(densities)),
                'min': float(np.min(densities)),
                'max': float(np.max(densities)),
            },
            'bank_magnitudes': {
                'bank1_mean': float(np.mean(bank1_mags)),
                'bank2_mean': float(np.mean(bank2_mags)),
                'bank3_mean': float(np.mean(bank3_mags)),
            },
            'correlations': {
                'density_vs_bank1': {'r': float(r_dens_b1), 'p': float(p_dens_b1)},
                'density_vs_bank2': {'r': float(r_dens_b2), 'p': float(p_dens_b2)},
                'density_vs_bank3': {'r': float(r_dens_b3), 'p': float(p_dens_b3)},
            },
        }

    # Save results
    output_path = Path(output_dir) / "motif_correlation_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Analysis saved to {output_path}")
    print('='*80)


if __name__ == '__main__':
    extreme_chunks_file = "/tmp/extreme_motifs_n50.json"
    output_dir = "genomevault/hdv_validation/hdc_experimentation/output"

    analyze_correlations(extreme_chunks_file, output_dir)
