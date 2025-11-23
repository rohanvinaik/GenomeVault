#!/usr/bin/env python3
"""
Extract Real Motif Magnitude Profiles from Actual Genome (Barbie Method - CORRECTED)

Instead of encoding synthetic consensus sequences, this extracts REAL motifs from
the actual encoded genome and measures their magnitude profiles.

Author: Phase 1 Week 4
Date: November 21, 2025
"""

import h5py
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_real_motifs_by_structure(h5_path: str, num_samples: int = 5) -> Dict[str, List[int]]:
    """
    Find real motif locations in encoded genome using structural fingerprints.

    Strategy: Scan for chunks with extreme structural properties:
    - GC-rich (ALU/CpG): High density + bank2/bank3 elevated
    - AT-rich (poly-A): Low density + bank1 elevated

    Returns:
        dict mapping motif_type -> list of chunk indices
    """
    logger.info("Scanning genome for real structural motifs...")

    motifs = {
        'ALU_REAL': [],
        'CPG_REAL': [],
        'POLYA_REAL': [],
    }

    with h5py.File(h5_path, 'r') as f:
        all_banks = f['all_bank_vectors']
        total_chunks = all_banks.shape[0]

        logger.info(f"Scanning {total_chunks:,} chunks...")

        for chunk_idx in range(0, total_chunks, 100):  # Sample every 100th chunk for speed
            if len(motifs['ALU_REAL']) >= num_samples and \
               len(motifs['CPG_REAL']) >= num_samples and \
               len(motifs['POLYA_REAL']) >= num_samples:
                break

            chunk = all_banks[chunk_idx, :, :]

            # Compute structural fingerprint
            total_zeros = np.sum(chunk == 0)
            density = 1 - (total_zeros / (3 * 5120))

            bank1_mag = np.linalg.norm(chunk[0, :])
            bank2_mag = np.linalg.norm(chunk[1, :])
            bank3_mag = np.linalg.norm(chunk[2, :])

            total_mag = bank1_mag + bank2_mag + bank3_mag
            if total_mag == 0:
                continue

            bank1_ratio = bank1_mag / total_mag
            bank2_ratio = bank2_mag / total_mag
            bank3_ratio = bank3_mag / total_mag

            # GC-RICH (ALU): High density + bank2 dominant
            if density >= 0.98 and bank2_ratio > 0.38 and len(motifs['ALU_REAL']) < num_samples:
                motifs['ALU_REAL'].append(chunk_idx)
                logger.info(f"  Found ALU at chunk {chunk_idx}: density={density:.3f}, bank2_ratio={bank2_ratio:.3f}")

            # GC-RICH (CpG): Very high density + bank2/bank3 both elevated
            elif density >= 0.95 and bank2_ratio > 0.35 and bank3_ratio > 0.35 and len(motifs['CPG_REAL']) < num_samples:
                motifs['CPG_REAL'].append(chunk_idx)
                logger.info(f"  Found CpG at chunk {chunk_idx}: density={density:.3f}, bank2={bank2_ratio:.3f}, bank3={bank3_ratio:.3f}")

            # AT-RICH (poly-A): Low density + bank1 dominant
            elif density < 0.20 and bank1_ratio > 0.60 and len(motifs['POLYA_REAL']) < num_samples:
                motifs['POLYA_REAL'].append(chunk_idx)
                logger.info(f"  Found poly-A at chunk {chunk_idx}: density={density:.3f}, bank1_ratio={bank1_ratio:.3f}")

    return motifs


def compute_magnitude_profiles(h5_path: str, motif_indices: Dict[str, List[int]]) -> Dict[str, Dict]:
    """
    Compute magnitude profiles for real motifs.

    Returns:
        dict mapping motif_type -> {bank1_mag, bank2_mag, bank3_mag, stats}
    """
    logger.info("\n" + "="*80)
    logger.info("COMPUTING MAGNITUDE PROFILES FROM REAL GENOME MOTIFS")
    logger.info("="*80)
    logger.info("")

    results = {}

    with h5py.File(h5_path, 'r') as f:
        all_banks = f['all_bank_vectors']

        for motif_type, chunk_indices in motif_indices.items():
            if not chunk_indices:
                logger.warning(f"No {motif_type} samples found, skipping...")
                continue

            logger.info(f"Processing {motif_type} ({len(chunk_indices)} samples)...")

            bank1_mags = []
            bank2_mags = []
            bank3_mags = []

            for i, chunk_idx in enumerate(chunk_indices):
                chunk = all_banks[chunk_idx, :, :]

                bank1_mag = float(np.linalg.norm(chunk[0, :]))
                bank2_mag = float(np.linalg.norm(chunk[1, :]))
                bank3_mag = float(np.linalg.norm(chunk[2, :]))

                bank1_mags.append(bank1_mag)
                bank2_mags.append(bank2_mag)
                bank3_mags.append(bank3_mag)

                logger.info(f"  Sample {i+1}: bank1={bank1_mag:.2f}, bank2={bank2_mag:.2f}, bank3={bank3_mag:.2f}")

            # Compute statistics
            results[motif_type] = {
                'bank1_mag': float(np.mean(bank1_mags)),
                'bank2_mag': float(np.mean(bank2_mags)),
                'bank3_mag': float(np.mean(bank3_mags)),
                'bank1_std': float(np.std(bank1_mags)),
                'bank2_std': float(np.std(bank2_mags)),
                'bank3_std': float(np.std(bank3_mags)),
                'num_samples': len(chunk_indices),
                'chunk_indices': chunk_indices,
            }

            logger.info(f"  ✓ Average: bank1={results[motif_type]['bank1_mag']:.2f} ± {results[motif_type]['bank1_std']:.2f}, "
                       f"bank2={results[motif_type]['bank2_mag']:.2f} ± {results[motif_type]['bank2_std']:.2f}, "
                       f"bank3={results[motif_type]['bank3_mag']:.2f} ± {results[motif_type]['bank3_std']:.2f}")
            logger.info("")

    return results


def print_threshold_recommendations(results: Dict[str, Dict]):
    """Print recommended thresholds for motif indexing."""
    logger.info("="*80)
    logger.info("GROUND TRUTH MAGNITUDE PROFILES (FROM REAL GENOME MOTIFS)")
    logger.info("="*80)
    logger.info("")

    for motif_type, profile in results.items():
        logger.info(f"{motif_type}:")
        logger.info(f"  bank1: {profile['bank1_mag']:.2f} ± {profile['bank1_std']:.2f}")
        logger.info(f"  bank2: {profile['bank2_mag']:.2f} ± {profile['bank2_std']:.2f}")
        logger.info(f"  bank3: {profile['bank3_mag']:.2f} ± {profile['bank3_std']:.2f}")
        logger.info(f"  samples: {profile['num_samples']}")
        logger.info("")

    logger.info("="*80)
    logger.info("RECOMMENDED THRESHOLDS FOR build_motif_index.py")
    logger.info("="*80)
    logger.info("")

    if 'ALU_REAL' in results and 'CPG_REAL' in results:
        # Use the weaker GC signal (ALU) for conservative thresholds
        alu = results['ALU_REAL']
        cpg = results['CPG_REAL']

        gc_bank1 = max(alu['bank1_mag'], cpg['bank1_mag'])
        gc_bank2 = min(alu['bank2_mag'], cpg['bank2_mag'])
        gc_bank3 = min(alu['bank3_mag'], cpg['bank3_mag'])

        logger.info("GC-RICH MOTIFS (ALU / CpG):")
        logger.info(f"  bank1 (suppressed): <= {gc_bank1:.2f}")
        logger.info(f"  bank2 (elevated):   >= {gc_bank2:.2f}")
        logger.info(f"  bank3 (elevated):   >= {gc_bank3:.2f}")
        logger.info("")

    if 'POLYA_REAL' in results:
        polya = results['POLYA_REAL']

        logger.info("AT-RICH MOTIFS (poly-A):")
        logger.info(f"  bank1 (elevated):   >= {polya['bank1_mag']:.2f}")
        logger.info(f"  bank2 (suppressed): <= {polya['bank2_mag']:.2f}")
        logger.info(f"  bank3 (suppressed): <= {polya['bank3_mag']:.2f}")
        logger.info("")

    logger.info("="*80)


def main():
    h5_path = "genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_3banks.h5"

    if not Path(h5_path).exists():
        logger.error(f"ERROR: Encoded genome not found at {h5_path}")
        exit(1)

    # Step 1: Find real motifs in genome
    motif_indices = find_real_motifs_by_structure(h5_path, num_samples=5)

    # Step 2: Compute magnitude profiles
    results = compute_magnitude_profiles(h5_path, motif_indices)

    # Step 3: Print recommendations
    print_threshold_recommendations(results)

    # Save results
    output_file = "genomevault/hdv_validation/hdc_experimentation/output/real_motif_profiles.json"
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
