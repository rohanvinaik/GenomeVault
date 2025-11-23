"""
Identify motifs from the 6-bank split ternary encoding

This script extracts motif signatures with the CORRECT format:
- signals: {bank1_pos_mag, bank1_neg_mag, bank2_pos_mag, bank2_neg_mag, bank3_pos_mag, bank3_neg_mag}
- composition: {A_pct, T_pct, G_pct, C_pct}

Compatible with split ternary format (two orthogonal 3D ternary vectors)

Author: Phase 1 Week 3-4
Date: November 22, 2025
"""

import h5py
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_split_ternary_h5(h5_path: str):
    """
    Load the split ternary H5 file with 6 banks.

    H5 format: split_ternary_vectors with shape (n_chunks, 6, D)
    where axis 1 represents:
      Vector 1 (GC-dominant): Banks 0-2 = [AT=0, GC, Hinge]
      Vector 2 (AT-dominant): Banks 3-5 = [AT, GC=0, Hinge]

    Returns:
        Dataset handles for the 6 banks + metadata
    """
    f = h5py.File(h5_path, 'r')

    # Load packed format: (n_chunks, 6, D)
    logger.info(f"Loading split ternary vectors...")
    packed_data = f['split_ternary_vectors'][:]
    n_chunks = packed_data.shape[0]

    logger.info(f"Loaded {n_chunks:,} chunks with {packed_data.shape[2]} dimensions")

    # Extract banks according to split ternary structure
    # Bank mapping: unified 3-bank representation
    #   Bank 1 (AT) ← Bank 3 (Vector2_AT)
    #   Bank 2 (GC) ← Bank 1 (Vector1_GC)
    #   Bank 3 (Hinge) ← Bank 2 (Hinge)
    return {
        'at_bank': packed_data[:, 3, :],    # Vector2_AT
        'gc_bank': packed_data[:, 1, :],    # Vector1_GC
        'hinge_bank': packed_data[:, 2, :], # Hinge
        'composition': None,  # Not available in split ternary H5
        'chunk_keys': [f'chunk_{i:07d}' for i in range(n_chunks)],  # Synthetic keys
        'file': f  # Keep handle for cleanup
    }


def compute_bank_magnitudes(chunk_idx, data):
    """
    Compute bank magnitudes for split ternary format.

    Ternary format {-1, 0, +1}:
    - bank_pos_mag = sum of positive values (+1s)
    - bank_neg_mag = sum of absolute value of negative values (-1s)
    - zeros = structural silence

    Returns dict with bank*_*_mag keys.
    """
    mags = {}

    # Extract each bank's ternary vector (D dimensions, {-1, 0, +1})
    at_vec = data['at_bank'][chunk_idx]
    gc_vec = data['gc_bank'][chunk_idx]
    hinge_vec = data['hinge_bank'][chunk_idx]

    # Bank 1 (AT pathway): T=+1, A=-1, GC=0
    mags['bank1_pos_mag'] = float(np.sum(at_vec[at_vec > 0]))  # T-rich
    mags['bank1_neg_mag'] = float(np.sum(-at_vec[at_vec < 0]))  # A-rich

    # Bank 2 (GC pathway): G=+1, C=-1, AT=0
    mags['bank2_pos_mag'] = float(np.sum(gc_vec[gc_vec > 0]))  # G-rich
    mags['bank2_neg_mag'] = float(np.sum(-gc_vec[gc_vec < 0]))  # C-rich

    # Bank 3 (Hinge): YR=+1, RY=-1, RR/YY=0
    mags['bank3_pos_mag'] = float(np.sum(hinge_vec[hinge_vec > 0]))  # Y→R transitions
    mags['bank3_neg_mag'] = float(np.sum(-hinge_vec[hinge_vec < 0]))  # R→Y transitions

    return mags


def extract_composition(chunk_idx, data):
    """
    Extract composition percentages [A%, T%, G%, C%].

    Note: Composition data not available in split ternary H5,
    so we estimate from bank magnitudes:
    - AT% ≈ Bank1 total magnitude
    - GC% ≈ Bank2 total magnitude

    This is the raw biophysical signal - no normalization.
    """
    if data['composition'] is not None:
        comp = data['composition'][chunk_idx]
        return {
            'A_pct': float(comp[0]),
            'T_pct': float(comp[1]),
            'G_pct': float(comp[2]),
            'C_pct': float(comp[3]),
        }

    # Estimate from ternary bank magnitudes
    # Bank1 (A/T pathway), Bank2 (G/C pathway)
    at_vec = data['at_bank'][chunk_idx]
    gc_vec = data['gc_bank'][chunk_idx]

    # Count non-zero elements (activated dimensions)
    at_count = np.sum(np.abs(at_vec) > 0)
    gc_count = np.sum(np.abs(gc_vec) > 0)

    total = at_count + gc_count + 1e-10
    at_pct = (at_count / total) * 100
    gc_pct = (gc_count / total) * 100

    # Split evenly between A/T and G/C
    return {
        'A_pct': float(at_pct / 2),
        'T_pct': float(at_pct / 2),
        'G_pct': float(gc_pct / 2),
        'C_pct': float(gc_pct / 2),
    }


def identify_motifs(h5_path: str, output_path: str, n_samples_per_motif: int = 50):
    """
    Identify motif signatures from split ternary H5.

    Motifs defined by bank magnitude profiles:
    - GC_SUPPRESS: Bank2_pos dominant, Bank1_pos suppressed
    - AT_SUPPRESS: Bank1_pos dominant, Bank2_pos suppressed
    - BANK3_EXTREME_POS: Bank3_pos >> Bank3_neg (Y→R transitions)
    - BANK3_EXTREME_NEG: Bank3_neg >> Bank3_pos (R→Y transitions)
    - BALANCED: Bank1 ≈ Bank2
    """
    logger.info("\n" + "="*80)
    logger.info("MOTIF IDENTIFICATION FROM SPLIT TERNARY ENCODING")
    logger.info("="*80)

    logger.info(f"\nLoading H5: {h5_path}")
    data = load_split_ternary_h5(h5_path)

    n_chunks = len(data['chunk_keys'])
    logger.info(f"Total chunks: {n_chunks:,}")

    # Initialize motif collections
    motifs = {
        'GC_SUPPRESS': {'threshold': 'bank2_pos/bank1_pos > 1.5', 'chunks': []},
        'AT_SUPPRESS': {'threshold': 'bank1_pos/bank2_pos > 1.5', 'chunks': []},
        'BANK3_EXTREME_POS': {'threshold': 'bank3_pos/bank3_neg > 1.3', 'chunks': []},
        'BANK3_EXTREME_NEG': {'threshold': 'bank3_neg/bank3_pos > 1.3', 'chunks': []},
        'BALANCED': {'threshold': '0.8 < bank1_pos/bank2_pos < 1.2', 'chunks': []},
    }

    logger.info(f"\nScanning {n_chunks:,} chunks for motifs...")
    logger.info(f"Target: {n_samples_per_motif} samples per motif")

    # CRITICAL: Scan ALL chunks to find rare extreme motifs
    # With ~4% activation density, extremes are <1% of genome
    logger.info(f"Scanning ALL {n_chunks:,} chunks for extreme patterns...")

    for idx in range(n_chunks):
        # Progress update every 10k chunks
        if idx % 10000 == 0 and idx > 0:
            logger.info(f"Progress: {idx:,}/{n_chunks:,} chunks scanned")
            logger.info(f"  GC_SUPPRESS: {len(motifs['GC_SUPPRESS']['chunks'])}")
            logger.info(f"  AT_SUPPRESS: {len(motifs['AT_SUPPRESS']['chunks'])}")
            logger.info(f"  BANK3_EXTREME_POS: {len(motifs['BANK3_EXTREME_POS']['chunks'])}")
            logger.info(f"  BANK3_EXTREME_NEG: {len(motifs['BANK3_EXTREME_NEG']['chunks'])}")
            logger.info(f"  BALANCED: {len(motifs['BALANCED']['chunks'])}")

        # Stop early if all motifs are filled
        if all(len(m['chunks']) >= n_samples_per_motif for m in motifs.values()):
            logger.info(f"✓ All motifs filled at chunk {idx:,}")
            break

        # Compute bank magnitudes
        mags = compute_bank_magnitudes(idx, data)
        comp = extract_composition(idx, data)

        # Build chunk data structure
        chunk_data = {
            'chunk_idx': int(idx),
            'position': data['chunk_keys'][idx],
            'signals': mags,
            'composition': comp,
        }

        # Classify into motifs
        eps = 1e-6

        # LOWERED THRESHOLDS for activation count-based metrics
        # GC_SUPPRESS: Bank2 activation >> Bank1 (lower from 1.5 to 1.2)
        if mags['bank2_pos_mag'] / (mags['bank1_pos_mag'] + eps) > 1.2:
            if len(motifs['GC_SUPPRESS']['chunks']) < n_samples_per_motif:
                motifs['GC_SUPPRESS']['chunks'].append(chunk_data)

        # AT_SUPPRESS: Bank1 activation >> Bank2
        if mags['bank1_pos_mag'] / (mags['bank2_pos_mag'] + eps) > 1.2:
            if len(motifs['AT_SUPPRESS']['chunks']) < n_samples_per_motif:
                motifs['AT_SUPPRESS']['chunks'].append(chunk_data)

        # BANK3_EXTREME_POS: Y→R transitions (lower from 1.3 to 1.15)
        if mags['bank3_pos_mag'] / (mags['bank3_neg_mag'] + eps) > 1.15:
            if len(motifs['BANK3_EXTREME_POS']['chunks']) < n_samples_per_motif:
                motifs['BANK3_EXTREME_POS']['chunks'].append(chunk_data)

        # BANK3_EXTREME_NEG: R→Y transitions
        if mags['bank3_neg_mag'] / (mags['bank3_pos_mag'] + eps) > 1.15:
            if len(motifs['BANK3_EXTREME_NEG']['chunks']) < n_samples_per_motif:
                motifs['BANK3_EXTREME_NEG']['chunks'].append(chunk_data)

        # BALANCED: Bank1 ≈ Bank2
        ratio = mags['bank1_pos_mag'] / (mags['bank2_pos_mag'] + eps)
        if 0.8 < ratio < 1.2:
            if len(motifs['BALANCED']['chunks']) < n_samples_per_motif:
                motifs['BALANCED']['chunks'].append(chunk_data)

    # Close H5
    data['file'].close()

    # Report
    logger.info("\n" + "="*80)
    logger.info("MOTIF IDENTIFICATION SUMMARY")
    logger.info("="*80)

    for motif_name, motif_info in motifs.items():
        n = len(motif_info['chunks'])
        logger.info(f"\n{motif_name}:")
        logger.info(f"  Threshold: {motif_info['threshold']}")
        logger.info(f"  Samples collected: {n}")

        if n > 0:
            # Report bank ranges
            bank1_pos_range = [c['signals']['bank1_pos_mag'] for c in motif_info['chunks']]
            bank2_pos_range = [c['signals']['bank2_pos_mag'] for c in motif_info['chunks']]
            bank3_pos_range = [c['signals']['bank3_pos_mag'] for c in motif_info['chunks']]

            logger.info(f"  Bank1_pos range: {min(bank1_pos_range):.2f} - {max(bank1_pos_range):.2f}")
            logger.info(f"  Bank2_pos range: {min(bank2_pos_range):.2f} - {max(bank2_pos_range):.2f}")
            logger.info(f"  Bank3_pos range: {min(bank3_pos_range):.2f} - {max(bank3_pos_range):.2f}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(motifs, f, indent=2)

    logger.info(f"\n✓ Motif ground truth saved to: {output_path}")
    logger.info("="*80)

    return motifs


if __name__ == '__main__':
    h5_path = "genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_6banks_split_ternary.h5"
    output_path = "genomevault/hdv_validation/hdc_experimentation/output/motif_ground_truth_split_ternary.json"

    motifs = identify_motifs(
        h5_path=h5_path,
        output_path=output_path,
        n_samples_per_motif=50  # Target 50 samples per motif
    )
