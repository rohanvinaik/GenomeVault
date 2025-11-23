#!/usr/bin/env python3
"""
Extract Motif Magnitude Profiles from Known Sequences (Barbie Method)

Encode known structural motifs and extract their actual bank magnitude profiles
to use as ground truth thresholds for motif detection.

Averages profiles across 5 variants of each motif for robust thresholds.

Author: Phase 1 Week 4
Date: November 21, 2025
"""

import numpy as np
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Architecture parameters (must match encoder)
D = 5_120
N = 1_024

# Biophysical bank definitions
PURINES = {'A', 'G'}
PYRIMIDINES = {'C', 'T'}


def generate_sparse_position_codebook(dimension: int, chunk_size: int, seed: int = 42) -> np.ndarray:
    """
    Generate sparse position codebook where each position has EXACTLY ONE ±1 element.

    This is locality-sensitive hashing: each position i → random dimension d_i with random sign.

    Args:
        dimension: Hypervector dimension (D=5120)
        chunk_size: Number of nucleotide positions (N=1024)
        seed: Random seed for reproducibility

    Returns:
        position_codebook: shape (chunk_size, dimension), sparse ternary {-1, 0, +1}
    """
    np.random.seed(seed)
    codebook = np.zeros((chunk_size, dimension), dtype=np.int8)

    for pos in range(chunk_size):
        # Each position activates exactly ONE random dimension
        dim_idx = np.random.randint(0, dimension)
        sign = np.random.choice([-1, +1])
        codebook[pos, dim_idx] = sign

    return codebook


def encode_3banks(sequence: str, position_codebook: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Encode sequence using 3-bank split architecture.

    Returns:
        dict with keys 'bank1', 'bank2', 'bank3' (each is int16 vector)
    """
    seq_upper = sequence.upper()

    # Initialize accumulation buffers (int16 to handle accumulation)
    bank1 = np.zeros(D, dtype=np.int16)  # Hydrophobic (T vs A)
    bank2 = np.zeros(D, dtype=np.int16)  # Major Groove (G vs C)
    bank3 = np.zeros(D, dtype=np.int16)  # Hinge (Y-R steps)

    # Encode each nucleotide
    for i, nuc in enumerate(seq_upper[:N]):  # Only use first N nucleotides
        if i >= len(position_codebook):
            break

        pos_vec = position_codebook[i]

        # Bank 1: Hydrophobic (A=-1, T=+1, GC=0)
        if nuc == 'A':
            bank1 -= pos_vec
        elif nuc == 'T':
            bank1 += pos_vec
        # G, C, N → transparent (no change)

        # Bank 2: Major Groove (C=-1, G=+1, AT=0)
        if nuc == 'C':
            bank2 -= pos_vec
        elif nuc == 'G':
            bank2 += pos_vec
        # A, T, N → transparent (no change)

        # Bank 3: Hinge (Y-R steps)
        # Compute hinge for dinucleotide steps
        if i > 0:
            prev_nuc = seq_upper[i-1]

            # Y→R transition (pyrimidine to purine): +1
            if prev_nuc in PYRIMIDINES and nuc in PURINES:
                bank3 += pos_vec
            # R→Y transition (purine to pyrimidine): -1
            elif prev_nuc in PURINES and nuc in PYRIMIDINES:
                bank3 -= pos_vec
            # Same category → transparent (no change)

    return {
        'bank1': bank1,
        'bank2': bank2,
        'bank3': bank3,
    }


def compute_bank_magnitudes(banks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute L2 magnitude of each bank."""
    return {
        'bank1_mag': float(np.linalg.norm(banks['bank1'])),
        'bank2_mag': float(np.linalg.norm(banks['bank2'])),
        'bank3_mag': float(np.linalg.norm(banks['bank3'])),
    }


def generate_motif_variants(motif_type: str, num_variants: int = 5):
    """Generate multiple variants of each motif type for robust averaging."""
    variants = []

    if motif_type == 'ALU':
        # ALU consensus with slight variations
        base = 'GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCACGAGGTCAGGAGATCGAGACCATCCCGGCTAAAACGGTGAAACCCCGTCTCTACTAAAAATACAAAAAATTAGCCGGGCGTAGTGGCGGGCGCCTGTAGTCCCAGCTACTTGGGAGGCTGAGGCAGGAGAATGGCGTGAACCCGGGAGGCGGAGCTTGCAGTGAGCCGAGATCCCGCCACTGCACTCCAGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA'

        for i in range(num_variants):
            # Repeat ALU consensus to fill 1024 bp
            full_seq = (base * 4)[:1024]
            variants.append(full_seq)

    elif motif_type == 'CPG':
        # CpG-rich sequences with varying patterns
        for i in range(num_variants):
            if i % 2 == 0:
                # Pure CG repeats
                variants.append('CG' * 512)
            else:
                # CG with some variation
                variants.append('CGCGCGCG' * 128)

    elif motif_type == 'POLYA':
        # Pure poly-A tails
        for i in range(num_variants):
            variants.append('A' * 1024)

    elif motif_type == 'TYPICAL':
        # Random DNA with ~50% GC
        for i in range(num_variants):
            np.random.seed(1000 + i)  # Use fixed seeds for reproducibility
            seq = ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=1024))
            variants.append(seq)

    return variants


def main():
    logger.info("="*80)
    logger.info("BARBIE METHOD: Extract Ground Truth Motif Magnitude Profiles")
    logger.info("="*80)
    logger.info("")
    logger.info(f"Architecture: D={D}, N={N}")
    logger.info(f"Encoding 5 variants of each motif and averaging profiles...")
    logger.info("")

    # Generate position codebook (same seed as encoder for consistency)
    position_codebook = generate_sparse_position_codebook(D, N, seed=42)

    motif_types = {
        'ALU_CONSENSUS': 'ALU',
        'CPG_ISLAND': 'CPG',
        'POLYA_TAIL': 'POLYA',
        'TYPICAL_DNA': 'TYPICAL',
    }

    results = {}

    for motif_name, motif_type in motif_types.items():
        logger.info(f"Processing {motif_name}...")

        # Generate 5 variants
        variants = generate_motif_variants(motif_type, num_variants=5)
        variant_profiles = []

        for i, sequence in enumerate(variants):
            # Encode sequence
            banks = encode_3banks(sequence, position_codebook)

            # Compute magnitudes
            mags = compute_bank_magnitudes(banks)
            variant_profiles.append(mags)

            logger.info(f"  Variant {i+1}: bank1={mags['bank1_mag']:.2f}, "
                       f"bank2={mags['bank2_mag']:.2f}, bank3={mags['bank3_mag']:.2f}")

        # Average the profiles
        avg_profile = {
            'name': motif_name,
            'bank1_mag': np.mean([p['bank1_mag'] for p in variant_profiles]),
            'bank2_mag': np.mean([p['bank2_mag'] for p in variant_profiles]),
            'bank3_mag': np.mean([p['bank3_mag'] for p in variant_profiles]),
            'bank1_std': np.std([p['bank1_mag'] for p in variant_profiles]),
            'bank2_std': np.std([p['bank2_mag'] for p in variant_profiles]),
            'bank3_std': np.std([p['bank3_mag'] for p in variant_profiles]),
        }

        results[motif_name] = avg_profile

        logger.info(f"  ✓ Average: bank1={avg_profile['bank1_mag']:.2f} ± {avg_profile['bank1_std']:.2f}, "
                   f"bank2={avg_profile['bank2_mag']:.2f} ± {avg_profile['bank2_std']:.2f}, "
                   f"bank3={avg_profile['bank3_mag']:.2f} ± {avg_profile['bank3_std']:.2f}")
        logger.info("")

    # Print summary
    logger.info("="*80)
    logger.info("GROUND TRUTH MAGNITUDE PROFILES (AVERAGED ACROSS 5 VARIANTS)")
    logger.info("="*80)
    logger.info("")

    for motif_name, profile in results.items():
        logger.info(f"{motif_name}:")
        logger.info(f"  bank1: {profile['bank1_mag']:.2f} ± {profile['bank1_std']:.2f}")
        logger.info(f"  bank2: {profile['bank2_mag']:.2f} ± {profile['bank2_std']:.2f}")
        logger.info(f"  bank3: {profile['bank3_mag']:.2f} ± {profile['bank3_std']:.2f}")
        logger.info("")

    # Recommend thresholds
    logger.info("="*80)
    logger.info("RECOMMENDED THRESHOLDS FOR build_motif_index.py")
    logger.info("="*80)
    logger.info("")

    # GC-rich motifs (ALU / CpG)
    gc_bank1 = max(results['ALU_CONSENSUS']['bank1_mag'], results['CPG_ISLAND']['bank1_mag'])
    gc_bank2 = min(results['ALU_CONSENSUS']['bank2_mag'], results['CPG_ISLAND']['bank2_mag'])
    gc_bank3 = min(results['ALU_CONSENSUS']['bank3_mag'], results['CPG_ISLAND']['bank3_mag'])

    logger.info("GC-RICH MOTIFS (ALU / CpG):")
    logger.info(f"  bank1 (suppressed): <= {gc_bank1:.2f}")
    logger.info(f"  bank2 (elevated):   >= {gc_bank2:.2f}")
    logger.info(f"  bank3 (elevated):   >= {gc_bank3:.2f}")
    logger.info("")

    # AT-rich motifs (poly-A)
    at_bank1 = results['POLYA_TAIL']['bank1_mag']
    at_bank2 = results['POLYA_TAIL']['bank2_mag']
    at_bank3 = results['POLYA_TAIL']['bank3_mag']

    logger.info("AT-RICH MOTIFS (poly-A):")
    logger.info(f"  bank1 (elevated):   >= {at_bank1:.2f}")
    logger.info(f"  bank2 (suppressed): <= {at_bank2:.2f}")
    logger.info(f"  bank3 (suppressed): <= {at_bank3:.2f}")
    logger.info("")

    logger.info("="*80)


if __name__ == '__main__':
    main()
