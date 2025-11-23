"""
Create Biologically Accurate Lens Definitions for Structural Motifs

ALU Structure:
- Repeated segments (GC-rich, ~130 bp each)
- Linked by poly-A tail (variable length, 5-50 bp)
- Total length: ~280-300 bp

This script:
1. Encodes real ALU sequences into 3-bank ternary representation
2. Creates lens patterns for both whole and split ALU structures
3. Tests lenses against known ALU positions

Author: Phase 1 Week 3 - Lens System Debugging
Date: November 21, 2025
"""

import sys
from pathlib import Path
import h5py
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from query.lens_aware_simd_query_engine import (
    generate_position_codebook,
    LensLibrary,
    cosine_similarity
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_alu_structure(alu_sequence: str):
    """
    Analyze the structure of an ALU repeat to identify segments and poly-A tail.

    Args:
        alu_sequence: Example ALU sequence

    Returns:
        dict with segment locations, poly-A tail position, composition
    """
    # Count GC content
    gc_count = alu_sequence.count('G') + alu_sequence.count('C')
    at_count = alu_sequence.count('A') + alu_sequence.count('T')

    gc_percent = 100 * gc_count / len(alu_sequence)

    # Look for poly-A stretches (A repeated 5+ times)
    polya_positions = []
    i = 0
    while i < len(alu_sequence) - 4:
        if alu_sequence[i:i+5] == 'AAAAA':
            start = i
            while i < len(alu_sequence) and alu_sequence[i] == 'A':
                i += 1
            polya_positions.append((start, i))
        else:
            i += 1

    # Analyze dinucleotide steps for hinge activity (YR/RY transitions)
    purines = set('AG')
    pyrimidines = set('CT')

    yr_transitions = 0
    ry_transitions = 0

    for i in range(len(alu_sequence) - 1):
        b1, b2 = alu_sequence[i], alu_sequence[i+1]
        if b1 in pyrimidines and b2 in purines:
            yr_transitions += 1
        elif b1 in purines and b2 in pyrimidines:
            ry_transitions += 1

    total_transitions = yr_transitions + ry_transitions
    transition_percent = 100 * total_transitions / (len(alu_sequence) - 1)

    return {
        'length': len(alu_sequence),
        'gc_percent': gc_percent,
        'at_percent': 100 - gc_percent,
        'polya_positions': polya_positions,
        'yr_transitions': yr_transitions,
        'ry_transitions': ry_transitions,
        'transition_percent': transition_percent
    }


def encode_sequence_to_banks(sequence: str, position_codebook: np.ndarray, D: int):
    """
    Encode a DNA sequence into 3-bank ternary representation.

    Uses the same biophysical encoding as the main encoder:
    - Bank 1: Hydrophobic (A/T vs G/C)
    - Bank 2: Major Groove (G/C pair strength)
    - Bank 3: Hinge (YR/RY flexibility)

    Args:
        sequence: DNA sequence (ACGT)
        position_codebook: Position codebook (N x D)
        D: Hypervector dimension

    Returns:
        dict with bank1, bank2, bank3 vectors
    """
    N = len(position_codebook)

    # Nucleotide base vectors (Bank 1: Hydrophobic)
    base_hydrophobic = {
        'A': -1,  # Hydrophobic
        'T': -1,  # Hydrophobic
        'G': +1,  # Hydrophilic
        'C': +1,  # Hydrophilic
    }

    # Bank 2: Major Groove (GC pair strength)
    base_major_groove = {
        'G': +1,  # Strong major groove
        'C': +1,  # Strong major groove
        'A': -1,  # Weak major groove
        'T': -1,  # Weak major groove
    }

    # Bank 3: Hinge (YR/RY transitions)
    # Y = pyrimidine (C/T), R = purine (A/G)
    purines = set('AG')
    pyrimidines = set('CT')

    # Initialize banks
    bank1 = np.zeros(D, dtype=np.int8)
    bank2 = np.zeros(D, dtype=np.int8)
    bank3 = np.zeros(D, dtype=np.int8)

    for i, nucleotide in enumerate(sequence):
        if i >= N:
            break  # Sequence longer than position codebook

        pos_vec = position_codebook[i, :]

        # Bank 1: Hydrophobic
        contribution1 = base_hydrophobic.get(nucleotide, 0) * pos_vec
        bank1 += contribution1.astype(np.int8)

        # Bank 2: Major Groove
        contribution2 = base_major_groove.get(nucleotide, 0) * pos_vec
        bank2 += contribution2.astype(np.int8)

        # Bank 3: Hinge (YR/RY transitions)
        if i > 0:
            prev_nuc = sequence[i-1]

            # YR transition (pyrimidine → purine)
            if prev_nuc in pyrimidines and nucleotide in purines:
                bank3 += pos_vec.astype(np.int8)
            # RY transition (purine → pyrimidine)
            elif prev_nuc in purines and nucleotide in pyrimidines:
                bank3 -= pos_vec.astype(np.int8)

    # Clip to ternary
    bank1 = np.clip(bank1, -1, 1).astype(np.int8)
    bank2 = np.clip(bank2, -1, 1).astype(np.int8)
    bank3 = np.clip(bank3, -1, 1).astype(np.int8)

    return {
        'bank1': bank1,
        'bank2': bank2,
        'bank3': bank3
    }


def main():
    print("=" * 80)
    print("CREATING BIOLOGICALLY ACCURATE LENS DEFINITIONS")
    print("=" * 80)
    print()

    # Example ALU sequence from user
    alu_example = "GCCGGGCGCGGTGGCGCGTGCCTGTAGTCCCAGCTACTCGGGAGGCTGAGGCTGGAGGATCGCTTGAGTCCAGGAGTTCTGGGCTGTAGTGCGCTATGCCGATCGGAATAGCCACTGCACTCCAGCCTGGGCAACATAGCGAGACCCCGTCTC"

    print("Analyzing example ALU sequence:")
    print(f"  Sequence: {alu_example}")
    print(f"  Length: {len(alu_example)} bp")
    print()

    analysis = analyze_alu_structure(alu_example)

    print("Structure Analysis:")
    print(f"  GC content: {analysis['gc_percent']:.1f}%")
    print(f"  AT content: {analysis['at_percent']:.1f}%")
    print(f"  YR transitions: {analysis['yr_transitions']}")
    print(f"  RY transitions: {analysis['ry_transitions']}")
    print(f"  Total transitions: {analysis['transition_percent']:.1f}%")
    print(f"  Poly-A stretches: {analysis['polya_positions']}")
    print()

    # Generate position codebook
    N = 1024  # Chunk size
    D = 5120  # Dimension

    logger.info(f"Generating position codebook (N={N}, D={D})...")
    position_codebook = generate_position_codebook(N, D, seed=42)

    # Encode the ALU sequence
    logger.info("Encoding ALU sequence into 3-bank representation...")
    alu_banks = encode_sequence_to_banks(alu_example, position_codebook, D)

    # Analyze the encoded representation
    print("Encoded Representation:")
    for bank_name, bank_vec in alu_banks.items():
        zeros = np.sum(bank_vec == 0)
        ones = np.sum(bank_vec == 1)
        neg_ones = np.sum(bank_vec == -1)

        zero_pct = 100 * zeros / D
        one_pct = 100 * ones / D
        neg_one_pct = 100 * neg_ones / D

        magnitude = np.linalg.norm(bank_vec)

        print(f"  {bank_name}:")
        print(f"    +1: {ones:4d} ({one_pct:4.1f}%)  |  0: {zeros:4d} ({zero_pct:4.1f}%)  |  -1: {neg_ones:4d} ({neg_one_pct:4.1f}%)")
        print(f"    L2 norm: {magnitude:.2f}")

    # Calculate overall density
    total_zeros = sum(np.sum(bank_vec == 0) for bank_vec in alu_banks.values())
    total_elements = 3 * D  # 3 banks × D dimensions
    density = 1 - (total_zeros / total_elements)

    print()
    print(f"Overall Density: {100 * density:.1f}%")
    print(f"  Total zeros: {total_zeros} / {total_elements}")
    print()

    # Calculate magnitude profile
    mag1 = np.linalg.norm(alu_banks['bank1'])
    mag2 = np.linalg.norm(alu_banks['bank2'])
    mag3 = np.linalg.norm(alu_banks['bank3'])
    total_mag = mag1 + mag2 + mag3

    ratio1 = mag1 / total_mag
    ratio2 = mag2 / total_mag
    ratio3 = mag3 / total_mag

    print("Magnitude Profile:")
    print(f"  Bank 1 (Hydrophobic):  {mag1:6.2f}  ({ratio1:.3f})")
    print(f"  Bank 2 (Major Groove): {mag2:6.2f}  ({ratio2:.3f})")
    print(f"  Bank 3 (Hinge):        {mag3:6.2f}  ({ratio3:.3f})")
    print()

    # Check if this matches ALU signature
    print("ALU Signature Detection:")
    if density >= 0.98:
        print(f"  ✓ High density ({100*density:.1f}%) - consistent with GC-rich ALU")
    else:
        print(f"  ⚠ Lower density ({100*density:.1f}%) - expected >98%")

    if ratio2 > 0.38:
        print(f"  ✓ Bank 2 dominant ({ratio2:.3f}) - consistent with GC-rich major groove")
    else:
        print(f"  ⚠ Bank 2 not dominant ({ratio2:.3f}) - expected >0.38")

    if ratio3 > 0.32:
        print(f"  ✓ Bank 3 elevated ({ratio3:.3f}) - consistent with YR/RY transitions")
    else:
        print(f"  ⚠ Bank 3 not elevated ({ratio3:.3f}) - expected >0.32")

    print()

    # Create poly-A tail signature
    print("Creating Poly-A Tail Signature:")
    polya_sequence = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"  # 34 A's
    polya_banks = encode_sequence_to_banks(polya_sequence, position_codebook, D)

    polya_zeros = sum(np.sum(bank_vec == 0) for bank_vec in polya_banks.values())
    polya_density = 1 - (polya_zeros / total_elements)

    polya_mag1 = np.linalg.norm(polya_banks['bank1'])
    polya_mag2 = np.linalg.norm(polya_banks['bank2'])
    polya_mag3 = np.linalg.norm(polya_banks['bank3'])
    polya_total = polya_mag1 + polya_mag2 + polya_mag3

    print(f"  Poly-A density: {100*polya_density:.1f}%")
    print(f"  Magnitude profile: {polya_mag1/polya_total:.3f} / {polya_mag2/polya_total:.3f} / {polya_mag3/polya_total:.3f}")
    print()

    # Save lens definitions
    output_file = Path("output/biological_lenses.npz")
    output_file.parent.mkdir(exist_ok=True)

    np.savez_compressed(
        output_file,
        alu_bank1=alu_banks['bank1'],
        alu_bank2=alu_banks['bank2'],
        alu_bank3=alu_banks['bank3'],
        alu_density=density,
        alu_mag_ratios=np.array([ratio1, ratio2, ratio3]),
        polya_bank1=polya_banks['bank1'],
        polya_bank2=polya_banks['bank2'],
        polya_bank3=polya_banks['bank3'],
        polya_density=polya_density,
        polya_mag_ratios=np.array([polya_mag1/polya_total, polya_mag2/polya_total, polya_mag3/polya_total]),
        metadata={
            'alu_sequence': alu_example,
            'polya_sequence': polya_sequence,
            'N': N,
            'D': D,
            'seed': 42
        }
    )

    logger.info(f"Saved lens definitions to {output_file}")

    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print()
    print("1. Test these lenses against known ALU positions in encoded genome")
    print("2. Create split ALU lens (ALU segment + polyA tail at chunk boundary)")
    print("3. Update offline indexing to detect both patterns")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
