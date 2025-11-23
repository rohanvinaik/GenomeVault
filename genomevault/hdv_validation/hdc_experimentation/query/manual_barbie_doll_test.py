"""
Manual "Barbie Doll" Test - Smash an ALU chunk against ALU lens

Find a real genomic chunk containing an ALU repeat and manually
compute similarity to the ALU lens to see what's actually happening.

Author: Investigation requested by user
Date: November 21, 2025
"""

import sys
from pathlib import Path
import h5py
import numpy as np
from pyfaidx import Fasta

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from query.lens_aware_simd_query_engine import (
    generate_position_codebook,
    LensLibrary
)


def find_alu_sequence_in_genome(fasta_path: str, search_length: int = 1024):
    """
    Search consensus FASTA for ALU repeat sequences.

    ALU consensus (simplified): GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTT

    Returns positions where ALU-like patterns are found.
    """
    print("Searching for ALU repeat sequences in consensus genome...")
    print()

    # ALU Alu consensus sequence (from RepBase)
    # Using a simplified signature that should appear in many ALUs
    alu_signature = "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTT"

    fasta = Fasta(fasta_path)

    # Search chromosome 1 (largest, most likely to have ALUs)
    chr1 = fasta['chr1_consensus']
    chr1_seq = str(chr1[:]).upper()

    print(f"Searching chr1 (length: {len(chr1_seq):,} bp) for ALU signature...")
    print(f"ALU signature: {alu_signature}")
    print()

    # Find all occurrences
    positions = []
    start = 0
    while True:
        pos = chr1_seq.find(alu_signature, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1

        if len(positions) >= 10:  # Find first 10
            break

    print(f"Found {len(positions)} occurrences of ALU signature in chr1")
    for i, pos in enumerate(positions[:5]):
        print(f"  {i+1}. Position: {pos:,} bp")
    print()

    return positions


def manually_encode_sequence(sequence: str, position_codebook: np.ndarray, D: int = 5120):
    """
    Manually encode a sequence using the 3-bank ternary encoding.

    This mimics what the encoder does, step by step.
    """
    N = len(position_codebook)
    sequence = sequence.upper()[:N]  # Truncate to N bases

    # Initialize banks
    bank1 = np.zeros(D, dtype=np.float32)  # Hydrophobic (T=+1, A=-1, GC=0)
    bank2 = np.zeros(D, dtype=np.float32)  # Major Groove (G=+1, C=-1, AT=0)
    bank3 = np.zeros(D, dtype=np.float32)  # Hinge (YR=+1, RY=-1)

    for i, base in enumerate(sequence):
        pos_vec = position_codebook[i]

        # Bank 1: Hydrophobic
        if base == 'T':
            bank1 += pos_vec
        elif base == 'A':
            bank1 -= pos_vec
        # G, C contribute 0

        # Bank 2: Major Groove
        if base == 'G':
            bank2 += pos_vec
        elif base == 'C':
            bank2 -= pos_vec
        # A, T contribute 0

        # Bank 3: Hinge (Pyrimidine-Purine transitions)
        # Y (C/T) to R (A/G) = +1
        # R (A/G) to Y (C/T) = -1
        if i > 0:
            prev_base = sequence[i-1]
            # YR transition: (C or T) → (A or G)
            if prev_base in 'CT' and base in 'AG':
                bank3 += pos_vec
            # RY transition: (A or G) → (C or T)
            elif prev_base in 'AG' and base in 'CT':
                bank3 -= pos_vec

    return {
        'bank1': bank1,
        'bank2': bank2,
        'bank3': bank3
    }


def main():
    print("=" * 80)
    print("MANUAL BARBIE DOLL TEST: ALU Chunk vs ALU Lens")
    print("=" * 80)
    print()

    # Paths
    consensus_fasta = "/Users/rohanvinaik/genomevault/benchmark_results/enhanced_privacy_k13_phase123_optimized/layer1_consensus/consensus.fa"
    h5_path = "output/encoded_genome_3banks.h5"

    if not Path(consensus_fasta).exists():
        print(f"ERROR: Consensus FASTA not found: {consensus_fasta}")
        return

    if not Path(h5_path).exists():
        print(f"ERROR: Encoded genome not found: {h5_path}")
        return

    # Find ALU sequences
    alu_positions = find_alu_sequence_in_genome(consensus_fasta)

    if not alu_positions:
        print("No ALU sequences found!")
        return

    # Pick first ALU occurrence
    alu_pos = alu_positions[0]
    print(f"Using ALU at position {alu_pos:,} bp on chr1")
    print()

    # Load parameters from H5
    with h5py.File(h5_path, 'r') as f:
        D = f.attrs.get('dimension', 5120)
        N = f.attrs.get('chunk_size', 1024)
        stride = f.attrs.get('stride', 896)

    print(f"Encoding parameters: D={D}, N={N}, stride={stride}")
    print()

    # Generate position codebook (must match encoder!)
    position_codebook = generate_position_codebook(N, D, seed=42)
    print(f"✓ Generated position codebook (N={N}, D={D})")
    print()

    # Calculate chunk index for ALU position
    chunk_idx = alu_pos // stride
    chunk_start = chunk_idx * stride
    chunk_end = chunk_start + N

    print(f"ALU chunk mapping:")
    print(f"  Chunk index: {chunk_idx}")
    print(f"  Chunk range: {chunk_start:,} - {chunk_end:,} bp")
    print(f"  ALU position in chunk: {alu_pos - chunk_start} bp")
    print()

    # Load chunk sequence from FASTA
    fasta = Fasta(consensus_fasta)
    chunk_seq = str(fasta['chr1_consensus'][chunk_start:chunk_end]).upper()
    print(f"Loaded chunk sequence: {len(chunk_seq)} bp")
    print(f"  First 60 bp: {chunk_seq[:60]}")
    print(f"  Last 60 bp:  {chunk_seq[-60:]}")
    print()

    # Load encoded chunk from H5
    with h5py.File(h5_path, 'r') as f:
        encoded_chunk = f['all_bank_vectors'][chunk_idx, :, :]  # Shape: (3, D)

    encoded_vectors = {
        'bank1': encoded_chunk[0, :].astype(np.float32),
        'bank2': encoded_chunk[1, :].astype(np.float32),
        'bank3': encoded_chunk[2, :].astype(np.float32),
    }

    print("✓ Loaded encoded chunk from H5 database")
    print(f"  Bank1: min={encoded_vectors['bank1'].min()}, max={encoded_vectors['bank1'].max()}, mean={encoded_vectors['bank1'].mean():.3f}")
    print(f"  Bank2: min={encoded_vectors['bank2'].min()}, max={encoded_vectors['bank2'].max()}, mean={encoded_vectors['bank2'].mean():.3f}")
    print(f"  Bank3: min={encoded_vectors['bank3'].min()}, max={encoded_vectors['bank3'].max()}, mean={encoded_vectors['bank3'].mean():.3f}")
    print()

    # Manually encode the chunk sequence to verify
    print("Manually encoding chunk sequence for verification...")
    manual_vectors = manually_encode_sequence(chunk_seq, position_codebook, D)

    # Compare manual encoding to H5 encoding
    diff1 = np.abs(manual_vectors['bank1'] - encoded_vectors['bank1']).sum()
    diff2 = np.abs(manual_vectors['bank2'] - encoded_vectors['bank2']).sum()
    diff3 = np.abs(manual_vectors['bank3'] - encoded_vectors['bank3']).sum()

    print(f"Verification: Manual encoding vs H5 encoding")
    print(f"  Bank1 difference: {diff1:.2f}")
    print(f"  Bank2 difference: {diff2:.2f}")
    print(f"  Bank3 difference: {diff3:.2f}")

    if diff1 < 1.0 and diff2 < 1.0 and diff3 < 1.0:
        print(f"  ✓ MATCH! Manual encoding matches H5")
    else:
        print(f"  ✗ MISMATCH! Manual encoding doesn't match H5")
    print()

    # Create ALU lens
    print("Creating ALU lens from synthetic sequence...")
    lens_lib = LensLibrary(D=D)
    lens_lib.build_simple_library(position_codebook)
    alu_lens = lens_lib.lenses['ALU_YI']

    print(f"ALU lens stats:")
    print(f"  Bank1: min={alu_lens.bank1.min()}, max={alu_lens.bank1.max()}, mean={alu_lens.bank1.mean():.3f}")
    print(f"  Bank2: min={alu_lens.bank2.min()}, max={alu_lens.bank2.max()}, mean={alu_lens.bank2.mean():.3f}")
    print(f"  Bank3: min={alu_lens.bank3.min()}, max={alu_lens.bank3.max()}, mean={alu_lens.bank3.mean():.3f}")
    print()

    # BARBIE DOLL SMASH: Compute similarity
    print("=" * 80)
    print("BARBIE DOLL SMASH: Computing similarity")
    print("=" * 80)
    print()

    # Use H5 encoded chunk
    sim1_h5 = np.dot(encoded_vectors['bank1'], alu_lens.bank1) / D
    sim2_h5 = np.dot(encoded_vectors['bank2'], alu_lens.bank2) / D
    sim3_h5 = np.dot(encoded_vectors['bank3'], alu_lens.bank3) / D
    combined_h5 = (sim1_h5 + sim2_h5 + sim3_h5) / 3.0

    print("H5 Encoded Chunk → ALU Lens:")
    print(f"  Bank1 similarity: {sim1_h5:.6f}")
    print(f"  Bank2 similarity: {sim2_h5:.6f}")
    print(f"  Bank3 similarity: {sim3_h5:.6f}")
    print(f"  Combined:         {combined_h5:.6f}")
    print()

    # Use manually encoded chunk
    sim1_manual = np.dot(manual_vectors['bank1'], alu_lens.bank1) / D
    sim2_manual = np.dot(manual_vectors['bank2'], alu_lens.bank2) / D
    sim3_manual = np.dot(manual_vectors['bank3'], alu_lens.bank3) / D
    combined_manual = (sim1_manual + sim2_manual + sim3_manual) / 3.0

    print("Manually Encoded Chunk → ALU Lens:")
    print(f"  Bank1 similarity: {sim1_manual:.6f}")
    print(f"  Bank2 similarity: {sim2_manual:.6f}")
    print(f"  Bank3 similarity: {sim3_manual:.6f}")
    print(f"  Combined:         {combined_manual:.6f}")
    print()

    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()

    if combined_h5 > 0.05:
        print(f"✓ SUCCESS! Similarity {combined_h5:.4f} > 0.05 threshold")
        print(f"  The lens system SHOULD work for ALU repeats.")
        print(f"  Problem is likely in the query engine logic.")
    else:
        print(f"✗ PROBLEM! Similarity {combined_h5:.4f} < 0.05 threshold")
        print(f"  Even a real ALU chunk shows low similarity.")
        print(f"  This suggests:")
        print(f"    1. Synthetic lens doesn't match real ALU encoding")
        print(f"    2. Encoding method may not preserve motif structure")
        print(f"    3. Threshold of 5% is too high for this approach")
    print()

    # Compute lens self-similarity for reference
    self_sim1 = np.dot(alu_lens.bank1, alu_lens.bank1) / D
    self_sim2 = np.dot(alu_lens.bank2, alu_lens.bank2) / D
    self_sim3 = np.dot(alu_lens.bank3, alu_lens.bank3) / D
    combined_self = (self_sim1 + self_sim2 + self_sim3) / 3.0

    print(f"Reference: ALU Lens → ALU Lens (self-similarity):")
    print(f"  Bank1: {self_sim1:.6f}")
    print(f"  Bank2: {self_sim2:.6f}")
    print(f"  Bank3: {self_sim3:.6f}")
    print(f"  Combined: {combined_self:.6f}")
    print()

    print("=" * 80)


if __name__ == '__main__':
    main()
