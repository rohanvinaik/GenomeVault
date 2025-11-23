"""
Inspect Per-Bank Density Patterns

Sample random chunks and print their per-bank density profiles to understand
what real genomic motifs actually look like in the encoding.
"""

import h5py
import numpy as np

def count_zeros(bank: np.ndarray) -> int:
    """Count number of zero elements in a bank."""
    return np.sum(bank == 0)

def analyze_chunk(banks: dict, chunk_idx: int):
    """Print detailed analysis of a chunk's per-bank properties."""

    # Per-bank statistics
    for name in ['bank1', 'bank2', 'bank3']:
        bank = banks[name]
        zeros = count_zeros(bank)
        density = 1 - (zeros / bank.size)
        magnitude = np.linalg.norm(bank)

        print(f"  {name}: density={density:.3f} ({zeros:5d} zeros), magnitude={magnitude:.1f}")

h5_path = "genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_3banks.h5"

print("="*80)
print("INSPECTING PER-BANK DENSITY PATTERNS")
print("="*80)
print()

with h5py.File(h5_path, 'r') as f:
    all_banks = f['all_bank_vectors']
    total_chunks = all_banks.shape[0]

    # Sample 10 random chunks
    np.random.seed(42)
    sample_indices = np.random.choice(total_chunks, 10, replace=False)

    for i, chunk_idx in enumerate(sample_indices):
        print(f"Chunk {chunk_idx:,} ({chunk_idx * 896:,} bp):")

        all_banks_data = all_banks[chunk_idx, :, :]
        banks = {
            'bank1': all_banks_data[0, :],
            'bank2': all_banks_data[1, :],
            'bank3': all_banks_data[2, :],
        }

        analyze_chunk(banks, chunk_idx)
        print()

print("="*80)
print("KEY INSIGHT:")
print("="*80)
print("If all banks have similar density (~95-100%), position-dependent encoding")
print("is making every position contribute equally, regardless of sequence content.")
print()
print("Expected for real structure:")
print("  - AT-rich: bank1 dense, bank2 sparse")
print("  - GC-rich: bank1 sparse, bank2 dense")
print("="*80)
