"""
Validate 2-bit ternary packing on REAL genome chunks.

This validates lossless compression on actual encoded genomic data,
not random test data. Critical for Phase 1 Week 1 completion.

Author: Claude Code
Date: November 21, 2025
"""

import h5py
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from quantization.ternary_2bit_packing import (
    pack_3bank_chunk,
    unpack_3bank_chunk,
    validate_packing_lossless,
    calculate_storage_reduction,
)


def validate_real_genome_chunks(
    h5_path: str,
    num_chunks: int = 100,
    seed: int = 42,
) -> dict:
    """
    Validate 2-bit packing on random real genome chunks.

    Args:
        h5_path: Path to encoded genome HDF5 file
        num_chunks: Number of random chunks to validate
        seed: Random seed for reproducibility

    Returns:
        Validation statistics dictionary
    """
    print("=" * 80)
    print("2-Bit Ternary Packing Validation on REAL Genome Chunks")
    print("=" * 80)
    print()

    # Load encoded genome
    print(f"Loading encoded genome: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        all_banks = f['all_bank_vectors']
        total_chunks = all_banks.shape[0]

        print(f"  Total chunks: {total_chunks:,}")
        print(f"  Shape: {all_banks.shape}")
        print(f"  Dtype: {all_banks.dtype}")
        print()

        # Select random chunks
        np.random.seed(seed)
        chunk_indices = np.random.choice(total_chunks, size=num_chunks, replace=False)
        chunk_indices.sort()

        print(f"Testing {num_chunks} random chunks (seed={seed})")
        print(f"  Chunk indices: {chunk_indices[:5].tolist()}... (showing first 5)")
        print()

        # Validate each chunk
        print("Validating lossless packing...")
        failures = []
        original_sizes = []
        packed_sizes = []

        for i, chunk_idx in enumerate(chunk_indices):
            # Load real chunk (3 banks × D dimensions)
            chunk = all_banks[chunk_idx, :, :]  # Shape: (3, 5120)
            bank1 = chunk[0, :].astype(np.int8)
            bank2 = chunk[1, :].astype(np.int8)
            bank3 = chunk[2, :].astype(np.int8)

            # Verify ternary values
            for bank_num, bank in enumerate([bank1, bank2, bank3], 1):
                unique_vals = np.unique(bank)
                if not np.all(np.isin(unique_vals, [-1, 0, 1])):
                    failures.append({
                        'chunk_idx': chunk_idx,
                        'bank': bank_num,
                        'error': f'Non-ternary values: {unique_vals}',
                    })
                    continue

            # Pack
            packed1, packed2, packed3 = pack_3bank_chunk(bank1, bank2, bank3)

            # Unpack
            unpacked1, unpacked2, unpacked3 = unpack_3bank_chunk(packed1, packed2, packed3)

            # Validate bit-identical
            if not (np.array_equal(bank1, unpacked1) and
                    np.array_equal(bank2, unpacked2) and
                    np.array_equal(bank3, unpacked3)):
                failures.append({
                    'chunk_idx': chunk_idx,
                    'error': 'Pack/unpack mismatch',
                })

            # Track sizes
            original_size = len(bank1) + len(bank2) + len(bank3)
            packed_size = len(packed1) + len(packed2) + len(packed3)
            original_sizes.append(original_size)
            packed_sizes.append(packed_size)

            # Progress
            if (i + 1) % 20 == 0:
                print(f"  Validated {i + 1}/{num_chunks} chunks...")

        print(f"  ✓ Completed {num_chunks} chunks")
        print()

        # Report results
        if failures:
            print("=" * 80)
            print(f"❌ VALIDATION FAILED: {len(failures)} chunks had errors")
            print("=" * 80)
            for failure in failures[:5]:  # Show first 5
                print(f"  Chunk {failure['chunk_idx']}: {failure['error']}")
            return {'success': False, 'failures': failures}

        print("=" * 80)
        print("✓ ALL CHUNKS VALIDATED SUCCESSFULLY - Packing is LOSSLESS")
        print("=" * 80)
        print()

        # Storage statistics
        total_original = sum(original_sizes)
        total_packed = sum(packed_sizes)
        reduction = total_original / total_packed

        print("Storage Reduction (100 chunks):")
        print(f"  Original: {total_original:,} bytes ({total_original / 1024:.2f} KB)")
        print(f"  Packed:   {total_packed:,} bytes ({total_packed / 1024:.2f} KB)")
        print(f"  Reduction: {reduction:.2f}×")
        print()

        # Extrapolate to full genome
        print("Extrapolation to Full Genome:")
        stats = calculate_storage_reduction(num_chunks=total_chunks, D=5120)
        print(f"  Total chunks: {stats['num_chunks']:,}")
        print(f"  Original size: {stats['total_original_GB']:.2f} GB")
        print(f"  Packed size: {stats['total_packed_GB']:.2f} GB")
        print(f"  Compressed size (est): {stats['total_compressed_GB']:.2f} GB")
        print(f"  Reduction (packing): {stats['reduction_packing']:.1f}×")
        print(f"  Reduction (with gzip): {stats['reduction_with_compression']:.1f}×")
        print()

        return {
            'success': True,
            'num_chunks_tested': num_chunks,
            'total_chunks': total_chunks,
            'original_bytes': total_original,
            'packed_bytes': total_packed,
            'reduction_factor': reduction,
            'full_genome_stats': stats,
        }


if __name__ == '__main__':
    h5_path = 'output/encoded_genome_3banks.h5'

    if not Path(h5_path).exists():
        print(f"Error: Encoded genome file not found: {h5_path}")
        sys.exit(1)

    results = validate_real_genome_chunks(h5_path, num_chunks=100, seed=42)

    if results['success']:
        print("=" * 80)
        print("✓ PHASE 1 WEEK 1 VALIDATION COMPLETE")
        print("=" * 80)
        print()
        print("2-bit packing is production-ready for real genomic data.")
        print("Ready to proceed to Phase 1 Week 2.")
    else:
        print("=" * 80)
        print("❌ VALIDATION FAILED - DO NOT PROCEED")
        print("=" * 80)
        sys.exit(1)
