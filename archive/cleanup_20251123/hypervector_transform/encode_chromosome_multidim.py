#!/usr/bin/env python3
"""
Encode a single chromosome with different dimensions for AT vs GC.

Strategy:
- Direct encoding from FASTA (no GDiff)
- Encode with all 5 lenses at max(D_at, D_gc)
- Extract AT subset at D_at
- Extract GC subset at D_gc
- Quantize to unipolar
- Bit-pack
"""

import h5py
import numpy as np
from pathlib import Path
import time
import sys
import argparse
import pysam

# Lens definitions
LENS_DEFINITIONS = {
    'AT': {'positive': {'A'}, 'negative': {'T'}},
    'GC': {'positive': {'G'}, 'negative': {'C'}},
    'PuPy': {'positive': {'A', 'G'}, 'negative': {'T', 'C'}},  # Purine vs Pyrimidine
    'AmKe': {'positive': {'A', 'C'}, 'negative': {'G', 'T'}},  # Amino vs Keto
    'StWk': {'positive': {'G', 'C'}, 'negative': {'A', 'T'}},  # Strong vs Weak
}

LENS_NAMES = ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']


def generate_random_codebook(dimension, chunk_size):
    """Generate random position codebook."""
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    return rng.randn(chunk_size, dimension).astype(np.float32)


def encode_chunk(sequence, position_codebook):
    """
    Encode a chunk with all 5 lenses.

    Returns: array of shape (5, D) where D is dimension
    """
    D = position_codebook.shape[1]

    # Initialize vectors for all 5 lenses
    lens_vectors = np.zeros((5, D), dtype=np.float32)

    # Process each position
    for offset, nucleotide in enumerate(sequence):
        if offset >= len(position_codebook):
            break

        pos_vec = position_codebook[offset]

        # Encode with all lenses
        for lens_idx, lens_name in enumerate(LENS_NAMES):
            lens_def = LENS_DEFINITIONS[lens_name]
            if nucleotide in lens_def['positive']:
                lens_vectors[lens_idx] += pos_vec
            elif nucleotide in lens_def['negative']:
                lens_vectors[lens_idx] -= pos_vec
            # 'N' contributes to neither (ternary: 0)

    return lens_vectors


def encode_chromosome_multidim(
    fasta_path,
    output_dir,
    chromosome,
    d_at=15000,
    d_gc=20000,
    chunk_size=2000,
):
    """
    Encode a single chromosome with different D for AT vs GC.

    Args:
        fasta_path: Path to reference FASTA (hg38, etc.)
        output_dir: Directory for output files
        chromosome: Chromosome to encode (e.g., 'chr22')
        d_at: Dimensionality for AT-focused vector
        d_gc: Dimensionality for GC-focused vector
        chunk_size: Base pairs per chunk
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    d_max = max(d_at, d_gc)

    print("=" * 80)
    print(f"ENCODING {chromosome} WITH MULTI-DIMENSIONAL ARCHITECTURE")
    print("=" * 80)
    print()
    print(f"Input FASTA: {fasta_path}")
    print(f"Chromosome: {chromosome}")
    print(f"Chunk size: {chunk_size:,} bp")
    print()
    print(f"Dimensions:")
    print(f"  AT-focused: {d_at:,}D")
    print(f"  GC-focused: {d_gc:,}D")
    print(f"  Encoding at: {d_max:,}D (max)")
    print()

    # Generate position codebook at max dimension
    print("Generating position codebook...")
    position_codebook = generate_random_codebook(d_max, chunk_size)
    print(f"  Codebook shape: {position_codebook.shape}")
    print()

    # Step 1: Encode full 5-lens at max dimension
    print("=" * 80)
    print("STEP 1: ENCODING FULL 5-LENS AT MAX DIMENSION")
    print("=" * 80)
    print()

    # Load chromosome sequence
    fasta = pysam.FastaFile(str(fasta_path))

    if chromosome not in fasta.references:
        print(f"ERROR: {chromosome} not found in FASTA")
        print(f"Available: {fasta.references[:10]}...")
        return False

    sequence = fasta.fetch(chromosome)
    seq_len = len(sequence)

    print(f"Chromosome length: {seq_len:,} bp")
    print(f"Expected chunks: {seq_len // chunk_size:,}")
    print()

    # Encode chromosome
    start_time = time.time()

    chunks = []
    chunk_keys = []

    for i in range(0, seq_len, chunk_size):
        end = min(i + chunk_size, seq_len)
        chunk_seq = sequence[i:end]

        if len(chunk_seq) < chunk_size:
            # Pad last chunk
            chunk_seq = chunk_seq + 'N' * (chunk_size - len(chunk_seq))

        # Encode this chunk
        vector = encode_chunk(chunk_seq, position_codebook)  # Shape: (5, d_max)
        chunks.append(vector)
        chunk_keys.append(f"{chromosome}:{i}")

        if (len(chunks)) % 1000 == 0:
            elapsed = time.time() - start_time
            pct = 100.0 * end / seq_len
            print(f"  Progress: {pct:5.1f}% | {len(chunks):,} chunks | {elapsed:.1f}s")

    elapsed = time.time() - start_time

    print()
    print(f"✓ Encoding complete: {len(chunks):,} chunks in {elapsed:.1f}s")
    print()

    # Convert to array
    all_chunks = np.array(chunks)  # Shape: (num_chunks, 5, d_max)

    print(f"Encoded shape: {all_chunks.shape}")
    print()

    # Step 2: Extract and quantize AT-focused
    print("=" * 80)
    print("STEP 2: AT-FOCUSED EXTRACTION & QUANTIZATION")
    print("=" * 80)
    print()

    at_lenses = [0, 2, 3, 4]  # AT, PuPy, AmKe, StWk

    # Extract AT lenses and truncate to d_at dimensions
    at_vectors = all_chunks[:, at_lenses, :d_at]  # Shape: (chunks, 4, d_at)

    print(f"AT-focused shape: {at_vectors.shape}")
    print(f"AT dimensions: {d_at:,}")
    print()

    # Unipolar quantization
    print("Quantizing to unipolar {0, 1}...")
    at_quantized = (at_vectors >= 0).astype(np.uint8)

    # Bit-pack
    print("Bit-packing (8 values → 1 byte)...")

    # Ensure dimensions are multiple of 8
    if d_at % 8 != 0:
        pad_size = ((d_at + 7) // 8) * 8 - d_at
        padding = np.zeros((at_quantized.shape[0], at_quantized.shape[1], pad_size), dtype=np.uint8)
        at_quantized = np.concatenate([at_quantized, padding], axis=2)

    at_packed = np.packbits(at_quantized, axis=-1)

    print(f"AT packed shape: {at_packed.shape}")
    print()

    # Save AT file
    at_file = output_dir / f"{chromosome}_at_focused_D{d_at}_packed.h5"

    print(f"Saving to: {at_file}")

    with h5py.File(at_file, 'w') as f:
        f.create_dataset(
            'lens_vectors_packed',
            data=at_packed
            # No compression - bit-packed data has max entropy, gzip won't help
        )
        f.create_dataset('chunk_keys', data=np.array(chunk_keys, dtype='S'))
        f.attrs['chromosome'] = chromosome
        f.attrs['dimension'] = d_at
        f.attrs['focus'] = 'AT'
        f.attrs['lenses'] = 'AT,PuPy,AmKe,StWk'
        f.attrs['bit_packed'] = True
        f.attrs['chunk_size'] = chunk_size

    at_size = at_file.stat().st_size / (1024**2)
    print(f"✓ AT file saved: {at_size:.2f} MB")
    print()

    # Step 3: Extract and quantize GC-focused
    print("=" * 80)
    print("STEP 3: GC-FOCUSED EXTRACTION & QUANTIZATION")
    print("=" * 80)
    print()

    gc_lenses = [1, 2, 3, 4]  # GC, PuPy, AmKe, StWk

    # Extract GC lenses and truncate to d_gc dimensions
    gc_vectors = all_chunks[:, gc_lenses, :d_gc]  # Shape: (chunks, 4, d_gc)

    print(f"GC-focused shape: {gc_vectors.shape}")
    print(f"GC dimensions: {d_gc:,}")
    print()

    # Unipolar quantization
    print("Quantizing to unipolar {0, 1}...")
    gc_quantized = (gc_vectors >= 0).astype(np.uint8)

    # Bit-pack
    print("Bit-packing (8 values → 1 byte)...")

    if d_gc % 8 != 0:
        pad_size = ((d_gc + 7) // 8) * 8 - d_gc
        padding = np.zeros((gc_quantized.shape[0], gc_quantized.shape[1], pad_size), dtype=np.uint8)
        gc_quantized = np.concatenate([gc_quantized, padding], axis=2)

    gc_packed = np.packbits(gc_quantized, axis=-1)

    print(f"GC packed shape: {gc_packed.shape}")
    print()

    # Save GC file
    gc_file = output_dir / f"{chromosome}_gc_focused_D{d_gc}_packed.h5"

    print(f"Saving to: {gc_file}")

    with h5py.File(gc_file, 'w') as f:
        f.create_dataset(
            'lens_vectors_packed',
            data=gc_packed
            # No compression - bit-packed data has max entropy, gzip won't help
        )
        f.create_dataset('chunk_keys', data=np.array(chunk_keys, dtype='S'))
        f.attrs['chromosome'] = chromosome
        f.attrs['dimension'] = d_gc
        f.attrs['focus'] = 'GC'
        f.attrs['lenses'] = 'GC,PuPy,AmKe,StWk'
        f.attrs['bit_packed'] = True
        f.attrs['chunk_size'] = chunk_size

    gc_size = gc_file.stat().st_size / (1024**2)
    print(f"✓ GC file saved: {gc_size:.2f} MB")
    print()

    # Summary
    total_time = time.time() - start_time

    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()
    print(f"Chromosome: {chromosome}")
    print(f"Total chunks: {len(chunks):,}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print()
    print(f"Files created:")
    print(f"  AT-focused ({d_at:,}D): {at_file.name} ({at_size:.2f} MB)")
    print(f"  GC-focused ({d_gc:,}D): {gc_file.name} ({gc_size:.2f} MB)")
    print(f"  Total: {at_size + gc_size:.2f} MB")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(description='Encode chromosome with multi-dimensional architecture')
    parser.add_argument('--fasta', required=True, help='Path to reference FASTA')
    parser.add_argument('--chromosome', default='chr22', help='Chromosome to encode (default: chr22)')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--d-at', type=int, default=15000, help='Dimensions for AT-focused (default: 15000)')
    parser.add_argument('--d-gc', type=int, default=20000, help='Dimensions for GC-focused (default: 20000)')
    parser.add_argument('--chunk-size', type=int, default=2000, help='Chunk size in bp (default: 2000)')

    args = parser.parse_args()

    success = encode_chromosome_multidim(
        fasta_path=args.fasta,
        output_dir=args.output,
        chromosome=args.chromosome,
        d_at=args.d_at,
        d_gc=args.d_gc,
        chunk_size=args.chunk_size,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
