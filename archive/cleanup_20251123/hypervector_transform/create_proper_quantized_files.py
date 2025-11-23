#!/usr/bin/env python3
"""
Create properly quantized int8 and int4 files from float32 3D file.

Proper quantization preserves BOTH magnitude and sign, not just sign.
"""

import h5py
import numpy as np
from pathlib import Path
import time
import sys

def compute_global_scale(h5_file, sample_size=10000):
    """
    Compute global scale factor by finding max absolute value across all lenses.

    Args:
        h5_file: Open HDF5 file handle
        sample_size: Number of chunks to sample for max estimation

    Returns:
        max_abs: Maximum absolute value across sampled data
    """
    print(f"Computing global scale factor (sampling {sample_size:,} chunks)...")

    all_lens_vectors = h5_file['all_lens_vectors']
    total_chunks = all_lens_vectors.shape[0]
    num_lenses = all_lens_vectors.shape[1]

    # Sample chunks evenly distributed across dataset
    if sample_size >= total_chunks:
        chunk_indices = np.arange(total_chunks)
    else:
        chunk_indices = np.linspace(0, total_chunks - 1, sample_size, dtype=int)

    max_abs = 0.0

    for i, chunk_idx in enumerate(chunk_indices):
        if i % 1000 == 0:
            print(f"  Progress: {i:,}/{len(chunk_indices):,} chunks sampled, current max: {max_abs:.2f}")

        # Read all lenses for this chunk
        chunk_data = all_lens_vectors[chunk_idx, :, :]  # Shape: (5, 10000)
        chunk_max = np.max(np.abs(chunk_data))
        max_abs = max(max_abs, chunk_max)

    print(f"✓ Global max absolute value: {max_abs:.2f}")
    return max_abs


def quantize_int8(float_data, scale):
    """
    Quantize float32 data to int8 range [-127, +127].

    Args:
        float_data: Float32 numpy array
        scale: Scale factor (max_abs / 127.0)

    Returns:
        int8 numpy array
    """
    quantized = np.clip(np.round(float_data / scale), -127, 127).astype(np.int8)
    return quantized


def quantize_int4(float_data, scale):
    """
    Quantize float32 data to int4 range [-7, +7].

    Args:
        float_data: Float32 numpy array
        scale: Scale factor (max_abs / 7.0)

    Returns:
        int8 numpy array (values in range [-7, +7])
    """
    quantized = np.clip(np.round(float_data / scale), -7, 7).astype(np.int8)
    return quantized


def create_quantized_file(input_path, output_path, quantization_type, batch_size=1000):
    """
    Create quantized HDF5 file from float32 source.

    Args:
        input_path: Path to float32 3D HDF5 file
        output_path: Path to output quantized file
        quantization_type: 'int8' or 'int4'
        batch_size: Number of chunks to process at once
    """
    print("=" * 80)
    print(f"CREATING {quantization_type.upper()} QUANTIZED FILE")
    print("=" * 80)
    print()
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    start_time = time.time()

    # Open input file
    print("Opening source file...")
    f_in = h5py.File(input_path, 'r')
    all_lens_vectors = f_in['all_lens_vectors']
    chunk_keys = f_in['chunk_keys']

    total_chunks = all_lens_vectors.shape[0]
    num_lenses = all_lens_vectors.shape[1]
    dimensions = all_lens_vectors.shape[2]

    print(f"  Total chunks: {total_chunks:,}")
    print(f"  Lenses: {num_lenses}")
    print(f"  Dimensions: {dimensions:,}")
    print()

    # Compute global scale
    max_abs = compute_global_scale(f_in, sample_size=10000)

    if quantization_type == 'int8':
        scale = max_abs / 127.0
        quantize_func = quantize_int8
    elif quantization_type == 'int4':
        scale = max_abs / 7.0
        quantize_func = quantize_int4
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")

    print()
    print(f"Scale factor: {scale:.4f}")
    print(f"  Example: float32=100 → {quantization_type}={quantize_func(np.array([100.0]), scale)[0]}")
    print(f"  Example: float32=-50 → {quantization_type}={quantize_func(np.array([-50.0]), scale)[0]}")
    print()

    # Create output file
    print("Creating output file...")
    f_out = h5py.File(output_path, 'w')

    # Create datasets
    quant_dataset = f_out.create_dataset(
        'all_lens_vectors',
        shape=(total_chunks, num_lenses, dimensions),
        dtype='int8',
        chunks=(1, num_lenses, dimensions),
        compression='gzip',
        compression_opts=4
    )

    # Copy chunk keys
    f_out.create_dataset('chunk_keys', data=chunk_keys[:])

    # Add metadata
    f_out.attrs['quantization_type'] = quantization_type
    f_out.attrs['scale_factor'] = scale
    f_out.attrs['global_max_abs'] = max_abs
    f_out.attrs['source_file'] = str(input_path)
    f_out.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')

    print(f"✓ Output file created")
    print()

    # Quantize data in batches
    print(f"Quantizing data (batch size: {batch_size:,} chunks)...")
    num_batches = (total_chunks + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_chunks)

        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            progress = (start_idx / total_chunks) * 100
            print(f"  Batch {batch_idx+1}/{num_batches} ({progress:.1f}%) | "
                  f"Chunks {start_idx:,}-{end_idx:,} | "
                  f"Elapsed: {elapsed:.1f}s")

        # Read batch
        batch_data = all_lens_vectors[start_idx:end_idx, :, :]

        # Quantize
        quantized_batch = quantize_func(batch_data, scale)

        # Write
        quant_dataset[start_idx:end_idx, :, :] = quantized_batch

    elapsed = time.time() - start_time
    file_size = Path(output_path).stat().st_size / (1024**3)  # GB

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output file size: {file_size:.1f} GB")
    print(f"Processing speed: {total_chunks/elapsed:.0f} chunks/sec")
    print()

    # Verify a sample
    print("Verification (first chunk, AT lens, first 20 values):")
    float_sample = f_in['all_lens_vectors'][0, 0, :20]
    quant_sample = quant_dataset[0, 0, :20]
    expected_sample = quantize_func(float_sample, scale)

    print(f"  Float32:  {float_sample}")
    print(f"  Quantized: {quant_sample}")
    print(f"  Expected:  {expected_sample}")
    print(f"  Match: {'✓ YES' if np.array_equal(quant_sample, expected_sample) else '❌ NO'}")

    # Check value distribution
    print()
    print("Value distribution (first 1000 chunks, all lenses):")
    sample_data = quant_dataset[:1000, :, :].flatten()
    unique, counts = np.unique(sample_data, return_counts=True)
    print(f"  Unique values: {len(unique)}")
    print(f"  Value range: [{unique.min()}, {unique.max()}]")
    if quantization_type == 'int8':
        expected_range = "[-127, +127]"
    else:
        expected_range = "[-7, +7]"
    print(f"  Expected range: {expected_range}")
    print(f"  Top 10 values: {unique[:10] if len(unique) > 10 else unique}")

    print()

    f_in.close()
    f_out.close()

    print(f"✓ File saved: {output_path}")
    print()


def main():
    # Paths
    base_dir = Path("data/experimental_strands/ERR3239334/hdv_encoding")
    float32_file = base_dir / "encoded_genome_5lenses_3d.h5"
    int8_file = base_dir / "encoded_genome_5lenses_3d_int8.h5"
    int4_file = base_dir / "encoded_genome_5lenses_3d_int4.h5"

    # Check source file exists
    if not float32_file.exists():
        print(f"ERROR: Source file not found: {float32_file}")
        sys.exit(1)

    # Create int8
    create_quantized_file(float32_file, int8_file, 'int8', batch_size=1000)

    # Create int4
    create_quantized_file(float32_file, int4_file, 'int4', batch_size=1000)

    print("=" * 80)
    print("ALL QUANTIZED FILES CREATED SUCCESSFULLY")
    print("=" * 80)


if __name__ == '__main__':
    main()
