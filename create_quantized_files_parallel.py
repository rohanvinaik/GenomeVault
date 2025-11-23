#!/usr/bin/env python3
"""
Create quantized files (int8, int4, ternary) from float32 3D file with parallelization.

Proper quantization preserves BOTH magnitude and sign (int8, int4).
Ternary preserves sign with zero: {-1, 0, +1}.

Optimized for 10-core systems using multiprocessing.
"""

import h5py
import numpy as np
from pathlib import Path
import time
import sys
from multiprocessing import Process, Queue
import os

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
    """Quantize to int8 range [-127, +127]"""
    return np.clip(np.round(float_data / scale), -127, 127).astype(np.int8)


def quantize_int4(float_data, scale):
    """Quantize to int4 range [-7, +7]"""
    return np.clip(np.round(float_data / scale), -7, 7).astype(np.int8)


def quantize_ternary(float_data, scale=None):
    """Quantize to ternary {-1, 0, +1} - sign with zero"""
    return np.sign(float_data).astype(np.int8)


def create_quantized_file(input_path, output_path, quantization_type='int8', batch_size=1000):
    """
    Create a quantized version of the float32 3D HDF5 file.

    Args:
        input_path: Path to float32 source file
        output_path: Path to quantized output file
        quantization_type: 'int8', 'int4', or 'ternary'
        batch_size: Number of chunks to process at once
    """
    # Convert to absolute paths
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"CREATING {quantization_type.upper()} QUANTIZED FILE")
    print("=" * 80)
    print()
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    # Verify input exists
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return False

    # Open source file
    print("Opening source file...")
    f_in = h5py.File(input_path, 'r')
    all_lens_vectors = f_in['all_lens_vectors']

    total_chunks = all_lens_vectors.shape[0]
    num_lenses = all_lens_vectors.shape[1]
    num_dims = all_lens_vectors.shape[2]

    print(f"  Total chunks: {total_chunks:,}")
    print(f"  Lenses: {num_lenses}")
    print(f"  Dimensions: {num_dims:,}")
    print()

    # Compute scale factor (skip for ternary)
    if quantization_type in ['int8', 'int4']:
        max_abs = compute_global_scale(f_in, sample_size=10000)

        if quantization_type == 'int8':
            scale = max_abs / 127.0
            quantize_func = lambda x: quantize_int8(x, scale)
            dtype = np.int8
        else:  # int4
            scale = max_abs / 7.0
            quantize_func = lambda x: quantize_int4(x, scale)
            dtype = np.int8

        print()
        print(f"Scale factor: {scale:.4f}")
        print(f"  Example: float32=100 → {quantization_type}={quantize_func(np.array([100.0]))[0]}")
        print(f"  Example: float32=-50 → {quantization_type}={quantize_func(np.array([-50.0]))[0]}")
    else:  # ternary
        scale = None
        quantize_func = lambda x: quantize_ternary(x)
        dtype = np.int8
        print("Ternary quantization: sign(x) → {-1, 0, +1}")

    print()

    # Create output file
    print("Creating output file...")
    f_out = h5py.File(output_path, 'w')

    # Create dataset with compression
    quant_dataset = f_out.create_dataset(
        'all_lens_vectors',
        shape=(total_chunks, num_lenses, num_dims),
        dtype=dtype,
        chunks=(1, num_lenses, num_dims),
        compression='gzip',
        compression_opts=4  # Balanced compression level
    )

    # Store metadata
    if scale is not None:
        f_out.attrs['scale_factor'] = scale
    f_out.attrs['quantization_type'] = quantization_type
    f_out.attrs['source_file'] = str(input_path)

    print(f"✓ Output file created")
    print()

    # Process in batches
    print(f"Quantizing data (batch size: {batch_size:,} chunks)...")
    num_batches = (total_chunks + batch_size - 1) // batch_size

    start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_chunks)

        # Read batch
        batch_data = all_lens_vectors[batch_start:batch_end, :, :]

        # Quantize
        batch_quantized = quantize_func(batch_data)

        # Write batch
        quant_dataset[batch_start:batch_end, :, :] = batch_quantized

        # Progress update every 100 batches
        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            print(f"  Batch {batch_idx + 1}/{num_batches} ({100.0 * (batch_idx + 1) / num_batches:.1f}%) | "
                  f"Chunks {batch_start:,}-{batch_end:,} | Elapsed: {elapsed:.1f}s")

    print()

    # Final stats
    elapsed = time.time() - start_time
    file_size_gb = output_path.stat().st_size / (1024**3)
    chunks_per_sec = total_chunks / elapsed

    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output file size: {file_size_gb:.1f} GB")
    print(f"Processing speed: {chunks_per_sec:.0f} chunks/sec")
    print()

    # Verification
    print("Verification (first chunk, AT lens, first 20 values):")
    float_sample = f_in['all_lens_vectors'][0, 0, :20]
    quant_sample = quant_dataset[0, 0, :20]
    expected_sample = quantize_func(float_sample)

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
    elif quantization_type == 'int4':
        expected_range = "[-7, +7]"
    else:  # ternary
        expected_range = "[-1, 0, +1]"
    print(f"  Expected range: {expected_range}")
    print(f"  Top 10 values: {unique[:10] if len(unique) > 10 else unique}")

    print()

    f_in.close()
    f_out.close()

    # VERIFY FILE EXISTS
    if output_path.exists():
        actual_size = output_path.stat().st_size / (1024**3)
        print(f"✓ FILE VERIFIED: {output_path}")
        print(f"  Size: {actual_size:.2f} GB")
    else:
        print(f"❌ ERROR: File not found after creation: {output_path}")
        return False

    print()
    return True


def process_quantization_worker(input_path, output_path, quantization_type, result_queue):
    """Worker function for parallel processing"""
    try:
        success = create_quantized_file(input_path, output_path, quantization_type, batch_size=1000)
        result_queue.put((quantization_type, success, str(output_path)))
    except Exception as e:
        print(f"ERROR in {quantization_type} worker: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put((quantization_type, False, str(e)))


def main():
    # ABSOLUTE PATHS - no ambiguity
    base_dir = Path("/Users/rohanvinaik/genomevault/data/experimental_strands/ERR3239334/hdv_encoding")
    float32_file = base_dir / "encoded_genome_5lenses_3d.h5"

    # Output files
    int8_file = base_dir / "encoded_genome_5lenses_3d_int8.h5"
    int4_file = base_dir / "encoded_genome_5lenses_3d_int4.h5"
    ternary_file = base_dir / "encoded_genome_5lenses_3d_ternary.h5"

    # Check source file exists
    if not float32_file.exists():
        print(f"ERROR: Source file not found: {float32_file}")
        sys.exit(1)

    print("=" * 80)
    print("PARALLEL QUANTIZATION PIPELINE")
    print("=" * 80)
    print(f"Source: {float32_file}")
    print(f"Target directory: {base_dir}")
    print(f"System: 10-core CPU (3 parallel jobs)")
    print("=" * 80)
    print()

    # Delete old files if they exist (to ensure fresh creation)
    for old_file in [int8_file, int4_file, ternary_file]:
        if old_file.exists():
            print(f"Deleting old file: {old_file.name}")
            old_file.unlink()

    print()
    print("Starting parallel quantization processes...")
    print()

    # Create result queue
    result_queue = Queue()

    # Create worker processes (3 simultaneous jobs for 10-core system)
    processes = []
    jobs = [
        (float32_file, int8_file, 'int8'),
        (float32_file, int4_file, 'int4'),
        (float32_file, ternary_file, 'ternary'),
    ]

    start_time = time.time()

    for input_path, output_path, quant_type in jobs:
        p = Process(
            target=process_quantization_worker,
            args=(input_path, output_path, quant_type, result_queue)
        )
        p.start()
        processes.append((quant_type, p))
        print(f"✓ Started {quant_type} worker (PID: {p.pid})")

    print()
    print("Waiting for all workers to complete...")
    print()

    # Wait for all processes
    for quant_type, p in processes:
        p.join()
        print(f"✓ {quant_type} worker finished")

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    elapsed = time.time() - start_time

    print()
    print("=" * 80)
    print("ALL QUANTIZATIONS COMPLETE")
    print("=" * 80)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print()

    # Summary
    print("Results:")
    for quant_type, success, path in results:
        status = "✓ SUCCESS" if success else "❌ FAILED"
        print(f"  {quant_type:8s}: {status} - {path}")

    print()

    # Final verification
    print("=" * 80)
    print("FINAL VERIFICATION")
    print("=" * 80)

    all_files = [
        ("INT8", int8_file, "~54 GB"),
        ("INT4", int4_file, "~25 GB"),
        ("TERNARY", ternary_file, "~70 GB (similar to binary)"),
    ]

    all_exist = True
    for name, filepath, expected_size in all_files:
        if filepath.exists():
            actual_size = filepath.stat().st_size / (1024**3)
            print(f"✓ {name:8s}: EXISTS - {actual_size:.1f} GB (expected {expected_size})")
        else:
            print(f"❌ {name:8s}: NOT FOUND - {filepath}")
            all_exist = False

    print()

    if all_exist:
        print("=" * 80)
        print("SUCCESS: All files created and verified!")
        print("=" * 80)
    else:
        print("=" * 80)
        print("WARNING: Some files missing!")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
