#!/usr/bin/env python3
"""
Create 4 new experimental binary split quantizations:
1. AT bipolar (-1/+1) - lens 0 only
2. GC bipolar (-1/+1) - lens 1 only
3. AT unipolar (0/1) - lens 0 only
4. GC unipolar (0/1) - lens 1 only

For scientific comparison against:
- Unified binary (70 GB, 3 values)
- Ternary (12.9 GB, 3 values)
"""

import h5py
import numpy as np
from pathlib import Path
import time
import sys
from multiprocessing import Process, Queue

def create_split_quantization(input_path, output_path, lens_idx, quantization_type, batch_size=1000):
    """
    Create a single-lens quantized file (AT or GC only).

    Args:
        input_path: Path to float32 source file
        output_path: Path for output
        lens_idx: 0 (AT) or 1 (GC)
        quantization_type: 'bipolar' (-1/+1) or 'unipolar' (0/1)
        batch_size: Chunks to process at once
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lens_names = ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']
    lens_name = lens_names[lens_idx]

    print("=" * 80)
    print(f"CREATING {lens_name} {quantization_type.upper()} FILE")
    print("=" * 80)
    print()
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Lens:   {lens_name} (index {lens_idx})")
    print(f"Type:   {quantization_type}")
    print()

    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        return False

    # Open source
    print("Opening source file...")
    f_in = h5py.File(input_path, 'r')
    all_lens_vectors = f_in['all_lens_vectors']

    total_chunks = all_lens_vectors.shape[0]
    num_dims = all_lens_vectors.shape[2]

    print(f"  Total chunks: {total_chunks:,}")
    print(f"  Target lens: {lens_name}")
    print(f"  Dimensions: {num_dims:,}")
    print()

    # Determine quantization function and dtype
    if quantization_type == 'bipolar':
        # -1 for negative, +1 for positive, 0 for zero
        def quantize_func(x):
            return np.sign(x).astype(np.int8)
        dtype = np.int8
        value_desc = "{-1, 0, +1}"
    else:  # unipolar
        # 0 for negative/zero, 1 for positive
        def quantize_func(x):
            return (x > 0).astype(np.uint8)
        dtype = np.uint8
        value_desc = "{0, 1}"

    print(f"Quantization: {quantization_type} → {value_desc}")
    print()

    # Create output file
    print("Creating output file...")
    f_out = h5py.File(output_path, 'w')

    # Single lens, so shape is (chunks, dims) not (chunks, lenses, dims)
    quant_dataset = f_out.create_dataset(
        'lens_vectors',
        shape=(total_chunks, num_dims),
        dtype=dtype,
        chunks=(1, num_dims),
        compression='gzip',
        compression_opts=4
    )

    # Metadata
    f_out.attrs['lens_name'] = lens_name
    f_out.attrs['lens_index'] = lens_idx
    f_out.attrs['quantization_type'] = quantization_type
    f_out.attrs['source_file'] = str(input_path)

    print(f"✓ Output file created")
    print()

    # Process in batches
    print(f"Quantizing {lens_name} lens data (batch size: {batch_size:,} chunks)...")
    num_batches = (total_chunks + batch_size - 1) // batch_size

    start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_chunks)

        # Read ONLY the target lens from this batch
        # Shape: (batch_size, 1 lens, dims) → extract → (batch_size, dims)
        batch_data = all_lens_vectors[batch_start:batch_end, lens_idx, :]

        # Quantize
        batch_quantized = quantize_func(batch_data)

        # Write
        quant_dataset[batch_start:batch_end, :] = batch_quantized

        # Progress
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
    print("Verification (first chunk, first 20 values):")
    float_sample = f_in['all_lens_vectors'][0, lens_idx, :20]
    quant_sample = quant_dataset[0, :20]
    expected_sample = quantize_func(float_sample)

    print(f"  Float32:  {float_sample}")
    print(f"  Quantized: {quant_sample}")
    print(f"  Expected:  {expected_sample}")
    print(f"  Match: {'✓ YES' if np.array_equal(quant_sample, expected_sample) else '❌ NO'}")

    # Value distribution
    print()
    print("Value distribution (first 1000 chunks):")
    sample_data = quant_dataset[:1000, :].flatten()
    unique, counts = np.unique(sample_data, return_counts=True)
    print(f"  Unique values: {len(unique)}")
    print(f"  Value range: [{unique.min()}, {unique.max()}]")
    print(f"  Expected range: {value_desc}")
    print(f"  Values: {unique}")
    print()

    f_in.close()
    f_out.close()

    # Verify exists
    if output_path.exists():
        actual_size = output_path.stat().st_size / (1024**3)
        print(f"✓ FILE VERIFIED: {output_path}")
        print(f"  Size: {actual_size:.2f} GB")
    else:
        print(f"❌ ERROR: File not found after creation: {output_path}")
        return False

    print()
    return True


def process_worker(input_path, output_path, lens_idx, quant_type, result_queue):
    """Worker for parallel processing"""
    try:
        lens_names = ['AT', 'GC']
        name = f"{lens_names[lens_idx]}_{quant_type}"
        success = create_split_quantization(input_path, output_path, lens_idx, quant_type, batch_size=1000)
        result_queue.put((name, success, str(output_path)))
    except Exception as e:
        print(f"ERROR in {name} worker: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put((name, False, str(e)))


def main():
    # Absolute paths
    base_dir = Path("/Users/rohanvinaik/genomevault/data/experimental_strands/ERR3239334/hdv_encoding")
    float32_file = base_dir / "encoded_genome_5lenses_3d.h5"

    # Output files
    at_bipolar_file = base_dir / "encoded_genome_at_bipolar.h5"
    gc_bipolar_file = base_dir / "encoded_genome_gc_bipolar.h5"
    at_unipolar_file = base_dir / "encoded_genome_at_unipolar.h5"
    gc_unipolar_file = base_dir / "encoded_genome_gc_unipolar.h5"

    # Check source
    if not float32_file.exists():
        print(f"ERROR: Source not found: {float32_file}")
        sys.exit(1)

    print("=" * 80)
    print("BINARY SPLIT QUANTIZATION PIPELINE")
    print("=" * 80)
    print(f"Source: {float32_file}")
    print(f"Target directory: {base_dir}")
    print(f"System: 10-core CPU (4 parallel jobs)")
    print()
    print("Creating 4 experimental quantizations:")
    print("  1. AT bipolar  (-1/+1)")
    print("  2. GC bipolar  (-1/+1)")
    print("  3. AT unipolar (0/1)")
    print("  4. GC unipolar (0/1)")
    print("=" * 80)
    print()

    # Delete old files
    for old_file in [at_bipolar_file, gc_bipolar_file, at_unipolar_file, gc_unipolar_file]:
        if old_file.exists():
            print(f"Deleting old file: {old_file.name}")
            old_file.unlink()

    print()
    print("Starting parallel quantization processes...")
    print()

    # Create result queue
    result_queue = Queue()

    # Create workers (4 simultaneous jobs)
    processes = []
    jobs = [
        (float32_file, at_bipolar_file, 0, 'bipolar'),   # AT lens, bipolar
        (float32_file, gc_bipolar_file, 1, 'bipolar'),   # GC lens, bipolar
        (float32_file, at_unipolar_file, 0, 'unipolar'), # AT lens, unipolar
        (float32_file, gc_unipolar_file, 1, 'unipolar'), # GC lens, unipolar
    ]

    start_time = time.time()

    for input_path, output_path, lens_idx, quant_type in jobs:
        p = Process(
            target=process_worker,
            args=(input_path, output_path, lens_idx, quant_type, result_queue)
        )
        p.start()
        lens_names = ['AT', 'GC']
        name = f"{lens_names[lens_idx]}_{quant_type}"
        processes.append((name, p))
        print(f"✓ Started {name} worker (PID: {p.pid})")

    print()
    print("Waiting for all workers to complete...")
    print()

    # Wait for all
    for name, p in processes:
        p.join()
        print(f"✓ {name} worker finished")

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    elapsed = time.time() - start_time

    print()
    print("=" * 80)
    print("ALL SPLIT QUANTIZATIONS COMPLETE")
    print("=" * 80)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print()

    # Summary
    print("Results:")
    for name, success, path in results:
        status = "✓ SUCCESS" if success else "❌ FAILED"
        print(f"  {name:14s}: {status} - {path}")

    print()

    # Final verification
    print("=" * 80)
    print("FINAL VERIFICATION")
    print("=" * 80)

    all_files = [
        ("AT_bipolar", at_bipolar_file, "~6-7 GB"),
        ("GC_bipolar", gc_bipolar_file, "~6-7 GB"),
        ("AT_unipolar", at_unipolar_file, "~3-4 GB"),
        ("GC_unipolar", gc_unipolar_file, "~3-4 GB"),
    ]

    all_exist = True
    for name, filepath, expected_size in all_files:
        if filepath.exists():
            actual_size = filepath.stat().st_size / (1024**3)
            print(f"✓ {name:14s}: EXISTS - {actual_size:.1f} GB (expected {expected_size})")
        else:
            print(f"❌ {name:14s}: NOT FOUND - {filepath}")
            all_exist = False

    print()

    if all_exist:
        print("=" * 80)
        print("SUCCESS: All split binary files created and verified!")
        print("=" * 80)
        print()
        print("Comparison Summary:")
        print(f"  Unified Binary:  70.0 GB  (5 lenses, 3 values)")
        print(f"  Ternary:         12.9 GB  (5 lenses, 3 values) 🔥")
        print(f"  AT Bipolar:      ~6-7 GB  (1 lens, 3 values)")
        print(f"  GC Bipolar:      ~6-7 GB  (1 lens, 3 values)")
        print(f"  AT Unipolar:     ~3-4 GB  (1 lens, 2 values)")
        print(f"  GC Unipolar:     ~3-4 GB  (1 lens, 2 values)")
        print()
        print("Total split binary storage: ~13-15 GB (AT bipolar + GC bipolar)")
        print("Total unipolar storage:     ~6-8 GB (AT unipolar + GC unipolar)")
    else:
        print("=" * 80)
        print("WARNING: Some files missing!")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
