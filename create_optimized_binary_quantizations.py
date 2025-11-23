#!/usr/bin/env python3
"""
Create 4 optimized binary quantizations:

AT-focused (4 lenses: AT, PuPy, AmKe, StWk - drops GC):
  1. AT bipolar (-1/+1, no zero)
  2. AT unipolar (0/1)

GC-focused (4 lenses: GC, PuPy, AmKe, StWk - drops AT):
  3. GC bipolar (-1/+1, no zero)
  4. GC unipolar (0/1)

Rationale: AT and GC are complementary, so we can drop one to save storage.
Optimization: Bipolar {-1, +1} or Unipolar {0, 1} reduces bits per dimension.
"""

import h5py
import numpy as np
from pathlib import Path
import time
import sys
from multiprocessing import Process, Queue

def create_optimized_binary(input_path, output_path, lens_subset, quantization_type, batch_size=1000):
    """
    Create optimized binary file with subset of lenses.

    Args:
        input_path: Path to float32 source file
        output_path: Path for output
        lens_subset: List of lens indices to include (e.g., [0, 2, 3, 4] for AT-focused)
        quantization_type: 'bipolar' (-1/+1) or 'unipolar' (0/1)
        batch_size: Chunks to process at once
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lens_names = ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']
    included_lenses = [lens_names[i] for i in lens_subset]
    focus = "AT" if 0 in lens_subset else "GC"

    print("=" * 80)
    print(f"{focus}-FOCUSED {quantization_type.upper()} QUANTIZATION")
    print("=" * 80)
    print()
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Lenses: {', '.join(included_lenses)} ({len(lens_subset)} lenses)")
    print(f"Dropped: {'GC' if focus == 'AT' else 'AT'} (complementary)")
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
    print(f"  Lenses included: {len(lens_subset)}/5")
    print(f"  Dimensions: {num_dims:,}")
    print()

    # Determine quantization function and dtype
    if quantization_type == 'bipolar':
        # -1 for negative, +1 for positive (NO ZERO - that's the key difference from ternary!)
        def quantize_func(x):
            return np.where(x >= 0, 1, -1).astype(np.int8)
        dtype = np.int8
        value_desc = "{-1, +1} (no zero)"
    else:  # unipolar
        # 0 for negative, 1 for positive/zero
        def quantize_func(x):
            return (x >= 0).astype(np.uint8)
        dtype = np.uint8
        value_desc = "{0, 1}"

    print(f"Quantization: {quantization_type} → {value_desc}")
    print()

    # Create output file
    print("Creating output file...")
    f_out = h5py.File(output_path, 'w')

    # Shape: (chunks, num_lenses_included, dims)
    quant_dataset = f_out.create_dataset(
        'lens_vectors',
        shape=(total_chunks, len(lens_subset), num_dims),
        dtype=dtype,
        chunks=(1, len(lens_subset), num_dims),
        compression='gzip',
        compression_opts=4
    )

    # Metadata
    f_out.attrs['lens_names'] = ','.join(included_lenses)
    f_out.attrs['lens_indices'] = lens_subset
    f_out.attrs['focus'] = focus
    f_out.attrs['quantization_type'] = quantization_type
    f_out.attrs['source_file'] = str(input_path)
    f_out.attrs['num_lenses'] = len(lens_subset)

    print(f"✓ Output file created")
    print()

    # Process in batches
    print(f"Quantizing {len(lens_subset)}-lens data (batch size: {batch_size:,} chunks)...")
    num_batches = (total_chunks + batch_size - 1) // batch_size

    start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_chunks)

        # Read only the lenses we need
        # Shape: (batch_size, all_lenses, dims) → extract → (batch_size, subset_lenses, dims)
        batch_data = all_lens_vectors[batch_start:batch_end, lens_subset, :]

        # Quantize
        batch_quantized = quantize_func(batch_data)

        # Write
        quant_dataset[batch_start:batch_end, :, :] = batch_quantized

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
    print("Verification (first chunk, first lens, first 20 values):")
    float_sample = f_in['all_lens_vectors'][0, lens_subset[0], :20]
    quant_sample = quant_dataset[0, 0, :20]
    expected_sample = quantize_func(float_sample)

    print(f"  Float32:  {float_sample}")
    print(f"  Quantized: {quant_sample}")
    print(f"  Expected:  {expected_sample}")
    print(f"  Match: {'✓ YES' if np.array_equal(quant_sample, expected_sample) else '❌ NO'}")

    # Value distribution
    print()
    print("Value distribution (first 1000 chunks, all lenses):")
    sample_data = quant_dataset[:1000, :, :].flatten()
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


def process_worker(input_path, output_path, lens_subset, quant_type, result_queue):
    """Worker for parallel processing"""
    try:
        lens_names = ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']
        focus = "AT" if 0 in lens_subset else "GC"
        name = f"{focus}_{quant_type}"
        success = create_optimized_binary(input_path, output_path, lens_subset, quant_type, batch_size=1000)
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
    at_bipolar_file = base_dir / "encoded_genome_at_focused_bipolar.h5"
    at_unipolar_file = base_dir / "encoded_genome_at_focused_unipolar.h5"
    gc_bipolar_file = base_dir / "encoded_genome_gc_focused_bipolar.h5"
    gc_unipolar_file = base_dir / "encoded_genome_gc_focused_unipolar.h5"

    # Lens subsets
    # AT-focused: indices [0, 2, 3, 4] = AT, PuPy, AmKe, StWk (drops GC at index 1)
    # GC-focused: indices [1, 2, 3, 4] = GC, PuPy, AmKe, StWk (drops AT at index 0)
    at_lenses = [0, 2, 3, 4]  # AT, PuPy, AmKe, StWk
    gc_lenses = [1, 2, 3, 4]  # GC, PuPy, AmKe, StWk

    # Check source
    if not float32_file.exists():
        print(f"ERROR: Source not found: {float32_file}")
        sys.exit(1)

    print("=" * 80)
    print("OPTIMIZED BINARY QUANTIZATION PIPELINE")
    print("=" * 80)
    print(f"Source: {float32_file}")
    print(f"Target directory: {base_dir}")
    print(f"System: 10-core CPU (4 parallel jobs)")
    print()
    print("Optimization Strategy:")
    print("  - Keep 4 lenses per file (drop complementary AT or GC)")
    print("  - Bipolar {-1, +1} or Unipolar {0, 1} encoding")
    print("  - Storage savings: 4/5 lenses = 20% reduction")
    print()
    print("Creating 4 optimized quantizations:")
    print("  1. AT-focused bipolar  (AT, PuPy, AmKe, StWk) → {-1, +1}")
    print("  2. AT-focused unipolar (AT, PuPy, AmKe, StWk) → {0, 1}")
    print("  3. GC-focused bipolar  (GC, PuPy, AmKe, StWk) → {-1, +1}")
    print("  4. GC-focused unipolar (GC, PuPy, AmKe, StWk) → {0, 1}")
    print("=" * 80)
    print()

    # Delete old files
    for old_file in [at_bipolar_file, at_unipolar_file, gc_bipolar_file, gc_unipolar_file]:
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
        (float32_file, at_bipolar_file, at_lenses, 'bipolar'),   # AT-focused, bipolar
        (float32_file, at_unipolar_file, at_lenses, 'unipolar'), # AT-focused, unipolar
        (float32_file, gc_bipolar_file, gc_lenses, 'bipolar'),   # GC-focused, bipolar
        (float32_file, gc_unipolar_file, gc_lenses, 'unipolar'), # GC-focused, unipolar
    ]

    start_time = time.time()

    for input_path, output_path, lens_subset, quant_type in jobs:
        p = Process(
            target=process_worker,
            args=(input_path, output_path, lens_subset, quant_type, result_queue)
        )
        p.start()
        focus = "AT" if 0 in lens_subset else "GC"
        name = f"{focus}_{quant_type}"
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
    print("ALL OPTIMIZED QUANTIZATIONS COMPLETE")
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
        ("AT_bipolar", at_bipolar_file, "~10-11 GB (4 lenses)"),
        ("AT_unipolar", at_unipolar_file, "~5-6 GB (4 lenses)"),
        ("GC_bipolar", gc_bipolar_file, "~10-11 GB (4 lenses)"),
        ("GC_unipolar", gc_unipolar_file, "~5-6 GB (4 lenses)"),
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
        print("SUCCESS: All optimized binary files created and verified!")
        print("=" * 80)
        print()
        print("Comparison Summary:")
        print(f"  Ternary (5 lenses):      12.9 GB  (sign with zero)")
        print(f"  AT Bipolar (4 lenses):   ~10-11 GB  (no zero, 20% savings)")
        print(f"  GC Bipolar (4 lenses):   ~10-11 GB  (no zero, 20% savings)")
        print(f"  AT Unipolar (4 lenses):  ~5-6 GB  (0/1 encoding)")
        print(f"  GC Unipolar (4 lenses):  ~5-6 GB  (0/1 encoding)")
        print()
        print("Key Difference from Ternary:")
        print("  - Ternary: {-1, 0, +1} (3 values)")
        print("  - Bipolar: {-1, +1} (2 values, no zero)")
        print("  - Unipolar: {0, 1} (2 values, SIMD-ready)")
    else:
        print("=" * 80)
        print("WARNING: Some files missing!")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
