#!/usr/bin/env python3
"""
Convert unipolar uint8 {0, 1} files to bit-packed format.

Uses np.packbits() to pack 8 values into 1 byte.
Reduces file size by ~8× (from 7.6 GB → ~1 GB).
"""

import h5py
import numpy as np
from pathlib import Path
import time
import sys

def create_bitpacked_file(input_path, output_path, batch_size=1000):
    """
    Convert uint8 {0,1} file to bit-packed format.

    Args:
        input_path: Path to unipolar uint8 file
        output_path: Path for bit-packed output
        batch_size: Chunks to process at once
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    print("=" * 80)
    print("BIT-PACKING UNIPOLAR FILE")
    print("=" * 80)
    print()
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        return False

    # Open input
    print("Opening input file...")
    with h5py.File(input_path, 'r') as f_in:
        ds_in = f_in['lens_vectors']

        total_chunks = ds_in.shape[0]
        num_lenses = ds_in.shape[1]
        num_dims = ds_in.shape[2]

        print(f"  Shape: ({total_chunks:,}, {num_lenses}, {num_dims:,})")
        print(f"  Dtype: {ds_in.dtype}")

        if ds_in.dtype != np.uint8:
            print(f"ERROR: Expected uint8, got {ds_in.dtype}")
            return False

        # Verify dimensions are multiple of 8 (required for packbits)
        if num_dims % 8 != 0:
            print(f"WARNING: Dimensions ({num_dims}) not multiple of 8")
            print(f"  Will pad to {((num_dims + 7) // 8) * 8}")

        # Calculate packed dimensions
        packed_dims = (num_dims + 7) // 8  # Round up to nearest byte

        print()
        print(f"Bit-packing: {num_dims:,} dims → {packed_dims:,} bytes per lens")
        print(f"  Compression: {num_dims / packed_dims:.1f}× (8 bits → 1 bit)")
        print()

        # Create output file
        print("Creating output file...")
        with h5py.File(output_path, 'w') as f_out:
            # Create bit-packed dataset (NO GZIP - bit-packed data is already incompressible!)
            ds_out = f_out.create_dataset(
                'lens_vectors_packed',
                shape=(total_chunks, num_lenses, packed_dims),
                dtype=np.uint8,
                chunks=(1, num_lenses, packed_dims)
                # compression=None  (default - bit-packed data has max entropy, gzip won't help)
            )

            # Copy metadata from input
            for key in f_in.attrs.keys():
                f_out.attrs[key] = f_in.attrs[key]

            # Add bit-packing metadata
            f_out.attrs['bit_packed'] = True
            f_out.attrs['original_dims'] = num_dims
            f_out.attrs['packed_dims'] = packed_dims

            print("  ✓ Output file created")
            print()

            # Process in batches
            print(f"Bit-packing data (batch size: {batch_size:,} chunks)...")
            num_batches = (total_chunks + batch_size - 1) // batch_size

            start_time = time.time()

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_chunks)

                # Read batch
                batch_data = ds_in[batch_start:batch_end, :, :]

                # Verify values are {0, 1}
                if batch_idx == 0:
                    unique = np.unique(batch_data)
                    if not np.array_equal(unique, np.array([0, 1])):
                        print(f"WARNING: Expected {{0, 1}}, found {unique}")

                # Pad if necessary
                if num_dims % 8 != 0:
                    pad_size = packed_dims * 8 - num_dims
                    padding = np.zeros((batch_data.shape[0], batch_data.shape[1], pad_size), dtype=np.uint8)
                    batch_data = np.concatenate([batch_data, padding], axis=2)

                # Bit-pack: 8 values → 1 byte
                # packbits operates on last axis by default
                packed = np.packbits(batch_data, axis=-1)

                # Write
                ds_out[batch_start:batch_end, :, :] = packed

                # Progress
                if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
                    elapsed = time.time() - start_time
                    pct = 100.0 * (batch_idx + 1) / num_batches
                    print(f"  Batch {batch_idx + 1}/{num_batches} ({pct:.1f}%) | "
                          f"Chunks {batch_start:,}-{batch_end:,} | Elapsed: {elapsed:.1f}s")

            print()

    # Final stats
    elapsed = time.time() - start_time

    input_size = input_path.stat().st_size / (1024**3)
    output_size = output_path.stat().st_size / (1024**3)
    compression = input_size / output_size

    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Input size:  {input_size:.2f} GB")
    print(f"Output size: {output_size:.2f} GB")
    print(f"Compression: {compression:.2f}×")
    print()

    # Verification
    print("Verification (unpacking first chunk, first lens, first 80 bits):")
    with h5py.File(input_path, 'r') as f_in:
        with h5py.File(output_path, 'r') as f_out:
            original = f_in['lens_vectors'][0, 0, :80]
            packed = f_out['lens_vectors_packed'][0, 0, :10]  # 80 bits = 10 bytes
            unpacked = np.unpackbits(packed)[:80]  # Unpack and take first 80

            print(f"  Original: {original}")
            print(f"  Packed:   {packed} (hex: {packed.tobytes().hex()})")
            print(f"  Unpacked: {unpacked}")
            print(f"  Match: {'✓ YES' if np.array_equal(original, unpacked) else '❌ NO'}")

    print()
    return True


def main():
    base_dir = Path("/Users/rohanvinaik/genomevault/data/experimental_strands/ERR3239334/hdv_encoding")

    files_to_pack = [
        ("AT-focused unipolar",
         base_dir / "encoded_genome_at_focused_unipolar.h5",
         base_dir / "encoded_genome_at_focused_unipolar_packed.h5"),
        ("GC-focused unipolar",
         base_dir / "encoded_genome_gc_focused_unipolar.h5",
         base_dir / "encoded_genome_gc_focused_unipolar_packed.h5"),
    ]

    print("=" * 80)
    print("BIT-PACKING PIPELINE")
    print("=" * 80)
    print()
    print("Converting unipolar uint8 files to bit-packed format")
    print("Using np.packbits() - 8× compression (8 bits → 1 bit)")
    print()
    print(f"Files to process: {len(files_to_pack)}")
    print()

    for name, input_file, output_file in files_to_pack:
        print(f"Processing: {name}")
        print()

        if not input_file.exists():
            print(f"  ⚠️  Input not found: {input_file.name}")
            print()
            continue

        if output_file.exists():
            print(f"  Deleting existing output: {output_file.name}")
            output_file.unlink()

        success = create_bitpacked_file(input_file, output_file, batch_size=1000)

        if success:
            print(f"  ✓ {name} complete")
        else:
            print(f"  ❌ {name} failed")

        print()

    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()

    total_before = 0
    total_after = 0

    for name, input_file, output_file in files_to_pack:
        if input_file.exists() and output_file.exists():
            before = input_file.stat().st_size / (1024**3)
            after = output_file.stat().st_size / (1024**3)
            total_before += before
            total_after += after

            print(f"{name}:")
            print(f"  Before: {before:.2f} GB")
            print(f"  After:  {after:.2f} GB")
            print(f"  Saved:  {before - after:.2f} GB ({100*(before-after)/before:.1f}%)")
            print()

    if total_before > 0:
        print(f"Total:")
        print(f"  Before: {total_before:.2f} GB")
        print(f"  After:  {total_after:.2f} GB")
        print(f"  Saved:  {total_before - total_after:.2f} GB ({100*(total_before-total_after)/total_before:.1f}%)")
        print()

    print("=" * 80)


if __name__ == '__main__':
    main()
