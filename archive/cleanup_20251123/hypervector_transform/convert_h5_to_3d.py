#!/usr/bin/env python3
"""
Convert 5-lens HDF5 from separate 2D datasets to single 3D array.

BEFORE:
  AT_vectors:   (n_chunks, D)
  GC_vectors:   (n_chunks, D)
  PuPy_vectors: (n_chunks, D)
  AmKe_vectors: (n_chunks, D)
  StWk_vectors: (n_chunks, D)
  chunk_keys:   (n_chunks,)

AFTER:
  all_lens_vectors: (n_chunks, 5, D)  where axis 1 = [AT, GC, PuPy, AmKe, StWk]
  chunk_keys:       (n_chunks,)

This enables true batch reading: 1 disk read instead of 5 per chunk.
"""

import h5py
import numpy as np
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def convert_h5_to_3d(input_path: Path, output_path: Path, chunk_size: int = 1000):
    """Convert 5 separate 2D datasets into single 3D array.

    Args:
        input_path: Path to existing 5-dataset H5 file
        output_path: Path for new 3D H5 file
        chunk_size: Number of chunks to process at once (for memory efficiency)
    """
    logger.info("=" * 80)
    logger.info("H5 FILE CONVERSION: 5×2D → 1×3D")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info("")

    # Open input file
    logger.info("Opening input file (read-only)...")
    start_time = time.time()

    with h5py.File(input_path, 'r') as f_in:
        # Get dimensions
        n_chunks = f_in['AT_vectors'].shape[0]
        D = f_in['AT_vectors'].shape[1]

        logger.info(f"  Dimensions: {n_chunks:,} chunks × {D:,} dimensions")
        logger.info(f"  Chunk size for processing: {chunk_size:,} chunks at a time")
        logger.info("")

        # Create output file
        logger.info("Creating output file...")
        with h5py.File(output_path, 'w') as f_out:
            # Create 3D dataset with optimal chunking for row access
            # Chunk shape: (1, 5, D) means reading 1 chunk gets all 5 lenses
            logger.info(f"  Creating 3D dataset: ({n_chunks:,}, 5, {D:,})")
            logger.info(f"  Chunk shape: (1, 5, {D:,}) - optimized for single-chunk reads")
            logger.info(f"  Compression: NONE (uncompressed for speed)")
            logger.info("")

            dset_3d = f_out.create_dataset(
                'all_lens_vectors',
                shape=(n_chunks, 5, D),
                dtype='float32',
                chunks=(1, 5, D)  # Optimize for reading all lenses of 1 chunk
                # No compression - faster conversion and negligible performance impact
            )

            # Copy chunk_keys
            logger.info("Copying chunk_keys dataset...")
            chunk_keys = f_in['chunk_keys'][:]
            f_out.create_dataset('chunk_keys', data=chunk_keys, dtype=h5py.string_dtype())
            logger.info("")

            # Process in batches
            lens_order = ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']
            logger.info(f"Converting lens data (processing {chunk_size:,} chunks at a time)...")
            logger.info(f"Lens order: {lens_order}")
            logger.info("")

            n_batches = (n_chunks + chunk_size - 1) // chunk_size

            for batch_idx in range(n_batches):
                batch_start = batch_idx * chunk_size
                batch_end = min(batch_start + chunk_size, n_chunks)
                batch_n = batch_end - batch_start

                batch_start_time = time.time()

                # Read all 5 lenses for this batch
                batch_data = np.zeros((batch_n, 5, D), dtype=np.float32)

                for lens_idx, lens_name in enumerate(lens_order):
                    batch_data[:, lens_idx, :] = f_in[f'{lens_name}_vectors'][batch_start:batch_end, :]

                # Write to output
                dset_3d[batch_start:batch_end, :, :] = batch_data

                # Progress reporting
                elapsed = time.time() - batch_start_time
                chunks_per_sec = batch_n / elapsed if elapsed > 0 else 0
                total_elapsed = time.time() - start_time
                pct = 100.0 * batch_end / n_chunks

                if batch_idx == 0:
                    # First batch - estimate total time
                    eta_sec = (total_elapsed / batch_end) * (n_chunks - batch_end)
                    logger.info(f"  Batch 0: {batch_n:,} chunks in {elapsed:.1f}s ({chunks_per_sec:.1f} chunks/s)")
                    logger.info(f"  Estimated total time: {eta_sec/60:.1f} min")
                    logger.info("")
                elif batch_idx % max(1, n_batches // 20) == 0:  # Report 20 times
                    remaining_chunks = n_chunks - batch_end
                    eta_sec = remaining_chunks / chunks_per_sec if chunks_per_sec > 0 else 0
                    logger.info(f"  Progress: {batch_end:,}/{n_chunks:,} chunks ({pct:.1f}%) - "
                              f"{chunks_per_sec:.1f} chunks/s - ETA: {eta_sec/60:.1f} min")

    total_time = time.time() - start_time
    logger.info("")
    logger.info(f"✓ Conversion complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("")

    # Verify output
    logger.info("Verifying output file...")
    with h5py.File(output_path, 'r') as f:
        all_lens = f['all_lens_vectors']
        keys = f['chunk_keys']
        logger.info(f"  all_lens_vectors shape: {all_lens.shape}")
        logger.info(f"  chunk_keys count: {len(keys):,}")
        logger.info("")

        # Test read performance
        logger.info("Testing read performance (10 random chunks)...")
        test_indices = np.random.choice(n_chunks, size=10, replace=False)

        read_start = time.time()
        for idx in test_indices:
            data = all_lens[idx, :, :]  # Read all 5 lenses for 1 chunk
        read_time = (time.time() - read_start) / 10

        logger.info(f"  Avg time per chunk (all 5 lenses): {read_time*1000:.2f} ms")
        logger.info("")

    # File size comparison
    input_size = input_path.stat().st_size / 1024**3
    output_size = output_path.stat().st_size / 1024**3
    logger.info("File sizes:")
    logger.info(f"  Input (5×2D):  {input_size:.2f} GB")
    logger.info(f"  Output (1×3D): {output_size:.2f} GB")
    logger.info(f"  Difference:    {(output_size - input_size):.2f} GB ({(output_size/input_size - 1)*100:+.1f}%)")
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ CONVERSION SUCCESSFUL")
    logger.info("=" * 80)
    logger.info("")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert 5-lens H5 file from 5×2D to 1×3D format')
    parser.add_argument('--input', type=str,
                       default='data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome_5lenses.h5',
                       help='Input H5 file path')
    parser.add_argument('--output', type=str,
                       default='data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome_5lenses_3d.h5',
                       help='Output H5 file path')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Number of chunks to process at once')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        exit(1)

    if output_path.exists():
        logger.warning(f"Output file already exists: {output_path}")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Aborted.")
            exit(0)

    convert_h5_to_3d(input_path, output_path, chunk_size=args.batch_size)
