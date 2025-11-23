#!/usr/bin/env python3
"""
Split Ternary Quantization: Create Two Orthogonal 3D Hypervectors

Input:  3-bank ternary (AT, GC, Hinge)
Output: 6-bank ternary (two 3D vectors)

Vector 1 (GC-dominant): [AT=0, GC, Hinge] - 3 ternary banks
Vector 2 (AT-dominant): [AT, GC=0, Hinge] - 3 ternary banks

Architecture:
- Hinge appears in BOTH vectors (grounding context)
- AT and GC are orthogonal (no cross-contamination)
- Each vector specializes in a biophysical regime
- √2 SNR improvement per vector

Author: Claude Code
Date: November 22, 2025
"""

import h5py
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_split_ternary_streaming(
    input_path: Path,
    output_path: Path,
    batch_size: int = 10000
):
    """
    Create split ternary encoding with streaming to avoid memory overflow.

    Memory usage: ~600 MB per batch (10k chunks × 6 banks × 5120 dims × int8)
    """
    logger.info("=" * 80)
    logger.info("SPLIT TERNARY QUANTIZATION (STREAMING)")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Batch size: {batch_size:,} chunks")
    logger.info("")

    # Get dimensions from input file
    with h5py.File(input_path, 'r') as f_in:
        total_chunks, num_banks, D = f_in['all_bank_vectors'].shape
        logger.info(f"Input shape: {total_chunks:,} chunks × {num_banks} banks × {D:,} dimensions")

        if num_banks != 3:
            logger.error(f"Expected 3 banks (AT, GC, Hinge), got {num_banks}")
            return 1

        batch_memory_gb = (batch_size * 6 * D) / (1024**3)
        logger.info(f"Memory per batch: {batch_memory_gb:.2f} GB")
        logger.info("")

    # Create output HDF5 file
    logger.info("Creating output HDF5 structure...")
    with h5py.File(output_path, 'w') as f_out:
        # Create dataset for 6 ternary banks
        dset = f_out.create_dataset(
            'split_ternary_vectors',
            shape=(total_chunks, 6, D),
            dtype='int8',
            chunks=(1, 6, D),
            compression='gzip',
            compression_opts=6
        )

        # Metadata
        dset.attrs['architecture'] = 'split_ternary_6bank'
        dset.attrs['num_banks'] = 6
        dset.attrs['dimension'] = D
        dset.attrs['bank_names'] = [
            'Vector1_AT_zeroed',
            'Vector1_GC',
            'Vector1_Hinge',
            'Vector2_AT',
            'Vector2_GC_zeroed',
            'Vector2_Hinge'
        ]
        dset.attrs['vector1_indices'] = [0, 1, 2]  # GC-dominant
        dset.attrs['vector2_indices'] = [3, 4, 5]  # AT-dominant
        dset.attrs['created'] = datetime.now().isoformat()
        dset.attrs['source_file'] = str(input_path)
        dset.attrs['description'] = 'Two orthogonal 3D ternary hypervectors: GC-dominant and AT-dominant'

        # Copy chunk_keys if exists
        with h5py.File(input_path, 'r') as f_in:
            if 'chunk_keys' in f_in:
                logger.info("Copying chunk_keys...")
                f_out.create_dataset(
                    'chunk_keys',
                    data=f_in['chunk_keys'][:],
                    compression='gzip',
                    compression_opts=6
                )

    logger.info("✓ Output file created")
    logger.info("")

    # Process in batches
    num_batches = (total_chunks + batch_size - 1) // batch_size
    logger.info(f"Processing {total_chunks:,} chunks in {num_batches} batches...")
    logger.info("")

    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'a') as f_out:
        dset_in = f_in['all_bank_vectors']
        dset_out = f_out['split_ternary_vectors']

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_chunks)
            actual_batch_size = batch_end - batch_start

            logger.info(f"Batch {batch_idx + 1}/{num_batches}: chunks {batch_start:,} - {batch_end-1:,} ({actual_batch_size:,} chunks)")

            # Load one batch
            ternary_batch = dset_in[batch_start:batch_end, :, :]  # Shape: (batch, 3, D)

            # Create split ternary batch
            split_batch = np.zeros((actual_batch_size, 6, D), dtype=np.int8)

            # Vector 1 (GC-dominant): Banks 0-2
            split_batch[:, 0, :] = 0                      # AT zeroed
            split_batch[:, 1, :] = ternary_batch[:, 1, :] # GC preserved
            split_batch[:, 2, :] = ternary_batch[:, 2, :] # Hinge preserved

            # Vector 2 (AT-dominant): Banks 3-5
            split_batch[:, 3, :] = ternary_batch[:, 0, :] # AT preserved
            split_batch[:, 4, :] = 0                      # GC zeroed
            split_batch[:, 5, :] = ternary_batch[:, 2, :] # Hinge preserved

            # Write to disk
            dset_out[batch_start:batch_end, :, :] = split_batch

            # Free memory
            del ternary_batch, split_batch

            logger.info(f"  ✓ Written to disk, batch memory freed")

    logger.info("")
    logger.info("=" * 80)
    logger.info("SPLIT TERNARY QUANTIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info("")

    # Report statistics
    file_size_gb = output_path.stat().st_size / (1024**3)
    original_size_gb = input_path.stat().st_size / (1024**3)
    ratio = file_size_gb / original_size_gb

    logger.info(f"Input file:  {input_path}")
    logger.info(f"Input size:  {original_size_gb:.2f} GB")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Output size: {file_size_gb:.2f} GB")
    logger.info(f"Size ratio:  {ratio:.2f}× (expected ~2× for doubling banks)")
    logger.info("")
    logger.info("Structure:")
    logger.info("  Vector 1 (GC-dominant): Banks 0-2 = [AT=0, GC, Hinge]")
    logger.info("  Vector 2 (AT-dominant): Banks 3-5 = [AT, GC=0, Hinge]")
    logger.info("")
    logger.info("✓ All done! Two orthogonal 3D ternary hypervectors created.")

    return 0


def main():
    """Create split ternary encoding"""

    # Paths
    base_dir = Path(__file__).parent.parent  # Go up to hdc_experimentation/
    input_path = base_dir / "output/encoded_genome_3banks.h5"
    output_path = base_dir / "output/encoded_genome_6banks_split_ternary.h5"

    # Verify input exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    # Remove old output if exists
    if output_path.exists():
        logger.info(f"Removing existing output file: {output_path}")
        output_path.unlink()

    # Create split ternary encoding
    return create_split_ternary_streaming(input_path, output_path, batch_size=10000)


if __name__ == '__main__':
    exit(main())
