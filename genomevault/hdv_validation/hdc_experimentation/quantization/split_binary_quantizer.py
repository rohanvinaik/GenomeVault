#!/usr/bin/env python3
"""
Split Binary Quantization: 3-Bank Ternary → 6-Bank Binary (STREAMING)

Processes in batches to avoid loading 96 GB into RAM.
Memory usage: ~1-2 GB max (only holds one batch at a time)
"""

import h5py
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def split_ternary_to_binary(ternary_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split ternary {-1, 0, +1} into two binary {0, 1} vectors

    Transformation:
      +1 → (1, 0)  positive bank active, negative bank inactive
      -1 → (0, 1)  positive bank inactive, negative bank active
       0 → (0, 0)  both banks inactive (sparsity preserved!)
    """
    positive = (ternary_data == 1).astype(np.uint8)
    negative = (ternary_data == -1).astype(np.uint8)
    return positive, negative


def convert_to_split_binary_streaming(
    input_path: Path,
    output_path: Path,
    batch_size: int = 10000  # Process 10k chunks at a time (~300 MB per batch)
):
    """
    Convert 3-bank ternary encoding to 6-bank split binary using streaming.

    Memory usage: ~1-2 GB max (only holds one batch at a time)
    """
    logger.info("="*80)
    logger.info("SPLIT BINARY QUANTIZATION (STREAMING)")
    logger.info("="*80)
    logger.info("")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Batch size: {batch_size:,} chunks")
    logger.info("")

    # Get dimensions from input file (don't load data yet!)
    with h5py.File(input_path, 'r') as f_in:
        total_chunks, num_banks, dimension = f_in['all_bank_vectors'].shape
        logger.info(f"Input shape: {total_chunks:,} chunks × {num_banks} banks × {dimension:,} dimensions")
        logger.info(f"Memory per batch: {(batch_size * 6 * dimension) / (1024**3):.2f} GB")
        logger.info("")

    # Create output HDF5 file with correct shape (but empty)
    logger.info("Creating output HDF5 structure...")
    with h5py.File(output_path, 'w') as f_out:
        dset = f_out.create_dataset(
            'binary_bank_vectors',
            shape=(total_chunks, 6, dimension),
            dtype='uint8',
            chunks=(1, 6, dimension),  # One chunk at a time for efficient writes
            compression='gzip',
            compression_opts=6  # Level 6 for balance of speed/compression
        )

        # Metadata
        dset.attrs['architecture'] = 'split_binary_6bank'
        dset.attrs['num_banks'] = 6
        dset.attrs['dimension'] = dimension
        dset.attrs['bank_names'] = [
            'Hydrophobic_A', 'Hydrophobic_T',
            'MajorGroove_G', 'MajorGroove_C',
            'Hinge_pos', 'Hinge_neg'
        ]
        dset.attrs['created'] = datetime.now().isoformat()
        dset.attrs['source_file'] = str(input_path)

    logger.info("✓ Output file created")
    logger.info("")

    # Process in batches
    num_batches = (total_chunks + batch_size - 1) // batch_size
    logger.info(f"Processing {total_chunks:,} chunks in {num_batches} batches...")
    logger.info("")

    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'a') as f_out:
        dset_in = f_in['all_bank_vectors']
        dset_out = f_out['binary_bank_vectors']

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_chunks)
            actual_batch_size = batch_end - batch_start

            logger.info(f"Batch {batch_idx + 1}/{num_batches}: chunks {batch_start:,} - {batch_end-1:,} ({actual_batch_size:,} chunks)")

            # Load one batch into memory
            ternary_batch = dset_in[batch_start:batch_end, :, :]  # Shape: (batch_size, 3, D)

            # Split each of the 3 banks
            hydrophobic_T, hydrophobic_A = split_ternary_to_binary(ternary_batch[:, 0, :])
            majorgroove_G, majorgroove_C = split_ternary_to_binary(ternary_batch[:, 1, :])
            hinge_pos, hinge_neg = split_ternary_to_binary(ternary_batch[:, 2, :])

            # Stack into 6-bank array (still in memory but only this batch)
            binary_batch = np.stack([
                hydrophobic_A,     # Bank 0: A detector
                hydrophobic_T,     # Bank 1: T detector
                majorgroove_G,     # Bank 2: G detector
                majorgroove_C,     # Bank 3: C detector
                hinge_pos,         # Bank 4: YR dinucleotide
                hinge_neg          # Bank 5: RY dinucleotide
            ], axis=1)  # Shape: (batch_size, 6, D)

            # Write batch to output file
            dset_out[batch_start:batch_end, :, :] = binary_batch

            # Free memory
            del ternary_batch, hydrophobic_T, hydrophobic_A
            del majorgroove_G, majorgroove_C, hinge_pos, hinge_neg, binary_batch

            logger.info(f"  ✓ Written to disk, batch memory freed")

    logger.info("")
    logger.info("="*80)
    logger.info("STREAMING QUANTIZATION COMPLETE")
    logger.info("="*80)
    logger.info("")

    file_size_gb = output_path.stat().st_size / (1024**3)
    logger.info(f"Output file: {output_path}")
    logger.info(f"File size: {file_size_gb:.2f} GB")
    logger.info("")

    # Analyze sparsity from small sample
    logger.info("Analyzing sparsity from sample of 1,000 chunks...")
    with h5py.File(output_path, 'r') as f:
        sample_indices = np.random.choice(total_chunks, size=1000, replace=False)
        sample_indices = np.sort(sample_indices)
        sample_data = f['binary_bank_vectors'][sample_indices, :, :]

        bank_names = [
            'Hydrophobic_A (A positions)', 'Hydrophobic_T (T positions)',
            'MajorGroove_G (G positions)', 'MajorGroove_C (C positions)',
            'Hinge_pos (YR steps)', 'Hinge_neg (RY steps)'
        ]

        for bank_idx in range(6):
            bank_data = sample_data[:, bank_idx, :]
            ones_pct = 100.0 * (bank_data == 1).sum() / bank_data.size
            zeros_pct = 100.0 * (bank_data == 0).sum() / bank_data.size
            logger.info(f"  {bank_names[bank_idx]}:")
            logger.info(f"    Active (1): {ones_pct:5.2f}%  |  Inactive (0): {zeros_pct:5.2f}%")

    logger.info("")
    logger.info("✓ All done!")


def main():
    """Convert 3-bank ternary to 6-bank split binary with streaming"""

    # Paths
    input_path = Path("genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_3banks.h5")
    output_path = Path("genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_6banks_split_binary.h5")

    # Verify input exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Please run the 3-bank encoder first!")
        return 1

    # Remove old output if exists
    if output_path.exists():
        logger.info(f"Removing existing output file: {output_path}")
        output_path.unlink()

    # Convert with streaming (max ~2 GB RAM usage)
    convert_to_split_binary_streaming(input_path, output_path, batch_size=10000)

    return 0


if __name__ == '__main__':
    main()
