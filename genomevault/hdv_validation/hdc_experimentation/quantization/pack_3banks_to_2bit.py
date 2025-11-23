#!/usr/bin/env python3
"""
Pack 3-Bank Ternary Encoding to 2-Bit Representation

Converts int8 ternary {-1, 0, +1} to LOSSLESS 2-bit packed format.

Input:  encoded_genome_3banks.h5 (5.31 GB, int8 ternary)
Output: encoded_genome_3banks_packed.h5 (~1.3 GB, uint8 2-bit packed)

Encoding: {-1 → 0b00, 0 → 0b01, +1 → 0b10}
Packing:  4 ternary values per byte (4× compression)
Gzip:     Additional ~2.5× compression on top

Expected final size: ~1.3 GB (10× smaller than uncompressed int8)

Author: Claude Code
Date: November 22, 2025
"""

import h5py
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import sys

# Import the 2-bit packing functions
sys.path.insert(0, str(Path(__file__).parent))
from ternary_2bit_packing import pack_3bank_chunk, validate_packing_lossless

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def pack_3banks_streaming(
    input_path: Path,
    output_path: Path,
    batch_size: int = 10000  # Process 10k chunks at a time
):
    """
    Pack 3-bank ternary encoding to 2-bit representation with streaming.

    Memory usage: ~150 MB max (only holds one batch at a time)
    """
    logger.info("=" * 80)
    logger.info("2-BIT TERNARY PACKING (STREAMING)")
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
        logger.info(f"Input dtype: {f_in['all_bank_vectors'].dtype}")

        # Verify dimensions work for packing
        if D % 4 != 0:
            logger.error(f"Dimension D={D} must be multiple of 4 for 2-bit packing!")
            return 1

        packed_D = D // 4
        logger.info(f"Packed dimension: {packed_D:,} bytes per bank")
        logger.info("")

    # Validate packing on a small sample first
    logger.info("Validating 2-bit packing on sample...")
    with h5py.File(input_path, 'r') as f_in:
        sample_chunk = f_in['all_bank_vectors'][0, 0, :]  # First bank of first chunk
        try:
            is_lossless = validate_packing_lossless(sample_chunk)
            logger.info(f"✓ Packing validation PASSED (lossless: {is_lossless})")
        except AssertionError as e:
            logger.error(f"✗ Packing validation FAILED: {e}")
            return 1
    logger.info("")

    # Create output HDF5 file
    logger.info("Creating output HDF5 structure...")
    with h5py.File(output_path, 'w') as f_out:
        # Create dataset for packed banks
        dset = f_out.create_dataset(
            'packed_bank_vectors',
            shape=(total_chunks, num_banks, packed_D),
            dtype='uint8',
            chunks=(1, num_banks, packed_D),  # One chunk at a time
            compression='gzip',
            compression_opts=9  # Maximum compression for ternary data
        )

        # Metadata
        dset.attrs['format'] = '2bit_ternary'
        dset.attrs['encoding'] = '{-1 → 0b00, 0 → 0b01, +1 → 0b10}'
        dset.attrs['num_banks'] = num_banks
        dset.attrs['dimension_original'] = D
        dset.attrs['dimension_packed'] = packed_D
        dset.attrs['packing_ratio'] = 4.0
        dset.attrs['bank_names'] = ['Hydrophobic', 'MajorGroove', 'Hinge']
        dset.attrs['created'] = datetime.now().isoformat()
        dset.attrs['source_file'] = str(input_path)
        dset.attrs['lossless'] = True

        # Copy position data if exists
        with h5py.File(input_path, 'r') as f_in:
            if 'positions' in f_in:
                logger.info("Copying position data...")
                f_out.create_dataset(
                    'positions',
                    data=f_in['positions'][:],
                    compression='gzip',
                    compression_opts=6
                )
                logger.info("✓ Position data copied")

    logger.info("✓ Output file created")
    logger.info("")

    # Pack in batches
    num_batches = (total_chunks + batch_size - 1) // batch_size
    logger.info(f"Processing {total_chunks:,} chunks in {num_batches} batches...")
    logger.info("")

    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'a') as f_out:
        dset_in = f_in['all_bank_vectors']
        dset_out = f_out['packed_bank_vectors']

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_chunks)
            actual_batch_size = batch_end - batch_start

            logger.info(f"Batch {batch_idx + 1}/{num_batches}: chunks {batch_start:,} - {batch_end-1:,} ({actual_batch_size:,} chunks)")

            # Load one batch into memory
            ternary_batch = dset_in[batch_start:batch_end, :, :]  # Shape: (batch_size, 3, D)

            # Pack each chunk in the batch
            packed_batch = np.zeros((actual_batch_size, num_banks, packed_D), dtype=np.uint8)

            for i in range(actual_batch_size):
                # Extract 3 banks for this chunk
                bank1 = ternary_batch[i, 0, :]
                bank2 = ternary_batch[i, 1, :]
                bank3 = ternary_batch[i, 2, :]

                # Pack to 2-bit
                packed1, packed2, packed3 = pack_3bank_chunk(bank1, bank2, bank3)

                # Store in batch array
                packed_batch[i, 0, :] = packed1
                packed_batch[i, 1, :] = packed2
                packed_batch[i, 2, :] = packed3

            # Write packed batch to output file
            dset_out[batch_start:batch_end, :, :] = packed_batch

            # Free memory
            del ternary_batch, packed_batch

            logger.info(f"  ✓ Packed and written to disk, batch memory freed")

    logger.info("")
    logger.info("=" * 80)
    logger.info("2-BIT PACKING COMPLETE")
    logger.info("=" * 80)
    logger.info("")

    # Report statistics
    file_size_gb = output_path.stat().st_size / (1024**3)
    original_size_gb = input_path.stat().st_size / (1024**3)
    reduction = original_size_gb / file_size_gb

    logger.info(f"Input file:  {input_path}")
    logger.info(f"Input size:  {original_size_gb:.2f} GB")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Output size: {file_size_gb:.2f} GB")
    logger.info(f"Reduction:   {reduction:.1f}× smaller")
    logger.info("")
    logger.info("✓ All done! Use unpack_2bit_to_ternary() to decode.")

    return 0


def main():
    """Pack 3-bank ternary to 2-bit format"""

    # Paths
    input_path = Path("output/encoded_genome_3banks.h5")
    output_path = Path("output/encoded_genome_3banks_packed.h5")

    # Verify input exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    # Remove old output if exists
    if output_path.exists():
        logger.info(f"Removing existing output file: {output_path}")
        output_path.unlink()

    # Pack with streaming (max ~150 MB RAM usage)
    return pack_3banks_streaming(input_path, output_path, batch_size=10000)


if __name__ == '__main__':
    exit(main())
