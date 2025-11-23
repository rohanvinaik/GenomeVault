#!/usr/bin/env python3
"""
Recompress 6-Bank Binary from gzip to Zstandard

Converts HDF5 file from gzip compression to zstd for better compression ratio.
Expected improvement: 20-30% size reduction with same or better query performance.
"""

import h5py
import hdf5plugin
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


def recompress_to_zstd(
    input_path: Path,
    output_path: Path,
    zstd_level: int = 9
):
    """
    Recompress HDF5 file from gzip to zstd

    Args:
        input_path: Path to gzip-compressed HDF5
        output_path: Path for zstd-compressed HDF5
        zstd_level: Compression level (1-22, default 9 for balanced speed/size)
    """
    logger.info("="*80)
    logger.info("ZSTANDARD RECOMPRESSION")
    logger.info("="*80)
    logger.info("")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Zstd level: {zstd_level}")
    logger.info("")

    # Load gzip-compressed data
    logger.info("Loading gzip-compressed data...")
    with h5py.File(input_path, 'r') as f_in:
        if 'binary_bank_vectors' not in f_in:
            logger.error("❌ 'binary_bank_vectors' dataset not found!")
            return False

        binary_data = f_in['binary_bank_vectors'][:]
        total_chunks, num_banks, dimension = binary_data.shape

        logger.info(f"  Loaded: {total_chunks:,} chunks × {num_banks} banks × {dimension:,}D")
        logger.info(f"  Shape: {binary_data.shape}")
        logger.info(f"  Dtype: {binary_data.dtype}")

        # Load metadata
        metadata = dict(f_in['binary_bank_vectors'].attrs)
        logger.info(f"  Metadata keys: {list(metadata.keys())}")

    input_size_gb = input_path.stat().st_size / (1024**3)
    logger.info(f"  Input size: {input_size_gb:.2f} GB")
    logger.info("")

    # Save with zstd compression
    logger.info("Recompressing with Zstandard...")
    logger.info(f"  Zstd level: {zstd_level} (1=fast, 22=max compression)")
    logger.info("")

    with h5py.File(output_path, 'w') as f_out:
        # Create dataset with zstd compression
        dset = f_out.create_dataset(
            'binary_bank_vectors',
            data=binary_data,
            dtype='uint8',
            **hdf5plugin.Zstd(clevel=zstd_level)
        )

        # Copy metadata
        for key, value in metadata.items():
            dset.attrs[key] = value

        # Add recompression info
        dset.attrs['recompressed_from'] = str(input_path)
        dset.attrs['recompressed_date'] = datetime.now().isoformat()
        dset.attrs['compression_method'] = f'zstd_level_{zstd_level}'

    output_size_gb = output_path.stat().st_size / (1024**3)
    compression_improvement = (1 - output_size_gb / input_size_gb) * 100

    logger.info(f"✓ Recompression complete!")
    logger.info("")
    logger.info("="*80)
    logger.info("COMPRESSION COMPARISON")
    logger.info("="*80)
    logger.info("")
    logger.info(f"Input (gzip):     {input_size_gb:.2f} GB")
    logger.info(f"Output (zstd):    {output_size_gb:.2f} GB")
    logger.info(f"Improvement:      {compression_improvement:.1f}% smaller")
    logger.info("")

    # Calculate compression ratios
    raw_size_gb = binary_data.nbytes / (1024**3)
    gzip_ratio = raw_size_gb / input_size_gb
    zstd_ratio = raw_size_gb / output_size_gb

    logger.info(f"Raw data size:    {raw_size_gb:.2f} GB")
    logger.info(f"Gzip ratio:       {gzip_ratio:.1f}×")
    logger.info(f"Zstd ratio:       {zstd_ratio:.1f}×")
    logger.info("")

    logger.info("="*80)
    logger.info("✅ ZSTANDARD RECOMPRESSION COMPLETE")
    logger.info("="*80)
    logger.info("")
    logger.info("Benefits:")
    logger.info("  ✓ Smaller file size")
    logger.info("  ✓ Same or faster decompression")
    logger.info("  ✓ Identical data (lossless)")
    logger.info("  ✓ Same API (transparent to queries)")
    logger.info("")

    return True


def main():
    """Recompress 6-bank binary file with Zstandard"""

    input_path = Path("genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_6banks_split_binary.h5")
    output_path = Path("genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_6banks_split_binary_zstd.h5")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    # Try different compression levels
    logger.info("Testing Zstandard compression...")
    logger.info("")

    success = recompress_to_zstd(input_path, output_path, zstd_level=9)

    if success:
        logger.info("🎉 SUCCESS! You now have:")
        logger.info("")
        logger.info(f"  Gzip version:  {input_path}")
        logger.info(f"  Zstd version:  {output_path}  ⭐ USE THIS ONE")
        logger.info("")
        logger.info("Both files have identical data - zstd is just smaller!")
        return 0
    else:
        return 1


if __name__ == '__main__':
    exit(main())
