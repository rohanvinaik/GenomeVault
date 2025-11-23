#!/usr/bin/env python3
"""
Generate int8 quantized version of 3D HDF5 file.

Memory-efficient: Processes in batches to avoid loading entire dataset.
"""

import h5py
import numpy as np
from pathlib import Path
import logging
from time import perf_counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def quantize_float32_to_int8(data: np.ndarray) -> np.ndarray:
    """
    Quantize float32 data in [-1, 1] to int8 in [-127, 127].

    Args:
        data: Float32 array with values in [-1, 1]

    Returns:
        Int8 array with values in [-127, 127]
    """
    # Clip to [-1, 1] range and scale to int8 range
    clipped = np.clip(data, -1.0, 1.0)
    quantized = np.round(clipped * 127.0).astype(np.int8)
    return quantized


def generate_int8_h5(input_path: Path, output_path: Path, batch_size: int = 10000):
    """
    Generate int8 quantized HDF5 file from float32 version.

    Args:
        input_path: Path to input float32 3D HDF5
        output_path: Path to output int8 3D HDF5
        batch_size: Number of chunks to process at once (default: 10000)
    """
    logger.info("=" * 80)
    logger.info("INT8 QUANTIZATION - 3D HDF5")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Batch size: {batch_size:,} chunks")
    logger.info("")

    start_time = perf_counter()

    # Open input file
    logger.info("Opening input file...")
    with h5py.File(input_path, 'r') as f_in:
        # Get dataset info
        if 'all_lens_vectors' not in f_in:
            raise ValueError("Input file must contain 'all_lens_vectors' dataset")

        chunks_in = f_in['all_lens_vectors']
        total_chunks, n_lenses, n_dims = chunks_in.shape

        logger.info(f"  Total chunks: {total_chunks:,}")
        logger.info(f"  Shape: ({total_chunks:,}, {n_lenses}, {n_dims})")
        logger.info(f"  Input dtype: {chunks_in.dtype}")
        logger.info(f"  Input size: {chunks_in.nbytes / 1e9:.2f} GB (uncompressed)")
        logger.info("")

        # Calculate expected output size
        expected_size_gb = (total_chunks * n_lenses * n_dims * 1) / 1e9  # 1 byte per value
        logger.info(f"Expected output size: {expected_size_gb:.2f} GB (uncompressed)")
        logger.info(f"  Note: No compression used - gzip overhead not worth it for quantized data")
        logger.info("")

        # Create output file
        logger.info("Creating output file (UNCOMPRESSED for speed)...")
        with h5py.File(output_path, 'w') as f_out:
            # Create dataset with same shape but int8 dtype
            # NO COMPRESSION - gzip overhead not worth it for quantized data
            chunks_out = f_out.create_dataset(
                'all_lens_vectors',
                shape=(total_chunks, n_lenses, n_dims),
                dtype=np.int8,
                chunks=(1, n_lenses, n_dims)  # Same chunking as input
            )

            # Copy attributes
            if 'chunk_keys' in f_in:
                logger.info("  Copying chunk keys...")
                f_out.create_dataset('chunk_keys', data=f_in['chunk_keys'][:])

            # Copy metadata attributes
            for key, value in chunks_in.attrs.items():
                chunks_out.attrs[key] = value

            # Add quantization metadata
            chunks_out.attrs['quantization'] = 'int8'
            chunks_out.attrs['scale_factor'] = 127.0
            chunks_out.attrs['original_dtype'] = 'float32'

            logger.info("  ✓ Dataset created")
            logger.info("")

            # Process in batches
            logger.info("Quantizing and writing data...")
            num_batches = (total_chunks + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_chunks)
                batch_size_actual = end_idx - start_idx

                # Read batch
                batch_data = chunks_in[start_idx:end_idx]

                # Quantize
                quantized_batch = quantize_float32_to_int8(batch_data)

                # Write batch
                chunks_out[start_idx:end_idx] = quantized_batch

                # Progress
                progress_pct = ((batch_idx + 1) / num_batches) * 100
                if (batch_idx + 1) % max(1, num_batches // 20) == 0 or batch_idx == num_batches - 1:
                    logger.info(f"  Progress: {progress_pct:5.1f}% ({end_idx:,}/{total_chunks:,} chunks)")

            logger.info("")
            logger.info("  ✓ Quantization complete")

    end_time = perf_counter()
    elapsed = end_time - start_time

    # Report final file size
    output_size_gb = output_path.stat().st_size / 1e9

    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Size: {output_size_gb:.2f} GB (uncompressed)")
    logger.info(f"Compression vs float32: {281/output_size_gb:.1f}× smaller")
    logger.info(f"Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Throughput: {total_chunks/elapsed:,.0f} chunks/second")
    logger.info("")
    logger.info("✓ Int8 quantization complete!")
    logger.info("")


if __name__ == '__main__':
    input_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome_5lenses_3d.h5")
    output_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome_5lenses_3d_int8.h5")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        exit(1)

    if output_path.exists():
        logger.warning(f"Output file already exists: {output_path}")
        logger.warning("Deleting existing file...")
        output_path.unlink()

    generate_int8_h5(input_path, output_path, batch_size=10000)
