#!/usr/bin/env python3
"""
Generate int4 quantized version of 3D HDF5 file.

Int4 quantization: -8 to +7, packed 2 values per byte.
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


def quantize_float32_to_int4_packed(data: np.ndarray) -> np.ndarray:
    """
    Quantize float32 data in [-1, 1] to int4 in [-8, 7], packed 2 per byte.

    Args:
        data: Float32 array with values in [-1, 1]
              Shape: (chunks, lenses, dims)

    Returns:
        Uint8 array with 2 int4 values packed per byte
        Shape: (chunks, lenses, dims//2) if dims is even
    """
    # Clip to [-1, 1] range and scale to int4 range [-8, 7]
    clipped = np.clip(data, -1.0, 1.0)
    # Scale to [-8, 7] range (15 quantization levels)
    scaled = np.round(clipped * 7.5).astype(np.int8)

    # Reshape to pack pairs
    chunks, lenses, dims = data.shape
    if dims % 2 != 0:
        raise ValueError(f"Dimension {dims} must be even for int4 packing")

    # Reshape to (chunks, lenses, dims//2, 2)
    pairs = scaled.reshape(chunks, lenses, dims // 2, 2)

    # Pack two int4 values into one uint8
    # First value in low nibble (bits 0-3), second in high nibble (bits 4-7)
    # Convert to uint8 for bit operations
    low_nibble = (pairs[:, :, :, 0] + 8).astype(np.uint8) & 0x0F  # Offset by 8 to make unsigned
    high_nibble = ((pairs[:, :, :, 1] + 8).astype(np.uint8) & 0x0F) << 4

    packed = low_nibble | high_nibble

    return packed


def unpack_int4_to_int8(packed: np.ndarray) -> np.ndarray:
    """
    Unpack int4 values from packed uint8 array back to int8.

    Args:
        packed: Uint8 array with 2 int4 values per byte
                Shape: (chunks, lenses, dims//2)

    Returns:
        Int8 array with unpacked values in [-8, 7]
        Shape: (chunks, lenses, dims)
    """
    # Extract low and high nibbles
    low_nibble = (packed & 0x0F).astype(np.int8) - 8
    high_nibble = ((packed >> 4) & 0x0F).astype(np.int8) - 8

    # Interleave to restore original order
    chunks, lenses, half_dims = packed.shape
    unpacked = np.empty((chunks, lenses, half_dims * 2), dtype=np.int8)
    unpacked[:, :, 0::2] = low_nibble
    unpacked[:, :, 1::2] = high_nibble

    return unpacked


def generate_int4_h5(input_path: Path, output_path: Path, batch_size: int = 10000):
    """
    Generate int4 quantized HDF5 file from float32 version.

    Args:
        input_path: Path to input float32 3D HDF5
        output_path: Path to output int4 3D HDF5 (packed)
        batch_size: Number of chunks to process at once (default: 10000)
    """
    logger.info("=" * 80)
    logger.info("INT4 QUANTIZATION - 3D HDF5 (BIT-PACKED)")
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

        if n_dims % 2 != 0:
            raise ValueError(f"Dimension {n_dims} must be even for int4 packing")

        # Calculate expected output size (0.5 bytes per value)
        expected_size_gb = (total_chunks * n_lenses * (n_dims // 2) * 1) / 1e9
        logger.info(f"Expected output size: {expected_size_gb:.2f} GB (uncompressed, packed)")
        logger.info(f"  Note: 2 int4 values packed per byte (0.5 bytes/value)")
        logger.info(f"  Note: No compression - uncompressed for speed")
        logger.info("")

        # Create output file
        logger.info("Creating output file (UNCOMPRESSED, BIT-PACKED)...")
        with h5py.File(output_path, 'w') as f_out:
            # Create dataset with packed shape (half the dimensions)
            # NO COMPRESSION - keep it simple and fast
            chunks_out = f_out.create_dataset(
                'all_lens_vectors_packed',
                shape=(total_chunks, n_lenses, n_dims // 2),
                dtype=np.uint8,
                chunks=(1, n_lenses, n_dims // 2)
            )

            # Copy attributes
            if 'chunk_keys' in f_in:
                logger.info("  Copying chunk keys...")
                f_out.create_dataset('chunk_keys', data=f_in['chunk_keys'][:])

            # Copy metadata attributes
            for key, value in chunks_in.attrs.items():
                chunks_out.attrs[key] = value

            # Add quantization metadata
            chunks_out.attrs['quantization'] = 'int4_packed'
            chunks_out.attrs['range'] = '[-8, 7]'
            chunks_out.attrs['packing'] = '2_values_per_byte'
            chunks_out.attrs['original_dtype'] = 'float32'
            chunks_out.attrs['original_dims'] = n_dims
            chunks_out.attrs['scale_factor'] = 7.5

            logger.info("  ✓ Dataset created")
            logger.info("")

            # Process in batches
            logger.info("Quantizing, packing, and writing data...")
            num_batches = (total_chunks + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_chunks)
                batch_size_actual = end_idx - start_idx

                # Read batch
                batch_data = chunks_in[start_idx:end_idx]

                # Quantize and pack
                packed_batch = quantize_float32_to_int4_packed(batch_data)

                # Write batch
                chunks_out[start_idx:end_idx] = packed_batch

                # Progress
                progress_pct = ((batch_idx + 1) / num_batches) * 100
                if (batch_idx + 1) % max(1, num_batches // 20) == 0 or batch_idx == num_batches - 1:
                    logger.info(f"  Progress: {progress_pct:5.1f}% ({end_idx:,}/{total_chunks:,} chunks)")

            logger.info("")
            logger.info("  ✓ Quantization and packing complete")

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
    logger.info(f"Size: {output_size_gb:.2f} GB (uncompressed, bit-packed)")
    logger.info(f"Compression vs float32: {281/output_size_gb:.1f}× smaller")
    logger.info(f"Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Throughput: {total_chunks/elapsed:,.0f} chunks/second")
    logger.info("")
    logger.info("✓ Int4 quantization and bit-packing complete!")
    logger.info("")


if __name__ == '__main__':
    input_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome_5lenses_3d.h5")
    output_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome_5lenses_3d_int4.h5")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        exit(1)

    if output_path.exists():
        logger.warning(f"Output file already exists: {output_path}")
        logger.warning("Deleting existing file...")
        output_path.unlink()

    generate_int4_h5(input_path, output_path, batch_size=10000)
