#!/usr/bin/env python3
"""
Re-encode genome with ALL 5 biophysical lenses using the SAME data sources as existing HDF5.

Uses ComplementaryPairEncoder to get nucleotides (GDiff + guide FASTAs),
then encodes with all 5 lenses.
"""

import h5py
import logging
import time
from pathlib import Path
import numpy as np

from genomevault.hypervector_transform.complementary_pair_encoder import ComplementaryPairEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

# Lens definitions
LENS_DEFINITIONS = {
    'AT': {'positive': {'A'}, 'negative': {'T'}},
    'GC': {'positive': {'G'}, 'negative': {'C'}},
    'PuPy': {'positive': {'A', 'G'}, 'negative': {'T', 'C'}},  # Purine vs Pyrimidine
    'AmKe': {'positive': {'A', 'C'}, 'negative': {'G', 'T'}},  # Amino vs Keto
    'StWk': {'positive': {'G', 'C'}, 'negative': {'A', 'T'}},  # Strong vs Weak
}


def encode_chunk_all_lenses(encoder: ComplementaryPairEncoder, chrom: str, chunk_start: int):
    """
    Encode chunk with ALL 5 lenses.

    Uses the SAME nucleotide fetching logic as existing HDF5 encoder.
    """
    # Initialize vectors for all 5 lenses
    lens_vectors = {
        lens_name: np.zeros(encoder.D, dtype=np.float32)
        for lens_name in LENS_DEFINITIONS.keys()
    }

    # Get guide for this region
    guide_id = encoder._get_guide_for_region(chrom, chunk_start)
    if not guide_id:
        guide_id = 'ref1'

    # Process each position
    for offset in range(encoder.N):
        pos = chunk_start + offset
        nucleotide = encoder._get_nucleotide_at_position(chrom, pos, guide_id)

        # Get position vector
        pos_vec = encoder.position_codebook[offset].astype(np.float32)

        # Encode with ALL lenses
        for lens_name, lens_def in LENS_DEFINITIONS.items():
            if nucleotide in lens_def['positive']:
                lens_vectors[lens_name] += pos_vec
            elif nucleotide in lens_def['negative']:
                lens_vectors[lens_name] -= pos_vec
            # 'N' contributes to neither (ternary: 0)

    return lens_vectors


def main():
    logger.info("=" * 80)
    logger.info("MULTI-LENS GENOME ENCODING (All 5 Lenses)")
    logger.info("=" * 80)
    logger.info("")

    # Configuration
    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    guide_fasta_dir = Path("/Volumes/1TBStorage/guide_strands")
    output_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome_5lenses.h5")

    dimension = 10000
    chunk_size = 2000

    logger.info("Configuration:")
    logger.info(f"  GDiff: {gdiff_path}")
    logger.info(f"  Guide FASTAs: {guide_fasta_dir}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Dimension: {dimension:,}D")
    logger.info(f"  Chunk size: {chunk_size:,} bp")
    logger.info("")

    # Load existing HDF5 to get chunk keys
    existing_h5 = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome.h5")
    logger.info(f"Loading chunk keys from existing HDF5: {existing_h5}")

    with h5py.File(existing_h5, 'r') as f:
        chunk_keys = [key.decode('utf-8') for key in f['chunk_keys'][:]]
        total_chunks = len(chunk_keys)

    logger.info(f"  Found {total_chunks:,} chunks")
    logger.info("")

    # Initialize encoder (uses SAME data sources as existing HDF5)
    logger.info("Initializing encoder...")
    encoder = ComplementaryPairEncoder(
        gdiff_path=gdiff_path,
        guide_fasta_dir=guide_fasta_dir,
        dimension=dimension,
        chunk_size=chunk_size
    )
    logger.info("  ✓ Encoder initialized")
    logger.info("")

    # Create output HDF5 with all 5 lenses
    logger.info("Creating output HDF5...")
    with h5py.File(output_path, 'w') as f_out:
        # Create datasets for all 5 lenses
        for lens_name in LENS_DEFINITIONS.keys():
            f_out.create_dataset(
                f'{lens_name}_vectors',
                shape=(total_chunks, dimension),
                dtype=np.float32,
                chunks=(1, dimension),
                compression='gzip',
                compression_opts=1
            )
            logger.info(f"  Created {lens_name}_vectors dataset")

        # Copy chunk keys
        f_out.create_dataset(
            'chunk_keys',
            data=[key.encode('utf-8') for key in chunk_keys],
            dtype=h5py.string_dtype('utf-8')
        )
        logger.info(f"  Created chunk_keys dataset")

    logger.info("  ✓ HDF5 structure created")
    logger.info("")

    # Encode all chunks
    logger.info("=" * 80)
    logger.info("ENCODING ALL CHUNKS")
    logger.info("=" * 80)
    logger.info("")

    start_time = time.time()

    with h5py.File(output_path, 'r+') as f_out:
        for chunk_idx, chunk_key in enumerate(chunk_keys):
            # Parse chunk key
            parts = chunk_key.split(':')
            chrom = parts[0]
            chunk_start = int(parts[1])

            # Encode with all 5 lenses
            lens_vectors = encode_chunk_all_lenses(encoder, chrom, chunk_start)

            # Save to HDF5
            for lens_name, vector in lens_vectors.items():
                f_out[f'{lens_name}_vectors'][chunk_idx, :] = vector

            # Progress
            if (chunk_idx + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (chunk_idx + 1) / elapsed
                remaining = (total_chunks - chunk_idx - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  Progress: {chunk_idx+1:,}/{total_chunks:,} ({(chunk_idx+1)/total_chunks*100:.1f}%) | "
                    f"Rate: {rate:.1f} chunks/sec | ETA: {remaining/60:.1f} min"
                )

    total_time = time.time() - start_time

    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ ENCODING COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Output: {output_path}")
    logger.info(f"Total chunks: {total_chunks:,}")
    logger.info(f"Lenses encoded: {list(LENS_DEFINITIONS.keys())}")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
    logger.info(f"File size: {output_path.stat().st_size / (1024**3):.2f} GB")
    logger.info("")


if __name__ == '__main__':
    main()
