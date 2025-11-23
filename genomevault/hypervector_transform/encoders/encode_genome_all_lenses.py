#!/usr/bin/env python3
"""
Re-encode the experimental genome with ALL 5 biophysical lenses.

Current HDF5 only has AT and GC lenses. This script adds:
- PuPy (Purine vs Pyrimidine)
- AmKe (Amino vs Keto)
- StWk (Strong vs Weak)

Output: encoded_genome_5lenses.h5 with all lenses for multi-lens error correction.
"""

import h5py
import numpy as np
import gzip
import json
from pathlib import Path
from typing import Dict
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# LENS DEFINITIONS
# ============================================================================

LENS_DEFINITIONS = {
    'AT': {
        'positive': {'A'},
        'negative': {'T'}
    },
    'GC': {
        'positive': {'G'},
        'negative': {'C'}
    },
    'PuPy': {  # Purine vs Pyrimidine
        'positive': {'A', 'G'},  # Purines
        'negative': {'T', 'C'}   # Pyrimidines
    },
    'AmKe': {  # Amino vs Keto
        'positive': {'A', 'C'},  # Amino
        'negative': {'G', 'T'}   # Keto
    },
    'StWk': {  # Strong vs Weak (H-bonds)
        'positive': {'G', 'C'},  # Strong (3 H-bonds)
        'negative': {'A', 'T'}   # Weak (2 H-bonds)
    }
}

# ============================================================================
# MULTI-LENS ENCODER
# ============================================================================

class MultiLensHDCEncoder:
    """Encode genomic sequences using all 5 biophysical lenses."""

    def __init__(self, D: int = 10000, N: int = 2000, seed: int = 42):
        self.D = D  # Dimensionality
        self.N = N  # Chunk size (nucleotides)
        self.seed = seed

        np.random.seed(seed)

        # Generate position codebook (same for all lenses)
        logger.info(f"Generating position codebook (D={D}, N={N})...")
        self.position_codebook = np.random.randn(N, D).astype(np.float32)

        logger.info(f"✓ Multi-lens encoder initialized")
        logger.info(f"  Lenses: {list(LENS_DEFINITIONS.keys())}")

    def encode_chunk(self, sequence: str) -> Dict[str, np.ndarray]:
        """
        Encode a chunk through all 5 lenses.

        Returns:
            Dict mapping lens_name -> vector (D-dimensional)
        """
        assert len(sequence) == self.N, f"Expected {self.N} bp, got {len(sequence)}"

        lens_vectors = {}

        for lens_name, lens_def in LENS_DEFINITIONS.items():
            # Create bipolar encoding for this lens
            bipolar = np.zeros(self.N, dtype=np.float32)

            for i, nuc in enumerate(sequence):
                if nuc in lens_def['positive']:
                    bipolar[i] = 1.0
                elif nuc in lens_def['negative']:
                    bipolar[i] = -1.0
                else:  # N or other
                    bipolar[i] = 0.0

            # Bind with position codebook
            chunk_vector = bipolar @ self.position_codebook  # (N,) @ (N, D) = (D,)
            lens_vectors[lens_name] = chunk_vector

        return lens_vectors


# ============================================================================
# MAIN ENCODING PIPELINE
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("MULTI-LENS GENOME ENCODING (All 5 Lenses)")
    logger.info("=" * 80)
    logger.info("")

    # Paths
    h5_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome.h5")
    output_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome_5lenses.h5")

    logger.info(f"Loading existing HDF5: {h5_path}")
    logger.info(f"Output path: {output_path}")
    logger.info("")

    # Load chunk keys from existing HDF5
    with h5py.File(h5_path, 'r') as f:
        chunk_keys = [key.decode('utf-8') for key in f['chunk_keys'][:]]
        total_chunks = len(chunk_keys)

        # Get parameters
        D = f['AT_vectors'].shape[1]
        N = 2000  # Chunk size

    logger.info(f"Found {total_chunks:,} chunks")
    logger.info(f"Dimensionality: {D:,}")
    logger.info(f"Chunk size: {N:,} bp")
    logger.info("")

    # Initialize encoder
    encoder = MultiLensHDCEncoder(D=D, N=N, seed=42)

    # Load GDiff to get ground truth sequences
    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    logger.info(f"Loading GDiff: {gdiff_path}")

    with gzip.open(gdiff_path, 'rt') as f:
        gdiff = json.load(f)

    variants = gdiff['differential_variants']
    logger.info(f"  Loaded {len(variants):,} variants")
    logger.info("")

    # Load consensus reference
    import pysam
    consensus_path = Path("data/reference_genomes/consensus.fa.gz")
    logger.info(f"Loading consensus reference: {consensus_path}")
    consensus_fasta = pysam.FastaFile(str(consensus_path))
    logger.info("  ✓ Consensus loaded")
    logger.info("")

    # Index variants by chunk
    logger.info("Indexing variants by chunk...")
    variants_by_chunk = {}
    for v in variants:
        # Determine chunk
        chrom = v['chrom'].replace('_consensus', '')
        pos = v['pos']
        chunk_start = (pos // N) * N
        chunk_key = f"{chrom}_consensus:{chunk_start}"

        if chunk_key not in variants_by_chunk:
            variants_by_chunk[chunk_key] = []
        variants_by_chunk[chunk_key].append(v)

    logger.info(f"  ✓ Variants indexed into {len(variants_by_chunk):,} chunks")
    logger.info("")

    # Create output HDF5
    logger.info("Creating output HDF5 with all 5 lenses...")
    with h5py.File(output_path, 'w') as f_out:
        # Create datasets for all 5 lenses
        for lens_name in LENS_DEFINITIONS.keys():
            f_out.create_dataset(
                f'{lens_name}_vectors',
                shape=(total_chunks, D),
                dtype=np.float32,
                chunks=(1, D),
                compression='gzip',
                compression_opts=1
            )

        # Copy chunk keys
        f_out.create_dataset(
            'chunk_keys',
            data=[key.encode('utf-8') for key in chunk_keys],
            dtype=h5py.string_dtype('utf-8')
        )

        logger.info("  ✓ Datasets created")

    logger.info("")
    logger.info("Encoding genome with all 5 lenses...")
    logger.info("")

    # Encode all chunks
    with h5py.File(output_path, 'r+') as f_out:
        for chunk_idx, chunk_key in enumerate(tqdm(chunk_keys, desc="Encoding")):
            # Parse chunk key
            parts = chunk_key.split(':')
            chrom = parts[0]
            chunk_start = int(parts[1])

            # Get reference sequence
            ref_seq = consensus_fasta.fetch(chrom, chunk_start, chunk_start + N).upper()

            # Apply variants to get experimental sequence
            seq_list = list(ref_seq)
            if chunk_key in variants_by_chunk:
                for var in variants_by_chunk[chunk_key]:
                    local_pos = var['pos'] - chunk_start
                    if 0 <= local_pos < len(seq_list):
                        seq_list[local_pos] = var['alt']

            experimental_seq = ''.join(seq_list)

            # Pad if needed
            if len(experimental_seq) < N:
                experimental_seq += 'N' * (N - len(experimental_seq))

            # Encode with all lenses
            lens_vectors = encoder.encode_chunk(experimental_seq)

            # Save to HDF5
            for lens_name, vector in lens_vectors.items():
                f_out[f'{lens_name}_vectors'][chunk_idx, :] = vector

    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ ENCODING COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Output: {output_path}")
    logger.info(f"Total chunks: {total_chunks:,}")
    logger.info(f"Lenses encoded: {list(LENS_DEFINITIONS.keys())}")
    logger.info(f"File size: {output_path.stat().st_size / (1024**3):.2f} GB")
    logger.info("")


if __name__ == '__main__':
    main()
