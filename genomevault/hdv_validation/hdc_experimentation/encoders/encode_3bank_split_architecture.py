#!/usr/bin/env python3
"""
3-Bank Split Architecture Encoder with Overlap - OPTIMIZED
===========================================================

Encodes genome using biophysically-motivated split banks:
- Bank 1: Hydrophobic (T vs A, transparent to G/C)
- Bank 2: Major Groove (G vs C, transparent to A/T)
- Bank 3: Hinge (Y-R vs R-Y structural flexibility)

Optimized Parameters (2025-11-21):
- N = 1,024 bp (chunk size, reduced for genome structure exploitation)
- D = 5,120 bits (dimension, 2× faster queries, exploits repetitive elements)
- Overlap = 128 bp (12.5%, higher % for smaller chunks)
- Stride = 896 bp (N - OVERLAP)
- SNR = D/N = 5.0 (proven effective ratio)
- Sparsity: Keep 100% of accumulated data (no aggressive thresholding)

Rationale: Lower D forces system to find genome's natural patterns (45% repeats,
conserved motifs). Same storage as D=10K,N=2K but 2× faster queries.

Memory optimization: <50 GB RAM limit
CPU optimization: 10-core parallel processing with proper worker initialization
"""

import h5py
import numpy as np
import logging
import time
from pathlib import Path
from datetime import datetime
import json
import sys
from multiprocessing import Pool, cpu_count
import gzip

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from genomevault.hypervector_transform.complementary_pair_encoder import ComplementaryPairEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s'
)
logger = logging.getLogger(__name__)

# === ARCHITECTURE PARAMETERS (OPTIMIZED 2025-11-21) ===
N = 1_024   # Chunk size (bp) - reduced for genome structure exploitation
D = 5_120   # Dimension (bits) - 2× faster queries, exploits repetitive elements
OVERLAP = 128  # 12.5% overlap (higher % for smaller chunks to handle edge effects)
STRIDE = 896   # N - OVERLAP
SPARSITY_PERCENTILE = 50  # Keep 100% of accumulated data (50th percentile = median = no thresholding)

# Biophysical bank definitions
PURINES = {'A', 'G'}
PYRIMIDINES = {'C', 'T'}

BANK_DEFINITIONS = {
    'Hydrophobic': {
        'positive': {'T'},  # Methylated
        'negative': {'A'},  # Non-hydrophobic
        'transparent': {'G', 'C', 'N'}
    },
    'MajorGroove': {
        'positive': {'G'},  # Acceptor-heavy
        'negative': {'C'},  # Donor-heavy
        'transparent': {'A', 'T', 'N'}
    },
    'Hinge': {
        'type': 'contextual'  # Special handling for Y-R steps
    }
}

# === GLOBAL ENCODER FOR WORKER PROCESSES ===
_worker_encoder = None


def init_worker(gdiff_path, guide_fasta_dir):
    """
    Initialize worker process with a single ComplementaryPairEncoder instance.

    This function is called ONCE per worker process at startup.
    The encoder is stored in a global variable accessible to all tasks in that worker.
    """
    global _worker_encoder
    _worker_encoder = ComplementaryPairEncoder(
        gdiff_path=gdiff_path,
        guide_fasta_dir=guide_fasta_dir,
        dimension=D,
        chunk_size=N
    )
    logger.info(f"Worker initialized (PID {__import__('os').getpid()})")


def sparsify_bipolar(vector: np.ndarray, percentile: float = 92) -> np.ndarray:
    """
    Keep top percentile of positive values and top percentile of negative values.
    Zero out the middle (100 - 2*percentile)% of values.

    With percentile=92:
    - Top 8% positive → +1
    - Top 8% negative → -1
    - Middle 84% → 0

    Args:
        vector: int16 accumulated vector
        percentile: Percentile threshold (92 = keep top 8%)

    Returns:
        Ternary vector {-1, 0, +1} as int8
    """
    result = np.zeros_like(vector, dtype=np.int8)

    # Threshold for positive values
    positive_vals = vector[vector > 0]
    if len(positive_vals) > 0:
        pos_thresh = np.percentile(positive_vals, percentile)
        result[vector > pos_thresh] = 1

    # Threshold for negative values
    negative_vals = vector[vector < 0]
    if len(negative_vals) > 0:
        neg_thresh = np.percentile(negative_vals, 100 - percentile)
        result[vector < neg_thresh] = -1

    return result


def generate_chunk_keys(chrom_sizes: dict) -> list:
    """
    Generate all chunk keys with overlap.

    Args:
        chrom_sizes: Dict mapping chromosome names to sizes

    Returns:
        List of (chrom, start, end) tuples
    """
    chunks = []

    for chrom, size in chrom_sizes.items():
        pos = 0
        while pos < size:
            chunk_end = min(pos + N, size)
            chunks.append((chrom, pos, chunk_end))

            if chunk_end >= size:
                break

            # Move by stride (with overlap)
            pos += STRIDE

    return chunks


def encode_chunk_3banks(encoder: ComplementaryPairEncoder,
                        chrom: str,
                        chunk_start: int,
                        chunk_end: int,
                        guide_id: str) -> tuple:
    """
    Encode single chunk using 3-bank split architecture.

    Uses int16 accumulation, then sparsifies to int8 ternary.

    Args:
        encoder: ComplementaryPairEncoder instance
        chrom: Chromosome name
        chunk_start: Start position (0-based)
        chunk_end: End position (exclusive)
        guide_id: Guide reference ID

    Returns:
        Tuple of (bank1_vec, bank2_vec, bank3_vec) as int8 arrays
    """
    # Initialize int16 accumulators (prevent overflow)
    acc_hydro = np.zeros(D, dtype=np.int16)
    acc_groove = np.zeros(D, dtype=np.int16)
    acc_hinge = np.zeros(D, dtype=np.int16)

    # Process each position in chunk
    chunk_length = chunk_end - chunk_start

    for offset in range(chunk_length):
        pos = chunk_start + offset

        # Get nucleotide (with GDiff variant integration)
        nucleotide = encoder._get_nucleotide_at_position(chrom, pos, guide_id)

        # Get position vector (sparse random)
        pos_vec = encoder.position_codebook[offset % N].astype(np.int16)

        # === Bank 1: Hydrophobic (AT-exclusive) ===
        if nucleotide == 'T':
            acc_hydro += pos_vec  # Methylated (+1)
        elif nucleotide == 'A':
            acc_hydro -= pos_vec  # Hydrophilic (-1)
        # G, C, N → contribute 0 (transparent)

        # === Bank 2: Major Groove (GC-exclusive) ===
        if nucleotide == 'G':
            acc_groove += pos_vec  # Acceptor-heavy (+1)
        elif nucleotide == 'C':
            acc_groove -= pos_vec  # Donor-heavy (-1)
        # A, T, N → contribute 0 (transparent)

        # === Bank 3: Hinge (Contextual Y-R steps) ===
        # Need to look ahead 1 base for dinucleotide context
        if offset < chunk_length - 1:
            next_pos = pos + 1
            next_nucleotide = encoder._get_nucleotide_at_position(chrom, next_pos, guide_id)

            is_YR = (nucleotide in PYRIMIDINES) and (next_nucleotide in PURINES)
            is_RY = (nucleotide in PURINES) and (next_nucleotide in PYRIMIDINES)

            if is_YR:
                acc_hinge += pos_vec  # Flexible hinge (+1)
            elif is_RY:
                acc_hinge -= pos_vec  # Stiff lock (-1)
            # R-R and Y-Y steps → 0 (neutral stacking)

    # Direct ternary quantization - sparsity comes naturally from D/N ratio and bank transparency
    # NO percentile thresholding - we keep ALL accumulated information!
    bank1 = np.sign(acc_hydro).astype(np.int8)  # T=+1, A=-1, GC=0 (natural 50% sparsity)
    bank2 = np.sign(acc_groove).astype(np.int8)  # G=+1, C=-1, AT=0 (natural 50% sparsity)
    bank3 = np.sign(acc_hinge).astype(np.int8)   # YR=+1, RY=-1, neutral=0 (natural ~70% sparsity)

    return (bank1, bank2, bank3)


def encode_chunk_worker(args):
    """
    Worker function that uses the pre-initialized global encoder.

    Args:
        args: Tuple of (chrom, start, end, guide_id, chunk_idx)

    Returns:
        Tuple of (chunk_idx, (bank1, bank2, bank3))
    """
    global _worker_encoder
    chrom, start, end, guide_id, chunk_idx = args

    # Use the global encoder (initialized ONCE per worker)
    banks = encode_chunk_3banks(_worker_encoder, chrom, start, end, guide_id)

    return (chunk_idx, banks)


def main():
    """Main encoding pipeline."""
    logger.info("=" * 80)
    logger.info("3-BANK SPLIT ARCHITECTURE ENCODER (FIXED MULTIPROCESSING)")
    logger.info("=" * 80)
    logger.info("")

    # === CONFIGURATION ===
    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    guide_fasta_dir = Path("/Volumes/1TBStorage/guide_strands")
    output_dir = Path("genomevault/hdv_validation/hdc_experimentation/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "encoded_genome_3banks.h5"

    logger.info(f"GDiff: {gdiff_path}")
    logger.info(f"Guide FASTAs: {guide_fasta_dir}")
    logger.info(f"Output: {output_path}")
    logger.info("")

    # === PARAMETERS ===
    logger.info("Architecture Parameters:")
    logger.info(f"  Chunk size (N): {N:,} bp")
    logger.info(f"  Dimension (D): {D:,} bits")
    logger.info(f"  Overlap: {OVERLAP:,} bp (10%)")
    logger.info(f"  Stride: {STRIDE:,} bp")
    logger.info(f"  Sparsity target: {100 - 2*(100-SPARSITY_PERCENTILE):.0f}% "
                f"({100-SPARSITY_PERCENTILE}% pos + {100-SPARSITY_PERCENTILE}% neg)")
    logger.info("")

    # === INITIALIZE ENCODER (for main process only, to get metadata) ===
    logger.info("Initializing encoder (main process)...")
    start_time = time.time()

    encoder = ComplementaryPairEncoder(
        gdiff_path=gdiff_path,
        guide_fasta_dir=guide_fasta_dir,
        dimension=D,
        chunk_size=N
    )

    # Get chromosome sizes from GDiff region_guide_map
    chrom_sizes = {}
    logger.info("Reading chromosome sizes from GDiff...")

    with gzip.open(gdiff_path, 'rt') as f:
        gdiff_data = json.load(f)
        region_map = gdiff_data['region_guide_map']

        for region_key in region_map.keys():
            # Parse: 'chr1_consensus:0-10000000'
            chrom_part, range_part = region_key.split(':')
            chrom = chrom_part.replace('_consensus', '')
            start, end = map(int, range_part.split('-'))

            if chrom not in chrom_sizes:
                chrom_sizes[chrom] = 0
            chrom_sizes[chrom] = max(chrom_sizes[chrom], end)

    logger.info(f"  Found {len(chrom_sizes)} chromosomes")
    logger.info("")

    # === GENERATE CHUNK KEYS ===
    logger.info("Generating chunk keys with overlap...")
    chunk_keys = generate_chunk_keys(chrom_sizes)
    total_chunks = len(chunk_keys)

    logger.info(f"  Total chunks: {total_chunks:,}")
    logger.info(f"  Total genome coverage: {sum(chrom_sizes.values()):,} bp")
    logger.info(f"  Effective coverage with overlaps: ~{total_chunks * N:,} bp")
    logger.info("")

    # === ESTIMATE MEMORY USAGE ===
    bytes_per_chunk = 3 * D * 1  # 3 banks × D × int8
    estimated_ram = (total_chunks * bytes_per_chunk) / (1024**3)

    logger.info(f"Memory estimate:")
    logger.info(f"  Per chunk: {bytes_per_chunk / 1024:.1f} KB")
    logger.info(f"  Total file size: {estimated_ram:.1f} GB")
    logger.info("")

    # === CREATE OUTPUT HDF5 ===
    logger.info("Creating output HDF5...")
    with h5py.File(output_path, 'w') as f_out:
        # Create dataset for all 3 banks
        # Shape: (chunks, 3 banks, D)
        # Chunk shape: (1, 3, D) for optimal single-chunk reads
        dset = f_out.create_dataset(
            'all_bank_vectors',
            shape=(total_chunks, 3, D),
            dtype=np.int8,  # Ternary {-1, 0, +1}
            chunks=(1, 3, D),
            compression='gzip',
            compression_opts=6
        )

        # Create chunk_keys dataset
        chunk_key_strings = [f"{chrom}:{start}-{end}"
                             for chrom, start, end in chunk_keys]
        f_out.create_dataset(
            'chunk_keys',
            data=[s.encode('utf-8') for s in chunk_key_strings],
            dtype=h5py.string_dtype('utf-8')
        )

        # Store metadata
        metadata = {
            'genome_size': sum(chrom_sizes.values()),
            'chunk_size': N,
            'overlap': OVERLAP,
            'stride': STRIDE,
            'dimension': D,
            'sparsity_percentile': SPARSITY_PERCENTILE,
            'encoding_date': datetime.now().isoformat(),
            'gdiff_source': str(gdiff_path),
            'guide_fasta_dir': str(guide_fasta_dir),
            'num_chunks': total_chunks
        }

        for key, value in metadata.items():
            f_out.attrs[key] = value

    logger.info("  ✓ HDF5 structure created")
    logger.info("")

    # === PARALLEL ENCODING WITH PROPER WORKER INITIALIZATION ===
    logger.info("=" * 80)
    logger.info("ENCODING CHUNKS (Parallel with Worker Initialization)")
    logger.info("=" * 80)
    logger.info("")

    # Use 8 cores (leave 2 for system)
    num_processes = min(8, cpu_count() - 2)
    logger.info(f"Parallel processing: {num_processes} workers")
    logger.info("Each worker will initialize ComplementaryPairEncoder ONCE")
    logger.info("")

    # Process in batches to control memory
    BATCH_SIZE = 5000  # Process 5000 chunks at a time (~120 MB per batch with int8)
    num_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE

    logger.info(f"Processing {total_chunks:,} chunks in {num_batches} batches")
    logger.info(f"Batch size: {BATCH_SIZE:,} chunks")
    logger.info("")

    encoding_start = time.time()
    chunks_processed = 0

    # Create pool ONCE with initializer (not per batch!)
    logger.info("Initializing worker pool...")
    with Pool(processes=num_processes,
              initializer=init_worker,
              initargs=(gdiff_path, guide_fasta_dir)) as pool:

        logger.info("  ✓ Worker pool initialized")
        logger.info("")

        for batch_idx in range(num_batches):
            batch_start_idx = batch_idx * BATCH_SIZE
            batch_end_idx = min(batch_start_idx + BATCH_SIZE, total_chunks)
            batch_size = batch_end_idx - batch_start_idx

            batch_start_time = time.time()

            logger.info(f"Batch {batch_idx + 1}/{num_batches}: chunks {batch_start_idx:,} - {batch_end_idx-1:,}")

            # Prepare arguments for this batch
            # Determine guide for each chunk using GDiff's region-to-guide mapping
            batch_args = []
            for i in range(batch_start_idx, batch_end_idx):
                chrom, start, end = chunk_keys[i]
                # Use encoder to get correct guide for this region (handles 10M region swaps)
                guide_id = encoder._get_guide_for_region(chrom, start)
                if not guide_id:
                    guide_id = 'ref1'  # Fallback
                batch_args.append((chrom, start, end, guide_id, i))

            # Parallel encode using pre-initialized workers
            results = pool.map(encode_chunk_worker, batch_args)

            # Write results to HDF5
            with h5py.File(output_path, 'a') as f_out:
                dset = f_out['all_bank_vectors']

                for chunk_idx, (bank1, bank2, bank3) in results:
                    dset[chunk_idx, 0, :] = bank1
                    dset[chunk_idx, 1, :] = bank2
                    dset[chunk_idx, 2, :] = bank3

            chunks_processed += batch_size
            batch_time = time.time() - batch_start_time
            chunks_per_sec = batch_size / batch_time

            # Progress reporting
            total_elapsed = time.time() - encoding_start
            pct = 100.0 * chunks_processed / total_chunks
            chunks_remaining = total_chunks - chunks_processed
            eta_sec = chunks_remaining / chunks_per_sec if chunks_per_sec > 0 else 0

            logger.info(f"  ✓ Batch complete: {batch_size:,} chunks in {batch_time:.1f}s "
                        f"({chunks_per_sec:.1f} chunks/s)")
            logger.info(f"  Progress: {chunks_processed:,}/{total_chunks:,} ({pct:.1f}%) "
                        f"- ETA: {eta_sec/3600:.1f} hours")
            logger.info("")

    total_encoding_time = time.time() - encoding_start

    # === FINAL STATISTICS ===
    logger.info("=" * 80)
    logger.info("ENCODING COMPLETE")
    logger.info("=" * 80)
    logger.info("")

    file_size_gb = output_path.stat().st_size / (1024**3)

    logger.info(f"Total time: {total_encoding_time:.1f}s ({total_encoding_time/3600:.2f} hours)")
    logger.info(f"Throughput: {total_chunks / total_encoding_time:.1f} chunks/second")
    logger.info(f"Output file: {output_path}")
    logger.info(f"File size: {file_size_gb:.2f} GB")
    logger.info("")

    # Verify and analyze sparsity
    logger.info("Analyzing sparsity...")
    with h5py.File(output_path, 'r') as f:
        dset = f['all_bank_vectors']

        # Sample 1000 random chunks
        sample_indices = np.random.choice(total_chunks, size=min(1000, total_chunks), replace=False)
        sample_indices = np.sort(sample_indices)  # HDF5 requires sorted indices for fancy indexing
        sample_data = dset[sample_indices, :, :]

        # Analyze each bank
        for bank_idx in range(3):
            bank_name = ['Hydrophobic', 'MajorGroove', 'Hinge'][bank_idx]
            bank_data = sample_data[:, bank_idx, :]

            pos_pct = 100.0 * (bank_data == 1).sum() / bank_data.size
            neg_pct = 100.0 * (bank_data == -1).sum() / bank_data.size
            zero_pct = 100.0 * (bank_data == 0).sum() / bank_data.size

            logger.info(f"  Bank {bank_idx + 1} ({bank_name}):")
            logger.info(f"    +1: {pos_pct:.2f}%  |  -1: {neg_pct:.2f}%  |  0: {zero_pct:.2f}%")

    logger.info("")
    logger.info("✓ All done!")


if __name__ == '__main__':
    main()
