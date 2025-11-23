#!/usr/bin/env python3
"""
HDV Query Performance Benchmark - Isolate Compute vs I/O

Tests the hypothesis that the ~293ms query time is dominated by HDF5 disk I/O
rather than pure compute time.

Expected results:
- Pure compute (dot product): ~10-15 microseconds
- Full query (with HDF5 load): ~293 milliseconds
- Bottleneck: Disk I/O overhead
"""

import logging
import time
from pathlib import Path

import numpy as np
import h5py

from genomevault.hypervector_transform.complementary_pair_encoder import ComplementaryPairEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("HDV QUERY PERFORMANCE BENCHMARK")
    logger.info("=" * 80)
    logger.info("")

    # Configuration
    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    guide_fasta_dir = Path("/Volumes/1TBStorage/guide_strands")
    hdf5_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome.h5")

    dimension = 10000
    chunk_size = 2000
    num_iterations = 10000

    logger.info("Configuration:")
    logger.info(f"  HDF5 file: {hdf5_path}")
    logger.info(f"  Dimension: {dimension:,}D")
    logger.info(f"  Chunk size: {chunk_size:,} bp")
    logger.info(f"  Test iterations: {num_iterations:,}")
    logger.info("")

    # Initialize encoder for position codebook
    logger.info("Initializing encoder (for position codebook)...")
    encoder = ComplementaryPairEncoder(
        gdiff_path=gdiff_path,
        guide_fasta_dir=guide_fasta_dir,
        dimension=dimension,
        chunk_size=chunk_size
    )
    logger.info("  ✓ Encoder ready")
    logger.info("")

    # Open HDF5 file
    logger.info("Opening HDF5 file...")
    h5file = h5py.File(hdf5_path, 'r')

    # Build chunk index
    chunk_index = {}
    chunk_keys = h5file['chunk_keys'][:]
    for idx, key in enumerate(chunk_keys):
        chunk_index[key.decode('utf-8')] = idx

    logger.info(f"  ✓ Loaded index with {len(chunk_index):,} chunks")
    logger.info("")

    # Select a random chunk to test with
    test_chunk_key = list(chunk_index.keys())[500000]  # Mid-genome chunk
    test_chunk_idx = chunk_index[test_chunk_key]
    logger.info(f"Test chunk: {test_chunk_key} (index {test_chunk_idx:,})")
    logger.info("")

    # =========================================================================
    # TEST 1: PURE COMPUTE (DOT PRODUCT ONLY)
    # =========================================================================

    logger.info("=" * 80)
    logger.info("TEST 1: PURE COMPUTE (DOT PRODUCT ONLY)")
    logger.info("=" * 80)
    logger.info("")

    # Pre-load chunk into memory
    logger.info("Pre-loading chunk into memory...")
    AT_vec = h5file['AT_vectors'][test_chunk_idx]
    GC_vec = h5file['GC_vectors'][test_chunk_idx]

    # Copy to ensure they're in RAM, not memory-mapped
    AT_vec = np.array(AT_vec, dtype=np.float32)
    GC_vec = np.array(GC_vec, dtype=np.float32)
    logger.info("  ✓ Chunk loaded into RAM")
    logger.info("")

    # Select a position in the chunk
    test_offset = 1000  # Middle of 2000bp chunk
    pos_vec = encoder.position_codebook[test_offset].astype(np.float32)

    logger.info(f"Running {num_iterations:,} iterations of pure compute...")

    # Warm up
    for _ in range(100):
        sim_AT = np.dot(pos_vec, AT_vec)
        sim_GC = np.dot(pos_vec, GC_vec)

    # Benchmark pure compute
    start = time.perf_counter()
    for _ in range(num_iterations):
        sim_AT = np.dot(pos_vec, AT_vec)
        sim_GC = np.dot(pos_vec, GC_vec)

        # Two-stage retrieval (same as validation)
        if abs(sim_AT) > abs(sim_GC):
            nucleotide = 'A' if sim_AT > 0 else 'T'
        else:
            nucleotide = 'G' if sim_GC > 0 else 'C'

    elapsed_compute = (time.perf_counter() - start) / num_iterations

    logger.info("")
    logger.info("PURE COMPUTE RESULTS:")
    logger.info(f"  Average time per query: {elapsed_compute * 1e6:.2f} microseconds")
    logger.info(f"  Expected: ~10-15 microseconds")
    logger.info(f"  Throughput: {1/elapsed_compute:,.0f} queries/second")
    logger.info("")

    # =========================================================================
    # TEST 2: FULL QUERY (WITH HDF5 LOADING)
    # =========================================================================

    logger.info("=" * 80)
    logger.info("TEST 2: FULL QUERY (WITH HDF5 LOADING)")
    logger.info("=" * 80)
    logger.info("")

    logger.info(f"Running {num_iterations:,} iterations of full query (with HDF5 load)...")

    # Benchmark full query with disk I/O
    start = time.perf_counter()
    for _ in range(num_iterations):
        # Load chunk from HDF5 (simulates disk I/O)
        AT_vec_loaded = h5file['AT_vectors'][test_chunk_idx]
        GC_vec_loaded = h5file['GC_vectors'][test_chunk_idx]

        # Compute
        sim_AT = np.dot(pos_vec, AT_vec_loaded)
        sim_GC = np.dot(pos_vec, GC_vec_loaded)

        # Two-stage retrieval
        if abs(sim_AT) > abs(sim_GC):
            nucleotide = 'A' if sim_AT > 0 else 'T'
        else:
            nucleotide = 'G' if sim_GC > 0 else 'C'

    elapsed_full = (time.perf_counter() - start) / num_iterations

    logger.info("")
    logger.info("FULL QUERY RESULTS:")
    logger.info(f"  Average time per query: {elapsed_full * 1e3:.2f} milliseconds")
    logger.info(f"  Expected: ~293 milliseconds")
    logger.info(f"  Throughput: {1/elapsed_full:.2f} queries/second")
    logger.info("")

    # =========================================================================
    # TEST 3: HDF5 CACHING EFFECT
    # =========================================================================

    logger.info("=" * 80)
    logger.info("TEST 3: HDF5 CACHING EFFECT (REPEATED ACCESS)")
    logger.info("=" * 80)
    logger.info("")

    logger.info(f"Testing if HDF5 caches chunks on repeated access...")
    logger.info("")

    # First access (cold)
    start = time.perf_counter()
    AT_vec_cold = h5file['AT_vectors'][test_chunk_idx]
    GC_vec_cold = h5file['GC_vectors'][test_chunk_idx]
    sim_AT = np.dot(pos_vec, AT_vec_cold)
    elapsed_cold = time.perf_counter() - start

    # Second access (potentially cached)
    start = time.perf_counter()
    AT_vec_warm = h5file['AT_vectors'][test_chunk_idx]
    GC_vec_warm = h5file['GC_vectors'][test_chunk_idx]
    sim_AT = np.dot(pos_vec, AT_vec_warm)
    elapsed_warm = time.perf_counter() - start

    logger.info("CACHING RESULTS:")
    logger.info(f"  First access (cold): {elapsed_cold * 1e3:.2f} ms")
    logger.info(f"  Second access (warm): {elapsed_warm * 1e3:.2f} ms")
    logger.info(f"  Speedup: {elapsed_cold / elapsed_warm:.2f}×")
    logger.info("")

    # =========================================================================
    # ANALYSIS & SUMMARY
    # =========================================================================

    logger.info("=" * 80)
    logger.info("PERFORMANCE ANALYSIS")
    logger.info("=" * 80)
    logger.info("")

    io_overhead = elapsed_full - elapsed_compute
    io_overhead_percent = (io_overhead / elapsed_full) * 100

    logger.info("BOTTLENECK ANALYSIS:")
    logger.info(f"  Pure compute time: {elapsed_compute * 1e6:.2f} μs")
    logger.info(f"  Full query time: {elapsed_full * 1e3:.2f} ms")
    logger.info(f"  I/O overhead: {io_overhead * 1e3:.2f} ms ({io_overhead_percent:.1f}% of total)")
    logger.info(f"  Slowdown factor: {elapsed_full / elapsed_compute:,.0f}×")
    logger.info("")

    logger.info("THEORETICAL VS ACTUAL:")
    logger.info(f"  Expected pure compute: ~10-15 μs")
    logger.info(f"  Actual pure compute: {elapsed_compute * 1e6:.2f} μs")
    logger.info(f"  Expected full query: ~293 ms (from validation)")
    logger.info(f"  Actual full query: {elapsed_full * 1e3:.2f} ms")
    logger.info("")

    logger.info("CONCLUSIONS:")
    if elapsed_compute * 1e6 < 20:
        logger.info("  ✅ Pure compute is in expected microsecond range")
    else:
        logger.info(f"  ⚠️ Pure compute ({elapsed_compute * 1e6:.2f} μs) slower than expected")

    if io_overhead_percent > 90:
        logger.info("  ✅ Hypothesis CONFIRMED: I/O overhead is the bottleneck")
        logger.info(f"      - {io_overhead_percent:.1f}% of query time is disk I/O")
    else:
        logger.info(f"  ⚠️ Hypothesis PARTIAL: I/O is {io_overhead_percent:.1f}% of total time")

    logger.info("")
    logger.info("OPTIMIZATION OPPORTUNITIES:")
    logger.info("  1. LRU cache for frequently accessed chunks")
    logger.info("  2. Memory-mapped HDF5 with OS page cache")
    logger.info("  3. Pre-load hot chunks into RAM")
    logger.info("  4. SSD vs HDD storage (if not already on SSD)")
    logger.info("  5. Reduce HDF5 compression level (trade space for speed)")
    logger.info("")

    # Cleanup
    h5file.close()

    logger.info("=" * 80)
    logger.info("✅ BENCHMARK COMPLETE")
    logger.info("=" * 80)
    logger.info("")


if __name__ == "__main__":
    main()
