"""
Benchmark 2-bit unpacking overhead to verify <50ns per chunk target.

This measures the query-path overhead of unpacking ternary data from
2-bit representation. Critical for ensuring compression doesn't slow queries.

Target: <50ns per chunk unpacking (Phase 1 Week 2)

Author: Claude Code
Date: November 21, 2025
"""

import h5py
import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from quantization.ternary_2bit_packing import (
    pack_3bank_chunk,
    unpack_3bank_chunk,
)


def benchmark_unpacking_overhead(
    h5_path: str,
    num_chunks: int = 1000,
    num_iterations: int = 100,
    seed: int = 42,
) -> Dict:
    """
    Benchmark unpacking overhead on real genome chunks.

    Measures the time to unpack a 3-bank chunk from 2-bit representation,
    which is the overhead added to the query path.

    Args:
        h5_path: Path to encoded genome HDF5 file
        num_chunks: Number of random chunks to benchmark
        num_iterations: Number of unpacking iterations per chunk
        seed: Random seed for reproducibility

    Returns:
        Benchmark statistics dictionary
    """
    print("=" * 80)
    print("2-Bit Unpacking Overhead Benchmark")
    print("=" * 80)
    print()
    print(f"Target: <50ns per chunk unpacking")
    print(f"Testing: {num_chunks} chunks × {num_iterations} iterations = {num_chunks * num_iterations:,} operations")
    print()

    # Load encoded genome
    print(f"Loading encoded genome: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        all_banks = f['all_bank_vectors']
        total_chunks = all_banks.shape[0]

        print(f"  Total chunks: {total_chunks:,}")
        print()

        # Select random chunks
        np.random.seed(seed)
        chunk_indices = np.random.choice(total_chunks, size=num_chunks, replace=False)

        # Pre-pack all chunks (one-time cost, not part of query path)
        print("Pre-packing chunks (one-time setup cost)...")
        packed_chunks = []
        for i, chunk_idx in enumerate(chunk_indices):
            chunk = all_banks[chunk_idx, :, :]
            bank1 = chunk[0, :].astype(np.int8)
            bank2 = chunk[1, :].astype(np.int8)
            bank3 = chunk[2, :].astype(np.int8)

            packed1, packed2, packed3 = pack_3bank_chunk(bank1, bank2, bank3)
            packed_chunks.append((packed1, packed2, packed3))

            if (i + 1) % 200 == 0:
                print(f"  Packed {i + 1}/{num_chunks} chunks...")

        print(f"  ✓ Packed {num_chunks} chunks")
        print()

    # Benchmark unpacking (query-path overhead)
    print("Benchmarking unpacking overhead (query-path cost)...")
    print()

    times_per_chunk = []

    for iteration in range(num_iterations):
        for packed1, packed2, packed3 in packed_chunks:
            # Measure unpacking time
            start_time = time.perf_counter()
            unpacked1, unpacked2, unpacked3 = unpack_3bank_chunk(packed1, packed2, packed3)
            end_time = time.perf_counter()

            elapsed_ns = (end_time - start_time) * 1e9
            times_per_chunk.append(elapsed_ns)

        if (iteration + 1) % 20 == 0:
            print(f"  Completed iteration {iteration + 1}/{num_iterations}...")

    print(f"  ✓ Completed {num_iterations} iterations")
    print()

    # Calculate statistics
    times_per_chunk = np.array(times_per_chunk)

    stats = {
        'num_chunks': num_chunks,
        'num_iterations': num_iterations,
        'total_operations': len(times_per_chunk),
        'min_ns': np.min(times_per_chunk),
        'median_ns': np.median(times_per_chunk),
        'mean_ns': np.mean(times_per_chunk),
        'p95_ns': np.percentile(times_per_chunk, 95),
        'p99_ns': np.percentile(times_per_chunk, 99),
        'max_ns': np.max(times_per_chunk),
        'std_ns': np.std(times_per_chunk),
        'target_ns': 50.0,
    }

    stats['meets_target'] = stats['median_ns'] < stats['target_ns']
    stats['p95_meets_target'] = stats['p95_ns'] < stats['target_ns']

    return stats


def print_benchmark_results(stats: Dict):
    """Print formatted benchmark results."""
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print()

    print(f"Operations: {stats['total_operations']:,} unpacking operations")
    print()

    print("Unpacking time per chunk:")
    print(f"  Min:    {stats['min_ns']:>8.2f} ns")
    print(f"  Median: {stats['median_ns']:>8.2f} ns  {'✓ PASSES' if stats['meets_target'] else '✗ FAILS'}")
    print(f"  Mean:   {stats['mean_ns']:>8.2f} ns")
    print(f"  P95:    {stats['p95_ns']:>8.2f} ns  {'✓ PASSES' if stats['p95_meets_target'] else '✗ FAILS'}")
    print(f"  P99:    {stats['p99_ns']:>8.2f} ns")
    print(f"  Max:    {stats['max_ns']:>8.2f} ns")
    print(f"  StdDev: {stats['std_ns']:>8.2f} ns")
    print()

    print(f"Target: <{stats['target_ns']:.0f} ns per chunk")
    print()

    # Overhead analysis
    print("Query-path overhead analysis:")
    overhead_percent = (stats['median_ns'] / stats['target_ns']) * 100
    print(f"  Median overhead: {overhead_percent:.1f}% of target")

    if stats['meets_target']:
        margin = stats['target_ns'] - stats['median_ns']
        print(f"  Margin: {margin:.2f} ns below target ({margin/stats['target_ns']*100:.1f}% headroom)")
    else:
        excess = stats['median_ns'] - stats['target_ns']
        print(f"  ⚠️  Exceeds target by: {excess:.2f} ns ({excess/stats['target_ns']*100:.1f}% over)")
    print()

    # Comparison to query time budget
    query_budget_us = 10.0  # 10 μs from Phase 1 Week 4 validation gate
    overhead_percent_of_budget = (stats['median_ns'] / 1000) / query_budget_us * 100
    print(f"Query time budget context:")
    print(f"  Target query time: {query_budget_us:.1f} μs (Phase 1 Week 4 gate)")
    print(f"  Unpacking overhead: {stats['median_ns']/1000:.3f} μs ({overhead_percent_of_budget:.2f}% of budget)")
    print()

    print("=" * 80)
    if stats['meets_target']:
        print("✓ BENCHMARK PASSED - Unpacking overhead meets <50ns target")
    else:
        print("✗ BENCHMARK FAILED - Unpacking overhead exceeds target")
    print("=" * 80)


if __name__ == '__main__':
    h5_path = 'output/encoded_genome_3banks.h5'

    if not Path(h5_path).exists():
        print(f"Error: Encoded genome file not found: {h5_path}")
        sys.exit(1)

    # Run benchmark
    stats = benchmark_unpacking_overhead(
        h5_path=h5_path,
        num_chunks=1000,
        num_iterations=100,
        seed=42,
    )

    # Print results
    print_benchmark_results(stats)

    # Exit with appropriate code
    if stats['meets_target']:
        print()
        print("Ready to proceed to SIMD dot products and cache-aligned storage.")
        sys.exit(0)
    else:
        print()
        print("⚠️  Unpacking overhead exceeds target. Consider optimization:")
        print("  1. Numba JIT compilation (may need warmup)")
        print("  2. Vectorized bit operations")
        print("  3. SIMD instructions")
        sys.exit(1)
