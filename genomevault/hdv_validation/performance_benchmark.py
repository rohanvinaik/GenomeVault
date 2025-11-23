#!/usr/bin/env python3
"""
Performance benchmarking for HDV quantization modes.

Measures:
- Load time
- Query latency (single and batch)
- Memory usage
- Throughput (queries per second)
- Scalability with different query patterns

Usage:
    python performance_benchmark.py --quantization float32
    python performance_benchmark.py --all-quantizations --iterations 5
"""

import argparse
import logging
import sys
import time
import json
import h5py
import numpy as np
import psutil
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add parent directory to path

from genomevault.hdv_validation.query_engine import PreEncodedMultiLensHDV

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def benchmark_quantization(
    quantization='float32',
    n_queries=1000,
    n_iterations=3,
    output_dir=None
):
    """
    Comprehensive performance benchmark for a quantization mode.
    """
    if output_dir is None:
        output_dir = Path("HDV_VALIDATION_PACKAGE/architecture_testing/benchmarks")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info(f"PERFORMANCE BENCHMARK: {quantization.upper()}")
    logger.info("=" * 80)
    logger.info("")

    # Paths - use correct quantized 3D files
    base_dir = Path("data/experimental_strands/ERR3239334/hdv_encoding")
    if quantization == 'float32':
        hdf5_path = base_dir / "encoded_genome_5lenses_3d.h5"
    elif quantization == 'int8':
        hdf5_path = base_dir / "encoded_genome_5lenses_3d_int8.h5"
    elif quantization == 'int4':
        hdf5_path = base_dir / "encoded_genome_5lenses_3d_int4.h5"
    elif quantization == 'binary':
        hdf5_path = base_dir / "encoded_genome_5lenses_3d_binary.h5"
    else:
        raise ValueError(f"Unknown quantization mode: {quantization}")

    guide_fasta_dir = Path("/Volumes/1TBStorage/guide_strands")

    logger.info(f"H5 file: {hdf5_path}")
    logger.info("")

    # Get chunk keys for random sampling
    with h5py.File(hdf5_path, 'r') as f:
        chunk_keys_bytes = f['chunk_keys'][:]
        chunk_keys = [k.decode('utf-8') for k in chunk_keys_bytes]
    
    logger.info(f"Total chunks available: {len(chunk_keys):,}")
    logger.info(f"Query count: {n_queries:,}")
    logger.info(f"Iterations: {n_iterations}")
    logger.info("")
    
    results = {
        'quantization': quantization,
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'n_queries': n_queries,
            'n_iterations': n_iterations
        },
        'benchmarks': {}
    }
    
    # =========================================================================
    # BENCHMARK 1: System Load Time
    # =========================================================================
    logger.info("=" * 80)
    logger.info("BENCHMARK 1: System Load Time")
    logger.info("=" * 80)
    logger.info("")
    
    load_times = []
    memory_before_load = []
    memory_after_load = []
    
    for i in range(n_iterations):
        logger.info(f"Iteration {i+1}/{n_iterations}...")
        
        mem_before = get_memory_usage()
        memory_before_load.append(mem_before)
        
        start_time = time.time()
        hdv = PreEncodedMultiLensHDV(hdf5_path, guide_fasta_dir=guide_fasta_dir, quantization=quantization)
        load_time = time.time() - start_time
        
        mem_after = get_memory_usage()
        memory_after_load.append(mem_after)
        
        load_times.append(load_time)
        
        logger.info(f"  Load time: {load_time:.3f}s")
        logger.info(f"  Memory: {mem_before:.1f} MB → {mem_after:.1f} MB (Δ {mem_after - mem_before:.1f} MB)")
        
        hdv.close()
        
        # Force garbage collection between iterations
        import gc
        gc.collect()
        time.sleep(0.5)
    
    logger.info("")
    logger.info("Load Time Statistics:")
    logger.info(f"  Mean:   {np.mean(load_times):.3f}s")
    logger.info(f"  Median: {np.median(load_times):.3f}s")
    logger.info(f"  Std:    {np.std(load_times):.3f}s")
    logger.info(f"  Min:    {np.min(load_times):.3f}s")
    logger.info(f"  Max:    {np.max(load_times):.3f}s")
    logger.info("")
    
    memory_deltas = [after - before for before, after in zip(memory_before_load, memory_after_load)]
    logger.info("Memory Usage Statistics:")
    logger.info(f"  Mean delta:   {np.mean(memory_deltas):.1f} MB")
    logger.info(f"  Median delta: {np.median(memory_deltas):.1f} MB")
    logger.info("")
    
    results['benchmarks']['load_time'] = {
        'mean_seconds': float(np.mean(load_times)),
        'median_seconds': float(np.median(load_times)),
        'std_seconds': float(np.std(load_times)),
        'min_seconds': float(np.min(load_times)),
        'max_seconds': float(np.max(load_times)),
        'all_times': [float(t) for t in load_times]
    }
    
    results['benchmarks']['memory_usage'] = {
        'mean_delta_mb': float(np.mean(memory_deltas)),
        'median_delta_mb': float(np.median(memory_deltas)),
        'all_deltas_mb': [float(d) for d in memory_deltas]
    }
    
    # =========================================================================
    # BENCHMARK 2: Single Query Latency
    # =========================================================================
    logger.info("=" * 80)
    logger.info("BENCHMARK 2: Single Query Latency")
    logger.info("=" * 80)
    logger.info("")
    
    # Load system once for remaining benchmarks
    hdv = PreEncodedMultiLensHDV(hdf5_path, guide_fasta_dir=guide_fasta_dir, quantization=quantization)
    
    # Sample random positions
    N = 2000
    sample_positions = []
    for _ in range(n_queries):
        chunk_key = np.random.choice(chunk_keys)
        chrom, chunk_start_str = chunk_key.split(':')
        chunk_start = int(chunk_start_str)
        pos = chunk_start + np.random.randint(0, N)
        sample_positions.append((chrom, pos))
    
    logger.info(f"Sampled {len(sample_positions):,} random positions")
    logger.info("")
    
    # Warm-up queries (not counted)
    logger.info("Warming up (10 queries)...")
    for i in range(10):
        chrom, pos = sample_positions[i]
        hdv.query_position_all_lenses(chrom, pos)
    logger.info("  ✓ Warm-up complete")
    logger.info("")
    
    # Measure individual query latencies
    logger.info("Measuring single query latency...")
    query_times = []
    
    start_idx = 10  # Skip warm-up queries
    for i in range(start_idx, min(start_idx + n_queries, len(sample_positions))):
        chrom, pos = sample_positions[i]
        
        start_time = time.perf_counter()
        hdv.query_position_all_lenses(chrom, pos)
        query_time = time.perf_counter() - start_time
        
        query_times.append(query_time * 1000)  # Convert to ms
        
        if (i - start_idx + 1) % 100 == 0:
            logger.info(f"  Progress: {i - start_idx + 1}/{n_queries}")
    
    logger.info("")
    logger.info("Single Query Latency Statistics:")
    logger.info(f"  Mean:   {np.mean(query_times):.3f} ms")
    logger.info(f"  Median: {np.median(query_times):.3f} ms")
    logger.info(f"  Std:    {np.std(query_times):.3f} ms")
    logger.info(f"  Min:    {np.min(query_times):.3f} ms")
    logger.info(f"  Max:    {np.max(query_times):.3f} ms")
    logger.info(f"  P50:    {np.percentile(query_times, 50):.3f} ms")
    logger.info(f"  P95:    {np.percentile(query_times, 95):.3f} ms")
    logger.info(f"  P99:    {np.percentile(query_times, 99):.3f} ms")
    logger.info("")
    
    results['benchmarks']['single_query_latency'] = {
        'mean_ms': float(np.mean(query_times)),
        'median_ms': float(np.median(query_times)),
        'std_ms': float(np.std(query_times)),
        'min_ms': float(np.min(query_times)),
        'max_ms': float(np.max(query_times)),
        'p50_ms': float(np.percentile(query_times, 50)),
        'p95_ms': float(np.percentile(query_times, 95)),
        'p99_ms': float(np.percentile(query_times, 99)),
        'sample_size': len(query_times)
    }
    
    # =========================================================================
    # BENCHMARK 3: Batch Query Throughput
    # =========================================================================
    logger.info("=" * 80)
    logger.info("BENCHMARK 3: Batch Query Throughput")
    logger.info("=" * 80)
    logger.info("")
    
    batch_sizes = [100, 500, 1000, 5000]
    throughput_results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(sample_positions):
            continue
        
        logger.info(f"Testing batch size: {batch_size}")
        
        batch_positions = sample_positions[:batch_size]
        
        start_time = time.perf_counter()
        for chrom, pos in batch_positions:
            hdv.query_position_all_lenses(chrom, pos)
        elapsed_time = time.perf_counter() - start_time
        
        throughput = batch_size / elapsed_time
        avg_latency = (elapsed_time / batch_size) * 1000
        
        throughput_results[str(batch_size)] = {
            'elapsed_seconds': float(elapsed_time),
            'queries_per_second': float(throughput),
            'avg_latency_ms': float(avg_latency)
        }
        
        logger.info(f"  Elapsed: {elapsed_time:.3f}s")
        logger.info(f"  Throughput: {throughput:.1f} queries/sec")
        logger.info(f"  Avg latency: {avg_latency:.3f} ms")
        logger.info("")
    
    results['benchmarks']['batch_throughput'] = throughput_results
    
    # =========================================================================
    # BENCHMARK 4: Query Pattern Analysis
    # =========================================================================
    logger.info("=" * 80)
    logger.info("BENCHMARK 4: Query Pattern Analysis")
    logger.info("=" * 80)
    logger.info("")
    
    # Sequential vs Random access patterns
    
    # Sequential: Same chunk, different positions
    logger.info("Testing sequential access (same chunk)...")
    chunk_key = np.random.choice(chunk_keys)
    chrom, chunk_start_str = chunk_key.split(':')
    chunk_start = int(chunk_start_str)
    
    sequential_times = []
    for i in range(min(100, N)):
        pos = chunk_start + i
        start_time = time.perf_counter()
        hdv.query_position_all_lenses(chrom, pos)
        sequential_times.append((time.perf_counter() - start_time) * 1000)
    
    logger.info(f"  Sequential access mean latency: {np.mean(sequential_times):.3f} ms")
    logger.info("")
    
    # Random: Different chunks
    logger.info("Testing random access (different chunks)...")
    random_times = []
    for _ in range(100):
        chunk_key = np.random.choice(chunk_keys)
        chrom, chunk_start_str = chunk_key.split(':')
        chunk_start = int(chunk_start_str)
        pos = chunk_start + np.random.randint(0, N)
        
        start_time = time.perf_counter()
        hdv.query_position_all_lenses(chrom, pos)
        random_times.append((time.perf_counter() - start_time) * 1000)
    
    logger.info(f"  Random access mean latency: {np.mean(random_times):.3f} ms")
    logger.info("")
    
    logger.info(f"Access Pattern Comparison:")
    logger.info(f"  Sequential: {np.mean(sequential_times):.3f} ms")
    logger.info(f"  Random:     {np.mean(random_times):.3f} ms")
    logger.info(f"  Ratio:      {np.mean(random_times) / np.mean(sequential_times):.2f}x slower for random")
    logger.info("")
    
    results['benchmarks']['query_patterns'] = {
        'sequential': {
            'mean_ms': float(np.mean(sequential_times)),
            'median_ms': float(np.median(sequential_times)),
            'std_ms': float(np.std(sequential_times))
        },
        'random': {
            'mean_ms': float(np.mean(random_times)),
            'median_ms': float(np.median(random_times)),
            'std_ms': float(np.std(random_times))
        },
        'random_to_sequential_ratio': float(np.mean(random_times) / np.mean(sequential_times))
    }
    
    # Close system
    hdv.close()
    
    # =========================================================================
    # Save Results
    # =========================================================================
    results_file = output_dir / f"{quantization}_performance_benchmark.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✓ Results saved to: {results_file}")
    logger.info("")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Quantization: {quantization.upper()}")
    logger.info("")
    logger.info(f"Load Time:        {results['benchmarks']['load_time']['mean_seconds']:.3f}s")
    logger.info(f"Memory Delta:     {results['benchmarks']['memory_usage']['mean_delta_mb']:.1f} MB")
    logger.info(f"Query Latency:    {results['benchmarks']['single_query_latency']['mean_ms']:.3f} ms (mean)")
    logger.info(f"                  {results['benchmarks']['single_query_latency']['p95_ms']:.3f} ms (p95)")
    logger.info(f"Throughput:       {results['benchmarks']['batch_throughput']['1000']['queries_per_second']:.1f} queries/sec")
    logger.info("")
    
    return results


def benchmark_all_quantizations(n_queries=1000, n_iterations=3, output_dir=None):
    """Run benchmarks for all quantization modes and generate comparison."""
    
    if output_dir is None:
        output_dir = Path("HDV_VALIDATION_PACKAGE/architecture_testing/benchmarks")
    else:
        output_dir = Path(output_dir)
    
    quantizations = ['float32', 'int8', 'int4', 'binary']
    all_results = {}
    
    for quant in quantizations:
        logger.info("")
        logger.info("#" * 80)
        logger.info(f"# BENCHMARKING: {quant.upper()}")
        logger.info("#" * 80)
        logger.info("")
        
        result = benchmark_quantization(
            quantization=quant,
            n_queries=n_queries,
            n_iterations=n_iterations,
            output_dir=output_dir
        )
        all_results[quant] = result
    
    # Generate comparison
    logger.info("")
    logger.info("=" * 80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 80)
    logger.info("")
    
    logger.info(f"{'Quantization':<12} {'Load (s)':<10} {'Memory (MB)':<12} {'Latency (ms)':<14} {'Throughput (qps)':<18}")
    logger.info("-" * 80)
    
    for quant in quantizations:
        r = all_results[quant]['benchmarks']
        logger.info(
            f"{quant:<12} "
            f"{r['load_time']['mean_seconds']:>9.3f} "
            f"{r['memory_usage']['mean_delta_mb']:>11.1f} "
            f"{r['single_query_latency']['mean_ms']:>13.3f} "
            f"{r['batch_throughput']['1000']['queries_per_second']:>17.1f}"
        )
    
    logger.info("")
    
    # Save comparison
    comparison_file = output_dir / "performance_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2)
    
    logger.info(f"✓ Comparison saved to: {comparison_file}")
    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description='Performance benchmarking for HDV quantization modes'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        choices=['float32', 'int8', 'int4', 'binary'],
        help='Quantization mode to benchmark (required if not using --all-quantizations)'
    )
    parser.add_argument(
        '--all-quantizations',
        action='store_true',
        help='Benchmark all quantization modes'
    )
    parser.add_argument(
        '--n-queries',
        type=int,
        default=1000,
        help='Number of queries for latency testing (default: 1000)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='Number of iterations for load time testing (default: 3)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    if args.all_quantizations:
        benchmark_all_quantizations(
            n_queries=args.n_queries,
            n_iterations=args.iterations,
            output_dir=args.output_dir
        )
    elif args.quantization:
        benchmark_quantization(
            quantization=args.quantization,
            n_queries=args.n_queries,
            n_iterations=args.iterations,
            output_dir=args.output_dir
        )
    else:
        parser.error("Either --quantization or --all-quantizations must be specified")


if __name__ == '__main__':
    main()
