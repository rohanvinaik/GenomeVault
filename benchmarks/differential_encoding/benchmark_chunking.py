#!/usr/bin/env python3
"""
Benchmark: Adaptive Genomic Chunking Performance

Tests all chunking strategies with realistic genomic data.
Measures time, memory, chunk counts, and distribution.
"""

import json
import argparse
import time
import tracemalloc
import random
from typing import Dict, Any, List

from genomevault.differential_encoding import (
    Genome,
    Variant,
    AnalysisType,
    CryptographicChunker,
    STRATEGY_CONFIGS,
    CryptoRNG,
)


def create_test_genome(n_variants: int = 10000, chromosome: str = "chr1") -> Genome:
    """Create realistic test genome with variable variant distribution."""
    random.seed(42)
    variants = []
    position = 100000

    for i in range(n_variants):
        position += random.randint(100, 10000)  # Variable spacing
        variants.append(Variant(
            chromosome=chromosome,
            position=position,
            ref=random.choice(['A', 'C', 'G', 'T']),
            alt=random.choice(['A', 'C', 'G', 'T']),
            genotype=random.choice(['0/1', '1/1']),
            quality=random.uniform(20, 99),
        ))

    return Genome(
        genome_id="benchmark_test",
        assembly="GRCh38",
        chromosomes={chromosome: variants}
    )


def benchmark_strategy(
    genome: Genome,
    analysis_type: AnalysisType,
    rng: CryptoRNG,
    iterations: int = 3
) -> Dict[str, Any]:
    """Benchmark a single chunking strategy."""

    strategy = STRATEGY_CONFIGS[analysis_type]
    chunker = CryptographicChunker(strategy, rng)

    times = []
    memory_peaks = []
    chunk_counts = []

    for _ in range(iterations):
        # Measure memory
        tracemalloc.start()

        # Measure time
        start_time = time.perf_counter()
        master_seed = rng.derive_seed(f"{analysis_type.value}_bench".encode())
        chunks = chunker.chunk_genome(genome, analysis_type, master_seed)
        end_time = time.perf_counter()

        # Get memory peak
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append((end_time - start_time) * 1000)  # Convert to ms
        memory_peaks.append(peak / (1024 * 1024))  # Convert to MB
        chunk_counts.append(len(chunks))

    # Calculate statistics
    avg_time = sum(times) / len(times)
    avg_memory = sum(memory_peaks) / len(memory_peaks)
    avg_chunks = sum(chunk_counts) / len(chunk_counts)

    return {
        "time_ms": round(avg_time, 2),
        "memory_mb": round(avg_memory, 1),
        "chunks": int(avg_chunks),
        "min_time_ms": round(min(times), 2),
        "max_time_ms": round(max(times), 2),
    }


def run_benchmarks(quick: bool = False) -> int:
    """Run all chunking strategy benchmarks."""

    # Create test genome
    n_variants = 5000 if quick else 10000
    genome = create_test_genome(n_variants=n_variants)

    rng = CryptoRNG()
    iterations = 3 if quick else 5

    # Strategies to benchmark
    strategies_to_test = [
        AnalysisType.SLIDING_WINDOW,
        AnalysisType.GENE_REGION,
        AnalysisType.WHOLE_CHROMOSOME,
        AnalysisType.GWAS_ASSOCIATION,
        AnalysisType.STRUCTURAL_VARIANT,
    ]

    results = {
        "test_genome": {
            "variants": n_variants,
            "iterations": iterations,
        },
        "strategies": {}
    }

    # Run benchmarks for each strategy
    best_time = float('inf')
    best_strategy = None

    for analysis_type in strategies_to_test:
        try:
            print(f"Benchmarking {analysis_type.value}...", flush=True)
            strategy_results = benchmark_strategy(genome, analysis_type, rng, iterations)
            results["strategies"][analysis_type.value] = strategy_results

            if strategy_results["time_ms"] < best_time:
                best_time = strategy_results["time_ms"]
                best_strategy = analysis_type.value

        except Exception as e:
            print(f"Error benchmarking {analysis_type.value}: {e}", flush=True)
            results["strategies"][analysis_type.value] = {
                "error": str(e),
                "time_ms": 0,
                "memory_mb": 0,
                "chunks": 0
            }

    # Calculate averages
    valid_results = [r for r in results["strategies"].values() if "error" not in r]
    if valid_results:
        avg_time = sum(r["time_ms"] for r in valid_results) / len(valid_results)
        results["best_strategy"] = best_strategy
        results["avg_time_ms"] = round(avg_time, 2)
    else:
        results["best_strategy"] = None
        results["avg_time_ms"] = 0

    # Output JSON
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark adaptive chunking strategies"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmarks with reduced iterations'
    )
    args = parser.parse_args()

    exit(run_benchmarks(quick=args.quick))
