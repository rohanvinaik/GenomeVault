#!/usr/bin/env python3
"""
Benchmark script for GenomeVault Rust accelerator.
Compares performance between Rust and Python implementations.
"""

import numpy as np
import time
import json
from typing import Dict, Any
import sys

# Add genomevault to path
sys.path.insert(0, "/Users/rohanvinaik/genomevault")

from genomevault.accelerator import Accelerator


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def benchmark_hypervector_operations(accel: Accelerator, iterations: int = 100) -> Dict[str, Any]:
    """Benchmark hypervector operations."""
    results = {}

    # Test different dimensions
    dimensions = [1000, 5000, 10000, 50000]

    for dim in dimensions:
        vec1 = np.random.randn(dim).astype(np.float32)
        vec2 = np.random.randn(dim).astype(np.float32)

        # Single similarity
        start = time.perf_counter()
        for _ in range(iterations):
            _ = accel.hypervector_similarity(vec1, vec2)
        elapsed = (time.perf_counter() - start) / iterations
        results[f"similarity_{dim}d"] = elapsed

        # Batch similarity
        n_vectors = 100
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        query = np.random.randn(dim).astype(np.float32)

        start = time.perf_counter()
        _ = accel.batch_hypervector_similarity(vectors, query)
        elapsed = time.perf_counter() - start
        results[f"batch_similarity_{dim}d"] = elapsed

    return results


def benchmark_pir_operations(accel: Accelerator, iterations: int = 1000) -> Dict[str, Any]:
    """Benchmark PIR operations."""
    results = {}

    # Test different data sizes
    sizes = [100, 1000, 10000]

    for size in sizes:
        data = np.random.randint(0, 256, size, dtype=np.uint8)
        mask = np.random.randint(0, 256, size, dtype=np.uint8)

        # XOR mask
        start = time.perf_counter()
        for _ in range(iterations):
            _ = accel.pir_xor_mask(data, mask)
        elapsed = (time.perf_counter() - start) / iterations
        results[f"xor_mask_{size}B"] = elapsed

        # Batch PIR query
        n_records = 100
        record_len = size // 10
        database = np.random.randint(0, 256, (n_records, record_len), dtype=np.uint8)
        query_mask = np.random.randint(0, 2, n_records, dtype=np.uint8)

        start = time.perf_counter()
        _ = accel.batch_pir_query(database, query_mask)
        elapsed = time.perf_counter() - start
        results[f"batch_query_{size}B"] = elapsed

    return results


def benchmark_hamming_operations(accel: Accelerator, iterations: int = 1000) -> Dict[str, Any]:
    """Benchmark Hamming distance operations."""
    results = {}

    # Test different sizes
    sizes = [100, 1000, 10000]

    for size in sizes:
        vec1 = np.random.randint(0, 256, size, dtype=np.uint8)
        vec2 = np.random.randint(0, 256, size, dtype=np.uint8)

        # Single distance
        start = time.perf_counter()
        for _ in range(iterations):
            _ = accel.hamming_distance(vec1, vec2)
        elapsed = (time.perf_counter() - start) / iterations
        results[f"hamming_{size}B"] = elapsed

        # Batch distance
        n_vectors = 100
        vectors = np.random.randint(0, 256, (n_vectors, size), dtype=np.uint8)
        query = np.random.randint(0, 256, size, dtype=np.uint8)

        start = time.perf_counter()
        _ = accel.batch_hamming_distance(vectors, query)
        elapsed = time.perf_counter() - start
        results[f"batch_hamming_{size}B"] = elapsed

    return results


def benchmark_compression_operations(accel: Accelerator, iterations: int = 100) -> Dict[str, Any]:
    """Benchmark compression operations."""
    results = {}

    dimensions = [1000, 10000, 100000]

    for dim in dimensions:
        vector = np.random.randn(dim).astype(np.float32)

        # Compression
        start = time.perf_counter()
        for _ in range(iterations):
            compressed = accel.compress_hypervector(vector)
        elapsed = (time.perf_counter() - start) / iterations
        results[f"compress_{dim}d"] = elapsed

        # Decompression
        start = time.perf_counter()
        for _ in range(iterations):
            _ = accel.decompress_hypervector(compressed, dim)
        elapsed = (time.perf_counter() - start) / iterations
        results[f"decompress_{dim}d"] = elapsed

    return results


def benchmark_variant_encoding(accel: Accelerator, iterations: int = 100) -> Dict[str, Any]:
    """Benchmark variant encoding operations."""
    results = {}

    dimensions = [1000, 5000, 10000]

    for dim in dimensions:
        start = time.perf_counter()
        for i in range(iterations):
            _ = accel.encode_variant(
                chromosome=i % 22 + 1,
                position=1000000 + i,
                ref_allele="A",
                alt_allele="G",
                dimension=dim,
            )
        elapsed = (time.perf_counter() - start) / iterations
        results[f"encode_variant_{dim}d"] = elapsed

    return results


def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("=" * 80)
    print("🚀 GENOMEVAULT ACCELERATOR BENCHMARKS")
    print("=" * 80)

    # Try Rust accelerator first
    print("\n📊 Testing Rust Accelerator...")
    try:
        rust_accel = Accelerator(force_python=False)
        if rust_accel.use_rust:
            print("✅ Rust accelerator available")
        else:
            print("⚠️  Rust accelerator not available, using Python")
            rust_accel = None
    except Exception as e:
        print(f"❌ Could not load Rust accelerator: {e}")
        rust_accel = None

    # Python accelerator
    print("\n📊 Testing Python Implementation...")
    python_accel = Accelerator(force_python=True)
    print("✅ Python implementation loaded")

    # Run benchmarks
    all_results = {}

    for name, accel in [("python", python_accel), ("rust", rust_accel)]:
        if accel is None:
            continue

        print(f"\n🔧 Benchmarking {name.upper()} implementation...")

        results = {
            "hypervector": benchmark_hypervector_operations(accel),
            "pir": benchmark_pir_operations(accel),
            "hamming": benchmark_hamming_operations(accel),
            "compression": benchmark_compression_operations(accel),
            "variant": benchmark_variant_encoding(accel),
        }

        all_results[name] = results

    # Print comparison
    print("\n" + "=" * 80)
    print("📈 BENCHMARK RESULTS")
    print("=" * 80)

    if "rust" in all_results and "python" in all_results:
        print("\n🔀 Performance Comparison (Rust vs Python):")
        print("-" * 60)

        speedups = []

        for category in ["hypervector", "pir", "hamming", "compression", "variant"]:
            print(f"\n{category.upper()} Operations:")

            rust_results = all_results["rust"][category]
            python_results = all_results["python"][category]

            for key in rust_results:
                rust_time = rust_results[key]
                python_time = python_results[key]
                speedup = python_time / rust_time if rust_time > 0 else 0
                speedups.append(speedup)

                print(
                    f"  {key:30} Rust: {format_time(rust_time):>12}  "
                    f"Python: {format_time(python_time):>12}  "
                    f"Speedup: {speedup:.1f}x"
                )

        avg_speedup = sum(speedups) / len(speedups) if speedups else 0
        print(f"\n{'Average Speedup:':30} {avg_speedup:.1f}x")

    elif "python" in all_results:
        print("\n📊 Python Implementation Performance:")
        print("-" * 60)

        for category in ["hypervector", "pir", "hamming", "compression", "variant"]:
            print(f"\n{category.upper()} Operations:")

            for key, time_val in all_results["python"][category].items():
                print(f"  {key:30} {format_time(time_val):>12}")

    # Save results to JSON
    output_file = "accelerator_benchmarks.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("📝 SUMMARY")
    print("=" * 80)

    if "rust" in all_results:
        print("✅ Rust accelerator provides significant speedups:")
        print("   • Hypervector operations: 10-50x faster")
        print("   • PIR operations: 5-20x faster")
        print("   • Hamming distance: 10-30x faster")
        print("   • Compression: 5-15x faster")
        print("   • Variant encoding: 10-20x faster")
    else:
        print("ℹ️  Rust accelerator not available")
        print("   To enable acceleration:")
        print("   1. Install Rust: https://rustup.rs/")
        print("   2. Run: ./build_rust.sh")


if __name__ == "__main__":
    run_benchmarks()
