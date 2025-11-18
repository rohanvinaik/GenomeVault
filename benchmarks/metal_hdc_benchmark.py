#!/usr/bin/env python3
"""
Metal HDC Acceleration Benchmark

Demonstrates the performance difference between CPU-only and Metal GPU-accelerated
hypervector operations on Apple Silicon.

This is Phase 1 of the Apple Silicon optimization plan.
"""

import time
import numpy as np
from typing import Dict, Any

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from genomevault.compute.metal_backend import MetalBackend
from genomevault.compute.cpu_backend import CPUBackend


def benchmark_cpu_bundling(vectors: np.ndarray, num_runs: int = 10) -> Dict[str, Any]:
    """Benchmark CPU-only HDC bundling (majority vote)."""
    times = []

    for _ in range(num_runs):
        start = time.perf_counter()

        # CPU bundling: sum and threshold
        summed = np.sum(vectors, axis=0)
        threshold = vectors.shape[0] / 2.0
        result = (summed > threshold).astype(np.float32)

        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        "backend": "CPU",
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "throughput_ops_per_sec": 1000 / np.mean(times),
    }


def benchmark_metal_bundling(vectors: np.ndarray, num_runs: int = 10) -> Dict[str, Any]:
    """Benchmark Metal GPU-accelerated HDC bundling."""
    if not MLX_AVAILABLE:
        return {"error": "MLX not available"}

    backend = MetalBackend()

    times = []

    # Warmup
    _ = backend.bundle_vectors(vectors[:10])

    for _ in range(num_runs):
        start = time.perf_counter()

        # Metal bundling
        result = backend.bundle_vectors(vectors)

        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        "backend": "Metal (Apple Silicon)",
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "throughput_ops_per_sec": 1000 / np.mean(times),
    }


def benchmark_encoding(num_samples: int = 1000, input_dim: int = 1000) -> Dict[str, Any]:
    """Benchmark Metal encoding vs CPU baseline."""
    if not MLX_AVAILABLE:
        return {"error": "MLX not available"}

    metal_backend = MetalBackend()
    cpu_backend = CPUBackend()

    # Generate test data
    test_variants = [np.random.randn(100, input_dim).astype(np.float32) for _ in range(num_samples)]

    # Benchmark Metal encoding
    start = time.perf_counter()
    metal_results = metal_backend.encode_batch(test_variants)
    metal_time = (time.perf_counter() - start) * 1000

    # Benchmark CPU encoding
    start = time.perf_counter()
    cpu_results = cpu_backend.encode_batch(test_variants)
    cpu_time = (time.perf_counter() - start) * 1000

    return {
        "samples": num_samples,
        "input_dim": input_dim,
        "output_dim": metal_results.shape[1] if len(metal_results.shape) > 1 else 8192,
        "encoding_time_ms": metal_time,
        "cpu_encoding_time_ms": cpu_time,
        "samples_per_second": num_samples / (metal_time / 1000),
        "speedup": cpu_time / metal_time,
    }


def print_benchmark_results(cpu_results: Dict, metal_results: Dict, test_name: str):
    """Pretty-print benchmark comparison."""
    print(f"\n{'=' * 80}")
    print(f"  {test_name}")
    print(f"{'=' * 80}")

    if "error" in metal_results:
        print(f"⚠️  Metal not available: {metal_results['error']}")
        return

    print(f"\n{'Backend':<25} {'Mean (ms)':<12} {'Throughput (ops/s)':<20}")
    print(f"{'-' * 80}")

    print(f"{cpu_results['backend']:<25} "
          f"{cpu_results['mean_ms']:>10.2f}  "
          f"{cpu_results['throughput_ops_per_sec']:>18.1f}")

    print(f"{metal_results['backend']:<25} "
          f"{metal_results['mean_ms']:>10.2f}  "
          f"{metal_results['throughput_ops_per_sec']:>18.1f}")

    # Calculate speedup
    speedup = cpu_results['mean_ms'] / metal_results['mean_ms']
    print(f"\n{'Speedup:':<25} {speedup:>10.2f}×")

    # Performance improvement
    improvement = ((cpu_results['mean_ms'] - metal_results['mean_ms']) /
                   cpu_results['mean_ms'] * 100)
    print(f"{'Improvement:':<25} {improvement:>10.1f}%")


def main():
    """Run comprehensive Metal HDC benchmark."""
    print("\n" + "=" * 80)
    print("  METAL HDC ACCELERATION BENCHMARK")
    print("  GenomeVault - Apple Silicon Optimization (Phase 1)")
    print("=" * 80)

    if not MLX_AVAILABLE:
        print("\n❌ MLX not available. Please install: pip install mlx")
        print("   Note: MLX only works on Apple Silicon (M1/M2/M3/M4)")
        return

    print(f"\n✅ MLX {mx.__version__ if hasattr(mx, '__version__') else 'available'}")
    print(f"✅ Device: {mx.default_device()}")

    # Test 1: Small bundling (representative of GenomeVault usage)
    print("\n" + "=" * 80)
    print("TEST 1: HDC Bundling - Small (100 vectors × 10,000D)")
    print("=" * 80)

    num_vectors = 100
    dimension = 10000
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)

    cpu_results = benchmark_cpu_bundling(vectors, num_runs=10)
    metal_results = benchmark_metal_bundling(vectors, num_runs=10)

    print_benchmark_results(cpu_results, metal_results, "HDC Bundling (Small)")

    # Test 2: Large bundling (stress test)
    print("\n" + "=" * 80)
    print("TEST 2: HDC Bundling - Large (1000 vectors × 10,000D)")
    print("=" * 80)

    num_vectors = 1000
    vectors_large = np.random.randn(num_vectors, dimension).astype(np.float32)

    cpu_results_large = benchmark_cpu_bundling(vectors_large, num_runs=5)
    metal_results_large = benchmark_metal_bundling(vectors_large, num_runs=5)

    print_benchmark_results(cpu_results_large, metal_results_large, "HDC Bundling (Large)")

    # Test 3: Encoding benchmark
    print("\n" + "=" * 80)
    print("TEST 3: HDC Encoding (100 samples × 1000 features → 8192D)")
    print("=" * 80)

    encoding_results = benchmark_encoding(num_samples=100, input_dim=1000)

    if "error" not in encoding_results:
        print(f"\nEncoding Performance:")
        print(f"  Samples:             {encoding_results['samples']}")
        print(f"  Input dimension:     {encoding_results['input_dim']}")
        print(f"  Output dimension:    {encoding_results['output_dim']}")
        print(f"  Metal time:          {encoding_results['encoding_time_ms']:.2f} ms")
        print(f"  CPU time:            {encoding_results['cpu_encoding_time_ms']:.2f} ms")
        print(f"  Speedup:             {encoding_results['speedup']:.2f}×")
        print(f"  Throughput:          {encoding_results['samples_per_second']:.0f} samples/sec")
        print(f"  Latency per sample:  {encoding_results['encoding_time_ms']/encoding_results['samples']:.3f} ms")

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    bundling_speedup = cpu_results['mean_ms'] / metal_results['mean_ms']
    large_bundling_speedup = cpu_results_large['mean_ms'] / metal_results_large['mean_ms']

    print(f"\nMetal GPU Acceleration Results:")
    print(f"  Small bundling speedup:  {bundling_speedup:.2f}×")
    print(f"  Large bundling speedup:  {large_bundling_speedup:.2f}×")
    if "error" not in encoding_results:
        print(f"  Encoding speedup:        {encoding_results['speedup']:.2f}×")
        print(f"  Encoding throughput:     {encoding_results['samples_per_second']:.0f} samples/sec")

    print(f"\nRecommendation:")
    if bundling_speedup > 2.0:
        print(f"  ✅ Metal GPU provides {bundling_speedup:.1f}× speedup - RECOMMENDED for production!")
        print(f"  ✅ Enable Metal backend in enhanced_privacy_pipeline.py")
    elif bundling_speedup > 1.5:
        print(f"  ⚠️  Metal GPU provides {bundling_speedup:.1f}× speedup - moderate improvement")
        print(f"  ℹ️  Consider enabling for large-scale processing")
    else:
        print(f"  ⚠️  Metal GPU speedup is only {bundling_speedup:.1f}× - overhead may dominate")
        print(f"  ℹ️  CPU may be sufficient for current workload")

    print("\n" + "=" * 80)
    print("  Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
