"""
Benchmark RAW SIMD Dot Product Performance (M1/M2)

This benchmarks the pure dot product operation: bank · position_vector
without database scanning overhead.

Target: <10 μs for single dot product (D=5,120)

Author: Claude Code
Date: November 21, 2025
"""

import numpy as np
import time
from numba import njit


@njit(cache=True, fastmath=True)
def dotproduct_numba(a: np.ndarray, b: np.ndarray) -> float:
    """
    Numba-JIT optimized dot product for ternary int8 vectors.

    Args:
        a: First vector (D,) int8
        b: Second vector (D,) int8

    Returns:
        Dot product (scalar)
    """
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


def benchmark_dotproduct(
    D: int = 5120,
    num_iterations: int = 10_000,
    seed: int = 42
) -> dict:
    """
    Benchmark raw dot product performance.

    Tests both NumPy (BLAS/Accelerate) and Numba implementations.

    Args:
        D: Dimension of vectors
        num_iterations: Number of dot products to compute
        seed: Random seed

    Returns:
        Performance statistics
    """
    print("=" * 80)
    print("Raw SIMD Dot Product Benchmark (M1/M2)")
    print("=" * 80)
    print()
    print(f"Dimension: {D:,}")
    print(f"Iterations: {num_iterations:,}")
    print(f"Target: <10 μs per dot product")
    print()

    # Generate random ternary vectors
    np.random.seed(seed)
    vectors_a = np.random.choice([-1, 0, 1], size=(num_iterations, D)).astype(np.int8)
    vectors_b = np.random.choice([-1, 0, 1], size=(num_iterations, D)).astype(np.int8)

    # Warm-up for NumPy
    print("Warming up NumPy (BLAS/Accelerate)...")
    for i in range(100):
        _ = np.dot(vectors_a[i], vectors_b[i])
    print("  ✓ NumPy warm-up complete")

    # Warm-up for Numba
    print("Warming up Numba JIT...")
    for i in range(100):
        _ = dotproduct_numba(vectors_a[i], vectors_b[i])
    print("  ✓ Numba warm-up complete")
    print()

    # Benchmark NumPy
    print("Benchmarking NumPy (Apple Accelerate SIMD)...")
    times_numpy_ns = []
    for i in range(num_iterations):
        start = time.perf_counter_ns()
        result = np.dot(vectors_a[i], vectors_b[i])
        end = time.perf_counter_ns()
        times_numpy_ns.append(end - start)

        if (i + 1) % 2000 == 0:
            print(f"  Progress: {i + 1}/{num_iterations}")

    print(f"  ✓ Completed {num_iterations} NumPy dot products")
    print()

    # Benchmark Numba
    print("Benchmarking Numba JIT...")
    times_numba_ns = []
    for i in range(num_iterations):
        start = time.perf_counter_ns()
        result = dotproduct_numba(vectors_a[i], vectors_b[i])
        end = time.perf_counter_ns()
        times_numba_ns.append(end - start)

        if (i + 1) % 2000 == 0:
            print(f"  Progress: {i + 1}/{num_iterations}")

    print(f"  ✓ Completed {num_iterations} Numba dot products")
    print()

    # Calculate statistics
    times_numpy_ns = np.array(times_numpy_ns)
    times_numba_ns = np.array(times_numba_ns)

    stats = {
        'num_iterations': num_iterations,
        'D': D,
        'target_ns': 10_000.0,  # 10 μs
        'numpy': {
            'min_ns': float(np.min(times_numpy_ns)),
            'median_ns': float(np.median(times_numpy_ns)),
            'mean_ns': float(np.mean(times_numpy_ns)),
            'p95_ns': float(np.percentile(times_numpy_ns, 95)),
            'p99_ns': float(np.percentile(times_numpy_ns, 99)),
            'max_ns': float(np.max(times_numpy_ns)),
            'std_ns': float(np.std(times_numpy_ns)),
        },
        'numba': {
            'min_ns': float(np.min(times_numba_ns)),
            'median_ns': float(np.median(times_numba_ns)),
            'mean_ns': float(np.mean(times_numba_ns)),
            'p95_ns': float(np.percentile(times_numba_ns, 95)),
            'p99_ns': float(np.percentile(times_numba_ns, 99)),
            'max_ns': float(np.max(times_numba_ns)),
            'std_ns': float(np.std(times_numba_ns)),
        },
    }

    stats['numpy']['meets_target'] = stats['numpy']['median_ns'] < stats['target_ns']
    stats['numba']['meets_target'] = stats['numba']['median_ns'] < stats['target_ns']

    # Determine best
    if stats['numpy']['median_ns'] < stats['numba']['median_ns']:
        stats['best_implementation'] = 'numpy'
        stats['best_median_ns'] = stats['numpy']['median_ns']
    else:
        stats['best_implementation'] = 'numba'
        stats['best_median_ns'] = stats['numba']['median_ns']

    stats['meets_target'] = stats['best_median_ns'] < stats['target_ns']

    return stats


def print_benchmark_results(stats: dict):
    """Print formatted benchmark results."""
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print()

    print(f"Vector dimension: D = {stats['D']:,}")
    print(f"Iterations: {stats['num_iterations']:,}")
    print(f"Target: <{stats['target_ns'] / 1000:.1f} μs per dot product")
    print()

    print("-" * 80)
    print("NumPy (Apple Accelerate BLAS + SIMD)")
    print("-" * 80)
    numpy_stats = stats['numpy']
    print(f"  Min:    {numpy_stats['min_ns'] / 1000:>8.2f} μs")
    print(f"  Median: {numpy_stats['median_ns'] / 1000:>8.2f} μs  {'✓ MEETS' if numpy_stats['meets_target'] else '✗ EXCEEDS'}")
    print(f"  Mean:   {numpy_stats['mean_ns'] / 1000:>8.2f} μs")
    print(f"  P95:    {numpy_stats['p95_ns'] / 1000:>8.2f} μs")
    print(f"  P99:    {numpy_stats['p99_ns'] / 1000:>8.2f} μs")
    print(f"  Max:    {numpy_stats['max_ns'] / 1000:>8.2f} μs")
    print(f"  StdDev: {numpy_stats['std_ns'] / 1000:>8.2f} μs")
    print()

    print("-" * 80)
    print("Numba JIT (with fastmath)")
    print("-" * 80)
    numba_stats = stats['numba']
    print(f"  Min:    {numba_stats['min_ns'] / 1000:>8.2f} μs")
    print(f"  Median: {numba_stats['median_ns'] / 1000:>8.2f} μs  {'✓ MEETS' if numba_stats['meets_target'] else '✗ EXCEEDS'}")
    print(f"  Mean:   {numba_stats['mean_ns'] / 1000:>8.2f} μs")
    print(f"  P95:    {numba_stats['p95_ns'] / 1000:>8.2f} μs")
    print(f"  P99:    {numba_stats['p99_ns'] / 1000:>8.2f} μs")
    print(f"  Max:    {numba_stats['max_ns'] / 1000:>8.2f} μs")
    print(f"  StdDev: {numba_stats['std_ns'] / 1000:>8.2f} μs")
    print()

    print("=" * 80)
    print(f"BEST: {stats['best_implementation'].upper()} ({stats['best_median_ns'] / 1000:.2f} μs median)")
    print("=" * 80)
    print()

    if stats['meets_target']:
        margin = stats['target_ns'] - stats['best_median_ns']
        print("✓ BENCHMARK PASSED")
        print(f"  Margin: {margin / 1000:.2f} μs below target ({margin / stats['target_ns'] * 100:.1f}% headroom)")
        print()
        print("  SIMD dot products are ready for production use.")
        print("  Proceed to Phase 1 Week 3: Smart binary search and lens integration.")
    else:
        excess = stats['best_median_ns'] - stats['target_ns']
        print("✗ BENCHMARK FAILED")
        print(f"  Exceeds target by: {excess / 1000:.2f} μs ({excess / stats['target_ns'] * 100:.1f}% over)")
        print()
        print("  Optimization recommendations:")
        print("  1. Profile with Instruments.app (Xcode) to find hot paths")
        print("  2. Verify Apple Accelerate framework is being used (import scipy.show_config())")
        print("  3. Try explicit SIMD intrinsics via Cython")
        print("  4. Consider dimension reduction (but verify accuracy impact)")

    print()
    print("=" * 80)


if __name__ == '__main__':
    # Run benchmark
    stats = benchmark_dotproduct(
        D=5120,
        num_iterations=10_000,
        seed=42
    )

    # Print results
    print_benchmark_results(stats)

    # Exit with appropriate code
    import sys
    sys.exit(0 if stats['meets_target'] else 1)
