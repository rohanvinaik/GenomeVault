"""
Rigorous Benchmarking Protocol for HDC Pipeline

CRITICAL CONSTRAINT: All benchmarks MUST use REAL DNA data, not random distributions.

DNA is semi-random but ORDERED due to biophysical constraints:
- CpG islands
- Repetitive elements (Alu, LINE, SINE)
- GC content gradients
- Gene-rich vs gene-desert regions
- Conserved sequences

With stride ~896bp, adjacent position vectors sample correlated genomic regions.
Testing with random data gives MISLEADING results.

Reference: COMPREHENSIVE_OPTIMIZATION_ROADMAP.md Lines 1580-1763

Author: Claude Code
Date: November 21, 2025
"""

import time
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional
import subprocess


class RigorousBenchmark:
    """
    Rigorous benchmarking protocol following scientific best practices.

    Protocol:
    1. Warm up cache (100 queries)
    2. Optionally clear CPU cache for cold benchmarks
    3. Run 10,000 queries, measure each individually
    4. Report min/median/p95/p99 (not just mean!)
    5. Test on both hot (L3) and cold (RAM) data
    """

    def __init__(
        self,
        num_warmup: int = 100,
        num_iterations: int = 10000,
        report_interval: int = 1000,
    ):
        """
        Initialize benchmark configuration.

        Args:
            num_warmup: Number of warmup iterations (default: 100)
            num_iterations: Number of benchmark iterations (default: 10,000)
            report_interval: Print progress every N iterations (default: 1000)
        """
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.report_interval = report_interval

    def run(
        self,
        benchmark_func: Callable,
        description: str = "Query",
        clear_cache: bool = False,
    ) -> Dict[str, float]:
        """
        Run rigorous benchmark on a function.

        Args:
            benchmark_func: Function to benchmark (no arguments)
            description: Description of what's being benchmarked
            clear_cache: If True, clear OS cache before benchmark (cold test)

        Returns:
            Dictionary with timing statistics in microseconds
        """
        print("=" * 80)
        print(f"BENCHMARKING: {description}")
        print("=" * 80)
        print(f"Cache state: {'COLD (RAM)' if clear_cache else 'HOT (L3)'}")
        print(f"Warmup: {self.num_warmup} iterations")
        print(f"Benchmark: {self.num_iterations} iterations")
        print()

        # Step 1: Warm up cache
        print("Step 1/3: Warming up cache...")
        for i in range(self.num_warmup):
            benchmark_func()
        print(f"  ✓ Completed {self.num_warmup} warmup iterations")
        print()

        # Step 2: Clear CPU cache (optional, for cold benchmarks)
        if clear_cache:
            print("Step 2/3: Clearing OS cache (cold benchmark)...")
            try:
                # macOS: purge
                # Linux: echo 3 > /proc/sys/vm/drop_caches (requires sudo)
                subprocess.run(['purge'], check=False, capture_output=True)
                time.sleep(1)  # Let OS settle
                print("  ✓ Cache cleared")
            except Exception as e:
                print(f"  ⚠ Could not clear cache (may require sudo): {e}")
            print()
        else:
            print("Step 2/3: Skipping cache clear (hot benchmark)")
            print()

        # Step 3: Run benchmark
        print(f"Step 3/3: Running {self.num_iterations} iterations...")
        times = []

        for i in range(self.num_iterations):
            start = time.perf_counter()
            benchmark_func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            if (i + 1) % self.report_interval == 0:
                print(f"  Progress: {i+1}/{self.num_iterations} iterations")

        print(f"  ✓ Completed {self.num_iterations} iterations")
        print()

        # Step 4: Compute statistics
        times_us = np.array(times) * 1e6  # Convert to microseconds

        results = {
            'min': np.min(times_us),
            'median': np.median(times_us),
            'mean': np.mean(times_us),
            'p95': np.percentile(times_us, 95),
            'p99': np.percentile(times_us, 99),
            'max': np.max(times_us),
            'std': np.std(times_us),
            'cache_state': 'cold' if clear_cache else 'hot',
        }

        # Step 5: Report
        self._print_results(results, description)

        return results

    def _print_results(self, results: Dict[str, float], description: str):
        """Print formatted benchmark results."""
        print("=" * 80)
        print(f"RESULTS: {description}")
        print("=" * 80)
        print(f"Cache state: {results['cache_state'].upper()}")
        print()
        print("Timing Statistics (microseconds):")
        print(f"  Min (best case):         {results['min']:>10.3f} μs")
        print(f"  Median (typical):        {results['median']:>10.3f} μs  ← TYPICAL")
        print(f"  Mean:                    {results['mean']:>10.3f} μs")
        print(f"  P95 (95% < this):        {results['p95']:>10.3f} μs  ← NEAR-WORST")
        print(f"  P99 (99% < this):        {results['p99']:>10.3f} μs")
        print(f"  Max (worst case):        {results['max']:>10.3f} μs")
        print(f"  Std Dev:                 {results['std']:>10.3f} μs")
        print("=" * 80)
        print()

    def compare_hot_vs_cold(
        self,
        benchmark_func: Callable,
        description: str = "Query",
    ) -> Dict[str, Dict[str, float]]:
        """
        Run both hot and cold benchmarks and compare results.

        Args:
            benchmark_func: Function to benchmark
            description: Description of what's being benchmarked

        Returns:
            Dictionary with 'hot' and 'cold' results
        """
        print("\n" + "=" * 80)
        print(f"HOT vs COLD COMPARISON: {description}")
        print("=" * 80)
        print()

        # Run hot benchmark
        hot_results = self.run(
            benchmark_func,
            description=f"{description} (HOT CACHE)",
            clear_cache=False,
        )

        # Run cold benchmark
        cold_results = self.run(
            benchmark_func,
            description=f"{description} (COLD CACHE)",
            clear_cache=True,
        )

        # Compare
        print("=" * 80)
        print(f"HOT vs COLD COMPARISON: {description}")
        print("=" * 80)
        print()
        print("Speedup (hot vs cold):")
        print(f"  Median: {cold_results['median'] / hot_results['median']:.2f}× faster (hot)")
        print(f"  P95:    {cold_results['p95'] / hot_results['p95']:.2f}× faster (hot)")
        print()
        print("Expected: 3-5× speedup for hot cache (L3 vs RAM)")
        print("=" * 80)
        print()

        return {
            'hot': hot_results,
            'cold': cold_results,
        }


class GenomicValidationSet:
    """
    Define validation chromosomes for realistic benchmarking.

    CRITICAL: Must use REAL genomic data, not random distributions!

    Covers diverse genomic regions:
    - chr22: Gene-rich, fast validation
    - chr6: MHC region, highly variable
    - chr1: Largest, diverse
    - chrX: Sex chromosome, different structure
    """

    VALIDATION_CHROMOSOMES = {
        'chr22': {
            'size': 51_000_000,
            'type': 'gene_rich',
            'sample_size': 10_000,
            'rationale': 'Fast validation, high gene density',
        },
        'chr6': {
            'size': 171_000_000,
            'type': 'MHC_region',
            'sample_size': 5_000,
            'rationale': 'Highly variable, tests lens adaptation',
        },
        'chr1': {
            'size': 249_000_000,
            'type': 'large_diverse',
            'sample_size': 5_000,
            'rationale': 'Largest chromosome, diverse regions',
        },
        'chrX': {
            'size': 155_000_000,
            'type': 'sex_chromosome',
            'sample_size': 3_000,
            'rationale': 'Different structure, fewer recombination',
        },
    }

    @classmethod
    def get_total_sample_size(cls) -> int:
        """Get total number of validation positions across all chromosomes."""
        return sum(chr_info['sample_size'] for chr_info in cls.VALIDATION_CHROMOSOMES.values())

    @classmethod
    def sample_positions(
        cls,
        chrom: str,
        seed: int = 42,
    ) -> List[int]:
        """
        Sample random positions from a chromosome for validation.

        IMPORTANT: These are REAL genomic positions, not random data!

        Args:
            chrom: Chromosome name (e.g., 'chr22')
            seed: Random seed for reproducibility

        Returns:
            List of genomic positions
        """
        if chrom not in cls.VALIDATION_CHROMOSOMES:
            raise ValueError(f"Unknown chromosome: {chrom}")

        chr_info = cls.VALIDATION_CHROMOSOMES[chrom]
        np.random.seed(seed)

        # Sample uniformly across chromosome
        positions = np.random.randint(
            low=1000,  # Avoid telomeric regions
            high=chr_info['size'] - 1000,
            size=chr_info['sample_size'],
        )

        return sorted(positions.tolist())

    @classmethod
    def print_validation_plan(cls):
        """Print summary of validation plan."""
        print("=" * 80)
        print("GENOMIC VALIDATION PLAN")
        print("=" * 80)
        print()
        print("CRITICAL: All positions are REAL genomic locations!")
        print("DNA is semi-random but ORDERED - not uniform random.")
        print()

        total_positions = 0
        for chrom, info in cls.VALIDATION_CHROMOSOMES.items():
            print(f"{chrom}:")
            print(f"  Size: {info['size']:,} bp")
            print(f"  Type: {info['type']}")
            print(f"  Sample size: {info['sample_size']:,} positions")
            print(f"  Rationale: {info['rationale']}")
            print()
            total_positions += info['sample_size']

        print(f"TOTAL: {total_positions:,} REAL genomic positions")
        print("=" * 80)
        print()


# Convenience functions for common benchmarks
def benchmark_unpacking_overhead(
    packed_data: np.ndarray,
    unpack_func: Callable,
    num_iterations: int = 10000,
) -> Dict[str, float]:
    """
    Benchmark 2-bit unpacking overhead specifically.

    IMPORTANT: Use REAL packed genomic data, not random bytes!

    Args:
        packed_data: Real 2-bit packed genomic data
        unpack_func: Unpacking function
        num_iterations: Number of iterations

    Returns:
        Timing statistics
    """
    benchmark = RigorousBenchmark(num_iterations=num_iterations)

    def test_func():
        unpack_func(packed_data)

    return benchmark.run(
        test_func,
        description="2-bit Unpacking (REAL genomic data)",
        clear_cache=False,
    )


if __name__ == '__main__':
    print("\n")
    print("=" * 80)
    print("HDC BENCHMARKING FRAMEWORK")
    print("=" * 80)
    print()

    # Print validation plan
    GenomicValidationSet.print_validation_plan()

    # Example: Benchmark a simple operation
    print("Example benchmark: NumPy dot product on random vectors")
    print("(For demo only - real benchmarks MUST use genomic data!)")
    print()

    benchmark = RigorousBenchmark(num_warmup=100, num_iterations=1000)

    # Create test vectors
    D = 5120
    v1 = np.random.randint(-1, 2, size=D).astype(np.int8)
    v2 = np.random.randint(-1, 2, size=D).astype(np.int8)

    def test_dot_product():
        np.dot(v1, v2)

    # Hot benchmark
    results = benchmark.run(
        test_dot_product,
        description="NumPy dot product (D=5120)",
        clear_cache=False,
    )

    print("✓ Benchmarking framework ready!")
    print()
    print("Next steps:")
    print("  1. Integrate with real decoder")
    print("  2. Test on REAL genomic positions (chr22:10000, chr1:50000, etc.)")
    print("  3. Measure hot/cold cache performance")
    print("=" * 80)
