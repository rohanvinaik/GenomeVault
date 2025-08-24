#!/usr/bin/env python3
"""
Benchmark Metal vs CPU performance for HDC encoding.
"""

import time
from genomevault.compression.tiered_compression import TieredCompressor, CompressionTier
from genomevault.core.constants import OmicsType
from genomevault.hypervector_transform.encoding import HypervectorConfig


def benchmark_hdc(use_metal: bool, num_variants: int = 100000):
    """Benchmark HDC encoding with/without Metal."""

    # Configure to use or not use Metal
    config = HypervectorConfig(use_metal=use_metal)

    # Initialize compressor with custom config
    compressor = TieredCompressor(hdc_config=config)

    # Create test data
    test_data = {"sample_id": f"BENCH_{'METAL' if use_metal else 'CPU'}", "variants": {}}

    for i in range(num_variants):
        test_data["variants"][f"rs{i}"] = int(i % 3)

    # Run compression
    start = time.time()
    compressed, metrics = compressor.compress_to_target(
        test_data, CompressionTier.FULL_HDC, OmicsType.GENOMIC
    )
    elapsed = time.time() - start

    return elapsed, metrics


def main():
    print("\n" + "=" * 60)
    print("METAL vs CPU BENCHMARK")
    print("=" * 60)

    variant_counts = [50000, 100000, 200000]

    for num_variants in variant_counts:
        print(f"\nTesting with {num_variants:,} variants:")
        print("-" * 40)

        # Test with Metal
        print("  Metal acceleration: ", end="", flush=True)
        metal_time, metal_metrics = benchmark_hdc(True, num_variants)
        print(f"{metal_time:.2f}s (ratio: {metal_metrics.compression_ratio:.0f}x)")

        # Test without Metal
        print("  CPU only:          ", end="", flush=True)
        cpu_time, cpu_metrics = benchmark_hdc(False, num_variants)
        print(f"{cpu_time:.2f}s (ratio: {cpu_metrics.compression_ratio:.0f}x)")

        # Calculate speedup
        speedup = cpu_time / metal_time
        print(f"  🚀 Metal speedup:   {speedup:.2f}x faster")

    print("\n" + "=" * 60)
    print("Note: First run includes initialization overhead")
    print("=" * 60)


if __name__ == "__main__":
    main()
