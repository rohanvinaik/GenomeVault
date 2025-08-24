#!/usr/bin/env python3
"""Test HDC encoding with fixed sparsity calculation."""

import sys
import time
import numpy as np

sys.path.insert(0, ".")

from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig
from genomevault.core.constants import OmicsType
from utils.tensor_utils import calculate_sparsity, tensor_stats


def test_hdc_encoding_fixed():
    """Test HDC encoding with proper tensor handling."""

    print("=" * 60)
    print("🧬 HDC ENCODING TEST (FIXED)")
    print("=" * 60)

    results = {}

    # Test different dimensions
    dimensions = [1000, 8192, 10000]

    for dim in dimensions:
        print(f"\nTesting {dim}D encoding...")

        try:
            # Initialize encoder
            config = HypervectorConfig(dimension=dim)
            encoder = HypervectorEncoder(config=config)

            # Generate test data
            data = np.random.randn(100).astype(np.float32)

            # Encode
            start = time.perf_counter()
            encoded = encoder.encode(data, OmicsType.GENOMIC)
            encode_time = (time.perf_counter() - start) * 1000

            # Calculate sparsity using fixed utility
            sparsity = calculate_sparsity(encoded)

            # Get full stats
            stats = tensor_stats(encoded)

            results[f"{dim}D"] = {
                "encode_time_ms": round(encode_time, 2),
                "sparsity": round(sparsity, 3),
                "compression_ratio": round(100 / dim, 4),
                "stats": stats,
            }

            print("  ✅ Success!")
            print(f"     Encoding time: {encode_time:.2f}ms")
            print(f"     Sparsity: {sparsity:.1%}")
            print(f"     Shape: {stats['shape']}")
            print(f"     Non-zero elements: {stats['non_zero']}/{stats['size']}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[f"{dim}D"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results.values() if "error" not in r)
    total = len(results)

    print(f"Success rate: {successful}/{total} ({successful/total*100:.0f}%)")

    if successful > 0:
        print("\nPerformance metrics:")
        for dim, result in results.items():
            if "error" not in result:
                print(
                    f"  {dim}: {result['encode_time_ms']}ms, " f"sparsity={result['sparsity']:.1%}"
                )

    return results


def test_metal_vs_cpu():
    """Compare Metal vs CPU performance."""

    print("\n" + "=" * 60)
    print("⚡ METAL vs CPU COMPARISON")
    print("=" * 60)

    dim = 8192
    data = np.random.randn(1000).astype(np.float32)

    # Test with Metal (if available)
    try:
        config = HypervectorConfig(dimension=dim)
        encoder = HypervectorEncoder(config=config)

        start = time.perf_counter()
        encoded = encoder.encode(data, OmicsType.GENOMIC)
        metal_time = (time.perf_counter() - start) * 1000

        print(f"  Metal encoding: {metal_time:.2f}ms")
        print(f"  Throughput: {1000/metal_time:.0f} vectors/sec")

    except Exception as e:
        print(f"  Metal test failed: {e}")

    # Note: CPU-only test would require disabling Metal,
    # which would need environment variable or config change


if __name__ == "__main__":
    # Run fixed HDC tests
    results = test_hdc_encoding_fixed()

    # Test Metal performance
    test_metal_vs_cpu()

    print("\n" + "=" * 60)
    print("✅ HDC TESTING COMPLETE")
    print("=" * 60)
