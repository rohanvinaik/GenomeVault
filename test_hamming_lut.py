"""Test script for Hamming LUT implementation"""

import numpy as np
import torch

from genomevault.hypervector.operations import HammingLUT, generate_popcount_lut
from genomevault.hypervector.operations.binding import HypervectorBinder
from genomevault.observability.logging import configure_logging

logger = configure_logging()


def test_lut_generation():
    """Test LUT generation"""
    print("Testing LUT generation...")
    lut = generate_popcount_lut()

    # Verify some known values
    assert lut[0] == 0, "Popcount of 0 should be 0"
    assert lut[0xFFFF] == 16, "Popcount of 0xFFFF should be 16"
    assert lut[0xFF] == 8, "Popcount of 0xFF should be 8"
    assert lut[0x5555] == 8, "Popcount of 0x5555 should be 8"

    print(f"✓ LUT generated successfully, size: {lut.nbytes / 1024:.1f} KB")


def test_hamming_distance():
    """Test Hamming distance computation"""
    print("\nTesting Hamming distance computation...")

    # Create test vectors
    dim = 10000
    vec1 = np.random.randint(0, 2, dim, dtype=np.uint8)
    vec2 = np.random.randint(0, 2, dim, dtype=np.uint8)

    # Pack vectors
    vec1_packed = np.packbits(vec1).view(np.uint64)
    vec2_packed = np.packbits(vec2).view(np.uint64)

    # Compute with LUT
    lut_calc = HammingLUT(use_gpu=False)
    lut_distance = lut_calc.distance(vec1_packed, vec2_packed)

    # Compute standard way
    standard_distance = np.sum(vec1 != vec2)

    print(f"Standard distance: {standard_distance}")
    print(f"LUT distance: {lut_distance}")
    print(f"Match: {standard_distance == lut_distance}")

    assert standard_distance == lut_distance, "Distances should match!"
    print("✓ Hamming distance computation correct")


def test_batch_computation():
    """Test batch Hamming distance computation"""
    print("\nTesting batch Hamming distance...")

    # Create test batches
    n, m, dim = 10, 15, 5000
    vecs1 = np.random.randint(0, 2, (n, dim), dtype=np.uint8)
    vecs2 = np.random.randint(0, 2, (m, dim), dtype=np.uint8)

    # Pack vectors
    vecs1_packed = np.array([np.packbits(v).view(np.uint64) for v in vecs1])
    vecs2_packed = np.array([np.packbits(v).view(np.uint64) for v in vecs2])

    # Compute with LUT
    lut_calc = HammingLUT(use_gpu=False)
    lut_distances = lut_calc.distance_batch(vecs1_packed, vecs2_packed)

    # Verify shape
    assert lut_distances.shape == (
        n,
        m,
    ), f"Expected shape {(n, m)}, got {lut_distances.shape}"

    # Spot check a few values
    for i in range(min(3, n)):
        for j in range(min(3, m)):
            expected = np.sum(vecs1[i] != vecs2[j])
            actual = lut_distances[i, j]
            assert expected == actual, f"Mismatch at ({i},{j}): {expected} vs {actual}"

    print(f"✓ Batch computation correct, shape: {lut_distances.shape}")


def test_integration_with_binder():
    """Test integration with HypervectorBinder"""
    print("\nTesting integration with HypervectorBinder...")

    # Create binder with LUT
    binder = HypervectorBinder(dimension=10000, use_gpu=False)

    # Create test vectors
    vec1 = torch.randn(10000)
    vec2 = torch.randn(10000)

    # Test Hamming similarity
    similarity = binder.hamming_similarity(vec1, vec2)
    print(f"Hamming similarity: {similarity:.4f}")

    assert 0 <= similarity <= 1, "Similarity should be between 0 and 1"
    print("✓ HypervectorBinder integration successful")


def test_performance_comparison():
    """Quick performance comparison"""
    print("\nQuick performance test (10000D vectors)...")

    import time

    dim = 10000
    vec1 = np.random.randint(0, 2, dim, dtype=np.uint8)
    vec2 = np.random.randint(0, 2, dim, dtype=np.uint8)

    # Standard computation
    start = time.perf_counter()
    for _ in range(100):
        _ = np.sum(vec1 != vec2)
    standard_time = time.perf_counter() - start

    # LUT computation
    vec1_packed = np.packbits(vec1).view(np.uint64)
    vec2_packed = np.packbits(vec2).view(np.uint64)
    lut_calc = HammingLUT(use_gpu=False)

    start = time.perf_counter()
    for _ in range(100):
        _ = lut_calc.distance(vec1_packed, vec2_packed)
    lut_time = time.perf_counter() - start

    speedup = standard_time / lut_time
    print(f"Standard time: {standard_time * 1000:.2f} ms")
    print(f"LUT time: {lut_time * 1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")


def main():
    """Run all tests"""
    print("=" * 60)
    print("HAMMING LUT IMPLEMENTATION TEST")
    print("=" * 60)

    test_lut_generation()
    test_hamming_distance()
    test_batch_computation()
    test_integration_with_binder()
    test_performance_comparison()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
