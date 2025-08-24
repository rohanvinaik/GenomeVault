#!/usr/bin/env python3
"""Test the secure RNG and XOF module."""

from genomevault.crypto.rng import secure_bytes, xof, xof_uint_mod


def test_secure_rng():
    """Test secure random number generation and XOF."""

    print("=" * 50)

    # Test secure_bytes
    random1 = secure_bytes(32)
    random2 = secure_bytes(32)

    assert len(random1) == 32
    assert len(random2) == 32
    assert random1 != random2  # Should be different with overwhelming probability

    print(f"  Random 1: {random1.hex()[:32]}...")
    print(f"  Random 2: {random2.hex()[:32]}...")
    print("  ✓ Generates different random bytes each call")

    # Test zero-length
    zero_bytes = secure_bytes(0)
    assert zero_bytes == b""
    print("  ✓ Handles zero-length request")

    # Test XOF (Extensible Output Function)
    pass  # Debug print removed

    label = b"TEST_LABEL"
    seed = b"test_seed_12345"

    # Generate different length outputs
    out16 = xof(label, seed, 16)
    out32 = xof(label, seed, 32)
    out64 = xof(label, seed, 64)

    assert len(out16) == 16
    assert len(out32) == 32
    assert len(out64) == 64

    # First 16 bytes should match
    assert out32[:16] == out16
    assert out64[:16] == out16
    assert out64[:32] == out32

    print(f"  XOF(16): {out16.hex()}")
    print(f"  XOF(32): {out32.hex()[:32]}...")
    print(f"  XOF(64): {out64.hex()[:32]}...")
    print("  ✓ XOF produces consistent extensible output")

    # Different labels produce different outputs
    out_diff = xof(b"DIFFERENT_LABEL", seed, 32)
    assert out_diff != out32
    print("  ✓ Different labels produce different outputs")

    # Different seeds produce different outputs
    out_diff_seed = xof(label, b"different_seed", 32)
    assert out_diff_seed != out32
    print("  ✓ Different seeds produce different outputs")

    # Test domain separation with length prefixing
    concat_out = xof(b"AB", b"CD", 32)
    split_out = xof(b"A", b"BCD", 32)
    assert concat_out != split_out
    print("  ✓ Length prefixing prevents concatenation attacks")

    # Test xof_uint_mod
    pass  # Debug print removed

    modulus = 1000
    samples = []

    # Generate samples with different seeds
    for i in range(100):
        seed_i = seed + i.to_bytes(4, "big")
        sample = xof_uint_mod(label, seed_i, modulus)
        assert 0 <= sample < modulus
        samples.append(sample)

    # Check distribution (should be roughly uniform)
    min_val = min(samples)
    max_val = max(samples)
    avg_val = sum(samples) / len(samples)

    print(f"  100 samples mod {modulus}:")
    print(f"    Min: {min_val}, Max: {max_val}, Avg: {avg_val:.1f}")
    print(f"    Expected avg: {(modulus-1)/2:.1f}")

    # With 100 samples, we should see good coverage
    unique_samples = len(set(samples))
    print(f"    Unique values: {unique_samples}/100")
    assert unique_samples > 50  # Should have many unique values
    print("  ✓ Produces well-distributed values")

    # Test determinism
    val1 = xof_uint_mod(label, seed, modulus)
    val2 = xof_uint_mod(label, seed, modulus)
    assert val1 == val2
    print("  ✓ Deterministic for same inputs")

    # Test large modulus
    large_mod = 2**60
    large_val = xof_uint_mod(label, seed, large_mod)
    assert 0 <= large_val < large_mod
    print(f"  ✓ Handles large modulus (2^60): {large_val}")

    pass  # Debug print removed


if __name__ == "__main__":
    test_secure_rng()
