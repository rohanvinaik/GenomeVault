#!/usr/bin/env python3
"""
Test that proof compression/decompression works correctly.
Verifies that proofs are never truncated and can be fully recovered.
"""

import json
import zlib
import hashlib


def compress_proof(proof_data: dict) -> bytes:
    """
    Compress proof data using the CPROOF format.

    Args:
        proof_data: Dictionary containing proof components

    Returns:
        Compressed proof with header
    """
    # Serialize to JSON with deterministic ordering
    proof_json = json.dumps(proof_data, sort_keys=True, separators=(",", ":"))
    proof_bytes = proof_json.encode("utf-8")

    # Compress with maximum compression
    compressed = zlib.compress(proof_bytes, level=9)

    # Add header: CPROOF + original size (4 bytes) + compressed data
    header = b"CPROOF"
    size_bytes = len(proof_bytes).to_bytes(4, "big")

    return header + size_bytes + compressed


def decompress_proof(compressed_data: bytes) -> dict:
    """
    Decompress proof data from CPROOF format.

    Args:
        compressed_data: Compressed proof with header

    Returns:
        Original proof dictionary
    """
    # Check header
    if not compressed_data.startswith(b"CPROOF"):
        raise ValueError("Invalid proof format - missing CPROOF header")

    # Extract original size
    original_size = int.from_bytes(compressed_data[6:10], "big")

    # Extract compressed data
    compressed_part = compressed_data[10:]

    # Decompress
    decompressed = zlib.decompress(compressed_part)

    # Verify size
    if len(decompressed) != original_size:
        raise ValueError(f"Size mismatch: expected {original_size}, got {len(decompressed)}")

    # Parse JSON
    return json.loads(decompressed.decode("utf-8"))


def test_proof_compression():
    """Test proof compression and decompression."""

    print("=" * 50)

    # Test 1: Small proof
    small_proof = {
        "variant_commitment": hashlib.sha256(b"test_variant").hexdigest(),
        "computed_root": hashlib.sha256(b"root1").hexdigest(),
        "expected_root": hashlib.sha256(b"root1").hexdigest(),
        "verification_passed": True,
        "path_length": 20,
        "catalytic_verification": True,
    }

    compressed_small = compress_proof(small_proof)
    original_json = json.dumps(small_proof, sort_keys=True, separators=(",", ":"))

    print(f"  Original size: {len(original_json)} bytes")
    print(f"  Compressed size: {len(compressed_small)} bytes (with header)")
    print(f"  Compression ratio: {len(original_json) / len(compressed_small):.2f}x")

    # Test decompression
    recovered_small = decompress_proof(compressed_small)
    assert recovered_small == small_proof, "Small proof recovery failed!"
    print("  ✓ Decompression successful")

    # Test 2: Large proof with many fields
    large_proof = {
        "proof_id": hashlib.sha256(b"proof123").hexdigest(),
        "circuit_name": "variant_presence",
        "variant_commitments": [hashlib.sha256(f"var{i}".encode()).hexdigest() for i in range(100)],
        "merkle_paths": [
            [hashlib.sha256(f"node{i}_{j}".encode()).hexdigest() for j in range(20)]
            for i in range(10)
        ],
        "public_inputs": {
            "threshold": 0.95,
            "reference": "GRCh38",
            "timestamp": 1234567890,
        },
        "verification_passed": True,
        "metadata": {
            "prover_version": "1.2.3",
            "circuit_constraints": 50000,
            "proof_system": "PLONK",
        },
    }

    compressed_large = compress_proof(large_proof)
    original_json = json.dumps(large_proof, sort_keys=True, separators=(",", ":"))

    print(f"  Original size: {len(original_json)} bytes")
    print(f"  Compressed size: {len(compressed_large)} bytes (with header)")
    print(f"  Compression ratio: {len(original_json) / len(compressed_large):.2f}x")

    # Test decompression
    recovered_large = decompress_proof(compressed_large)
    assert recovered_large == large_proof, "Large proof recovery failed!"
    print("  ✓ Decompression successful")

    # Test 3: Show why truncation is bad
    pass  # Debug print removed

    # Truncate the original JSON (old bad approach)
    truncated_json = original_json[:256]
    print(f"  Truncated to 256 bytes: '{truncated_json[:50]}...{truncated_json[-20:]}'")

    try:
        # Try to parse truncated JSON
        json.loads(truncated_json)
        print("  ✗ Truncated JSON somehow parsed (shouldn't happen)")
    except json.JSONDecodeError as e:
        print(f"  ✓ Truncated JSON is invalid: {e}")

    # Show that compressed proof preserves everything
    print(
        f"\n  Compressed proof preserves all {len(large_proof['variant_commitments'])} commitments"
    )
    print(f"  Compressed proof preserves all {len(large_proof['merkle_paths'])} paths")
    print("  ✓ No data loss with compression")

    # Test 4: Compression efficiency for different data types
    pass  # Debug print removed

    # Random hashes (don't compress well)
    random_proof = {
        "hashes": [hashlib.sha256(f"random{i}".encode()).hexdigest() for i in range(100)]
    }
    compressed_random = compress_proof(random_proof)
    random_json = json.dumps(random_proof, sort_keys=True, separators=(",", ":"))
    print(
        f"  Random hashes: {len(random_json)} → {len(compressed_random)} bytes "
        f"({len(random_json) / len(compressed_random):.2f}x)"
    )

    # Repeated data (compresses well)
    repeated_proof = {"values": ["same_value"] * 100}
    compressed_repeated = compress_proof(repeated_proof)
    repeated_json = json.dumps(repeated_proof, sort_keys=True, separators=(",", ":"))
    print(
        f"  Repeated data: {len(repeated_json)} → {len(compressed_repeated)} bytes "
        f"({len(repeated_json) / len(compressed_repeated):.2f}x)"
    )

    # Structured data (moderate compression)
    structured_proof = {
        f"field_{i}": {
            "type": "measurement",
            "value": i * 0.1,
            "unit": "mg/dL",
            "timestamp": 1234567890 + i,
        }
        for i in range(50)
    }
    compressed_structured = compress_proof(structured_proof)
    structured_json = json.dumps(structured_proof, sort_keys=True, separators=(",", ":"))
    print(
        f"  Structured data: {len(structured_json)} → {len(compressed_structured)} bytes "
        f"({len(structured_json) / len(compressed_structured):.2f}x)"
    )

    print("Conclusion: Compression preserves all proof data while reducing size.")
    print("Truncation destroys proofs and must NEVER be used!")


if __name__ == "__main__":
    test_proof_compression()
