#!/usr/bin/env python3
"""Test the proof I/O compression module."""

from genomevault.crypto.proof_io import compress_proof, decompress_proof
from genomevault.crypto.serialization import pack_proof_components, be_int
import hashlib


def test_proof_io():
    """Test proof compression and decompression."""

    print("=" * 50)

    # Test basic compression/decompression
    pass  # Debug print removed

    # Create a simple proof
    simple_proof = b"This is a test proof " * 10
    print(f"  Original size: {len(simple_proof)} bytes")

    compressed = compress_proof(simple_proof)
    print(f"  Compressed size: {len(compressed)} bytes")
    print(f"  Compression ratio: {len(simple_proof) / len(compressed):.2f}x")

    # Check format
    assert compressed.startswith(b"CPROOF\0")
    print("  ✓ Has correct magic header")

    # Decompress
    decompressed = decompress_proof(compressed)
    assert decompressed == simple_proof
    print("  ✓ Decompresses to original")

    # Test with typical proof data
    pass  # Debug print removed

    proof_components = {
        "commitment": hashlib.sha256(b"commitment").digest(),
        "challenge": hashlib.sha256(b"challenge").digest(),
        "response": hashlib.sha256(b"response").digest(),
        "merkle_root": hashlib.sha256(b"root").digest(),
        "path_length": be_int(20, 4),
        "witness": b"\xaa" * 1024,  # 1KB witness data
    }

    raw_proof = pack_proof_components(proof_components)
    print(f"  Raw proof size: {len(raw_proof)} bytes")

    compressed_proof = compress_proof(raw_proof)
    print(f"  Compressed size: {len(compressed_proof)} bytes")
    print(f"  Compression ratio: {len(raw_proof) / len(compressed_proof):.2f}x")

    decompressed_proof = decompress_proof(compressed_proof)
    assert decompressed_proof == raw_proof
    print("  ✓ Proof components round-trip correctly")

    # Test with highly compressible data
    pass  # Debug print removed

    repetitive = b"A" * 10000
    compressed_rep = compress_proof(repetitive)
    print(f"  10KB of 'A's -> {len(compressed_rep)} bytes")
    print(f"  Compression ratio: {len(repetitive) / len(compressed_rep):.2f}x")

    assert decompress_proof(compressed_rep) == repetitive
    print("  ✓ Highly compressible data works")

    # Test with random (incompressible) data
    pass  # Debug print removed

    random_data = hashlib.sha256(b"seed").digest() * 100  # 3.2KB of hashes
    compressed_rand = compress_proof(random_data)
    print(f"  3.2KB random -> {len(compressed_rand)} bytes")
    print(f"  Compression ratio: {len(random_data) / len(compressed_rand):.2f}x")

    assert decompress_proof(compressed_rand) == random_data
    print("  ✓ Random data works (low compression expected)")

    # Test empty proof
    pass  # Debug print removed

    empty = b""
    compressed_empty = compress_proof(empty)
    decompressed_empty = decompress_proof(compressed_empty)
    assert decompressed_empty == empty
    print("  ✓ Empty proof works")

    # Test large proof (no truncation!)
    large = b"X" * 1000000  # 1MB
    compressed_large = compress_proof(large)
    decompressed_large = decompress_proof(compressed_large)
    assert len(decompressed_large) == 1000000
    assert decompressed_large == large
    print("  ✓ 1MB proof preserves all data (NO TRUNCATION)")

    # Test version checking
    pass  # Debug print removed

    # Bad magic
    bad_magic = b"BADMAGIC" + compressed[7:]
    try:
        decompress_proof(bad_magic)
        assert False, "Should have rejected bad magic"
    except ValueError as e:
        print(f"  ✓ Bad magic rejected: {e}")

    # Bad version
    bad_version = b"CPROOF\0" + b"\x00\x02" + compressed[9:]
    try:
        decompress_proof(bad_version)
        assert False, "Should have rejected bad version"
    except ValueError as e:
        print(f"  ✓ Bad version rejected: {e}")

    # Corrupted size
    bad_size = compressed[:9] + b"\xff" * 8 + compressed[17:]
    try:
        decompress_proof(bad_size)
        assert False, "Should have detected size mismatch"
    except ValueError as e:
        print(f"  ✓ Size mismatch detected: {e}")

    # Test format details
    pass  # Debug print removed

    test_data = b"test"
    test_compressed = compress_proof(test_data)

    # Parse format
    magic = test_compressed[:7]
    version = int.from_bytes(test_compressed[7:9], "big")
    orig_size = int.from_bytes(test_compressed[9:17], "big")

    print(f"  Magic: {magic}")
    print(f"  Version: {version}")
    print(f"  Original size field: {orig_size}")
    pass  # Debug print removed

    assert magic == b"CPROOF\0"
    assert version == 1
    assert orig_size == len(test_data)
    print("  ✓ Format structure correct")

    # Demonstrate NO TRUNCATION
    pass  # Debug print removed

    # Create a proof with critical data at the end
    critical_proof = b"START" + b"\x00" * 10000 + b"CRITICAL_DATA_AT_END"
    compressed_critical = compress_proof(critical_proof)
    decompressed_critical = decompress_proof(compressed_critical)

    assert decompressed_critical.endswith(b"CRITICAL_DATA_AT_END")
    print("  ✓ Critical data at end preserved")

    # Unlike truncation, compression preserves everything
    truncated = critical_proof[:256]  # BAD: loses data
    assert not truncated.endswith(b"CRITICAL_DATA_AT_END")
    print("  ✗ Truncation would lose critical data")
    print("  ✓ Compression preserves ALL data")

    pass  # Debug print removed


if __name__ == "__main__":
    test_proof_io()
