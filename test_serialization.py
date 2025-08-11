#!/usr/bin/env python3
"""Test the canonical serialization module."""

from genomevault.crypto.serialization import (
    be_int,
    bstr,
    varbytes,
    pack_bytes_seq,
    pack_str_list,
    pack_int_list,
    pack_kv_map,
    pack_proof_components,
)
from genomevault.crypto.commit import H, TAGS


def test_canonical_serialization():
    """Test that canonical serialization works correctly."""

    print("Testing Canonical Serialization")
    print("=" * 50)

    # Test be_int
    print("Testing be_int:")
    assert be_int(0, 4) == b"\x00\x00\x00\x00"
    assert be_int(255, 4) == b"\x00\x00\x00\xff"
    assert be_int(256, 4) == b"\x00\x00\x01\x00"
    print(f"  be_int(256, 4) = {be_int(256, 4).hex()}")

    try:
        be_int(-1, 4)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Negative integers rejected: {e}")

    # Test bstr
    print("\nTesting bstr:")
    assert bstr("hello") == b"hello"
    assert bstr("") == b""
    assert bstr("🧬") == b"\xf0\x9f\xa7\xac"  # Unicode emoji
    print(f"  bstr('hello') = {bstr('hello')}")
    print(f"  bstr('🧬') = {bstr('🧬').hex()}")

    # Test varbytes
    print("\nTesting varbytes:")
    vb = varbytes(b"test")
    assert vb == b"\x00\x00\x00\x04test"
    print(f"  varbytes(b'test') = {vb.hex()}")

    # Test pack_bytes_seq
    print("\nTesting pack_bytes_seq:")
    seq = pack_bytes_seq([b"one", b"two", b"three"])
    print(f"  pack_bytes_seq(['one', 'two', 'three']) = {seq.hex()[:32]}...")

    # Verify deterministic
    seq2 = pack_bytes_seq([b"one", b"two", b"three"])
    assert seq == seq2
    print("  ✓ Deterministic encoding")

    # Test pack_str_list
    print("\nTesting pack_str_list:")
    str_list = pack_str_list(["alpha", "beta", "gamma"])
    print(f"  pack_str_list(['alpha', 'beta', 'gamma']) = {str_list.hex()[:32]}...")

    # Test pack_int_list
    print("\nTesting pack_int_list:")
    int_list = pack_int_list([1, 2, 3, 255], limb=4)
    print(f"  pack_int_list([1, 2, 3, 255], limb=4) = {int_list.hex()[:32]}...")

    # Test pack_kv_map - deterministic ordering
    print("\nTesting pack_kv_map (deterministic ordering):")
    map1 = pack_kv_map({"z": b"last", "a": b"first", "m": b"middle"})
    map2 = pack_kv_map({"m": b"middle", "z": b"last", "a": b"first"})
    assert map1 == map2
    print("  ✓ Deterministic ordering by key")
    print(f"  pack_kv_map({{z, a, m}}) = {map1.hex()[:32]}...")

    # Test pack_proof_components
    print("\nTesting pack_proof_components:")
    components = {
        "commitment": b"\xaa" * 32,
        "challenge": b"\xbb" * 32,
        "response": b"\xcc" * 32,
    }
    packed = pack_proof_components(components)
    print(f"  Packed size: {len(packed)} bytes")
    print(f"  Packed data: {packed.hex()[:32]}...")

    # Components should be deterministically ordered
    components2 = {
        "response": b"\xcc" * 32,
        "commitment": b"\xaa" * 32,
        "challenge": b"\xbb" * 32,
    }
    packed2 = pack_proof_components(components2)
    assert packed == packed2
    print("  ✓ Deterministic component ordering")

    # Test canonical serialization for commitments
    print("\nTesting canonical serialization for commitments:")

    # Create proof data
    proof_data = pack_proof_components(
        {
            "variant_hash": be_int(0x12345678, 32),
            "merkle_root": b"\xff" * 32,
            "path_length": be_int(20, 4),
        }
    )

    # Commit to it using domain separation
    commitment = H(TAGS["PROOF_ID"], proof_data)
    print(f"  Commitment: {commitment.hex()[:16]}...")

    # Same data, different order, should produce same commitment
    proof_data2 = pack_proof_components(
        {
            "path_length": be_int(20, 4),
            "variant_hash": be_int(0x12345678, 32),
            "merkle_root": b"\xff" * 32,
        }
    )
    commitment2 = H(TAGS["PROOF_ID"], proof_data2)
    assert commitment == commitment2
    print("  ✓ Order-independent commitment (via canonical serialization)")

    print("\n" + "=" * 50)
    print("All serialization tests passed!")


if __name__ == "__main__":
    test_canonical_serialization()
