#!/usr/bin/env python3
"""Test the canonical commitment module."""

from genomevault.crypto.commit import H, hexH, TAGS


def test_domain_separation():
    """Test that domain separation works correctly."""

    print("Testing Domain-Separated Commitments")
    print("=" * 50)

    # Same data with different tags should produce different hashes
    data = b"test data"

    leaf_commit = H(TAGS["LEAF"], data)
    node_commit = H(TAGS["NODE"], data)
    acc_commit = H(TAGS["ACC"], data)

    print(f"LEAF commit: {leaf_commit.hex()[:16]}...")
    print(f"NODE commit: {node_commit.hex()[:16]}...")
    print(f"ACC commit:  {acc_commit.hex()[:16]}...")

    # All should be different
    assert leaf_commit != node_commit
    assert node_commit != acc_commit
    assert leaf_commit != acc_commit
    print("✓ Domain separation working - different tags produce different hashes")

    # Test multi-part hashing
    print("\nTesting multi-part commitments:")
    part1 = b"part one"
    part2 = b"part two"
    part3 = b"part three"

    multi_commit = H(TAGS["PROOF_ID"], part1, part2, part3)
    print(f"Multi-part: {multi_commit.hex()[:16]}...")

    # Order matters
    reversed_commit = H(TAGS["PROOF_ID"], part3, part2, part1)
    print(f"Reversed:   {reversed_commit.hex()[:16]}...")

    assert multi_commit != reversed_commit
    print("✓ Order matters in multi-part commitments")

    # Length prefixing prevents ambiguity
    concat_commit = H(TAGS["INT"], b"abc", b"def")
    single_commit = H(TAGS["INT"], b"abcdef")

    print(f"\nH('abc', 'def'): {concat_commit.hex()[:16]}...")
    print(f"H('abcdef'):     {single_commit.hex()[:16]}...")

    assert concat_commit != single_commit
    print("✓ Length prefixing prevents concatenation ambiguity")

    # Test hexH convenience
    hex_result = hexH(TAGS["VK_AGG"], b"verification", b"key")
    assert isinstance(hex_result, str)
    assert len(hex_result) == 64  # SHA-256 produces 32 bytes = 64 hex chars
    print(f"\nhexH result: {hex_result[:16]}...")
    print("✓ hexH convenience function works")

    print("\n" + "=" * 50)
    print("All commitment tests passed!")


if __name__ == "__main__":
    test_domain_separation()
