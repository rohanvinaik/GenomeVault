"""
Comprehensive tests for crypto refactoring.
Ensures all the mechanical refactoring items are properly implemented.
"""
import hashlib
import json
import pytest

from genomevault.crypto import (

    H,
    hexH,
    TAGS,
    be_int,
    bstr,
    pack_proof_components,
    compress_proof,
    decompress_proof,
    secure_bytes,
    xof,
    xof_uint_mod,
    Transcript,
    merkle,
)
from genomevault.zk_proofs.core.accumulator import INITIAL_ACC, step, verify_chain


def test_no_proof_truncation():
    """Ensure proofs are never truncated."""
    # Create a large proof with critical data at the end
    critical_marker = b"CRITICAL_END_DATA_MUST_NOT_BE_LOST"
    proof_data = b"X" * 10000 + critical_marker

    # Compress (not truncate!)
    compressed = compress_proof(proof_data)
    decompressed = decompress_proof(compressed)

    # Critical data must be preserved
    assert decompressed.endswith(critical_marker)
    assert len(decompressed) == len(proof_data)

    # Truncation would lose data
    truncated = proof_data[:256]
    assert not truncated.endswith(critical_marker)


def test_canonical_commitments():
    """Test that commitments use canonical serialization, not JSON."""
    # Bad: JSON is not canonical
    data = {"b": 2, "a": 1}  # Order matters in JSON
    json1 = json.dumps(data)
    json2 = json.dumps({"a": 1, "b": 2})
    assert json1 != json2  # JSON order is not stable

    # Good: Canonical serialization
    proof_components = {
        "commitment": b"\xaa" * 32,
        "challenge": b"\xbb" * 32,
    }

    # Use canonical serialization
    packed = pack_proof_components(proof_components)
    commitment = H(TAGS["PROOF_ID"], packed)

    # Should be deterministic
    packed2 = pack_proof_components(proof_components)
    commitment2 = H(TAGS["PROOF_ID"], packed2)
    assert commitment == commitment2


def test_secure_rng_usage():
    """Test that cryptographic operations use secure RNG."""
    # Secure randomness for crypto
    nonce = secure_bytes(32)
    assert len(nonce) == 32

    # XOF for deterministic derivation
    seed = b"test_seed"
    derived1 = xof(b"LABEL", seed, 32)
    derived2 = xof(b"LABEL", seed, 32)
    assert derived1 == derived2  # Deterministic

    # Different labels produce different outputs
    derived3 = xof(b"OTHER", seed, 32)
    assert derived1 != derived3


def test_merkle_with_direction_bits():
    """Test Merkle trees use proper direction bits."""
    # Create a tree
    leaves = []
    for i in range(4):
        leaves.append(merkle.leaf_bytes([i]))

    tree = merkle.build(leaves)

    # Get path with direction bits
    path = merkle.path(tree, 1)

    # Each path element has (sibling_hash, sibling_is_right)
    for sibling_hash, sibling_is_right in path:
        assert isinstance(sibling_hash, bytes)
        assert isinstance(sibling_is_right, bool)

    # Verify with correct path
    assert merkle.verify([1], path, tree["root"])

    # Flipping direction bits should fail
    flipped_path = [(sib, not is_right) for sib, is_right in path]
    assert not merkle.verify([1], flipped_path, tree["root"])


def test_accumulator_binding():
    """Test accumulator properly binds proofs."""
    # Build a chain
    chain = []
    acc = INITIAL_ACC

    for i in range(3):
        proof_commit = hashlib.sha256(f"proof_{i}".encode()).digest()
        vk_hash = hashlib.sha256(f"vk_{i}".encode()).digest()

        acc = step(acc, proof_commit, vk_hash)
        chain.append(
            {
                "proof_commit": proof_commit.hex(),
                "vk_hash": vk_hash.hex(),
                "proof_id": f"proof_{i}",
            }
        )

    # Valid chain verifies
    assert verify_chain(chain, acc.hex())

    # Dropping a link fails
    assert not verify_chain(chain[:-1], acc.hex())

    # Reordering fails
    reordered = [chain[1], chain[0], chain[2]]
    assert not verify_chain(reordered, acc.hex())


def test_transcript_determinism():
    """Test Fiat-Shamir transcript is deterministic."""
    # Create transcript
    t = Transcript()
    t.append("commitment", b"\xaa" * 32)
    t.append("public_input", be_int(42, 8))

    # Get challenge
    challenge1 = t.challenge("alpha", 32)

    # Create identical transcript
    t2 = Transcript()
    t2.append("commitment", b"\xaa" * 32)
    t2.append("public_input", be_int(42, 8))

    # Should get same challenge
    challenge2 = t2.challenge("alpha", 32)
    assert challenge1 == challenge2


def test_no_json_in_commitments():
    """Ensure JSON is not used for cryptographic commitments."""
    # This would be bad (non-canonical)
    bad_data = {"key": "value", "num": 123}
    bad_commit = hashlib.sha256(json.dumps(bad_data).encode()).digest()

    # This is good (canonical)
    good_data = {"key": bstr("value"), "num": be_int(123, 8)}
    good_packed = pack_proof_components(good_data)
    good_commit = H(TAGS["PROOF_ID"], good_packed)

    # They should be different (JSON vs canonical)
    assert bad_commit.hex() != good_commit.hex()

    # Canonical should be stable
    good_packed2 = pack_proof_components(good_data)
    good_commit2 = H(TAGS["PROOF_ID"], good_packed2)
    assert good_commit == good_commit2


def test_domain_separation():
    """Test that different operations use different domain tags."""
    data = b"test_data"

    # Different tags produce different outputs
    leaf_hash = H(TAGS["LEAF"], data)
    node_hash = H(TAGS["NODE"], data)
    acc_hash = H(TAGS["ACC"], data)
    proof_hash = H(TAGS["PROOF_ID"], data)

    # All should be different
    hashes = [leaf_hash, node_hash, acc_hash, proof_hash]
    assert len(set(hashes)) == len(hashes)


def test_xof_vs_random():
    """Test XOF for deterministic derivation vs random for secrets."""
    # XOF is deterministic (good for public values)
    seed = b"public_seed"
    xof1 = xof(b"DERIVE", seed, 32)
    xof2 = xof(b"DERIVE", seed, 32)
    assert xof1 == xof2

    # secure_bytes is random (good for secrets)
    rand1 = secure_bytes(32)
    rand2 = secure_bytes(32)
    assert rand1 != rand2  # Should be different with overwhelming probability


if __name__ == "__main__":
    # Run all tests
    test_no_proof_truncation()
    print("✓ No proof truncation")

    test_canonical_commitments()
    print("✓ Canonical commitments")

    test_secure_rng_usage()
    print("✓ Secure RNG usage")

    test_merkle_with_direction_bits()
    print("✓ Merkle with direction bits")

    test_accumulator_binding()
    print("✓ Accumulator binding")

    test_transcript_determinism()
    print("✓ Transcript determinism")

    test_no_json_in_commitments()
    print("✓ No JSON in commitments")

    test_domain_separation()
    print("✓ Domain separation")

    test_xof_vs_random()
    print("✓ XOF vs random")

    print("\nAll crypto refactor tests passed!")
