#!/usr/bin/env python3
"""Test the accumulator binding primitive."""

from genomevault.zk_proofs.core.accumulator import INITIAL_ACC, step, verify_chain
from genomevault.crypto.commit import H, TAGS
import hashlib


def test_accumulator():
    """Test accumulator binding for proof aggregation."""

    print("Testing Accumulator Binding Primitive")
    print("=" * 50)

    # Test initial accumulator
    print("Testing initial accumulator:")
    print(f"  INITIAL_ACC: {INITIAL_ACC.hex()[:16]}...")

    # Verify it's deterministic
    initial2 = H(TAGS["ACC"], b"INITIAL_ACC")
    assert INITIAL_ACC == initial2
    print("  ✓ Initial accumulator is deterministic")

    # Test single step
    print("\nTesting accumulator step:")

    proof_commit1 = hashlib.sha256(b"proof1").digest()
    vk_hash1 = hashlib.sha256(b"vk1").digest()

    acc1 = step(INITIAL_ACC, proof_commit1, vk_hash1)
    print(f"  After step 1: {acc1.hex()[:16]}...")

    assert acc1 != INITIAL_ACC
    print("  ✓ Accumulator changes after step")

    # Test chaining
    print("\nTesting accumulator chaining:")

    proof_commit2 = hashlib.sha256(b"proof2").digest()
    vk_hash2 = hashlib.sha256(b"vk2").digest()

    acc2 = step(acc1, proof_commit2, vk_hash2)
    print(f"  After step 2: {acc2.hex()[:16]}...")

    assert acc2 != acc1
    assert acc2 != INITIAL_ACC
    print("  ✓ Accumulator evolves with each step")

    # Test order matters
    print("\nTesting order dependence:")

    # Reverse order
    acc1_rev = step(INITIAL_ACC, proof_commit2, vk_hash2)
    acc2_rev = step(acc1_rev, proof_commit1, vk_hash1)

    assert acc2 != acc2_rev
    print("  ✓ Order matters (non-commutative)")

    # Test binding property
    print("\nTesting binding properties:")

    # Different proof, same VK
    proof_commit3 = hashlib.sha256(b"proof3").digest()
    acc3a = step(INITIAL_ACC, proof_commit3, vk_hash1)
    acc3b = step(INITIAL_ACC, proof_commit1, vk_hash1)

    assert acc3a != acc3b
    print("  ✓ Different proofs produce different accumulators")

    # Same proof, different VK
    vk_hash3 = hashlib.sha256(b"vk3").digest()
    acc4a = step(INITIAL_ACC, proof_commit1, vk_hash3)
    acc4b = step(INITIAL_ACC, proof_commit1, vk_hash1)

    assert acc4a != acc4b
    print("  ✓ Different VKs produce different accumulators")

    # Test chain verification
    print("\nTesting chain verification:")

    # Build a chain
    chain = [
        {
            "proof_commit": proof_commit1.hex(),
            "vk_hash": vk_hash1.hex(),
            "proof_id": "proof1",
        },
        {
            "proof_commit": proof_commit2.hex(),
            "vk_hash": vk_hash2.hex(),
            "proof_id": "proof2",
        },
        {
            "proof_commit": proof_commit3.hex(),
            "vk_hash": vk_hash3.hex(),
            "proof_id": "proof3",
        },
    ]

    # Compute expected final accumulator
    acc_final = INITIAL_ACC
    acc_final = step(acc_final, proof_commit1, vk_hash1)
    acc_final = step(acc_final, proof_commit2, vk_hash2)
    acc_final = step(acc_final, proof_commit3, vk_hash3)

    # Verify with correct final
    assert verify_chain(chain, acc_final.hex())
    print("  ✓ Valid chain verifies")

    # Verify with wrong final
    wrong_final = hashlib.sha256(b"wrong").digest().hex()
    assert not verify_chain(chain, wrong_final)
    print("  ✓ Wrong final rejects")

    # Verify with None final
    assert not verify_chain(chain, None)
    print("  ✓ None final rejects")

    # Test empty chain
    empty_chain = []
    assert verify_chain(empty_chain, INITIAL_ACC.hex())
    print("  ✓ Empty chain verifies to initial")

    # Test tampered chain
    print("\nTesting tamper detection:")

    # Tamper with middle proof
    tampered_chain = [
        chain[0],
        {
            "proof_commit": hashlib.sha256(b"tampered").digest().hex(),
            "vk_hash": chain[1]["vk_hash"],
            "proof_id": "tampered",
        },
        chain[2],
    ]

    assert not verify_chain(tampered_chain, acc_final.hex())
    print("  ✓ Tampered proof detected")

    # Reorder chain
    reordered_chain = [chain[1], chain[0], chain[2]]
    assert not verify_chain(reordered_chain, acc_final.hex())
    print("  ✓ Reordered chain detected")

    # Test domain separation
    print("\nTesting domain separation:")

    # Manual computation should use ACC tag
    test_acc = H(TAGS["ACC"], INITIAL_ACC, proof_commit1, vk_hash1)
    test_acc2 = step(INITIAL_ACC, proof_commit1, vk_hash1)
    assert test_acc == test_acc2
    print("  ✓ Step uses ACC domain tag")

    # Different tag would produce different result
    wrong_tag_acc = H(TAGS["NODE"], INITIAL_ACC, proof_commit1, vk_hash1)
    assert wrong_tag_acc != test_acc
    print("  ✓ Domain separation prevents cross-protocol attacks")

    # Test large chain
    print("\nTesting large chain:")

    large_chain = []
    acc_large = INITIAL_ACC

    for i in range(100):
        pc = hashlib.sha256(f"proof_{i}".encode()).digest()
        vk = hashlib.sha256(f"vk_{i}".encode()).digest()

        large_chain.append(
            {
                "proof_commit": pc.hex(),
                "vk_hash": vk.hex(),
                "proof_id": f"proof_{i}",
            }
        )

        acc_large = step(acc_large, pc, vk)

    assert verify_chain(large_chain, acc_large.hex())
    print("  ✓ 100-proof chain verifies")

    # Tamper with one proof in large chain
    large_chain[50]["proof_commit"] = hashlib.sha256(b"tampered").digest().hex()
    assert not verify_chain(large_chain, acc_large.hex())
    print("  ✓ Single tampered proof in large chain detected")

    print("\n" + "=" * 50)
    print("All accumulator tests passed!")


if __name__ == "__main__":
    test_accumulator()
