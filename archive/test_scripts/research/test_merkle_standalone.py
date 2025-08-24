#!/usr/bin/env python3
"""
Standalone test to verify Merkle proof fix works correctly.
"""

import hashlib


def compute_merkle_root(leaf: bytes, siblings: list[bytes], directions: list[int]) -> bytes:
    """
    Compute Merkle root from leaf and proof path.

    Args:
        leaf: Leaf hash
        siblings: Sibling hashes along the path
        directions: Direction bits (0 = current on left, 1 = current on right)

    Returns:
        Computed root hash
    """
    current = leaf

    for sibling, direction in zip(siblings, directions):
        if direction == 0:
            # Current node is on left
            current = hashlib.sha256(current + sibling).digest()
        else:
            # Current node is on right
            current = hashlib.sha256(sibling + current).digest()

    return current


def test_merkle_verification():
    """Test the Merkle verification logic."""

    print("=" * 50)

    # Create a simple 4-leaf tree
    leaves = [
        hashlib.sha256(b"leaf0").digest(),
        hashlib.sha256(b"leaf1").digest(),
        hashlib.sha256(b"leaf2").digest(),
        hashlib.sha256(b"leaf3").digest(),
    ]

    # Build tree manually
    # Level 0: leaves
    # Level 1: [hash(leaf0||leaf1), hash(leaf2||leaf3)]
    level1_left = hashlib.sha256(leaves[0] + leaves[1]).digest()
    level1_right = hashlib.sha256(leaves[2] + leaves[3]).digest()

    # Level 2: root = hash(level1_left||level1_right)
    root = hashlib.sha256(level1_left + level1_right).digest()

    print("Tree structure:")
    print(f"  Root: {root.hex()[:16]}...")
    print(f"  Level 1: [{level1_left.hex()[:8]}..., {level1_right.hex()[:8]}...]")
    print(
        f"  Leaves: [{leaves[0].hex()[:8]}..., {leaves[1].hex()[:8]}...,
            {leaves[2].hex()[:8]}..., {leaves[3].hex()[:8]}...]"
    )

    # Test 1: Proof for leaf 0
    siblings_0 = [leaves[1], level1_right]  # sibling at each level
    directions_0 = [0, 0]  # leaf 0 is on left at both levels

    computed_root_0 = compute_merkle_root(leaves[0], siblings_0, directions_0)
    print(f"  Computed root: {computed_root_0.hex()[:16]}...")
    print(f"  Expected root: {root.hex()[:16]}...")
    print(
        f"  Match: {computed_root_0 == root} ✓"
        if computed_root_0 == root
        else f"  Match: {computed_root_0 == root} ✗"
    )

    # Test 2: Proof for leaf 2
    siblings_2 = [leaves[3], level1_left]  # sibling at each level
    directions_2 = [
        0,
        1,
    ]  # leaf 2 is on left at level 0, but its parent is on right at level 1

    computed_root_2 = compute_merkle_root(leaves[2], siblings_2, directions_2)
    print(f"  Computed root: {computed_root_2.hex()[:16]}...")
    print(f"  Expected root: {root.hex()[:16]}...")
    print(
        f"  Match: {computed_root_2 == root} ✓"
        if computed_root_2 == root
        else f"  Match: {computed_root_2 == root} ✗"
    )

    # Test 3: Wrong direction bits (using parity)
    wrong_directions_0 = [0, 1]  # This is WRONG for leaf 0 (should be [0, 0])

    wrong_root_0 = compute_merkle_root(leaves[0], siblings_0, wrong_directions_0)
    print("  Leaf 0 with wrong directions [0,1]:")
    print(f"    Computed: {wrong_root_0.hex()[:16]}...")
    print(f"    Expected: {root.hex()[:16]}...")
    print(f"    Match: {wrong_root_0 == root} - This should be False!")

    # Test 4: Show why parity doesn't work
    print("  For leaf at index 0:")
    print("    - Level 0: index 0, parity = 0 (correct: on left)")
    print("    - Level 1: index 0, parity = 0 (correct: on left)")
    print("  For leaf at index 2:")
    print("    - Level 0: index 2, parity = 0 (correct: on left)")
    print("    - Level 1: index 1, parity = 1 (correct: on right)")
    print("  For leaf at index 1:")
    print("    - Level 0: index 1, parity = 1 (correct: on right)")
    print("    - Level 1: index 0, parity = 0 (correct: on left)")
    pass  # Debug print removed

    print("Conclusion: Direction bits are REQUIRED for correct Merkle proofs!")


if __name__ == "__main__":
    test_merkle_verification()
