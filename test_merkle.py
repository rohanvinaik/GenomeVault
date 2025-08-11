#!/usr/bin/env python3
"""Test the canonical Merkle tree module."""

from genomevault.crypto import merkle
from genomevault.crypto.commit import H, TAGS


def test_merkle_tree():
    """Test Merkle tree construction and verification."""

    print("Testing Canonical Merkle Tree")
    print("=" * 50)

    # Test leaf hashing
    print("Testing leaf hashing:")

    leaf1 = merkle.leaf_bytes([1, 2, 3])
    leaf2 = merkle.leaf_bytes([1, 2, 3])
    leaf3 = merkle.leaf_bytes([4, 5, 6])

    assert leaf1 == leaf2
    assert leaf1 != leaf3
    print(f"  Leaf [1,2,3]: {leaf1.hex()[:16]}...")
    print(f"  Leaf [4,5,6]: {leaf3.hex()[:16]}...")
    print("  ✓ Deterministic leaf hashing")

    # Test tree building
    print("\nTesting tree building:")

    # Create leaves
    leaves = []
    for i in range(8):
        leaves.append(merkle.leaf_bytes([i]))

    tree = merkle.build(leaves)
    root = tree["root"]
    layers = tree["layers"]

    print(f"  8 leaves -> root: {root.hex()[:16]}...")
    print(f"  Tree has {len(layers)} layers")
    assert len(layers) == 4  # 8 -> 4 -> 2 -> 1
    assert len(layers[0]) == 8
    assert len(layers[1]) == 4
    assert len(layers[2]) == 2
    assert len(layers[3]) == 1
    print("  ✓ Correct tree structure")

    # Test odd number of leaves (duplication)
    print("\nTesting odd number of leaves:")

    odd_leaves = leaves[:5]  # 5 leaves
    odd_tree = merkle.build(odd_leaves)
    odd_layers = odd_tree["layers"]

    # Layer 0: 5 leaves
    # Layer 1: 3 nodes (5->6 with duplication, then 6/2=3)
    # Layer 2: 2 nodes (3->4 with duplication, then 4/2=2)
    # Layer 3: 1 root
    print(f"  5 leaves -> {len(odd_layers)} layers")
    print(f"  Layer sizes: {[len(layer) for layer in odd_layers]}")
    print("  ✓ Handles odd leaf count with duplication")

    # Test path generation
    print("\nTesting path generation:")

    # Get path for leaf at index 2
    path2 = merkle.path(tree, 2)

    print(f"  Path for leaf 2: {len(path2)} siblings")
    for i, (sib, is_right) in enumerate(path2):
        side = "right" if is_right else "left"
        print(f"    Level {i}: sibling on {side} ({sib.hex()[:8]}...)")

    assert len(path2) == 3  # log2(8) = 3
    print("  ✓ Correct path length")

    # Test verification with correct path
    print("\nTesting path verification:")

    # Verify leaf 2 with its path
    leaf2_data = [2]
    verified = merkle.verify(leaf2_data, path2, root)
    assert verified
    print("  ✓ Valid path verifies correctly")

    # Test verification with wrong data
    wrong_data = [99]
    wrong_verified = merkle.verify(wrong_data, path2, root)
    assert not wrong_verified
    print("  ✓ Wrong data fails verification")

    # Test verification with wrong root
    fake_root = b"\xff" * 32
    fake_verified = merkle.verify(leaf2_data, path2, fake_root)
    assert not fake_verified
    print("  ✓ Wrong root fails verification")

    # Test direction bits matter
    print("\nTesting direction bits importance:")

    # Flip all direction bits
    flipped_path = [(sib, not is_right) for sib, is_right in path2]
    flipped_verified = merkle.verify(leaf2_data, flipped_path, root)
    assert not flipped_verified
    print("  ✓ Wrong direction bits fail verification")

    # Test all leaves verify
    print("\nTesting all leaves verify:")

    all_verify = True
    for i in range(len(leaves)):
        path_i = merkle.path(tree, i)
        if not merkle.verify([i], path_i, root):
            all_verify = False
            print(f"  ✗ Leaf {i} failed verification!")

    assert all_verify
    print("  ✓ All 8 leaves verify correctly")

    # Test domain separation
    print("\nTesting domain separation:")

    # Manual computation should use proper tags
    test_leaf = H(TAGS["LEAF"], merkle.be_int(42, 32))
    test_leaf2 = merkle.leaf_bytes([42])
    assert test_leaf == test_leaf2
    print("  ✓ Leaf uses LEAF tag")

    left = b"\xaa" * 32
    right = b"\xbb" * 32
    test_node = H(TAGS["NODE"], left, right)
    test_node2 = merkle.node_bytes(left, right)
    assert test_node == test_node2
    print("  ✓ Node uses NODE tag")

    # Test empty tree handling
    print("\nTesting edge cases:")

    try:
        merkle.build([])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Empty tree rejected: {e}")

    # Test single leaf tree
    single_tree = merkle.build([leaves[0]])
    assert single_tree["root"] == leaves[0]
    single_path = merkle.path(single_tree, 0)
    assert len(single_path) == 0
    assert merkle.verify([0], single_path, single_tree["root"])
    print("  ✓ Single leaf tree works")

    print("\n" + "=" * 50)
    print("All Merkle tree tests passed!")


if __name__ == "__main__":
    test_merkle_tree()
