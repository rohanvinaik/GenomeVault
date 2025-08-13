import pytest

from genomevault.crypto.merkle import leaf_bytes, build, path, verify

def test_merkle_path_bitflip_fails():
    rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    leaves = [leaf_bytes(r) for r in rows]
    tree = build(leaves)
    for i in range(len(rows)):
        p = path(tree, i)
        assert verify(rows[i], p, tree["root"])
        # flip one bit in first sibling
        bad = list(p)
        h, is_right = bad[0]
        bad[0] = (bytes([h[0] ^ 1]) + h[1:], is_right)
        assert not verify(rows[i], bad, tree["root"])
