from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Any, Sequence, NamedTuple

from .commit import TAGS, H
from .serialization import be_int


class PathItem(NamedTuple):
    """Single step in a Merkle proof.

    Attributes:
        sibling: Hash of the sibling node.
        is_right: ``True`` if ``sibling`` is on the right-hand side of the
            current node, ``False`` if it is on the left.
    """

    sibling: bytes
    is_right: bool


def leaf_bytes(vals: Iterable[int]) -> bytes:
    """Hash a sequence of integers into a Merkle leaf node.

    Each value is encoded as a 32-byte big-endian integer and concatenated
    before being hashed with the ``LEAF`` domain-separation tag.
    """

    data = b"".join(be_int(v, 32) for v in vals)
    return H(TAGS["LEAF"], data)


def node_bytes(left: bytes, right: bytes) -> bytes:
    """Hash two child nodes to derive their parent node."""

    return H(TAGS["NODE"], left, right)


def build(leaves: Sequence[bytes]) -> Dict[str, Any]:
    """Build a Merkle tree and return its root and all layers.

    The returned dictionary contains the Merkle ``root`` and ``layers`` where
    ``layers[0]`` is the original list of ``leaves``.  If a layer has an odd
    number of nodes, the last node is duplicated (Bitcoin-style padding).
    """

    if not leaves:
        raise ValueError("Empty leaf set")

    layers: List[List[bytes]] = [list(leaves)]
    cur = layers[0]
    while len(cur) > 1:
        nxt: List[bytes] = []
        if len(cur) & 1:
            cur = cur + [cur[-1]]
        for i in range(0, len(cur), 2):
            nxt.append(node_bytes(cur[i], cur[i + 1]))
        layers.append(nxt)
        cur = nxt
    return {"root": cur[0], "layers": layers}


def path(tree: Dict[str, Any], index: int) -> List[PathItem]:
    """Return the Merkle proof for a leaf at ``index``.

    The proof is returned as a list of :class:`PathItem` objects describing the
    sibling hash at each layer and whether that sibling is to the right of the
    current node.
    """

    layers = tree["layers"]
    idx = index
    out: List[PathItem] = []
    for lvl in range(len(layers) - 1):
        layer = layers[lvl]
        if idx % 2 == 0:
            sib_idx = idx + 1
            sib_right = True
        else:
            sib_idx = idx - 1
            sib_right = False
        if sib_idx >= len(layer):
            sib_idx = idx
        out.append(PathItem(layer[sib_idx], sib_right))
        idx //= 2
    return out


def verify(
    leaf_data_vals: Iterable[int], path_items: List[PathItem], root: bytes
) -> bool:
    """Verify a Merkle proof for ``leaf_data_vals`` against ``root``.

    ``path_items`` should be the list returned by :func:`path` for the leaf in
    question.  The function recomputes the hashes up the tree and checks whether
    the resulting value equals ``root``.
    """

    cur = leaf_bytes(list(leaf_data_vals))
    for sib, sib_is_right in path_items:
        if sib_is_right:
            cur = node_bytes(cur, sib)
        else:
            cur = node_bytes(sib, cur)
    return cur == root
