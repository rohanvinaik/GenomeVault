from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Any

from .commit import TAGS, H
from .serialization import be_int

def leaf_bytes(vals: Iterable[int]) -> bytes:
    data = b"".join(be_int(v, 32) for v in vals)
    return H(TAGS["LEAF"], data)


def node_bytes(left: bytes, right: bytes) -> bytes:
    return H(TAGS["NODE"], left, right)


def build(leaves: List[bytes]) -> Dict[str, Any]:
    """
    Returns {"root": bytes, "layers": List[List[bytes]]} where layers[0] = leaves.
    Duplicates last node on odd layers (Bitcoin-style padding).
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


def path(tree: Dict[str, Any], index: int) -> List[Tuple[bytes, bool]]:
    """
    Returns list of (sibling_hash, sibling_is_right).
    """
    layers = tree["layers"]
    idx = index
    out: List[Tuple[bytes, bool]] = []
    for lvl in range(len(layers) - 1):
        layer = layers[lvl]
        if idx % 2 == 0:
            # left child
            sib_idx = idx + 1
            sib_right = True
        else:
            sib_idx = idx - 1
            sib_right = False
        if sib_idx >= len(layer):
            sib_idx = idx
        out.append((layer[sib_idx], sib_right))
        idx //= 2
    return out


def verify(
    leaf_data_vals: Iterable[int], path_items: List[Tuple[bytes, bool]], root: bytes
) -> bool:
    cur = leaf_bytes(list(leaf_data_vals))
    for sib, sib_is_right in path_items:
        if sib_is_right:
            cur = node_bytes(cur, sib)
        else:
            cur = node_bytes(sib, cur)
    return cur == root
