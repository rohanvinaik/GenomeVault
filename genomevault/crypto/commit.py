from __future__ import annotations

import hashlib

TAGS = {
    "LEAF": b"\x00LEAF",
    "NODE": b"\x01NODE",
    "VK_AGG": b"VK_AGG",
    "SUBPROOF": b"SUBPROOF",
    "ACC": b"ACC",
    "PROOF_ID": b"PROOF_ID",
    "TRACE": b"TRACE",
    "CONS": b"CONS",
    "INT": b"INT",  # integrity binding
    "EMPTY_VK": b"EMPTY_VK",
}


def _len_prefix(b: bytes) -> bytes:
    """Return a 4-byte big-endian length prefix for ``b``."""

    # 4-byte BE length prefix — stable, portable
    return len(b).to_bytes(4, "big")


def H(tag: bytes, *parts: bytes) -> bytes:
    """Domain-separated SHA-256 hash.

    The hash is computed as ``H(tag || len(tag) || len(part_i)||part_i ...)``.
    This keeps the API stable if the underlying hash function is swapped out in
    the future.
    """

    h = hashlib.sha256()
    h.update(_len_prefix(tag))
    h.update(tag)
    for part in parts:
        h.update(_len_prefix(part))
        h.update(part)
    return h.digest()


def hexH(tag: bytes, *parts: bytes) -> str:
    """Return the hexadecimal representation of :func:`H`."""

    return H(tag, *parts).hex()
