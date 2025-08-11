# genomevault/crypto/commit.py
from __future__ import annotations
import hashlib

# Centralized domain tags (bytes!) to avoid typos across the codebase
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
    # 4-byte BE length prefix — stable, portable
    return len(b).to_bytes(4, "big")


def H(tag: bytes, *parts: bytes) -> bytes:
    """
    Domain-separated SHA-256: H(tag || len(tag) || len(part_i)||part_i ...)
    Keep it simple & reproducible; if you later swap to a different hash,
    keep this API stable and change the implementation here.
    """
    h = hashlib.sha256()
    h.update(_len_prefix(tag))
    h.update(tag)
    for part in parts:
        h.update(_len_prefix(part))
        h.update(part)
    return h.digest()


def hexH(tag: bytes, *parts: bytes) -> str:
    return H(tag, *parts).hex()
