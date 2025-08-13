from __future__ import annotations

import zlib

MAGIC = b"CPROOF\0"  # format tag
VERSION = (1).to_bytes(2, "big")


def compress_proof(raw: bytes) -> bytes:
    """Compress proof.
    Args:        raw: Parameter value.
    Returns:
        bytes"""
    comp = zlib.compress(raw, level=6)
    return MAGIC + VERSION + len(raw).to_bytes(8, "big") + comp


def decompress_proof(blob: bytes) -> bytes:
    """Decompress proof.
    Args:        blob: Parameter value.
    Returns:
        bytes"""
    if not blob.startswith(MAGIC):
        raise ValueError("Unknown proof payload format")
    version = int.from_bytes(blob[len(MAGIC) : len(MAGIC) + 2], "big")
    if version != 1:
        raise ValueError(f"Unsupported proof format version: {version}")
    size_off = len(MAGIC) + 2
    orig = int.from_bytes(blob[size_off : size_off + 8], "big")
    comp = blob[size_off + 8 :]
    out = zlib.decompress(comp)
    if len(out) != orig:
        raise ValueError("Decompressed size mismatch")
    return out
