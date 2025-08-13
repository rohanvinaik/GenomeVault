from __future__ import annotations

from typing import Iterable, Mapping


def be_int(x: int, size: int = 32) -> bytes:
    """Be int.
    Args:        x: Parameter value.        size: Number or count value.
    Returns:
        bytes"""
    if x < 0:
        raise ValueError("Negative integers not supported in canonical encoding")
    return int(x).to_bytes(size, "big")


def bstr(s: str) -> bytes:
    """Bstr.
    Args:        s: List of items.
    Returns:
        bytes"""
    return s.encode("utf-8")


def varbytes(b: bytes) -> bytes:
    """Varbytes.
    Args:        b: Parameter value.
    Returns:
        bytes"""
    return len(b).to_bytes(4, "big") + b


def pack_bytes_seq(seq: Iterable[bytes]) -> bytes:
    """Pack bytes seq.
    Args:        seq: Parameter value.
    Returns:
        bytes"""
    out = len(tuple(seq)).to_bytes(4, "big")  # consume once
    # Note: to avoid double-iteration, callers should pass tuples/lists.
    for b in seq:
        out += varbytes(b)
    return out


def pack_str_list(items: Iterable[str]) -> bytes:
    """Pack str list.
    Args:        items: List of items.
    Returns:
        bytes"""
    items = tuple(items)
    return pack_bytes_seq(tuple(bstr(s) for s in items))


def pack_int_list(items: Iterable[int], limb: int = 32) -> bytes:
    """Pack int list.
    Args:        items: List of items.        limb: Parameter value.
    Returns:
        bytes"""
    items = tuple(items)
    return pack_bytes_seq(tuple(be_int(i, limb) for i in items))


def pack_kv_map(m: Mapping[str, bytes]) -> bytes:
    """Pack kv map.
    Args:        m: Parameter value.
    Returns:
        bytes"""
    # deterministic ordering by key
    items = sorted(m.items())
    return pack_bytes_seq(tuple(bstr(k) + varbytes(v) for k, v in items))


# Minimal, explicit TLV for "proof components"
def pack_proof_components(components: Mapping[str, bytes]) -> bytes:
    """
    DO NOT truncate. Callers may compress this for transport.
    """
    return pack_kv_map(components)
