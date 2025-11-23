"""Recursive Aggregator module."""

from __future__ import annotations

from typing import Iterable

AGG_PREFIX = b"AGG("


def aggregate(proofs: Iterable[bytes]) -> bytes:
    """Aggregate.

    Args:
        proofs: Zero-knowledge proof.

    Returns:
        bytes instance.
    """
    parts = [p for p in proofs if isinstance(p, (bytes, bytearray))]
    return AGG_PREFIX + b",".join(parts) + b")"


def verify_aggregate(agg: bytes, expected_count: int | None = None) -> bool:
    """Verify aggregate.

    Args:
        agg: Agg.
        expected_count: Number of items.

    Returns:
        Boolean result.
    """
    if (
        not isinstance(agg, (bytes, bytearray))
        or not agg.startswith(AGG_PREFIX)
        or not agg.endswith(b")")
    ):
        return False
    inner = agg[len(AGG_PREFIX) : -1]
    items = [] if inner == b"" else inner.split(b",")
    return expected_count is None or len(items) == expected_count
