from __future__ import annotations

"""Servers module."""
import numpy as np


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


class PIRServer:
    """Information-theoretic PIR server holding a replicated DB of equal-length byte records."""

    def __init__(self, db: list[bytes]):
        """Initialize instance.

        Args:
            db: Db.

        Raises:
            ValueError: When operation fails.
        """
        if not db:
            raise ValueError("db must be non-empty")
        L = len(db[0])
        if any(len(x) != L for x in db):
            raise ValueError("all records must be the same length")
        self.db = list(db)
        self.record_len = L

    def answer(self, mask: np.ndarray) -> bytes:
        """Return XOR of records where mask[k] == 1."""
        if mask.ndim != 1 or mask.dtype != np.uint8:
            raise ValueError("mask must be 1-D uint8 array")
        res = bytes([0] * self.record_len)
        for k, bit in enumerate(mask):
            if bit & 1:
                res = _xor_bytes(res, self.db[k])
        return res
