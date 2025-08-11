from __future__ import annotations

"""Engine module."""
from hashlib import sha256

import numpy as np

from genomevault.pir.servers import PIRServer


def _normalize_db(items: list[bytes]) -> list[bytes]:
    """Convert arbitrary byte-like items into fixed-length 32-byte records using SHA-256."""
    out: list[bytes] = []
    for it in items:
        if not isinstance(it, (bytes, bytearray)):
            it = bytes(it)
        out.append(sha256(it).digest())
    return out


class PIREngine:
    """Simple m-server IT-PIR engine (replicated DB, XOR aggregation)."""

    def __init__(self, db_items: list[bytes], n_servers: int = 3):
        """Initialize instance.

        Args:
            db_items: Db items.
            n_servers: N servers.

        Raises:
            ValueError: When operation fails.
        """
        if n_servers < 2:
            raise ValueError("n_servers must be >= 2")
        self.db = _normalize_db(db_items)
        self.n = len(self.db)
        self.m = int(n_servers)
        self.servers = [PIRServer(self.db) for _ in range(self.m)]

    def _random_masks(self, index: int) -> list[np.ndarray]:
        rng = np.random.default_rng()
        masks = []
        # Generate m-1 random masks of length n over GF(2)
        for _ in range(self.m - 1):
            masks.append(rng.integers(0, 2, size=self.n, dtype=np.uint8))
        # Final mask so that XOR of masks equals one-hot e_index
        e = np.zeros(self.n, dtype=np.uint8)
        e[index] = 1
        acc = np.zeros(self.n, dtype=np.uint8)
        for r in masks:
            acc ^= r
        last = acc ^ e
        masks.append(last)
        return masks

    def query(self, index: int) -> bytes:
        """Query.

        Args:
            index: Index position.

        Returns:
            bytes instance.

        Raises:
            IndexError: When operation fails.
        """
        if not (0 <= index < self.n):
            raise IndexError("index out of range")
        masks = self._random_masks(index)
        answers = [srv.answer(mask) for srv, mask in zip(self.servers, masks)]
        # XOR aggregate answers to recover the target record
        res = bytes([0] * len(self.db[0]))
        for a in answers:
            res = bytes(x ^ y for x, y in zip(res, a))
        return res
