from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from time import time
from typing import Any


def _hash_bytes(b: bytes) -> str:
    return sha256(b).hexdigest()


def _hash_obj(obj: dict[str, Any]) -> str:
    return _hash_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8"))


@dataclass(frozen=True)
class LedgerEntry:
    index: int
    timestamp: float
    data: dict[str, Any]
    prev_hash: str
    hash: str


class InMemoryLedger:
    """Minimal append-only ledger with hash chaining (NOT a consensus blockchain)."""

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    def _compute_hash(self, index: int, ts: float, data: dict[str, Any], prev_hash: str) -> str:
        payload = {
            "index": index,
            "timestamp": ts,
            "data": data,
            "prev_hash": prev_hash,
        }
        return _hash_obj(payload)

    def append(self, data: dict[str, Any]) -> LedgerEntry:
        idx = len(self._entries)
        ts = time()
        prev = self._entries[-1].hash if self._entries else "GENESIS"
        h = self._compute_hash(idx, ts, data, prev)
        entry = LedgerEntry(index=idx, timestamp=ts, data=data, prev_hash=prev, hash=h)
        self._entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        prev = "GENESIS"
        for i, e in enumerate(self._entries):
            if e.index != i:
                return False
            expected = self._compute_hash(e.index, e.timestamp, e.data, prev)
            if e.hash != expected or e.prev_hash != prev:
                return False
            prev = e.hash
        return True

    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)
