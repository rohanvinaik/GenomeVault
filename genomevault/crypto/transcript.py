from __future__ import annotations

import hashlib

from .rng import xof


class Transcript:
    """
    Domain-separated running hash transcript with XOF challenges.
    """

    def __init__(self):
        self._h = hashlib.sha256()
        self._round = 0

    def append(self, label: str, message: bytes) -> None:
        lbl = label.encode("utf-8")
        self._h.update(len(lbl).to_bytes(4, "big"))
        self._h.update(lbl)
        self._h.update(len(message).to_bytes(4, "big"))
        self._h.update(message)

    def digest(self) -> bytes:
        return self._h.digest()

    def challenge(self, label: str, nbytes: int) -> bytes:
        self._round += 1
        seed = self.digest() + self._round.to_bytes(4, "big")
        return xof(label.encode("utf-8"), seed, nbytes)
