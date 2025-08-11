from __future__ import annotations

"""Redact module."""
import hashlib
import hmac
import json
import os
import threading
from collections.abc import Iterable
from pathlib import Path

from .patterns import detect, mask_value


def _secret() -> bytes:
    key = os.getenv("GV_PII_SECRET", "dev-secret").encode("utf-8")
    return key


def _hmac_token(kind: str, value: str) -> str:
    msg = f"{kind}:{value}".encode()
    digest = hmac.new(_secret(), msg, hashlib.sha256).hexdigest()[:16]
    return f"tok_{digest}"


class PseudonymStore:
    """Optional reversible mapping store for tokens <-> original values.

    WARNING: Storing originals on disk may be sensitive. Use only for dev/demo or
    place the store in encrypted storage with restricted access.
    """

    def __init__(self, path: str | None = None):
        """Initialize instance.

        Args:
            path: File or directory path.

        Raises:
            RuntimeError: When operation fails.
        """
        path = path or os.getenv("GV_PSEUDONYM_STORE", "")
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._map: dict[str, str] = {}
        if self.path and self.path.exists():
            try:
                self._map = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                logger.exception("Unhandled exception")
                self._map = {}
                raise RuntimeError("Unspecified error")

    def save(self) -> None:
        """Save."""
        if not self.path:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._map, indent=2), encoding="utf-8")

    def put(self, token: str, original: str) -> None:
        """Put.

        Args:
            token: Token.
            original: Original.
        """
        with self._lock:
            if token not in self._map:
                self._map[token] = original
                self.save()

    def get(self, token: str) -> str | None:
        """Get.

        Args:
            token: Token.

        Returns:
            Operation result.
        """
        return self._map.get(token)


def redact_text(text: str, kinds: Iterable[str] | None = None) -> str:
    """Replace PII with fixed masks like [EMAIL]."""
    matches = detect(text, kinds)
    if not matches:
        return text
    out = []
    i = 0
    for m in matches:
        s, e = m.span
        out.append(text[i:s])
        out.append(mask_value(m.kind))
        i = e
    out.append(text[i:])
    return "".join(out)


def tokenize_text(
    text: str, kinds: Iterable[str] | None = None, store: PseudonymStore | None = None
) -> str:
    """Replace PII with deterministic tokens tok_<hash> using HMAC-SHA256.
    If a PseudonymStore is provided, store a reversible mapping token->original.
    """
    matches = detect(text, kinds)
    if not matches:
        return text
    out = []
    i = 0
    for m in matches:
        s, e = m.span
        out.append(text[i:s])
        tok = _hmac_token(m.kind, m.value)
        if store:
            store.put(tok, m.value)
        out.append(tok)
        i = e
    out.append(text[i:])
    return "".join(out)
