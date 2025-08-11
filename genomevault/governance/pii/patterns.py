from __future__ import annotations

"""Patterns module."""
import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Match:
    """Match implementation."""

    kind: str
    span: tuple[int, int]
    value: str


EMAIL = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")  # US only

# Order matters for overlapping matches (more specific first)
_PATTERNS: dict[str, re.Pattern] = {
    "email": EMAIL,
    "phone": PHONE,
    "ipv4": IPV4,
    "ssn": SSN,
}


def detect(text: str, kinds: Iterable[str] | None = None) -> list[Match]:
    """Return list of detected PII matches with kind, span, value.")"""
    if not text:
        return []
    kinds = list(kinds) if kinds else list(_PATTERNS.keys())
    found: list[Match] = []
    for k in kinds:
        pat = _PATTERNS.get(k)
        if not pat:
            continue
        for m in pat.finditer(text):
            found.append(Match(kind=k, span=m.span(), value=m.group(0)))
    # sort by start offset, then length desc
    found.sort(key=lambda x: (x.span[0], -(x.span[1] - x.span[0])))
    return found


def mask_value(kind: str) -> str:
    """Mask value.

    Args:
        kind: Kind.

    Returns:
        String result.
    """
    return {
        "email": "[EMAIL]",
        "phone": "[PHONE]",
        "ipv4": "[IPV4]",
        "ssn": "[SSN]",
    }.get(kind, "[PII]")
