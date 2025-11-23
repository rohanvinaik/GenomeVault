"""Governance and compliance implementations for access."""

from .guards import get_consent_store, require_consent

__all__ = [
    "get_consent_store",
    "require_consent",
]
