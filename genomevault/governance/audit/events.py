"""Events module."""
from __future__ import annotations

from typing import Any

from genomevault.ledger.store import InMemoryLedger

_LEDGER = InMemoryLedger()


def record_event(
    """Record event.

        Args:
            kind: Kind.
            subject_id: Subject id.
            scope: Scope.
            details: Details.
        """
    kind: str, subject_id: str, scope: str, details: dict[str, Any] | None = None
) -> None:
    data = {
        "type": "data_event",
        "kind": kind,  # access/export/erase/consent_granted/consent_revoked
        "subject_id": subject_id,
        "scope": scope,
        "details": details or {},
    }
    _LEDGER.append(data)


def list_events() -> list[dict]:
    """List events.

        Returns:
            Operation result.
        """
    return [e.__dict__ for e in _LEDGER.entries() if e.data.get("type") == "data_event"]
