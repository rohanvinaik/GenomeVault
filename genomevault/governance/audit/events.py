from __future__ import annotations

from typing import Any

from genomevault.ledger.store import InMemoryLedger

# global singleton (simple)
_LEDGER = InMemoryLedger()


def record_event(
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
    return [e.__dict__ for e in _LEDGER.entries() if e.data.get("type") == "data_event"]
