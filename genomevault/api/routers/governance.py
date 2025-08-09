from __future__ import annotations

from fastapi import APIRouter

from genomevault.governance.audit.events import list_events, record_event
from genomevault.governance.consent.store import ConsentStore
from genomevault.governance.pii.redact import redact_text

router = APIRouter(prefix="/governance", tags=["governance"])

_CONSENT = ConsentStore()


@router.post("/consent/grant")
def consent_grant(payload):
    rec = _CONSENT.grant(payload.subject_id, payload.scope, ttl_days=payload.ttl_days)
    record_event(
        "consent_granted",
        payload.subject_id,
        payload.scope,
        {"ttl_days": payload.ttl_days},
    )
    return {"ok": True, "granted_at": rec.granted_at.isoformat()}


@router.post("/consent/revoke")
def consent_revoke(payload):
    ok = _CONSENT.revoke(payload.subject_id, payload.scope)
    if ok:
        record_event("consent_revoked", payload.subject_id, payload.scope, {})
    return {"ok": ok}


@router.get("/consent/check")
def consent_check(subject_id: str, scope: str):
    active = _CONSENT.has_consent(subject_id, scope)
    return {"subject_id": subject_id, "scope": scope, "active": active}


@router.post("/dsar/export")
def dsar_export(payload):
    # In a real system, locate subject's records and redact PII as needed.
    # Here we return a minimal synthetic response for demonstration.
    sample = [{"note": redact_text(f"Contact {payload.subject_id}@example.com for updates")}]
    record_event("export", payload.subject_id, "dsar", {"items": len(sample)})
    return {"subject_id": payload.subject_id, "redacted": True, "data": sample}


@router.post("/dsar/erase")
def dsar_erase(payload):
    # In a real system, erase or anonymize subject's data per retention policy.
    record_event("erase", payload.subject_id, "dsar", {})
    return {"ok": True}


@router.get("/ropa")
def ropa():
    # Record of Processing Activities from audit events
    return {"events": list_events()}
