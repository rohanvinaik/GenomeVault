from __future__ import annotations

from fastapi import Depends, HTTPException, Header, status
from genomevault.governance.consent.store import ConsentStore

_CONSENT = ConsentStore()  # simple singleton for app process


def get_consent_store() -> ConsentStore:
    return _CONSENT


def require_consent(scope: str):
    def dep(subject_id: str | None = Header(default=None, alias="X-Subject-ID"), store: ConsentStore = Depends(get_consent_store)):
        if subject_id is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing X-Subject-ID")
        if not store.has_consent(subject_id, scope):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"No consent for scope '{scope}'")
        return subject_id
    return dep
