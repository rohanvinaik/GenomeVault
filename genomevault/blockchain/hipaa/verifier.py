from __future__ import annotations
from typing import Dict, Any

REQUIRED_FIELDS = {"patient_id", "purpose", "consent_hash"}


def verify_access(request: Dict[str, Any]) -> bool:
    # Minimal policy: all required fields present and non-empty
    if not isinstance(request, dict):
        return False
    for f in REQUIRED_FIELDS:
        if not request.get(f):
            return False
    # Example denylist rule
    if request.get("purpose") == "sell_data":
        return False
    return True
