from __future__ import annotations

"""Verifier module."""
from typing import Any, Dict

REQUIRED_FIELDS = {"patient_id", "purpose", "consent_hash"}


def verify_access(request: Dict[str, Any]) -> bool:
    """Verify access.

    Args:
        request: Client request.

    Returns:
        Boolean result.
    """
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
