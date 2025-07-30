from __future__ import annotations

import os
from typing import List
from fastapi import Header, HTTPException, status, Depends

_HEADER = "X-API-Key"


def _load_keys() -> List[str]:
    raw = os.getenv("GV_API_KEYS", "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def require_api_key(x_api_key: str | None = Header(default=None, alias=_HEADER)) -> str:
    keys = _load_keys()
    if not keys:
        # Security off: allow (development mode). In prod, set GV_API_KEYS!
        return "ANON"
    if x_api_key is None or x_api_key not in keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API key")
    return x_api_key
