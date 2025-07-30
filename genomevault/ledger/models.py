from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class LedgerAppendRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Arbitrary JSON-serializable payload")


class LedgerEntryModel(BaseModel):
    index: int
    timestamp: float
    data: Dict[str, Any]
    prev_hash: str
    hash: str


class LedgerVerifyResponse(BaseModel):
    valid: bool


class LedgerEntriesResponse(BaseModel):
    entries: List[LedgerEntryModel]
