"""Models module."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any
class LedgerAppendRequest(BaseModel):
    """LedgerAppendRequest implementation."""

    data: dict[str, Any] = Field(..., description="Arbitrary JSON-serializable payload")


class LedgerEntryModel(BaseModel):
    """LedgerEntryModel implementation."""

    index: int
    timestamp: float
    data: dict[str, Any]
    prev_hash: str
    hash: str


class LedgerVerifyResponse(BaseModel):
    """LedgerVerifyResponse implementation."""

    valid: bool


class LedgerEntriesResponse(BaseModel):
    """LedgerEntriesResponse implementation."""

    entries: list[LedgerEntryModel]
