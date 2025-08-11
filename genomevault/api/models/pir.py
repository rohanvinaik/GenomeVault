from __future__ import annotations

"""Pir module."""
from pydantic import BaseModel


class PIRQueryRequest(BaseModel):
    """Private information retrieval queryrequest component."""

    index: int


class PIRQueryResponse(BaseModel):
    """Private information retrieval queryresponse component."""

    index: int
    item_base64: str  # base64 of 32-byte record
