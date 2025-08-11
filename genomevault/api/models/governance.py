from __future__ import annotations

"""Governance module."""
from typing import Any

from pydantic import BaseModel, Field


class ConsentGrantRequest(BaseModel):
    """ConsentGrantRequest implementation."""

    subject_id: str = Field(...)
    scope: str = Field(...)
    ttl_days: int | None = Field(None, ge=1)


class ConsentRevokeRequest(BaseModel):
    """ConsentRevokeRequest implementation."""

    subject_id: str = Field(...)
    scope: str = Field(...)


class ConsentCheckResponse(BaseModel):
    """ConsentCheckResponse implementation."""

    subject_id: str
    scope: str
    active: bool


class DSARExportRequest(BaseModel):
    """DSARExportRequest implementation."""

    subject_id: str = Field(...)


class DSAREraseRequest(BaseModel):
    """DSAREraseRequest implementation."""

    subject_id: str = Field(...)


class DSARExportResponse(BaseModel):
    """DSARExportResponse implementation."""

    subject_id: str
    redacted: bool = True
    data: list[dict[str, Any]] = []
