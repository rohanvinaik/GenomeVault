from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ConsentGrantRequest(BaseModel):
    subject_id: str = Field(...)
    scope: str = Field(...)
    ttl_days: int | None = Field(None, ge=1)


class ConsentRevokeRequest(BaseModel):
    subject_id: str = Field(...)
    scope: str = Field(...)


class ConsentCheckResponse(BaseModel):
    subject_id: str
    scope: str
    active: bool


class DSARExportRequest(BaseModel):
    subject_id: str = Field(...)


class DSAREraseRequest(BaseModel):
    subject_id: str = Field(...)


class DSARExportResponse(BaseModel):
    subject_id: str
    redacted: bool = True
    data: list[dict[str, Any]] = []
