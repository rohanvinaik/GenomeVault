"""Models module."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any


class ProofCreateRequest(BaseModel):
    """Zero-knowledge proof createrequest component."""

    circuit_type: str = Field(..., description="variant | prs | diabetes_risk (placeholder types)")
    inputs: dict[str, Any]


class ProofVerifyRequest(BaseModel):
    """Zero-knowledge proof verifyrequest component."""

    proof: str  # base64
    public_inputs: dict[str, Any]
