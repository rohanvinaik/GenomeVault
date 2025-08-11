from __future__ import annotations

"""Models module."""
from typing import Any

from pydantic import BaseModel, Field


class ProofCreateRequest(BaseModel):
    """Zero-knowledge proof createrequest component."""

    circuit_type: str = Field(..., description="variant | prs | diabetes_risk (placeholder types)")
    inputs: dict[str, Any]


class ProofVerifyRequest(BaseModel):
    """Zero-knowledge proof verifyrequest component."""

    proof: str  # base64
    public_inputs: dict[str, Any]
