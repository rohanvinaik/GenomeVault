from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProofCreateRequest(BaseModel):
    circuit_type: str = Field(..., description="variant | prs | diabetes_risk (placeholder types)")
    inputs: dict[str, Any]


class ProofVerifyRequest(BaseModel):
    proof: str  # base64
    public_inputs: dict[str, Any]
