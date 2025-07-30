from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, Any


class ProofCreateRequest(BaseModel):
    circuit_type: str = Field(..., description="variant | prs | diabetes_risk (placeholder types)")
    inputs: Dict[str, Any]


class ProofVerifyRequest(BaseModel):
    proof: str  # base64
    public_inputs: Dict[str, Any]
