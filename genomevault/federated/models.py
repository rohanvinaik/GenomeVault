from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class ModelUpdate(BaseModel):
    client_id: str = Field(..., description="Unique identifier for the client")
    weights: List[float] = Field(..., description="Flattened model weights")
    num_examples: int = Field(..., ge=1, description="Number of examples used to compute the update")
    signature: Optional[str] = Field(None, description="Optional signature (not verified in this scaffold)")

    @validator("weights")
    def _non_empty(cls, v: List[float]):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("weights must be a non-empty list[float]")
        return v


class AggregateRequest(BaseModel):
    updates: List[ModelUpdate] = Field(..., min_items=1)
    clip_norm: Optional[float] = Field(None, ge=0.0, description="Optional L2 clip per-update before averaging")


class AggregateResponse(BaseModel):
    aggregated_weights: List[float]
    total_examples: int
    client_count: int
    details: Dict[str, Any] = {}
