from __future__ import annotations

"""Models module."""
from typing import Any

from pydantic import BaseModel, Field, validator


class ModelUpdate(BaseModel):
    """ModelUpdate implementation."""

    client_id: str = Field(..., description="Unique identifier for the client")
    weights: list[float] = Field(..., description="Flattened model weights")
    num_examples: int = Field(
        ..., ge=1, description="Number of examples used to compute the update"
    )
    signature: str | None = Field(
        None, description="Optional signature (not verified in this scaffold)"
    )

    @validator("weights")
    def _non_empty(cls, v: list[float]):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("weights must be a non-empty list[float]")
        return v


class AggregateRequest(BaseModel):
    """AggregateRequest implementation."""

    updates: list[ModelUpdate] = Field(..., min_items=1)
    clip_norm: float | None = Field(
        None, ge=0.0, description="Optional L2 clip per-update before averaging"
    )


class AggregateResponse(BaseModel):
    """AggregateResponse implementation."""

    aggregated_weights: list[float]
    total_examples: int
    client_count: int
    details: dict[str, Any] = {}
