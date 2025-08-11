from __future__ import annotations

"""Vectors module."""
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class VectorDimension(str, Enum):
    """VectorDimension implementation."""

    D10K = "10000"
    D15K = "15000"
    D20K = "20000"


class VectorEncodeRequest(BaseModel):
    """VectorEncodeRequest implementation."""

    data: dict[str, list[float]] = Field(..., description="Multi-modal input data")
    dimension: VectorDimension = Field(VectorDimension.D10K, description="Target dimension")
    compression_tier: str = Field("full", description="mini, clinical, or full")

    @validator("data")
    def _validate_data(cls, v: dict[str, list[float]]):
        if not isinstance(v, dict) or not v:
            raise ValueError("data must be a non-empty dict[str, list[float]]")
        return v


class VectorEncodeResponse(BaseModel):
    """VectorEncodeResponse implementation."""

    vector_id: str
    dimension: int
    sparsity: float
    compression_tier: str


class VectorOperationRequest(BaseModel):
    """VectorOperationRequest implementation."""

    operation: str = Field(..., description="bundle, bind, permute, multiply, convolve")
    vector_ids: list[str]
    parameters: dict[str, Any] | None = None
