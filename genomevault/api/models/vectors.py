"""Vectors module."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Dict

from pydantic import BaseModel, Field, validator
from genomevault.api.utils import dict_for_update as _dict_for_update


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
    vector_ids: list[str] = Field(..., description="List of vector IDs to operate on")
    parameters: dict[str, Any] | None = None


class VectorOperationPatch(BaseModel):
    """PATCH model for updating vector operations - all fields optional."""

    operation: Optional[str] = Field(None, description="bundle, bind, permute, multiply, convolve")
    vector_ids: Optional[list[str]] = Field(None, description="List of vector IDs to operate on")
    parameters: Optional[dict[str, Any]] = None

    def dict_for_update(self) -> Dict[str, Any]:
        """Return only set fields for update."""
        return _dict_for_update(self)


class VectorEncodeConfigPatch(BaseModel):
    """PATCH model for updating encoding configuration."""

    dimension: Optional[VectorDimension] = Field(None, description="Target dimension")
    compression_tier: Optional[str] = Field(None, description="mini, clinical, or full")

    def dict_for_update(self) -> Dict[str, Any]:
        """Return only set fields for database update."""
        return self.dict(exclude_unset=True, exclude_none=True)
