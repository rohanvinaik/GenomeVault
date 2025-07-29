from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator


class VectorDimension(str, Enum):
    D10K = "10000"
    D15K = "15000"
    D20K = "20000"


class VectorEncodeRequest(BaseModel):
    data: Dict[str, List[float]] = Field(..., description="Multi-modal input data")
    dimension: VectorDimension = Field(VectorDimension.D10K, description="Target dimension")
    compression_tier: str = Field("full", description="mini, clinical, or full")

    @validator("data")
    def _validate_data(cls, v: Dict[str, List[float]]):
        if not isinstance(v, dict) or not v:
            raise ValueError("data must be a non-empty dict[str, list[float]]")
        return v


class VectorEncodeResponse(BaseModel):
    vector_id: str
    dimension: int
    sparsity: float
    compression_tier: str


class VectorOperationRequest(BaseModel):
    operation: str = Field(..., description="bundle, bind, permute, multiply, convolve")
    vector_ids: List[str]
    parameters: Optional[Dict[str, Any]] = None
