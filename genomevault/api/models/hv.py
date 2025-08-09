"""
Pydantic models for Hypervector encoding API endpoints.
"""

from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, validator


class EncodingVersion(str, Enum):
    """Supported encoding versions."""

    V1 = "v1"
    V2 = "v2"
    LATEST = "latest"


class CompressionTier(str, Enum):
    """Compression tier for hypervectors."""

    MINI = "mini"
    CLINICAL = "clinical"
    FULL = "full"


class HVEncodeRequest(BaseModel):
    """Request model for hypervector encoding."""

    data: str = Field(
        ...,
        description="Raw genomic data to encode (VCF, FASTA, or JSON format)",
        min_length=1,
        max_length=10_000_000,  # 10MB limit for direct encoding
    )
    version: EncodingVersion = Field(
        default=EncodingVersion.V1, description="Encoding version to use"
    )
    dimension: int = Field(
        default=10000, description="Target hypervector dimension", ge=1000, le=100000
    )
    compression_tier: CompressionTier = Field(
        default=CompressionTier.CLINICAL, description="Compression tier for the output"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata to include with encoding"
    )

    @validator("dimension")
    def validate_dimension(cls, v: int) -> int:
        """Ensure dimension is in allowed values."""
        allowed = [10000, 15000, 20000, 50000, 100000]
        if v not in allowed:
            raise ValueError(f"Dimension must be one of {allowed}")
        return v

    class Config:
        """Config implementation."""
        schema_extra = {
            "example": {
                "data": "chr1:123456 A>G\nchr2:789012 C>T",
                "version": "v1",
                "dimension": 10000,
                "compression_tier": "clinical",
                "metadata": {"sample_id": "patient_001", "source": "exome_sequencing"},
            }
        }


class HVEncodeResponse(BaseModel):
    """Response model for hypervector encoding."""

    vector: List[float] = Field(..., description="Encoded hypervector")
    dimension: int = Field(..., description="Actual dimension of the vector")
    version: str = Field(..., description="Encoding version used")
    compression_tier: str = Field(..., description="Compression tier applied")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Encoding metadata and statistics"
    )

    class Config:
        """Config implementation."""
        schema_extra = {
            "example": {
                "vector": [0.123, -0.456, 0.789],
                "dimension": 10000,
                "version": "v1",
                "compression_tier": "clinical",
                "metadata": {
                    "encoding_time_ms": 42.3,
                    "input_size_bytes": 1024,
                    "compression_ratio": 50.2,
                    "variants_encoded": 2,
                },
            }
        }


class HVBatchEncodeRequest(BaseModel):
    """Request model for batch hypervector encoding."""

    samples: List[Dict[str, Any]] = Field(
        ..., description="List of samples to encode", min_items=1, max_items=100
    )
    version: EncodingVersion = Field(
        default=EncodingVersion.V1, description="Encoding version to use"
    )
    dimension: int = Field(default=10000, description="Target hypervector dimension")
    parallel: bool = Field(default=True, description="Whether to process samples in parallel")


class HVBatchEncodeResponse(BaseModel):
    """Response model for batch hypervector encoding."""

    results: List[Dict[str, Any]] = Field(..., description="Encoded results for each sample")
    total_samples: int = Field(..., description="Total number of samples processed")
    successful: int = Field(..., description="Number of successfully encoded samples")
    failed: int = Field(..., description="Number of failed encodings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Batch processing metadata")


class HVSimilarityRequest(BaseModel):
    """Request model for computing hypervector similarity."""

    vector1: List[float] = Field(..., description="First hypervector")
    vector2: List[float] = Field(..., description="Second hypervector")
    metric: str = Field(
        default="cosine",
        description="Similarity metric to use",
        regex="^(cosine|euclidean|hamming)$",
    )


class HVSimilarityResponse(BaseModel):
    """Response model for hypervector similarity."""

    similarity: float = Field(..., description="Similarity score between vectors", ge=-1.0, le=1.0)
    metric: str = Field(..., description="Metric used for comparison")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional similarity metadata"
    )
