"""
Pydantic models for Hypervector encoding API endpoints.
"""

from typing import Any, Dict, List, Optional
from enum import Enum
import re

from pydantic import BaseModel, Field, validator, root_validator


# Strict validation models with deterministic ordering


class VariantInput(BaseModel):
    """Validates genomic variant in 'chr:pos ref>alt' format."""

    variant: str = Field(
        ...,
        description="Variant in format 'chr:pos ref>alt' (e.g., 'chr1:123456 A>G')",
        regex=r"^chr([1-9]|1[0-9]|2[0-2]|X|Y|M|MT):(\d+)\s+([ACGT]+)>([ACGT]+)$",
        min_length=10,
        max_length=100,
    )

    @validator("variant")
    def validate_variant_format(cls, v: str) -> str:
        """Validate and normalize variant format."""
        # Normalize chromosome naming (ensure 'chr' prefix)
        if not v.startswith("chr"):
            v = "chr" + v

        # Validate format with regex
        pattern = r"^chr([1-9]|1[0-9]|2[0-2]|X|Y|M|MT):(\d+)\s+([ACGT]+)>([ACGT]+)$"
        if not re.match(pattern, v, re.IGNORECASE):
            raise ValueError(
                f"Invalid variant format: '{v}'. Expected 'chr:pos ref>alt' "
                f"(e.g., 'chr1:123456 A>G')"
            )

        # Convert to uppercase for consistency
        parts = v.split()
        if len(parts) == 2:
            chr_pos = parts[0]
            alleles = parts[1].upper()
            v = f"{chr_pos} {alleles}"

        return v

    class Config:
        """Config implementation."""

        schema_extra = {"example": {"variant": "chr1:123456 A>G"}}


class VectorInput(BaseModel):
    """Validates numeric arrays for vector operations."""

    vector: List[float] = Field(
        ..., description="Numeric vector/array", min_items=100, max_items=100000
    )

    @validator("vector")
    def validate_vector(cls, v: List[float]) -> List[float]:
        """Validate vector contains only finite numbers."""
        if not v:
            raise ValueError("Vector cannot be empty")

        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Vector element at index {i} must be numeric, got {type(val)}")
            if not (-1e10 < val < 1e10):  # Reasonable bounds
                raise ValueError(f"Vector element at index {i} is out of bounds: {val}")

        return v

    class Config:
        """Config implementation."""

        schema_extra = {"example": {"vector": [0.1, -0.2, 0.3, 0.4, -0.5]}}


class EncodingRequest(BaseModel):
    """
    Accepts either variants OR vector for encoding, plus dimension.
    Ensures deterministic ordering and clear validation.
    """

    variants: Optional[List[str]] = Field(
        None,
        description="List of variants in 'chr:pos ref>alt' format",
        min_items=1,
        max_items=10000,
    )
    vector: Optional[List[float]] = Field(
        None,
        description="Pre-computed vector to process",
        min_items=100,
        max_items=100000,
    )
    dimension: int = Field(
        default=10000, description="Target hypervector dimension", ge=1000, le=100000
    )

    @root_validator
    def validate_input_type(cls, values: dict) -> dict:
        """Ensure exactly one input type is provided."""
        variants = values.get("variants")
        vector = values.get("vector")

        if variants is not None and vector is not None:
            raise ValueError(
                "Cannot provide both 'variants' and 'vector'. "
                "Please provide exactly one input type."
            )

        if variants is None and vector is None:
            raise ValueError(
                "Must provide either 'variants' or 'vector' input. " "Neither was provided."
            )

        return values

    @validator("variants", each_item=True)
    def validate_each_variant(cls, v: str) -> str:
        """Validate each variant in the list."""
        if not v:
            raise ValueError("Empty variant string")

        # Normalize and validate format
        if not v.startswith("chr"):
            v = "chr" + v

        pattern = r"^chr([1-9]|1[0-9]|2[0-2]|X|Y|M|MT):(\d+)\s+([ACGT]+)>([ACGT]+)$"
        if not re.match(pattern, v, re.IGNORECASE):
            raise ValueError(
                f"Invalid variant format: '{v}'. Expected 'chr:pos ref>alt' "
                f"(e.g., 'chr1:123456 A>G')"
            )

        # Normalize to uppercase
        parts = v.split()
        if len(parts) == 2:
            chr_pos = parts[0]
            alleles = parts[1].upper()
            v = f"{chr_pos} {alleles}"

        return v

    @validator("variants")
    def ensure_deterministic_order(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Sort variants for deterministic processing."""
        if v is None:
            return v

        # Sort by chromosome, position, ref, alt for deterministic ordering
        def sort_key(variant: str) -> tuple:
            match = re.match(
                r"^chr([1-9]|1[0-9]|2[0-2]|X|Y|M|MT):(\d+)\s+([ACGT]+)>([ACGT]+)$",
                variant,
                re.IGNORECASE,
            )
            if match:
                chr_str, pos, ref, alt = match.groups()
                # Convert chromosome to sortable format
                if chr_str.isdigit():
                    chr_num = int(chr_str)
                elif chr_str == "X":
                    chr_num = 23
                elif chr_str == "Y":
                    chr_num = 24
                elif chr_str in ["M", "MT"]:
                    chr_num = 25
                else:
                    chr_num = 99
                return (chr_num, int(pos), ref, alt)
            return (99, 0, "", "")

        return sorted(v, key=sort_key)

    @validator("vector")
    def validate_vector_values(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate vector contains only finite numbers."""
        if v is None:
            return v

        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"Vector element at index {i} must be numeric, got {type(val).__name__}"
                )
            if not (-1e10 < val < 1e10):
                raise ValueError(
                    f"Vector element at index {i} is out of bounds: {val}. "
                    f"Values must be between -1e10 and 1e10."
                )

        return v

    @validator("dimension")
    def validate_dimension(cls, v: int) -> int:
        """Ensure dimension is in allowed values."""
        allowed = [1000, 5000, 10000, 15000, 20000, 50000, 100000]
        if v not in allowed:
            raise ValueError(
                f"Dimension {v} not allowed. Must be one of: {', '.join(map(str, allowed))}"
            )
        return v

    class Config:
        """Config implementation."""

        schema_extra = {
            "examples": {
                "variants_input": {
                    "summary": "Encode genomic variants",
                    "value": {
                        "variants": [
                            "chr1:123456 A>G",
                            "chr2:789012 C>T",
                            "chrX:555555 GG>AA",
                        ],
                        "dimension": 10000,
                    },
                },
                "vector_input": {
                    "summary": "Process existing vector",
                    "value": {
                        "vector": [0.1, -0.2, 0.3, 0.4, -0.5],
                        "dimension": 10000,
                    },
                },
            }
        }


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


class SearchRequest(BaseModel):
    """Request model for searching hypervector index."""

    query_vector: List[float] = Field(
        ..., description="Query hypervector for search", min_items=100, max_items=100000
    )
    index_path: str = Field(default="demo_index/", description="Path to the index directory")
    k: int = Field(default=5, description="Number of nearest neighbors to return", ge=1, le=100)
    metric: str = Field(
        default="hamming",
        description="Distance metric to use",
        regex="^(hamming|cosine|euclidean)$",
    )

    @validator("query_vector")
    def validate_query_vector(cls, v: List[float]) -> List[float]:
        """Validate query vector contains finite numbers."""
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"Query vector element at index {i} must be numeric, got {type(val).__name__}"
                )
            if not (-1e10 < val < 1e10):
                raise ValueError(f"Query vector element at index {i} is out of bounds: {val}")
        return v

    @validator("index_path")
    def validate_index_path(cls, v: str) -> str:
        """Ensure index path is safe."""
        # Basic path validation to prevent directory traversal
        if ".." in v or v.startswith("/etc") or v.startswith("/root"):
            raise ValueError("Invalid index path")
        return v

    class Config:
        """Config implementation."""

        schema_extra = {
            "example": {
                "query_vector": [0.1, -0.2, 0.3, 0.4, -0.5],
                "index_path": "demo_index/",
                "k": 5,
                "metric": "hamming",
            }
        }


class SearchResult(BaseModel):
    """Single search result."""

    id: str = Field(..., description="Identifier of the matched vector")
    distance: float = Field(..., description="Distance from query vector", ge=0.0)

    class Config:
        """Config implementation."""

        schema_extra = {"example": {"id": "sample_001", "distance": 0.125}}


class SearchResponse(BaseModel):
    """Response model for hypervector search."""

    results: List[SearchResult] = Field(
        ..., description="List of search results ordered by distance"
    )
    query_dimension: int = Field(..., description="Dimension of the query vector")
    index_size: int = Field(..., description="Total number of vectors in the index")
    metric: str = Field(..., description="Distance metric used for search")
    search_time_ms: float = Field(..., description="Time taken for search in milliseconds")

    class Config:
        """Config implementation."""

        schema_extra = {
            "example": {
                "results": [
                    {"id": "sample_001", "distance": 0.0},
                    {"id": "sample_005", "distance": 0.125},
                    {"id": "sample_003", "distance": 0.250},
                ],
                "query_dimension": 10000,
                "index_size": 1000,
                "metric": "hamming",
                "search_time_ms": 12.5,
            }
        }
