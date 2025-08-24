"""
Vector operation models for GenomeVault API Gateway.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, validator

from genomevault.api.gateway.models.base import BaseModel, GenomicVariant, PrivacyLevel


class VectorType(str, Enum):
    """Hypervector types."""
    
    BINARY = "binary"
    BIPOLAR = "bipolar"
    REAL_VALUED = "real_valued"
    SPARSE = "sparse"


class EncodingType(str, Enum):
    """Encoding algorithm types."""
    
    STANDARD = "standard"
    PACKED = "packed"
    ORTHOGONAL_PROJECTION = "orthogonal_projection"
    SPARSE_PROJECTION = "sparse_projection"
    UNIFIED = "unified"


class SimilarityMetric(str, Enum):
    """Vector similarity metrics."""
    
    HAMMING = "hamming"
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    JACCARD = "jaccard"
    DOT_PRODUCT = "dot_product"


class VectorEncodeRequest(BaseModel):
    """Request model for vector encoding."""
    
    # Input data (mutually exclusive)
    numeric: Optional[List[float]] = Field(
        None,
        description="Numeric feature array (alternative to variants)",
        example=[0.1, 0.8, 0.3, 0.9, 0.2]
    )
    variants: Optional[List[GenomicVariant]] = Field(
        None,
        description="Genomic variants to encode (alternative to numeric)"
    )
    
    # Encoding parameters
    dimension: int = Field(
        8192,
        ge=1024,
        le=100000,
        description="Hypervector dimension"
    )
    encoding_type: EncodingType = Field(
        EncodingType.STANDARD,
        description="Encoding algorithm to use"
    )
    vector_type: VectorType = Field(
        VectorType.BINARY,
        description="Output vector type"
    )
    
    # Privacy parameters
    privacy_level: Optional[PrivacyLevel] = Field(
        None,
        description="Desired privacy level"
    )
    noise_level: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Differential privacy noise level"
    )
    
    # Additional parameters
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional encoding metadata")
    
    @field_validator("variants", "numeric")
    def validate_input_data(cls, v, values, field):
        """Ensure exactly one input type is provided."""
        if field.name == "variants" and v is not None:
            if values.get("numeric") is not None:
                raise ValueError("Cannot specify both 'variants' and 'numeric'")
        elif field.name == "numeric" and v is not None:
            if values.get("variants") is not None:
                raise ValueError("Cannot specify both 'variants' and 'numeric'")
        return v
    
    @field_validator("numeric")
    def validate_numeric_input(cls, v):
        """Validate numeric input array."""
        if v is not None and len(v) == 0:
            raise ValueError("Numeric array cannot be empty")
        return v
    
    @field_validator("variants")
    def validate_variants_input(cls, v):
        """Validate variants input."""
        if v is not None and len(v) == 0:
            raise ValueError("Variants array cannot be empty")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": {
                "numeric_encoding": {
                    "summary": "Encode numeric features",
                    "value": {
                        "numeric": [0.1, 0.8, 0.3, 0.9, 0.2],
                        "dimension": 4096,
                        "encoding_type": "standard",
                        "vector_type": "binary"
                    }
                },
                "variant_encoding": {
                    "summary": "Encode genomic variants",
                    "value": {
                        "variants": [
                            {
                                "chrom": "1",
                                "pos": 1234567,
                                "ref": "A",
                                "alt": "T",
                                "impact": "missense"
                            }
                        ],
                        "dimension": 8192,
                        "encoding_type": "unified",
                        "vector_type": "bipolar"
                    }
                }
            }
        }


class VectorEncodeResponse(BaseModel):
    """Response model for vector encoding."""
    
    vector_id: str = Field(..., description="Unique vector identifier")
    dimension: int = Field(..., description="Hypervector dimension")
    vector_type: VectorType = Field(..., description="Vector type")
    encoding_type: EncodingType = Field(..., description="Encoding algorithm used")
    
    # Vector data
    vector: List[Union[int, float]] = Field(..., description="Encoded hypervector")
    
    # Privacy guarantees
    privacy_level: PrivacyLevel = Field(..., description="Achieved privacy level")
    compression_ratio: Optional[float] = Field(None, description="Data compression ratio achieved")
    
    # Metadata
    encoding_time_ms: float = Field(..., description="Encoding time in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "vector_id": "vec_1234567890abcdef",
                "dimension": 8192,
                "vector_type": "binary",
                "encoding_type": "standard",
                "vector": [1, -1, 1, 1, -1],  # Truncated for example
                "privacy_level": "k-anonymous",
                "compression_ratio": 87.3,
                "encoding_time_ms": 125.7
            }
        }


class VectorCompareRequest(BaseModel):
    """Request model for vector comparison."""
    
    vector1_id: Optional[str] = Field(None, description="First vector ID (if stored)")
    vector2_id: Optional[str] = Field(None, description="Second vector ID (if stored)")
    
    vector1: Optional[List[Union[int, float]]] = Field(None, description="First vector data")
    vector2: Optional[List[Union[int, float]]] = Field(None, description="Second vector data")
    
    metrics: List[SimilarityMetric] = Field(
        [SimilarityMetric.HAMMING],
        description="Similarity metrics to compute"
    )
    
    normalize: bool = Field(False, description="Whether to normalize similarity scores")
    
    @field_validator("vector1", "vector2")
    def validate_vector_or_id(cls, v, values, field):
        """Ensure either vector ID or data is provided."""
        id_field = f"{field.name}_id"
        if v is None and values.get(id_field) is None:
            raise ValueError(f"Must provide either {field.name} or {id_field}")
        if v is not None and values.get(id_field) is not None:
            raise ValueError(f"Cannot specify both {field.name} and {id_field}")
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "vector1_id": "vec_1234567890abcdef",
                "vector2": [1, -1, 1, 1, -1],
                "metrics": ["hamming", "cosine"],
                "normalize": True
            }
        }


class VectorCompareResponse(BaseModel):
    """Response model for vector comparison."""
    
    similarity_scores: Dict[str, float] = Field(..., description="Similarity scores by metric")
    comparison_time_ms: float = Field(..., description="Comparison time in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional comparison metadata")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "similarity_scores": {
                    "hamming": 0.73,
                    "cosine": 0.85
                },
                "comparison_time_ms": 15.2
            }
        }


class VectorSearchRequest(BaseModel):
    """Request model for vector similarity search."""
    
    query_vector: List[Union[int, float]] = Field(..., description="Query vector")
    search_space: Optional[str] = Field(None, description="Search space identifier")
    top_k: int = Field(10, ge=1, le=1000, description="Number of top results to return")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity threshold")
    metric: SimilarityMetric = Field(SimilarityMetric.HAMMING, description="Similarity metric")
    
    # Filtering parameters
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    privacy_preserving: bool = Field(True, description="Use privacy-preserving search")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "query_vector": [1, -1, 1, 1, -1],
                "top_k": 20,
                "similarity_threshold": 0.7,
                "metric": "hamming",
                "privacy_preserving": True
            }
        }


class VectorSearchResult(BaseModel):
    """Individual vector search result."""
    
    vector_id: str = Field(..., description="Vector identifier")
    similarity_score: float = Field(..., description="Similarity score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Vector metadata")


class VectorSearchResponse(BaseModel):
    """Response model for vector similarity search."""
    
    results: List[VectorSearchResult] = Field(..., description="Search results")
    total_candidates: int = Field(..., description="Total number of candidates searched")
    search_time_ms: float = Field(..., description="Search time in milliseconds")
    privacy_guarantees: Optional[Dict[str, str]] = Field(None, description="Privacy guarantees provided")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "results": [
                    {
                        "vector_id": "vec_abc123",
                        "similarity_score": 0.92,
                        "metadata": {"source": "patient_cohort_1"}
                    },
                    {
                        "vector_id": "vec_def456", 
                        "similarity_score": 0.87,
                        "metadata": {"source": "population_reference"}
                    }
                ],
                "total_candidates": 10000,
                "search_time_ms": 245.3,
                "privacy_guarantees": {
                    "level": "differential_private",
                    "epsilon": "0.1"
                }
            }
        }


class VectorStoreRequest(BaseModel):
    """Request model for storing vectors."""
    
    vector_id: Optional[str] = Field(None, description="Vector identifier (generated if not provided)")
    vector: List[Union[int, float]] = Field(..., description="Vector data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Vector metadata")
    ttl_seconds: Optional[int] = Field(None, ge=60, description="Time to live in seconds")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "vector": [1, -1, 1, 1, -1],
                "metadata": {
                    "source": "genomic_analysis",
                    "patient_id_hash": "sha256_hash",
                    "created_by": "analysis_pipeline_v2"
                },
                "ttl_seconds": 3600
            }
        }


class VectorStoreResponse(BaseModel):
    """Response model for storing vectors."""
    
    vector_id: str = Field(..., description="Stored vector identifier")
    storage_time_ms: float = Field(..., description="Storage time in milliseconds")
    expiration_time: Optional[str] = Field(None, description="Vector expiration time (ISO format)")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "vector_id": "vec_1234567890abcdef",
                "storage_time_ms": 12.5,
                "expiration_time": "2024-01-15T11:30:00Z"
            }
        }
                                }
                            }
                        }
                    }
                }
            }
        }
    }