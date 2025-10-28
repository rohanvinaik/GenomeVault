"""
Pydantic models for GDiff/HDV API endpoints.

These models support the GDiff → Selective HDV workflow with caching,
schema selection, and k-anonymity configuration.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class GDiffHDVGenerateRequest(BaseModel):
    """Request model for generating HDV encoding from GDiff."""

    vcf_path: str = Field(
        ...,
        description="Path to query VCF file",
        min_length=1,
        max_length=1000
    )
    reference_pool: Optional[str] = Field(
        None,
        description="Path to reference pool directory (auto-detected if not provided)",
    )
    schema: str = Field(
        default="clinical_risk",
        description="Analysis schema to use (simple_snp_lookup, clinical_risk, etc.)",
    )
    k_anonymity: Optional[int] = Field(
        None,
        description="k-anonymity level (auto-selected based on available references if not provided)",
        ge=2,
        le=15,
    )
    cache_dir: str = Field(
        default="data/hdv_cache",
        description="HDV cache directory",
    )
    gdiff_path: Optional[str] = Field(
        None,
        description="Path to existing GDiff file (if already generated)",
    )
    force: bool = Field(
        default=False,
        description="Force regeneration even if cached",
    )
    enable_encryption: bool = Field(
        default=False,
        description="Enable AES-256-GCM encryption for GDiff files",
    )
    encryption_password: Optional[str] = Field(
        None,
        description="Password for encryption (if enable_encryption=true)",
    )

    @field_validator("schema")
    def validate_schema(cls, v: str) -> str:
        """Validate schema name."""
        valid_schemas = [
            "simple_snp_lookup",
            "clinical_risk",
            "pharmacogenomics",
            "ancestry_inference",
            "nanopore_structural_inference",
            "epigenetic_landscape",
            "full_research_profile",
        ]
        if v not in valid_schemas:
            raise ValueError(
                f"Invalid schema: {v}. "
                f"Must be one of: {', '.join(valid_schemas)}"
            )
        return v

    class Config:
        """Config implementation."""

        schema_extra = {
            "example": {
                "vcf_path": "data/query.vcf.gz",
                "reference_pool": "benchmark_results/layer2_reference_pool",
                "schema": "clinical_risk",
                "k_anonymity": 3,
                "cache_dir": "data/hdv_cache",
                "force": False,
            }
        }


class GDiffHDVGenerateResponse(BaseModel):
    """Response model for HDV generation."""

    status: str = Field(..., description="Status: 'cached' or 'generated'")
    query_id: str = Field(..., description="Unique query ID")
    k_anonymity: int = Field(..., description="k-anonymity level used")
    schema: str = Field(..., description="Analysis schema used")
    dimension: int = Field(..., description="HDV dimension")
    hdv_size_kb: float = Field(..., description="HDV size in KB")
    encoding_time_ms: Optional[float] = Field(None, description="Encoding time (if generated)")
    num_variants: int = Field(..., description="Number of variants encoded")
    features_used: List[str] = Field(..., description="Feature categories used")
    hdv_path: str = Field(..., description="Path to cached HDV file")
    cache_stats: Dict[str, Any] = Field(..., description="Cache statistics")

    class Config:
        """Config implementation."""

        schema_extra = {
            "example": {
                "status": "generated",
                "query_id": "a1b2c3d4e5f6g7h8",
                "k_anonymity": 3,
                "schema": "clinical_risk",
                "dimension": 2048,
                "hdv_size_kb": 8.0,
                "encoding_time_ms": 52.3,
                "num_variants": 120,
                "features_used": ["position", "allele", "differential", "functional", "quality"],
                "hdv_path": "data/hdv_cache/a1b2c3d4/k3/clinical_risk.hdc",
                "cache_stats": {
                    "num_encodings": 1,
                    "k_levels_available": [3],
                    "schemas_available": ["clinical_risk"],
                },
            }
        }


class GDiffBatchGenerateRequest(BaseModel):
    """Request model for batch HDV generation."""

    vcf_path: str = Field(..., description="Path to query VCF file")
    reference_pool: Optional[str] = Field(None, description="Path to reference pool directory")
    schemas: List[str] = Field(
        default=["clinical_risk", "pharmacogenomics"],
        description="List of schemas to generate",
        min_items=1,
        max_items=7,
    )
    k_levels: List[int] = Field(
        default=[3, 7, 13],
        description="List of k-anonymity levels",
        min_items=1,
        max_items=10,
    )
    cache_dir: str = Field(default="data/hdv_cache", description="HDV cache directory")
    gdiff_path: Optional[str] = Field(None, description="Path to existing GDiff file")

    @field_validator("schemas")
    def validate_schemas(cls, v: List[str]) -> List[str]:
        """Validate all schema names."""
        valid_schemas = [
            "simple_snp_lookup",
            "clinical_risk",
            "pharmacogenomics",
            "ancestry_inference",
            "nanopore_structural_inference",
            "epigenetic_landscape",
            "full_research_profile",
        ]
        for schema in v:
            if schema not in valid_schemas:
                raise ValueError(
                    f"Invalid schema: {schema}. "
                    f"Must be one of: {', '.join(valid_schemas)}"
                )
        return v

    @field_validator("k_levels")
    def validate_k_levels(cls, v: List[int]) -> List[int]:
        """Validate k-anonymity levels."""
        for k in v:
            if k < 2 or k > 15:
                raise ValueError(f"k-anonymity level {k} out of range [2, 15]")
        return sorted(set(v))  # Remove duplicates and sort

    class Config:
        """Config implementation."""

        schema_extra = {
            "example": {
                "vcf_path": "data/query.vcf.gz",
                "schemas": ["clinical_risk", "pharmacogenomics", "ancestry_inference"],
                "k_levels": [3, 7, 13],
            }
        }


class GDiffBatchGenerateResponse(BaseModel):
    """Response model for batch HDV generation."""

    total: int = Field(..., description="Total number of combinations")
    success: int = Field(..., description="Number of successful generations")
    skipped: int = Field(..., description="Number of cached (skipped)")
    errors: int = Field(..., description="Number of errors")
    results: List[Dict[str, Any]] = Field(..., description="Individual results")
    cache_stats: Dict[str, Any] = Field(..., description="Final cache statistics")

    class Config:
        """Config implementation."""

        schema_extra = {
            "example": {
                "total": 9,
                "success": 6,
                "skipped": 3,
                "errors": 0,
                "results": [
                    {
                        "schema": "clinical_risk",
                        "k": 3,
                        "status": "generated",
                        "encoding_time_ms": 52.3,
                    },
                    {
                        "schema": "clinical_risk",
                        "k": 7,
                        "status": "cached",
                    },
                ],
                "cache_stats": {
                    "num_encodings": 9,
                    "k_levels_available": [3, 7, 13],
                    "schemas_available": ["clinical_risk", "pharmacogenomics", "ancestry_inference"],
                },
            }
        }


class SchemaInfo(BaseModel):
    """Schema information model."""

    schema_name: str = Field(..., description="Schema identifier")
    dimension: int = Field(..., description="HDV dimension")
    encoding_time_ms: int = Field(..., description="Expected encoding time")
    hdv_size_kb: float = Field(..., description="Expected HDV size in KB")
    num_features: int = Field(..., description="Number of feature categories")
    privacy_level: str = Field(..., description="Privacy level (standard/enhanced/maximum)")
    description: str = Field(..., description="Schema description")
    use_cases: List[str] = Field(..., description="Recommended use cases")
    requires_nanopore: bool = Field(..., description="Requires Nanopore data")
    requires_epigenetic: bool = Field(..., description="Requires epigenetic data")


class SchemasListResponse(BaseModel):
    """Response model for listing schemas."""

    schemas: List[SchemaInfo] = Field(..., description="Available analysis schemas")
    total: int = Field(..., description="Total number of schemas")


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""

    query_id: str = Field(..., description="Query ID")
    num_encodings: int = Field(..., description="Total number of cached encodings")
    k_levels_available: List[int] = Field(..., description="Available k-anonymity levels")
    schemas_available: List[str] = Field(..., description="Available schemas")
    gdiff_exists: bool = Field(..., description="Whether GDiff exists in cache")
    total_hdv_size_mb: float = Field(..., description="Total HDV cache size in MB")

    class Config:
        """Config implementation."""

        schema_extra = {
            "example": {
                "query_id": "a1b2c3d4e5f6g7h8",
                "num_encodings": 6,
                "k_levels_available": [3, 7, 13],
                "schemas_available": ["clinical_risk", "pharmacogenomics"],
                "gdiff_exists": True,
                "total_hdv_size_mb": 0.15,
            }
        }


class CachedHDVInfo(BaseModel):
    """Information about a cached HDV."""

    k_anonymity: int = Field(..., description="k-anonymity level")
    schema_name: str = Field(..., description="Schema name")
    dimension: int = Field(..., description="HDV dimension")
    hdv_size_kb: float = Field(..., description="HDV size in KB")
    num_variants: int = Field(..., description="Number of variants encoded")
    encoding_time_ms: float = Field(..., description="Encoding time")
    created_timestamp: float = Field(..., description="Creation timestamp (Unix time)")
    hdv_path: str = Field(..., description="Path to HDV file")


class ListCachedHDVsResponse(BaseModel):
    """Response model for listing cached HDVs."""

    query_id: str = Field(..., description="Query ID")
    cached_hdvs: List[CachedHDVInfo] = Field(..., description="Cached HDV encodings")
    total: int = Field(..., description="Total number of cached HDVs")

    class Config:
        """Config implementation."""

        schema_extra = {
            "example": {
                "query_id": "a1b2c3d4e5f6g7h8",
                "cached_hdvs": [
                    {
                        "k_anonymity": 3,
                        "schema_name": "clinical_risk",
                        "dimension": 2048,
                        "hdv_size_kb": 8.0,
                        "num_variants": 120,
                        "encoding_time_ms": 52.3,
                        "created_timestamp": 1698765432.0,
                        "hdv_path": "data/hdv_cache/a1b2c3d4/k3/clinical_risk.hdc",
                    }
                ],
                "total": 1,
            }
        }
