"""Pydantic models for GenomeVault SDK."""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Union, Any
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class GenomicVariant(BaseModel):
    """Genomic variant model."""

    chrom: str = Field(..., description="Chromosome (1-22, X, Y, M)")
    pos: int = Field(..., ge=1, description="Genomic position (1-based)")
    ref: str = Field(..., regex=r"^[ATCGN]+$", description="Reference allele")
    alt: str = Field(..., regex=r"^[ATCGN]+$", description="Alternative allele")
    impact: Optional[str] = Field(None, description="Predicted functional impact")
    quality: Optional[float] = Field(None, ge=0, le=100, description="Variant quality score")

    @validator("chrom")
    def validate_chromosome(cls, v):
        """Validate chromosome format."""
        valid_chroms = [str(i) for i in range(1, 23)] + ["X", "Y", "M", "MT"]
        clean_chrom = v.replace("chr", "").upper()
        if clean_chrom not in valid_chroms:
            raise ValueError(f"Invalid chromosome: {v}")
        return clean_chrom

    @validator("impact")
    def validate_impact(cls, v):
        """Validate impact type."""
        if v is None:
            return v
        valid_impacts = [
            "missense",
            "nonsense",
            "synonymous",
            "frameshift",
            "splice_site",
            "intron",
            "intergenic",
        ]
        if v not in valid_impacts:
            raise ValueError(f"Invalid impact: {v}")
        return v


class EncodeRequest(BaseModel):
    """Request model for hypervector encoding."""

    numeric: Optional[List[float]] = Field(None, description="Numeric feature array")
    variants: Optional[List[Dict[str, Any]]] = Field(None, description="Genomic variants to encode")
    dim: int = Field(8192, ge=1024, le=100000, description="Hypervector dimension")
    binary: bool = Field(False, description="Return binary (-1/+1) or continuous values")

    @validator("variants")
    def validate_variants_or_numeric(cls, v, values):
        """Ensure either variants or numeric is provided."""
        if v is None and values.get("numeric") is None:
            raise ValueError("Either 'variants' or 'numeric' must be provided")
        return v


class EncodeResponse(BaseModel):
    """Response model for hypervector encoding."""

    dim: int = Field(..., description="Hypervector dimension")
    binary: bool = Field(..., description="Whether vector contains binary values")
    vector: List[Union[int, float]] = Field(..., description="Encoded hypervector")
    privacy_level: Optional[str] = Field(None, description="Privacy guarantee level")
    compression_ratio: Optional[float] = Field(None, description="Data compression ratio achieved")


class PIRQueryRequest(BaseModel):
    """Request model for PIR queries."""

    index: int = Field(..., ge=0, description="Index to query")
    query_id: Optional[str] = Field(None, description="Unique query identifier")
    timeout_seconds: int = Field(30, ge=1, le=300, description="Query timeout")

    @validator("query_id", pre=True, always=True)
    def set_query_id(cls, v):
        """Set default query ID if not provided."""
        return v or str(uuid4())


class PIRQueryResponse(BaseModel):
    """Response model for PIR queries."""

    index: int = Field(..., description="Queried index")
    item_base64: str = Field(..., description="Base64-encoded retrieved item")
    privacy_proof: Optional[str] = Field(None, description="Cryptographic proof of privacy")
    query_time_ms: Optional[int] = Field(None, description="Query execution time in milliseconds")


class ProofRequest(BaseModel):
    """Request model for zero-knowledge proofs."""

    proof_type: str = Field(..., description="Type of proof to generate")
    public_inputs: Dict[str, Any] = Field(..., description="Public inputs visible to verifiers")
    private_inputs_hash: str = Field(
        ..., regex=r"^[a-f0-9]{64}$", description="SHA-256 hash of private inputs"
    )
    circuit_params: Dict[str, Any] = Field(
        default_factory=dict, description="Circuit-specific parameters"
    )

    @validator("proof_type")
    def validate_proof_type(cls, v):
        """Validate proof type."""
        valid_types = ["genomic", "clinical", "research"]
        if v not in valid_types:
            raise ValueError(f"Invalid proof type: {v}. Must be one of {valid_types}")
        return v


class ProofResponse(BaseModel):
    """Response model for zero-knowledge proofs."""

    proof_id: str = Field(..., description="Unique proof identifier")
    proof_data: str = Field(..., description="Hex-encoded zk-SNARK proof")
    verification_key: str = Field(..., description="Verification key for proof validation")
    public_signals: List[str] = Field(
        default_factory=list, description="Public signals from the proof"
    )
    validity_period_hours: Optional[int] = Field(None, description="Proof validity period")


class ClinicalVariant(BaseModel):
    """Clinical variant model."""

    gene: str = Field(..., regex=r"^[A-Z0-9-]+$", description="Gene symbol (HGNC approved)")
    variant: str = Field(..., description="HGVS notation variant")
    classification: Optional[str] = Field(None, description="Clinical variant classification")
    evidence_level: Optional[str] = Field(None, description="Evidence level (ClinGen guidelines)")

    @validator("classification")
    def validate_classification(cls, v):
        """Validate clinical classification."""
        if v is None:
            return v
        valid_classifications = [
            "pathogenic",
            "likely_pathogenic",
            "uncertain_significance",
            "likely_benign",
            "benign",
        ]
        if v not in valid_classifications:
            raise ValueError(f"Invalid classification: {v}")
        return v

    @validator("evidence_level")
    def validate_evidence_level(cls, v):
        """Validate evidence level."""
        if v is None:
            return v
        valid_levels = ["A", "B", "C", "D"]
        if v not in valid_levels:
            raise ValueError(f"Invalid evidence level: {v}")
        return v


class ClinicalAnalysisRequest(BaseModel):
    """Request model for clinical analysis."""

    patient_id_hash: str = Field(
        ..., regex=r"^[a-f0-9]{64}$", description="SHA-256 hash of patient identifier"
    )
    variants: List[ClinicalVariant] = Field(..., description="Clinical variants for analysis")
    analysis_type: str = Field(..., description="Type of clinical analysis")
    population_reference: str = Field("gnomAD", description="Population reference database")
    consent_hash: Optional[str] = Field(
        None, regex=r"^[a-f0-9]{64}$", description="Hash of patient consent"
    )

    @validator("analysis_type")
    def validate_analysis_type(cls, v):
        """Validate analysis type."""
        valid_types = ["risk_assessment", "pharmacogenomics", "carrier_screening", "diagnostic"]
        if v not in valid_types:
            raise ValueError(f"Invalid analysis type: {v}")
        return v

    @validator("population_reference")
    def validate_population_reference(cls, v):
        """Validate population reference."""
        valid_refs = ["gnomAD", "1000G", "ESP", "ExAC"]
        if v not in valid_refs:
            raise ValueError(f"Invalid population reference: {v}")
        return v


class ClinicalAnalysisResponse(BaseModel):
    """Response model for clinical analysis."""

    analysis_id: str = Field(..., description="Unique analysis identifier")
    risk_score: float = Field(..., ge=0, le=1, description="Calculated risk score")
    confidence_interval: List[float] = Field(
        ..., min_items=2, max_items=2, description="95% confidence interval"
    )
    recommendations: List[str] = Field(default_factory=list, description="Clinical recommendations")
    audit_trail_hash: str = Field(..., description="Cryptographic hash of audit trail")
    differential_privacy_epsilon: Optional[float] = Field(
        None, description="Differential privacy parameter"
    )


class HealthResponse(BaseModel):
    """Response model for health checks."""

    status: str = Field(..., description="Overall system status")
    timestamp: datetime.datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(default_factory=dict, description="Service health status")

    @validator("status")
    def validate_status(cls, v):
        """Validate health status."""
        valid_statuses = ["healthy", "degraded", "unhealthy"]
        if v not in valid_statuses:
            raise ValueError(f"Invalid status: {v}")
        return v


class APIError(BaseModel):
    """API error response model."""

    type: str = Field(..., description="Error type classification")
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime.datetime = Field(..., description="Error timestamp")
    trace_id: Optional[str] = Field(None, description="Distributed tracing identifier")


class ErrorDetail(BaseModel):
    """Individual error detail for validation errors."""

    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message for this field")
    code: str = Field(..., description="Error code for this field")
    value: Optional[Any] = Field(None, description="Invalid value (if safe to expose)")
    allowed_values: Optional[List[str]] = Field(None, description="List of allowed values")


class ValidationErrorResponse(APIError):
    """Extended error response for validation errors."""

    errors: List[ErrorDetail] = Field(
        default_factory=list, description="Field-level validation errors"
    )


# Utility models for batch operations
class BatchEncodeRequest(BaseModel):
    """Request model for batch encoding operations."""

    requests: List[EncodeRequest] = Field(..., description="List of encoding requests")
    max_concurrent: int = Field(10, ge=1, le=50, description="Maximum concurrent requests")


class BatchEncodeResponse(BaseModel):
    """Response model for batch encoding operations."""

    results: List[EncodeResponse] = Field(..., description="List of encoding responses")
    success_count: int = Field(..., description="Number of successful requests")
    error_count: int = Field(..., description="Number of failed requests")
    errors: List[APIError] = Field(
        default_factory=list, description="List of errors for failed requests"
    )


# Pagination models
class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""

    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(50, ge=1, le=1000, description="Number of items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: str = Field("asc", regex=r"^(asc|desc)$", description="Sort order")


class PaginatedResponse(BaseModel):
    """Base model for paginated responses."""

    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_count: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")


# Configuration models
class ClientConfig(BaseModel):
    """Client configuration model."""

    base_url: str = Field("https://api.genomevault.io", description="Base API URL")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    oauth_token: Optional[str] = Field(None, description="OAuth2 token")
    timeout: float = Field(30.0, ge=1.0, description="Request timeout in seconds")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum number of retries")
    retry_backoff: float = Field(1.0, ge=0.1, description="Backoff factor for retries")

    class Config:
        """Pydantic config."""

        use_enum_values = True
        validate_assignment = True
