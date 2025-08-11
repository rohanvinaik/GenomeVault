"""
Type definitions for API models.

This module provides TypedDict and structured type definitions
to replace "magical" dict[str, Any] in the API layer.
"""

from typing import TypedDict, List, Optional, Literal


# Health check types
class HealthCheckResult(TypedDict):
    """Result of a single health check."""

    status: Literal["healthy", "unhealthy", "degraded"]
    message: Optional[str]
    error: Optional[str]
    latency_ms: Optional[float]


class ComponentHealth(TypedDict):
    """Health status of a system component."""

    status: str
    message: str
    last_check: Optional[str]
    metadata: Optional[dict]


# Hypervector metadata types
class EncodingMetadata(TypedDict):
    """Metadata for hypervector encoding."""

    encoding_time_ms: float
    input_size_bytes: int
    compression_ratio: float
    variants_encoded: int
    full_vector_available: bool
    truncated_for_response: bool


class SampleData(TypedDict):
    """Data for a single sample in batch processing."""

    sample_id: str
    data: str
    metadata: Optional[dict]


class BatchResult(TypedDict):
    """Result for a single sample in batch processing."""

    sample_id: str
    vector: List[float]
    dimension: int
    encoding_time_ms: float
    error: Optional[str]


class BatchMetadata(TypedDict):
    """Metadata for batch processing."""

    total_samples: int
    successful: int
    failed: int
    total_time_ms: float
    average_time_per_sample_ms: float


# Variant and genomic data types
class Variant(TypedDict):
    """Genomic variant representation."""

    chromosome: str
    position: int
    ref: str
    alt: str
    type: Optional[Literal["SNP", "INDEL", "CNV", "SV"]]
    quality: Optional[float]
    depth: Optional[int]


class GenomicFeatures(TypedDict):
    """Extracted genomic features."""

    variants: List[Variant]
    total_variants: int
    snp_count: int
    indel_count: int
    quality_metrics: Optional[dict]


# Query and search types
class QueryParams(TypedDict):
    """Parameters for database queries."""

    cohort_id: str
    statistic: str
    filters: Optional[dict]
    limit: Optional[int]
    offset: Optional[int]


class SearchMetadata(TypedDict):
    """Metadata for search operations."""

    query_dimension: int
    index_size: int
    search_time_ms: float
    distance_metric: str
    pruning_enabled: bool


# Clinical evaluation types
class ConfusionMatrix(TypedDict):
    """Confusion matrix for binary classification."""

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


class CalibrationBin(TypedDict):
    """Calibration bin for probability calibration."""

    bin_start: float
    bin_end: float
    count: int
    positive_fraction: float
    mean_predicted: float


class ClinicalMetrics(TypedDict):
    """Clinical evaluation metrics."""

    auc: float
    accuracy: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1_score: float
    mcc: float


# Governance and consent types
class ConsentRecord(TypedDict):
    """Consent record for data usage."""

    subject_id: str
    scope: str
    granted: bool
    granted_at: Optional[str]
    expires_at: Optional[str]
    revoked_at: Optional[str]


class DataSubjectInfo(TypedDict):
    """Information about a data subject."""

    subject_id: str
    created_at: str
    last_updated: str
    data_categories: List[str]
    processing_purposes: List[str]
    retention_period_days: int


# Configuration types
class HypervectorConfig(TypedDict):
    """Configuration for hypervector encoding."""

    dimension: int
    projection_type: str
    sparsity: float
    normalize: bool
    quantize: bool
    quantization_bits: int
    seed: Optional[int]


class StorageConfig(TypedDict):
    """Storage configuration."""

    backend: Literal["local", "s3", "gcs", "azure"]
    bucket: Optional[str]
    prefix: Optional[str]
    encryption_enabled: bool
    compression_enabled: bool


# Index manifest types
class IndexManifest(TypedDict):
    """Manifest for vector index."""

    ids: List[str]
    dimension: int
    metric: Literal["hamming", "cosine", "euclidean"]
    n_vectors: int
    dtype: str
    version: str
    created_at: str
    metadata: Optional[dict]


# Proof and verification types
class ProofMetadata(TypedDict):
    """Metadata for zero-knowledge proofs."""

    circuit_type: str
    prover_id: str
    timestamp: str
    computation_hash: str
    public_inputs_hash: str


class VerificationResult(TypedDict):
    """Result of proof verification."""

    valid: bool
    verifier_id: str
    timestamp: str
    error: Optional[str]


# Performance metrics types
class PerformanceMetrics(TypedDict):
    """Performance metrics for operations."""

    operation: str
    duration_ms: float
    memory_used_mb: float
    cpu_percent: float
    timestamp: str


# Error response types
class ErrorDetail(TypedDict):
    """Detailed error information."""

    code: str
    message: str
    field: Optional[str]
    context: Optional[dict]


class ErrorResponse(TypedDict):
    """Standard error response."""

    error: str
    details: Optional[List[ErrorDetail]]
    request_id: Optional[str]
    timestamp: str
