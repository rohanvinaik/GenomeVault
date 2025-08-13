"""
Core type definitions for GenomeVault internal boundaries.

This module provides TypedDict and structured types to replace
"magical" dict[str, Any] throughout the codebase.
"""

"""
Core type definitions for GenomeVault internal boundaries.

This module provides TypedDict and structured types to replace
"magical" dict[str, Any] throughout the codebase.
"""
from typing import TypedDict, List, Optional, Dict, Literal, Any


class ShardMetadata(TypedDict):
    """Metadata for a data shard."""

    shard_id: str
    chromosome: str
    start_position: int
    end_position: int
    data_path: str
    index_path: str
    compressed: bool
    size_bytes: int
    created_at: str
    checksum: str


class CacheStats(TypedDict):
    """Cache statistics."""

    hits: int
    misses: int
    evictions: int
    size_bytes: int
    items: int


# Model and training types
class ModelMetadata(TypedDict):
    """Metadata for trained models."""

    model_id: str
    model_type: str
    version: str
    training_dataset: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    created_at: str
    checksum: str


class TrainingConfig(TypedDict):
    """Configuration for model training."""

    batch_size: int
    learning_rate: float
    epochs: int
    optimizer: str
    loss_function: str
    early_stopping: bool
    validation_split: float


# Processing pipeline types
class PipelineStep(TypedDict):
    """Single step in processing pipeline."""

    name: str
    function: str
    parameters: Dict[str, Any]
    input_type: str
    output_type: str
    timeout_seconds: Optional[int]


class PipelineResult(TypedDict):
    """Result from pipeline execution."""

    pipeline_id: str
    steps_completed: int
    total_steps: int
    status: Literal["success", "failed", "partial"]
    outputs: List[Any]
    errors: List[str]
    execution_time_ms: float


# Genomic data types
class GenomicAnnotation(TypedDict):
    """Annotation for genomic features."""

    feature_type: str
    gene_name: Optional[str]
    transcript_id: Optional[str]
    consequence: Optional[str]
    impact: Optional[Literal["high", "moderate", "low", "modifier"]]
    allele_frequency: Optional[float]


class ReferenceGenome(TypedDict):
    """Reference genome information."""

    name: str
    version: str
    assembly: str
    species: str
    total_length: int
    num_chromosomes: int
    source_url: str


# Security and audit types
class AuditLog(TypedDict):
    """Audit log entry."""

    event_id: str
    timestamp: str
    user_id: str
    action: str
    resource: str
    resource_id: str
    success: bool
    ip_address: Optional[str]
    user_agent: Optional[str]
    details: Optional[Dict[str, Any]]


class AccessControl(TypedDict):
    """Access control rules."""

    resource: str
    permissions: List[str]
    roles: List[str]
    conditions: Optional[Dict[str, Any]]


# Federated learning types
class FederatedNode(TypedDict):
    """Node in federated learning network."""

    node_id: str
    hostname: str
    port: int
    status: Literal["active", "inactive", "failed"]
    last_heartbeat: str
    capabilities: List[str]
    trust_score: float


class FederatedUpdate(TypedDict):
    """Update from federated learning node."""

    node_id: str
    round_number: int
    model_delta: bytes
    metrics: Dict[str, float]
    samples_processed: int
    timestamp: str


# Blockchain types
class BlockData(TypedDict):
    """Data for blockchain block."""

    block_number: int
    previous_hash: str
    timestamp: str
    transactions: List[str]
    merkle_root: str
    nonce: int
    difficulty: int


class TransactionData(TypedDict):
    """Blockchain transaction data."""

    transaction_id: str
    from_address: str
    to_address: str
    amount: float
    data: Optional[str]
    gas_price: float
    gas_limit: int
    timestamp: str


# Metrics and monitoring types
class SystemMetrics(TypedDict):
    """System performance metrics."""

    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io_bytes: int
    active_connections: int
    timestamp: str


class QueryMetrics(TypedDict):
    """Query performance metrics."""

    query_id: str
    query_type: str
    duration_ms: float
    rows_processed: int
    cache_hit: bool
    timestamp: str


# Configuration types
class DatabaseConfig(TypedDict):
    """Database configuration."""

    host: str
    port: int
    database: str
    username: str
    pool_size: int
    timeout_seconds: int
    ssl_enabled: bool


class CacheConfig(TypedDict):
    """Cache configuration."""

    backend: Literal["redis", "memcached", "in-memory"]
    host: Optional[str]
    port: Optional[int]
    max_size_mb: int
    ttl_seconds: int
    eviction_policy: Literal["lru", "lfu", "fifo"]


# Validation and error types
class ValidationResult(TypedDict):
    """Result of data validation."""

    valid: bool
    errors: List[str]
    warnings: List[str]
    processed_items: int
    failed_items: int


class ErrorContext(TypedDict):
    """Context information for errors."""

    error_code: str
    error_type: str
    message: str
    stack_trace: Optional[str]
    context: Dict[str, Any]
    timestamp: str
