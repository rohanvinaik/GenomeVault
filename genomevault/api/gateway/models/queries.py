"""
Query and PIR models for GenomeVault API Gateway.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, validator

from genomevault.api.gateway.models.base import BaseModel


class QueryType(str, Enum):
    """Types of queries supported."""
    
    PIR = "pir"
    FEDERATED = "federated"
    STATISTICAL = "statistical"
    GENOMIC_SEARCH = "genomic_search"
    CLINICAL_LOOKUP = "clinical_lookup"


class QueryStatus(str, Enum):
    """Query execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PIRProtocol(str, Enum):
    """PIR protocol types."""
    
    LATTICE_BASED = "lattice_based"
    COMPUTATIONAL = "computational"
    INFORMATION_THEORETIC = "information_theoretic"
    XPIR = "xpir"
    MULPIR = "mulpir"


class PIRQueryRequest(BaseModel):
    """Request model for Private Information Retrieval query."""
    
    index: int = Field(..., ge=0, description="Index to query (kept private from server)")
    database_id: str = Field(..., description="Database identifier")
    protocol: PIRProtocol = Field(PIRProtocol.LATTICE_BASED, description="PIR protocol to use")
    
    # Query parameters
    query_id: Optional[str] = Field(None, description="Unique query identifier for tracking")
    timeout_seconds: int = Field(30, ge=1, le=300, description="Query timeout")
    
    # Privacy parameters
    noise_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="Additional noise level")
    batch_size: Optional[int] = Field(None, ge=1, le=1000, description="Batch query size")
    
    # Client parameters
    client_public_key: Optional[str] = Field(None, description="Client public key for encryption")
    encryption_params: Optional[Dict[str, Any]] = Field(None, description="Encryption parameters")
    
    model_config = {
        "json_schema_extra": {
            "examples": {
                "basic_pir": {
                    "summary": "Basic PIR query",
                    "value": {
                        "index": 42,
                        "database_id": "genomic_variants_db",
                        "protocol": "lattice_based",
                        "timeout_seconds": 60
                    }
                },
                "batch_pir": {
                    "summary": "Batch PIR query",
                    "value": {
                        "index": 100,
                        "database_id": "clinical_records_db",
                        "protocol": "computational",
                        "batch_size": 10,
                        "query_id": "batch_query_001"
                    }
                }
            }
        }


class PIRQueryResponse(BaseModel):
    """Response model for PIR query."""
    
    query_id: str = Field(..., description="Query identifier")
    index: int = Field(..., description="Queried index (for client verification)")
    
    # Retrieved data
    item_base64: str = Field(..., description="Base64-encoded retrieved item")
    item_size_bytes: int = Field(..., description="Size of retrieved item in bytes")
    
    # Privacy guarantees
    privacy_proof: Optional[str] = Field(None, description="Cryptographic proof of privacy preservation")
    privacy_level: str = Field(..., description="Privacy guarantee level")
    
    # Performance metrics
    query_time_ms: int = Field(..., description="Query execution time in milliseconds")
    network_overhead_bytes: Optional[int] = Field(None, description="Network overhead in bytes")
    
    # Metadata
    protocol_used: PIRProtocol = Field(..., description="PIR protocol used")
    server_count: int = Field(..., description="Number of PIR servers involved")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "query_id": "query_1234567890",
                "index": 42,
                "item_base64": "YWxwaGE=",
                "item_size_bytes": 1024,
                "privacy_proof": "zk_proof_hash_example",
                "privacy_level": "information_theoretic",
                "query_time_ms": 125,
                "network_overhead_bytes": 4096,
                "protocol_used": "lattice_based",
                "server_count": 3
            }
        }


class QueryExecutionRequest(BaseModel):
    """Request model for general query execution."""
    
    query_type: QueryType = Field(..., description="Type of query to execute")
    query_params: Dict[str, Any] = Field(..., description="Query-specific parameters")
    
    # Execution options
    async_execution: bool = Field(False, description="Execute query asynchronously")
    priority: int = Field(5, ge=1, le=10, description="Query priority (1=highest, 10=lowest)")
    timeout_seconds: int = Field(300, ge=1, le=3600, description="Query timeout")
    
    # Privacy options
    privacy_preserving: bool = Field(True, description="Use privacy-preserving execution")
    differential_privacy: bool = Field(False, description="Apply differential privacy")
    epsilon: Optional[float] = Field(None, ge=0.001, le=10.0, description="Differential privacy epsilon")
    
    # Result options
    result_format: str = Field("json", pattern=r"^(json|csv|binary)$", description="Result format")
    compression: Optional[str] = Field(None, pattern=r"^(gzip|brotli|lz4)$", description="Result compression")
    
    @field_validator("query_params")
    def validate_query_params(cls, v, values):
        """Validate query parameters based on query type."""
        query_type = values.get("query_type")
        if query_type == QueryType.PIR:
            if "index" not in v or "database_id" not in v:
                raise ValueError("PIR queries require 'index' and 'database_id' parameters")
        elif query_type == QueryType.GENOMIC_SEARCH:
            if "search_terms" not in v:
                raise ValueError("Genomic search queries require 'search_terms' parameter")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": {
                "pir_query": {
                    "summary": "PIR query execution",
                    "value": {
                        "query_type": "pir",
                        "query_params": {
                            "index": 42,
                            "database_id": "genomic_variants_db"
                        },
                        "privacy_preserving": True,
                        "timeout_seconds": 60
                    }
                },
                "genomic_search": {
                    "summary": "Genomic search query",
                    "value": {
                        "query_type": "genomic_search",
                        "query_params": {
                            "search_terms": ["BRCA1", "pathogenic"],
                            "population": "EUR",
                            "max_results": 100
                        },
                        "differential_privacy": True,
                        "epsilon": 0.1
                    }
                }
            }
        }


class QueryExecutionResponse(BaseModel):
    """Response model for query execution."""
    
    query_id: str = Field(..., description="Unique query identifier")
    query_type: QueryType = Field(..., description="Type of query executed")
    status: QueryStatus = Field(..., description="Query execution status")
    
    # Results (if completed)
    results: Optional[Union[Dict[str, Any], List[Any], str]] = Field(None, description="Query results")
    result_count: Optional[int] = Field(None, description="Number of results returned")
    result_size_bytes: Optional[int] = Field(None, description="Size of results in bytes")
    
    # Execution metadata
    execution_time_ms: Optional[int] = Field(None, description="Execution time in milliseconds")
    started_at: datetime = Field(..., description="Query start time")
    completed_at: Optional[datetime] = Field(None, description="Query completion time")
    
    # Privacy information
    privacy_guarantees: Optional[Dict[str, str]] = Field(None, description="Privacy guarantees provided")
    audit_trail_hash: Optional[str] = Field(None, description="Audit trail hash")
    
    # Error information (if failed)
    error_message: Optional[str] = Field(None, description="Error message if query failed")
    error_details: Optional[Dict[str, str]] = Field(None, description="Detailed error information")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "query_id": "query_abc123456789",
                "query_type": "genomic_search",
                "status": "completed",
                "results": [
                    {
                        "variant_id": "rs123456",
                        "gene": "BRCA1",
                        "clinical_significance": "pathogenic"
                    }
                ],
                "result_count": 1,
                "result_size_bytes": 256,
                "execution_time_ms": 1250,
                "started_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:30:01Z",
                "privacy_guarantees": {
                    "level": "differential_private",
                    "epsilon": "0.1"
                }
            }
        }


class QueryStatusRequest(BaseModel):
    """Request model for query status check."""
    
    query_id: str = Field(..., description="Query identifier")
    include_results: bool = Field(False, description="Include results in response if completed")
    include_logs: bool = Field(False, description="Include execution logs")


class QueryStatusResponse(BaseModel):
    """Response model for query status."""
    
    query_id: str = Field(..., description="Query identifier")
    status: QueryStatus = Field(..., description="Current query status")
    progress_percent: Optional[int] = Field(None, ge=0, le=100, description="Execution progress percentage")
    
    # Timing information
    created_at: datetime = Field(..., description="Query creation time")
    started_at: Optional[datetime] = Field(None, description="Query start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    completed_at: Optional[datetime] = Field(None, description="Actual completion time")
    
    # Resource usage
    cpu_time_ms: Optional[int] = Field(None, description="CPU time used in milliseconds")
    memory_usage_mb: Optional[float] = Field(None, description="Peak memory usage in MB")
    network_io_bytes: Optional[int] = Field(None, description="Network I/O in bytes")
    
    # Results (if requested and available)
    results: Optional[Union[Dict[str, Any], List[Any]]] = Field(None, description="Query results")
    logs: Optional[List[str]] = Field(None, description="Execution logs")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "query_id": "query_abc123456789",
                "status": "running",
                "progress_percent": 75,
                "created_at": "2024-01-15T10:30:00Z",
                "started_at": "2024-01-15T10:30:05Z",
                "estimated_completion": "2024-01-15T10:32:00Z",
                "cpu_time_ms": 45000,
                "memory_usage_mb": 256.7,
                "network_io_bytes": 1048576
            }
        }


class QueryCancelRequest(BaseModel):
    """Request model for query cancellation."""
    
    query_id: str = Field(..., description="Query identifier")
    reason: Optional[str] = Field(None, description="Reason for cancellation")
    force: bool = Field(False, description="Force cancellation even if in critical phase")


class QueryCancelResponse(BaseModel):
    """Response model for query cancellation."""
    
    query_id: str = Field(..., description="Query identifier")
    cancelled: bool = Field(..., description="Whether query was successfully cancelled")
    final_status: QueryStatus = Field(..., description="Final query status")
    message: Optional[str] = Field(None, description="Cancellation message")
    partial_results: Optional[Dict[str, Any]] = Field(None, description="Partial results if available")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "query_id": "query_abc123456789",
                "cancelled": True,
                "final_status": "cancelled",
                "message": "Query cancelled by user request"
            }
        }


class DatabaseListRequest(BaseModel):
    """Request model for listing available databases."""
    
    database_type: Optional[str] = Field(None, description="Filter by database type")
    access_level: Optional[str] = Field(None, description="Filter by required access level")
    include_stats: bool = Field(False, description="Include database statistics")


class DatabaseInfo(BaseModel):
    """Information about available database."""
    
    database_id: str = Field(..., description="Database identifier")
    name: str = Field(..., description="Human-readable database name")
    description: Optional[str] = Field(None, description="Database description")
    database_type: str = Field(..., description="Database type")
    
    # Access information
    access_level: str = Field(..., description="Required access level")
    public: bool = Field(..., description="Whether database is publicly accessible")
    
    # Statistics (if requested)
    record_count: Optional[int] = Field(None, description="Number of records")
    size_bytes: Optional[int] = Field(None, description="Database size in bytes")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    
    # PIR capabilities
    supports_pir: bool = Field(..., description="Whether database supports PIR queries")
    pir_protocols: List[PIRProtocol] = Field(..., description="Supported PIR protocols")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "database_id": "genomic_variants_db",
                "name": "Genomic Variants Database",
                "description": "Curated genomic variants with clinical annotations",
                "database_type": "genomic",
                "access_level": "research",
                "public": False,
                "record_count": 1000000,
                "size_bytes": 1073741824,
                "last_updated": "2024-01-15T00:00:00Z",
                "supports_pir": True,
                "pir_protocols": ["lattice_based", "computational"]
            }
        }


class DatabaseListResponse(BaseModel):
    """Response model for database listing."""
    
    databases: List[DatabaseInfo] = Field(..., description="Available databases")
    total: int = Field(..., description="Total number of databases")
    accessible: int = Field(..., description="Number of databases accessible to user")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "databases": [
                    {
                        "database_id": "genomic_variants_db",
                        "name": "Genomic Variants Database",
                        "database_type": "genomic",
                        "access_level": "research",
                        "public": False,
                        "supports_pir": True,
                        "pir_protocols": ["lattice_based"]
                    }
                ],
                "total": 15,
                "accessible": 8
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