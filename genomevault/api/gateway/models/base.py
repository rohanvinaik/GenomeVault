"""
Base Pydantic models for GenomeVault API Gateway.

Common models and validation patterns used across all endpoints.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, field_validator


T = TypeVar("T")


class BaseModel(PydanticBaseModel):
    """Base model with common configuration."""

    model_config = {
        "use_enum_values": True,
        "validate_assignment": True,
        "populate_by_name": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: str,
        },
    }


class RequestMetadata(BaseModel):
    """Metadata included in API requests."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    client_ip: Optional[str] = Field(None, description="Client IP address")
    trace_id: Optional[str] = Field(None, description="Distributed tracing identifier")


class ErrorType(str, Enum):
    """Error type classifications."""

    VALIDATION_ERROR = "ValidationError"
    AUTHENTICATION_ERROR = "AuthenticationError"
    AUTHORIZATION_ERROR = "AuthorizationError"
    RATE_LIMIT_ERROR = "RateLimitError"
    RESOURCE_NOT_FOUND = "ResourceNotFound"
    CONFLICT_ERROR = "ConflictError"
    SERVICE_UNAVAILABLE = "ServiceUnavailable"
    INTERNAL_ERROR = "InternalError"


class ErrorResponse(BaseModel):
    """Standard error response model."""

    type: ErrorType = Field(..., description="Error type classification")
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")
    request_id: str = Field(..., description="Unique request identifier for support")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    trace_id: Optional[str] = Field(None, description="Distributed tracing identifier")

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "ValidationError",
                "code": "GV_INVALID_INPUT",
                "message": "Invalid genomic coordinate format",
                "details": {
                    "field": "variants[0].chrom",
                    "allowed_values": ["1", "2", "3", "...", "22", "X", "Y", "M"],
                },
                "request_id": "req_1234567890",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }
    }


class SuccessResponse(BaseModel, Generic[T]):
    """Standard success response wrapper."""

    success: bool = Field(True, description="Operation success indicator")
    data: T = Field(..., description="Response data")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class PaginationParams(BaseModel):
    """Pagination parameters."""

    page: int = Field(1, ge=1, description="Page number (1-based)")
    per_page: int = Field(20, ge=1, le=100, description="Items per page")

    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.per_page


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""

    items: List[T] = Field(..., description="Page items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")

    @field_validator("total_pages", mode="before")
    @classmethod
    def calculate_total_pages(cls, v, info):
        """Calculate total pages."""
        total = info.data.get("total", 0)
        per_page = info.data.get("per_page", 1)
        return (total + per_page - 1) // per_page if total > 0 else 0

    @field_validator("has_next", mode="before")
    @classmethod
    def calculate_has_next(cls, v, info):
        """Calculate if there are more pages."""
        page = info.data.get("page", 1)
        total_pages = info.data.get("total_pages", 0)
        return page < total_pages

    @field_validator("has_previous", mode="before")
    @classmethod
    def calculate_has_previous(cls, v, info):
        """Calculate if there are previous pages."""
        page = info.data.get("page", 1)
        return page > 1


class PrivacyLevel(str, Enum):
    """Privacy guarantee levels."""

    K_ANONYMOUS = "k-anonymous"
    DIFFERENTIAL_PRIVATE = "differential_private"
    INFORMATION_THEORETIC = "information_theoretic"


class ProcessingStatus(str, Enum):
    """Processing status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenomicVariant(BaseModel):
    """Genomic variant representation."""

    chrom: str = Field(
        ...,
        pattern=r"^(chr)?(1[0-9]|2[0-2]|[1-9]|X|Y|M|MT)$",
        description="Chromosome (1-22, X, Y, M)",
    )
    pos: int = Field(..., ge=1, description="Genomic position (1-based)")
    ref: str = Field(..., pattern=r"^[ATCGN]+$", description="Reference allele")
    alt: str = Field(..., pattern=r"^[ATCGN]+$", description="Alternative allele")
    impact: Optional[str] = Field(
        None,
        pattern=r"^(missense|nonsense|synonymous|frameshift|splice_site|intron|intergenic)$",
        description="Predicted functional impact",
    )
    quality: Optional[float] = Field(None, ge=0, le=100, description="Variant quality score")

    model_config = {
        "json_schema_extra": {
            "example": {
                "chrom": "1",
                "pos": 1234567,
                "ref": "A",
                "alt": "T",
                "impact": "missense",
                "quality": 99.5,
            }
        }
    }


class RateLimitInfo(BaseModel):
    """Rate limiting information."""

    limit: int = Field(..., description="Request limit per window")
    remaining: int = Field(..., description="Requests remaining in current window")
    reset_time: int = Field(..., description="Time when rate limit resets (Unix timestamp)")
    window_seconds: int = Field(..., description="Rate limit window duration in seconds")


class AuditTrail(BaseModel):
    """Audit trail information."""

    user_id: Optional[str] = Field(None, description="User identifier")
    action: str = Field(..., description="Action performed")
    resource: str = Field(..., description="Resource accessed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Action timestamp")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    request_id: str = Field(..., description="Request identifier")
    status: str = Field(..., description="Action status (success/failure)")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional audit details")
