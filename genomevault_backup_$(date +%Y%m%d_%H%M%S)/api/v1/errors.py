"""Comprehensive error handling for API v1 with PHI-safe responses."""

from __future__ import annotations

import uuid
import logging
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from genomevault.exceptions import GVError


logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    """Individual error detail."""

    field: Optional[str] = None
    message: str
    code: str
    value: Optional[Any] = None
    allowed_values: Optional[List[str]] = None


class APIError(BaseModel):
    """Comprehensive API error response model."""

    type: str = Field(..., description="Error type classification")
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message (PHI-safe)")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")
    errors: Optional[List[ErrorDetail]] = Field(None, description="Field-level validation errors")
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    trace_id: Optional[str] = Field(None, description="Distributed tracing identifier")


# PHI-safe error messages
PHI_SAFE_MESSAGES = {
    "genomic_coordinate_invalid": "Invalid genomic coordinate format",
    "variant_format_invalid": "Variant data format is invalid",
    "patient_id_invalid": "Patient identifier format is invalid",
    "clinical_data_invalid": "Clinical data validation failed",
    "sequence_data_invalid": "Sequence data format is invalid",
    "quality_score_invalid": "Quality score is out of valid range",
}


class ErrorCodes:
    """Standardized error codes for GenomeVault API."""

    # Generic errors
    VALIDATION_ERROR = "GV_VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "GV_AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "GV_AUTHORIZATION_ERROR"
    RATE_LIMIT_ERROR = "GV_RATE_LIMIT_ERROR"
    INTERNAL_ERROR = "GV_INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "GV_SERVICE_UNAVAILABLE"

    # Genomic-specific errors
    INVALID_GENOMIC_COORDINATE = "GV_INVALID_GENOMIC_COORDINATE"
    INVALID_VARIANT_FORMAT = "GV_INVALID_VARIANT_FORMAT"
    INVALID_SEQUENCE_DATA = "GV_INVALID_SEQUENCE_DATA"
    UNSUPPORTED_CHROMOSOME = "GV_UNSUPPORTED_CHROMOSOME"
    QUALITY_SCORE_OUT_OF_RANGE = "GV_QUALITY_SCORE_OUT_OF_RANGE"

    # Privacy and security errors
    PHI_DETECTED = "GV_PHI_DETECTED"
    ENCRYPTION_ERROR = "GV_ENCRYPTION_ERROR"
    PROOF_VERIFICATION_FAILED = "GV_PROOF_VERIFICATION_FAILED"
    PIR_QUERY_FAILED = "GV_PIR_QUERY_FAILED"

    # Clinical errors
    CLINICAL_DATA_INCOMPLETE = "GV_CLINICAL_DATA_INCOMPLETE"
    CONSENT_REQUIRED = "GV_CONSENT_REQUIRED"
    REGULATORY_COMPLIANCE_ERROR = "GV_REGULATORY_COMPLIANCE_ERROR"

    # System errors
    HYPERVECTOR_ENGINE_ERROR = "GV_HYPERVECTOR_ENGINE_ERROR"
    DATABASE_CONNECTION_ERROR = "GV_DATABASE_CONNECTION_ERROR"
    EXTERNAL_SERVICE_ERROR = "GV_EXTERNAL_SERVICE_ERROR"


def sanitize_error_message(message: str, field: Optional[str] = None) -> str:
    """
    Sanitize error messages to ensure they don't contain PHI.

    Args:
        message: Original error message
        field: Field name that caused the error

    Returns:
        PHI-safe error message
    """
    # Remove potential PHI patterns
    sanitized = message

    # Remove specific genomic coordinates
    import re

    sanitized = re.sub(
        r"\bchr[0-9XYM]+:\d+\b", "[genomic_coordinate]", sanitized, flags=re.IGNORECASE
    )

    # Remove specific sequences
    sanitized = re.sub(r"\b[ATCGN]{10,}\b", "[sequence_data]", sanitized, flags=re.IGNORECASE)

    # Remove potential patient identifiers
    sanitized = re.sub(r"\b[A-Z0-9]{6,20}\b", "[identifier]", sanitized)

    # Use pre-defined safe messages for common genomic errors
    if field and any(term in message.lower() for term in ["chromosome", "position", "genomic"]):
        return PHI_SAFE_MESSAGES.get("genomic_coordinate_invalid", sanitized)

    if field and any(term in message.lower() for term in ["variant", "allele"]):
        return PHI_SAFE_MESSAGES.get("variant_format_invalid", sanitized)

    if field and "patient" in message.lower():
        return PHI_SAFE_MESSAGES.get("patient_id_invalid", sanitized)

    return sanitized


def create_error_response(
    error_type: str,
    error_code: str,
    message: str,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
    errors: Optional[List[ErrorDetail]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> JSONResponse:
    """
    Create standardized error response.

    Args:
        error_type: Type of error (e.g., ValidationError)
        error_code: Machine-readable error code
        message: Human-readable error message
        status_code: HTTP status code
        details: Additional error details
        errors: Field-level validation errors
        request_id: Request identifier
        trace_id: Tracing identifier

    Returns:
        JSONResponse with error details
    """
    error = APIError(
        type=error_type,
        code=error_code,
        message=sanitize_error_message(message),
        details=details or {},
        errors=errors,
        request_id=request_id or str(uuid.uuid4()),
        trace_id=trace_id,
    )

    return JSONResponse(status_code=status_code, content=error.dict())


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle FastAPI validation errors with PHI-safe messages."""
    request_id = str(uuid.uuid4())
    trace_id = request.headers.get("X-Trace-Id")

    errors = []
    for error in exc.errors():
        field_path = ".".join(str(loc) for loc in error["loc"]) if error.get("loc") else None
        error_detail = ErrorDetail(
            field=field_path,
            message=sanitize_error_message(error["msg"], field_path),
            code=ErrorCodes.VALIDATION_ERROR,
            value=None,  # Don't expose potentially sensitive values
        )
        errors.append(error_detail)

    logger.warning(
        f"Validation error for request {request_id}: {len(errors)} validation errors",
        extra={"request_id": request_id, "error_count": len(errors)},
    )

    return create_error_response(
        error_type="ValidationError",
        error_code=ErrorCodes.VALIDATION_ERROR,
        message=f"Request validation failed with {len(errors)} error(s)",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        errors=errors,
        request_id=request_id,
        trace_id=trace_id,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    request_id = str(uuid.uuid4())
    trace_id = request.headers.get("X-Trace-Id")

    # Map HTTP status codes to error types and codes
    error_mapping = {
        400: ("BadRequestError", ErrorCodes.VALIDATION_ERROR),
        401: ("AuthenticationError", ErrorCodes.AUTHENTICATION_ERROR),
        403: ("AuthorizationError", ErrorCodes.AUTHORIZATION_ERROR),
        404: ("NotFoundError", "GV_NOT_FOUND"),
        429: ("RateLimitError", ErrorCodes.RATE_LIMIT_ERROR),
        500: ("InternalServerError", ErrorCodes.INTERNAL_ERROR),
        503: ("ServiceUnavailableError", ErrorCodes.SERVICE_UNAVAILABLE),
    }

    error_type, error_code = error_mapping.get(exc.status_code, ("HTTPError", "GV_HTTP_ERROR"))

    details = {}
    if exc.status_code == 429:
        details.update(
            {
                "retry_after": request.headers.get("Retry-After", "3600"),
                "limit": request.headers.get("X-RateLimit-Limit", "unknown"),
            }
        )

    logger.warning(
        f"HTTP {exc.status_code} error for request {request_id}: {exc.detail}",
        extra={"request_id": request_id, "status_code": exc.status_code},
    )

    return create_error_response(
        error_type=error_type,
        error_code=error_code,
        message=sanitize_error_message(str(exc.detail)),
        status_code=exc.status_code,
        details=details,
        request_id=request_id,
        trace_id=trace_id,
    )


async def genomevault_exception_handler(request: Request, exc: GVError) -> JSONResponse:
    """Handle custom GenomeVault exceptions."""
    request_id = str(uuid.uuid4())
    trace_id = request.headers.get("X-Trace-Id")

    # Map GVError to appropriate HTTP status and error details
    status_code = getattr(exc, "http_status", 500)
    error_code = getattr(exc, "error_code", ErrorCodes.INTERNAL_ERROR)

    logger.error(
        f"GenomeVault error for request {request_id}: {exc}",
        extra={"request_id": request_id, "error_type": type(exc).__name__},
    )

    return create_error_response(
        error_type=type(exc).__name__,
        error_code=error_code,
        message=sanitize_error_message(str(exc)),
        status_code=status_code,
        request_id=request_id,
        trace_id=trace_id,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with minimal information exposure."""
    request_id = str(uuid.uuid4())
    trace_id = request.headers.get("X-Trace-Id")

    # Log full exception details for debugging
    logger.error(
        f"Unhandled exception for request {request_id}: {type(exc).__name__}",
        extra={
            "request_id": request_id,
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        },
    )

    # Return minimal error information to client
    return create_error_response(
        error_type="InternalServerError",
        error_code=ErrorCodes.INTERNAL_ERROR,
        message="An unexpected error occurred. Please contact support with the request ID.",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details={"support_contact": "support@genomevault.io"},
        request_id=request_id,
        trace_id=trace_id,
    )


class PHIDetectionError(Exception):
    """Raised when PHI is detected in request data."""

    def __init__(self, field: str, phi_type: str):
        self.field = field
        self.phi_type = phi_type
        super().__init__(f"PHI detected in field '{field}': {phi_type}")


async def phi_detection_handler(request: Request, exc: PHIDetectionError) -> JSONResponse:
    """Handle PHI detection errors with secure logging."""
    request_id = str(uuid.uuid4())
    trace_id = request.headers.get("X-Trace-Id")

    # Log PHI detection for security audit (without exposing PHI)
    logger.error(
        f"PHI detected in request {request_id}: field='{exc.field}', type='{exc.phi_type}'",
        extra={
            "request_id": request_id,
            "phi_field": exc.field,
            "phi_type": exc.phi_type,
            "security_event": "PHI_DETECTION",
        },
    )

    return create_error_response(
        error_type="PHIDetectionError",
        error_code=ErrorCodes.PHI_DETECTED,
        message="Protected health information detected in request",
        status_code=status.HTTP_400_BAD_REQUEST,
        details={
            "field": exc.field,
            "remediation": "Remove any protected health information and resubmit the request",
        },
        request_id=request_id,
        trace_id=trace_id,
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all error handlers for the FastAPI app."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(GVError, genomevault_exception_handler)
    app.add_exception_handler(PHIDetectionError, phi_detection_handler)
    app.add_exception_handler(Exception, generic_exception_handler)


# Utility functions for custom error raising


def raise_validation_error(
    message: str,
    field: Optional[str] = None,
    value: Optional[Any] = None,
    allowed_values: Optional[List[str]] = None,
) -> None:
    """Raise a validation error with standardized format."""
    errors = [
        ErrorDetail(
            field=field,
            message=sanitize_error_message(message, field),
            code=ErrorCodes.VALIDATION_ERROR,
            value=None,  # Don't expose potentially sensitive values
            allowed_values=allowed_values,
        )
    ]

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Validation failed: {sanitize_error_message(message, field)}",
    )


def raise_genomic_error(
    message: str,
    error_code: str = ErrorCodes.INVALID_GENOMIC_COORDINATE,
    field: Optional[str] = None,
) -> None:
    """Raise a genomic data validation error."""
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, detail=sanitize_error_message(message, field)
    )


def raise_clinical_error(
    message: str,
    error_code: str = ErrorCodes.CLINICAL_DATA_INCOMPLETE,
    requires_consent: bool = False,
) -> None:
    """Raise a clinical data error."""
    status_code = status.HTTP_403_FORBIDDEN if requires_consent else status.HTTP_400_BAD_REQUEST

    raise HTTPException(status_code=status_code, detail=sanitize_error_message(message))
