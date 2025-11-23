"""
Error handling middleware for GenomeVault API Gateway.
"""

from __future__ import annotations

import traceback
import uuid
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from genomevault.api.gateway.models.base import ErrorResponse, ErrorType
from genomevault.observability.logging import get_logger

logger = get_logger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive error handling middleware.

    Handles different types of errors:
    - HTTP exceptions
    - Validation errors
    - Application-specific errors
    - Unexpected exceptions
    """

    def __init__(self, app, include_debug_info: bool = False):
        """
        Initialize error handling middleware.

        Args:
            app: FastAPI application instance
            include_debug_info: Whether to include debug information in error responses
        """
        super().__init__(app)
        self.include_debug_info = include_debug_info

        # Error type mappings
        self.error_mappings = {
            400: ErrorType.VALIDATION_ERROR,
            401: ErrorType.AUTHENTICATION_ERROR,
            403: ErrorType.AUTHORIZATION_ERROR,
            404: ErrorType.RESOURCE_NOT_FOUND,
            409: ErrorType.CONFLICT_ERROR,
            429: ErrorType.RATE_LIMIT_ERROR,
            503: ErrorType.SERVICE_UNAVAILABLE,
            500: ErrorType.INTERNAL_ERROR,
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with comprehensive error handling.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware or route handler

        Returns:
            HTTP response with proper error formatting
        """
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        try:
            # Process the request
            response = await call_next(request)
            return response

        except HTTPException as exc:
            return await self._handle_http_exception(exc, request_id, request)

        except RequestValidationError as exc:
            return await self._handle_validation_error(exc, request_id, request)

        except ValidationError as exc:
            return await self._handle_pydantic_validation_error(exc, request_id, request)

        except Exception as exc:
            return await self._handle_unexpected_error(exc, request_id, request)

    async def _handle_http_exception(
        self, exc: HTTPException, request_id: str, request: Request
    ) -> JSONResponse:
        """
        Handle FastAPI HTTP exceptions.

        Args:
            exc: HTTP exception
            request_id: Request identifier
            request: HTTP request

        Returns:
            JSON error response
        """
        # Determine error type
        error_type = self.error_mappings.get(exc.status_code, ErrorType.INTERNAL_ERROR)

        # Extract error details
        if isinstance(exc.detail, dict):
            # Structured error detail
            error_code = exc.detail.get("code", f"GV_HTTP_{exc.status_code}")
            message = exc.detail.get("message", str(exc.detail))
            details = exc.detail.get("details")
        else:
            # Simple string detail
            error_code = f"GV_HTTP_{exc.status_code}"
            message = str(exc.detail)
            details = None

        # Create error response
        error_response = ErrorResponse(
            type=error_type,
            code=error_code,
            message=message,
            details=details,
            request_id=request_id,
        )

        # Log the error
        logger.warning(
            f"HTTP exception: {exc.status_code} {message}",
            extra={
                "request_id": request_id,
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method,
                "error_code": error_code,
                "client_ip": request.client.host if request.client else None,
            },
        )

        # Prepare response headers
        headers = {"X-Request-ID": request_id}
        if hasattr(exc, "headers") and exc.headers:
            headers.update(exc.headers)

        return JSONResponse(
            status_code=exc.status_code, content=error_response.dict(), headers=headers
        )

    async def _handle_validation_error(
        self, exc: RequestValidationError, request_id: str, request: Request
    ) -> JSONResponse:
        """
        Handle FastAPI request validation errors.

        Args:
            exc: Validation error
            request_id: Request identifier
            request: HTTP request

        Returns:
            JSON error response
        """
        # Extract validation error details
        error_details = []
        for error in exc.errors():
            location = " -> ".join(str(loc) for loc in error["loc"])
            error_details.append(
                {"field": location, "message": error["msg"], "type": error["type"]}
            )

        # Create error response
        error_response = ErrorResponse(
            type=ErrorType.VALIDATION_ERROR,
            code="GV_VALIDATION_ERROR",
            message="Request validation failed",
            details={"validation_errors": error_details, "request_id": request_id},
            request_id=request_id,
        )

        # Log validation error
        logger.info(
            f"Request validation error: {len(error_details)} errors",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "validation_errors": error_details,
            },
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.dict(),
            headers={"X-Request-ID": request_id},
        )

    async def _handle_pydantic_validation_error(
        self, exc: ValidationError, request_id: str, request: Request
    ) -> JSONResponse:
        """
        Handle Pydantic model validation errors.

        Args:
            exc: Pydantic validation error
            request_id: Request identifier
            request: HTTP request

        Returns:
            JSON error response
        """
        # Extract Pydantic validation error details
        error_details = []
        for error in exc.errors():
            location = " -> ".join(str(loc) for loc in error["loc"])
            error_details.append(
                {"field": location, "message": error["msg"], "type": error["type"]}
            )

        # Create error response
        error_response = ErrorResponse(
            type=ErrorType.VALIDATION_ERROR,
            code="GV_MODEL_VALIDATION_ERROR",
            message="Data model validation failed",
            details={"validation_errors": error_details, "request_id": request_id},
            request_id=request_id,
        )

        # Log validation error
        logger.info(
            f"Pydantic validation error: {len(error_details)} errors",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "validation_errors": error_details,
            },
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.dict(),
            headers={"X-Request-ID": request_id},
        )

    async def _handle_unexpected_error(
        self, exc: Exception, request_id: str, request: Request
    ) -> JSONResponse:
        """
        Handle unexpected application errors.

        Args:
            exc: Unexpected exception
            request_id: Request identifier
            request: HTTP request

        Returns:
            JSON error response
        """
        # Get exception details
        exc_type = type(exc).__name__
        exc_message = str(exc)

        # Prepare debug information
        debug_info = None
        if self.include_debug_info:
            debug_info = {
                "exception_type": exc_type,
                "exception_message": exc_message,
                "traceback": traceback.format_exc(),
            }

        # Create error response
        error_response = ErrorResponse(
            type=ErrorType.INTERNAL_ERROR,
            code="GV_INTERNAL_ERROR",
            message="An unexpected error occurred",
            details={
                "request_id": request_id,
                "error_type": exc_type,
                **({"debug_info": debug_info} if debug_info else {}),
            },
            request_id=request_id,
        )

        # Log the error with full details
        logger.error(
            f"Unexpected error: {exc_type}: {exc_message}",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "exception_type": exc_type,
                "client_ip": request.client.host if request.client else None,
            },
            exc_info=True,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.dict(),
            headers={"X-Request-ID": request_id},
        )

    def _sanitize_error_message(self, message: str) -> str:
        """
        Sanitize error message to prevent information leakage.

        Args:
            message: Original error message

        Returns:
            Sanitized error message
        """
        # Remove potential sensitive information patterns
        sensitive_patterns = [
            r"password=\w+",
            r"token=\w+",
            r"key=\w+",
            r"secret=\w+",
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card numbers
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        ]

        sanitized_message = message
        for pattern in sensitive_patterns:
            import re

            sanitized_message = re.sub(
                pattern, "[REDACTED]", sanitized_message, flags=re.IGNORECASE
            )

        return sanitized_message

    def create_error_response(
        self,
        error_type: ErrorType,
        code: str,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> JSONResponse:
        """
        Create a standardized error response.

        Args:
            error_type: Type of error
            code: Error code
            message: Error message
            status_code: HTTP status code
            details: Additional error details
            request_id: Request identifier

        Returns:
            JSON error response
        """
        if not request_id:
            request_id = str(uuid.uuid4())

        error_response = ErrorResponse(
            type=error_type,
            code=code,
            message=self._sanitize_error_message(message),
            details=details,
            request_id=request_id,
        )

        return JSONResponse(
            status_code=status_code,
            content=error_response.dict(),
            headers={"X-Request-ID": request_id},
        )
