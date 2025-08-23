"""Exceptions for GenomeVault Python SDK."""

from __future__ import annotations

from typing import Dict, List, Optional, Any

import httpx


class GenomeVaultAPIError(Exception):
    """Base exception for GenomeVault API errors."""

    def __init__(
        self,
        message: str,
        response: Optional[httpx.Response] = None,
        request_id: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        """
        Initialize API error.

        Args:
            message: Error message
            response: HTTP response object
            request_id: Request identifier for support
            error_code: GenomeVault error code
        """
        super().__init__(message)
        self.message = message
        self.response = response
        self.request_id = request_id
        self.error_code = error_code

        # Extract additional info from response
        if response:
            self.status_code = response.status_code
            self.headers = dict(response.headers)

            # Try to extract error details from response body
            try:
                error_data = response.json()
                self.request_id = self.request_id or error_data.get("request_id")
                self.error_code = self.error_code or error_data.get("code")
                self.error_type = error_data.get("type")
                self.details = error_data.get("details", {})
            except Exception:
                self.error_type = None
                self.details = {}
        else:
            self.status_code = None
            self.headers = {}
            self.error_type = None
            self.details = {}

    def __str__(self) -> str:
        """String representation of the error."""
        parts = [self.message]

        if self.error_code:
            parts.append(f"Code: {self.error_code}")

        if self.request_id:
            parts.append(f"Request ID: {self.request_id}")

        if self.status_code:
            parts.append(f"HTTP {self.status_code}")

        return " | ".join(parts)

    def __repr__(self) -> str:
        """Representation of the error."""
        return f"{self.__class__.__name__}('{self.message}', error_code='{self.error_code}')"


class AuthenticationError(GenomeVaultAPIError):
    """Authentication failed (401)."""

    def __init__(self, message: str = "Authentication required", **kwargs):
        super().__init__(message, **kwargs)


class AuthorizationError(GenomeVaultAPIError):
    """Authorization failed (403)."""

    def __init__(self, message: str = "Insufficient permissions", **kwargs):
        super().__init__(message, **kwargs)


class ValidationError(GenomeVaultAPIError):
    """Request validation failed (422)."""

    def __init__(
        self,
        message: str = "Request validation failed",
        errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.validation_errors = errors or []

    def __str__(self) -> str:
        """String representation including validation errors."""
        base_str = super().__str__()

        if self.validation_errors:
            error_details = []
            for error in self.validation_errors:
                field = error.get("field", "unknown")
                msg = error.get("message", "validation error")
                error_details.append(f"{field}: {msg}")

            if error_details:
                base_str += f" | Errors: {', '.join(error_details)}"

        return base_str


class RateLimitError(GenomeVaultAPIError):
    """Rate limit exceeded (429)."""

    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(message, **kwargs)

        # Extract rate limit info from headers
        if self.response:
            self.limit = self.response.headers.get("X-RateLimit-Limit")
            self.remaining = self.response.headers.get("X-RateLimit-Remaining")
            self.reset_time = self.response.headers.get("X-RateLimit-Reset")
            self.retry_after = self.response.headers.get("Retry-After")
        else:
            self.limit = None
            self.remaining = None
            self.reset_time = None
            self.retry_after = None

    def __str__(self) -> str:
        """String representation including rate limit info."""
        base_str = super().__str__()

        if self.retry_after:
            base_str += f" | Retry after: {self.retry_after}s"

        if self.limit and self.remaining:
            base_str += f" | Limit: {self.remaining}/{self.limit}"

        return base_str


class ServiceUnavailableError(GenomeVaultAPIError):
    """Service temporarily unavailable (503)."""

    def __init__(self, message: str = "Service temporarily unavailable", **kwargs):
        super().__init__(message, **kwargs)

        # Extract retry info from headers
        if self.response:
            self.retry_after = self.response.headers.get("Retry-After")
        else:
            self.retry_after = None


class NotFoundError(GenomeVaultAPIError):
    """Resource not found (404)."""

    def __init__(self, message: str = "Resource not found", **kwargs):
        super().__init__(message, **kwargs)


class ConflictError(GenomeVaultAPIError):
    """Resource conflict (409)."""

    def __init__(self, message: str = "Resource conflict", **kwargs):
        super().__init__(message, **kwargs)


class TimeoutError(GenomeVaultAPIError):
    """Request timeout."""

    def __init__(self, message: str = "Request timed out", **kwargs):
        super().__init__(message, **kwargs)


class NetworkError(GenomeVaultAPIError):
    """Network connectivity error."""

    def __init__(self, message: str = "Network error", **kwargs):
        super().__init__(message, **kwargs)


# Domain-specific exceptions
class GenomicDataError(ValidationError):
    """Error in genomic data format or content."""

    def __init__(self, message: str = "Invalid genomic data", **kwargs):
        super().__init__(message, **kwargs)


class PIRQueryError(GenomeVaultAPIError):
    """Error in PIR query execution."""

    def __init__(self, message: str = "PIR query failed", **kwargs):
        super().__init__(message, **kwargs)


class ProofGenerationError(GenomeVaultAPIError):
    """Error in zero-knowledge proof generation."""

    def __init__(self, message: str = "Proof generation failed", **kwargs):
        super().__init__(message, **kwargs)


class ClinicalAnalysisError(GenomeVaultAPIError):
    """Error in clinical analysis."""

    def __init__(self, message: str = "Clinical analysis failed", **kwargs):
        super().__init__(message, **kwargs)


class PHIDetectedError(ValidationError):
    """Protected health information detected in request."""

    def __init__(
        self,
        message: str = "Protected health information detected",
        phi_fields: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.phi_fields = phi_fields or []

    def __str__(self) -> str:
        """String representation including PHI fields."""
        base_str = super().__str__()

        if self.phi_fields:
            base_str += f" | Fields: {', '.join(self.phi_fields)}"

        return base_str


# Error factory functions
def create_error_from_response(response: httpx.Response) -> GenomeVaultAPIError:
    """
    Create appropriate exception from HTTP response.

    Args:
        response: HTTP response object

    Returns:
        Appropriate exception instance
    """
    try:
        error_data = response.json()
        message = error_data.get("message", f"HTTP {response.status_code} error")
        error_code = error_data.get("code")
        request_id = error_data.get("request_id")
        error_type = error_data.get("type")

    except Exception:
        message = response.text or f"HTTP {response.status_code} error"
        error_code = None
        request_id = response.headers.get("X-Request-ID")
        error_type = None

    # Map status codes to exception types
    status_exceptions = {
        401: AuthenticationError,
        403: AuthorizationError,
        404: NotFoundError,
        409: ConflictError,
        422: ValidationError,
        429: RateLimitError,
        503: ServiceUnavailableError,
    }

    # Map error codes to exception types
    code_exceptions = {
        "GV_PHI_DETECTED": PHIDetectedError,
        "GV_PIR_QUERY_FAILED": PIRQueryError,
        "GV_PROOF_VERIFICATION_FAILED": ProofGenerationError,
        "GV_CLINICAL_DATA_INCOMPLETE": ClinicalAnalysisError,
        "GV_INVALID_GENOMIC_COORDINATE": GenomicDataError,
        "GV_INVALID_VARIANT_FORMAT": GenomicDataError,
    }

    # Choose exception class
    if error_code and error_code in code_exceptions:
        exception_class = code_exceptions[error_code]
    elif response.status_code in status_exceptions:
        exception_class = status_exceptions[response.status_code]
    else:
        exception_class = GenomeVaultAPIError

    # Create exception with appropriate parameters
    if exception_class == ValidationError and error_data:
        return exception_class(
            message=message,
            errors=error_data.get("errors"),
            response=response,
            request_id=request_id,
            error_code=error_code,
        )
    elif exception_class == PHIDetectedError and error_data:
        phi_fields = []
        for error in error_data.get("errors", []):
            if error.get("field"):
                phi_fields.append(error["field"])

        return exception_class(
            message=message,
            phi_fields=phi_fields,
            response=response,
            request_id=request_id,
            error_code=error_code,
        )
    else:
        return exception_class(
            message=message,
            response=response,
            request_id=request_id,
            error_code=error_code,
        )


def create_timeout_error(timeout: float) -> TimeoutError:
    """
    Create timeout error.

    Args:
        timeout: Timeout value in seconds

    Returns:
        TimeoutError instance
    """
    return TimeoutError(f"Request timed out after {timeout} seconds")


def create_network_error(original_error: Exception) -> NetworkError:
    """
    Create network error from underlying exception.

    Args:
        original_error: Original exception

    Returns:
        NetworkError instance
    """
    return NetworkError(f"Network error: {str(original_error)}")


# Context manager for error handling
class GenomeVaultErrorHandler:
    """Context manager for handling GenomeVault API errors."""

    def __init__(self, operation: str):
        """
        Initialize error handler.

        Args:
            operation: Description of the operation being performed
        """
        self.operation = operation

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager with error handling."""
        if exc_type is None:
            return False

        if isinstance(exc_val, GenomeVaultAPIError):
            # Re-raise GenomeVault errors as-is
            return False

        if isinstance(exc_val, httpx.TimeoutException):
            # Convert timeout exceptions
            new_exc = TimeoutError(f"{self.operation} timed out")
            raise new_exc from exc_val

        if isinstance(exc_val, httpx.RequestError):
            # Convert network errors
            new_exc = NetworkError(f"{self.operation} failed due to network error: {str(exc_val)}")
            raise new_exc from exc_val

        # Let other exceptions pass through
        return False
