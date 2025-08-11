# genomevault/core/exceptions.py
from __future__ import annotations

"""Exceptions module."""
from typing import Any, Dict, Optional


class GVError(Exception):
    """Base class for all GenomeVault errors."""

    code: str = "GV_ERROR"
    http_status: int = 500

    def __init__(self, message: str = "", *, details: Optional[Dict[str, Any]] = None):
        """Initialize instance.

        Args:
            message: Message string.
        """
        super().__init__(message or self.__class__.__name__)
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """To dict.

        Returns:
            Operation result.
        """
        return {
            "type": self.__class__.__name__,
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }


class GVConfigError(GVError):
    """Exception raised for gvconfig errors."""

    code = "GV_CONFIG"


class GVInputError(GVError):
    """Exception raised for gvinput errors."""

    code = "GV_INPUT"
    http_status = 400


class GVNotFound(GVError):
    """GVNotFound implementation."""

    code = "GV_NOT_FOUND"
    http_status = 404


class GVStateError(GVError):
    """Exception raised for gvstate errors."""

    code = "GV_STATE"
    http_status = 409


class GVComputeError(GVError):
    """Exception raised for gvcompute errors."""

    code = "GV_COMPUTE"


class GVSecurityError(GVError):
    """Exception raised for gvsecurity errors."""

    code = "GV_SECURITY"
    http_status = 403


class GVTimeout(GVError):
    """GVTimeout implementation."""

    code = "GV_TIMEOUT"
    http_status = 504


# Domain-specific aliases (used in HV/ZK code)
class EncodingError(GVComputeError):
    """Exception raised for encoding errors."""

    code = "GV_ENCODING"


class ProjectionError(GVComputeError):
    """Exception raised for projection errors."""

    code = "GV_PROJECTION"

    def __init__(
        self,
        message: str = "",
        *,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize instance.

        Args:
            message: Message string.
            details: Details dictionary.
            context: Context dictionary (deprecated, use details).
        """
        # Accept both 'details' and 'context' for backwards compatibility
        super().__init__(message, details=details or context)


class HypervectorError(GVComputeError):
    """Exception raised for hypervector errors."""

    code = "GV_HYPERVECTOR"


class ValidationError(GVInputError):
    """Exception raised for validation errors."""

    code = "GV_VALIDATION"


class VerificationError(GVSecurityError):
    """Exception raised for verification errors."""

    code = "GV_VERIFICATION"


class PIRError(GVComputeError):
    """Exception raised for pir errors."""

    code = "GV_PIR"


class ProcessingError(GVComputeError):
    """Exception raised for processing errors."""

    code = "GV_PROCESSING"


class BindingError(GVComputeError):
    """Exception raised for binding errors."""

    code = "GV_BINDING"


# Legacy alias for backwards compatibility
GenomeVaultError = GVError
