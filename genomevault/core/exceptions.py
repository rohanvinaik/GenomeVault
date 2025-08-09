# genomevault/core/exceptions.py
from __future__ import annotations
from typing import Any, Dict, Optional

class GVError(Exception):
    """Base class for all GenomeVault errors."""
    code: str = "GV_ERROR"
    http_status: int = 500

    def __init__(self, message: str = "", *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message or self.__class__.__name__)
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }

class GVConfigError(GVError):
    code = "GV_CONFIG"

class GVInputError(GVError):
    code = "GV_INPUT"
    http_status = 400

class GVNotFound(GVError):
    code = "GV_NOT_FOUND"
    http_status = 404

class GVStateError(GVError):
    code = "GV_STATE"
    http_status = 409

class GVComputeError(GVError):
    code = "GV_COMPUTE"

class GVSecurityError(GVError):
    code = "GV_SECURITY"
    http_status = 403

class GVTimeout(GVError):
    code = "GV_TIMEOUT"
    http_status = 504

# Domain-specific aliases (used in HV/ZK code)
class EncodingError(GVComputeError):
    code = "GV_ENCODING"

class ProjectionError(GVComputeError):
    code = "GV_PROJECTION"

class HypervectorError(GVComputeError):
    code = "GV_HYPERVECTOR"