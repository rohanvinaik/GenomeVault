# genomevault/exceptions.py
from __future__ import annotations
from typing import Any, Dict, Type
from .core.exceptions import (
    GVError, GVConfigError, GVInputError, GVNotFound, GVStateError,
    GVComputeError, GVSecurityError, GVTimeout, EncodingError, ProjectionError, HypervectorError
)

__all__ = [
    "GVError", "GVConfigError", "GVInputError", "GVNotFound",
    "GVStateError", "GVComputeError", "GVSecurityError", "GVTimeout",
    "EncodingError", "ProjectionError", "HypervectorError", "error_response", "CODEC",
]

def error_response(exc: GVError) -> Dict[str, Any]:
    return exc.to_dict()

CODEC: Dict[str, Type[GVError]] = {
    c.code: c for c in [
        GVError, GVConfigError, GVInputError, GVNotFound, GVStateError,
        GVComputeError, GVSecurityError, GVTimeout, EncodingError, ProjectionError, HypervectorError
    ]
}