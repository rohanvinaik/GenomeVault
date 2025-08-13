"""Exceptions module."""

from __future__ import annotations

from typing import Any, Dict, Type

from .core.exceptions import (
    EncodingError,
    GVComputeError,
    GVConfigError,
    GVError,
    GVInputError,
    GVNotFound,
    GVSecurityError,
    GVStateError,
    GVTimeout,
    HypervectorError,
    ProjectionError,
)

__all__ = [
    "GVError",
    "GVConfigError",
    "GVInputError",
    "GVNotFound",
    "GVStateError",
    "GVComputeError",
    "GVSecurityError",
    "GVTimeout",
    "EncodingError",
    "ProjectionError",
    "HypervectorError",
    "error_response",
    "CODEC",
]


def error_response(exc: GVError) -> Dict[str, Any]:
    """Error response.

    Args:
        exc: Exc.

    Returns:
        Operation result.
    """
    return exc.to_dict()


CODEC: Dict[str, Type[GVError]] = {
    c.code: c
    for c in [
        GVError,
        GVConfigError,
        GVInputError,
        GVNotFound,
        GVStateError,
        GVComputeError,
        GVSecurityError,
        GVTimeout,
        EncodingError,
        ProjectionError,
        HypervectorError,
    ]
}
