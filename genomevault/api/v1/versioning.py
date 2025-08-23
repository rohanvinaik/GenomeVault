"""API versioning and deprecation management."""

from __future__ import annotations

import datetime
import warnings
from enum import Enum
from typing import Optional, Dict, Any

from fastapi import Request, Response
from pydantic import BaseModel


class APIVersion(str, Enum):
    """API version enumeration."""

    V1 = "v1"
    V2 = "v2"  # Future version


class DeprecationStatus(str, Enum):
    """API deprecation status levels."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"
    REMOVED = "removed"


class VersionInfo(BaseModel):
    """Version information model."""

    version: str
    status: DeprecationStatus
    release_date: datetime.date
    deprecation_date: Optional[datetime.date] = None
    sunset_date: Optional[datetime.date] = None
    successor_version: Optional[str] = None


# Version registry with deprecation timeline
VERSION_REGISTRY: Dict[str, VersionInfo] = {
    "v1": VersionInfo(
        version="v1",
        status=DeprecationStatus.ACTIVE,
        release_date=datetime.date(2024, 1, 1),
        deprecation_date=None,
        sunset_date=None,
        successor_version=None,
    ),
    # Future v2 example:
    # "v2": VersionInfo(
    #     version="v2",
    #     status=DeprecationStatus.ACTIVE,
    #     release_date=datetime.date(2024, 6, 1),
    #     deprecation_date=None,
    #     sunset_date=None,
    #     successor_version=None
    # )
}


class DeprecationWarning:
    """Handles API deprecation warnings and headers."""

    def __init__(self, version: str):
        self.version = version
        self.version_info = VERSION_REGISTRY.get(version)

    def add_deprecation_headers(self, response: Response) -> None:
        """Add deprecation headers to response if applicable."""
        if not self.version_info:
            return

        if self.version_info.status == DeprecationStatus.DEPRECATED:
            response.headers["Deprecation"] = "true"
            if self.version_info.sunset_date:
                response.headers["Sunset"] = self.version_info.sunset_date.isoformat()
            if self.version_info.successor_version:
                response.headers["Link"] = (
                    f'</api/{self.version_info.successor_version}>; rel="successor-version"'
                )

        elif self.version_info.status == DeprecationStatus.SUNSET:
            response.headers["Deprecation"] = "true"
            response.headers["Sunset"] = datetime.date.today().isoformat()
            if self.version_info.successor_version:
                response.headers["Link"] = (
                    f'</api/{self.version_info.successor_version}>; rel="successor-version"'
                )

    def emit_warning(self) -> None:
        """Emit Python warning for deprecated API usage."""
        if not self.version_info or self.version_info.status not in [
            DeprecationStatus.DEPRECATED,
            DeprecationStatus.SUNSET,
        ]:
            return

        message = f"API version {self.version} is {self.version_info.status.value}"
        if self.version_info.successor_version:
            message += f", please migrate to {self.version_info.successor_version}"
        if self.version_info.sunset_date:
            message += f", sunset date: {self.version_info.sunset_date}"

        warnings.warn(message, DeprecationWarning, stacklevel=2)


def get_api_version_from_path(path: str) -> Optional[str]:
    """Extract API version from URL path."""
    path_parts = path.strip("/").split("/")
    if len(path_parts) > 0 and path_parts[0].startswith("v"):
        return path_parts[0]
    return None


def validate_api_version(version: str) -> bool:
    """Validate if API version is supported."""
    return version in VERSION_REGISTRY


def get_version_info(version: str) -> Optional[VersionInfo]:
    """Get version information."""
    return VERSION_REGISTRY.get(version)


async def deprecation_middleware(request: Request, call_next):
    """Middleware to handle API deprecation."""
    # Extract version from path
    api_version = get_api_version_from_path(request.url.path)

    if api_version and validate_api_version(api_version):
        # Add version info to request state
        request.state.api_version = api_version
        request.state.version_info = get_version_info(api_version)

        # Process request
        response = await call_next(request)

        # Add deprecation headers
        deprecation = DeprecationWarning(api_version)
        deprecation.add_deprecation_headers(response)

        return response

    # Version not found or invalid
    response = await call_next(request)
    return response


def deprecate_version(
    version: str,
    deprecation_date: datetime.date,
    sunset_date: datetime.date,
    successor_version: Optional[str] = None,
) -> None:
    """Mark a version as deprecated with timeline."""
    if version in VERSION_REGISTRY:
        version_info = VERSION_REGISTRY[version]
        version_info.status = DeprecationStatus.DEPRECATED
        version_info.deprecation_date = deprecation_date
        version_info.sunset_date = sunset_date
        version_info.successor_version = successor_version


def sunset_version(version: str) -> None:
    """Mark a version as sunset (no longer accepting new requests)."""
    if version in VERSION_REGISTRY:
        VERSION_REGISTRY[version].status = DeprecationStatus.SUNSET


def remove_version(version: str) -> None:
    """Remove a version from the registry (no longer available)."""
    if version in VERSION_REGISTRY:
        VERSION_REGISTRY[version].status = DeprecationStatus.REMOVED


# Feature flags for experimental endpoints
FEATURE_FLAGS: Dict[str, Dict[str, Any]] = {
    "v1": {
        "experimental_federated_learning": False,
        "advanced_zk_circuits": False,
        "quantum_resistant_crypto": False,
    }
}


def is_feature_enabled(version: str, feature: str) -> bool:
    """Check if a feature flag is enabled for a version."""
    return FEATURE_FLAGS.get(version, {}).get(feature, False)


def enable_feature(version: str, feature: str) -> None:
    """Enable a feature flag for a version."""
    if version not in FEATURE_FLAGS:
        FEATURE_FLAGS[version] = {}
    FEATURE_FLAGS[version][feature] = True


def disable_feature(version: str, feature: str) -> None:
    """Disable a feature flag for a version."""
    if version in FEATURE_FLAGS:
        FEATURE_FLAGS[version][feature] = False
