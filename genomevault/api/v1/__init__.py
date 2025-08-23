"""GenomeVault API v1 module.

This module provides version 1 of the GenomeVault API with comprehensive
privacy-preserving genomic computing capabilities.

Version: 1.0.0
Stability: Stable
Deprecation: None (current version)
"""

from __future__ import annotations

from .app import create_app
from .versioning import APIVersion, DeprecationWarning

__version__ = "1.0.0"
__api_version__ = APIVersion.V1
__deprecated__ = False

__all__ = ["create_app", "APIVersion", "DeprecationWarning"]
