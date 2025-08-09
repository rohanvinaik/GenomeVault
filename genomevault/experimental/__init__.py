"""
Experimental features for GenomeVault.

⚠️ WARNING: This module contains experimental features.
APIs may change without deprecation notices.
Use at your own risk in production environments.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "This module contains experimental features. "
    "APIs may change without deprecation notices. "
    "Use at your own risk in production environments.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "kan",
    "pir_advanced",
    "zk_circuits",
]

# Version tracking for experimental features
EXPERIMENTAL_VERSION = "0.1.0-alpha"


def is_experimental_enabled() -> bool:
    """Check if experimental features are enabled."""
    import os

    return os.environ.get("GENOMEVAULT_EXPERIMENTAL", "false").lower() == "true"


def require_experimental(feature_name: str) -> None:
    """
    Require experimental features to be explicitly enabled.

    Args:
        feature_name: Name of the experimental feature being accessed

    Raises:
        RuntimeError: If experimental features are not enabled
    """
    if not is_experimental_enabled():
        raise RuntimeError(
            f"Experimental feature '{feature_name}' requires explicit opt-in. "
            "Set environment variable GENOMEVAULT_EXPERIMENTAL=true to enable."
        )
