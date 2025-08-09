"""
KAN module - compatibility shim to experimental.

This module has been moved to genomevault.experimental.kan
"""

import warnings

warnings.warn(
    "genomevault.kan has been moved to genomevault.experimental.kan. "
    "Please update your imports. This compatibility shim will be removed in v2.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from experimental for backward compatibility
from genomevault.experimental.kan import *  # noqa: F401, F403

__all__ = ["KANHDEncoder"]
