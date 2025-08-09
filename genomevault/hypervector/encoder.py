"""
Compatibility shim for deprecated module path.

This module maintains backward compatibility for code using the old import path.
Please migrate to genomevault.hypervector_transform.encoding
"""

import warnings
from genomevault.hypervector_transform.encoding import *  # noqa: F401, F403

warnings.warn(
    "genomevault.hypervector.encoder is deprecated. "
    "Use genomevault.hypervector_transform.encoding instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export main classes for compatibility
from genomevault.hypervector_transform.encoding import (  # noqa: F401, E402
    HypervectorEncoder,
    HypervectorConfig,
)

__all__ = ["HypervectorEncoder", "HypervectorConfig"]
