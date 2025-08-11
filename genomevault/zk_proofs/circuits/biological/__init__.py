"""Package initialization for biological."""

from . import multi_omics, variant

"""
Biological Package
"""

# Too many exports in variant, import module directly
# Too many exports in multi_omics, import module directly

__all__ = [
    "multi_omics",
    "variant",
]
