"""
Biological Package
"""

# Too many exports in variant, import module directly
# Too many exports in multi_omics, import module directly
from . import multi_omics, variant

__all__ = [
    "multi_omics",
    "variant",
]
