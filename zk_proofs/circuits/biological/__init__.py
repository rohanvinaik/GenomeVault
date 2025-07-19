"""
Biological Package
"""

# Too many exports in multi_omics, import module directly
from . import multi_omics

# Too many exports in variant, import module directly
from . import variant

__all__ = [
    'multi_omics',
    'variant',
]
