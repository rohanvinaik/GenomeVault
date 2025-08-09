"""
Lightweight package init.

Avoid importing heavy submodules (e.g., sequencing with pysam) at import time.
Import them lazily inside the functions that need them.
"""

from .common import process  # safe, lightweight export

__all__ = ["process"]
