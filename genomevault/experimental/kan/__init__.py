"""
Experimental KAN (Kolmogorov-Arnold Networks) implementation.

EXPERIMENTAL: This module is under active development.
The API may change significantly between versions.
"""
from typing import TYPE_CHECKING
import warnings
warnings.warn(
    "KAN networks are experimental. Performance and API stability not guaranteed.",
    FutureWarning,
    stacklevel=2,
)

if TYPE_CHECKING:
    from .hybrid import KANHDEncoder

__all__ = ["KANHDEncoder"]


# Lazy imports to avoid loading heavy dependencies unless needed
def __getattr__(name: str):
    if name == "KANHDEncoder":
        from .hybrid import KANHDEncoder

        return KANHDEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
