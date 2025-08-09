from __future__ import annotations

# Thin wrapper to keep imports stable; re-export main encoder
from .encoding import HypervectorEncoder, HypervectorConfig, ProjectionType

__all__ = ["HypervectorEncoder", "HypervectorConfig", "ProjectionType"]
