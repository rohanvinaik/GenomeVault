"""GenomeVault API v1 routers."""

from __future__ import annotations

from .health import router as health_router
from .hv import router as hv_router
from .pir import router as pir_router
from .zk import router as zk_router
from .clinical import router as clinical_router

__all__ = [
    "health_router",
    "hv_router",
    "pir_router",
    "zk_router",
    "clinical_router",
]
