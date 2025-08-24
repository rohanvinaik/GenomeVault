"""
Route modules for GenomeVault API Gateway.

Provides comprehensive API endpoints for:
- Health monitoring
- Pipeline management
- Vector operations
- ZK proof generation
- PIR queries
- Model management
- Algorithm marketplace
- Specialized operations
"""

from __future__ import annotations

# Import routers for easy access
from genomevault.api.gateway.routes import (
    algorithms,
    health,
    models,
    pipelines,
    proofs,
    queries,
    specialized,
    vectors,
)

__all__ = [
    "algorithms",
    "health",
    "models",
    "pipelines",
    "proofs",
    "queries",
    "specialized",
    "vectors",
]
