"""
GenomeVault API Gateway

A comprehensive FastAPI gateway implementing OpenAPI specifications for
privacy-preserving genomic computing platform.

Provides unified access to:
- Pipeline management
- Hypervector operations
- Zero-knowledge proof generation
- Private information retrieval
- Federated learning models
- Algorithm marketplace
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "GenomeVault Team"
__all__ = ["app"]

from genomevault.api.gateway.main import app
