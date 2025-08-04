"""Core constants for GenomeVault system."""

from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    """Node types in the blockchain network."""

    LIGHT = "light"
    FULL = "full"
    ARCHIVE = "archive"


# Use a single canonical mapping
NODE_CLASS_WEIGHT: dict[NodeType, int] = {
    NodeType.LIGHT: 1,
    NodeType.FULL: 2,
    NodeType.ARCHIVE: 3,
}

# Centralize allowed pipeline types
DEFAULT_PIPELINE_TYPES: tuple[str, ...] = (
    "ancestry",
    "prs",
    "pca",
    "pathway_enrichment",
)

# Hypervector dimensions for different tiers
HYPERVECTOR_DIMENSIONS = {
    "base": 10000,
    "medium": 15000,
    "high": 20000,
}

__all__ = [
    "DEFAULT_PIPELINE_TYPES",
    "HYPERVECTOR_DIMENSIONS",
    "NODE_CLASS_WEIGHT",
    "NodeType",
]
