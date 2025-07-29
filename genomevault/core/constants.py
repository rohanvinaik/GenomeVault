from __future__ import annotations

# Hypervector dimension tiers
HYPERVECTOR_DIMENSIONS = {
    "base": 10000,
    "mid": 15000,
    "high": 20000,
}

# Defaults for encoding
DEFAULT_SEED = 42
DEFAULT_DENSITY = 0.1

__all__ = ["HYPERVECTOR_DIMENSIONS", "DEFAULT_SEED", "DEFAULT_DENSITY"]
