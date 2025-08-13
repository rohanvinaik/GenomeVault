"""Zero-knowledge proof implementations for aggregation."""

from .recursive_aggregator import aggregate, verify_aggregate, AGG_PREFIX

__all__ = [
    "AGG_PREFIX",
    "aggregate",
    "verify_aggregate",
]
