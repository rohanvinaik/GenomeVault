"""Core ZK proof primitives."""

from .accumulator import step, verify_chain, INITIAL_ACC

__all__ = [
    "INITIAL_ACC",
    "step",
    "verify_chain",
]
