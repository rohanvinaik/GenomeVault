"""Core ZK proof primitives."""
from genomevault.zk_proofs.core.accumulator import (

    INITIAL_ACC,
    step,
    verify_chain,
)

__all__ = [
    "INITIAL_ACC",
    "step",
    "verify_chain",
]
