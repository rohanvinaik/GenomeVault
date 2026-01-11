"""
Hypervector Transform Module for GenomeVault

This module implements Hyperdimensional Computing (HDC) for
privacy-preserving genomic data encoding.

Primary Components:
- AdaptiveEncoder: Production encoder with adaptive k-selection (99.2% accuracy)
- AdaptiveSparseHadamardCodebook: The core codebook implementation
- DifficultyScorer: Adaptive k-selection based on sequence difficulty

Usage:
    from genomevault.hypervector_transform import AdaptiveEncoder

    encoder = AdaptiveEncoder()
    encoded, _ = encoder.encode_chunk(sequence, k=6)
    decoded = encoder.decode_chunk(encoded, k=6)
"""

# Primary production encoder (promoted from experimental)
from .adaptive_encoder import (
    AdaptiveEncoder,
    AdaptiveSparseHadamardCodebook,
    DifficultyScorer,
    GenomicDataLoader,
    EncodingConfig,
    run_production_encoding,
    sanity_check_encoding,
    # Constants
    D, N, STEP, N_BANKS, BASE_SEED,
    DIFFICULTY_THRESHOLDS,
)

# Binding operations (core functionality, no broken dependencies)
from .binding_operations import (
    BindingOperation,
    BindingType,
    HypervectorBinder,
    BindingOperations,
    bind,
    superpose,
    circular_bind,
    fourier_bind,
    protect_vector,
)

__all__ = [
    # Primary encoder
    "AdaptiveEncoder",
    "AdaptiveSparseHadamardCodebook",
    "DifficultyScorer",
    "GenomicDataLoader",
    "EncodingConfig",
    "run_production_encoding",
    "sanity_check_encoding",
    # Constants
    "D", "N", "STEP", "N_BANKS", "BASE_SEED",
    "DIFFICULTY_THRESHOLDS",
    # Binding
    "BindingOperation",
    "BindingOperations",
    "BindingType",
    "HypervectorBinder",
    "bind",
    "superpose",
    "circular_bind",
    "fourier_bind",
    "protect_vector",
]
