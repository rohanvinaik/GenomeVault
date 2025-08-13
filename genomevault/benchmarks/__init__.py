"""Module for benchmarks functionality."""

from .benchmark_hamming_lut import (
    standard_hamming_distance,
    standard_hamming_batch,
    benchmark_single_vector,
    benchmark_batch,
    benchmark_hdc_encoder,
    print_results,
    create_performance_plots,
    main,
    VECTOR_DIMENSIONS,
    BATCH_SIZES,
    NUM_TRIALS,
)

__all__ = [
    "BATCH_SIZES",
    "NUM_TRIALS",
    "VECTOR_DIMENSIONS",
    "benchmark_batch",
    "benchmark_hdc_encoder",
    "benchmark_single_vector",
    "create_performance_plots",
    "main",
    "print_results",
    "standard_hamming_batch",
    "standard_hamming_distance",
]
