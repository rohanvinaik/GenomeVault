"""Tracing module for GenomeVault observability."""

from .opentelemetry import (
    get_tracing_manager,
    get_tracer,
    trace_hdc_operation,
    trace_zk_operation,
    trace_pir_operation,
    trace_operation,
    TracingManager,
    GenomeVaultAttributes
)

__all__ = [
    "get_tracing_manager",
    "get_tracer",
    "trace_hdc_operation",
    "trace_zk_operation", 
    "trace_pir_operation",
    "trace_operation",
    "TracingManager",
    "GenomeVaultAttributes"
]