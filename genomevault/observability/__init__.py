"""Module for observability functionality."""

# Import from the original logging.py file (not the directory)
from .logging import configure_logging, get_logger as get_basic_logger
from .metrics import MetricsRegistry, REGISTRY
from .otel import try_enable_otel
from .middleware import ObservabilityMiddleware, add_observability_middleware

# Import from new enhanced modules
try:
    from .metrics.prometheus import get_metrics_collector, get_prometheus_metrics
    from .logging.structured import get_structured_logger, configure_structured_logging
    from .tracing.opentelemetry import get_tracing_manager
    from .middleware.enhanced import (
        add_enhanced_observability_middleware,
        add_performance_timing_middleware
    )
except ImportError as e:
    # Fallback if enhanced modules aren't available
    print(f"Enhanced observability modules not available: {e}")

__all__ = [
    "MetricsRegistry",
    "ObservabilityMiddleware", 
    "REGISTRY",
    "add_observability_middleware",
    "configure_logging",
    "try_enable_otel",
    "get_basic_logger",
    # Enhanced functionality
    "get_metrics_collector",
    "get_prometheus_metrics",
    "get_structured_logger",
    "configure_structured_logging",
    "get_tracing_manager",
    "add_enhanced_observability_middleware",
    "add_performance_timing_middleware",
]
