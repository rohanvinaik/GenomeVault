"""Metrics module for GenomeVault observability."""

# Import enhanced metrics
from .prometheus import (
    get_metrics_collector,
    get_prometheus_metrics,
    get_metrics_content_type,
    MetricsCollector,
    GENOMEVAULT_REGISTRY,
)

# Import basic metrics from parent for compatibility
from typing import Dict

try:
    from genomevault.utils.metrics import Counter, Gauge, Histogram
except ImportError:
    # Fallback simple implementations
    class Counter:
        def __init__(self):
            self.value = 0

        def inc(self, amount=1):
            self.value += amount

    class Gauge:
        def __init__(self):
            self.value = 0

        def set(self, value):
            self.value = value

    class Histogram:
        def __init__(self):
            self.values = []

        def observe(self, value):
            self.values.append(value)


class MetricsRegistry:
    """MetricsRegistry implementation."""

    def __init__(self) -> None:
        """Initialize instance."""
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.hists: Dict[str, Histogram] = {}

    def counter(self, name: str) -> Counter:
        """Counter."""
        self.counters.setdefault(name, Counter())
        return self.counters[name]

    def gauge(self, name: str) -> Gauge:
        """Gauge."""
        self.gauges.setdefault(name, Gauge())
        return self.gauges[name]

    def histogram(self, name: str) -> Histogram:
        """Histogram."""
        self.hists.setdefault(name, Histogram())
        return self.hists[name]


REGISTRY = MetricsRegistry()

__all__ = [
    "get_metrics_collector",
    "get_prometheus_metrics",
    "get_metrics_content_type",
    "MetricsCollector",
    "GENOMEVAULT_REGISTRY",
    "MetricsRegistry",
    "REGISTRY",
    "Counter",
    "Gauge",
    "Histogram",
]
