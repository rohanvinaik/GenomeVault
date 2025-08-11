from __future__ import annotations

"""Metrics module."""
"""Metrics module."""
from typing import Dict, Optional


class Counter:
    """Counter implementation."""

    def __init__(self):
        """Initialize instance."""
        self.v = 0

    def inc(self, n: int = 1) -> None:
        """Inc.

        Args:
            n: N.
        """
        self.v += n

    def get(self) -> int:
        """Get.

        Returns:
            Integer result.
        """
        return self.v


class Gauge:
    """Gauge implementation."""

    def __init__(self):
        """Initialize instance."""
        self.v = 0.0

    def set(self, x: float) -> None:
        """Set.

        Args:
            x: X.
        """
        self.v = float(x)

    def get(self) -> float:
        """Get.

        Returns:
            Float result.
        """
        return self.v


class Histogram:
    """Histogram implementation."""

    def __init__(self):
        """Initialize instance."""
        self.buckets: Dict[str, int] = {}

    def observe(self, value: float) -> None:
        """Observe.

        Args:
            value: Value to set.
        """
        # simple ms buckets
        ms = int(value * 1000)
        key = f"{(ms//10)*10}-{(ms//10)*10+9}ms"
        self.buckets[key] = self.buckets.get(key, 0) + 1


class MetricsCollector:
    """Collects and manages application metrics."""

    def __init__(self) -> None:
        """Initialize instance."""
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}

    def counter(self, name: str) -> Counter:
        """Counter.

        Args:
            name: Name.

        Returns:
            Counter instance.
        """
        if name not in self.counters:
            self.counters[name] = Counter()
        return self.counters[name]

    def gauge(self, name: str) -> Gauge:
        """Gauge.

        Args:
            name: Name.

        Returns:
            Gauge instance.
        """
        if name not in self.gauges:
            self.gauges[name] = Gauge()
        return self.gauges[name]

    def histogram(self, name: str) -> Histogram:
        """Histogram.

        Args:
            name: Name.

        Returns:
            Histogram instance.
        """
        if name not in self.histograms:
            self.histograms[name] = Histogram()
        return self.histograms[name]

    def get_all_metrics(self) -> Dict[str, Dict[str, float | int | Dict[str, int]]]:
        """Retrieve all metrics.

        Returns:
            The all metrics.
        """
        return {
            "counters": {name: c.get() for name, c in self.counters.items()},
            "gauges": {name: g.get() for name, g in self.gauges.items()},
            "histograms": {name: h.buckets for name, h in self.histograms.items()},
        }


# Global metrics instance
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get or create the global metrics collector instance."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
