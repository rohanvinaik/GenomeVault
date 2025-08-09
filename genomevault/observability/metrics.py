from __future__ import annotations

"""Metrics module."""
"""Metrics module."""
from typing import Dict

from genomevault.utils.metrics import Counter, Gauge, Histogram


class MetricsRegistry:
    """MetricsRegistry implementation."""
    def __init__(self) -> None:
        """Initialize instance.
            """
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.hists: Dict[str, Histogram] = {}

    def counter(self, name: str) -> Counter:
        """Counter.

            Args:
                name: Name.

            Returns:
                Counter instance.
            """
        self.counters.setdefault(name, Counter())
        return self.counters[name]

    def gauge(self, name: str) -> Gauge:
        """Gauge.

            Args:
                name: Name.

            Returns:
                Gauge instance.
            """
        self.gauges.setdefault(name, Gauge())
        return self.gauges[name]

    def histogram(self, name: str) -> Histogram:
        """Histogram.

            Args:
                name: Name.

            Returns:
                Histogram instance.
            """
        self.hists.setdefault(name, Histogram())
        return self.hists[name]


REGISTRY = MetricsRegistry()
