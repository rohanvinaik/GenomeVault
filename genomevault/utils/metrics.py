from __future__ import annotations
from typing import Dict, Optional


class Counter:
    def __init__(self):
        self.v = 0

    def inc(self, n: int = 1) -> None:
        self.v += n

    def get(self) -> int:
        return self.v


class Gauge:
    def __init__(self):
        self.v = 0.0

    def set(self, x: float) -> None:
        self.v = float(x)

    def get(self) -> float:
        return self.v


class Histogram:
    def __init__(self):
        self.buckets: Dict[str, int] = {}

    def observe(self, value: float) -> None:
        # simple ms buckets
        ms = int(value * 1000)
        key = f"{(ms//10)*10}-{(ms//10)*10+9}ms"
        self.buckets[key] = self.buckets.get(key, 0) + 1


class MetricsCollector:
    """Collects and manages application metrics."""

    def __init__(self) -> None:
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}

    def counter(self, name: str) -> Counter:
        if name not in self.counters:
            self.counters[name] = Counter()
        return self.counters[name]

    def gauge(self, name: str) -> Gauge:
        if name not in self.gauges:
            self.gauges[name] = Gauge()
        return self.gauges[name]

    def histogram(self, name: str) -> Histogram:
        if name not in self.histograms:
            self.histograms[name] = Histogram()
        return self.histograms[name]

    def get_all_metrics(self) -> Dict[str, Dict[str, float | int | Dict[str, int]]]:
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
