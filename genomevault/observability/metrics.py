from __future__ import annotations
from typing import Dict
from genomevault.utils.metrics import Counter, Gauge, Histogram


class MetricsRegistry:
    def __init__(self) -> None:
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.hists: Dict[str, Histogram] = {}

    def counter(self, name: str) -> Counter:
        self.counters.setdefault(name, Counter())
        return self.counters[name]

    def gauge(self, name: str) -> Gauge:
        self.gauges.setdefault(name, Gauge())
        return self.gauges[name]

    def histogram(self, name: str) -> Histogram:
        self.hists.setdefault(name, Histogram())
        return self.hists[name]


REGISTRY = MetricsRegistry()
