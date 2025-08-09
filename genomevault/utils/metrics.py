from __future__ import annotations
from typing import Dict


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
