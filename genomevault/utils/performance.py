from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Callable, Tuple

def time_fn(fn: Callable, *args, **kwargs) -> Tuple[float, any]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return dt, out

@contextmanager
def time_block():
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0