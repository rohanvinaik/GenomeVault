"""Performance module."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Tuple
import time
def time_fn(fn: Callable, *args, **kwargs) -> Tuple[float, any]:
    """Time fn.

    Args:
        fn: Fn.

    Returns:
        Operation result.
    """
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return dt, out


@contextmanager
def time_block():
    """Time block."""
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0
