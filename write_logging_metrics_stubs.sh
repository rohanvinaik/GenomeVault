#!/bin/bash
###############################################################################
#  write logging + metrics stubs  ➜  importable + satisfy unit tests
###############################################################################
/bin/sh <<'FIX'
set -e
mkdir -p genomevault/utils
# 1) utils/logging.py ────────────────────────────────────────────────────────
cat > genomevault/utils/logging.py <<'PY'
import logging
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"
def configure(level: int = logging.INFO) -> None:
    """Configure root logger exactly once."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=_FMT)
def get_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger pre-configured for GenomeVault."""
    configure(level)
    return logging.getLogger(name)
PY
echo "• utils/logging.py written"
# 2) utils/metrics.py ────────────────────────────────────────────────────────
cat > genomevault/utils/metrics.py <<'PY'
"""Tiny in-memory metrics helper so tests can import MetricsCollector /
MetricsContext / get_metrics without pulling in Prometheus etc."""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from typing import Generator
class MetricsCollector:
    def __init__(self) -> None:
        self._ctr: Counter[str] = Counter()
    def inc(self, name: str, value: int = 1) -> None:
        self._ctr[name] += value
    def get(self, name: str) -> int:
        return self._ctr[name]
_GLOBAL = MetricsCollector()
def get_metrics() -> MetricsCollector:
    return _GLOBAL
@contextmanager
def MetricsContext() -> Generator[MetricsCollector, None, None]:
    """Simple context manager used by some unit tests."""
    yield _GLOBAL
PY
echo "• utils/metrics.py written"
# 3) ensure utils/__init__.py re-exports these  ──────────────────────────────
cat > genomevault/utils/__init__.py <<'PY'
from .logging import get_logger
from .metrics import MetricsCollector, MetricsContext, get_metrics
__all__ = [
    "get_logger",
    "MetricsCollector",
    "MetricsContext",
    "get_metrics",
]
PY
echo "• utils/__init__.py rewritten"
# 4) wire get_logger into local_processing/sequencing.py  ────────────────────
if [ -f genomevault/local_processing/sequencing.py ]; then
    sed -i '' -E '1s/^/from genomevault.utils.logging import get_logger\n/' \
      genomevault/local_processing/sequencing.py
    echo "• added get_logger import to sequencing.py"
else
    echo "• sequencing.py not found, skipping import addition"
fi
# 5) format new files (requires black / isort)  ──────────────────────────────
python3 -m black genomevault/utils/logging.py genomevault/utils/metrics.py --quiet || true
python3 -m isort genomevault/utils/logging.py genomevault/utils/metrics.py --quiet || true
# 6) stage & commit  ─────────────────────────────────────────────────────────
git add genomevault/utils
if [ -f genomevault/local_processing/sequencing.py ]; then
    git add genomevault/local_processing/sequencing.py
fi
git commit -m "feat(utils): add logging & metrics stubs; wire get_logger in sequencing" || echo "Nothing to commit"
echo "✓ stubs committed — run:  pytest -q tests/unit/test_config.py"
FIX