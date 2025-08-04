#!/bin/bash
################################################################################
#  add minimal utils.logging & utils.metrics to satisfy imports
################################################################################
/bin/sh <<'PATCH'
set -e
mkdir -p genomevault/utils
# 1) logging helper -----------------------------------------------------------
cat > genomevault/utils/logging.py <<'PY'
import logging
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"

def configure_root(level: int = logging.INFO) -> None:
    """Configure the root logger once per process."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=_LOG_FORMAT)

def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger pre-configured for GenomeVault."""
    configure_root(level)
    return logging.getLogger(name)
PY
echo "✓  utils/logging.py written"

# 2) metrics stub -------------------------------------------------------------
cat > genomevault/utils/metrics.py <<'PY'
"""Very small stub so tests can import MetricsCollector / get_metrics.
Replace with a real implementation when observability package is ready.
"""
from collections import Counter
from typing import Any

class MetricsCollector:
    def __init__(self) -> None:
        self._counter: Counter[str] = Counter()
    
    def inc(self, name: str, value: int = 1) -> None:
        self._counter[name] += value
    
    def get(self, name: str) -> int:
        return self._counter[name]

def get_metrics() -> MetricsCollector:  # used by unit tests
    return _GLOBAL_METRICS

_GLOBAL_METRICS = MetricsCollector()
PY
echo "✓  utils/metrics.py stubbed"

# 3) import the helper in sequencing.py ---------------------------------------
if [ -f "genomevault/local_processing/sequencing.py" ]; then
    sed -i '' -E '1i\
from genomevault.utils.logging import get_logger
' genomevault/local_processing/sequencing.py
    echo "✓  added get_logger import to sequencing.py"
else
    echo "⚠️  sequencing.py not found, skipping import addition"
fi

# 4) run formatters just on new files -----------------------------------------
python3 -m black genomevault/utils/logging.py genomevault/utils/metrics.py --quiet || true
python3 -m isort genomevault/utils/logging.py genomevault/utils/metrics.py --quiet || true

# 5) stage & commit -----------------------------------------------------------
git add genomevault/utils/logging.py genomevault/utils/metrics.py
if [ -f "genomevault/local_processing/sequencing.py" ]; then
    git add genomevault/local_processing/sequencing.py
fi
git commit -m "feat(utils): add logging & metrics stubs; wire into sequencing" || echo "Nothing to commit"

echo "🟢  Stubs added & committed.  Re-run your test:"
echo "    pytest -q tests/unit/test_config.py"
PATCH