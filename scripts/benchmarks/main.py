#!/usr/bin/env python3
"""Main benchmark runner - imports and runs all benchmarks."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import at module level to satisfy flake8 E402
from genomevault.benchmarks import run_all_benchmarks  # noqa: E402

if __name__ == "__main__":
    run_all_benchmarks()
