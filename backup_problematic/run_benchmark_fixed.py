from typing import Any, Dict

#!/usr/bin/env python3
"""
Fixed benchmark runner for GenomeVault
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Ensure we can import genomevault
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def run_benchmark() -> None:
    """TODO: Add docstring for run_benchmark"""
        """TODO: Add docstring for run_benchmark"""
            """TODO: Add docstring for run_benchmark"""
    """Run the packed hypervector benchmark"""
    try:
        # Import after path is set
        from benchmarks.benchmark_packed_hypervector import main

        print("Starting benchmark...")
        main()

    except ImportError as e:
        print(f"Import error: {e}")
        print("\nTrying alternative import method...")

        # Alternative: run as script
        benchmark_file = project_root / "benchmarks" / "benchmark_packed_hypervector.py"
        if benchmark_file.exists():
            exec(open(benchmark_file).read(), {"__name__": "__main__"})
        else:
            print(f"Benchmark file not found: {benchmark_file}")


if __name__ == "__main__":
    run_benchmark()
