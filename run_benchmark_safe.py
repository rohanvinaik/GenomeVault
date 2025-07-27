#!/usr/bin/env python3
"""
Safe benchmark runner for GenomeVault
"""

import locale
import os
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.environ["PYTHONPATH"] = str(project_root)

# Set encoding
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

try:
    # Try to run benchmark
    from benchmarks.benchmark_packed_hypervector import main

    print("Running benchmark...")
    main()
except ImportError as e:
    print(f"Import error: {e}")
    print("\nTrying direct execution...")

    bench_file = project_root / "benchmarks" / "benchmark_packed_hypervector.py"
    if bench_file.exists():
        # Read and execute
        with open(bench_file, "r", encoding="utf-8") as f:
            code = f.read()
        exec(code, {"__name__": "__main__", "__file__": str(bench_file)})
except Exception as e:
    print(f"Error: {e}")
    print("\nPlease check that all dependencies are installed:")
    print("  pip install -r requirements.txt")
