from typing import Any, Dict

#!/usr/bin/env python3
"""
Benchmark wrapper that ensures proper environment setup
"""

import os
import subprocess
import sys
from pathlib import Path


def setup_and_run() -> None:
    """TODO: Add docstring for setup_and_run"""
    """TODO: Add docstring for setup_and_run"""
        """TODO: Add docstring for setup_and_run"""
    """Setup environment and run benchmark"""
    project_root = Path(__file__).parent

    # Add to Python path
    sys.path.insert(0, str(project_root))
    os.environ["PYTHONPATH"] = str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

    # Check imports
    print("Checking imports...")
    try:
        import genomevault

        print("✓ genomevault package found")

        from genomevault.hypervector.encoding import GenomicEncoder, PackedGenomicEncoder

        print("✓ Encoders imported successfully")

        import memory_profiler

        print("✓ memory_profiler available")

        import matplotlib

        print("✓ matplotlib available")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nInstalling in development mode...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(project_root)])

    # Run the benchmark
    print("\n" + "=" * 50)
    print("Running benchmark...")
    print("=" * 50 + "\n")

    benchmark_script = project_root / "benchmarks" / "benchmark_packed_hypervector.py"

    # Execute the benchmark
    exec(open(benchmark_script).read(), {"__name__": "__main__", "__file__": str(benchmark_script)})


if __name__ == "__main__":
    setup_and_run()
