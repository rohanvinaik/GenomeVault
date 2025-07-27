#!/usr/bin/env python3
"""
Clean benchmark runner - bypasses virtual environment issues
"""

import sys
import os

# Use system Python packages instead of venv
sys.path = [p for p in sys.path if 'venv' not in p]

# Add project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Now import and run
try:
    # Import required packages using system installation
    import numpy as np
    import torch
    print("✓ NumPy and PyTorch imported from system")

    # Try importing matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported")
    except ImportError:
        print("⚠️ Matplotlib not available, will skip plotting")
        plt = None

    # Import memory_profiler if available
    try:
        from memory_profiler import profile
        print("✓ memory_profiler imported")
    except ImportError:
        print("⚠️ memory_profiler not available, creating dummy decorator")

        def profile(func):
            return func

    # Now run the actual benchmark
    print("\n" + "=" * 50)
    print("Running benchmark...")
    print("=" * 50 + "\n")

    # Import benchmark code
    benchmark_file = os.path.join(project_dir, "benchmarks", "benchmark_packed_hypervector.py")

    # Read and execute (to avoid import issues)
    with open(benchmark_file, 'r') as f:
        benchmark_code = f.read()

    # Create a namespace with required imports
    namespace = {
        '__name__': '__main__',
        '__file__': benchmark_file,
        'np': np,
        'numpy': np,
        'torch': torch,
        'plt': plt,
        'matplotlib': matplotlib if 'matplotlib' in locals() else None,
        'profile': profile,
        'time': __import__('time'),
    }

    # Execute the benchmark
    exec(benchmark_code, namespace)

except Exception as e:
    print(f"\n❌ Error running benchmark: {e}")
    import traceback
    traceback.print_exc()
