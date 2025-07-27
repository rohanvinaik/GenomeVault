#!/usr/bin/env python3
"""
Fix virtual environment and run benchmark
"""

import os
import subprocess
import sys
from pathlib import Path


def fix_venv_issues():
    """Fix virtual environment issues"""
    print("🔧 Fixing virtual environment issues...")

    base_path = Path.home() / "genomevault"
    venv_path = base_path / "venv"

    # Remove problematic .pth files
    pth_files = [
        venv_path / "lib/python3.9/site-packages/__editable__.genomevault-3.0.0.pth",
        venv_path / "lib/python3.9/site-packages/distutils-precedence.pth",
    ]

    for pth_file in pth_files:
        if pth_file.exists():
            print(f"  Removing problematic file: {pth_file.name}")
            pth_file.unlink()

    print("  ✅ Cleaned up .pth files")


def create_clean_runner():
    """Create a clean benchmark runner that bypasses venv issues"""
    base_path = Path.home() / "genomevault"

    runner_content = '''#!/usr/bin/env python3
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
    print("\\n" + "="*50)
    print("Running benchmark...")
    print("="*50 + "\\n")

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
    print(f"\\n❌ Error running benchmark: {e}")
    import traceback
    traceback.print_exc()
'''

    runner_path = base_path / "clean_benchmark_runner.py"
    runner_path.write_text(runner_content)
    runner_path.chmod(0o755)
    print(f"\n✅ Created clean runner: {runner_path}")
    return runner_path


def use_system_python_benchmark():
    """Run benchmark using system Python"""
    base_path = Path.home() / "genomevault"

    print("\n🐍 Using system Python to run benchmark...")

    # Deactivate venv if active
    if "VIRTUAL_ENV" in os.environ:
        print("  Deactivating virtual environment...")
        del os.environ["VIRTUAL_ENV"]
        # Remove venv from PATH
        path_parts = os.environ["PATH"].split(":")
        os.environ["PATH"] = ":".join([p for p in path_parts if "venv" not in p])

    # Install required packages in user space if needed
    print("\n📦 Checking system packages...")
    packages = ["numpy", "torch", "matplotlib", "memory-profiler"]

    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✓ {package} available")
        except ImportError:
            print(f"  📥 Installing {package} in user space...")
            subprocess.run(
                ["/usr/bin/python3", "-m", "pip", "install", "--user", package], capture_output=True
            )

    # Run the clean benchmark
    runner = create_clean_runner()
    print(f"\n🚀 Running benchmark with clean runner...")

    result = subprocess.run(
        ["/usr/bin/python3", str(runner)], cwd=str(base_path), capture_output=True, text=True
    )

    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)

    return result.returncode == 0


def main():
    """Main function"""
    print("🛠️ GenomeVault Benchmark Fixer")
    print("=" * 60)

    base_path = Path.home() / "genomevault"
    os.chdir(base_path)

    # Fix venv issues
    fix_venv_issues()

    # Try using system Python
    success = use_system_python_benchmark()

    if success:
        print("\n✅ Benchmark completed successfully!")
    else:
        print("\n❌ Benchmark failed")
        print("\nAlternative: Use the minimal benchmark:")
        print(f"  python3 {base_path}/minimal_benchmark.py")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
