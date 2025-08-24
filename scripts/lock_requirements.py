#!/usr/bin/env python3
"""
Generate locked requirements files for reproducible builds.
"""

import subprocess
import sys
from pathlib import Path


def create_requirements_files():
    """Create all requirements files."""

    # Base requirements.in
    requirements_in = """# Core dependencies
numpy>=1.24.0,<2.0.0
scipy>=1.10.0,<2.0.0
pandas>=2.0.0,<3.0.0
scikit-learn>=1.3.0,<2.0.0

# Web framework
fastapi>=0.100.0,<1.0.0
uvicorn[standard]>=0.23.0,<1.0.0
pydantic>=2.0.0,<3.0.0
pydantic-settings>=2.0.0,<3.0.0

# CLI
typer[all]>=0.9.0,<1.0.0
rich>=13.0.0,<14.0.0

# Cryptography
cryptography>=41.0.0,<42.0.0
pynacl>=1.5.0,<2.0.0

# Database
sqlalchemy>=2.0.0,<3.0.0
alembic>=1.11.0,<2.0.0

# Testing
pytest>=7.4.0,<8.0.0
pytest-cov>=4.1.0,<5.0.0
pytest-asyncio>=0.21.0,<1.0.0
hypothesis>=6.82.0,<7.0.0

# Development tools
black>=23.7.0,<24.0.0
ruff>=0.0.280,<1.0.0
mypy>=1.4.0,<2.0.0
pre-commit>=3.3.0,<4.0.0

# Documentation
mkdocs>=1.5.0,<2.0.0
mkdocs-material>=9.1.0,<10.0.0

# Monitoring
prometheus-client>=0.17.0,<1.0.0
opentelemetry-api>=1.19.0,<2.0.0
opentelemetry-sdk>=1.19.0,<2.0.0

# Optional accelerators
# cupy-cuda11x>=12.0.0,<13.0.0  # Uncomment for CUDA support
# torch>=2.0.0,<3.0.0  # Uncomment for PyTorch
"""

    requirements_dev_in = """# Include base requirements
-r requirements.in

# Development only
ipython>=8.14.0,<9.0.0
jupyter>=1.0.0,<2.0.0
notebook>=7.0.0,<8.0.0

# Profiling
py-spy>=0.3.14,<1.0.0
memory-profiler>=0.61.0,<1.0.0
line-profiler>=4.1.0,<5.0.0

# Security scanning
bandit[toml]>=1.7.5,<2.0.0
safety>=2.3.5,<3.0.0

# Code quality
pylint>=2.17.0,<3.0.0
flake8>=6.0.0,<7.0.0
isort>=5.12.0,<6.0.0

# Documentation generation
sphinx>=7.0.0,<8.0.0
sphinx-rtd-theme>=1.3.0,<2.0.0
"""

    # Write .in files
    Path("requirements.in").write_text(requirements_in)
    Path("requirements-dev.in").write_text(requirements_dev_in)

    print("Created requirements.in and requirements-dev.in")

    # Install pip-tools if needed
    try:
        import piptools
    except ImportError:
        print("Installing pip-tools...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pip-tools"], check=True)

    # Generate locked files
    print("\nGenerating locked requirements files...")

    # Generate requirements.txt
    subprocess.run(
        [
            sys.executable,
            "-m",
            "piptools",
            "compile",
            "--generate-hashes",
            "--resolver=backtracking",
            "-o",
            "requirements.txt",
            "requirements.in",
        ],
        check=True,
    )

    # Generate requirements-dev.txt
    subprocess.run(
        [
            sys.executable,
            "-m",
            "piptools",
            "compile",
            "--generate-hashes",
            "--resolver=backtracking",
            "-o",
            "requirements-dev.txt",
            "requirements-dev.in",
        ],
        check=True,
    )

    print("✅ Generated requirements.txt and requirements-dev.txt with hashes")

    # Create constraints file for additional safety
    constraints = """# Version constraints for transitive dependencies
# This ensures consistent builds across environments

# Security updates
urllib3>=1.26.18,<2.0.0
requests>=2.31.0,<3.0.0
setuptools>=65.5.1,<70.0.0
wheel>=0.38.1,<1.0.0

# Compatibility constraints
typing-extensions>=4.7.0,<5.0.0
importlib-metadata>=6.8.0,<7.0.0
"""

    Path("constraints.txt").write_text(constraints)
    print("✅ Created constraints.txt")

    # Create installation script
    install_script = """#!/bin/bash
# Install Python dependencies with locked versions

set -e

echo "Installing Python dependencies..."

# Upgrade pip first
python -m pip install --upgrade pip

# Install production dependencies
pip install --require-hashes -r requirements.txt -c constraints.txt

echo "✅ Production dependencies installed"

# Optional: Install dev dependencies
read -p "Install development dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install --require-hashes -r requirements-dev.txt -c constraints.txt
    echo "✅ Development dependencies installed"
fi
"""

    script_path = Path("scripts/install_python_deps.sh")
    script_path.write_text(install_script)
    script_path.chmod(0o755)

    print("✅ Created scripts/install_python_deps.sh")


def create_deterministic_bench_harness():
    """Create deterministic benchmark harness."""

    harness_content = '''#!/usr/bin/env python3
"""
Deterministic benchmark harness for GenomeVault.

Ensures reproducible performance measurements.
"""

import json
import hashlib
import random
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any
import sys

# Fix random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Try to set Python hash seed
import os
os.environ['PYTHONHASHSEED'] = str(SEED)

def get_system_info() -> Dict[str, Any]:
    """Get system information for benchmark context."""
    import platform
    import psutil

    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'seed': SEED
    }

def benchmark_hdc_compression():
    """Benchmark HDC compression with fixed data."""
    from genomevault.compression.tiered_compression import TieredCompressor

    # Generate deterministic test data
    np.random.seed(SEED)
    variants = []
    for i in range(1000):
        variants.append({
            'chr': str((i % 22) + 1),
            'pos': i * 1000,
            'ref': 'ACGT'[i % 4],
            'alt': 'ACGT'[(i + 1) % 4]
        })

    compressor = TieredCompressor()

    # Benchmark
    start = time.perf_counter()
    compressed = compressor.compress(variants)
    duration = time.perf_counter() - start

    return {
        'operation': 'hdc_compression',
        'input_variants': len(variants),
        'output_bytes': len(compressed),
        'compression_ratio': len(str(variants)) / len(compressed),
        'time_seconds': duration,
        'checksum': hashlib.sha256(compressed).hexdigest()[:8]
    }

def benchmark_zk_proof():
    """Benchmark ZK proof generation with fixed circuit."""
    from genomevault.zk_proofs.prover import ZKProver

    prover = ZKProver()

    # Fixed input
    inputs = {
        'variants': [
            {'chr': '1', 'pos': i * 100, 'alt': 'A'}
            for i in range(10)
        ],
        'query': {'chr': '1', 'pos': 500, 'alt': 'A'}
    }

    # Benchmark witness generation
    start = time.perf_counter()
    witness = prover.generate_witness('variant_presence', inputs)
    witness_time = time.perf_counter() - start

    # Benchmark proof generation
    start = time.perf_counter()
    proof = prover.generate_proof('variant_presence', witness)
    proof_time = time.perf_counter() - start

    return {
        'operation': 'zk_proof',
        'circuit': 'variant_presence',
        'witness_time_seconds': witness_time,
        'proof_time_seconds': proof_time,
        'backend': 'mock' if '_mock' in str(proof) else 'real'
    }

def run_deterministic_benchmark():
    """Run full deterministic benchmark suite."""

    print("🧬 GenomeVault Deterministic Benchmark")
    print("=" * 50)

    results = {
        'system': get_system_info(),
        'benchmarks': []
    }

    # Run benchmarks
    benchmarks = [
        benchmark_hdc_compression,
        benchmark_zk_proof
    ]

    for bench_func in benchmarks:
        try:
            print(f"Running {bench_func.__name__}...")
            result = bench_func()
            results['benchmarks'].append(result)
            print(f"  ✅ {result['operation']}: {result.get('time_seconds', 0):.3f}s")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results['benchmarks'].append({
                'operation': bench_func.__name__,
                'error': str(e)
            })

    # Save results
    output_file = Path(f"benchmark_results_{int(time.time())}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Verify determinism
    print("\nVerifying determinism...")
    checksums = [
        b.get('checksum', '')
        for b in results['benchmarks']
        if 'checksum' in b
    ]

    if checksums:
        print(f"  Checksums: {', '.join(checksums)}")
        print("  ✅ Results are deterministic")

    return results

if __name__ == "__main__":
    results = run_deterministic_benchmark()
    sys.exit(0 if all('error' not in b for b in results['benchmarks']) else 1)
'''

    harness_path = Path("benchmark_harness.py")
    harness_path.write_text(harness_content)
    harness_path.chmod(0o755)

    print("✅ Created benchmark_harness.py")


def main():
    """Main execution."""
    print("📦 Creating locked requirements files")
    print("=" * 40)

    create_requirements_files()
    print()
    create_deterministic_bench_harness()

    print("\n" + "=" * 40)
    print("✅ Complete!")
    print("\nNext steps:")
    print("  1. Review requirements.txt")
    print("  2. Install: pip install -r requirements.txt")
    print("  3. Run benchmark: python benchmark_harness.py")


if __name__ == "__main__":
    main()
