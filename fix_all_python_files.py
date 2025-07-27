#!/usr/bin/env python3
"""
Fix all indentation issues in GenomeVault Python files automatically
"""

import os
import sys
from pathlib import Path
import subprocess


def fix_all_python_files(base_path):
    """Fix indentation in all Python files"""
    print("🔧 Fixing all Python files in GenomeVault...")

    # Install autopep8 if not available
    try:
        import autopep8
    except ImportError:
        print("📦 Installing autopep8...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "autopep8"])
        import autopep8

    fixed_count = 0
    error_count = 0

    # Find all Python files
    for py_file in base_path.rglob("*.py"):
        # Skip venv and cache
        if 'venv' in str(py_file) or '__pycache__' in str(py_file):
            continue

        try:
            # Read file
            with open(py_file, 'r', encoding='utf-8') as f:
                original = f.read()

            # Fix with autopep8
            fixed = autopep8.fix_code(
                original,
                options={
                    'aggressive': 2,  # More aggressive fixing
                    'max_line_length': 100,
                    'indent_size': 4,
                    'ignore': ['E501'],  # Ignore line too long
                }
            )

            # Write back if changed
            if fixed != original:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(fixed)
                fixed_count += 1
                print(f"  ✅ Fixed {py_file.relative_to(base_path)}")

        except Exception as e:
            error_count += 1
            print(f"  ❌ Error with {py_file.name}: {e}")

    print(f"\nSummary:")
    print(f"  Files fixed: {fixed_count}")
    print(f"  Errors: {error_count}")

    return fixed_count, error_count


def main():
    """Main function"""
    print("🚀 GenomeVault Complete Python Fixer")
    print("=" * 60)

    base_path = Path.home() / "genomevault"
    if not base_path.exists():
        print(f"❌ GenomeVault not found at {base_path}")
        return 1

    os.chdir(base_path)

    # Fix all Python files
    fixed, errors = fix_all_python_files(base_path)

    if fixed > 0:
        print(f"\n✨ Fixed {fixed} files!")

    print("\n🚀 Now trying to run the benchmark...")

    # Try to run the benchmark
    env = os.environ.copy()
    env['PYTHONPATH'] = str(base_path)

    result = subprocess.run(
        [sys.executable, "benchmarks/benchmark_packed_hypervector.py"],
        env=env,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("\n✅ Benchmark ran successfully!")
        print(result.stdout)
    else:
        print("\n❌ Benchmark failed:")
        print(result.stderr)

        # Try minimal benchmark as fallback
        print("\n🔄 Trying minimal benchmark...")
        result2 = subprocess.run(
            [sys.executable, "minimal_benchmark.py"],
            env=env,
            capture_output=True,
            text=True
        )

        if result2.returncode == 0:
            print("\n✅ Minimal benchmark works!")
            print(result2.stdout)
        else:
            print("\n❌ Even minimal benchmark failed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
