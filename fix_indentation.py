#!/usr/bin/env python3
"""
Fix all indentation issues in GenomeVault Python files
"""

import os
import sys
from pathlib import Path


def fix_indentation_issues(base_path):
    """Fix indentation issues in all Python files"""
    print("🔧 Fixing indentation issues in all Python files...")

    try:
        import autopep8
    except ImportError:
        print("❌ autopep8 not installed yet")
        return

    fixed_count = 0
    error_count = 0

    for py_file in base_path.rglob("*.py"):
        # Skip venv and cache
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue

        try:
            # Read file
            with open(py_file, "r", encoding="utf-8") as f:
                original_content = f.read()

            # Use autopep8 to fix indentation
            fixed_content = autopep8.fix_code(
                original_content,
                options={
                    "aggressive": 1,
                    "max_line_length": 100,
                    "indent_size": 4,
                    "ignore": ["E501"],  # Ignore line too long
                },
            )

            # Check if content changed
            if fixed_content != original_content:
                # Write back
                with open(py_file, "w", encoding="utf-8") as f:
                    f.write(fixed_content)

                fixed_count += 1
                print(f"  ✅ Fixed {py_file.relative_to(base_path)}")

        except Exception as e:
            error_count += 1
            print(f"  ❌ Error with {py_file.name}: {e}")

    print(f"\nSummary:")
    print(f"  Files fixed: {fixed_count}")
    print(f"  Errors: {error_count}")


def install_autopep8():
    """Install autopep8 if not available"""
    try:
        import autopep8

        print("✅ autopep8 is already installed")
    except ImportError:
        print("📦 Installing autopep8...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "autopep8"])
        print("✅ autopep8 installed")


def main():
    """Main function"""
    print("🚀 GenomeVault Indentation Fixer")
    print("=" * 60)

    base_path = Path.home() / "genomevault"
    if not base_path.exists():
        print(f"❌ GenomeVault not found at {base_path}")
        return 1

    # Install autopep8
    install_autopep8()

    # Fix indentation
    fix_indentation_issues(base_path)

    print("\n✨ Indentation fixes complete!")
    print("\nNow try running the benchmark again:")
    print(f"  python {base_path}/run_benchmark_safe.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
