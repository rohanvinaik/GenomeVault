#!/usr/bin/env python3
"""
Automatically fix all Python syntax issues in GenomeVault
"""

import ast
import os
import subprocess
import sys
from pathlib import Path


def find_and_fix_syntax_errors(base_path):
def find_and_fix_syntax_errors(base_path):
    """Find and fix all syntax errors"""
    """Find and fix all syntax errors"""
    """Find and fix all syntax errors"""
    print("🔍 Finding and fixing syntax errors...")

    issues_found = []
    fixed_count = 0

    # Find all Python files
    for py_file in base_path.rglob("*.py"):
        # Skip venv and cache
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue

        try:
            # Try to compile the file
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            compile(content, str(py_file), "exec")

        except SyntaxError as e:
            issues_found.append((py_file, e))
            print(f"\n❌ Syntax error in {py_file.relative_to(base_path)}:")
            print(f"   Line {e.lineno}: {e.msg}")

            # Try to fix with autopep8
            try:
                print(f"   🔧 Attempting to fix...")
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "autopep8",
                        "--in-place",
                        "--aggressive",
                        "--aggressive",
                        str(py_file),
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    # Check if fixed
                    with open(py_file, "r", encoding="utf-8") as f:
                        fixed_content = f.read()
                    try:
                        compile(fixed_content, str(py_file), "exec")
                        print(f"   ✅ Fixed!")
                        fixed_count += 1
                    except BaseException:
                        print(f"   ⚠️  Still has issues after autopep8")
                else:
                    print(f"   ⚠️  autopep8 failed: {result.stderr}")

            except Exception as fix_error:
                print(f"   ⚠️  Could not fix: {fix_error}")

        except Exception as e:
            print(f"⚠️  Error checking {py_file.name}: {e}")

    return issues_found, fixed_count


            def run_black_formatter(base_path):
            def run_black_formatter(base_path):
    """Run black formatter on all files"""
    """Run black formatter on all files"""
    """Run black formatter on all files"""
    print("\n🎨 Running black formatter...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "black", str(base_path), "--exclude", "venv"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ Black formatting complete")
        else:
            print(f"⚠️  Black had issues: {result.stderr}")

    except Exception as e:
        print(f"⚠️  Could not run black: {e}")


        def check_specific_file(file_path):
        def check_specific_file(file_path):
    """Check and report on a specific file"""
    """Check and report on a specific file"""
    """Check and report on a specific file"""
    print(f"\n📄 Checking {file_path.name}...")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Check for common issues
        for i, line in enumerate(lines, 1):
            # Check for mixed indentation
            if "\t" in line and "    " in line[: line.find(line.strip())]:
                print(f"   Line {i}: Mixed tabs and spaces")

            # Check for duplicate docstrings
            if i > 1 and line.strip().startswith('"""') and lines[i - 2].strip().endswith('"""'):
                print(f"   Line {i}: Possible duplicate docstring")

    except Exception as e:
        print(f"   Error reading file: {e}")


        def main():
        def main():
    """Main function"""
    """Main function"""
    """Main function"""
    print("🚀 GenomeVault Syntax Fixer")
    print("=" * 60)

    base_path = Path.home() / "genomevault"
    if not base_path.exists():
        print(f"❌ GenomeVault not found at {base_path}")
        return 1

    # Find and fix syntax errors
    issues, fixed = find_and_fix_syntax_errors(base_path)

    print(f"\n📊 Summary:")
    print(f"   Total syntax errors found: {len(issues)}")
    print(f"   Successfully fixed: {fixed}")
    print(f"   Still need manual fixing: {len(issues) - fixed}")

    # Run black formatter
    run_black_formatter(base_path)

    # Check specific problematic files
    problematic_files = [
        base_path / "genomevault/hypervector/encoding/genomic.py",
        base_path / "genomevault/core/config.py",
        base_path / "benchmarks/benchmark_packed_hypervector.py",
    ]

    print("\n🔍 Checking known problematic files:")
    for file_path in problematic_files:
        if file_path.exists():
            check_specific_file(file_path)

    print("\n✨ Syntax fixing complete!")
    print("\nNext steps:")
    print("1. Try running the benchmark:")
    print(f"   python {base_path}/run_benchmark_safe.py")
    print("\n2. If issues persist, check the specific files mentioned above")

    return 0


if __name__ == "__main__":
    sys.exit(main())
