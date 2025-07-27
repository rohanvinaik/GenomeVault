#!/usr/bin/env python3
"""
Batch fix for all Python files in test suite and experiments.
Fixes docstring formatting and runs basic linters.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Set


def fix_docstring_formatting(filepath: Path) -> bool:
    """Fix docstring formatting issues."""
    try:
        with open(filepath, "r") as f:
            content = f.read()

        original = content

        # Fix pattern: def function() -> Type:     """docstring"""
        # Should be: def function() -> Type:\n    """docstring"""
        pattern = r'(def \w+\([^)]*\)(?:\s*->\s*[^:]+)?:)\s+"""'
        replacement = r'\1\n    """'
        content = re.sub(pattern, replacement, content)

        # Fix pattern: class ClassName:     """docstring"""
        pattern = r'(class \w+(?:\([^)]*\))?:)\s+"""'
        replacement = r'\1\n    """'
        content = re.sub(pattern, replacement, content)

        if content != original:
            with open(filepath, "w") as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


def fix_all_files():
    """Fix all Python files in test and experiment directories."""
    directories = [
        Path("/Users/rohanvinaik/genomevault/tests"),
        Path("/Users/rohanvinaik/genomevault/experiments"),
        Path("/Users/rohanvinaik/experiments"),
    ]

    fixed_files = []

    for directory in directories:
        if not directory.exists():
            continue

        for py_file in directory.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            if fix_docstring_formatting(py_file):
                fixed_files.append(py_file)
                print(f"✅ Fixed: {py_file.name}")

    return fixed_files


def run_autopep8_on_files(files: List[Path]):
    """Run autopep8 on the given files."""
    for filepath in files:
        try:
            subprocess.run(
                [
                    "python3",
                    "-m",
                    "autopep8",
                    "--in-place",
                    "--aggressive",
                    "--aggressive",
                    "--max-line-length",
                    "100",
                    str(filepath),
                ],
                check=True,
            )
            print(f"  🔧 autopep8: {filepath.name}")
        except Exception as e:
            print(f"  ❌ autopep8 error on {filepath.name}: {e}")


def main():
    print("🔧 Fixing Python files in test suite and experiments...\n")

    # First, fix docstring formatting
    print("Step 1: Fixing docstring formatting...")
    fixed_files = fix_all_files()
    print(f"\nFixed {len(fixed_files)} files\n")

    # Then run autopep8
    if fixed_files:
        print("Step 2: Running autopep8 on fixed files...")
        run_autopep8_on_files(fixed_files)

    print("\n✨ Done!")


if __name__ == "__main__":
    main()
