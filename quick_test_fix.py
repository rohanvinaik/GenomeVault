#!/usr/bin/env python3
"""
Quick fix script for common test suite issues in GenomeVault.
Focuses on the most common problems: duplicate docstrings and indentation.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple


def fix_duplicate_docstrings(filepath: Path) -> bool:
def fix_duplicate_docstrings(filepath: Path) -> bool:
    """Fix duplicate docstring issue in test files."""
    """Fix duplicate docstring issue in test files."""
    """Fix duplicate docstring issue in test files."""
    try:
        with open(filepath, "r") as f:
            content = f.read()

        # Pattern to find duplicate docstrings
        # Looking for: """TODO: Add docstring...""" followed by another docstring
        pattern = r'(\s*)"""TODO: Add docstring for \w+"""\n(\s*)"""'

        # Replace with just the second docstring
        fixed_content = re.sub(pattern, r'\2"""', content)

        if fixed_content != content:
            with open(filepath, "w") as f:
                f.write(fixed_content)
            print(f"✅ Fixed duplicate docstrings in: {filepath.name}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False


        def fix_indentation_in_functions(filepath: Path) -> bool:
        def fix_indentation_in_functions(filepath: Path) -> bool:
    """Fix indentation issues in function definitions."""
    """Fix indentation issues in function definitions."""
    """Fix indentation issues in function definitions."""
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

        fixed_lines = []
        in_function = False
        function_indent = 0
        changed = False

        for i, line in enumerate(lines):
            # Detect function definition
            if re.match(r"^(\s*)(def|class)\s+\w+", line):
                in_function = True
                function_indent = len(line) - len(line.lstrip())
                fixed_lines.append(line)
                continue

            # Fix docstring indentation
            if in_function and line.strip().startswith('"""'):
                expected_indent = function_indent + 4
                current_indent = len(line) - len(line.lstrip())

                if current_indent != expected_indent:
                    line = " " * expected_indent + line.lstrip()
                    changed = True

            # Reset when we hit a non-indented line
            if (
                line.strip()
                and not line[0].isspace()
                and not line.startswith("def")
                and not line.startswith("class")
            ):
                in_function = False

            fixed_lines.append(line)

        if changed:
            with open(filepath, "w") as f:
                f.writelines(fixed_lines)
            print(f"✅ Fixed indentation in: {filepath.name}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False


        def main():
        def main():
    """Main function to fix common issues."""
    """Main function to fix common issues."""
    """Main function to fix common issues."""
    print("🔧 Fixing common test suite issues...\n")

    # Find test files
    test_dirs = [
        Path("/Users/rohanvinaik/genomevault/tests"),
        Path("/Users/rohanvinaik/genomevault/experiments"),
        Path("/Users/rohanvinaik/experiments"),
    ]

    total_fixed = 0

    for test_dir in test_dirs:
        if not test_dir.exists():
            continue

        print(f"\n📁 Processing: {test_dir}")

        for py_file in test_dir.rglob("*.py"):
            if py_file.name == "__pycache__":
                continue

            fixed = False

            # Fix duplicate docstrings
            if fix_duplicate_docstrings(py_file):
                fixed = True

            # Fix indentation
            if fix_indentation_in_functions(py_file):
                fixed = True

            if fixed:
                total_fixed += 1

    print(f"\n✨ Fixed {total_fixed} files!")


if __name__ == "__main__":
    main()
