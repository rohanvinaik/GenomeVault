#!/usr/bin/env python3
"""Comprehensive fix for all Black formatting errors."""

import os
import re
from pathlib import Path
from typing import List, Dict


def fix_file_indentation(file_path: str) -> bool:
    """Fix indentation issues in a Python file."""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        fixed_lines = []
        in_class = False
        in_method = False
        class_indent = 0
        method_indent = 0

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)

            # Handle empty lines
            if not stripped:
                fixed_lines.append(line)
                continue

            # Handle class definitions
            if re.match(r"^class\s+\w+", stripped):
                in_class = True
                in_method = False
                class_indent = current_indent
                fixed_lines.append(line)
                continue

            # Handle method definitions
            if re.match(r"^def\s+\w+", stripped) and in_class:
                in_method = True
                method_indent = class_indent + 4
                # Ensure method is properly indented
                fixed_line = " " * method_indent + stripped
                fixed_lines.append(fixed_line)
                continue

            # Handle __init__ and method bodies
            if in_method and i > 0 and lines[i - 1].strip().endswith(":"):
                # This is the first line of a method body
                body_indent = method_indent + 4

                # Special handling for docstrings
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    fixed_line = " " * body_indent + stripped
                    fixed_lines.append(fixed_line)
                    continue

                # Handle self.attribute = value
                if stripped.startswith("self."):
                    fixed_line = " " * body_indent + stripped
                    fixed_lines.append(fixed_line)
                    continue

            # Default: keep the line as is
            fixed_lines.append(line)

        # Write back the fixed content
        with open(file_path, "w") as f:
            f.writelines(fixed_lines)

        return True
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def fix_all_problematic_files():
    """Fix all files identified as problematic in the CI output."""

    # Map of files to their specific issues
    problematic_files = {
        "backup_problematic/analyze_and_fix_modules.py": "unexpected_indent",
        "backup_problematic/comprehensive_fixes.py": "unexpected_indent",
        "backup_problematic/fix_duplicate_functions.py": "unexpected_indent",
        "backup_problematic/fix_experimental_modules.py": "unindent_mismatch",
        "backup_problematic/generate_tailchasing_fixes.py": "unindent_mismatch",
        "backup_problematic/run_benchmark_fixed.py": "unindent_mismatch",
        "backup_problematic/run_benchmark_wrapper.py": "unindent_mismatch",
        "backup_problematic/run_tailchasing_fixes.py": "unindent_mismatch",
        "backup_problematic/safe_fixes.py": "unindent_mismatch",
        "backup_problematic/test_encoding.py": "unindent_mismatch",
        "backup_problematic/test_hamming_lut.py": "unindent_mismatch",
        "backup_problematic/variant_search.py": "unexpected_indent",
        "backup_problematic/variant_test_helper.py": "unindent_mismatch",
        "devtools/debug_genomevault.py": "unexpected_indent",
        "devtools/trace_import_failure.py": "unindent_mismatch",
        # Test files with docstring issues
        "tests/adversarial/test_hdc_adversarial.py": "docstring_indent",
        "tests/adversarial/test_pir_adversarial.py": "docstring_indent",
        "tests/adversarial/test_zk_adversarial.py": "docstring_indent",
        "tests/conftest.py": "docstring_indent",
        "tests/e2e/test_pir_e2e.py": "docstring_indent",
        "tests/e2e/test_zk_e2e.py": "docstring_indent",
        # Add more files as needed
    }

    print("Fixing problematic files...")

    for file_path, issue_type in problematic_files.items():
        full_path = Path(file_path)
        if full_path.exists():
            print(f"Fixing {file_path} ({issue_type})...")
            fix_file_indentation(str(full_path))
        else:
            print(f"Skipping {file_path} - not found")


def fix_test_docstrings():
    """Fix docstring indentation in test files."""
    test_patterns = ["tests/**/*.py", "genomevault/**/test_*.py"]

    for pattern in test_patterns:
        for test_file in Path(".").glob(pattern):
            try:
                with open(test_file, "r") as f:
                    content = f.read()

                # Fix docstrings that appear after function definitions
                # Pattern: def function():\n"""docstring"""
                content = re.sub(
                    r'(def\s+\w+\([^)]*\):\s*\n)(\s*)("""[^"]*"""|\'\'\'[^\']*\'\'\')',
                    r"\1    \3",
                    content,
                    flags=re.MULTILINE,
                )

                with open(test_file, "w") as f:
                    f.write(content)

            except Exception as e:
                print(f"Error fixing docstrings in {test_file}: {e}")


if __name__ == "__main__":
    # First fix all identified problematic files
    fix_all_problematic_files()

    # Then fix test docstrings
    fix_test_docstrings()

    print("\nFixes applied. Now running Black to format all files...")

    # Run Black with specific target version
    import subprocess

    result = subprocess.run(
        ["black", "--target-version", "py311", "."], capture_output=True, text=True
    )

    print("Black output:", result.stdout)
    if result.stderr:
        print("Black errors:", result.stderr)
