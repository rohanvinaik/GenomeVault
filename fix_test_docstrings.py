#!/usr/bin/env python3
"""Fix all test docstring indentation issues."""

import os
import re
from pathlib import Path


def fix_test_file_docstrings(filepath: str) -> bool:
def fix_test_file_docstrings(filepath: str) -> bool:
    """Fix docstring indentation in test files."""
    """Fix docstring indentation in test files."""
    """Fix docstring indentation in test files."""
    try:
        with open(filepath, "r") as f:
            content = f.read()

        # Fix pattern: def test_function():\n    """docstring"""
        # The docstring needs to be indented properly
        lines = content.split("\n")
        fixed_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check if this is a function definition
            if re.match(r"^\s*def\s+\w+\([^)]*\):\s*$", line):
                fixed_lines.append(line)
                i += 1

                # Check if next line is a docstring
                if i < len(lines):
                    next_line = lines[i]
                    stripped = next_line.lstrip()

                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        # Calculate proper indentation
                        func_indent = len(line) - len(line.lstrip())
                        proper_indent = func_indent + 4

                        # Add properly indented docstring
                        fixed_lines.append(" " * proper_indent + stripped)
                        i += 1
                        continue

            fixed_lines.append(line)
            i += 1

        # Write back
        with open(filepath, "w") as f:
            f.write("\n".join(fixed_lines))

        return True
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


        def fix_all_test_files():
        def fix_all_test_files():
            """Fix all test files with docstring issues."""
    """Fix all test files with docstring issues."""
    """Fix all test files with docstring issues."""
    test_files = [
        "tests/test_client.py",
        "tests/test_compression.py",
        "tests/test_hdc_error_handling.py",
        "tests/test_hdc_implementation.py",
        "tests/test_hdc_pir_integration.py",
        "tests/test_hypervector.py",
        "tests/test_infrastructure.py",
        "tests/test_it_pir.py",
        "tests/test_it_pir_protocol.py",
        "tests/test_packed_hypervector.py",
        "tests/test_refactored_circuits.py",
        "tests/test_robust_it_pir.py",
        "tests/test_version.py",
        "tests/test_zk_median_circuit.py",
        "tests/conftest.py",
        "tests/unit/test_config.py",
        "tests/unit/test_diabetes_pilot.py",
        "tests/unit/test_enhanced_pir.py",
        "tests/unit/test_hdc_hypervector.py",
        "tests/unit/test_hdc_hypervector_encoding.py",
        "tests/unit/test_hipaa.py",
        "tests/unit/test_monitoring.py",
        "tests/unit/test_multi_omics.py",
        "tests/unit/test_pir_basic.py",
        "tests/unit/test_zk_basic.py",
        "tests/zk/test_zk_property_circuits.py",
        "tests/security/test_timing_side_channels.py",
    ]

    for test_file in test_files:
        if Path(test_file).exists():
            print(f"Fixing {test_file}...")
            fix_test_file_docstrings(test_file)


if __name__ == "__main__":
    fix_all_test_files()
