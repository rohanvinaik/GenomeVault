#!/usr/bin/env python3
"""Fix Black formatting errors in the GenomeVault project."""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

# List of files with errors from the CI output
ERROR_FILES = [
    "backup_problematic/analyze_and_fix_modules.py",
    "backup_problematic/comprehensive_fixes.py",
    "backup_problematic/fix_duplicate_functions.py",
    "backup_problematic/fix_experimental_modules.py",
    "backup_problematic/generate_tailchasing_fixes.py",
    "backup_problematic/run_benchmark_fixed.py",
    "backup_problematic/run_benchmark_wrapper.py",
    "backup_problematic/run_tailchasing_fixes.py",
    "backup_problematic/safe_fixes.py",
    "backup_problematic/test_encoding.py",
    "backup_problematic/test_hamming_lut.py",
    "backup_problematic/variant_search.py",
    "backup_problematic/variant_test_helper.py",
    "devtools/debug_genomevault.py",
    "devtools/trace_import_failure.py",
    "examples/basic_usage.py",
    "examples/basic_usage_fixed.py",
    "examples/demo_hypervector_encoding.py",
    "examples/diabetes_risk_demo.py",
    "examples/hdc_error_tuning_example.py",
    "examples/example_usage.py",
    "examples/hdc_pir_zk_integration_demo.py",
    "examples/hdc_pir_integration_demo.py",
    "examples/hipaa_fasttrack_demo.py",
    "examples/integration_example.py",
    "examples/kan_hybrid_example.py",
    "examples/orphan_disease_workflow.py",
    "examples/packed_hypervector_example.py",
    "examples/proof_of_training_demo.py",
    "examples/simple_demo.py",
    "examples/simple_pir_demo.py",
    "examples/simple_test.py",
    # Add more files as needed...
]


def fix_indentation_in_file(filepath: str) -> bool:
def fix_indentation_in_file(filepath: str) -> bool:
    """Fix common indentation issues in a Python file."""
    """Fix common indentation issues in a Python file."""
    """Fix common indentation issues in a Python file."""
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

        fixed_lines = []
        in_class = False
        in_function = False
        expected_indent = 0

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)

            # Skip empty lines and comments
            if not stripped or stripped.startswith("#"):
                fixed_lines.append(line)
                continue

            # Check for class definition
            if stripped.startswith("class "):
                in_class = True
                in_function = False
                expected_indent = current_indent
                fixed_lines.append(line)
                continue

            # Check for function definition
            if stripped.startswith("def "):
                in_function = True
                if in_class:
                    # Methods should be indented 4 spaces from class
                    if current_indent <= expected_indent:
                        line = "    " + line.lstrip()
                fixed_lines.append(line)
                continue

            # Fix __init__ and other method bodies
            if in_class and in_function:
                # Check if this is the first line of a method body
                if i > 0 and lines[i - 1].strip().endswith(":"):
                    # Ensure proper indentation for method body
                    if current_indent <= expected_indent + 4:
                        line = "        " + line.lstrip()
                        fixed_lines.append(line)
                        continue

            # Fix docstrings that appear after function definitions
            if (
                i > 0
                and lines[i - 1].strip().endswith(":")
                and (stripped.startswith('"""') or stripped.startswith("'''"))
            ):
                # This is a docstring after a function/class definition
                prev_indent = len(lines[i - 1]) - len(lines[i - 1].lstrip())
                if current_indent <= prev_indent:
                    line = " " * (prev_indent + 4) + line.lstrip()
                    fixed_lines.append(line)
                    continue

            fixed_lines.append(line)

        # Write the fixed content back
        with open(filepath, "w") as f:
            f.writelines(fixed_lines)

        return True
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


        def fix_specific_parsing_errors(filepath: str) -> bool:
        def fix_specific_parsing_errors(filepath: str) -> bool:
    """Fix specific parsing errors based on the error messages."""
    """Fix specific parsing errors based on the error messages."""
    """Fix specific parsing errors based on the error messages."""
    try:
        with open(filepath, "r") as f:
            content = f.read()

        # Fix common patterns that cause parsing errors
        patterns = [
            # Fix self.attribute = value lines that are improperly indented
            (
                r"^(\s*)self\.(\w+)\s*=\s*(.+)$",
                lambda m: "        " + m.group(0).lstrip() if len(m.group(1)) < 8 else m.group(0),
            ),
            # Fix docstrings after function definitions
            (
                r'^(\s*)("""[^"]*"""|\'\'\'[^\']*\'\'\')$',
                lambda m: "    " + m.group(0).lstrip() if len(m.group(1)) < 4 else m.group(0),
            ),
        ]

        lines = content.split("\n")
        fixed_lines = []

        for i, line in enumerate(lines):
            fixed_line = line
            for pattern, replacement in patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    fixed_line = replacement(match)
                    break
            fixed_lines.append(fixed_line)

        # Write the fixed content back
        with open(filepath, "w") as f:
            f.write("\n".join(fixed_lines))

        return True
    except Exception as e:
        print(f"Error fixing parsing errors in {filepath}: {e}")
        return False


        def run_black_on_file(filepath: str) -> Tuple[bool, str]:
        def run_black_on_file(filepath: str) -> Tuple[bool, str]:
    """Run black on a single file and return success status and output."""
    """Run black on a single file and return success status and output."""
    """Run black on a single file and return success status and output."""
    try:
        result = subprocess.run(
            ["black", "--target-version", "py311", filepath], capture_output=True, text=True
        )
        return result.returncode == 0, result.stderr
    except Exception as e:
        return False, str(e)


        def main():
        def main():
    """Main function to fix all files with Black errors."""
    """Main function to fix all files with Black errors."""
    """Main function to fix all files with Black errors."""
    print("Starting to fix Black formatting errors...")

    fixed_count = 0
    failed_count = 0

    for file_path in ERROR_FILES:
        full_path = Path(file_path)
        if not full_path.exists():
            print(f"Skipping {file_path} - file not found")
            continue

        print(f"\nProcessing {file_path}...")

        # First, try to fix indentation issues
        if fix_indentation_in_file(str(full_path)):
            print(f"  Fixed indentation issues")

        # Then, fix specific parsing errors
        if fix_specific_parsing_errors(str(full_path)):
            print(f"  Fixed parsing errors")

        # Finally, run Black
        success, error = run_black_on_file(str(full_path))
        if success:
            print(f"  ✓ Black formatting successful")
            fixed_count += 1
        else:
            print(f"  ✗ Black formatting failed: {error}")
            failed_count += 1

    print(f"\n\nSummary:")
    print(f"  Fixed: {fixed_count} files")
    print(f"  Failed: {failed_count} files")

    # Now run Black on the entire project to catch any remaining issues
    print("\nRunning Black on entire project...")
    subprocess.run(["black", "--target-version", "py311", "."])


if __name__ == "__main__":
    main()
