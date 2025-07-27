#!/usr/bin/env python3
"""Remove all incorrectly indented TODO docstrings from files."""

import re
from pathlib import Path


def clean_todo_docstrings(filepath):
    """Remove all TODO docstrings that are incorrectly placed."""
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

        cleaned_lines = []
        skip_next = False

        for i, line in enumerate(lines):
            # Skip if we marked this line for skipping
            if skip_next:
                skip_next = False
                continue

            stripped = line.strip()

            # Check if this is a TODO docstring at wrong indentation
            if (stripped.startswith('"""TODO:') and stripped.endswith('"""')) or (
                stripped.startswith("'''TODO:") and stripped.endswith("'''")
            ):
                # Check if it's at column 0 or wrong indentation
                indent = len(line) - len(line.lstrip())  # noqa: F841

                # Look at previous line to see if this makes sense
                if i > 0:
                    prev_line = lines[i - 1].strip()

                    # If previous line is a function/method definition, keep it but fix indentation
                    if prev_line.endswith(":") and ("def " in prev_line or "class " in prev_line):
                        # This is a legitimate docstring, fix its indentation
                        prev_indent = len(lines[i - 1]) - len(lines[i - 1].lstrip())
                        proper_indent = prev_indent + 4
                        cleaned_lines.append(" " * proper_indent + stripped + "\n")
                        continue

                # Otherwise, this is a duplicate or misplaced docstring - skip it
                continue

            # Keep all other lines
            cleaned_lines.append(line)

        # Write back
        with open(filepath, "w") as f:
            f.writelines(cleaned_lines)

        return True
    except Exception as e:
        print(f"Error cleaning {filepath}: {e}")
        return False


def main():
    """Clean all Python files."""
    cleaned_count = 0

    for py_file in Path(".").rglob("*.py"):
        if any(skip in str(py_file) for skip in [".venv", "venv", "__pycache__", ".git"]):
            continue

        if clean_todo_docstrings(py_file):
            cleaned_count += 1

    print(f"Cleaned {cleaned_count} files")


if __name__ == "__main__":
    main()
