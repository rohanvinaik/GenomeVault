#!/usr/bin/env python3
"""Fix all TODO docstring indentation issues in one pass."""

import re
from pathlib import Path


def fix_todo_docstrings(filepath):  # noqa: C901
    """Fix TODO docstring indentation in a file."""
    try:
        with open(filepath, "r") as f:
            content = f.read()

        # Fix pattern: incorrectly indented TODO docstrings
        # This regex finds function/method definitions followed by incorrectly indented docstrings
        pattern = r'(^[ \t]*)(def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:\s*\n)(?:[ \t]*\n)*([ \t]*)("""TODO:.*?""")'

        def fix_indent(match):
            indent = match.group(1)  # Indentation of the def line
            def_line = match.group(2)  # The def line itself
            docstring = match.group(4)  # The TODO docstring

            # Calculate proper indentation (4 spaces more than def)
            proper_indent = indent + "    "

            # Return fixed version
            return indent + def_line + proper_indent + docstring

        # Apply the fix
        fixed_content = re.sub(pattern, fix_indent, content, flags=re.MULTILINE)

        # Also fix standalone TODO docstrings that are at wrong indentation
        # Pattern: line with only spaces followed by """TODO
        pattern2 = r'^([ \t]*)("""TODO:.*?""")\s*$'

        def fix_standalone(match):
            current_indent = match.group(1)
            docstring = match.group(2)

            # If it's at column 0 or weird indentation, assume it should be indented by 4 or 8
            indent_len = len(current_indent)
            if indent_len == 0 or indent_len % 4 != 0:
                # Find the previous non-empty line to determine context
                lines = fixed_content.split("\n")
                for i, line in enumerate(lines):
                    if docstring in line:
                        # Look backwards for context
                        for j in range(i - 1, -1, -1):
                            prev_line = lines[j].rstrip()
                            if prev_line and not prev_line.isspace():
                                if prev_line.strip().endswith(":"):
                                    # Previous line ends with colon, indent by 4 more
                                    prev_indent = len(lines[j]) - len(lines[j].lstrip())
                                    return " " * (prev_indent + 4) + docstring
                                break
                        break

                # Default to 8 spaces (method body level)
                return "        " + docstring

            return match.group(0)  # Keep as is if already properly indented

        fixed_content = re.sub(pattern2, fix_standalone, fixed_content, flags=re.MULTILINE)

        if fixed_content != content:
            with open(filepath, "w") as f:
                f.write(fixed_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Fix all Python files with TODO docstring issues."""
    fixed_count = 0
    error_count = 0

    # Process all Python files
    for py_file in Path(".").rglob("*.py"):
        # Skip virtual environments and caches
        if any(skip in str(py_file) for skip in [".venv", "venv", "__pycache__", ".git"]):
            continue

        if fix_todo_docstrings(py_file):
            fixed_count += 1
        else:
            # Try to read the file to see if it has TODO docstrings
            try:
                with open(py_file, "r") as f:
                    if "TODO:" in f.read():
                        error_count += 1
            except:
                pass

    print(f"Fixed {fixed_count} files")
    if error_count > 0:
        print(f"Failed to fix {error_count} files with TODO docstrings")


if __name__ == "__main__":
    main()
