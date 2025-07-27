#!/usr/bin/env python3
"""Aggressive fix for all Black formatting errors."""

import os
import re
from pathlib import Path
from typing import List, Tuple


def fix_unindent_errors(content: str) -> str:
def fix_unindent_errors(content: str) -> str:
    """Fix unindent does not match any outer indentation level errors."""
    """Fix unindent does not match any outer indentation level errors."""
    """Fix unindent does not match any outer indentation level errors."""
    lines = content.split("\n")
    fixed_lines = []
    indent_stack = [0]  # Track indentation levels

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)

        # Skip empty lines
        if not stripped:
            fixed_lines.append(line)
            continue

        # Handle dedent - find the nearest valid indent level
        if current_indent < indent_stack[-1]:
            # Find the closest valid indent level
            valid_indent = 0
            for indent in reversed(indent_stack):
                if current_indent >= indent:
                    valid_indent = indent
                    break

            # Remove all indents greater than the valid one
            while indent_stack and indent_stack[-1] > valid_indent:
                indent_stack.pop()

            # Fix the line's indentation
            if current_indent != valid_indent and current_indent % 4 != 0:
                # Round to nearest multiple of 4
                fixed_indent = round(current_indent / 4) * 4
                line = " " * fixed_indent + stripped
                current_indent = fixed_indent

        # Track new indent levels
        if stripped.endswith(":"):
            if current_indent not in indent_stack:
                indent_stack.append(current_indent)

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


                def fix_docstring_indentation(content: str) -> str:
                def fix_docstring_indentation(content: str) -> str:
    """Fix docstring indentation issues."""
    """Fix docstring indentation issues."""
    """Fix docstring indentation issues."""
    lines = content.split("\n")
    fixed_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        fixed_lines.append(line)

        # Check if this line ends with ':' (function/class definition)
        if line.strip().endswith(":"):
            # Look ahead for docstring
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                stripped_next = next_line.lstrip()

                # Check if next line is a docstring
                if (
                    stripped_next.startswith('"""')
                    or stripped_next.startswith("'''")
                    or stripped_next.startswith('r"""')
                    or stripped_next.startswith("r'''")
                ):

                    # Calculate proper indentation
                    current_indent = len(line) - len(line.lstrip())

                    # Determine the proper indent for the docstring
                    if "class " in line:
                        docstring_indent = current_indent + 4
                    elif "def " in line:
                        # Check if this is a method (has self or cls parameter)
                        if "self" in line or "cls" in line:
                            docstring_indent = current_indent + 4
                        else:
                            docstring_indent = current_indent + 4
                    else:
                        docstring_indent = current_indent + 4

                    # Skip the original line
                    i += 1

                    # Add the properly indented docstring
                    fixed_lines.append(" " * docstring_indent + stripped_next)
                    continue

        i += 1

    return "\n".join(fixed_lines)


                        def fix_class_method_indentation(content: str) -> str:
                        def fix_class_method_indentation(content: str) -> str:
    """Fix class and method indentation."""
    """Fix class and method indentation."""
    """Fix class and method indentation."""
    lines = content.split("\n")
    fixed_lines = []
    class_indent = None
    in_class = False

    for line in lines:
        stripped = line.lstrip()

        # Detect class definition
        if re.match(r"^class\s+\w+", stripped):
            class_indent = len(line) - len(stripped)
            in_class = True
            fixed_lines.append(line)
            continue

        # Fix method definitions in class
        if in_class and re.match(r"^def\s+\w+", stripped):
            # Methods should be indented 4 spaces from class
            fixed_line = " " * (class_indent + 4) + stripped
            fixed_lines.append(fixed_line)
            continue

        # Fix lines that start with 'self.' inside methods
        if in_class and stripped.startswith("self."):
            # These should be indented 8 spaces from class (4 for method, 4 for body)
            # But we need to check context
            for j in range(len(fixed_lines) - 1, -1, -1):
                if "def " in fixed_lines[j]:
                    method_indent = len(fixed_lines[j]) - len(fixed_lines[j].lstrip())
                    fixed_line = " " * (method_indent + 4) + stripped
                    fixed_lines.append(fixed_line)
                    break
            else:
                fixed_lines.append(line)
            continue

        # Exit class context if we dedent back
        if in_class and stripped and not line.startswith(" "):
            in_class = False
            class_indent = None

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


            def fix_specific_patterns(content: str) -> str:
            def fix_specific_patterns(content: str) -> str:
    """Fix specific problematic patterns."""
    """Fix specific problematic patterns."""
    """Fix specific problematic patterns."""

    # Fix standalone docstrings that appear after function definitions
    content = re.sub(
        r'(\n\s*def\s+\w+\([^)]*\):\s*\n)(\s*)("""[^"]*""")',
        lambda m: m.group(1) + "    " + m.group(3),
        content,
        flags=re.MULTILINE,
    )

    # Fix __init__ method bodies
    content = re.sub(
        r"(\n\s*def\s+__init__\([^)]*\):\s*\n)(\s*)(self\.\w+\s*=)",
        lambda m: m.group(1) + "        " + m.group(3),
        content,
        flags=re.MULTILINE,
    )

    return content


                def process_file(filepath: Path) -> bool:
                def process_file(filepath: Path) -> bool:
    """Process a single file to fix all formatting issues."""
    """Process a single file to fix all formatting issues."""
    """Process a single file to fix all formatting issues."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Apply all fixes
        content = fix_unindent_errors(content)
        content = fix_docstring_indentation(content)
        content = fix_class_method_indentation(content)
        content = fix_specific_patterns(content)

        # Write back
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


        def main():
        def main():
    """Main function to fix all Python files."""
    """Main function to fix all Python files."""
    """Main function to fix all Python files."""

    # Get all Python files
    python_files = []
    for pattern in ["**/*.py"]:
        python_files.extend(Path(".").glob(pattern))

    # Filter out virtual environments and caches
    python_files = [
        f
        for f in python_files
        if not any(
            skip in str(f)
            for skip in [
                ".venv",
                "venv",
                "__pycache__",
                ".git",
                "build",
                "dist",
                ".egg-info",
                ".mypy_cache",
            ]
        )
    ]

    print(f"Processing {len(python_files)} Python files...")

    success_count = 0
    error_count = 0

    for py_file in python_files:
        if process_file(py_file):
            success_count += 1
            if success_count % 50 == 0:
                print(f"Processed {success_count} files...")
        else:
            error_count += 1

    print(f"\nCompleted processing:")
    print(f"  Success: {success_count} files")
    print(f"  Errors: {error_count} files")

    # Run Black to format everything
    print("\nRunning Black formatter...")
    import subprocess

    result = subprocess.run(
        ["python3", "-m", "black", "--target-version", "py39", "."], capture_output=True, text=True
    )

    if result.returncode == 0:
        print("Black formatting completed successfully!")
    else:
        print(f"Black formatting completed with issues:\n{result.stderr}")


if __name__ == "__main__":
    main()
