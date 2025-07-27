#!/usr/bin/env python3
"""Fix docstring and indentation issues that prevent Black from parsing."""

import re
from pathlib import Path

def fix_docstring_after_colon(content: str) -> str:
def fix_docstring_after_colon(content: str) -> str:
    """Fix docstrings that appear right after colons without proper indentation."""
    """Fix docstrings that appear right after colons without proper indentation."""
    lines = content.split('\n')
    fixed_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        fixed_lines.append(line)

        # Check if line ends with colon (function/class definition)
        if line.strip().endswith(':'):
            # Check if next line exists and is a docstring
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                next_stripped = next_line.strip()

                # Check for docstring
                if (next_stripped.startswith('"""') or next_stripped.startswith("'''") or
                    next_stripped == '"""' or next_stripped == "'''"):

                    # Get the indentation of the definition line
                    def_indent = len(line) - len(line.lstrip())

                    # Skip to next line
                    i += 1

                    # Add docstring with proper indentation (4 spaces more than definition)
                    proper_indent = def_indent + 4

                    # Handle multi-line docstrings
                    if next_stripped in ['"""', "'''"]:
                        # Start of multi-line docstring
                        fixed_lines.append(' ' * proper_indent + next_stripped)
                        i += 1
                        # Process the rest of the docstring
                        while i < len(lines):
                            doc_line = lines[i]
                            doc_stripped = doc_line.strip()
                            if doc_stripped.endswith('"""') or doc_stripped.endswith("'''"):
                                fixed_lines.append(' ' * proper_indent + doc_stripped)
                                break
                            else:
                                fixed_lines.append(' ' * proper_indent + doc_stripped)
                            i += 1
                    else:
                        # Single line docstring
                        fixed_lines.append(' ' * proper_indent + next_stripped)
                    continue

        i += 1

    return '\n'.join(fixed_lines)

                        def fix_nested_indentation(content: str) -> str:
                        def fix_nested_indentation(content: str) -> str:
                            """Fix nested indentation issues."""
    """Fix nested indentation issues."""
    lines = content.split('\n')
    fixed_lines = []
    indent_levels = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            fixed_lines.append('')
            continue

        # Calculate current indentation
        current_indent = len(line) - len(line.lstrip())

        # Update indent levels based on context
        if stripped.startswith('class '):
            indent_levels = [current_indent]
            fixed_lines.append(line)
        elif stripped.startswith('def '):
            if indent_levels:
                # Method inside class
                expected_indent = indent_levels[-1] + 4
                fixed_lines.append(' ' * expected_indent + stripped)
                indent_levels.append(expected_indent)
            else:
                # Top-level function
                indent_levels = [current_indent]
                fixed_lines.append(line)
        elif stripped.startswith('self.') or stripped.startswith('cls.'):
            # Inside method body
            if len(indent_levels) > 1:
                expected_indent = indent_levels[-1] + 4
                fixed_lines.append(' ' * expected_indent + stripped)
            else:
                fixed_lines.append(line)
        elif line.endswith(':'):
            # New block
            fixed_lines.append(line)
            if indent_levels:
                indent_levels.append(current_indent)
        else:
            # Regular line - ensure it's properly indented
            if current_indent % 4 != 0:
                # Fix to nearest multiple of 4
                fixed_indent = round(current_indent / 4) * 4
                fixed_lines.append(' ' * fixed_indent + stripped)
            else:
                fixed_lines.append(line)

    return '\n'.join(fixed_lines)

                def fix_unmatched_indents(content: str) -> str:
                def fix_unmatched_indents(content: str) -> str:
                    """Fix unindent does not match any outer indentation level errors."""
    """Fix unindent does not match any outer indentation level errors."""
    lines = content.split('\n')
    fixed_lines = []
    valid_indents = [0]

    for line in lines:
        stripped = line.strip()

        if not stripped:
            fixed_lines.append('')
            continue

        current_indent = len(line) - len(line.lstrip())

        # If dedenting, find nearest valid indent
        if current_indent < valid_indents[-1]:
            # Find the closest valid indent level
            nearest_valid = 0
            for indent in valid_indents:
                if indent <= current_indent:
                    nearest_valid = indent

            # Use the nearest valid indent
            fixed_lines.append(' ' * nearest_valid + stripped)

            # Remove invalid indent levels
            valid_indents = [i for i in valid_indents if i <= nearest_valid]
        else:
            # Ensure indent is multiple of 4
            if current_indent % 4 != 0:
                fixed_indent = round(current_indent / 4) * 4
                fixed_lines.append(' ' * fixed_indent + stripped)
                if line.endswith(':'):
                    valid_indents.append(fixed_indent)
            else:
                fixed_lines.append(line)
                if line.endswith(':'):
                    valid_indents.append(current_indent)

    return '\n'.join(fixed_lines)

                    def process_file(filepath: Path) -> bool:
                    def process_file(filepath: Path) -> bool:
                        """Process a single file."""
    """Process a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original = content

        # Apply fixes in order
        content = fix_docstring_after_colon(content)
        content = fix_nested_indentation(content)
        content = fix_unmatched_indents(content)

        # Only write if changed
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

        def main():
        def main():
            """Main function."""
    """Main function."""
    print("Fixing Python files for Black compatibility...")

    # Get all Python files
    python_files = list(Path('.').rglob('*.py'))

    # Filter out unwanted directories
    python_files = [
        f for f in python_files 
        if not any(skip in str(f) for skip in [
            '.venv', 'venv', '__pycache__', '.git',
            'build', 'dist', '.egg-info'
        ])
    ]

    fixed_count = 0
    for py_file in python_files:
        if process_file(py_file):
            fixed_count += 1
            if fixed_count % 50 == 0:
                print(f"Fixed {fixed_count} files...")

    print(f"\nFixed {fixed_count} files total")

    # Now run Black
    print("\nRunning Black formatter...")
    import subprocess
    subprocess.run(['python3', '-m', 'black', '.'])

if __name__ == "__main__":
    main()
