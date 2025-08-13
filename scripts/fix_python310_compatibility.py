#!/usr/bin/env python3
"""
Fix Python 3.10 compatibility issues by adding future annotations import
where needed for generic type hints.
"""

import ast
import os
from pathlib import Path
from typing import Set, List, Tuple


def check_needs_future_annotations(file_path: Path) -> bool:
    """Check if a file uses Python 3.9+ generic type syntax."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for generic type syntax
        if any(
            pattern in content
            for pattern in [
                "list[",
                "dict[",
                "tuple[",
                "set[",
                "List[",
                "Dict[",
                "Tuple[",
                "Set[",
            ]
        ):
            # Check if already has future import
            if "from __future__ import annotations" not in content:
                return True
    except Exception as e:
        pass  # Debug print removed

    return False


def add_future_annotations_import(file_path: Path) -> bool:
    """Add future annotations import to a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find the position to insert the import
        insert_pos = 0
        has_docstring = False
        in_docstring = False
        docstring_char = None

        for i, line in enumerate(lines):
            # Skip shebang
            if i == 0 and line.startswith("#!"):
                insert_pos = i + 1
                continue

            # Skip encoding declaration
            if i < 2 and "# -*- coding" in line:
                insert_pos = i + 1
                continue

            # Handle docstrings
            stripped = line.strip()
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_char = '"""' if stripped.startswith('"""') else "'''"
                    in_docstring = True
                    has_docstring = True
                    # Check if docstring ends on same line
                    if stripped.count(docstring_char) >= 2:
                        in_docstring = False
                        insert_pos = i + 1
                    continue
                elif stripped and not stripped.startswith("#"):
                    # First non-comment, non-empty line
                    break
            else:
                # In multiline docstring
                if docstring_char and docstring_char in line:
                    in_docstring = False
                    insert_pos = i + 1
                continue

            # Update insert position for empty lines after module docstring
            if not stripped and has_docstring:
                insert_pos = i + 1

        # Check if import already exists
        for line in lines:
            if "from __future__ import annotations" in line:
                return False

        # Insert the import
        import_line = "from __future__ import annotations\n"

        # Add appropriate spacing
        if insert_pos < len(lines):
            # Check if we need a blank line before
            if insert_pos > 0 and lines[insert_pos - 1].strip():
                import_line = "\n" + import_line
            # Check if we need a blank line after
            if lines[insert_pos].strip() and not lines[insert_pos].startswith("from __future__"):
                import_line = import_line + "\n"

        lines.insert(insert_pos, import_line)

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return True

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main():
    """Main function to fix Python 3.10 compatibility."""
    root_dir = Path(__file__).parent.parent

    # Find all Python files
    python_files = []
    for dir_name in [
        "genomevault",
        "tests",
        "scripts",
        "tools",
        "examples",
        "benchmarks",
    ]:
        dir_path = root_dir / dir_name
        if dir_path.exists():
            python_files.extend(dir_path.rglob("*.py"))

    files_needing_fix = []

    for file_path in python_files:
        if check_needs_future_annotations(file_path):
            files_needing_fix.append(file_path)

    if not files_needing_fix:
        print("✅ All files are already Python 3.10 compatible!")
        return

    print(f"\n📝 Found {len(files_needing_fix)} files needing future annotations import")

    fixed_count = 0
    for file_path in files_needing_fix:
        if add_future_annotations_import(file_path):
            print(f"  ✓ Fixed {file_path.relative_to(root_dir)}")
            fixed_count += 1
        else:
            print(f"  ⚠️  Skipped {file_path.relative_to(root_dir)}")

    print(f"\n✨ Fixed {fixed_count} files for Python 3.10 compatibility")


if __name__ == "__main__":
    main()
