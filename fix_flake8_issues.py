#!/usr/bin/env python3
"""
Script to fix common flake8 issues in the GenomeVault codebase.
"""

import re
from pathlib import Path


def fix_f_string_without_placeholders(content: str) -> str:
    """Fix F541: f-string is missing placeholders."""
    # Pattern to match f-strings without placeholders
    pattern = r'"([^"]*?)"'

    def replace_f_string(match):
        string_content = match.group(1)
        # Check if the string actually contains placeholders
        if "{" not in string_content:
            # Remove the 'f' prefix
            return '"{string_content}"'
        return match.group(0)

    content = re.sub(pattern, replace_f_string, content)

    # Do the same for single quotes
    pattern = r"'([^']*?)'"

    def replace_f_string_single(match):
        string_content = match.group(1)
        if "{" not in string_content:
            return "'{string_content}'"
        return match.group(0)

    content = re.sub(pattern, replace_f_string_single, content)

    return content


def fix_bare_except(content: str) -> str:
    """Fix E722: do not use bare 'except'."""
    # Replace bare except with except Exception
    content = re.sub(r"\bexcept\s*:", "except Exception:", content)
    return content


def remove_unused_imports(content: str, file_path: str) -> str:
    """Remove unused imports (F401) - basic implementation."""
    lines = content.split("\n")
    new_lines = []
    imports_to_check = []

    # First pass: collect all imports
    for i, line in enumerate(lines):
        if line.strip().startswith(("import ", "from ")):
            imports_to_check.append((i, line))

    # For now, we'll keep all imports but add a comment for manual review
    # A proper implementation would use AST to check usage
    for i, line in enumerate(lines):
        if any(i == imp[0] for imp in imports_to_check):
            # Check if it's marked as unused in our error list
            if is_import_unused(line, file_path):
                # Comment it out instead of removing
                new_lines.append("# {line}  # TODO: Remove if truly unused")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def is_import_unused(import_line: str, file_path: str) -> bool:
    """Check if an import is marked as unused based on flake8 output."""
    # This is a simplified check - in reality, you'd parse the flake8 output
    # For now, return False to be safe
    return False


def fix_file(file_path: Path) -> bool:
    """Fix common flake8 issues in a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Apply fixes
        content = fix_f_string_without_placeholders(content)
        content = fix_bare_except(content)
        # Skip unused imports for now - requires more sophisticated analysis
        # content = remove_unused_imports(content, str(file_path))

        # Only write if content changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        return False
    except Exception:
        print("Error processing {file_path}: {e}")
        return False


def main():
    """Main function to fix flake8 issues."""
    project_root = Path(__file__).parent

    # Get all Python files
    python_files = []
    for pattern in ["*.py"]:
        python_files.extend(project_root.rglob(pattern))

    # Exclude virtual environments and other directories
    exclude_dirs = {".venv", "venv", "__pycache__", ".git", "build", "dist", ".eggs"}
    python_files = [
        f
        for f in python_files
        if not any(excluded in f.parts for excluded in exclude_dirs)
    ]

    print("Found {len(python_files)} Python files to process")

    fixed_count = 0
    for file_path in python_files:
        if fix_file(file_path):
            fixed_count += 1
            print("Fixed: {file_path}")

    print("\nFixed {fixed_count} files")
    print("\nNote: This script only fixes simple issues like:")
    print("- F541: f-strings without placeholders")
    print("- E722: bare except clauses")
    print("\nFor unused imports and other issues, manual review is recommended.")
    print("You can use 'autoflake' to remove unused imports more safely:")
    print("  pip install autoflake")
    print("  autoflake --remove-all-unused-imports --in-place --recursive .")


if __name__ == "__main__":
    main()
