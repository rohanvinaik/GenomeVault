#!/usr/bin/env python3
"""
Simple Debug Print Removal Script
==================================

A robust regex-based approach to replace print statements with logging.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Tuple

# Regex patterns for different print scenarios
PRINT_PATTERNS = [
    # Standard print statements
    (r"^(\s*)print\((.*?)\)(\s*#.*)?$", r"\1logger.debug(\2)\3"),
    # Print with f-strings
    (r'^(\s*)print\(f["\'](.+?)["\']\)(\s*#.*)?$', r'\1logger.info(f"\2")\3'),
    # Print with .format()
    (
        r"^(\s*)print\((.+?)\.format\((.*?)\)\)(\s*#.*)?$",
        r"\1logger.info(\2.format(\3))\4",
    ),
    # Print with % formatting
    (r"^(\s*)print\((.+?)\s*%\s*(.+?)\)(\s*#.*)?$", r"\1logger.info(\2 % \3)\4"),
    # Multi-line prints (simplified)
    (r"^(\s*)print\(\s*$", r"\1logger.info("),
]

# Keywords to determine log level
ERROR_KEYWORDS = ["error", "fail", "exception", "critical", "fatal", "abort"]
WARNING_KEYWORDS = ["warn", "caution", "alert", "attention"]
DEBUG_KEYWORDS = ["debug", "trace", "verbose", "detail", "dump", "raw"]
SUCCESS_KEYWORDS = ["success", "complete", "done", "✓", "✔", "finished", "passed"]

# Import statements to add
IMPORT_STATEMENT = "from genomevault.utils.logging import get_logger\n"
LOGGER_INIT = "logger = get_logger(__name__)\n"


def determine_log_level(content: str) -> str:
    """Determine appropriate log level based on content."""
    content_lower = content.lower()

    # Check for keywords
    if any(keyword in content_lower for keyword in ERROR_KEYWORDS):
        return "error"
    elif any(keyword in content_lower for keyword in WARNING_KEYWORDS):
        return "warning"
    elif any(keyword in content_lower for keyword in DEBUG_KEYWORDS):
        return "debug"
    elif any(keyword in content_lower for keyword in SUCCESS_KEYWORDS):
        return "info"

    # Default to debug for most prints
    return "debug"


def process_line(line: str, in_test_file: bool = False) -> Tuple[str, bool]:
    """Process a single line and return modified line and whether it was changed."""
    # Skip certain patterns
    if any(skip in line for skip in ['"""', "'''", "#"]):
        if "print(" in line and not line.strip().startswith("#"):
            # Print in docstring or comment - might need handling
            pass
        else:
            return line, False

    # Check if line contains print
    if "print(" not in line:
        return line, False

    # Try to match print patterns
    original_line = line
    modified = False

    # Simple replacement for basic prints
    if re.match(r"^\s*print\(", line):
        # Extract the content inside print()
        match = re.match(r"^(\s*)print\((.*?)\)(\s*#.*)?$", line)
        if match:
            indent, content, comment = match.groups()
            comment = comment or ""

            # Skip empty prints
            if not content or content.strip() == "":
                return f"{indent}# Removed empty print(){comment}\n", True

            # Determine log level
            log_level = determine_log_level(content)

            # Special handling for test files
            if in_test_file:
                # Keep prints in test files but comment them or use debug
                if "=" * 10 in content or "-" * 10 in content:
                    return f"{indent}# {line.strip()}\n", True
                log_level = "debug"

            # Create logging statement
            new_line = f"{indent}logger.{log_level}({content}){comment}\n"
            return new_line, True

    return line, False


def process_file(file_path: Path, dry_run: bool = False, verbose: bool = False) -> Tuple[int, int]:
    """Process a single Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return 0, 0

        # Check if file is a test file
        is_test_file = "test" in str(file_path).lower()

        # Process lines
        new_lines = []
        modified_count = 0
        needs_import = False
        has_logger_import = False
        has_logger_init = False

        for line in lines:
            # Check for existing imports
            if "from genomevault.utils.logging import" in line or "import logging" in line:
                has_logger_import = True
            if "logger = get_logger" in line or "logger = logging.getLogger" in line:
                has_logger_init = True

            new_line, was_modified = process_line(line, is_test_file)
            if was_modified:
                modified_count += 1
                needs_import = True
            new_lines.append(new_line)

        # Add imports if needed
        if needs_import and not has_logger_import:
            # Find where to insert imports
            insert_pos = 0
            for i, line in enumerate(new_lines):
                if (
                    line.strip()
                    and not line.strip().startswith("#")
                    and not line.strip().startswith('"""')
                ):
                    if line.startswith("from ") or line.startswith("import "):
                        insert_pos = i + 1
                    else:
                        break

            # Insert import and logger init
            if not has_logger_import:
                new_lines.insert(insert_pos, IMPORT_STATEMENT)
                if not has_logger_init:
                    new_lines.insert(insert_pos + 1, LOGGER_INIT)
                    new_lines.insert(insert_pos + 2, "\n")

        if modified_count > 0:
            if verbose:
                print(f"Processing {file_path}: {modified_count} print statements found")

            if not dry_run:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)

        return 1 if modified_count > 0 else 0, modified_count

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0


def process_directory(
    directory: Path, dry_run: bool = False, verbose: bool = False
) -> Tuple[int, int]:
    """Process all Python files in a directory."""
    total_files = 0
    total_prints = 0

    for py_file in directory.rglob("*.py"):
        # Skip __pycache__ and the script itself
        if "__pycache__" in str(py_file) or "remove_debug_prints" in str(py_file):
            continue

        files_modified, prints_found = process_file(py_file, dry_run, verbose)
        total_files += files_modified
        total_prints += prints_found

    return total_files, total_prints


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Remove debug print statements")
    parser.add_argument(
        "directories",
        nargs="*",
        default=["devtools", "examples", "tests"],
        help="Directories to process",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("Debug Print Removal Script")
    print("=" * 50)

    if args.dry_run:
        print("DRY RUN MODE - No changes will be made\n")

    total_files = 0
    total_prints = 0

    for dir_name in args.directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            print(f"Warning: Directory not found: {dir_name}")
            continue

        print(f"Processing {dir_name}/...")
        files_modified, prints_found = process_directory(dir_path, args.dry_run, args.verbose)

        total_files += files_modified
        total_prints += prints_found

        print(f"  Modified {files_modified} files, {prints_found} print statements replaced")

    print("\n" + "=" * 50)
    print(f"Total: {total_prints} print statements in {total_files} files")

    if args.dry_run:
        print("\nRun without --dry-run to apply changes")
    else:
        print("\nChanges applied successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
