#!/usr/bin/env python3
"""
Migration script to update deprecated import paths to new locations.

Usage:
    python scripts/migrate_imports.py [--dry-run] [--backup] [path]

    --dry-run: Show what would be changed without modifying files
    --backup: Create .bak files before modifying
    path: Directory or file to migrate (default: current directory)
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple

# Define import mappings (old -> new)
IMPORT_MAPPINGS = [
    # Direct module mappings
    (
        r"from genomevault\.hypervector\.encoder import",
        "from genomevault.hypervector_transform.encoding import",
    ),
    (
        r"from genomevault\.hypervector\.encoding\.genomic import",
        "from genomevault.hypervector_transform.encoding import",
    ),
    (
        r"from genomevault\.hypervector\.operations\.binding import",
        "from genomevault.hypervector_transform.binding_operations import",
    ),
    (
        r"from genomevault\.hypervector\.encoding import",
        "from genomevault.hypervector_transform.encoding import",
    ),
    (
        r"from genomevault\.hypervector\.operations import",
        "from genomevault.hypervector_transform.binding_operations import",
    ),
    # Import module patterns
    (
        r"import genomevault\.hypervector\.encoder",
        "import genomevault.hypervector_transform.encoding",
    ),
    (
        r"genomevault\.hypervector\.encoder\.",
        "genomevault.hypervector_transform.encoding.",
    ),
]

# Specific class/function renames
CLASS_MAPPINGS = [
    # If GenomicEncoder is imported, suggest using HypervectorEncoder
    (r"\bGenomicEncoder\b", "HypervectorEncoder  # Note: GenomicEncoder renamed"),
    # Binding operation renames
    (r"\bcircular_convolution\b", "circular_bind"),
    (r"\belement_wise_multiply\b", "element_wise_bind"),
]


def find_python_files(path: Path) -> List[Path]:
    """Find all Python files in the given path."""
    if path.is_file() and path.suffix == ".py":
        return [path]

    python_files = []
    for root, _, files in os.walk(path):
        root_path = Path(root)
        # Skip virtual environments and hidden directories
        if any(
            part.startswith(".") or part == "venv" or part == "__pycache__"
            for part in root_path.parts
        ):
            continue

        for file in files:
            if file.endswith(".py"):
                python_files.append(root_path / file)

    return python_files


def migrate_file(file_path: Path, dry_run: bool = False, backup: bool = False) -> List[str]:
    """
    Migrate imports in a single file.

    Returns list of changes made.
    """
    changes = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return [f"Error reading {file_path}: {e}"]

    original_content = content

    # Apply import mappings
    for old_pattern, new_pattern in IMPORT_MAPPINGS:
        if re.search(old_pattern, content):
            content = re.sub(old_pattern, new_pattern, content)
            changes.append(f"  Updated import: {old_pattern} -> {new_pattern}")

    # Apply class/function mappings (only in files that had import changes)
    if changes and any("hypervector_transform" in change for change in changes):
        for old_pattern, new_pattern in CLASS_MAPPINGS:
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_pattern, content)
                changes.append(f"  Updated reference: {old_pattern} -> {new_pattern}")

    # Write changes if not dry run and there were changes
    if changes and not dry_run:
        if backup:
            shutil.copy2(file_path, f"{file_path}.bak")

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            return [f"Error writing {file_path}: {e}"]

    return changes


def main():
    parser = argparse.ArgumentParser(description="Migrate deprecated imports to new paths")
    parser.add_argument("path", nargs="?", default=".", help="Path to file or directory to migrate")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument("--backup", action="store_true", help="Create .bak files before modifying")

    args = parser.parse_args()

    path = Path(args.path).resolve()

    if not path.exists():
        print(f"Error: {path} does not exist")
        return 1

    # Find Python files
    python_files = find_python_files(path)

    if not python_files:
        print(f"No Python files found in {path}")
        return 0

    print(f"{'DRY RUN: ' if args.dry_run else ''}Migrating imports in {len(python_files)} files...")
    print()

    total_changes = 0
    files_changed = 0

    for file_path in python_files:
        relative_path = file_path.relative_to(Path.cwd()) if file_path.is_absolute() else file_path
        changes = migrate_file(file_path, dry_run=args.dry_run, backup=args.backup)

        if changes:
            print(f"{relative_path}:")
            for change in changes:
                print(change)
            print()
            files_changed += 1
            total_changes += len(changes)

    # Summary
    print("=" * 60)
    if args.dry_run:
        print(f"DRY RUN SUMMARY: Would modify {files_changed} files with {total_changes} changes")
        print("Run without --dry-run to apply changes")
    else:
        print(f"MIGRATION COMPLETE: Modified {files_changed} files with {total_changes} changes")
        if args.backup:
            print("Backup files created with .bak extension")

    return 0


if __name__ == "__main__":
    exit(main())
