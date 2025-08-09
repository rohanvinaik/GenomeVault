#!/usr/bin/env python3
"""
Clean up the root directory by organizing files into appropriate locations.
This script follows best practices for project structure.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple

# Define the root directory
ROOT_DIR = Path(__file__).parent.parent

# Files to move to tools/ directory
MOVE_TO_TOOLS = [
    "apply_area_fixes.py",
    "apply_common_fixes.py",
    "apply_lint_fixes.py",
    "fix_imports.py",
    "fix_prover.py",
    "fix_python_compatibility.py",
    "quick_fix_init_files.py",
    "run_complete_lint_fix.py",
    "proper_ruff_upgrade.py",
    "upgrade_ruff.py",
    "check_lint_status.py",
    "comprehensive_status.py",
    "final_validation.py",
    "preflight_check.py",
    "validate_lint_clean.py",
    "validate_lint_fixes.py",
    "verify_fixes.py",
    "verify_phase3_ready.py",
    "verify_ruff.py",
    "diagnostic.py",
    "focused_green_impl.py",
    "generate_comparison_report.py",
    "simple_autofix_demo.py",
    "test_autofix_example.py",
]

# Files to delete (temporary test files)
DELETE_FILES = [
    "test_api_startup.py",
    "test_constant.py",
    "test_fixed_phase3.py",
    "test_fixes.py",
    "test_hamming_lut.py",
    "test_phase3.py",
    "test_ruff_upgrade.py",
    "test_run_api.py",
    "test_simple.py",
    # Task-specific scripts
    "run_mypy_task4.sh",
    "run_pytest_task5.sh",
    "run_ruff_fix_task3.sh",
    "run_ruff_task3.sh",
    # Temporary push/commit scripts
    "commit_and_push_now.sh",
    "push_audit_implementation.sh",
    "push_to_git.sh",
    "push_to_github.sh",
    "quick_push_clean_slate.sh",
    # Status check scripts
    "check_git_status.sh",
    "check_implementation_status.sh",
    # Task outputs
    "task4_mypy_output.txt",
    # Test registries
    "test_e2e_registry.json",
    "hypervector_registry.json",
]

# Directories to delete
DELETE_DIRS = [
    "htmlcov",
    "task3_ruff_output",
    "__pycache__",
    ".pytest_cache",
    "genomevault.egg-info",
]

# Files to keep in .gitignore but delete if present
COVERAGE_FILES = [
    "coverage.xml",
    ".coverage",
    "*.pyc",
    "*.pyo",
]


def create_backup_dir() -> Path:
    """Create a backup directory for safety."""
    backup_dir = ROOT_DIR / ".cleanup_backup"
    backup_dir.mkdir(exist_ok=True)
    return backup_dir


def move_file_to_tools(file_name: str, backup_dir: Path) -> bool:
    """Move a file from root to tools/ directory."""
    src = ROOT_DIR / file_name
    if not src.exists():
        return False

    # Create backup
    backup_dest = backup_dir / file_name
    shutil.copy2(src, backup_dest)

    # Move to tools
    dest = ROOT_DIR / "tools" / file_name
    if dest.exists():
        print(f"  ⚠️  {file_name} already exists in tools/, skipping")
        return False

    shutil.move(str(src), str(dest))
    print(f"  ✓ Moved {file_name} to tools/")
    return True


def delete_file(file_name: str, backup_dir: Path) -> bool:
    """Delete a file from root directory."""
    file_path = ROOT_DIR / file_name
    if not file_path.exists():
        return False

    # Create backup
    backup_dest = backup_dir / file_name
    shutil.copy2(file_path, backup_dest)

    # Delete file
    file_path.unlink()
    print(f"  ✓ Deleted {file_name}")
    return True


def delete_directory(dir_name: str, backup_dir: Path) -> bool:
    """Delete a directory from root."""
    dir_path = ROOT_DIR / dir_name
    if not dir_path.exists():
        return False

    # Create backup
    backup_dest = backup_dir / dir_name
    if backup_dest.exists():
        shutil.rmtree(backup_dest)
    shutil.copytree(dir_path, backup_dest)

    # Delete directory
    shutil.rmtree(dir_path)
    print(f"  ✓ Deleted directory {dir_name}/")
    return True


def clean_coverage_files() -> int:
    """Clean up coverage-related files."""
    count = 0
    for pattern in COVERAGE_FILES:
        if "*" in pattern:
            # Handle wildcard patterns
            import glob

            for file_path in glob.glob(str(ROOT_DIR / pattern)):
                Path(file_path).unlink()
                count += 1
        else:
            file_path = ROOT_DIR / pattern
            if file_path.exists():
                file_path.unlink()
                count += 1
    if count > 0:
        print(f"  ✓ Cleaned {count} coverage-related files")
    return count


def main():
    """Main cleanup function."""
    print("🧹 Starting root directory cleanup...")
    print(f"   Working in: {ROOT_DIR}")

    # Create backup directory
    backup_dir = create_backup_dir()
    print(f"   Backup directory: {backup_dir}")
    print()

    # Statistics
    moved_count = 0
    deleted_files = 0
    deleted_dirs = 0

    # Move files to tools/
    print("📦 Moving files to tools/...")
    for file_name in MOVE_TO_TOOLS:
        if move_file_to_tools(file_name, backup_dir):
            moved_count += 1

    print()
    print("🗑️  Deleting temporary files...")
    for file_name in DELETE_FILES:
        if delete_file(file_name, backup_dir):
            deleted_files += 1

    print()
    print("🗑️  Deleting temporary directories...")
    for dir_name in DELETE_DIRS:
        if delete_directory(dir_name, backup_dir):
            deleted_dirs += 1

    print()
    print("🧹 Cleaning coverage files...")
    coverage_count = clean_coverage_files()

    # Summary
    print()
    print("✨ Cleanup complete!")
    print(f"   • Moved {moved_count} files to tools/")
    print(f"   • Deleted {deleted_files} temporary files")
    print(f"   • Deleted {deleted_dirs} temporary directories")
    print(f"   • Cleaned {coverage_count} coverage files")
    print()
    print(f"💾 Backup saved to: {backup_dir}")
    print("   (You can delete this after verifying everything works)")

    # Suggest .gitignore additions
    if moved_count + deleted_files + deleted_dirs > 0:
        print()
        print("📝 Consider adding these to .gitignore:")
        print("   # Generated files")
        print("   htmlcov/")
        print("   coverage.xml")
        print("   .coverage")
        print("   *.egg-info/")
        print("   task*_output/")
        print("   test_*.py  # One-off test files in root")


if __name__ == "__main__":
    main()
