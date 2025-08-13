#!/usr/bin/env python3
"""
Organize documentation files in the root directory.
Move implementation reports to docs/reports/ directory.
"""

import shutil
from pathlib import Path

# Define the root directory
ROOT_DIR = Path(__file__).parent.parent

# Core documentation that should stay in root
KEEP_IN_ROOT = [
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "SECURITY.md",
    "INSTALL.md",
    "CLAUDE.md",  # Important for Claude Code
    "VERSION.md",
]

# Implementation reports to move to docs/reports/
MOVE_TO_REPORTS = [
    "AUDIT_IMPLEMENTATION_SUMMARY.md",
    "IMPLEMENTATION_REPORT.md",
    "LINT_FIX_IMPLEMENTATION_SUMMARY.md",
    "MVP_IMPLEMENTATION_PLAN.md",
    "MVP_SUCCESS_SUMMARY.md",
    "VALIDATION_REPORT.md",
]


def create_reports_dir() -> Path:
    """Create docs/reports directory if it doesn't exist."""
    reports_dir = ROOT_DIR / "docs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def move_report_file(file_name: str, reports_dir: Path) -> bool:
    """Move a report file from root to docs/reports."""
    src = ROOT_DIR / file_name
    if not src.exists():
        return False

    dest = reports_dir / file_name
    if dest.exists():
        print(f"  ⚠️  {file_name} already exists in docs/reports/, skipping")
        return False

    shutil.move(str(src), str(dest))
    print(f"  ✓ Moved {file_name} to docs/reports/")
    return True


def create_reports_index(reports_dir: Path):
    """Create an index file for the reports."""
    index_content = """# Implementation Reports

This directory contains various implementation and validation reports for the GenomeVault project.

## Reports

- [Audit Implementation Summary](AUDIT_IMPLEMENTATION_SUMMARY.md) - Code audit and fixes summary
- [Implementation Report](IMPLEMENTATION_REPORT.md) - Overall implementation status report
- [Lint Fix Implementation Summary](LINT_FIX_IMPLEMENTATION_SUMMARY.md) - Linting and code quality fixes
- [MVP Implementation Plan](MVP_IMPLEMENTATION_PLAN.md) - Minimum Viable Product implementation strategy
- [MVP Success Summary](MVP_SUCCESS_SUMMARY.md) - MVP completion and success metrics
- [Validation Report](VALIDATION_REPORT.md) - System validation and testing report

## Navigation

- [Back to main docs](../README.md)
- [Back to project root](../../README.md)
"""

    index_file = reports_dir / "README.md"
    index_file.write_text(index_content)
    print("  ✓ Created reports index file")


def main():
    """Main organization function."""
    pass  # Debug print removed
    print(f"   Working in: {ROOT_DIR}")
    pass  # Debug print removed

    # Create reports directory
    reports_dir = create_reports_dir()
    print(f"   Reports directory: {reports_dir}")
    pass  # Debug print removed

    # Move report files
    moved_count = 0
    pass  # Debug print removed
    for file_name in MOVE_TO_REPORTS:
        if move_report_file(file_name, reports_dir):
            moved_count += 1

    # Create index file
    if moved_count > 0:
        create_reports_index(reports_dir)

    # Summary
    print("✨ Documentation organization complete!")
    print(f"   • Moved {moved_count} reports to docs/reports/")
    print(f"   • Kept {len(KEEP_IN_ROOT)} core docs in root")

    # Show what's left in root
    print("📄 Documentation files remaining in root:")
    for doc in KEEP_IN_ROOT:
        if (ROOT_DIR / doc).exists():
            print(f"   • {doc}")


if __name__ == "__main__":
    main()
