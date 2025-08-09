#!/usr/bin/env python3
"""
Quick status check of the lint fix implementation.
"""

import subprocess
from pathlib import Path


def main():
    print("GenomeVault Lint Fix Status")
    print("=" * 60)

    # Check current branch
    result = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True
    )
    print(f"Current branch: {result.stdout.strip()}")

    # Check for uncommitted changes
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    if result.stdout.strip():
        print(
            f"\nUncommitted changes: {len(result.stdout.strip().split(chr(10)))} files"
        )
    else:
        print("\nNo uncommitted changes")

    # Show recent commits
    print("\nRecent commits:")
    result = subprocess.run(
        ["git", "log", "--oneline", "-5"], capture_output=True, text=True
    )
    print(result.stdout)

    # Quick lint check
    print("\nQuick lint check:")

    # Black
    result = subprocess.run(
        ["black", "--check", "genomevault", "--quiet"], capture_output=True, text=True
    )
    if result.returncode == 0:
        print("✅ Black: Clean")
    else:
        print("❌ Black: Needs formatting")

    # Ruff
    result = subprocess.run(
        ["ruff", "genomevault", "--quiet"], capture_output=True, text=True
    )
    if result.returncode == 0:
        print("✅ Ruff: No issues")
    else:
        # Count issues
        issues = len([l for l in result.stdout.split("\n") if l.strip()])
        print(f"⚠️  Ruff: {issues} issues found")

    print("=" * 60)
    print("\nConfiguration files status:")

    configs = [
        "pyproject.toml",
        "mypy.ini",
        ".pylintrc",
        ".editorconfig",
        ".pre-commit-config.yaml",
        "scripts/lint_check.sh",
        "scripts/lint_fix.sh",
        "scripts/lint_ratchet.sh",
    ]

    for config in configs:
        if Path(config).exists():
            print(f"✅ {config}")
        else:
            print(f"❌ {config} (missing)")

    print("\nTo continue fixing, run: python run_complete_lint_fix.py")
    print("To validate, run: python validate_lint_fixes.py")


if __name__ == "__main__":
    main()
