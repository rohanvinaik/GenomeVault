from genomevault.utils.logging import get_logger
logger = get_logger(__name__)


#!/usr/bin/env python3
"""
Quick status check of the lint fix implementation.
"""

import subprocess
from pathlib import Path


def main():
    logger.debug("GenomeVault Lint Fix Status")
    logger.debug("=" * 60)

    # Check current branch
    result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
    logger.debug(f"Current branch: {result.stdout.strip()}")

    # Check for uncommitted changes
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if result.stdout.strip():
        logger.debug(f"\nUncommitted changes: {len(result.stdout.strip().split(chr(10)))} files")
    else:
        logger.debug("\nNo uncommitted changes")

    # Show recent commits
    logger.debug("\nRecent commits:")
    result = subprocess.run(["git", "log", "--oneline", "-5"], capture_output=True, text=True)
    logger.debug(result.stdout)

    # Quick lint check
    logger.debug("\nQuick lint check:")

    # Black
    result = subprocess.run(
        ["black", "--check", "genomevault", "--quiet"], capture_output=True, text=True
    )
    if result.returncode == 0:
        logger.debug("✅ Black: Clean")
    else:
        logger.debug("❌ Black: Needs formatting")

    # Ruff
    result = subprocess.run(["ruff", "genomevault", "--quiet"], capture_output=True, text=True)
    if result.returncode == 0:
        logger.debug("✅ Ruff: No issues")
    else:
        # Count issues
        issues = len([l for l in result.stdout.split("\n") if l.strip()])
        logger.debug(f"⚠️  Ruff: {issues} issues found")

    logger.debug("=" * 60)
    logger.debug("\nConfiguration files status:")

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
            logger.debug(f"✅ {config}")
        else:
            logger.debug(f"❌ {config} (missing)")

    logger.info("\nTo continue fixing, run: python run_complete_lint_fix.py")
    logger.debug("To validate, run: python validate_lint_fixes.py")


if __name__ == "__main__":
    main()
