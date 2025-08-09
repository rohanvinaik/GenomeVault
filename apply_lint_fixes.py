#!/usr/bin/env python3
"""
Systematic lint fix implementation for GenomeVault codebase.
This script applies black, ruff, and isort fixes as per the plan.
"""

import logging
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd, check=False):
    """Run a shell command and return the result."""
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        return False


def ensure_dependencies():
    """Ensure required tools are installed."""
    deps = ["black", "ruff", "mypy", "pylint"]
    missing = []

    for dep in deps:
        result = subprocess.run(["which", dep], capture_output=True)
        if result.returncode != 0:
            missing.append(dep)

    if missing:
        logger.info(f"Installing missing dependencies: {missing}")
        run_command(["pip", "install"] + missing)
    else:
        logger.info("All dependencies are installed")


def apply_black_formatting(path="genomevault"):
    """Apply black formatting to the codebase."""
    logger.info("Applying Black formatting...")
    return run_command(["black", "--line-length=100", path])


def apply_ruff_fixes(path="genomevault"):
    """Apply ruff auto-fixes."""
    logger.info("Applying Ruff auto-fixes...")
    # First pass: general fixes
    run_command(["ruff", path, "--fix"])
    # Second pass: import sorting
    run_command(["ruff", path, "--select", "I", "--fix"])
    return True


def check_remaining_issues(path="genomevault"):
    """Check for remaining issues."""
    logger.info("Checking remaining issues...")

    # Run ruff check
    logger.info("Ruff check:")
    run_command(["ruff", path])

    # Run mypy (informational)
    logger.info("MyPy check (informational):")
    run_command(["mypy", path])

    # Run pylint (informational)
    logger.info("PyLint check (informational):")
    run_command(["pylint", path])


def commit_changes(message):
    """Commit the changes."""
    logger.info(f"Committing changes: {message}")
    run_command(["git", "add", "-A"])
    run_command(["git", "commit", "-m", message])


def main():
    """Main execution flow."""
    logger.info("Starting GenomeVault lint fix implementation")

    # Ensure we're in the right directory
    cwd = Path.cwd()
    if not (cwd / "genomevault").exists():
        logger.error("Not in GenomeVault root directory!")
        sys.exit(1)

    # Step 1: Ensure dependencies
    ensure_dependencies()

    # Step 2: Apply Black formatting
    if apply_black_formatting():
        logger.info("Black formatting applied successfully")

    # Step 3: Apply Ruff fixes
    if apply_ruff_fixes():
        logger.info("Ruff fixes applied successfully")

    # Step 4: Check remaining issues
    check_remaining_issues()

    # Step 5: Commit if there are changes
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if result.stdout.strip():
        commit_changes("style: black+ruff autofix baseline (no functional changes)")
        logger.info("Changes committed successfully")
    else:
        logger.info("No changes to commit")

    logger.info("Lint fix implementation complete!")


if __name__ == "__main__":
    main()
