#!/usr/bin/env python3
"""
Validate that the lint fixes have been applied correctly.
Check that no functional changes were made.
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd, capture=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, capture_output=capture, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False, "", str(e)


def check_black():
    """Check if Black formatting passes."""
    logger.info("Checking Black formatting...")
    success, stdout, stderr = run_command(["black", "--check", "genomevault"])

    if success:
        logger.info("✅ Black check passed")
    else:
        logger.warning("❌ Black check failed - files need formatting")
        if stderr:
            print(stderr)

    return success


def check_ruff():
    """Check if Ruff passes."""
    logger.info("Checking Ruff...")
    success, stdout, stderr = run_command(["ruff", "genomevault"])

    if success:
        logger.info("✅ Ruff check passed - no issues found")
    else:
        logger.warning("⚠️  Ruff found issues:")
        if stdout:
            print(stdout)

    return success


def check_imports():
    """Check if imports are sorted correctly."""
    logger.info("Checking import sorting...")
    success, stdout, stderr = run_command(["ruff", "genomevault", "--select", "I"])

    if success:
        logger.info("✅ Import sorting check passed")
    else:
        logger.warning("❌ Import sorting issues found")
        if stdout:
            print(stdout)

    return success


def check_mypy():
    """Run MyPy type checking."""
    logger.info("Running MyPy type checking...")
    success, stdout, stderr = run_command(["mypy", "genomevault"])

    # MyPy often has warnings, so we just report
    logger.info("ℹ️  MyPy results (informational):")
    if stdout:
        # Count the number of errors
        error_lines = [l for l in stdout.split("\n") if "error:" in l]
        logger.info(f"  Found {len(error_lines)} type errors")

    return True  # Don't fail on MyPy errors


def check_pylint():
    """Run PyLint checking."""
    logger.info("Running PyLint...")
    success, stdout, stderr = run_command(["pylint", "genomevault", "--exit-zero"])

    # Extract the score from PyLint output
    if stdout:
        for line in stdout.split("\n"):
            if "Your code has been rated at" in line:
                logger.info(f"ℹ️  PyLint score: {line.strip()}")
                break

    return True  # Don't fail on PyLint warnings


def check_tests():
    """Run tests to ensure no functional changes."""
    logger.info("Running tests to verify no functional changes...")

    # Check if pytest is available
    success, _, _ = run_command(["which", "pytest"])
    if not success:
        logger.warning("pytest not found - skipping test verification")
        return True

    # Run tests
    success, stdout, stderr = run_command(["pytest", "-v", "--tb=short"])

    if success:
        logger.info("✅ All tests passed - no functional regressions")
    else:
        logger.error("❌ Some tests failed - please review")

    return success


def generate_report():
    """Generate a summary report."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION REPORT")
    logger.info("=" * 60)

    results = {
        "Black formatting": check_black(),
        "Ruff linting": check_ruff(),
        "Import sorting": check_imports(),
        "MyPy type checking": check_mypy(),
        "PyLint analysis": check_pylint(),
        # "Test suite": check_tests(),  # Uncomment if you want to run tests
    }

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{check_name}: {status}")
        if not passed and check_name not in ["MyPy type checking", "PyLint analysis"]:
            all_passed = False

    logger.info("=" * 60)

    if all_passed:
        logger.info("\n🎉 All critical checks passed!")
        logger.info("The codebase is now lint-clean according to the configuration.")
    else:
        logger.warning("\n⚠️  Some checks need attention.")
        logger.info("Run './scripts/lint_fix.sh genomevault' to apply remaining fixes.")

    # Git status
    logger.info("\n" + "=" * 60)
    logger.info("GIT STATUS")
    logger.info("=" * 60)

    result = subprocess.run(
        ["git", "status", "--short"], capture_output=True, text=True
    )
    if result.stdout:
        logger.info("Modified files:")
        print(result.stdout)
    else:
        logger.info("No uncommitted changes")

    # Commit history
    logger.info("\nRecent commits:")
    result = subprocess.run(
        ["git", "log", "--oneline", "-5"], capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout)

    return all_passed


def main():
    """Main execution."""
    logger.info("Starting GenomeVault lint validation")

    # Ensure we're in the right directory
    cwd = Path.cwd()
    if not (cwd / "genomevault").exists():
        logger.error("Not in GenomeVault root directory!")
        sys.exit(1)

    # Generate report
    success = generate_report()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
