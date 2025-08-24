#!/usr/bin/env python3

"""Quick validation script for lint clean implementation."""

import subprocess
import sys
from pathlib import Path
from genomevault.utils.logging import get_logger
logger = get_logger(__name__)



def run_command(cmd, description):
    """Run a command and return success status."""
    logger.debug(f"\n🔧 {description}")
    logger.debug(f"Running: {' '.join(cmd)}")
    logger.debug("-" * 50)

    try:
        result = subprocess.run(
            cmd,
            cwd=".",
            capture_output=True,
            text=True,
            timeout=60,
        )

        logger.debug(f"Exit code: {result.returncode}")
        if result.stdout:
            logger.debug(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.debug(f"STDERR:\n{result.stderr}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        logger.debug("❌ Command timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error running command: {e}")
        return False


def main():
    logger.debug("GenomeVault Lint Clean Validation")
    logger.debug("=" * 50)

    # Check that we're in the right directory
    ruff_config = Path("./.ruff.toml")
    if not ruff_config.exists():
        logger.debug("❌ .ruff.toml not found - not in project root")
        sys.exit(1)

    logger.debug("✅ Found .ruff.toml - in correct directory")

    # Run validation sequence
    validations = [
        (["ruff", "format", "--check", "."], "Ruff format check"),
        (["ruff", "check", "."], "Ruff lint check"),
    ]

    all_passed = True

    for cmd, desc in validations:
        success = run_command(cmd, desc)
        if success:
            logger.info(f"✅ {desc} PASSED")
        else:
            logger.error(f"❌ {desc} FAILED")
            all_passed = False

    # Summary
    logger.debug("\n" + "=" * 50)
    if all_passed:
        logger.info("🎉 ALL VALIDATIONS PASSED!")
        logger.info("Lint clean implementation successful.")
    else:
        logger.error("❌ Some validations failed.")
        logger.debug("Review the output above for details.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
