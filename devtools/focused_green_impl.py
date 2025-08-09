from genomevault.utils.logging import get_logger
logger = get_logger(__name__)


#!/usr/bin/env python3
"""
Focused Green Toolchain Implementation
"""

import os
import subprocess
from pathlib import Path


def main():
    project_root = Path("/Users/rohanvinaik/genomevault")
    os.chdir(project_root)

    logger.debug("Starting focused Green Toolchain implementation...")

    # First, install updated dependencies
    logger.debug("\n=== Installing updated dependencies ===")
    subprocess.run(["pip", "install", "-e", ".[dev]"], check=False)

    # Run quick validation
    logger.debug("\n=== Running validation ===")

    logger.debug("1. Checking ruff...")
    result = subprocess.run(["ruff", "check", "."], capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("✓ Ruff check passed")
    else:
        logger.debug(f"⚠ Ruff found issues:\n{result.stdout}{result.stderr}")

    logger.debug("\n2. Checking mypy on core packages...")
    result = subprocess.run(
        ["mypy", "genomevault/hypervector", "genomevault/core"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info("✓ MyPy check passed")
    else:
        logger.debug(f"⚠ MyPy found issues:\n{result.stdout}{result.stderr}")

    logger.debug("\n3. Running pytest...")
    result = subprocess.run(["pytest", "-q", "--tb=short"], capture_output=True, text=True)
    logger.debug(f"Pytest result: {result.returncode}")
    if result.stdout:
        logger.debug(f"Output:\n{result.stdout}")
    if result.stderr:
        logger.error(f"Errors:\n{result.stderr}")

    logger.info("\n=== Green Toolchain basic implementation completed ===")


if __name__ == "__main__":
    main()
