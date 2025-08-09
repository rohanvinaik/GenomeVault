#!/usr/bin/env python3
"""
Area-by-area lint fix implementation for GenomeVault.
This script applies fixes to each module systematically.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Define the areas to process in order
AREAS = [
    "genomevault/api",
    "genomevault/local_processing",
    "genomevault/hdc",
    "genomevault/pir",
    "genomevault/zk_proofs",
    "genomevault/core",
    "genomevault/crypto",
    "genomevault/blockchain",
    "genomevault/clinical",
    "genomevault/federated",
    "genomevault/federation",
    "genomevault/governance",
    "genomevault/hypervector",
    "genomevault/hypervector_transform",
    "genomevault/integration",
    "genomevault/kan",
    "genomevault/ledger",
    "genomevault/nanopore",
    "genomevault/observability",
    "genomevault/pipelines",
    "genomevault/security",
    "genomevault/utils",
    "genomevault/zk",
    "genomevault/cli",
    "genomevault/config",
    "genomevault/contracts",
    "genomevault/advanced_analysis",
    "genomevault/benchmarks",
]


def run_command(cmd, check=False):
    """Run a shell command and return the result."""
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(result.stderr, file=sys.stderr)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        return False


def process_area(area):
    """Process a single area with all lint fixes."""
    area_path = Path(area)

    if not area_path.exists():
        logger.warning(f"Skipping {area} - path does not exist")
        return False

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing area: {area}")
    logger.info(f"{'='*60}")

    # Apply Black formatting
    logger.info(f"Applying Black to {area}...")
    run_command(["black", "--line-length=100", area])

    # Apply Ruff fixes
    logger.info(f"Applying Ruff fixes to {area}...")
    run_command(["ruff", area, "--fix"])

    # Apply import sorting
    logger.info(f"Sorting imports in {area}...")
    run_command(["ruff", area, "--select", "I", "--fix"])

    # Check for remaining issues (informational)
    logger.info(f"Checking remaining issues in {area}...")
    run_command(["ruff", area])

    return True


def commit_area(area):
    """Commit changes for a specific area."""
    # Check if there are changes
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)

    if result.stdout.strip():
        logger.info(f"Committing changes for {area}")
        run_command(["git", "add", "-A"])
        message = f"lint: {area} (black+ruff+isort, no functional changes)"
        run_command(["git", "commit", "-m", message])
        return True
    else:
        logger.info(f"No changes to commit for {area}")
        return False


def main():
    """Main execution flow."""
    logger.info("Starting area-by-area lint fix implementation")

    # Ensure we're in the right directory
    cwd = Path.cwd()
    if not (cwd / "genomevault").exists():
        logger.error("Not in GenomeVault root directory!")
        sys.exit(1)

    # Process each area
    for area in AREAS:
        if process_area(area):
            commit_area(area)
            # Small delay between areas
            time.sleep(0.5)

    # Process top-level files
    logger.info("\n" + "=" * 60)
    logger.info("Processing top-level genomevault files")
    logger.info("=" * 60)

    top_level_files = list(Path("genomevault").glob("*.py"))
    if top_level_files:
        for f in top_level_files:
            logger.info(f"Processing {f}")
            run_command(["black", "--line-length=100", str(f)])
            run_command(["ruff", str(f), "--fix"])

        # Commit top-level changes
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if result.stdout.strip():
            run_command(["git", "add", "-A"])
            run_command(
                [
                    "git",
                    "commit",
                    "-m",
                    "lint: genomevault top-level files (black+ruff, no functional changes)",
                ]
            )

    logger.info("\n" + "=" * 60)
    logger.info("Area-by-area lint fix implementation complete!")
    logger.info("=" * 60)

    # Final summary
    logger.info("\nFinal check of the entire codebase:")
    run_command(["ruff", "genomevault"])

    logger.info("\nYou can now run: ./scripts/lint_check.sh genomevault")
    logger.info("to verify all fixes have been applied.")


if __name__ == "__main__":
    main()
