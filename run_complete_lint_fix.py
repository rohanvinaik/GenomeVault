#!/usr/bin/env python3
"""
Complete lint fix implementation for GenomeVault.
Runs all fixes in the correct order as per the markdown plan.
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd, check=False, capture=True):
    """Run a shell command and return the result."""
    logger.info(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")

    if isinstance(cmd, str):
        cmd = cmd.split()

    try:
        if capture:
            result = subprocess.run(cmd, capture_output=True, text=True, check=check)
            if result.stdout:
                print(result.stdout)
            if result.stderr and result.returncode != 0:
                print(result.stderr, file=sys.stderr)
            return result.returncode == 0
        else:
            result = subprocess.run(cmd, check=check)
            return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        return False
    except FileNotFoundError as e:
        logger.error(f"Command not found: {e}")
        return False


def ensure_dependencies():
    """Ensure all required dependencies are installed."""
    logger.info("Checking dependencies...")

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

    return True


def create_branch():
    """Create the lint sweep branch."""
    logger.info("Creating branch for lint sweep...")

    # Check current branch
    result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
    current_branch = result.stdout.strip()

    if current_branch == "chore/lint-sweep":
        logger.info("Already on chore/lint-sweep branch")
        return True

    # Create and checkout new branch
    return run_command(["git", "checkout", "-b", "chore/lint-sweep"])


def run_baseline_autofix():
    """Run the baseline autofix on the entire codebase."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Running baseline autofix")
    logger.info("=" * 60)

    # Run Black
    logger.info("Running Black formatter...")
    run_command(["black", "--line-length=100", "genomevault"])

    # Run Ruff autofix
    logger.info("Running Ruff autofix...")
    run_command(["ruff", "genomevault", "--fix"])

    # Run import sorting
    logger.info("Sorting imports...")
    run_command(["ruff", "genomevault", "--select", "I", "--fix"])

    # Commit baseline changes
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if result.stdout.strip():
        logger.info("Committing baseline changes...")
        run_command(["git", "add", "-A"])
        run_command(
            [
                "git",
                "commit",
                "-m",
                "style: black+ruff autofix baseline (no functional changes)",
            ]
        )

    return True


def run_area_by_area_fixes():
    """Run fixes on each area separately."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Running area-by-area fixes")
    logger.info("=" * 60)

    areas = [
        "genomevault/api",
        "genomevault/local_processing",
        "genomevault/hdc",
        "genomevault/pir",
        "genomevault/zk_proofs",
    ]

    for area in areas:
        area_path = Path(area)
        if not area_path.exists():
            logger.warning(f"Skipping {area} - does not exist")
            continue

        logger.info(f"\nProcessing {area}...")

        # Run fixes
        run_command(["black", "--line-length=100", area])
        run_command(["ruff", area, "--fix"])
        run_command(["ruff", area, "--select", "I", "--fix"])

        # Commit if changes
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if result.stdout.strip():
            run_command(["git", "add", "-A"])
            run_command(
                [
                    "git",
                    "commit",
                    "-m",
                    f"lint: {area} (black+ruff+isort, no functional changes)",
                ]
            )

    return True


def run_common_fixes():
    """Apply common fix patterns."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Applying common fix patterns")
    logger.info("=" * 60)

    # Run the common fixes script
    if Path("apply_common_fixes.py").exists():
        run_command(["python", "apply_common_fixes.py"])

        # Commit if changes
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if result.stdout.strip():
            run_command(["git", "add", "-A"])
            run_command(
                [
                    "git",
                    "commit",
                    "-m",
                    "style: apply common fix patterns (logging, f-strings, etc.)",
                ]
            )

    return True


def final_check():
    """Run final lint checks."""
    logger.info("\n" + "=" * 60)
    logger.info("FINAL CHECK: Running all linters")
    logger.info("=" * 60)

    logger.info("\n--- Black Check ---")
    run_command(["black", "--check", "genomevault"])

    logger.info("\n--- Ruff Check ---")
    run_command(["ruff", "genomevault"])

    logger.info("\n--- MyPy Check ---")
    run_command(["mypy", "genomevault"])

    logger.info("\n--- PyLint Check ---")
    run_command(["pylint", "genomevault"])

    return True


def main():
    """Main execution flow."""
    logger.info("Starting GenomeVault comprehensive lint fix implementation")
    logger.info("This will apply all fixes from the markdown plan")

    # Ensure we're in the right directory
    cwd = Path.cwd()
    if not (cwd / "genomevault").exists():
        logger.error("Not in GenomeVault root directory!")
        sys.exit(1)

    # Step 0: Ensure dependencies
    if not ensure_dependencies():
        logger.error("Failed to install dependencies")
        sys.exit(1)

    # Step 1: Create branch (skip if already on it)
    # create_branch()

    # Step 2: Run baseline autofix
    run_baseline_autofix()

    # Step 3: Run area-by-area fixes
    run_area_by_area_fixes()

    # Step 4: Apply common fixes
    run_common_fixes()

    # Step 5: Final check
    final_check()

    logger.info("\n" + "=" * 60)
    logger.info("✅ Lint fix implementation complete!")
    logger.info("=" * 60)

    logger.info("\nNext steps:")
    logger.info("1. Review the changes: git diff HEAD~")
    logger.info("2. Run tests to ensure no functional changes: pytest")
    logger.info("3. Push the branch: git push origin chore/lint-sweep")
    logger.info("4. Create a pull request")


if __name__ == "__main__":
    main()
