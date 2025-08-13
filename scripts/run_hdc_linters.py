#!/usr/bin/env python3
"""
Run basic linter checks for HDC implementation
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    """Run a command and return success status"""
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        logger.exception("Unhandled exception")
        print(f"Error running command: {e}")
        return False
        raise


def main():
    """Run all linter checks"""
    # Change to project root
    project_root = Path(__file__).parent.parent

    # Define what to check
    hdc_module = project_root / "genomevault" / "hypervector_transform"
    hdc_tests = [
        project_root / "tests" / "test_hdc_implementation.py",
        project_root / "tests" / "test_hdc_quality.py",
        project_root / "tests" / "property" / "test_hdc_properties.py",
        project_root / "tests" / "adversarial" / "test_hdc_adversarial.py",
    ]

    results = {}

    # 1. Black formatting check
    black_files = [str(hdc_module)] + [str(f) for f in hdc_tests if f.exists()]
    black_result = run_command(["black", "--check", "--diff"] + black_files)
    results["black"] = black_result

    # 2. isort import sorting check
    isort_result = run_command(["isort", "--check-only", "--diff", str(hdc_module)])
    results["isort"] = isort_result

    # 3. Flake8 linting
    flake8_result = run_command(
        [
            "flake8",
            str(hdc_module),
            "--max-line-length=100",
            "--extend-ignore=E203,W503",
        ]
    )
    results["flake8"] = flake8_result

    # 4. Type checking with mypy (if available)
    try:
        mypy_result = run_command(
            [
                "mypy",
                str(hdc_module),
                "--ignore-missing-imports",
                "--no-strict-optional",
            ]
        )
        results["mypy"] = mypy_result
    except (FileNotFoundError, subprocess.CalledProcessError, Exception):
        logger.exception("Unhandled exception")
        results["mypy"] = None
        raise

    # 5. Run quick tests
    test_result = run_command(
        [
            "pytest",
            str(project_root / "tests" / "test_hdc_implementation.py::TestHDCDeterminism"),
            "-v",
            "--tb=short",
        ]
    )
    results["tests"] = test_result

    # Summary
    print("=" * 60)

    all_passed = True
    for check, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "PASSED ✓"
        else:
            status = "FAILED ✗"
            all_passed = False
        pass  # Debug print removed

    if all_passed:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
