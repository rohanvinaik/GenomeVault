from typing import Any, Dict

# !/usr/bin/env python3
"""
Run basic linter checks for HDC implementation
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd) -> None:  # noqa: C901
    """TODO: Add docstring for run_command"""
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

        def main() -> None:
            """TODO: Add docstring for main"""

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
    print("\n1. Checking code formatting with Black...")
    black_files = [str(hdc_module)] + [str(f) for f in hdc_tests if f.exists()]
    black_result = run_command(["black", "--check", "--diff"] + black_files)
    results["black"] = black_result

    # 2. isort import sorting check
    print("\n2. Checking import sorting with isort...")
    isort_result = run_command(["isort", "--check-only", "--diff", str(hdc_module)])
    results["isort"] = isort_result

    # 3. Flake8 linting
    print("\n3. Running flake8 linter...")
    flake8_result = run_command(
        ["flake8", str(hdc_module), "--max-line-length=100", "--extend-ignore=E203,W503"]
    )
    results["flake8"] = flake8_result

    # 4. Type checking with mypy (if available)
    print("\n4. Running mypy type checker...")
    try:
        mypy_result = run_command(
            ["mypy", str(hdc_module), "--ignore-missing-imports", "--no-strict-optional"]
        )
        results["mypy"] = mypy_result
    except:
        print("mypy not available, skipping type checking")
        results["mypy"] = None

    # 5. Run quick tests
    print("\n5. Running quick unit tests...")
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
    print("\n" + "=" * 60)
    print("LINTER CHECK SUMMARY")
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
        print(f"{check:.<20} {status}")

    print("\n" + "=" * 60)
    if all_passed:
        print("All checks passed! ✓")
        return 0
    else:
        print("Some checks failed. ✗")
        print("\nTo fix formatting issues, run:")
        print("  black genomevault/hypervector_transform tests/")
        print("  isort genomevault/hypervector_transform tests/")
        return 1


if __name__ == "__main__":
    sys.exit(main())  # noqa: F821
