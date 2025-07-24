#!/usr/bin/env python3
"""
Comprehensive linter check for GenomeVault
"""
import os
import subprocess
import sys
from typing import Dict, List, Tuple


class Colors:
    """Terminal colors"""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def run_command(cmd: str) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")


def print_result(passed: bool, message: str):
    """Print a result with appropriate color"""
    if passed:
        print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")
    else:
        print(f"{Colors.RED}❌ {message}{Colors.RESET}")


def check_black() -> Tuple[bool, List[str]]:
    """Check Black formatting"""
    print_header("BLACK FORMATTER")

    returncode, stdout, stderr = run_command("black --check .")

    if returncode == 0:
        print_result(True, "All files are properly formatted")
        return True, []
    else:
        # Extract files that need formatting
        files = []
        for line in stdout.split("\n"):
            if line.startswith("would reformat"):
                files.append(line.replace("would reformat ", "").strip())

        print_result(False, f"{len(files)} files need formatting")
        if files:
            print(f"\n{Colors.YELLOW}Files needing formatting:{Colors.RESET}")
            for f in files[:10]:  # Show first 10
                print(f"  • {f}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")

        return False, files


def check_isort() -> Tuple[bool, List[str]]:
    """Check import sorting"""
    print_header("ISORT (Import Sorting)")

    returncode, stdout, stderr = run_command("isort --check-only --diff .")

    if returncode == 0:
        print_result(True, "All imports are properly sorted")
        return True, []
    else:
        # Count files with import issues
        files = []
        for line in stdout.split("\n"):
            if line.startswith("---") and ".py" in line:
                files.append(line.replace("---", "").strip())

        print_result(False, f"Import sorting issues found")
        if files:
            print(f"\n{Colors.YELLOW}Files with import issues:{Colors.RESET}")
            for f in files[:5]:
                print(f"  • {f}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")

        return False, files


def check_flake8() -> Tuple[bool, Dict[str, int]]:
    """Check Flake8 style violations"""
    print_header("FLAKE8 (Style Checker)")

    returncode, stdout, stderr = run_command(
        "flake8 genomevault/ tests/ examples/ --count --statistics"
    )

    if returncode == 0:
        print_result(True, "No style violations found")
        return True, {}
    else:
        # Parse statistics
        violations = {}
        total = 0

        for line in stdout.split("\n"):
            if line.strip():
                parts = line.split()
                if (
                    len(parts) >= 2
                    and parts[1].startswith("E")
                    or parts[1].startswith("F")
                    or parts[1].startswith("W")
                ):
                    code = parts[1]
                    count = int(parts[0])
                    violations[code] = count
                    total += count

        print_result(False, f"{total} style violations found")

        if violations:
            print(f"\n{Colors.YELLOW}Top violations:{Colors.RESET}")
            sorted_violations = sorted(violations.items(), key=lambda x: x[1], reverse=True)
            for code, count in sorted_violations[:5]:
                print(f"  • {code}: {count} occurrences")

        return False, violations


def check_pylint() -> Tuple[bool, float]:
    """Check Pylint code quality"""
    print_header("PYLINT (Code Quality)")

    returncode, stdout, stderr = run_command("pylint genomevault/ --exit-zero")

    # Extract score from output
    score = 0.0
    for line in stdout.split("\n"):
        if "Your code has been rated at" in line:
            try:
                score_str = line.split("rated at")[1].split("/")[0].strip()
                score = float(score_str)
            except:
                pass

    passed = score >= 7.0  # Reasonable threshold

    if passed:
        print_result(True, f"Code quality score: {score:.2f}/10.00")
    else:
        print_result(False, f"Code quality score: {score:.2f}/10.00 (target: 7.00)")

    # Show some common issues
    issues = []
    for line in stdout.split("\n"):
        if ":" in line and ("genomevault/" in line or "tests/" in line):
            if any(x in line for x in ["C0111", "C0103", "R0913", "R0903"]):
                continue  # Skip disabled warnings
            issues.append(line.strip())

    if issues and len(issues) > 0:
        print(f"\n{Colors.YELLOW}Sample issues:{Colors.RESET}")
        for issue in issues[:5]:
            print(f"  • {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more")

    return passed, score


def main():
    """Run all linters and provide summary"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print(f"{Colors.BOLD}Running comprehensive linter checks for GenomeVault...{Colors.RESET}")

    # Run all checks
    black_passed, black_files = check_black()
    isort_passed, isort_files = check_isort()
    flake8_passed, flake8_violations = check_flake8()
    pylint_passed, pylint_score = check_pylint()

    # Summary
    print_header("SUMMARY")

    all_passed = black_passed and isort_passed and flake8_passed and pylint_passed

    print(f"\n{Colors.BOLD}Linter Results:{Colors.RESET}")
    print_result(
        black_passed, f"Black: {'PASSED' if black_passed else f'FAILED ({len(black_files)} files)'}"
    )
    print_result(isort_passed, f"isort: {'PASSED' if isort_passed else 'FAILED'}")
    print_result(
        flake8_passed,
        f"Flake8: {'PASSED' if flake8_passed else f'FAILED ({sum(flake8_violations.values())} violations)'}",
    )
    print_result(
        pylint_passed,
        f"Pylint: {'PASSED' if pylint_passed else 'FAILED'} (score: {pylint_score:.2f}/10.00)",
    )

    print(f"\n{Colors.BOLD}Quick Fixes:{Colors.RESET}")

    if not black_passed:
        print(f"\n{Colors.YELLOW}To fix Black formatting:{Colors.RESET}")
        print("  black .")

    if not isort_passed:
        print(f"\n{Colors.YELLOW}To fix import sorting:{Colors.RESET}")
        print("  isort .")

    if not flake8_passed:
        print(f"\n{Colors.YELLOW}To see detailed Flake8 violations:{Colors.RESET}")
        print("  flake8 genomevault/ tests/ examples/")

    if not pylint_passed:
        print(f"\n{Colors.YELLOW}To see detailed Pylint report:{Colors.RESET}")
        print("  pylint genomevault/")

    if all_passed:
        print(
            f"\n{Colors.GREEN}{Colors.BOLD}🎉 All linters passed! Your code is clean.{Colors.RESET}"
        )
        return 0
    else:
        print(
            f"\n{Colors.RED}{Colors.BOLD}Some linters failed. Please fix the issues above.{Colors.RESET}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
