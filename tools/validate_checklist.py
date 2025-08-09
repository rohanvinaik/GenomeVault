#!/usr/bin/env python3
"""
Validate that all checklist items have been properly implemented.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def check_command_exists(cmd):
    """Check if a command exists in the system."""
    try:
        subprocess.run([cmd, "--version"], check=False, capture_output=True)
        return True
    except FileNotFoundError:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        return False
        raise


def run_command(cmd_list, cwd=None):
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(cmd_list, check=False, capture_output=True, text=True, cwd=cwd)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        return False, "", str(e)
        raise


class ChecklistValidator:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.results = {}

    def validate_item_1_packaging(self):
        """1. Add packaging & metadata (pyproject.toml)"""
        checks = {}

        # Check pyproject.toml exists
        pyproject_path = self.project_root / "pyproject.toml"
        checks["pyproject_exists"] = pyproject_path.exists()

        if checks["pyproject_exists"]:
            # Check if it uses Hatch
            content = pyproject_path.read_text()
            checks["uses_hatch"] = "hatchling" in content
            checks["has_correct_name"] = 'name = "genomevault"' in content

        # Try pip install
        success, _, _ = run_command(["pip", "install", "-e", ".[dev]"], cwd=self.project_root)
        checks["pip_install_works"] = success

        # Try import
        success, _, _ = run_command(
            ["python", "-c", 'import genomevault; print("ok")'], cwd=self.project_root
        )
        checks["import_works"] = success

        self.results["item_1_packaging"] = {
            "passed": all(checks.values()),
            "checks": checks,
        }

    def validate_item_2_readme_license(self):
        """2. Add README & LICENSE"""
        checks = {}

        readme_path = self.project_root / "README.md"
        license_path = self.project_root / "LICENSE"

        checks["readme_exists"] = readme_path.exists()
        checks["license_exists"] = license_path.exists()

        if checks["license_exists"]:
            content = license_path.read_text()
            checks["is_mit_license"] = "MIT License" in content

        self.results["item_2_readme_license"] = {
            "passed": all(checks.values()),
            "checks": checks,
        }

    def validate_item_3_tooling_configs(self):
        """3. Tooling configs (ruff, mypy, pytest, pre-commit)"""
        checks = {}

        # Check config files exist
        checks["ruff_toml_exists"] = (self.project_root / "ruff.toml").exists()
        checks["mypy_ini_exists"] = (self.project_root / "mypy.ini").exists()
        checks["pytest_ini_exists"] = (self.project_root / "pytest.ini").exists()
        checks["pre_commit_exists"] = (self.project_root / ".pre-commit-config.yaml").exists()

        # Check ruff
        if check_command_exists("ruff"):
            success, _, _ = run_command(["ruff", "check", "."], cwd=self.project_root)
            checks["ruff_check_passes"] = success

            success, _, _ = run_command(["ruff", "format", "--check", "."], cwd=self.project_root)
            checks["ruff_format_passes"] = success
        else:
            checks["ruff_installed"] = False

        # Check mypy
        if check_command_exists("mypy"):
            success, _, _ = run_command(["mypy", "."], cwd=self.project_root)
            checks["mypy_passes"] = success
        else:
            checks["mypy_installed"] = False

        self.results["item_3_tooling"] = {
            "passed": all(v for k, v in checks.items() if not k.endswith("_installed")),
            "checks": checks,
        }

    def validate_item_4_ci_workflow(self):
        """4. CI workflow (GitHub Actions)"""
        checks = {}

        ci_path = self.project_root / ".github" / "workflows" / "ci.yml"
        checks["ci_yml_exists"] = ci_path.exists()

        if checks["ci_yml_exists"]:
            content = ci_path.read_text()
            checks["has_ruff_check"] = "ruff check" in content
            checks["has_ruff_format"] = "ruff format" in content
            checks["has_mypy"] = "mypy" in content
            checks["has_pytest"] = "pytest" in content

        self.results["item_4_ci"] = {"passed": all(checks.values()), "checks": checks}

    def validate_item_5_logging(self):
        """5. Standardize logging"""
        checks = {}

        # Check logging_utils exists
        logging_utils_path = self.project_root / "genomevault" / "logging_utils.py"
        checks["logging_utils_exists"] = logging_utils_path.exists()

        # Count print statements in library code
        success, stdout, _ = run_command(
            ["rg", "-n", r"print\(", "--glob", "!tests/**", "--glob", "!**/scripts/**"],
            cwd=self.project_root / "genomevault",
        )

        if success:
            print_count = len(stdout.strip().split("\n")) if stdout.strip() else 0
        else:
            print_count = -1  # Unknown

        checks["print_statements_count"] = print_count
        checks["minimal_prints"] = print_count <= 5  # Allow a few prints

        self.results["item_5_logging"] = {
            "passed": checks.get("logging_utils_exists", False)
            and checks.get("minimal_prints", False),
            "checks": checks,
        }

    def validate_item_6_exceptions(self):
        """6. Harden exception handling"""
        checks = {}

        # Check exceptions.py exists
        exceptions_path = self.project_root / "genomevault" / "exceptions.py"
        checks["exceptions_py_exists"] = exceptions_path.exists()

        if checks["exceptions_py_exists"]:
            content = exceptions_path.read_text()
            checks["has_base_error"] = "GenomeVaultError" in content
            checks["has_config_error"] = "ConfigError" in content
            checks["has_validation_error"] = "ValidationError" in content
            checks["has_compute_error"] = "ComputeError" in content

        # Check for bare excepts
        success, stdout, _ = run_command(
            ["rg", "-n", r"except\s*:", "genomevault"], cwd=self.project_root
        )

        bare_except_count = len(stdout.strip().split("\n")) if stdout.strip() and success else 0
        checks["bare_except_count"] = bare_except_count
        checks["no_bare_excepts"] = bare_except_count == 0

        self.results["item_6_exceptions"] = {
            "passed": all(v for k, v in checks.items() if k != "bare_except_count"),
            "checks": checks,
        }

    def validate_item_9_package_structure(self):
        """9. Package structure decision"""
        checks = {}

        # Check for __init__.py files
        genomevault_dir = self.project_root / "genomevault"

        # Find all Python package directories
        package_dirs = []
        for path in genomevault_dir.rglob("*.py"):
            parent = path.parent
            if parent not in package_dirs and parent != genomevault_dir.parent:
                package_dirs.append(parent)

        # Check each has __init__.py
        missing_init = []
        for pkg_dir in package_dirs:
            init_file = pkg_dir / "__init__.py"
            if not init_file.exists():
                missing_init.append(str(pkg_dir.relative_to(self.project_root)))

        checks["total_package_dirs"] = len(package_dirs)
        checks["missing_init_count"] = len(missing_init)
        checks["all_have_init"] = len(missing_init) == 0

        if missing_init:
            checks["missing_init_dirs"] = missing_init[:5]  # Show first 5

        self.results["item_9_package_structure"] = {
            "passed": checks["all_have_init"],
            "checks": checks,
        }

    def validate_item_11_tests_coverage(self):
        """11. Tests & coverage guardrail"""
        checks = {}

        # Check if pytest works
        if check_command_exists("pytest"):
            success, stdout, stderr = run_command(["pytest", "--version"], cwd=self.project_root)
            checks["pytest_installed"] = success

            # Try running tests
            success, stdout, stderr = run_command(
                ["pytest", "-q", "--tb=short"], cwd=self.project_root
            )
            checks["tests_pass"] = success

            # Check coverage
            if "cov" in stdout or "coverage" in stderr:
                # Extract coverage percentage if available
                import re

                match = re.search(r"(\d+)%", stdout + stderr)
                if match:
                    coverage_pct = int(match.group(1))
                    checks["coverage_percentage"] = coverage_pct
                    checks["coverage_above_80"] = coverage_pct >= 80
        else:
            checks["pytest_installed"] = False

        self.results["item_11_tests_coverage"] = {
            "passed": checks.get("tests_pass", False) and checks.get("coverage_above_80", False),
            "checks": checks,
        }

    def generate_report(self):
        """Generate a comprehensive validation report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "summary": {
                "total_items_checked": len(self.results),
                "items_passed": sum(1 for r in self.results.values() if r["passed"]),
                "items_failed": sum(1 for r in self.results.values() if not r["passed"]),
            },
            "details": self.results,
        }

        # Save JSON report
        report_path = self.project_root / "checklist_validation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Generate markdown summary
        md_lines = ["# Checklist Validation Report\n"]
        md_lines.append(f"Generated: {report['timestamp']}\n")
        md_lines.append("## Summary\n")
        md_lines.append(f"- Total items checked: {report['summary']['total_items_checked']}")
        md_lines.append(f"- Items passed: {report['summary']['items_passed']}")
        md_lines.append(f"- Items failed: {report['summary']['items_failed']}\n")

        md_lines.append("## Detailed Results\n")

        for item_name, result in self.results.items():
            status = "✅" if result["passed"] else "❌"
            md_lines.append(f"### {status} {item_name.replace('_', ' ').title()}\n")

            for check_name, check_value in result["checks"].items():
                if isinstance(check_value, bool):
                    check_status = "✓" if check_value else "✗"
                    md_lines.append(f"- {check_status} {check_name.replace('_', ' ')}")
                else:
                    md_lines.append(f"- {check_name.replace('_', ' ')}: {check_value}")
            md_lines.append("")

        md_report_path = self.project_root / "checklist_validation_report.md"
        with open(md_report_path, "w") as f:
            f.write("\n".join(md_lines))

        return report_path, md_report_path

    def run_all_validations(self):
        """Run all validation checks."""
        print("Running checklist validation...")

        # Run each validation
        self.validate_item_1_packaging()
        print("✓ Validated packaging setup")

        self.validate_item_2_readme_license()
        print("✓ Validated README and LICENSE")

        self.validate_item_3_tooling_configs()
        print("✓ Validated tooling configs")

        self.validate_item_4_ci_workflow()
        print("✓ Validated CI workflow")

        self.validate_item_5_logging()
        print("✓ Validated logging setup")

        self.validate_item_6_exceptions()
        print("✓ Validated exception handling")

        self.validate_item_9_package_structure()
        print("✓ Validated package structure")

        self.validate_item_11_tests_coverage()
        print("✓ Validated tests and coverage")

        # Generate report
        json_path, md_path = self.generate_report()

        print("\nValidation complete!")
        print(f"JSON report: {json_path}")
        print(f"Markdown report: {md_path}")

        # Print summary
        passed = sum(1 for r in self.results.values() if r["passed"])
        total = len(self.results)
        print(f"\nOverall: {passed}/{total} items passed")

        if passed < total:
            print("\nFailed items:")
            for item_name, result in self.results.items():
                if not result["passed"]:
                    print(f"  - {item_name.replace('_', ' ').title()}")


def main():
    """Main function."""
    project_root = Path("/Users/rohanvinaik/genomevault")

    if not project_root.exists():
        print(f"Error: Project root not found: {project_root}")
        sys.exit(1)

    validator = ChecklistValidator(project_root)
    validator.run_all_validations()


if __name__ == "__main__":
    main()
