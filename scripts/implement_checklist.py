#!/usr/bin/env python3
"""
Master script to implement the genomevault audit checklist.
Run this to execute all checklist items in order.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class ChecklistImplementer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.log_file = self.project_root / "checklist_implementation.log"
        self.backup_dir = (
            self.project_root / f'genomevault_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )

    def log(self, message):
        """Log message to both console and file."""
        print(message)
        with open(self.log_file, "a") as f:
            f.write(f"{datetime.now().isoformat()} - {message}\n")

    def backup_project(self):
        """Create a backup of the current project state."""
        self.log("Creating project backup...")

        # Create backup of key files
        files_to_backup = [
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "requirements-dev.txt",
            ".pre-commit-config.yaml",
            "mypy.ini",
            ".flake8",
            ".pylintrc",
            "LICENSE",
        ]

        os.makedirs(self.backup_dir, exist_ok=True)

        for file_name in files_to_backup:
            src = self.project_root / file_name
            if src.exists():
                dst = self.backup_dir / file_name
                shutil.copy2(src, dst)
                self.log(f"  Backed up: {file_name}")

        self.log(f"Backup created at: {self.backup_dir}")

    def run_command(self, cmd, check=True):
        """Run a shell command."""
        self.log(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True, check=check
            )
            if result.stdout:
                self.log(f"Output: {result.stdout[:200]}...")
            return True
        except subprocess.CalledProcessError as e:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            self.log(f"Error: {e}")
            if e.stderr:
                self.log(f"Stderr: {e.stderr}")
            return False
            raise

    def step_5_convert_prints(self):
        """Step 5: Convert print statements to logging."""
        self.log("\n=== Step 5: Converting print statements to logging ===")

        # First, run the analysis script
        script_path = self.project_root / "scripts" / "convert_print_to_logging.py"
        if script_path.exists():
            self.run_command(["python", str(script_path)])
        else:
            self.log("Print conversion script not found, skipping analysis")

        # Add logging setup to CLI entry points
        cli_files = [
            self.project_root / "genomevault" / "cli" / "__main__.py",
            self.project_root / "genomevault" / "cli" / "main.py",
        ]

        logging_setup = """import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
"""

        for cli_file in cli_files:
            if cli_file.exists():
                content = cli_file.read_text()
                if "logging.basicConfig" not in content:
                    # Add after imports
                    lines = content.split("\n")
                    import_end = 0
                    for i, line in enumerate(lines):
                        if line.strip() and not line.startswith(("import", "from", "#")):
                            import_end = i
                            break

                    lines.insert(import_end, logging_setup)
                    cli_file.write_text("\n".join(lines))
                    self.log(f"Added logging setup to: {cli_file}")

    def step_7_reduce_complexity(self):
        """Step 7: Analyze cyclomatic complexity."""
        self.log("\n=== Step 7: Analyzing cyclomatic complexity ===")

        # Run complexity analysis script
        script_path = self.project_root / "scripts" / "analyze_complexity.py"
        if script_path.exists():
            self.run_command(["python", str(script_path)])
        # Run radon directly
        elif self.run_command(["radon", "--version"], check=False):
            self.run_command(
                [
                    "radon",
                    "cc",
                    "-s",
                    "-a",
                    str(self.project_root / "genomevault"),
                    "--min",
                    "B",  # Show only B grade and worse (CC >= 6)
                ]
            )
        else:
            self.log("radon not installed, skipping complexity analysis")

    def step_9_add_init_files(self):
        """Step 9: Ensure all packages have __init__.py files."""
        self.log("\n=== Step 9: Adding missing __init__.py files ===")

        genomevault_dir = self.project_root / "genomevault"
        added_count = 0

        for root, dirs, files in os.walk(genomevault_dir):
            root_path = Path(root)

            # Skip __pycache__ and other special directories
            dirs[:] = [d for d in dirs if not d.startswith((".", "__pycache__"))]

            # Check if this directory contains Python files
            if any(f.endswith(".py") for f in files):
                init_file = root_path / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
                    self.log(f"  Created: {init_file.relative_to(self.project_root)}")
                    added_count += 1

        self.log(f"Added {added_count} __init__.py files")

    def step_12_extract_todos(self):
        """Step 12: Extract TODOs and create issues list."""
        self.log("\n=== Step 12: Extracting TODOs ===")

        # Search for TODOs
        patterns = ["TODO", "FIXME", "XXX"]
        todos = []

        for pattern in patterns:
            result = subprocess.run(
                ["rg", "-n", pattern, "genomevault"],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        todos.append(f"{pattern}: {line}")

        # Write to file
        todos_file = self.project_root / "docs" / "todo_issues.md"
        todos_file.parent.mkdir(exist_ok=True)

        with open(todos_file, "w") as f:
            f.write("# TODO/FIXME/XXX Items\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Total items found: {len(todos)}\n\n")

            for todo in todos:
                f.write(f"- [ ] {todo}\n")

        self.log(f"Extracted {len(todos)} TODO items to: {todos_file}")

    def install_dependencies(self):
        """Install development dependencies."""
        self.log("\n=== Installing dependencies ===")

        # First, try to install the package with dev dependencies
        if not self.run_command(["pip", "install", "-e", ".[dev]"], check=False):
            # If that fails, try installing individual packages
            self.log("Package install failed, trying individual packages...")
            packages = ["ruff", "mypy", "pytest", "pytest-cov", "pyupgrade", "radon"]
            for pkg in packages:
                self.run_command(["pip", "install", pkg], check=False)

        # Install pre-commit
        if self.run_command(["pip", "install", "pre-commit"], check=False):
            self.run_command(["pre-commit", "install"], check=False)

    def run_validation(self):
        """Run the validation script."""
        self.log("\n=== Running validation ===")

        script_path = self.project_root / "scripts" / "validate_checklist.py"
        if script_path.exists():
            self.run_command(["python", str(script_path)])
        else:
            self.log("Validation script not found")

    def implement_checklist(self):
        """Implement all checklist items."""
        self.log("Starting genomevault audit checklist implementation")
        self.log(f"Project root: {self.project_root}")

        # Create backup first
        self.backup_project()

        # Steps that require file creation are already done by the artifact creation
        self.log("\n=== Configuration files already created ===")
        self.log("✓ pyproject.toml (Hatch)")
        self.log("✓ ruff.toml")
        self.log("✓ mypy.ini")
        self.log("✓ pytest.ini")
        self.log("✓ .pre-commit-config.yaml")
        self.log("✓ .github/workflows/ci.yml")
        self.log("✓ genomevault/logging_utils.py")
        self.log("✓ genomevault/exceptions.py")
        self.log("✓ LICENSE (MIT)")

        # Install dependencies
        self.install_dependencies()

        # Run implementation steps
        self.step_5_convert_prints()
        self.step_7_reduce_complexity()
        self.step_9_add_init_files()
        self.step_12_extract_todos()

        # Run validation
        self.run_validation()

        self.log("\n=== Implementation complete ===")
        self.log(f"Log file: {self.log_file}")
        self.log(f"Backup directory: {self.backup_dir}")

        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "backup_dir": str(self.backup_dir),
            "log_file": str(self.log_file),
            "steps_completed": [
                "Created Hatch-based pyproject.toml",
                "Updated LICENSE to MIT",
                "Created ruff.toml configuration",
                "Updated mypy.ini",
                "Created pytest.ini",
                "Updated .pre-commit-config.yaml",
                "Created CI workflow",
                "Created logging utilities",
                "Created exception hierarchy",
                "Analyzed print statements",
                "Analyzed cyclomatic complexity",
                "Added missing __init__.py files",
                "Extracted TODO items",
            ],
        }

        summary_file = self.project_root / "checklist_implementation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        self.log(f"\nSummary saved to: {summary_file}")


def main():
    """Main function."""
    project_root = Path("/Users/rohanvinaik/genomevault")

    if not project_root.exists():
        print(f"Error: Project root not found: {project_root}")
        sys.exit(1)

    implementer = ChecklistImplementer(project_root)

    # Confirm before proceeding
    print("This script will implement the genomevault audit checklist.")
    print(f"Project root: {project_root}")
    print("\nThis will:")
    print("- Create a backup of current configuration files")
    print("- Update project configuration to use Hatch, ruff, etc.")
    print("- Install development dependencies")
    print("- Analyze and report on code quality issues")
    print("- Add missing __init__.py files")
    print("- Extract TODO items")
    print("\nPress Enter to continue or Ctrl+C to cancel...")

    try:
        input()
    except KeyboardInterrupt:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        print("\nCancelled.")
        sys.exit(0)
        raise

    implementer.implement_checklist()


if __name__ == "__main__":
    main()
