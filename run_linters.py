#!/usr/bin/env python3
"""
Comprehensive linter for GenomeVault test suite and experiments.
Runs multiple linters and generates a detailed report.
"""

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class LinterRunner:
    def __init__(self):
        self.directories = [
            Path("/Users/rohanvinaik/genomevault/tests"),
            Path("/Users/rohanvinaik/genomevault/experiments"),
            Path("/Users/rohanvinaik/experiments"),
        ]
        self.results = defaultdict(lambda: defaultdict(list))

    def find_python_files(self) -> List[Path]:
        """Find all Python files."""
        files = []
        for directory in self.directories:
            if directory.exists():
                files.extend([f for f in directory.rglob("*.py") if "__pycache__" not in str(f)])
        return files

    def run_flake8(self, files: List[Path]):
        """Run flake8 linter."""
        print("Running flake8...")
        for filepath in files:
            try:
                result = subprocess.run(
                    ["flake8", "--max-line-length=100", "--ignore=E501,W503", str(filepath)],
                    capture_output=True,
                    text=True,
                )
                if result.stdout:
                    issues = result.stdout.strip().split("\n")
                    self.results["flake8"][filepath] = issues
            except Exception as e:
                self.results["flake8"][filepath] = [f"Error: {e}"]

    def run_pylint(self, files: List[Path]):
        """Run pylint."""
        print("Running pylint...")
        for filepath in files:
            try:
                result = subprocess.run(
                    ["pylint", "--output-format=json", str(filepath)],
                    capture_output=True,
                    text=True,
                )
                if result.stdout:
                    try:
                        issues = json.loads(result.stdout)
                        # Filter out documentation warnings
                        filtered = [
                            i
                            for i in issues
                            if i["symbol"]
                            not in [
                                "missing-module-docstring",
                                "missing-class-docstring",
                                "missing-function-docstring",
                            ]
                        ]
                        if filtered:
                            self.results["pylint"][filepath] = filtered
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                self.results["pylint"][filepath] = [{"message": f"Error: {e}"}]

    def run_mypy(self, files: List[Path]):
        """Run mypy type checker."""
        print("Running mypy...")
        for filepath in files:
            try:
                result = subprocess.run(
                    ["mypy", "--ignore-missing-imports", str(filepath)],
                    capture_output=True,
                    text=True,
                )
                if result.stdout:
                    issues = result.stdout.strip().split("\n")
                    self.results["mypy"][filepath] = issues
            except Exception as e:
                self.results["mypy"][filepath] = [f"Error: {e}"]

    def generate_report(self):
        """Generate a comprehensive report."""
        report_path = Path("/Users/rohanvinaik/genomevault/linting_report.md")

        with open(report_path, "w") as f:
            f.write("# GenomeVault Test Suite Linting Report\n\n")

            # Summary
            f.write("## Summary\n\n")
            for linter in ["flake8", "pylint", "mypy"]:
                total_issues = sum(len(issues) for issues in self.results[linter].values())
                files_with_issues = len(self.results[linter])
                f.write(f"- **{linter}**: {total_issues} issues in {files_with_issues} files\n")
            f.write("\n")

            # Detailed issues by linter
            for linter in ["flake8", "pylint", "mypy"]:
                if self.results[linter]:
                    f.write(f"## {linter.capitalize()} Issues\n\n")

                    for filepath, issues in sorted(self.results[linter].items()):
                        f.write(f"### {filepath.name}\n")
                        f.write(f"Path: `{filepath}`\n\n")

                        if linter == "pylint":
                            for issue in issues[:10]:  # Limit to 10 issues per file
                                f.write(
                                    f"- **{issue.get('symbol', 'unknown')}**: {issue.get('message', 'No message')}\n"
                                )
                                f.write(
                                    f"  Line: {issue.get('line', '?')}, Column: {issue.get('column', '?')}\n"
                                )
                        else:
                            for issue in issues[:10]:  # Limit to 10 issues per file
                                f.write(f"- {issue}\n")

                        if len(issues) > 10:
                            f.write(f"- ... and {len(issues) - 10} more issues\n")
                        f.write("\n")

            # Priority fixes
            f.write("## Priority Fixes\n\n")
            f.write("Based on the analysis, here are the priority fixes:\n\n")

            # Count issue types
            issue_counts = defaultdict(int)
            for filepath, issues in self.results["flake8"].items():
                for issue in issues:
                    if ":" in issue:
                        code = issue.split(":")[-1].strip().split()[0]
                        issue_counts[code] += 1

            # Sort by frequency
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)

            f.write("### Most Common Issues\n\n")
            for code, count in sorted_issues[:10]:
                f.write(f"- **{code}**: {count} occurrences\n")

        print(f"\n📄 Report generated: {report_path}")

    def run(self):
        """Run all linters and generate report."""
        print("🔍 Running comprehensive linting...\n")

        # Find files
        files = self.find_python_files()
        print(f"Found {len(files)} Python files\n")

        # Run linters
        self.run_flake8(files)
        self.run_pylint(files)
        self.run_mypy(files)

        # Generate report
        self.generate_report()

        print("\n✅ Linting complete!")


def main():
    """Main entry point."""
    # Ensure linters are installed
    required = ["flake8", "pylint", "mypy"]
    for tool in required:
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
        except:
            print(f"Installing {tool}...")
            subprocess.run(["pip", "install", tool], check=True)

    # Run linter
    runner = LinterRunner()
    runner.run()


if __name__ == "__main__":
    main()
