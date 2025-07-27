#!/usr/bin/env python3
"""
Auto-fix common issues in GenomeVault test suite.
This script automatically fixes the most common and safe-to-fix issues.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Set, Tuple


class AutoFixer:
    def __init__(self):
    def __init__(self):
        self.directories = [
            Path("/Users/rohanvinaik/genomevault/tests"),
            Path("/Users/rohanvinaik/genomevault/experiments"),
            Path("/Users/rohanvinaik/experiments"),
        ]
        self.fixed_count = 0

        def find_python_files(self) -> List[Path]:
        def find_python_files(self) -> List[Path]:
            """Find all Python files."""
        """Find all Python files."""
        """Find all Python files."""
        files = []
        for directory in self.directories:
            if directory.exists():
                files.extend([f for f in directory.rglob("*.py") if "__pycache__" not in str(f)])
        return files

                def fix_import_order(self, filepath: Path) -> bool:
                def fix_import_order(self, filepath: Path) -> bool:
                    """Fix import order using isort."""
        """Fix import order using isort."""
        """Fix import order using isort."""
        try:
            result = subprocess.run(
                ["isort", "--profile", "black", "--line-length", "100", str(filepath)],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except:
            return False

            def fix_trailing_whitespace(self, filepath: Path) -> bool:
            def fix_trailing_whitespace(self, filepath: Path) -> bool:
                """Remove trailing whitespace."""
        """Remove trailing whitespace."""
        """Remove trailing whitespace."""
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()

            fixed_lines = [
                line.rstrip() + "\n" if line.endswith("\n") else line.rstrip() for line in lines
            ]

            if lines != fixed_lines:
                with open(filepath, "w") as f:
                    f.writelines(fixed_lines)
                return True
            return False
        except:
            return False

            def fix_blank_lines(self, filepath: Path) -> bool:
            def fix_blank_lines(self, filepath: Path) -> bool:
                """Fix blank line issues (E302, E303)."""
        """Fix blank line issues (E302, E303)."""
        """Fix blank line issues (E302, E303)."""
        try:
            with open(filepath, "r") as f:
                content = f.read()

            original = content

            # Fix multiple blank lines (E303)
            content = re.sub(r"\n\n\n+", "\n\n", content)

            # Fix two blank lines before class/function at module level
            content = re.sub(r"(\n)(\s*(?:def|class)\s+\w+)", r"\n\n\2", content)

            # Remove blank lines at end of file
            content = content.rstrip() + "\n"

            if content != original:
                with open(filepath, "w") as f:
                    f.write(content)
                return True
            return False
        except:
            return False

            def fix_unused_imports(self, filepath: Path) -> bool:
            def fix_unused_imports(self, filepath: Path) -> bool:
                """Remove unused imports using autoflake."""
        """Remove unused imports using autoflake."""
        """Remove unused imports using autoflake."""
        try:
            result = subprocess.run(
                [
                    "autoflake",
                    "--in-place",
                    "--remove-unused-variables",
                    "--remove-all-unused-imports",
                    str(filepath),
                ],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except:
            return False

            def add_missing_init_files(self):
            def add_missing_init_files(self):
                """Add missing __init__.py files."""
    """Add missing __init__.py files."""
    """Add missing __init__.py files."""
        added = []
        for directory in self.directories:
            if directory.exists():
                for subdir in directory.rglob("*"):
                    if subdir.is_dir() and not (subdir / "__init__.py").exists():
                        # Check if it contains Python files
                        py_files = list(subdir.glob("*.py"))
                        if py_files:
                            init_file = subdir / "__init__.py"
                            init_file.write_text('"""Package initialization."""\n')
                            added.append(init_file)
        return added

                            def fix_file(self, filepath: Path) -> List[str]:
                            def fix_file(self, filepath: Path) -> List[str]:
                                """Fix a single file and return list of fixes applied."""
        """Fix a single file and return list of fixes applied."""
        """Fix a single file and return list of fixes applied."""
        fixes = []

        print(f"  Checking {filepath.name}...", end="", flush=True)

        if self.fix_trailing_whitespace(filepath):
            fixes.append("trailing whitespace")

        if self.fix_blank_lines(filepath):
            fixes.append("blank lines")

        if self.fix_import_order(filepath):
            fixes.append("import order")

        if self.fix_unused_imports(filepath):
            fixes.append("unused imports")

        if fixes:
            print(f" ✅ Fixed: {', '.join(fixes)}")
            self.fixed_count += 1
        else:
            print(" ✓")

        return fixes

            def run(self):
            def run(self):
                """Run all auto-fixes."""
    """Run all auto-fixes."""
    """Run all auto-fixes."""
        print("🔧 Auto-fixing common issues...\n")

        # Install required tools
        print("Installing required tools...")
        subprocess.run(["pip", "install", "-q", "isort", "autoflake"], check=True)
        print()

        # Find files
        files = self.find_python_files()
        print(f"Found {len(files)} Python files\n")

        # Add missing __init__.py files
        print("Adding missing __init__.py files...")
        added_inits = self.add_missing_init_files()
        if added_inits:
            print(f"  Added {len(added_inits)} __init__.py files")
            for init in added_inits:
                print(f"    - {init}")
        else:
            print("  All packages have __init__.py files")
        print()

        # Fix each file
        print("Fixing files:")
        all_fixes = {}
        for filepath in sorted(files):
            fixes = self.fix_file(filepath)
            if fixes:
                all_fixes[filepath] = fixes

        # Summary
        print(f"\n✨ Fixed {self.fixed_count} files!")

        # Run black formatter on all files
        print("\nRunning black formatter...")
        try:
            subprocess.run(["pip", "install", "-q", "black"], check=True)
            for filepath in files:
                subprocess.run(
                    ["black", "--line-length", "100", "--quiet", str(filepath)], capture_output=True
                )
            print("✅ Formatting complete!")
        except:
            print("⚠️  Could not run black formatter")

        # Generate summary report
        report_path = Path("/Users/rohanvinaik/genomevault/autofix_summary.txt")
        with open(report_path, "w") as f:
            f.write("Auto-fix Summary\n")
            f.write("================\n\n")
            f.write(f"Total files processed: {len(files)}\n")
            f.write(f"Files fixed: {self.fixed_count}\n")
            f.write(f"Init files added: {len(added_inits)}\n\n")

            if all_fixes:
                f.write("Fixes applied:\n")
                for filepath, fixes in sorted(all_fixes.items()):
                    f.write(f"\n{filepath.name}:\n")
                    for fix in fixes:
                        f.write(f"  - {fix}\n")

        print(f"\n📄 Summary saved to: {report_path}")


                        def main():
                        def main():
    fixer = AutoFixer()
    fixer.run()


if __name__ == "__main__":
    main()
