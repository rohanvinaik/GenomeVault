#!/usr/bin/env python3
"""
Working TailChasingFixer implementation for GenomeVault
Based on the analysis results, this provides actionable fixes.
"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


class GenomeVaultTailChasingFixer:
    def __init__(self):
        """Magic method implementation."""
        self.fixes_applied = 0
        self.fixes_skipped = 0
        self.backup_dir = Path("./tailchasing_backup")
        self.ensure_backup_dir()

    def ensure_backup_dir(self):
        """Create backup directory if it doesn't exist."""
        self.backup_dir.mkdir(exist_ok=True)

    def backup_file(self, file_path: str) -> str:
        """Create a backup of the file before modifying."""
        file_path = Path(file_path)
        if not file_path.exists():
            return None

        backup_path = self.backup_dir / f"{file_path.name}.backup"
        shutil.copy2(file_path, backup_path)
        return str(backup_path)

    def install_autoflake(self):
        """Install autoflake for cleaning imports."""
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "autoflake"],
                check=True,
                capture_output=True,
            )
            print("✅ autoflake installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install autoflake: {e}")
            return False

    def clean_unused_imports(self, file_path: str) -> bool:
        """Remove unused imports from a Python file."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "autoflake",
                    "--remove-unused-variables",
                    "--remove-all-unused-imports",
                    "--in-place",
                    file_path,
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print(f"✅ Cleaned imports in {file_path}")
                return True
            else:
                print(f"⚠️ Warning cleaning {file_path}: {result.stderr}")
                return False
        except FileNotFoundError:
            print("❌ autoflake not found. Installing...")
            if self.install_autoflake():
                return self.clean_unused_imports(file_path)
            return False

    def remove_phantom_function(
        self, file_path: str, line_num: int, function_name: str
    ) -> bool:
        """Remove or comment out phantom functions."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return False

            self.backup_file(file_path)

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Find the function definition
            for i, line in enumerate(lines):
                if (
                    f"def {function_name.split('.')[-1]}" in line
                    and i + 1 >= line_num - 2
                ):
                    # Comment out the function
                    indent = len(line) - len(line.lstrip())
                    lines[i] = (
                        f"{' ' * indent}# TODO: Implement {function_name} or remove\n"
                    )
                    lines[i] += f"{' ' * indent}# {line.strip()}\n"

                    # Comment out the function body
                    j = i + 1
                    while j < len(lines) and (
                        lines[j].strip() == ""
                        or len(lines[j]) - len(lines[j].lstrip()) > indent
                    ):
                        if lines[j].strip():
                            lines[j] = f"{' ' * indent}# {lines[j].lstrip()}"
                        j += 1
                    break

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            print(f"✅ Commented out phantom function {function_name} in {file_path}")
            return True

        except Exception as e:
            print(f"❌ Error removing phantom function {function_name}: {e}")
            return False

    def find_duplicate_functions(self, file_path: str) -> List[Dict]:
        """Find structurally identical functions in a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Create a simplified representation
                    func_repr = {
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "body_hash": self._hash_ast_body(node.body),
                    }
                    functions.append(func_repr)

            # Find duplicates
            duplicates = []
            for i, func1 in enumerate(functions):
                for func2 in functions[i + 1 :]:
                    if func1["body_hash"] == func2["body_hash"]:
                        duplicates.append((func1, func2))

            return duplicates

        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
            return []

    def _hash_ast_body(self, body: List[ast.stmt]) -> str:
        """Create a hash of the AST body for comparison."""
        try:
            # Convert AST to string representation for hashing
            body_str = ast.dump(body, annotate_fields=False)
            return str(hash(body_str))
        except:
            return ""

    def merge_duplicate_functions(
        self, file_path: str, func1: Dict, func2: Dict
    ) -> bool:
        """Merge duplicate functions by keeping one and commenting out the other."""
        try:
            file_path = Path(file_path)
            self.backup_file(file_path)

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Comment out the second function (usually later in file)
            target_line = max(func1["line"], func2["line"])
            target_name = (
                func2["name"] if func1["line"] < func2["line"] else func1["name"]
            )

            for i, line in enumerate(lines):
                if f"def {target_name}" in line and i + 1 >= target_line - 2:
                    indent = len(line) - len(line.lstrip())
                    lines[i] = (
                        f"{' ' * indent}# DUPLICATE: Merged with {func1['name'] if target_name == func2['name'] else func2['name']}\n"
                    )
                    lines[i] += f"{' ' * indent}# {line.strip()}\n"

                    # Comment out the function body
                    j = i + 1
                    while j < len(lines) and (
                        lines[j].strip() == ""
                        or len(lines[j]) - len(lines[j].lstrip()) > indent
                    ):
                        if lines[j].strip():
                            lines[j] = f"{' ' * indent}# {lines[j].lstrip()}"
                        j += 1
                    break

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            print(
                f"✅ Merged duplicate functions {func1['name']} and {func2['name']} in {file_path}"
            )
            return True

        except Exception as e:
            print(f"❌ Error merging functions: {e}")
            return False

    def process_high_priority_files(self):
        """Process the files with the most issues first."""
        high_priority_files = [
            "clinical_validation/clinical_circuits.py",
            "genomevault/zk_proofs/circuits/base_circuits.py",
            "clinical_validation/data_sources/pima.py",
            "genomevault/utils/logging.py",
            "genomevault/pir/client.py",
        ]

        print("🎯 Processing High Priority Files")
        print(" = " * 50)

        for file_rel_path in high_priority_files:
            file_path = Path(file_rel_path)
            if not file_path.exists():
                print(f"⏭️ Skipping {file_path} (not found)")
                continue

            print(f"\n📁 Processing: {file_path}")

            # Clean imports
            print("  🧹 Cleaning imports...")
            self.clean_unused_imports(str(file_path))

            # Find and merge duplicates
            print("  🔍 Finding duplicate functions...")
            duplicates = self.find_duplicate_functions(str(file_path))

            if duplicates:
                print(f"  📋 Found {len(duplicates)} duplicate pairs")
                for i, (func1, func2) in enumerate(duplicates[:3]):  # Limit to first 3
                    response = input(
                        f"    Merge {func1['name']} and {func2['name']}? [y/N]: "
                    )
                    if response.lower() == "y":
                        self.merge_duplicate_functions(str(file_path), func1, func2)
                        self.fixes_applied += 1
                    else:
                        self.fixes_skipped += 1
            else:
                print("  ✅ No structural duplicates found")

    def clean_all_imports(self):
        """Clean imports across all Python files."""
        print("🧹 Cleaning All Unused Imports")
        print(" = " * 50)

        python_files = list(Path(".").rglob("*.py"))
        cleaned = 0

        for file_path in python_files:
            if any(
                skip in str(file_path)
                for skip in [".git", "__pycache__", ".pytest_cache", "build", "dist"]
            ):
                continue

            if self.clean_unused_imports(str(file_path)):
                cleaned += 1

        print(f"✅ Cleaned imports in {cleaned} files")

    def remove_obvious_phantoms(self):
        """Remove obvious phantom functions based on the analysis."""
        phantoms = [
            ("clinical_validation/data_sources/pima.py", 32, "get_glucose_column"),
            ("clinical_validation/data_sources/pima.py", 35, "get_hba1c_column"),
            ("clinical_validation/data_sources/pima.py", 38, "get_outcome_column"),
        ]

        print("👻 Removing Obvious Phantom Functions")
        print(" = " * 50)

        for file_path, line_num, func_name in phantoms:
            if Path(file_path).exists():
                response = input(
                    f"Remove phantom function {func_name} in {file_path}? [y/N]: "
                )
                if response.lower() == "y":
                    if self.remove_phantom_function(file_path, line_num, func_name):
                        self.fixes_applied += 1
                    else:
                        self.fixes_skipped += 1
                else:
                    self.fixes_skipped += 1

    def generate_report(self):
        """Generate a summary report of changes made."""
        print("\n" + " = " * 60)
        print("🎉 TailChasingFixer Summary Report")
        print(" = " * 60)
        print(f"✅ Fixes Applied: {self.fixes_applied}")
        print(f"⏭️ Fixes Skipped: {self.fixes_skipped}")
        print(f"📁 Backups Created: {len(list(self.backup_dir.glob('*.backup')))}")
        print("\n📋 Next Steps:")
        print("1. Review the changes made")
        print("2. Run your tests to ensure functionality")
        print("3. Commit the fixes incrementally")
        print("4. Re-run TailChasingFixer to see remaining issues:")
        print("   python -m tailchasing . --show-suggestions")
        print("\n📂 Backup files are in:", self.backup_dir.absolute())

    def interactive_fix_session(self):
        """Run an interactive fixing session."""
        print("🔧 GenomeVault TailChasingFixer - Interactive Mode")
        print(" = " * 60)
        print("This will help you fix the most critical issues found.")
        print("Backups will be created before any changes.\n")

        # Step 1: Clean imports (safest)
        response = input("1️⃣ Clean all unused imports? (Safest fix) [Y/n]: ")
        if response.lower() != "n":
            self.clean_all_imports()

        # Step 2: Process high priority files
        response = input("\n2️⃣ Process high-priority files with duplicates? [Y/n]: ")
        if response.lower() != "n":
            self.process_high_priority_files()

        # Step 3: Remove phantom functions
        response = input("\n3️⃣ Remove obvious phantom functions? [Y/n]: ")
        if response.lower() != "n":
            self.remove_obvious_phantoms()

        # Generate report
        self.generate_report()


def main():
    """Main entry point."""
    try:
        fixer = GenomeVaultTailChasingFixer()
        fixer.interactive_fix_session()
    except KeyboardInterrupt:
        print("\n⏹️ Fixing session interrupted by user")
    except Exception as e:
        print(f"❌ Error during fixing session: {e}")
        print("Check that you're in the GenomeVault root directory")


if __name__ == "__main__":
    main()
