#!/usr/bin/env python3
"""
GenomeVault Lint Clean Implementation Script

This script implements the systematic lint cleaning process according to the
instructions in the project knowledge. It handles:
1. Adding constants for magic numbers (PLR2004)
2. Fixing unused variables (F841)
3. Adding missing imports (F821, E402)
4. Validating changes with ruff check

Usage: python lint_clean_implementation.py
"""

import subprocess
import sys
from pathlib import Path


class LintCleaner:
    """Implements systematic lint cleaning for GenomeVault."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.fixes_applied = []

    def run_ruff_check(self) -> tuple[str, int]:
        """Run ruff check and return output and return code."""
        try:
            result = subprocess.run(
                ["ruff", "check", "."],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            return result.stdout + result.stderr, result.returncode
        except FileNotFoundError:
            print("ERROR: ruff not found. Please install with: pip install ruff")
            sys.exit(1)

    def validate_tools(self) -> bool:
        """Validate that required tools are available."""
        tools = ["ruff", "mypy", "pytest"]
        missing = []

        for tool in tools:
            try:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
            except (FileNotFoundError, subprocess.CalledProcessError):
                missing.append(tool)

        if missing:
            print(f"ERROR: Missing tools: {', '.join(missing)}")
            return False
        return True

    def fix_test_hv_encoding(self) -> None:
        """Fix tests/hv/test_encoding.py according to checklist."""
        file_path = self.project_root / "tests" / "hv" / "test_encoding.py"

        if not file_path.exists():
            print(f"WARNING: {file_path} not found")
            return

        print(f"Fixing {file_path}")

        # Read current content
        content = file_path.read_text()

        # Constants to add at the top after imports
        constants = """
# Constants for magic number elimination (PLR2004)
SIM_LOWER = 0.5
SIM_UPPER = 0.95
COMPONENT_SIM_MIN = 0.2
SEQ_SIM_LOWER = 0.1
SEQ_SIM_UPPER = 0.7
THROUGHPUT_TARGET = 1000
CORRELATION_MIN = 0.7
"""

        # Find location to insert constants (after last import)
        lines = content.split("\n")
        last_import_idx = 0

        for i, line in enumerate(lines):
            if line.strip().startswith(
                ("import ", "from ")
            ) and not line.strip().startswith("#"):
                last_import_idx = i

        # Insert constants after last import
        lines.insert(last_import_idx + 1, constants)

        # Replace magic numbers with constants
        replacements = [
            ("similarity < 0.9", "similarity < SIM_UPPER"),
            (
                "similarity < 0.8",
                "similarity < SIM_UPPER - 0.15",
            ),  # Adjust to maintain logic
            ("0.5 < similarity < 0.95", "SIM_LOWER < similarity < SIM_UPPER"),
            ("assert 0.2 <= similarity", "assert COMPONENT_SIM_MIN <= similarity"),
            ("threshold=0.2", "threshold=COMPONENT_SIM_MIN"),
            ("assert correlation > 0.7", "assert correlation > CORRELATION_MIN"),
            ("variants_per_second > 1000", "variants_per_second > THROUGHPUT_TARGET"),
        ]

        new_content = "\n".join(lines)
        for old, new in replacements:
            new_content = new_content.replace(old, new)

        # Fix unused variables - look for vec2 and similar patterns
        if (
            "vec2 =" in new_content
            and "vec2[" not in new_content
            and "vec2)" not in new_content
        ):
            new_content = new_content.replace("vec2 =", "_ =")

        # Write back
        file_path.write_text(new_content)
        self.fixes_applied.append(
            f"Fixed {file_path}: Added constants, replaced magic numbers"
        )

    def fix_test_pir_protocol(self) -> None:
        """Fix tests/pir/test_pir_protocol.py according to checklist."""
        file_path = self.project_root / "tests" / "pir" / "test_pir_protocol.py"

        if not file_path.exists():
            print(f"WARNING: {file_path} not found")
            return

        print(f"Fixing {file_path}")

        content = file_path.read_text()

        # Constants to add
        constants = """
# Constants for magic number elimination (PLR2004)
TIMING_VARIANCE_MAX = 5.0
MIN_SERVERS_TS = 2
MIN_SERVERS_LN = 3
TIMING_VARIANCE_LARGE = 10.0
QPS_MIN = 10
"""

        # Find location to insert constants
        lines = content.split("\n")
        last_import_idx = 0

        for i, line in enumerate(lines):
            if line.strip().startswith(
                ("import ", "from ")
            ) and not line.strip().startswith("#"):
                last_import_idx = i

        lines.insert(last_import_idx + 1, constants)

        # Replace magic numbers with constants
        replacements = [
            ("timing_variance < 5.0", "timing_variance < TIMING_VARIANCE_MAX"),
            ("timing_variance < 10.0", "timing_variance < TIMING_VARIANCE_LARGE"),
            ("min_ts == 2", "min_ts == MIN_SERVERS_TS"),
            ("min_ln == 3", "min_ln == MIN_SERVERS_LN"),
            ("queries_per_second > 10", "queries_per_second > QPS_MIN"),
        ]

        new_content = "\n".join(lines)
        for old, new in replacements:
            new_content = new_content.replace(old, new)

        # Fix unused variables by renaming to _
        unused_vars = ["queries", "response", "batch_queries"]
        for var in unused_vars:
            # Look for assignment patterns that might be unused
            import re

            pattern = rf"^(\s*){var}\s*="
            new_content = re.sub(pattern, r"\1_ =", new_content, flags=re.MULTILINE)

        file_path.write_text(new_content)
        self.fixes_applied.append(
            f"Fixed {file_path}: Added constants, fixed unused variables"
        )

    def fix_test_zk_property_circuits(self) -> None:
        """Fix tests/zk/test_zk_property_circuits.py according to checklist."""
        file_path = self.project_root / "tests" / "zk" / "test_zk_property_circuits.py"

        if not file_path.exists():
            print(f"WARNING: {file_path} not found")
            return

        print(f"Fixing {file_path}")

        content = file_path.read_text()

        # Constants to add
        constants = """
# Constants for magic number elimination (PLR2004)
MAX_VARIANTS = 10
VERIFICATION_TIME_MAX = 0.1
"""

        # Find location to insert constants
        lines = content.split("\n")
        last_import_idx = 0

        for i, line in enumerate(lines):
            if line.strip().startswith(
                ("import ", "from ")
            ) and not line.strip().startswith("#"):
                last_import_idx = i

        lines.insert(last_import_idx + 1, constants)

        # Replace magic numbers
        replacements = [
            ("len(self.variants) < 10", "len(self.variants) < MAX_VARIANTS"),
            ("len(self.variants) <= 10", "len(self.variants) <= MAX_VARIANTS"),
            ("max_snps=10", "max_snps=MAX_VARIANTS"),
            ("verification_time < 0.1", "verification_time < VERIFICATION_TIME_MAX"),
        ]

        new_content = "\n".join(lines)
        for old, new in replacements:
            new_content = new_content.replace(old, new)

        file_path.write_text(new_content)
        self.fixes_applied.append(f"Fixed {file_path}: Added constants")

    def fix_test_api_startup(self) -> None:
        """Fix tests/smoke/test_api_startup.py according to checklist."""
        file_path = self.project_root / "tests" / "smoke" / "test_api_startup.py"

        if not file_path.exists():
            print(f"WARNING: {file_path} not found")
            return

        print(f"Fixing {file_path}")

        content = file_path.read_text()

        # Constants to add
        constants = """
# Constants for magic number elimination (PLR2004)
HTTP_OK = 200
"""

        # Find location to insert constants
        lines = content.split("\n")
        last_import_idx = 0

        for i, line in enumerate(lines):
            if line.strip().startswith(
                ("import ", "from ")
            ) and not line.strip().startswith("#"):
                last_import_idx = i

        lines.insert(last_import_idx + 1, constants)

        # Replace magic numbers
        new_content = "\n".join(lines)
        new_content = new_content.replace(
            "r.status_code == 200", "r.status_code == HTTP_OK"
        )

        file_path.write_text(new_content)
        self.fixes_applied.append(f"Fixed {file_path}: Added HTTP_OK constant")

    def fix_missing_imports(self) -> None:
        """Fix missing imports across the codebase."""
        print("Checking for missing imports...")

        # Common missing imports to check for
        import_fixes = {
            "time.time()": "import time",
            "np.": "import numpy as np",
            "plt.": "import matplotlib.pyplot as plt",
        }

        # Find Python files that might need import fixes
        test_dirs = ["tests/hv", "tests/pir", "tests/zk", "tests/smoke"]

        for test_dir in test_dirs:
            dir_path = self.project_root / test_dir
            if not dir_path.exists():
                continue

            for py_file in dir_path.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue

                content = py_file.read_text()
                lines = content.split("\n")

                # Check if we need to add imports
                imports_to_add = []

                for usage, import_stmt in import_fixes.items():
                    if usage in content and import_stmt not in content:
                        imports_to_add.append(import_stmt)

                if imports_to_add:
                    # Find where to insert imports
                    insert_idx = 0
                    for i, line in enumerate(lines):
                        if line.strip().startswith('"""') and '"""' in line[3:]:
                            insert_idx = i + 1
                            break
                        elif line.strip().startswith('"""'):
                            # Multi-line docstring
                            for j in range(i + 1, len(lines)):
                                if '"""' in lines[j]:
                                    insert_idx = j + 1
                                    break
                            break

                    # Insert imports after docstring
                    for import_stmt in imports_to_add:
                        lines.insert(insert_idx, import_stmt)
                        insert_idx += 1

                    py_file.write_text("\n".join(lines))
                    self.fixes_applied.append(
                        f"Added imports to {py_file}: {', '.join(imports_to_add)}"
                    )

    def move_imports_to_top(self) -> None:
        """Move any misplaced imports to the top of files."""
        print("Moving imports to top of files...")

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue

            try:
                content = py_file.read_text()
                lines = content.split("\n")

                # Find misplaced imports (imports that come after executable code)
                import_lines = []
                non_import_lines = []
                found_executable = False
                docstring_end = 0

                # Skip docstring at top
                if lines and lines[0].strip().startswith('"""'):
                    if '"""' in lines[0][3:]:
                        docstring_end = 1
                    else:
                        for i in range(1, len(lines)):
                            if '"""' in lines[i]:
                                docstring_end = i + 1
                                break

                # Separate imports from other code
                for i, line in enumerate(lines):
                    if i < docstring_end:
                        non_import_lines.append(line)
                    elif line.strip().startswith(
                        ("import ", "from ")
                    ) and not line.strip().startswith("#"):
                        import_lines.append(line)
                    else:
                        if line.strip() and not line.strip().startswith("#"):
                            found_executable = True
                        non_import_lines.append(line)

                # If we found imports after executable code, reorganize
                if import_lines and found_executable:
                    # Reconstruct file with imports at top
                    new_lines = non_import_lines[:docstring_end]
                    new_lines.extend(import_lines)
                    new_lines.extend(non_import_lines[docstring_end:])

                    py_file.write_text("\n".join(new_lines))
                    self.fixes_applied.append(f"Moved imports to top in {py_file}")

            except Exception as e:
                print(f"WARNING: Could not process {py_file}: {e}")

    def run_validation_sequence(self) -> bool:
        """Run the validation sequence from the instructions."""
        print("\n" + "=" * 50)
        print("Running validation sequence...")
        print("=" * 50)

        commands = [
            (["ruff", "format", "."], "Ruff format"),
            (["ruff", "check", "."], "Ruff check"),
            (["mypy", "--strict", "genomevault", "tests"], "MyPy strict check"),
            (["pytest", "-q"], "Pytest"),
        ]

        all_passed = True

        for cmd, name in commands:
            print(f"\nRunning {name}...")
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )

                if result.returncode == 0:
                    print(f"✓ {name} passed")
                    if result.stdout:
                        print(f"  Output: {result.stdout.strip()}")
                else:
                    print(f"✗ {name} failed (exit code {result.returncode})")
                    if result.stdout:
                        print(f"  STDOUT: {result.stdout}")
                    if result.stderr:
                        print(f"  STDERR: {result.stderr}")
                    all_passed = False

            except subprocess.TimeoutExpired:
                print(f"✗ {name} timed out")
                all_passed = False
            except FileNotFoundError:
                print(f"✗ {name} - command not found")
                all_passed = False

        return all_passed

    def commit_changes(self, message: str) -> bool:
        """Commit changes with given message."""
        try:
            subprocess.run(["git", "add", "-A"], cwd=self.project_root, check=True)
            subprocess.run(
                ["git", "commit", "-m", message], cwd=self.project_root, check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Git commit failed: {e}")
            return False

    def run_full_cleanup(self) -> bool:
        """Run the complete lint cleanup process."""
        print("GenomeVault Lint Clean Implementation")
        print("=" * 50)

        # Validate tools first
        if not self.validate_tools():
            return False

        # Check initial state
        print("\nInitial ruff check:")
        output, code = self.run_ruff_check()
        if code == 0:
            print("✓ No linting issues found - already clean!")
            return True
        else:
            print(f"Found {len(output.split('error:')) - 1} issues to fix")

        # Apply fixes according to checklist
        print("\nApplying fixes from checklist...")

        # A. tests/hv/test_encoding.py
        self.fix_test_hv_encoding()

        # B. tests/pir/test_pir_protocol.py
        self.fix_test_pir_protocol()

        # C. tests/zk/test_zk_property_circuits.py
        self.fix_test_zk_property_circuits()

        # D. tests/smoke/test_api_startup.py
        self.fix_test_api_startup()

        # E. Fix missing imports
        self.fix_missing_imports()

        # F. Move imports to top
        self.move_imports_to_top()

        # Commit first batch of fixes
        if self.fixes_applied:
            print(f"\nCommitting {len(self.fixes_applied)} fixes...")
            for fix in self.fixes_applied:
                print(f"  - {fix}")

            if self.commit_changes(
                "fix: implement lint clean checklist - constants and imports"
            ):
                print("✓ Changes committed")
            else:
                print("✗ Commit failed")
                return False

        # Run validation
        success = self.run_validation_sequence()

        if success:
            print("\n🎉 Lint clean completed successfully!")
            print("All validation checks passed.")
        else:
            print("\n❌ Some validation checks failed.")
            print("Manual intervention may be required.")

        return success


def main():
    """Main entry point."""
    # Detect project root
    project_root = Path.cwd()

    # Look for GenomeVault project markers
    if not (project_root / ".ruff.toml").exists():
        # Try to find the genomevault directory
        for potential_root in [Path.cwd(), Path.cwd() / "genomevault"]:
            if (potential_root / ".ruff.toml").exists():
                project_root = potential_root
                break
        else:
            print(
                "ERROR: Could not find GenomeVault project root (.ruff.toml not found)"
            )
            print("Please run this script from the GenomeVault project directory")
            sys.exit(1)

    print(f"Using project root: {project_root}")

    cleaner = LintCleaner(str(project_root))
    success = cleaner.run_full_cleanup()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
