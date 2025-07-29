#!/usr/bin/env python3
"""
GenomeVault Comprehensive Fix Script
Based on audit report v2 from 2025-07-27
"""

import datetime
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GenomeVaultFixer:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.backup_dir = (
            self.base_path.parent
            / f"genomevault_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    def create_backup(self):
        """Create a backup of the entire codebase before making changes"""
        if self.backup_dir.exists():
            logger.warning(f"Backup directory already exists: {self.backup_dir}")
            return

        logger.info(f"Creating backup at: {self.backup_dir}")
        shutil.copytree(
            self.base_path,
            self.backup_dir,
            ignore=shutil.ignore_patterns(".git", "venv", "__pycache__"),
        )

    def add_missing_init_files(self):
        """Add missing __init__.py files to directories"""
        missing_init_dirs = [
            "genomevault/blockchain/contracts",
            "genomevault/blockchain/node",
            "genomevault/cli",
            "genomevault/clinical",
            "genomevault/clinical/diabetes_pilot",
            "genomevault/federation",
            "genomevault/governance",
            "genomevault/kan",
            "genomevault/pir/reference_data",
            "genomevault/zk_proofs/examples",
            "scripts",
            "tests",
            "tests/adversarial",
            "tests/e2e",
            "tests/integration",
            "tests/property",
            "tests/unit",
            "tests/zk",
        ]

        for dir_path in missing_init_dirs:
            full_path = self.base_path / dir_path
            init_file = full_path / "__init__.py"

            if full_path.exists() and not init_file.exists():
                logger.info(f"Creating __init__.py in {dir_path}")
                init_file.write_text('"""Module initialization."""\n')

    def replace_print_with_logging(self):
        """Replace print() calls with proper logging"""
        files_with_prints = [
            ("examples/minimal_verification.py", 33),
            ("devtools/verify_fix.py", 29),
            ("devtools/diagnose_imports.py", 27),
            ("devtools/diagnose_failures.py", 25),
            ("genomevault/hypervector_transform/advanced_compression.py", 24),
            ("genomevault/zk_proofs/advanced/catalytic_proof.py", 23),
            ("devtools/trace_import_failure.py", 22),
            ("genomevault/pir/advanced/it_pir.py", 20),
            ("genomevault/blockchain/node.py", 17),
            ("genomevault/pir/advanced/robust_it_pir.py", 16),
            ("tests/test_zk_median_circuit.py", 16),
            ("genomevault/clinical/diabetes_pilot/risk_calculator.py", 16),
            ("genomevault/blockchain/governance.py", 14),
            ("genomevault/hypervector_transform/hierarchical.py", 12),
            ("genomevault/security/phi_detector.py", 11),
        ]

        for file_path, _ in files_with_prints:
            full_path = self.base_path / file_path
            if full_path.exists():
                self._convert_prints_to_logging(full_path)

    def _convert_prints_to_logging(self, file_path: Path):
        """Convert print statements to logging in a single file"""
        logger.info(f"Converting print statements to logging in {file_path}")

        try:
            content = file_path.read_text()
            original_content = content

            # Add logging import if not present
            if "import logging" not in content:
                # Find the right place to add the import
                import_lines = []
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("import ") or line.startswith("from "):
                        import_lines.append(i)

                if import_lines:
                    insert_pos = max(import_lines) + 1
                else:
                    # Insert after module docstring
                    insert_pos = 0
                    if lines[0].startswith('"""') or lines[0].startswith("'''"):
                        for i, line in enumerate(lines[1:], 1):
                            if line.endswith('"""') or line.endswith("'''"):
                                insert_pos = i + 1
                                break

                lines.insert(insert_pos, "import logging")
                lines.insert(insert_pos + 1, "")
                lines.insert(insert_pos + 2, "logger = logging.getLogger(__name__)")
                lines.insert(insert_pos + 3, "")
                content = "\n".join(lines)

            # Replace print statements
            # Simple print() -> logger.info()
            content = re.sub(
                r"^(\s*)print\((.*?)\)$",
                r"\1logger.info(\2)",
                content,
                flags=re.MULTILINE,
            )

            # print(f"...") -> logger.info(f"...")
            content = re.sub(
                r'^(\s*)print\(f(["\'])(.*?)\2\)$',
                r"\1logger.info(f\2\3\2)",
                content,
                flags=re.MULTILINE,
            )

            # print("Error:", ...) -> logger.error(...)
            content = re.sub(
                r'^(\s*)print\(["\']Error:?["\'],?\s*(.*?)\)$',
                r"\1logger.error(\2)",
                content,
                flags=re.MULTILINE,
            )

            # print("Warning:", ...) -> logger.warning(...)
            content = re.sub(
                r'^(\s*)print\(["\']Warning:?["\'],?\s*(.*?)\)$',
                r"\1logger.warning(\2)",
                content,
                flags=re.MULTILINE,
            )

            # print("Debug:", ...) -> logger.debug(...)
            content = re.sub(
                r'^(\s*)print\(["\']Debug:?["\'],?\s*(.*?)\)$',
                r"\1logger.debug(\2)",
                content,
                flags=re.MULTILINE,
            )

            if content != original_content:
                file_path.write_text(content)
                logger.info(f"Updated {file_path}")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    def fix_broad_exceptions(self):
        """Replace broad 'except Exception' with more specific exception handling"""
        files_with_broad_excepts = [
            ("genomevault/utils/backup.py", 8),
            ("genomevault/zk_proofs/verifier.py", 8),
            ("genomevault/api/app.py", 6),
            ("genomevault/utils/common.py", 4),
            ("genomevault/api/routers/tuned_query.py", 3),
            ("genomevault/pir/client/pir_client.py", 3),
            ("genomevault/api/routers/query_tuned.py", 3),
            ("tests/adversarial/test_hdc_adversarial.py", 3),
            ("genomevault/pir/secure_wrapper.py", 3),
            ("scripts/run_bench.py", 3),
            ("genomevault/integration/proof_of_training.py", 3),
        ]

        for file_path, _ in files_with_broad_excepts:
            full_path = self.base_path / file_path
            if full_path.exists():
                self._fix_broad_exceptions_in_file(full_path)

    def _fix_broad_exceptions_in_file(self, file_path: Path):
        """Fix broad exceptions in a single file"""
        logger.info(f"Fixing broad exceptions in {file_path}")

        try:
            content = file_path.read_text()
            lines = content.split("\n")

            # Analyze the file to understand what exceptions might be raised
            new_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]

                if "except Exception" in line:
                    # Look at the try block to determine appropriate exceptions
                    try_start = i - 1
                    while try_start >= 0 and not lines[try_start].strip().startswith("try:"):
                        try_start -= 1

                    if try_start >= 0:
                        # Analyze the try block
                        try_block = "\n".join(lines[try_start + 1 : i])

                        # Determine appropriate exceptions based on common patterns
                        exceptions = self._determine_exceptions(try_block)

                        if exceptions:
                            # Replace with specific exceptions
                            indent = re.match(r"^(\s*)", line).group(1)
                            if len(exceptions) == 1:
                                new_lines.append(f"{indent}except {exceptions[0]} as e:")
                            else:
                                new_lines.append(f"{indent}except ({', '.join(exceptions)}) as e:")
                        else:
                            # Keep the original if we can't determine specific exceptions
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
                i += 1

            new_content = "\n".join(new_lines)
            if new_content != content:
                file_path.write_text(new_content)
                logger.info(f"Updated {file_path}")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    def _determine_exceptions(self, try_block: str) -> list[str]:
        """Determine appropriate exceptions based on code patterns"""
        exceptions = []

        # File operations
        if any(word in try_block for word in ["open(", "read(", "write(", "Path(", "os.path"]):
            exceptions.extend(["FileNotFoundError", "IOError", "PermissionError"])

        # Network operations
        if any(word in try_block for word in ["requests.", "urllib", "http", "socket"]):
            exceptions.extend(["ConnectionError", "TimeoutError", "RequestException"])

        # JSON operations
        if "json." in try_block:
            exceptions.append("json.JSONDecodeError")

        # Database operations
        if any(word in try_block for word in ["cursor", "execute", "commit", "rollback"]):
            exceptions.append("DatabaseError")

        # Value/Type errors
        if any(word in try_block for word in ["int(", "float(", "parse", "convert"]):
            exceptions.extend(["ValueError", "TypeError"])

        # Key errors
        if "[" in try_block and "]" in try_block:
            exceptions.append("KeyError")

        # Import errors
        if "import" in try_block:
            exceptions.append("ImportError")

        # Remove duplicates and return
        return list(set(exceptions))

    def refactor_complex_functions(self):
        """Refactor functions with high complexity"""
        complex_functions = [
            ("genomevault/cli/training_proof_cli.py", "verify_proof", 20),
            (
                "genomevault/local_processing/epigenetics.py",
                "find_differential_peaks",
                20,
            ),
            (
                "genomevault/hypervector_transform/hdc_encoder.py",
                "_extract_features",
                20,
            ),
            (
                "genomevault/local_processing/proteomics.py",
                "differential_expression",
                17,
            ),
            ("genomevault/local_processing/sequencing.py", "_run_quality_control", 17),
        ]

        for file_path, func_name, complexity in complex_functions[:3]:  # Start with top 3
            full_path = self.base_path / file_path
            if full_path.exists():
                logger.info(f"TODO: Refactor {func_name} in {file_path} (complexity: {complexity})")
                # Note: Actual refactoring would require understanding the business logic
                # For now, we'll add TODO comments
                self._add_refactor_todo(full_path, func_name)

    def _add_refactor_todo(self, file_path: Path, func_name: str):
        """Add a TODO comment for refactoring"""
        try:
            content = file_path.read_text()
            lines = content.split("\n")

            # Find the function
            for i, line in enumerate(lines):
                if f"def {func_name}(" in line:
                    # Add TODO comment before the function
                    indent = re.match(r"^(\s*)", line).group(1)
                    todo_comment = f"{indent}# TODO: Refactor this function to reduce complexity"
                    if i > 0 and lines[i - 1].strip() != todo_comment.strip():
                        lines.insert(i, todo_comment)
                        break

            new_content = "\n".join(lines)
            file_path.write_text(new_content)

        except Exception as e:
            logger.error(f"Error adding TODO to {file_path}: {e}")

    def add_type_annotations(self):
        """Add type annotations to functions missing them"""
        logger.info("Adding type annotations to improve coverage from 47.5% to 80%+")
        # This would require analyzing each function signature
        # For now, we'll focus on the most important files

    def create_missing_documentation(self):
        """Create missing README and improve documentation"""
        readme_path = self.base_path / "README.md"
        if not readme_path.exists():
            logger.info("Creating README.md")
            readme_content = """# GenomeVault

A secure, privacy-preserving genomic data management system.

## Features

- Hyperdimensional computing for genomic data encoding
- Zero-knowledge proofs for privacy preservation
- Private Information Retrieval (PIR)
- Blockchain integration for data integrity
- HIPAA compliance features

## Installation

```bash
pip install -r requirements.txt
python setup.py install
```

## Usage

See the `examples/` directory for usage examples.

## Documentation

Full documentation is available in the `docs/` directory.

## License

See LICENSE file for details.
"""
            readme_path.write_text(readme_content)

    def run_all_fixes(self):
        """Run all fixes in sequence"""
        logger.info("Starting GenomeVault comprehensive fixes...")

        # Create backup first
        self.create_backup()

        # Run fixes
        logger.info("\n1. Adding missing __init__.py files...")
        self.add_missing_init_files()

        logger.info("\n2. Replacing print() with logging...")
        self.replace_print_with_logging()

        logger.info("\n3. Fixing broad exception handlers...")
        self.fix_broad_exceptions()

        logger.info("\n4. Adding refactoring TODOs for complex functions...")
        self.refactor_complex_functions()

        logger.info("\n5. Creating missing documentation...")
        self.create_missing_documentation()

        logger.info("\nAll fixes completed!")
        logger.info(f"Backup created at: {self.backup_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "/Users/rohanvinaik/genomevault"

    fixer = GenomeVaultFixer(base_path)
    fixer.run_all_fixes()
