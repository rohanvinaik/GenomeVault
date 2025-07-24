#!/usr/bin/env python3
"""
GenomeVault Pylint Automated Fix Script
Systematically fixes common Pylint issues in the GenomeVault codebase.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple


class GenomeVaultPylintFixer:
    """Automated fixer for common Pylint issues in GenomeVault codebase."""

    def __init__(self, root_path: str):
        """Magic method implementation."""
        self.root_path = Path(root_path)
        self.genomevault_path = self.root_path / "genomevault"

        # Common typing imports that need to be added
        self.typing_imports = {
            "List",
            "Dict",
            "Any",
            "Optional",
            "Union",
            "Tuple",
            "Set",
            "Callable",
            "Iterator",
            "Mapping",
        }

        # Standard library imports that should be at the top
        self.stdlib_imports = {
            "os",
            "sys",
            "time",
            "json",
            "logging",
            "hashlib",
            "pathlib",
            "datetime",
            "uuid",
            "typing",
        }

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the genomevault package."""
        python_files = []
        for path in self.genomevault_path.rglob("*.py"):
            if not any(part.startswith(".") for part in path.parts):
                python_files.append(path)
        return python_files

    def analyze_file_imports(self, content: str) -> Tuple[Set[str], Set[str]]:
        """Analyze what typing imports and undefined variables are needed."""
        typing_needed = set()
        undefined_vars = set()

        # Look for typing usage without imports
        for typing_import in self.typing_imports:
            if re.search(rf"\b{typing_import}\b", content):
                typing_needed.add(typing_import)

        # Look for common undefined variable patterns
        if re.search(r"Optional\[", content):
            typing_needed.add("Optional")
        if re.search(r"List\[", content):
            typing_needed.add("List")
        if re.search(r"Dict\[", content):
            typing_needed.add("Dict")

        return typing_needed, undefined_vars

    def fix_imports(self, content: str, filepath: Path) -> str:
        """Fix import-related issues in a file."""
        lines = content.split("\n")
        new_lines = []

        # Track existing imports
        has_typing_import = False
        import_section_end = 0

        # First pass: analyze existing imports and find where to insert
        for i, line in enumerate(lines):
            if line.strip().startswith("from typing import") or line.strip().startswith(
                "import typing"
            ):
                has_typing_import = True
            if (
                line.strip()
                and not line.startswith("#")
                and not line.startswith("import")
                and not line.startswith("from")
            ):
                if import_section_end == 0:
                    import_section_end = i
                break

        # Analyze what typing imports are needed
        typing_needed, _ = self.analyze_file_imports(content)

        # Process lines
        for i, line in enumerate(lines):
            # Skip processing if we're in import section and need to add typing
            if i == import_section_end and typing_needed and not has_typing_import:
                # Add typing import
                typing_import_line = (
                    f"from typing import {', '.join(sorted(typing_needed))}"
                )
                new_lines.append(typing_import_line)
                new_lines.append("")

            new_lines.append(line)

        return "\n".join(new_lines)

    def fix_pydantic_validators(self, content: str) -> str:
        """Fix Pydantic validator method signatures."""
        # Fix validators missing @classmethod decorator and cls parameter
        patterns = [
            (
                r"(\s*)@validator\((.*?)\)\s*\n(\s*)def\s+(\w+)\s*\(\s*([^,)]*)\s*\):",
                r"\1@validator(\2)\n\1@classmethod\n\3def \4(cls, \5):",
            ),
            # Fix existing validators that have wrong signature
            (
                r"(\s*)@validator\((.*?)\)\s*\n(\s*)def\s+(\w+)\s*\(\s*([^,)]*)\s*, ?\s*([^)]*)\s*\):",
                r"\1@validator(\2)\n\1@classmethod\n\3def \4(cls, v):",
            ),
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        return content

    def fix_undefined_variables(self, content: str, filepath: Path) -> str:
        """Fix common undefined variable issues."""
        fixes = []

        # Common fixes for specific files
        if "prover.py" in str(filepath):
            fixes.extend(
                [
                    # Fix circuit variable
                    (
                        r"(\s+)([a-zA-Z_]\w*\s* = \s*)?circuit\s* = ",
                        r"\1circuit = self._get_circuit(circuit_type)",
                    ),
                    # Fix timing variables
                    (r"(\s+)start_time\s* = ", r"\1start_time = time.time()"),
                    # Fix undefined variables in method context
                    (
                        r"(\s+)proof_id\s* = ",
                        r'\1proof_id = f"proof_{uuid.uuid4().hex[:8]}"',
                    ),
                    (r"(\s+)proof_data\s* = ", r"\1proof_data = {}"),
                ]
            )

        if "app.py" in str(filepath):
            fixes.extend(
                [
                    # Fix FastAPI app variable
                    (
                        r"^(\s*)app\s* = ",
                        r'\1app = FastAPI(title = "GenomeVault API", version = "3.0.0")',
                    ),
                    # Fix undefined variables
                    (r"(\s+)config\s* = ", r"\1config = get_config()"),
                    (
                        r"(\s+)credit_ledger\s* = ",
                        r"\1credit_ledger = get_credit_ledger()",
                    ),
                ]
            )

        for pattern, replacement in fixes:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        return content

    def fix_code_quality_issues(self, content: str) -> str:
        """Fix various code quality issues."""
        # Remove superfluous parentheses after 'not'
        content = re.sub(r"\bnot\s*\(\s*([^)]+)\s*\)", r"not \1", content)

        # Fix unnecessary elif after return
        content = re.sub(
            r"(\s+return\s+[^\n]+\n\s+)elif\b", r"\1if", content, flags=re.MULTILINE
        )

        # Fix broad exception catching (basic cases)
        content = re.sub(
            r"except Exception as ([a-zA-Z_]\w*):", r"except Exception as \1:", content
        )

        # Add missing 'from' in raise statements
        content = re.sub(
            r"(\s+)raise\s+([A-Z]\w*Error?\([^)]*\))\s*$",
            r"\1raise \2 from e",
            content,
            flags=re.MULTILINE,
        )

        return content

    def add_missing_imports_for_file(self, content: str, filepath: Path) -> str:
        """Add commonly missing imports based on file content."""
        lines = content.split("\n")

        # Detect missing imports based on usage
        missing_imports = []

        if "FastAPI" in content and "from fastapi" not in content:
            missing_imports.append(
                "from fastapi import FastAPI, HTTPException, Depends"
            )

        if "BaseModel" in content and "from pydantic" not in content:
            missing_imports.append("from pydantic import BaseModel, validator")

        if "np." in content and "import numpy" not in content:
            missing_imports.append("import numpy as np")

        if "uuid." in content and "import uuid" not in content:
            missing_imports.append("import uuid")

        if "time." in content and "import time" not in content:
            missing_imports.append("import time")

        if missing_imports:
            # Find where to insert imports
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('"""') or line.strip().startswith("'''"):
                    # Skip docstrings
                    quote_type = '"""' if line.strip().startswith('"""') else "'''"
                    for j in range(i + 1, len(lines)):
                        if quote_type in lines[j]:
                            insert_idx = j + 1
                            break
                    break
                elif line.strip() and not line.startswith("#"):
                    insert_idx = i
                    break

            # Insert missing imports
            for imp in missing_imports:
                lines.insert(insert_idx, imp)
                insert_idx += 1

            # Add empty line after imports
            if missing_imports:
                lines.insert(insert_idx, "")

        return "\n".join(lines)

    def fix_file(self, filepath: Path) -> bool:
        """Fix a single Python file."""
        try:
            print(f"Fixing {filepath}...")

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Apply fixes in order
            content = self.add_missing_imports_for_file(content, filepath)
            content = self.fix_imports(content, filepath)
            content = self.fix_pydantic_validators(content)
            content = self.fix_undefined_variables(content, filepath)
            content = self.fix_code_quality_issues(content)

            # Only write if content changed
            if content != original_content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"  ✓ Fixed {filepath}")
                return True
            else:
                print(f"  - No changes needed for {filepath}")
                return False

        except Exception as e:
            print(f"  ✗ Error fixing {filepath}: {e}")
            return False

    def run_fixes(self) -> None:
        """Run all fixes on the GenomeVault codebase."""
        print("🔧 Starting GenomeVault Pylint fixes...")
        print(f"📁 Working directory: {self.root_path}")

        python_files = self.find_python_files()
        print(f"📄 Found {len(python_files)} Python files")

        fixed_count = 0
        error_count = 0

        for filepath in python_files:
            try:
                if self.fix_file(filepath):
                    fixed_count += 1
            except Exception as e:
                print(f"❌ Error processing {filepath}: {e}")
                error_count += 1

        print(f"\n✅ Fixes complete!")
        print(f"📊 Files fixed: {fixed_count}")
        print(f"📊 Files with errors: {error_count}")
        print(f"📊 Total files processed: {len(python_files)}")

        # Suggest next steps
        print(f"\n📋 Next steps:")
        print(f"1. Install dependencies: pip install -r requirements.txt")
        print(f"2. Run Pylint again: pylint genomevault/")
        print(f"3. Run tests: python -m pytest tests/")
        print(f"4. Check imports: python -c 'import genomevault'")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
    else:
        root_path = "/Users/rohanvinaik/genomevault"

    if not os.path.exists(root_path):
        print(f"❌ Path does not exist: {root_path}")
        sys.exit(1)

    fixer = GenomeVaultPylintFixer(root_path)
    fixer.run_fixes()


if __name__ == "__main__":
    main()
