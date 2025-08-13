#!/usr/bin/env python3
"""
Comprehensive fix script for all genomevault code issues.
Handles syntax errors, missing imports, and code organization.
"""

import os
import re
import subprocess
import sys
from typing import List, Tuple, Dict


class CodeFixer:
    """Comprehensive code fixer for Python files."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.fixed_files = []
        self.error_files = []

    def log(self, message: str, level: str = "info"):
        """Log message with color coding."""
        colors = {
            "info": "\033[94m",
            "success": "\033[92m",
            "warning": "\033[93m",
            "error": "\033[91m",
        }
        if self.verbose:
            print(f"{colors.get(level, '')}[{level.upper()}] {message}\033[0m")

    def get_syntax_errors(self) -> Dict[str, List[Tuple[int, str]]]:
        """Get all syntax errors from ruff."""
        result = subprocess.run(
            ["ruff", "check", ".", "--select", "E999"], capture_output=True, text=True
        )

        errors_by_file = {}
        for line in result.stderr.split("\n"):
            if "error: Failed to parse" in line:
                match = re.match(r"error: Failed to parse ([^:]+):(\d+):(\d+): (.+)", line)
                if match:
                    filepath, line_num, col_num, error_msg = match.groups()
                    if filepath not in errors_by_file:
                        errors_by_file[filepath] = []
                    errors_by_file[filepath].append((int(line_num), error_msg))

        return errors_by_file

    def fix_unclosed_docstring(self, filepath: str) -> bool:
        """Fix unclosed docstrings in a file."""
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()

            modified = False
            in_docstring = False
            quote_type = None

            new_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()

                if not in_docstring:
                    # Check for docstring start
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        quote_type = '"""' if '"""' in stripped else "'''"
                        # Check if it's a one-line docstring
                        if stripped.count(quote_type) >= 2:
                            new_lines.append(line)
                        else:
                            in_docstring = True
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                else:
                    # In docstring, look for end
                    if quote_type in line:
                        in_docstring = False
                        new_lines.append(line)
                    else:
                        # Check if next line is code without closing docstring
                        if i < len(lines) - 1:
                            next_line = lines[i + 1].strip()
                            # If next line looks like code, close docstring
                            if (
                                next_line
                                and not next_line.startswith("#")
                                and (
                                    next_line.startswith("from ")
                                    or next_line.startswith("import ")
                                    or next_line.startswith("class ")
                                    or next_line.startswith("def ")
                                    or next_line.startswith("@")
                                )
                            ):
                                # Add closing quotes
                                new_lines.append(line)
                                new_lines.append(quote_type + "\n")
                                in_docstring = False
                                modified = True
                            else:
                                new_lines.append(line)
                        else:
                            # End of file, close docstring if still open
                            new_lines.append(line)
                            if in_docstring:
                                new_lines.append(quote_type + "\n")
                                modified = True
                                in_docstring = False
                i += 1

            # If still in docstring at end, close it
            if in_docstring:
                new_lines.append(quote_type + "\n")
                modified = True

            if modified:
                with open(filepath, "w") as f:
                    f.writelines(new_lines)
                return True
        except Exception as e:
            self.log(f"Error fixing docstrings in {filepath}: {e}", "error")
        return False

    def fix_malformed_fstrings(self, filepath: str) -> bool:
        """Fix malformed f-strings."""
        try:
            with open(filepath, "r") as f:
                content = f.read()

            modified = False

            # Fix f-strings with wrong % formatting
            pattern = r'f"([^"]*?)%s([^"]*?)"'
            if re.search(pattern, content):
                content = re.sub(
                    pattern, lambda m: f'f"{m.group(1)}{{}}' + m.group(2) + '"', content
                )
                modified = True

            # Fix f-strings with missing closing braces
            lines = content.split("\n")
            new_lines = []
            for line in lines:
                if 'f"' in line or "f'" in line:
                    # Count braces
                    open_braces = line.count("{")
                    close_braces = line.count("}")
                    if open_braces > close_braces:
                        # Try to fix by adding closing braces before quote
                        if line.rstrip().endswith('"'):
                            line = line[:-1] + "}" * (open_braces - close_braces) + '"'
                            modified = True
                        elif line.rstrip().endswith("'"):
                            line = line[:-1] + "}" * (open_braces - close_braces) + "'"
                            modified = True
                new_lines.append(line)

            if modified:
                with open(filepath, "w") as f:
                    f.write("\n".join(new_lines))
                return True
        except Exception as e:
            self.log(f"Error fixing f-strings in {filepath}: {e}", "error")
        return False

    def add_missing_imports(self, filepath: str) -> bool:
        """Add missing imports to a file."""
        try:
            with open(filepath, "r") as f:
                content = f.read()

            # Check what's undefined
            result = subprocess.run(
                ["ruff", "check", filepath, "--select", "F821"],
                capture_output=True,
                text=True,
            )

            undefined_names = set()
            for line in result.stdout.split("\n"):
                if "F821" in line:
                    match = re.search(r"Undefined name `([^`]+)`", line)
                    if match:
                        undefined_names.add(match.group(1))

            if not undefined_names:
                return False

            # Common imports to add
            imports_to_add = []

            if "Callable" in undefined_names:
                if "from typing import" in content:
                    # Add to existing typing import
                    content = re.sub(
                        r"from typing import ([^\n]+)",
                        lambda m: (
                            f"from typing import {m.group(1)}, Callable"
                            if "Callable" not in m.group(1)
                            else m.group(0)
                        ),
                        content,
                        count=1,
                    )
                else:
                    imports_to_add.append("from typing import Callable")

            if (
                "Dict" in undefined_names
                or "List" in undefined_names
                or "Optional" in undefined_names
            ):
                typing_imports = []
                for name in ["Dict", "List", "Optional", "Any", "Tuple", "Union"]:
                    if name in undefined_names:
                        typing_imports.append(name)

                if typing_imports:
                    if "from typing import" in content:
                        # Add to existing
                        content = re.sub(
                            r"from typing import ([^\n]+)",
                            lambda m: (
                                f"from typing import {m.group(1)}, {', '.join(typing_imports)}"
                                if not all(t in m.group(1) for t in typing_imports)
                                else m.group(0)
                            ),
                            content,
                            count=1,
                        )
                    else:
                        imports_to_add.append(f"from typing import {', '.join(typing_imports)}")

            # Add common missing imports
            if "logger" in undefined_names and "get_logger" not in content:
                imports_to_add.append("from genomevault.utils.logging import get_logger")
                imports_to_add.append("logger = get_logger(__name__)")

            if "np" in undefined_names and "import numpy" not in content:
                imports_to_add.append("import numpy as np")

            if "pd" in undefined_names and "import pandas" not in content:
                imports_to_add.append("import pandas as pd")

            # Add imports after module docstring and future imports
            if imports_to_add:
                lines = content.split("\n")
                insert_idx = 0

                # Skip module docstring
                if lines[0].strip().startswith('"""') or lines[0].strip().startswith("'''"):
                    for i, line in enumerate(lines):
                        if i > 0 and (line.strip().endswith('"""') or line.strip().endswith("'''")):
                            insert_idx = i + 1
                            break

                # Skip future imports
                for i in range(insert_idx, len(lines)):
                    if lines[i].strip().startswith("from __future__"):
                        insert_idx = i + 1
                    elif lines[i].strip() and not lines[i].strip().startswith("#"):
                        break

                # Insert imports
                for imp in imports_to_add:
                    lines.insert(insert_idx + 1, imp)
                    insert_idx += 1

                content = "\n".join(lines)

                with open(filepath, "w") as f:
                    f.write(content)
                return True

        except Exception as e:
            self.log(f"Error adding imports to {filepath}: {e}", "error")
        return False

    def fix_import_order(self, filepath: str) -> bool:
        """Fix import ordering and placement."""
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()

            # Categorize lines
            docstring_lines = []
            future_imports = []
            stdlib_imports = []
            third_party_imports = []
            local_imports = []
            code_lines = []

            in_docstring = False
            docstring_quote = None
            found_code = False

            for line in lines:
                stripped = line.strip()

                # Handle docstring
                if not found_code and not in_docstring:
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        docstring_quote = '"""' if stripped.startswith('"""') else "'''"
                        docstring_lines.append(line)
                        if stripped.count(docstring_quote) >= 2:
                            continue  # One-line docstring
                        in_docstring = True
                        continue

                if in_docstring:
                    docstring_lines.append(line)
                    if docstring_quote in line:
                        in_docstring = False
                    continue

                # Categorize imports
                if stripped.startswith("from __future__ import"):
                    future_imports.append(line)
                elif stripped.startswith("import ") or stripped.startswith("from "):
                    if not found_code:
                        if "genomevault" in stripped or stripped.startswith("from ."):
                            local_imports.append(line)
                        elif any(
                            pkg in stripped for pkg in ["numpy", "pandas", "torch", "sklearn"]
                        ):
                            third_party_imports.append(line)
                        else:
                            stdlib_imports.append(line)
                    else:
                        code_lines.append(line)  # Import after code
                elif stripped and not stripped.startswith("#"):
                    found_code = True
                    code_lines.append(line)
                else:
                    if found_code or (
                        not future_imports
                        and not stdlib_imports
                        and not third_party_imports
                        and not local_imports
                    ):
                        code_lines.append(line)

            # Rebuild file
            new_lines = []

            # Add docstring
            if docstring_lines:
                new_lines.extend(docstring_lines)
                if not docstring_lines[-1].strip():
                    new_lines.append("\n")

            # Add imports in correct order
            if future_imports:
                new_lines.extend(sorted(future_imports))
                new_lines.append("\n")

            if stdlib_imports:
                new_lines.extend(sorted(stdlib_imports))
                if third_party_imports or local_imports:
                    new_lines.append("\n")

            if third_party_imports:
                new_lines.extend(sorted(third_party_imports))
                if local_imports:
                    new_lines.append("\n")

            if local_imports:
                new_lines.extend(sorted(local_imports))
                new_lines.append("\n")

            # Add code
            new_lines.extend(code_lines)

            # Write back
            with open(filepath, "w") as f:
                f.writelines(new_lines)

            return True

        except Exception as e:
            self.log(f"Error fixing import order in {filepath}: {e}", "error")
        return False

    def fix_file(self, filepath: str) -> bool:
        """Apply all fixes to a single file."""
        fixed_any = False

        # Fix docstrings first
        if self.fix_unclosed_docstring(filepath):
            fixed_any = True
            self.log(f"Fixed docstrings in {filepath}", "success")

        # Fix f-strings
        if self.fix_malformed_fstrings(filepath):
            fixed_any = True
            self.log(f"Fixed f-strings in {filepath}", "success")

        # Add missing imports
        if self.add_missing_imports(filepath):
            fixed_any = True
            self.log(f"Added missing imports to {filepath}", "success")

        # Fix import order
        if self.fix_import_order(filepath):
            fixed_any = True
            self.log(f"Fixed import order in {filepath}", "success")

        return fixed_any

    def fix_all(self):
        """Fix all Python files in the project."""
        self.log("Starting comprehensive fix process...", "info")

        # Get all Python files
        py_files = []
        for root, dirs, files in os.walk("genomevault"):
            for file in files:
                if file.endswith(".py"):
                    py_files.append(os.path.join(root, file))

        for root, dirs, files in os.walk("tests"):
            for file in files:
                if file.endswith(".py"):
                    py_files.append(os.path.join(root, file))

        for root, dirs, files in os.walk("examples"):
            for file in files:
                if file.endswith(".py"):
                    py_files.append(os.path.join(root, file))

        self.log(f"Found {len(py_files)} Python files to check", "info")

        # Fix each file
        for filepath in py_files:
            try:
                if self.fix_file(filepath):
                    self.fixed_files.append(filepath)
            except Exception as e:
                self.log(f"Error processing {filepath}: {e}", "error")
                self.error_files.append(filepath)

        # Run ruff auto-fix
        self.log("\nRunning ruff auto-fix...", "info")
        subprocess.run(["ruff", "check", ".", "--fix", "--unsafe-fixes"], capture_output=True)

        # Summary
        self.log(f"\n{'='*60}", "info")
        self.log(f"Fixed {len(self.fixed_files)} files", "success")
        if self.error_files:
            self.log(f"Errors in {len(self.error_files)} files", "warning")

        # Final check
        result = subprocess.run(
            ["ruff", "check", ".", "--statistics"], capture_output=True, text=True
        )

        self.log("\nRemaining issues:", "info")
        for line in result.stdout.split("\n")[:10]:
            if line.strip():
                self.log(f"  {line}", "info")

        return len(self.fixed_files) > 0


def main():
    """Main entry point."""
    fixer = CodeFixer(verbose=True)
    success = fixer.fix_all()

    if success:
        print("\n✅ Comprehensive fixes applied!")
        print("Run 'ruff check .' to see remaining issues")
    else:
        print("\n⚠️ No fixes were applied")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
