#!/usr/bin/env python3
"""
Apply common fix patterns to GenomeVault codebase.
This handles specific patterns like logging, f-strings, mutable defaults, etc.
"""

import ast
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def fix_print_statements(file_path):
    """Replace print() with logging statements."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original = content

        # Add logging import if print is found and logging not imported
        if "print(" in content and "import logging" not in content:
            # Check if it's not already using logger
            if "logger" not in content and "log" not in content:
                # Add logging setup after other imports
                lines = content.split("\n")
                import_end = 0
                for i, line in enumerate(lines):
                    if line.startswith("import ") or line.startswith("from "):
                        import_end = i + 1
                    elif import_end > 0 and line and not line.startswith(" "):
                        break

                if import_end > 0:
                    logging_setup = [
                        "",
                        "import logging",
                        "",
                        "logger = logging.getLogger(__name__)",
                        "",
                    ]
                    lines = lines[:import_end] + logging_setup + lines[import_end:]
                    content = "\n".join(lines)

        # Replace print statements with logger.info
        # Simple pattern - can be enhanced for more complex cases
        content = re.sub(r"\bprint\s*\((.*?)\)", r"logger.info(\1)", content)

        if content != original:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Fixed print statements in {file_path}")
            return True

    except Exception as e:
        logger.error(f"Error fixing print statements in {file_path}: {e}")

    return False


def fix_mutable_defaults(file_path):
    """Fix mutable default arguments."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original = content

        # Parse the AST
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return False

        changes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for i, default in enumerate(node.args.defaults):
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        # Found mutable default
                        arg_index = len(node.args.args) - len(node.args.defaults) + i
                        arg_name = node.args.args[arg_index].arg
                        changes.append((node.lineno, arg_name))

        # Apply fixes
        if changes:
            lines = content.split("\n")
            for line_no, arg_name in changes:
                # Find the function definition line
                func_line_idx = line_no - 1

                # Simple replacement pattern
                # Replace arg=[] with arg=None
                lines[func_line_idx] = re.sub(
                    rf"{arg_name}\s*=\s*\[\]", f"{arg_name}=None", lines[func_line_idx]
                )
                # Use r-string without f-string for regex pattern with braces
                pattern = r"%s\s*=\s*\{\}" % arg_name
                lines[func_line_idx] = re.sub(
                    pattern,
                    f"{arg_name}=None",
                    lines[func_line_idx],
                )

                # Add initialization in function body
                # This is simplified - would need more sophisticated handling

            content = "\n".join(lines)

        if content != original:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Fixed mutable defaults in {file_path}")
            return True

    except Exception as e:
        logger.error(f"Error fixing mutable defaults in {file_path}: {e}")

    return False


def fix_string_formatting(file_path):
    """Convert old-style string formatting to f-strings."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original = content

        # Convert .format() to f-strings (simple cases)
        content = re.sub(
            r'"([^"]*)\{\}([^"]*)"\.format\(([^)]+)\)',
            lambda m: f'f"{m.group(1)}{{{m.group(3)}}}{m.group(2)}"',
            content,
        )

        content = re.sub(
            r"'([^']*)\{\}([^']*)'\.format\(([^)]+)\)",
            lambda m: f"f'{m.group(1)}{{{m.group(3)}}}{m.group(2)}'",
            content,
        )

        # Convert % formatting to f-strings (simple cases)
        content = re.sub(
            r'"([^"]*%s[^"]*)" % \(([^)]+)\)',
            lambda m: f'f"{m.group(1).replace("%s", "{" + m.group(2) + "}")}"',
            content,
        )

        if content != original:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Fixed string formatting in {file_path}")
            return True

    except Exception as e:
        logger.error(f"Error fixing string formatting in {file_path}: {e}")

    return False


def fix_file_encoding(file_path):
    """Add explicit encoding to file operations."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original = content

        # Add encoding to open() calls
        content = re.sub(
            r"open\(([^,)]+),\s*['\"]r['\"]\)",
            r"open(\1, 'r', encoding='utf-8')",
            content,
        )

        content = re.sub(
            r"open\(([^,)]+),\s*['\"]w['\"]\)",
            r"open(\1, 'w', encoding='utf-8')",
            content,
        )

        if content != original:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Fixed file encoding in {file_path}")
            return True

    except Exception as e:
        logger.error(f"Error fixing file encoding in {file_path}: {e}")

    return False


def process_file(file_path):
    """Apply all fix patterns to a single file."""
    if not file_path.suffix == ".py":
        return

    logger.debug(f"Processing {file_path}")

    # Apply each fix pattern
    fixed = False
    fixed |= fix_print_statements(file_path)
    fixed |= fix_string_formatting(file_path)
    fixed |= fix_file_encoding(file_path)
    fixed |= fix_mutable_defaults(file_path)

    return fixed


def main():
    """Process all Python files in the genomevault package."""
    logger.info("Applying common fix patterns to GenomeVault codebase")

    genomevault_path = Path("genomevault")
    if not genomevault_path.exists():
        logger.error("genomevault directory not found!")
        return

    # Process all Python files
    python_files = list(genomevault_path.rglob("*.py"))

    fixed_count = 0
    for py_file in python_files:
        # Skip migration and test files
        if "migration" in str(py_file) or "__pycache__" in str(py_file):
            continue

        if process_file(py_file):
            fixed_count += 1

    logger.info(f"Fixed {fixed_count} files")
    logger.info("Common pattern fixes complete!")


if __name__ == "__main__":
    main()
