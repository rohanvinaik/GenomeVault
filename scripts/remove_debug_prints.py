#!/usr/bin/env python3
"""
Automated Debug Print Removal Script
=====================================

This script scans for debug print statements and replaces them with proper logging.
It handles various print patterns and converts them to appropriate log levels.

Usage:
    python remove_debug_prints.py [--dry-run] [--verbose] [directory...]
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Configure logging for the script itself
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
script_logger = logging.getLogger(__name__)


class PrintStatementTransformer(ast.NodeTransformer):
    """AST transformer to replace print statements with logging calls."""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.has_logger = False
        self.needs_import = False
        self.print_count = 0
        self.replacements = []

    def visit_ImportFrom(self, node):
        """Check if logging is already imported."""
        if node.module == "genomevault.utils.logging":
            self.has_logger = True
        return node

    def visit_Call(self, node):
        """Transform print() calls to logging calls."""
        self.generic_visit(node)

        if isinstance(node.func, ast.Name) and node.func.id == "print":
            self.print_count += 1
            self.needs_import = True

            # Determine log level based on content
            log_level = self._determine_log_level(node)

            # Create logging call
            if log_level == "skip":
                # Return None to remove the statement
                return None

            new_node = self._create_logging_call(node, log_level)
            return new_node

        return node

    def _determine_log_level(self, print_node) -> str:
        """Determine appropriate log level based on print content."""
        if not print_node.args:
            return "skip"  # Empty print statements

        first_arg = print_node.args[0]

        # Try to get string content
        content = self._get_string_content(first_arg)
        if content:
            content_lower = content.lower()

            # Skip certain patterns
            if any(
                pattern in content_lower
                for pattern in ["===", "---", "***", "test", "example", "demo"]
            ):
                return "skip" if "test" in self.module_name else "info"

            # Determine level based on keywords
            if any(word in content_lower for word in ["error", "fail", "exception", "critical"]):
                return "error"
            elif any(word in content_lower for word in ["warn", "caution", "alert"]):
                return "warning"
            elif any(
                word in content_lower for word in ["debug", "trace", "verbose", "detail", "print("]
            ):
                return "debug"
            elif any(
                word in content_lower
                for word in ["success", "complete", "done", "✓", "✔", "finished"]
            ):
                return "info"
            elif any(word in content_lower for word in ["start", "begin", "init", "load"]):
                return "info"

        # Default based on context
        if "test" in self.module_name or "example" in self.module_name:
            return "debug"
        elif "devtools" in self.module_name:
            return "debug"
        else:
            return "info"

    def _get_string_content(self, node) -> str:
        """Extract string content from AST node."""
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Str):  # Python 3.7 compatibility
            return node.s
        elif isinstance(node, ast.JoinedStr):  # f-string
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
                else:
                    parts.append("{}")
            return "".join(parts)
        return ""

    def _create_logging_call(self, print_node, log_level: str):
        """Create a logging call to replace print statement."""
        # Map log levels to logger methods
        log_methods = {
            "debug": "debug",
            "info": "info",
            "warning": "warning",
            "error": "error",
            "critical": "critical",
        }

        method_name = log_methods.get(log_level, "info")

        # Create logger.method() call
        logger_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="logger", ctx=ast.Load()),
                attr=method_name,
                ctx=ast.Load(),
            ),
            args=print_node.args,
            keywords=[],
        )

        return logger_call


def process_file(file_path: Path, dry_run: bool = False, verbose: bool = False) -> Tuple[int, int]:
    """Process a single Python file to replace print statements."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # Parse the file
        try:
            tree = ast.parse(original_content, filename=str(file_path))
        except SyntaxError as e:
            script_logger.warning(f"Syntax error in {file_path}: {e}")
            return 0, 0

        # Transform the AST
        module_name = str(file_path).replace("/", ".")
        transformer = PrintStatementTransformer(module_name)
        new_tree = transformer.visit(tree)

        if transformer.print_count == 0:
            return 0, 0

        # Add import if needed
        if transformer.needs_import and not transformer.has_logger:
            # Add import at the beginning after docstring and future imports
            import_node = ast.ImportFrom(
                module="genomevault.utils.logging",
                names=[ast.alias(name="get_logger", asname=None)],
                level=0,
            )

            # Add logger initialization
            logger_init = ast.Assign(
                targets=[ast.Name(id="logger", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="get_logger", ctx=ast.Load()),
                    args=[ast.Name(id="__name__", ctx=ast.Load())],
                    keywords=[],
                ),
            )

            # Find position to insert (after imports and docstrings)
            insert_pos = 0
            for i, node in enumerate(new_tree.body):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    insert_pos = i + 1
                elif isinstance(node, ast.Expr) and isinstance(node.value, (ast.Constant, ast.Str)):
                    # Docstring
                    insert_pos = i + 1
                else:
                    break

            # Insert import and logger initialization
            new_tree.body.insert(insert_pos, import_node)
            new_tree.body.insert(insert_pos + 1, logger_init)

        # Generate new code
        ast.fix_missing_locations(new_tree)
        new_content = ast.unparse(new_tree)

        # Clean up output
        new_content = clean_generated_code(new_content, original_content)

        if verbose:
            script_logger.info(
                f"Processing {file_path}: {transformer.print_count} print statements found"
            )

        if not dry_run and new_content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

        return transformer.print_count, transformer.print_count

    except Exception as e:
        script_logger.error(f"Error processing {file_path}: {e}")
        return 0, 0


def clean_generated_code(new_content: str, original_content: str) -> str:
    """Clean up generated code to maintain formatting consistency."""
    # Preserve original file encoding declaration if present
    encoding_match = re.match(r"^#.*?coding[:=].*$", original_content, re.MULTILINE)
    if encoding_match:
        new_content = encoding_match.group() + "\n" + new_content

    # Ensure proper newline at end of file
    if not new_content.endswith("\n"):
        new_content += "\n"

    return new_content


def process_directory(
    directory: Path, dry_run: bool = False, verbose: bool = False
) -> Tuple[int, int]:
    """Process all Python files in a directory."""
    total_files = 0
    total_prints = 0

    for py_file in directory.rglob("*.py"):
        # Skip __pycache__ directories
        if "__pycache__" in str(py_file):
            continue

        prints_found, prints_replaced = process_file(py_file, dry_run, verbose)
        if prints_found > 0:
            total_files += 1
            total_prints += prints_found

    return total_files, total_prints


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Remove debug print statements and replace with logging"
    )
    parser.add_argument(
        "directories",
        nargs="*",
        default=["devtools", "examples", "tests"],
        help="Directories to process (default: devtools, examples, tests)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    script_logger.info("Debug Print Removal Script")
    script_logger.info("=" * 50)

    if args.dry_run:
        script_logger.info("DRY RUN MODE - No changes will be made")

    total_files = 0
    total_prints = 0

    for dir_name in args.directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            script_logger.warning(f"Directory not found: {dir_name}")
            continue

        script_logger.info(f"\nProcessing {dir_name}/...")
        files_processed, prints_found = process_directory(dir_path, args.dry_run, args.verbose)

        total_files += files_processed
        total_prints += prints_found

        script_logger.info(f"  Found {prints_found} print statements in {files_processed} files")

    script_logger.info("\n" + "=" * 50)
    script_logger.info(f"Total: {total_prints} print statements in {total_files} files")

    if args.dry_run:
        script_logger.info("\nRun without --dry-run to apply changes")
    else:
        script_logger.info("\nChanges applied successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
