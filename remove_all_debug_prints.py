#!/usr/bin/env python3
"""Comprehensive removal of all debug print statements."""

import re
import os
import ast
from pathlib import Path


def is_essential_print(node, source_lines):
    """Check if a print call is essential (not debug)."""
    if not isinstance(node, ast.Call):
        return True

    # Get the actual print content if possible
    try:
        line_content = source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""

        # Essential patterns - keep these
        essential_patterns = [
            r"Usage:",
            r"Results?:",
            r"Output:",
            r"Help:",
            r"version",
            r"Version",
            r"--help",
            r"json\.dumps",
            r"\.to_json",
            r"\.format\(",
            r"__main__.*Results",  # Main block results
            r"DEMO COMPLETE",
            r"SUCCESS",
            r"COMPLETE",
            r"Test passed",
            r"All tests passed",
        ]

        for pattern in essential_patterns:
            if re.search(pattern, line_content, re.IGNORECASE):
                return True

        # Debug patterns - remove these
        debug_patterns = [
            r"debug",
            r"DEBUG",
            r"trace",
            r"TRACE",
            r"TODO",
            r"FIXME",
            r"XXX",
            r"HACK",
            r">>>>",
            r"----",
            r"====",
            r"\.\.\.\.",
            r"Processing",
            r"Checking",
            r"Starting",
            r"Found",
            r"Step \d+",
            r"Test \d+",
            r"Running test",
            r"^\s*print\s*\(\s*\)",  # Empty prints
            r"^\s*print\s*\([\'\"]\s*[\'\"]\)",  # Prints with just whitespace
            r"^\s*print\s*\([\'\"]\n[\'\"]\)",  # Just newlines
            r'print\s*\(\s*f[\'"].*\{.*\}.*[\'\"]\s*\)',  # f-strings (often debug)
        ]

        for pattern in debug_patterns:
            if re.search(pattern, line_content, re.IGNORECASE):
                return False

    except:
        pass

    return False  # When in doubt, remove it


class PrintRemover(ast.NodeTransformer):
    """AST transformer to remove debug print statements."""

    def __init__(self, source_lines):
        self.source_lines = source_lines
        self.removed_count = 0

    def visit_Expr(self, node):
        """Visit expression nodes and remove debug prints."""
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id == "print":
                if not is_essential_print(node.value, self.source_lines):
                    self.removed_count += 1
                    # Replace with pass to maintain structure
                    return ast.Pass()
        return self.generic_visit(node)


def remove_prints_from_file(filepath):
    """Remove debug prints from a single file using AST."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
            source_lines = source.splitlines()

        # Parse the AST
        tree = ast.parse(source)

        # Transform the AST
        remover = PrintRemover(source_lines)
        new_tree = remover.visit(tree)

        if remover.removed_count > 0:
            # Convert back to source code
            import astor

            new_source = astor.to_source(new_tree)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_source)

            return remover.removed_count
    except:
        # Fallback to regex-based removal
        return remove_prints_regex(filepath)

    return 0


def remove_prints_regex(filepath):
    """Fallback regex-based print removal."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        original = content

        # Remove obvious debug prints
        patterns = [
            (r"^\s*print\s*\([^)]*[Dd]ebug[^)]*\).*$", ""),
            (r"^\s*print\s*\([^)]*TODO[^)]*\).*$", ""),
            (r"^\s*print\s*\([^)]*FIXME[^)]*\).*$", ""),
            (r"^\s*print\s*\([^)]*XXX[^)]*\).*$", ""),
            (r"^\s*print\s*\(\s*[\'\"](>>>|---|===|\.\.\.).*?[\'\"]\s*\).*$", ""),
            (
                r"^\s*print\s*\(\s*[\'\"](Processing|Checking|Starting|Found).*?[\'\"]\s*\).*$",
                "",
            ),
            (r"^\s*print\s*\(\s*\).*$", ""),  # Empty prints
            (r"^\s*print\s*\(\s*[\'\"]\s*[\'\"]\s*\).*$", ""),  # Whitespace only
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        # Count removals
        removed = content.count("\n") - original.count("\n")

        if content != original:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return abs(removed)
    except:
        pass

    return 0


def process_directory(directory):
    """Process all Python files in a directory."""
    total_removed = 0
    processed_files = []

    for root, dirs, files in os.walk(directory):
        # Skip virtual environments and cache
        dirs[:] = [d for d in dirs if d not in {".venv", "venv", "__pycache__", ".git"}]

        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                removed = remove_prints_regex(filepath)  # Use regex method for speed
                if removed > 0:
                    total_removed += removed
                    processed_files.append((filepath, removed))

    return total_removed, processed_files


def main():
    """Main function."""
    print("=" * 70)
    pass  # Debug print removed
    print("=" * 70)

    # Priority directories
    directories = [
        "devtools",
        "examples",
        "tests",
        "genomevault",
        "tailchasing",
    ]

    total_removed = 0
    all_processed = []

    for directory in directories:
        if os.path.exists(directory):
            pass  # Debug print removed
            removed, processed = process_directory(directory)
            total_removed += removed
            all_processed.extend(processed)

    # Process root Python files
    pass  # Debug print removed
    for file in Path(".").glob("*.py"):
        if file.is_file():
            removed = remove_prints_regex(str(file))
            if removed > 0:
                total_removed += removed
                all_processed.append((str(file), removed))

    # Show summary
    print("\n" + "=" * 70)

    if all_processed:
        print("\nFiles modified:")
        for filepath, count in sorted(all_processed, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {filepath}: {count} prints removed")

    print("=" * 70)


if __name__ == "__main__":
    main()
