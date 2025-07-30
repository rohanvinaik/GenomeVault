#!/usr/bin/env python3
"""
Convert print statements to proper logging in the genomevault package.
This script helps with step 5 of the audit checklist.
"""

import ast
from pathlib import Path


def find_print_statements(file_path: Path) -> list[tuple[int, str]]:
    """Find all print statements in a Python file."""
    prints = []

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "print":
                    line_num = node.lineno
                    # Get the actual line content
                    lines = content.split("\n")
                    if 0 <= line_num - 1 < len(lines):
                        prints.append((line_num, lines[line_num - 1].strip()))

    except (SyntaxError, UnicodeDecodeError):
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        # Skip files with syntax errors or encoding issues
        pass
        raise

    return prints


def scan_directory(root_dir: Path, exclude_dirs: list[str] = None) -> dict:
    """Scan directory for Python files with print statements."""
    if exclude_dirs is None:
        exclude_dirs = [
            "tests",
            "scripts",
            "benchmarks",
            ".git",
            "__pycache__",
            "venv",
            ".venv",
        ]

    results = {}

    for file_path in root_dir.rglob("*.py"):
        # Skip excluded directories
        if any(excluded in file_path.parts for excluded in exclude_dirs):
            continue

        prints = find_print_statements(file_path)
        if prints:
            results[str(file_path)] = prints

    return results


def generate_conversion_guide(results: dict) -> str:
    """Generate a guide for converting print statements."""
    guide = []
    guide.append("# Print Statement to Logging Conversion Guide\n")
    guide.append("## Files with print statements that need conversion:\n")

    total_prints = sum(len(prints) for prints in results.values())
    guide.append(f"Total print statements found: {total_prints}\n")

    for file_path, prints in sorted(results.items()):
        guide.append(f"\n### {file_path}")
        guide.append(f"Lines with print statements: {len(prints)}")
        for line_num, line_content in prints:
            guide.append(f"  - Line {line_num}: `{line_content}`")

    guide.append("\n## Conversion recommendations:")
    guide.append("1. Add this import at the top of each file:")
    guide.append("   ```python")
    guide.append("   from genomevault.logging_utils import get_logger")
    guide.append("   logger = get_logger(__name__)")
    guide.append("   ```")
    guide.append("2. Replace print statements based on context:")
    guide.append("   - Status messages: `print(msg)` → `logger.info(msg)`")
    guide.append("   - Debug info: `print(msg)` → `logger.debug(msg)`")
    guide.append("   - Errors: `print(msg)` → `logger.error(msg)`")
    guide.append("   - Warnings: `print(msg)` → `logger.warning(msg)`")

    return "\n".join(guide)


def main():
    """Main function to scan for print statements."""
    genomevault_dir = Path("/Users/rohanvinaik/genomevault/genomevault")

    print("Scanning for print statements in genomevault package...")
    results = scan_directory(genomevault_dir)

    if results:
        guide = generate_conversion_guide(results)
        output_file = Path("/Users/rohanvinaik/genomevault/print_to_logging_guide.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(guide)
        print(f"Conversion guide written to: {output_file}")
        print(f"Total files with print statements: {len(results)}")
        print(f"Total print statements found: {sum(len(p) for p in results.values())}")
    else:
        print("No print statements found in library code!")


if __name__ == "__main__":
    main()
