#!/usr/bin/env python3
"""
GenomeVault Implementation Validator
Checks the current state of the codebase and identifies remaining issues
"""

import ast
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent
GENOMEVAULT_DIR = PROJECT_ROOT / "genomevault"


def check_syntax_errors() -> List[Dict]:
    """Check for Python syntax errors in all files."""
    errors = []

    for py_file in PROJECT_ROOT.rglob("*.py"):
        # Skip virtual environments and cache directories
        if any(part in str(py_file) for part in ["venv", "__pycache__", ".git", "node_modules"]):
            continue

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
            ast.parse(content)
        except SyntaxError as e:
            errors.append(
                {
                    "file": str(py_file.relative_to(PROJECT_ROOT)),
                    "line": e.lineno,
                    "error": str(e.msg),
                }
            )
        except Exception as e:
            errors.append(
                {
                    "file": str(py_file.relative_to(PROJECT_ROOT)),
                    "line": 0,
                    "error": str(e),
                }
            )

    return errors


def check_missing_init_files() -> List[str]:
    """Check for packages missing __init__.py files."""
    missing = []

    # Check genomevault subdirectories
    if GENOMEVAULT_DIR.exists():
        for dir_path in GENOMEVAULT_DIR.rglob("*"):
            if dir_path.is_dir() and not dir_path.name.startswith("__"):
                # Check if it contains Python files
                py_files = list(dir_path.glob("*.py"))
                if py_files and not (dir_path / "__init__.py").exists():
                    missing.append(str(dir_path.relative_to(PROJECT_ROOT)))

    return missing


def check_placeholders() -> List[Dict]:
    """Check for NotImplementedError placeholders."""
    placeholders = []

    for py_file in PROJECT_ROOT.rglob("*.py"):
        # Skip virtual environments and cache directories
        if any(part in str(py_file) for part in ["venv", "__pycache__", ".git"]):
            continue

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            if "NotImplementedError" in content:
                # Count occurrences
                count = content.count("NotImplementedError")
                placeholders.append(
                    {"file": str(py_file.relative_to(PROJECT_ROOT)), "count": count}
                )
        except Exception as e:
            logger.warning(f"Could not read {py_file}: {e}")

    return placeholders


def check_print_statements() -> List[Dict]:
    """Check for print statements in non-test code."""
    print_usage = []

    for py_file in PROJECT_ROOT.rglob("*.py"):
        # Skip virtual environments, cache, and test files
        if any(part in str(py_file) for part in ["venv", "__pycache__", ".git", "test"]):
            continue

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            print_lines = []
            for i, line in enumerate(lines, 1):
                # Simple check for print statements
                if "print(" in line and not line.strip().startswith("#"):
                    print_lines.append(i)

            if print_lines:
                print_usage.append(
                    {
                        "file": str(py_file.relative_to(PROJECT_ROOT)),
                        "lines": print_lines[:5],  # Show first 5 occurrences
                        "total": len(print_lines),
                    }
                )
        except Exception as e:
            logger.warning(f"Could not read {py_file}: {e}")

    return print_usage


def check_debug_code() -> List[Dict]:
    """Check for debug code (pdb, breakpoints)."""
    debug_usage = []

    for py_file in PROJECT_ROOT.rglob("*.py"):
        # Skip virtual environments and cache
        if any(part in str(py_file) for part in ["venv", "__pycache__", ".git"]):
            continue

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            debug_items = []
            if "pdb.set_trace()" in content:
                debug_items.append("pdb.set_trace()")
            if "breakpoint()" in content:
                debug_items.append("breakpoint()")
            if "import pdb" in content:
                debug_items.append("import pdb")

            if debug_items:
                debug_usage.append(
                    {
                        "file": str(py_file.relative_to(PROJECT_ROOT)),
                        "items": debug_items,
                    }
                )
        except Exception as e:
            logger.warning(f"Could not read {py_file}: {e}")

    return debug_usage


def generate_validation_report():
    """Generate a comprehensive validation report."""
    logger.info("Running GenomeVault Implementation Validation...")
    logger.info("=" * 60)

    # Run all checks
    syntax_errors = check_syntax_errors()
    missing_inits = check_missing_init_files()
    placeholders = check_placeholders()
    print_statements = check_print_statements()
    debug_code = check_debug_code()

    # Generate report
    report = []
    report.append("# GenomeVault Implementation Validation Report\n")
    report.append("## Summary\n")

    # Summary statistics
    total_issues = (
        len(syntax_errors)
        + len(missing_inits)
        + len(placeholders)
        + len(print_statements)
        + len(debug_code)
    )

    report.append(f"- **Total Issues Found**: {total_issues}\n")
    report.append(f"- **Syntax Errors**: {len(syntax_errors)}\n")
    report.append(f"- **Missing __init__.py**: {len(missing_inits)}\n")
    report.append(f"- **NotImplementedError Placeholders**: {len(placeholders)}\n")
    report.append(f"- **Print Statements**: {len(print_statements)}\n")
    report.append(f"- **Debug Code**: {len(debug_code)}\n")

    # Detailed sections
    if syntax_errors:
        report.append("\n## Syntax Errors (CRITICAL)\n")
        for error in syntax_errors:
            report.append(f"- **{error['file']}** (line {error['line']}): {error['error']}\n")
    else:
        report.append("\n## ✅ No Syntax Errors Found\n")

    if missing_inits:
        report.append("\n## Missing __init__.py Files\n")
        for path in missing_inits:
            report.append(f"- {path}\n")
    else:
        report.append("\n## ✅ All Packages Have __init__.py\n")

    if placeholders:
        report.append("\n## NotImplementedError Placeholders\n")
        for item in placeholders:
            report.append(f"- **{item['file']}**: {item['count']} occurrence(s)\n")
    else:
        report.append("\n## ✅ No NotImplementedError Placeholders\n")

    if print_statements:
        report.append("\n## Print Statements in Non-Test Code\n")
        for item in print_statements:
            lines_str = ", ".join(str(l) for l in item["lines"])
            report.append(f"- **{item['file']}**: {item['total']} total (lines: {lines_str}...)\n")
    else:
        report.append("\n## ✅ No Print Statements in Production Code\n")

    if debug_code:
        report.append("\n## Debug Code Found\n")
        for item in debug_code:
            items_str = ", ".join(item["items"])
            report.append(f"- **{item['file']}**: {items_str}\n")
    else:
        report.append("\n## ✅ No Debug Code Found\n")

    # Recommendations
    report.append("\n## Recommendations\n")

    if syntax_errors:
        report.append("1. **Fix syntax errors immediately** - These are blocking issues\n")
    if missing_inits:
        report.append(
            "2. **Add missing __init__.py files** - Required for proper package structure\n"
        )
    if placeholders:
        report.append("3. **Implement placeholder functions** - Replace with MVP implementations\n")
    if print_statements:
        report.append("4. **Replace print with logging** - Use proper logging framework\n")
    if debug_code:
        report.append("5. **Remove debug code** - Clean up before committing\n")

    if total_issues == 0:
        report.append("\n🎉 **All checks passed! The codebase is clean and ready.**\n")

    # Write report
    report_content = "".join(report)
    report_file = PROJECT_ROOT / "VALIDATION_REPORT.md"
    report_file.write_text(report_content)

    # Print to console
    logger.info(report_content)
    logger.info(f"\nReport saved to: {report_file}")

    return total_issues == 0


def main():
    """Main execution function."""
    all_valid = generate_validation_report()

    if all_valid:
        logger.info("\n✅ All validation checks passed!")
        return 0
    else:
        logger.warning("\n⚠️ Issues found. Please review the report above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
