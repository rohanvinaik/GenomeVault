#!/usr/bin/env python3
"""Analyze the codebase to identify missing docstrings."""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DocstringAnalyzer(ast.NodeVisitor):
    """Analyze Python files for missing docstrings."""

    def __init__(self, filepath: str):
        """Initialize the instance.
        Args:        filepath: Path to file or directory."""
        self.filepath = filepath
        self.missing_docstrings = []
        self.has_docstrings = []

    def visit_Module(self, node):
        """Visit module node."""
        if not ast.get_docstring(node):
            self.missing_docstrings.append(("Module", self.filepath, 1))
        else:
            self.has_docstrings.append(("Module", self.filepath, 1))
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Visit class definition."""
        if not ast.get_docstring(node):
            self.missing_docstrings.append(("Class", node.name, node.lineno))
        else:
            self.has_docstrings.append(("Class", node.name, node.lineno))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Visit function definition."""
        # Skip private functions starting with _
        if not node.name.startswith("_") or node.name in [
            "__init__",
            "__str__",
            "__repr__",
        ]:
            if not ast.get_docstring(node):
                self.missing_docstrings.append(("Function", node.name, node.lineno))
            else:
                self.has_docstrings.append(("Function", node.name, node.lineno))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition."""
        if not node.name.startswith("_"):
            if not ast.get_docstring(node):
                self.missing_docstrings.append(("AsyncFunction", node.name, node.lineno))
            else:
                self.has_docstrings.append(("AsyncFunction", node.name, node.lineno))
        self.generic_visit(node)


def analyze_file(filepath: Path) -> Tuple[List, List]:
    """Analyze a single Python file for docstrings."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))

        analyzer = DocstringAnalyzer(str(filepath))
        analyzer.visit(tree)
        return analyzer.missing_docstrings, analyzer.has_docstrings
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return [], []


def analyze_directory(directory: Path) -> Dict:
    """Analyze all Python files in a directory."""
    stats = {
        "total_missing": 0,
        "total_has": 0,
        "by_type": {
            "Module": {"missing": 0, "has": 0},
            "Class": {"missing": 0, "has": 0},
            "Function": {"missing": 0, "has": 0},
            "AsyncFunction": {"missing": 0, "has": 0},
        },
        "by_file": {},
        "priority_files": [],
    }

    python_files = list(directory.rglob("*.py"))

    for filepath in python_files:
        # Skip test files and __pycache__
        if "__pycache__" in str(filepath) or "test_" in filepath.name:
            continue

        missing, has = analyze_file(filepath)

        if missing:
            rel_path = filepath.relative_to(directory)
            stats["by_file"][str(rel_path)] = {
                "missing": len(missing),
                "details": missing,
            }

            # Identify priority files (public APIs)
            if any(part in str(rel_path) for part in ["__init__.py", "api.py", "core", "utils"]):
                stats["priority_files"].append(str(rel_path))

        # Update statistics
        for item_type, name, lineno in missing:
            stats["by_type"][item_type]["missing"] += 1
            stats["total_missing"] += 1

        for item_type, name, lineno in has:
            stats["by_type"][item_type]["has"] += 1
            stats["total_has"] += 1

    return stats


def main():
    """Main function to analyze the genomevault package."""
    root = Path(__file__).resolve().parents[1]
    genomevault_dir = root / "genomevault"

    print("=" * 60)
    print("Docstring Analysis for GenomeVault")
    print("=" * 60)

    stats = analyze_directory(genomevault_dir)

    print(f"\nTotal missing docstrings: {stats['total_missing']}")
    print(f"Total existing docstrings: {stats['total_has']}")
    print(f"Coverage: {stats['total_has']/(stats['total_has'] + stats['total_missing'])*100:.1f}%")

    print("Missing by type:")
    for item_type, counts in stats["by_type"].items():
        if counts["missing"] > 0:
            print(f"  {item_type}: {counts['missing']} missing, {counts['has']} present")

    print("Top 10 files with most missing docstrings:")
    sorted_files = sorted(stats["by_file"].items(), key=lambda x: x[1]["missing"], reverse=True)
    for filepath, info in sorted_files[:10]:
        print(f"  {filepath}: {info['missing']} missing")

    print("Priority files (public APIs) missing docstrings:")
    for filepath in stats["priority_files"][:10]:
        if filepath in stats["by_file"]:
            print(f"  {filepath}: {stats['by_file'][filepath]['missing']} missing")

    # Save detailed report
    report_path = root / "missing_docstrings_report.txt"
    with open(report_path, "w") as f:
        f.write("MISSING DOCSTRINGS DETAILED REPORT\n")
        f.write("=" * 60 + "\n\n")

        for filepath, info in sorted_files:
            f.write(f"\n{filepath} ({info['missing']} missing):\n")
            for item_type, name, lineno in info["details"]:
                f.write(f"  Line {lineno}: {item_type} '{name}'\n")

    print(f"\nDetailed report saved to: {report_path}")

    return stats["total_missing"]


if __name__ == "__main__":
    total = main()
    sys.exit(0 if total == 0 else 1)
