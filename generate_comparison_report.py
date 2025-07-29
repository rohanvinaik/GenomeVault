#!/usr/bin/env python3
"""
GenomeVault Before/After Comparison Report Generator
"""

import json
from datetime import datetime
from pathlib import Path


def generate_comparison_report():
    """Generate a comparison between the original audit and current state"""

    # Original audit findings from the report
    original_audit = {
        "files_total": 345,
        "py_files": 231,
        "tests_detected": 44,
        "has_pyproject": False,
        "has_requirements": True,
        "has_readme": False,
        "func_ann_cov": 0.47538440770315515,
        "func_ret_cov": 0.4489625251078286,
        "func_doc_cov": 0.9412108387732913,
        "avg_complexity": 3.0236194457706134,
        "max_complexity": 20.0,
        "n_bare_excepts": 0,
        "n_broad_excepts": 118,
        "n_print_calls": 456,
        "n_todos": 10,
        "missing_init_dirs": 19,
    }

    # Try to load current validation report
    current_report_path = Path("/Users/rohanvinaik/genomevault/audit_validation_report.json")
    if current_report_path.exists():
        with open(current_report_path) as f:
            current_audit = json.load(f)
    else:
        print("No current validation report found. Run validate_audit_fixes.py first.")
        return

    # Generate comparison report
    print("═" * 60)
    print("GenomeVault Audit Comparison Report")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)
    print()

    print("Project Structure Comparison:")
    print("-" * 40)
    print(f"{'Metric':<25} {'Original':<15} {'Current':<15} {'Status'}")
    print("-" * 40)

    # Compare basic metrics
    metrics = [
        ("Total files", "files_total", "files_total"),
        ("Python files", "py_files", "py_files"),
        ("Test files", "tests_detected", "tests_detected"),
        ("Has pyproject.toml", "has_pyproject", "has_pyproject"),
        ("Has requirements.txt", "has_requirements", "has_requirements"),
        ("Has README.md", "has_readme", "has_readme"),
    ]

    for label, orig_key, curr_key in metrics:
        orig_val = original_audit.get(orig_key, "N/A")
        curr_val = current_audit.get(curr_key, "N/A")

        # Determine status
        if isinstance(orig_val, bool) and isinstance(curr_val, bool):
            if not orig_val and curr_val:
                status = "✓ Fixed"
            elif orig_val and curr_val:
                status = "✓ OK"
            else:
                status = "✗ Needs fix"
        else:
            status = "→"

        print(f"{label:<25} {str(orig_val):<15} {str(curr_val):<15} {status}")

    print()
    print("Code Quality Issues:")
    print("-" * 40)

    # Missing init files
    orig_missing = original_audit.get("missing_init_dirs", 19)
    curr_missing = len(current_audit.get("missing_init_dirs", []))
    fixed_inits = orig_missing - curr_missing
    print(f"Missing __init__.py files: {orig_missing} → {curr_missing} ({fixed_inits} fixed)")

    # Print statements
    orig_prints = original_audit.get("n_print_calls", 456)
    curr_prints = sum(item["count"] for item in current_audit.get("files_with_prints", []))
    fixed_prints = orig_prints - curr_prints
    print(f"Print statements: {orig_prints} → {curr_prints} ({fixed_prints} converted to logging)")

    # Broad exceptions
    orig_excepts = original_audit.get("n_broad_excepts", 118)
    curr_excepts = sum(item["count"] for item in current_audit.get("files_with_broad_excepts", []))
    fixed_excepts = orig_excepts - curr_excepts
    print(f"Broad exceptions: {orig_excepts} → {curr_excepts} ({fixed_excepts} made specific)")

    # Complex functions
    curr_complex = len(current_audit.get("complex_functions", []))
    print(
        f"Complex functions: {original_audit.get('max_complexity', 20)} max → {curr_complex} functions > 10"
    )

    print()
    print("Overall Progress:")
    print("-" * 40)

    # Calculate overall progress
    total_issues = orig_missing + orig_prints + orig_excepts
    total_fixed = fixed_inits + fixed_prints + fixed_excepts
    progress = (total_fixed / total_issues * 100) if total_issues > 0 else 0

    print(f"Total issues fixed: {total_fixed} / {total_issues} ({progress:.1f}%)")

    # Recommendations
    print()
    print("Recommendations:")
    print("-" * 40)

    if curr_missing > 0:
        print(f"• Add {curr_missing} remaining __init__.py files")
    if curr_prints > 0:
        print(f"• Convert {curr_prints} remaining print statements to logging")
    if curr_excepts > 0:
        print(f"• Fix {curr_excepts} remaining broad exception handlers")
    if curr_complex > 5:
        print(f"• Refactor {curr_complex} complex functions")

    if progress >= 80:
        print("\n✓ Great progress! Most issues have been addressed.")
    elif progress >= 50:
        print("\n→ Good start! Continue with remaining fixes.")
    else:
        print("\n→ Run the comprehensive fix script to address most issues automatically.")

    print()


if __name__ == "__main__":
    generate_comparison_report()
