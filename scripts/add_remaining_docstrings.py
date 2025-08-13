#!/usr/bin/env python3
"""Add remaining docstrings manually to specific files."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def add_docstring_to_init_files():
    """Add docstrings to __init__ files."""
    root = Path(__file__).resolve().parents[1]

    init_files = {
        "genomevault/blockchain/__init__.py": '"""Blockchain integration for decentralized genomic data governance."""',
        "genomevault/blockchain/contracts/__init__.py": '"""Smart contracts for genomic data management."""',
        "genomevault/etl/__init__.py": '"""ETL pipelines for genomic data processing."""',
        "genomevault/security/__init__.py": '"""Security modules for genomic data protection."""',
        "genomevault/clinical/__init__.py": '"""Clinical genomics analysis modules."""',
        "genomevault/clinical/calibration/__init__.py": '"""Calibration tools for clinical models."""',
        "genomevault/hypervector_transform/__init__.py": '"""Hypervector transformation for genomic data encoding."""',
        "genomevault/monitoring/__init__.py": '"""System monitoring and observability."""',
        "genomevault/reference_data/__init__.py": '"""Reference genomic data management."""',
        "genomevault/utils/__init__.py": '"""Utility functions and helpers."""',
    }

    added = 0
    for filepath, docstring in init_files.items():
        full_path = root / filepath
        if full_path.exists():
            with open(full_path, "r") as f:
                content = f.read()

            # Check if already has docstring
            if content.strip() and not content.strip().startswith('"""'):
                # Add docstring at the beginning
                with open(full_path, "w") as f:
                    f.write(docstring + "\n\n" + content)
                print(f"✓ Added docstring to {filepath}")
                added += 1
            else:
                print(f"- {filepath} already has docstring or is empty")

    return added


def add_docstrings_to_specific_functions():
    """Add docstrings to specific functions that need them."""

    docstrings_to_add = [
        # Format: (file_path, function_name, line_after_def, docstring)
        (
            "genomevault/security/rate_limit.py",
            "check_rate_limit",
            '"""Check if request is within rate limits."""',
        ),
        (
            "genomevault/clinical/calibration/calibrators.py",
            "calibrate_model",
            '"""Calibrate machine learning model predictions."""',
        ),
        (
            "genomevault/etl/loaders.py",
            "load_vcf",
            '"""Load VCF file into genomic data structure."""',
        ),
        (
            "genomevault/etl/transformers.py",
            "transform_variants",
            '"""Transform genetic variants to standard format."""',
        ),
    ]

    # This would need more sophisticated implementation
    # For now, just report what needs to be done
    for item in docstrings_to_add[:5]:
        if len(item) >= 3:
            print(f"  - {item[0]}: {item[1]}")

    return 0


def main():
    """Main function."""
    print("=" * 60)
    print("Adding Remaining Docstrings")
    print("=" * 60)
    pass  # Debug print removed

    # Add to __init__ files
    pass  # Debug print removed
    init_added = add_docstring_to_init_files()

    # Report on remaining functions
    func_added = add_docstrings_to_specific_functions()

    total = init_added + func_added
    print(f"\nTotal docstrings added: {total}")

    # Run final analysis
    print("Final Analysis")
    print("=" * 60)

    import subprocess

    result = subprocess.run(
        [sys.executable, "scripts/analyze_missing_docstrings.py"],
        capture_output=True,
        text=True,
    )

    # Parse key metrics
    for line in result.stdout.split("\n"):
        if any(
            key in line for key in ["Total missing", "Coverage:", "Module:", "Class:", "Function:"]
        ):
            print(line)


if __name__ == "__main__":
    main()
