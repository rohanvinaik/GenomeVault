#!/usr/bin/env python3
"""
GenomeVault Pre-Flight Check
Shows current state before applying fixes
"""

import re
from pathlib import Path


def check_current_state():
    base_path = Path("/Users/rohanvinaik/genomevault")

    print("GenomeVault Pre-Flight Check")
    print("=" * 50)
    print()

    # Check project files
    print("Project Files:")
    print("  ✓ README.md exists" if (base_path / "README.md").exists() else "  ✗ README.md missing")
    print(
        "  ✓ requirements.txt exists"
        if (base_path / "requirements.txt").exists()
        else "  ✗ requirements.txt missing"
    )
    print(
        "  ✓ pyproject.toml exists"
        if (base_path / "pyproject.toml").exists()
        else "  ✗ pyproject.toml missing"
    )
    print()

    # Check for missing __init__.py files
    print("Checking for missing __init__.py files...")
    missing_init_dirs = [
        "genomevault/blockchain/contracts",
        "genomevault/blockchain/node",
        "genomevault/cli",
        "genomevault/clinical",
        "genomevault/clinical/diabetes_pilot",
        "genomevault/federation",
        "genomevault/governance",
        "genomevault/kan",
        "genomevault/pir/reference_data",
        "genomevault/zk_proofs/examples",
        "scripts",
        "tests",
        "tests/adversarial",
        "tests/e2e",
        "tests/integration",
        "tests/property",
        "tests/unit",
        "tests/zk",
    ]

    missing_count = 0
    for dir_path in missing_init_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            init_file = full_path / "__init__.py"
            if not init_file.exists():
                print(f"  ✗ Missing: {dir_path}/__init__.py")
                missing_count += 1
            else:
                print(f"  ✓ Found: {dir_path}/__init__.py")

    print(f"\nTotal missing __init__.py files: {missing_count}")
    print()

    # Sample check for print statements
    print("Sampling print statement usage...")
    sample_files = [
        "examples/minimal_verification.py",
        "genomevault/blockchain/node.py",
        "genomevault/api/app.py",
    ]

    total_prints = 0
    for file_path in sample_files:
        full_path = base_path / file_path
        if full_path.exists():
            try:
                content = full_path.read_text()
                print_count = len(re.findall(r"^\s*print\(", content, re.MULTILINE))
                if print_count > 0:
                    print(f"  {file_path}: {print_count} print statements")
                    total_prints += print_count
            except Exception as e:
                logger.exception("Unhandled exception")
                print(f"  Error reading {file_path}: {e}")
                raise

    print(f"\nTotal print statements in sample: {total_prints}")
    print()

    # Sample check for broad exceptions
    print("Sampling broad exception usage...")
    total_broad_excepts = 0
    for file_path in sample_files:
        full_path = base_path / file_path
        if full_path.exists():
            try:
                content = full_path.read_text()
                except_count = len(re.findall(r"except\s+Exception", content))
                if except_count > 0:
                    print(f"  {file_path}: {except_count} broad exceptions")
                    total_broad_excepts += except_count
            except Exception as e:
                logger.exception("Unhandled exception")
                print(f"  Error reading {file_path}: {e}")
                raise

    print(f"\nTotal broad exceptions in sample: {total_broad_excepts}")
    print()

    print("Pre-flight check complete!")
    print("\nTo apply fixes, run: ./apply_audit_fixes.sh")
    print("To apply only __init__.py fixes, run: python3 quick_fix_init_files.py")


if __name__ == "__main__":
    check_current_state()
