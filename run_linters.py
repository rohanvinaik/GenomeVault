#!/usr/bin/env python3
"""
Run linters on GenomeVault codebase
"""
import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print results"""
    print(f"\n{'='*60}")
    print(f"Running {description}...")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            print(f"❌ {description} failed with exit code {result.returncode}")
            return False
        else:
            print(f"✅ {description} passed")
            return True
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def main():
    """Run all linters"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Files that need Black formatting based on your output
    files_to_format = [
        "examples/proof_of_training_demo.py",
        "genomevault/advanced_analysis/federated_learning/model_lineage.py",
        "genomevault/blockchain/contracts/training_attestation.py",
        "genomevault/hypervector/visualization/__init__.py",
        "genomevault/cli/training_proof_cli.py",
        "genomevault/clinical/model_validation.py",  # Fixed syntax error
        "genomevault/integration/__init__.py",
        "genomevault/hypervector/visualization/projector.py",
        "genomevault/integration/proof_of_training.py",
        "genomevault/local_processing/differential_privacy_audit.py",
        "genomevault/local_processing/drift_detection.py",
        "genomevault/local_processing/model_snapshot.py",
        "genomevault/zk_proofs/circuits/training_proof.py",
        "genomevault/zk_proofs/circuits/multi_modal_training_proof.py",
        "genomevault/zk_proofs/circuits/test_training_proof.py",
        "tests/integration/test_proof_of_training.py",
    ]

    all_passed = True

    # Run Black
    print("\n" + "=" * 60)
    print("RUNNING BLACK")
    print("=" * 60)

    # First run Black in check mode on all files
    if not run_command("black --check .", "Black (check mode)"):
        print("\nFormatting files with Black...")
        for file in files_to_format:
            if os.path.exists(file):
                run_command(f"black {file}", f"Black format {file}")

        # Check again after formatting
        run_command("black --check .", "Black (verification after formatting)")

    # Run isort
    print("\n" + "=" * 60)
    print("RUNNING ISORT")
    print("=" * 60)

    if not run_command("isort --check-only --diff .", "isort (check mode)"):
        print("\nFixing import order with isort...")
        run_command("isort .", "isort (fix imports)")
        all_passed = False

    # Run Flake8
    print("\n" + "=" * 60)
    print("RUNNING FLAKE8")
    print("=" * 60)

    if not run_command("flake8 genomevault/ tests/ examples/", "Flake8"):
        all_passed = False

    # Run Pylint
    print("\n" + "=" * 60)
    print("RUNNING PYLINT")
    print("=" * 60)

    # Run pylint on the genomevault package
    if not run_command("pylint genomevault/", "Pylint"):
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("LINTING SUMMARY")
    print("=" * 60)

    if all_passed:
        print("✅ All linters passed!")
        return 0
    else:
        print("❌ Some linters failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
