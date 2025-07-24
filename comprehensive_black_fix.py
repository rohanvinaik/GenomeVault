#!/usr/bin/env python3
"""
Comprehensive Black formatter fix for GenomeVault
This script will fix all the files mentioned in the CI error
"""

import subprocess
import sys
from pathlib import Path


def run_black_on_files():
    """Run black on the specific files that failed CI"""
    files_to_fix = [
        "examples/diabetes_risk_demo.py",
        "devtools/pre_push_checklist.py",
        "genomevault/advanced_analysis/federated_learning/client.py",
        "genomevault/clinical/diabetes_pilot/risk_calculator.py",
        "genomevault/hypervector_transform/mapping.py",
        "genomevault/zk_proofs/circuits/__init__.py",
        "genomevault/zk_proofs/circuits/clinical_circuits.py",
        "genomevault/zk_proofs/circuits/biological/multi_omics.py",
        "genomevault/zk_proofs/circuits/implementations/constraint_system.py",
        "genomevault/zk_proofs/circuits/implementations/variant_proof_circuit.py",
        "genomevault/zk_proofs/circuits/implementations/plonk_circuits.py",
        "genomevault_zk_integration.py",
        "genomevault/zk_proofs/prover.py",
        "zk_api_integration.py",
    ]

    genomevault_dir = Path("/Users/rohanvinaik/genomevault")

    success_count = 0
    error_count = 0

    for file_path in files_to_fix:
        full_path = genomevault_dir / file_path

        if not full_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue

        try:
            result = subprocess.run(
                ["black", str(full_path)],
                capture_output=True,
                text=True,
                cwd=genomevault_dir,
            )

            if result.returncode == 0:
                print(f"✅ Fixed: {file_path}")
                success_count += 1
            else:
                print(f"❌ Failed to fix: {file_path}")
                print(f"   Error: {result.stderr}")
                error_count += 1

        except Exception as e:
            print(f"❌ Exception fixing {file_path}: {e}")
            error_count += 1

    print(f"\n📊 Summary:")
    print(f"   ✅ Successfully fixed: {success_count} files")
    print(f"   ❌ Failed to fix: {error_count} files")

    return error_count == 0


def run_black_check():
    """Run black --check to verify all formatting is correct"""
    genomevault_dir = Path("/Users/rohanvinaik/genomevault")

    try:
        result = subprocess.run(
            ["black", "--check", "."],
            capture_output=True,
            text=True,
            cwd=genomevault_dir,
        )

        if result.returncode == 0:
            print("✅ All files are properly formatted!")
            return True
        else:
            print("❌ Some files still need formatting:")
            print(result.stdout)
            return False

    except Exception as e:
        print(f"❌ Error running black --check: {e}")
        return False


def main():
    print("🔧 Fixing Black formatting issues in GenomeVault...")

    # First, try to fix the specific files
    if run_black_on_files():
        print("\n🔍 Verifying all files are properly formatted...")
        if run_black_check():
            print("\n🎉 All formatting issues fixed! CI should pass now.")
            return True

    # If individual file fixing didn't work, try formatting everything
    print("\n🔄 Trying to format entire codebase...")
    genomevault_dir = Path("/Users/rohanvinaik/genomevault")

    try:
        result = subprocess.run(["black", "."], capture_output=True, text=True, cwd=genomevault_dir)

        if result.returncode == 0:
            print("✅ Formatted entire codebase successfully!")
            if run_black_check():
                print("\n🎉 All formatting issues fixed! CI should pass now.")
                return True
        else:
            print(f"❌ Failed to format codebase: {result.stderr}")

    except Exception as e:
        print(f"❌ Error formatting codebase: {e}")

    print("\n❌ Could not fix all formatting issues. Manual intervention may be needed.")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
