#!/usr/bin/env python3
"""
Format the files that need Black formatting based on CI output
"""
import os
import subprocess
import sys

# Files that need formatting based on CI output
FILES_TO_FORMAT = [
    "examples/proof_of_training_demo.py",
    "genomevault/advanced_analysis/federated_learning/model_lineage.py",
    "genomevault/blockchain/contracts/training_attestation.py",
    "genomevault/hypervector/visualization/__init__.py",
    "genomevault/cli/training_proof_cli.py",
    "genomevault/clinical/model_validation.py",
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


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Formatting files with Black...")
    for file in FILES_TO_FORMAT:
        if os.path.exists(file):
            print(f"  Formatting {file}...")
            result = subprocess.run(f"black {file}", shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"    ❌ Error: {result.stderr}")
            else:
                print(f"    ✅ Done")
        else:
            print(f"  ⚠️  File not found: {file}")

    print("\nRunning Black check on entire codebase...")
    result = subprocess.run("black --check .", shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ All files are now properly formatted!")
        return 0
    else:
        print("❌ Some files still need formatting:")
        print(result.stdout)
        return 1


if __name__ == "__main__":
    sys.exit(main())
