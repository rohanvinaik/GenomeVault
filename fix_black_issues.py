#!/usr/bin/env python3
"""
Fix Black formatting issues in GenomeVault
"""
import os
import re
import sys


def fix_file_issues(filepath, issues):
    """Fix specific issues in a file"""
    try:
        with open(filepath, "r") as f:
            content = f.read()

        original = content

        # Fix based on the issue type
        for issue in issues:
            if "logger.error" in issue and 'f"' in issue:
                # Fix missing f-string prefix
                pattern = r'logger\.error\("([^"]*\{[^}]+\}[^"]*)"\)'
                content = re.sub(pattern, r'logger.error(f"\1")', content)

            elif "from .models import =" in issue:
                # Fix malformed import
                content = re.sub(
                    r"from \.models import =.*",
                    "from .models import HIPAAVerificationRecord",
                    content,
                )

            elif "import, logger" in issue:
                # Fix malformed logger import
                content = re.sub(
                    r"from genomevault\.utils\.logging import.*",
                    "from genomevault.utils.logging import get_logger",
                    content,
                )

            elif "from," in issue:
                # Remove trailing comma in from statement
                content = re.sub(
                    r"from typing import (.*?),\s*$",
                    r"from typing import \1",
                    content,
                    flags=re.MULTILINE,
                )

            elif "=," in issue:
                # Remove trailing comma after equals
                content = re.sub(r"=\s*,", "=", content)

        if content != original:
            with open(filepath, "w") as f:
                f.write(content)
            return True
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False
    return False


# Define files and their issues based on the Black output
file_issues = {
    "genomevault/blockchain/hipaa/verifier.py": [
        "from .models import =, **name**, from, genomevault.utils.logging, get_logger, import, logger"
    ],
    "genomevault/hypervector_transform/encoding.py": ['logger.error(f"Encoding error: {str(e)}")'],
    "genomevault/hypervector_transform/hdc_encoder.py": [
        'logger.error(f"Encoding error: {str(e)}")'
    ],
    "genomevault/hypervector_transform/registry.py": [
        'logger.error(f"Failed to load registry: {e}")'
    ],
    "genomevault/integration/proof_of_training.py": [
        'logger.error(f"Failed to submit attestation: {e}")'
    ],
    "genomevault/local_processing/pipeline.py": [
        'logger.error(f"Failed to process {omics_type}: {str(e)}")'
    ],
    "genomevault/pir/client.py": ['logger.error(f"Error querying server {server.server_id}: {e}")'],
    "genomevault/pir/network/coordinator.py": ['logger.error(f"Health monitor error: {e}")'],
    "genomevault/pir/server/handler.py": ['logger.error(f"Error handling PIR query: {str(e)}")'],
    "genomevault/pir/server/shard_manager.py": [
        'logger.error(f"Error creating shard {shard_index}: {e}")'
    ],
    "genomevault/security/phi_detector.py": [
        'logger.error(f"Error scanning file {filepath}: {e}")'
    ],
    "genomevault/pir/server/enhanced_pir_server.py": [
        'logger.error(f"Error processing query {query_id}: {e}")'
    ],
    "genomevault/utils/security_monitor.py": [
        'logger.error("anomaly_detector_training_failed", error=str(e))'
    ],
    "genomevault/zk_proofs/circuit_manager.py": ["=,"],
    "genomevault/zk_proofs/circuits/biological/multi_omics.py": ["from,"],
    "genomevault/zk_proofs/circuits/biological/variant.py": ["from,"],
    "genomevault/zk_proofs/circuits/implementations/variant_proof_circuit.py": ["from,"],
    "genomevault/zk_proofs/examples/integration_demo.py": ["=,"],
}


def main():
    """Main function to fix Black issues"""
    print("Fixing Black formatting issues in GenomeVault...")

    fixed_count = 0
    for filepath, issues in file_issues.items():
        if os.path.exists(filepath):
            if fix_file_issues(filepath, issues):
                fixed_count += 1
                print(f"✓ Fixed: {filepath}")
        else:
            print(f"⚠️  File not found: {filepath}")

    print(f"\nFixed {fixed_count} files")

    # Additional specific fixes for known patterns
    print("\nApplying additional fixes...")

    # Fix malformed imports in verifier.py
    verifier_path = "genomevault/blockchain/hipaa/verifier.py"
    if os.path.exists(verifier_path):
        with open(verifier_path, "r") as f:
            content = f.read()

        # Fix the specific malformed import line
        if "from .models import =" in content or "import, logger" in content:
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if "from .models import" in line and ("=" in line or "**" in line):
                    lines[i] = "from .models import HIPAAVerificationRecord"
                elif "from genomevault.utils.logging" in line and (
                    "=" in line or "import, logger" in line
                ):
                    lines[i] = "from genomevault.utils.logging import get_logger"
                    # Add logger initialization after imports
                    if i + 1 < len(lines) and not lines[i + 1].strip().startswith("logger"):
                        lines.insert(i + 1, "\nlogger = get_logger(__name__)")

            with open(verifier_path, "w") as f:
                f.write("\n".join(lines))
            print(f"✓ Fixed imports in {verifier_path}")

    # Fix trailing commas in type imports
    for root, dirs, files in os.walk("genomevault"):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r") as f:
                        content = f.read()

                    original = content

                    # Fix trailing commas in from statements
                    content = re.sub(
                        r"from typing import ([^,\n]+),\s*$",
                        r"from typing import \1",
                        content,
                        flags=re.MULTILINE,
                    )
                    content = re.sub(
                        r"from\s+([^,\n]+),\s*$", r"from \1", content, flags=re.MULTILINE
                    )

                    # Fix assignment with trailing comma
                    content = re.sub(r"(\w+)\s*=\s*,", r"\1 =", content)

                    # Fix logger.error without f-string prefix where needed
                    content = re.sub(
                        r'logger\.error\("([^"]*\{[^}]+\}[^"]*)"\)', r'logger.error(f"\1")', content
                    )
                    content = re.sub(
                        r"logger\.error\('([^']*\{[^}]+\}[^']*)'\)", r"logger.error(f'\1')", content
                    )

                    if content != original:
                        with open(filepath, "w") as f:
                            f.write(content)
                        print(f"✓ Fixed patterns in {filepath}")

                except Exception as e:
                    pass  # Skip files that can't be processed

    print("\nDone! Run 'black --check .' again to verify fixes.")


if __name__ == "__main__":
    main()
