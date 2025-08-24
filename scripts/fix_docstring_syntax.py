#!/usr/bin/env python3
"""Fix docstring syntax errors in Python files."""

import ast
import re
from pathlib import Path


def fix_unclosed_docstrings(content: str) -> str:
    """Fix unclosed docstrings in Python code.
    
    Args:
        content: The file content to fix.
        
    Returns:
        Fixed content.
    """
    lines = content.split('\n')
    fixed_lines = []
    in_docstring = False
    docstring_quote = None
    docstring_start_line = -1
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check for docstring start
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_quote = '"""' if stripped.startswith('"""') else "'''"
                # Check if it's a one-liner
                if stripped.count(docstring_quote) >= 2:
                    # One-line docstring
                    fixed_lines.append(line)
                else:
                    # Multi-line docstring starts
                    in_docstring = True
                    docstring_start_line = i
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        else:
            # We're inside a docstring
            if docstring_quote in line:
                # Docstring ends
                in_docstring = False
                fixed_lines.append(line)
            else:
                # Still inside docstring
                fixed_lines.append(line)
                
                # Check if next line is not part of docstring  
                # (e.g. starts with 'from', 'import', 'class', 'def', etc.)
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if (next_line.startswith('from ') or 
                        next_line.startswith('import ') or
                        next_line.startswith('class ') or
                        next_line.startswith('def ') or
                        next_line.startswith('__') or
                        next_line.startswith('#') or
                        (next_line and not next_line[0].isalpha() and next_line[0] not in '"\'`')):
                        # Looks like docstring should have ended, add closing quotes
                        fixed_lines.append(docstring_quote)
                        in_docstring = False
        
        i += 1
    
    # If we're still in a docstring at the end, close it
    if in_docstring:
        fixed_lines.append(docstring_quote)
    
    return '\n'.join(fixed_lines)


def check_and_fix_file(file_path: Path) -> bool:
    """Check and fix a Python file for docstring syntax errors.
    
    Args:
        file_path: Path to the file to check and fix.
        
    Returns:
        True if file was modified, False otherwise.
    """
    try:
        content = file_path.read_text()
        
        # Try to parse the file
        try:
            ast.parse(content)
            return False  # File is already valid
        except SyntaxError:
            pass  # File has syntax errors, try to fix
        
        # Fix unclosed docstrings
        fixed_content = fix_unclosed_docstrings(content)
        
        # Verify the fix
        try:
            ast.parse(fixed_content)
            # Fix worked, save the file
            if fixed_content != content:
                file_path.write_text(fixed_content)
                return True
        except SyntaxError as e:
            print(f"Could not fix {file_path}: {e}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False
    
    return False


def main():
    """Main function to fix docstring syntax errors."""
    root = Path(".")
    
    # List of files with syntax errors
    error_files = [
        "genomevault/experimental/pir_advanced/__init__.py",
        "genomevault/experimental/kan/hybrid.py",
        "genomevault/experimental/kan/__init__.py",
        "genomevault/experimental/zk_circuits/stark_prover.py",
        "genomevault/experimental/zk_circuits/__init__.py",
        "genomevault/pir/server/handler.py",
        "genomevault/pir/server/enhanced_pir_server.py",
        "genomevault/pir/server/shard_manager.py",
        "genomevault/pir/server/pir_server.py",
        "genomevault/pir/client/pir_client.py",
        "genomevault/pir/client/batched_query_builder.py",
        "genomevault/zk/proof.py",
        "genomevault/zk/circuits/median_verifier.py",
        "genomevault/advanced_analysis/__init__.py",
        "genomevault/advanced_analysis/tda/persistence.py",
        "genomevault/advanced_analysis/federated_learning/coordinator.py",
        "genomevault/advanced_analysis/federated_learning/model_lineage.py",
        "genomevault/advanced_analysis/federated_learning/client.py",
        "genomevault/security/phi_detector.py",
        "genomevault/integration/__init__.py",
        "genomevault/integration/proof_of_training.py",
        "genomevault/tests/test_hdc_quality.py",
        "genomevault/zk_proofs/__init__.py",
        "genomevault/zk_proofs/prover.py",
        "genomevault/zk_proofs/advanced/stark_prover.py",
        "genomevault/zk_proofs/circuits/clinical_circuits.py",
        "genomevault/zk_proofs/backends/gnark_backend.py",
        "genomevault/zk_proofs/cli/zk_cli.py",
        "genomevault/zk_proofs/examples/integration_demo.py",
        "genomevault/local_processing/__init__.py",
        "genomevault/local_processing/pipeline.py",
        "genomevault/local_processing/differential_privacy_audit.py",
        "genomevault/utils/backup.py",
        "genomevault/utils/encryption.py",
        "genomevault/utils/config.py",
        "genomevault/utils/monitoring.py",
        "genomevault/utils/common.py",
        "genomevault/utils/security_monitor.py",
        "genomevault/utils/dependencies.py",
        "genomevault/hypervector_transform/holographic.py",
        "genomevault/hypervector_transform/registry.py",
        "genomevault/hypervector_transform/hierarchical.py",
        "genomevault/hypervector_transform/hdc_api.py",
        "genomevault/hypervector_transform/advanced_compression.py",
        "genomevault/hypervector_transform/binding.py",
        "genomevault/hypervector_transform/mapping.py",
        "genomevault/blockchain/__init__.py",
        "genomevault/blockchain/hipaa/__init__.py",
        "genomevault/blockchain/hipaa/models.py",
        "genomevault/blockchain/hipaa/integration.py",
        "genomevault/blockchain/contracts/training_attestation.py",
        "genomevault/hypervector/encoder.py",
        "genomevault/hypervector/positional.py",
        "genomevault/hypervector/visualization/__init__.py",
        "genomevault/hypervector/visualization/projector.py",
        "genomevault/hypervector/encoding/__init__.py",
        "genomevault/hypervector/encoding/unified_encoder.py",
        "genomevault/hypervector/encoding/genomic.py",
        "genomevault/hypervector/operations/__init__.py",
        "genomevault/kan/__init__.py",
        "genomevault/api/__init__.py",
        "genomevault/api/example_usage.py",
        "genomevault/api/routers/metrics.py",
        "genomevault/api/routers/tuned_query.py",
        "genomevault/api/routers/topology.py",
        "genomevault/api/routers/healthz.py",
        "genomevault/api/routers/query_tuned.py",
        "genomevault/api/routers/credit.py",
        "genomevault/clinical/model_validation.py",
        "genomevault/clinical/diabetes_pilot/risk_calculator.py",
        "genomevault/nanopore/biological_signals.py",
        "genomevault/nanopore/__init__.py",
        "genomevault/nanopore/api.py",
        "genomevault/nanopore/gpu_kernels.py",
        "genomevault/nanopore/streaming.py",
    ]
    
    fixed_count = 0
    for file_path_str in error_files:
        file_path = root / file_path_str
        if file_path.exists():
            if check_and_fix_file(file_path):
                print(f"Fixed: {file_path_str}")
                fixed_count += 1
        else:
            print(f"File not found: {file_path_str}")
    
    # Also scan for any other files with issues
    for py_file in root.rglob("*.py"):
        if str(py_file.relative_to(root)) not in error_files:
            if check_and_fix_file(py_file):
                print(f"Fixed: {py_file.relative_to(root)}")
                fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()