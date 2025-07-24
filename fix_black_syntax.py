#!/usr/bin/env python3
"""
Fix specific Black syntax errors in GenomeVault
"""
import re

# List of files with their specific fixes
fixes = {
    "genomevault/blockchain/hipaa/verifier.py": {
        "line": 17,
        "pattern": r"from .models import.*",
        "replacement": "from .models import HIPAAVerificationRecord"
    },
    "genomevault/hypervector_transform/encoding.py": {
        "line": 125,
        "pattern": r'logger\.error\("Encoding error: \{str\(e\)\}"\)',
        "replacement": 'logger.error(f"Encoding error: {str(e)}")'
    },
    "genomevault/hypervector_transform/hdc_encoder.py": {
        "line": 188,
        "pattern": r'logger\.error\(f"Encoding error: \{str\(e\)\}"\)',
        "replacement": 'logger.error(f"Encoding error: {str(e)}")'
    },
    "genomevault/hypervector_transform/registry.py": {
        "line": 114,
        "pattern": r'logger\.error\("Failed to load registry: \{e\}"\)',
        "replacement": 'logger.error(f"Failed to load registry: {e}")'
    },
    "genomevault/integration/proof_of_training.py": {
        "line": 288,
        "pattern": r'logger\.error\("Failed to submit attestation: \{e\}"\)',
        "replacement": 'logger.error(f"Failed to submit attestation: {e}")'
    },
    "genomevault/local_processing/pipeline.py": {
        "line": 70,
        "pattern": r'logger\.error\("Failed to process \{omics_type\}: \{str\(e\)\}"\)',
        "replacement": 'logger.error(f"Failed to process {omics_type}: {str(e)}")'
    },
    "genomevault/pir/client.py": {
        "line": 311,
        "pattern": r'logger\.error\("Error querying server \{server\.server_id\}: \{e\}"\)',
        "replacement": 'logger.error(f"Error querying server {server.server_id}: {e}")'
    },
    "genomevault/pir/network/coordinator.py": {
        "line": 258,
        "pattern": r'logger\.error\("Health monitor error: \{e\}"\)',
        "replacement": 'logger.error(f"Health monitor error: {e}")'
    },
    "genomevault/pir/server/handler.py": {
        "line": 121,
        "pattern": r'logger\.error\("Error handling PIR query: \{str\(e\)\}"\)',
        "replacement": 'logger.error(f"Error handling PIR query: {str(e)}")'
    },
    "genomevault/pir/server/shard_manager.py": {
        "line": 267,
        "pattern": r'logger\.error\("Error creating shard \{shard_index\}: \{e\}"\)',
        "replacement": 'logger.error(f"Error creating shard {shard_index}: {e}")'
    },
    "genomevault/security/phi_detector.py": {
        "line": 127,
        "pattern": r'logger\.error\("Error scanning file \{filepath\}: \{e\}"\)',
        "replacement": 'logger.error(f"Error scanning file {filepath}: {e}")'
    },
    "genomevault/pir/server/enhanced_pir_server.py": {
        "line": 484,
        "pattern": r'logger\.error\("Error processing query \{query_id\}: \{e\}"\)',
        "replacement": 'logger.error(f"Error processing query {query_id}: {e}")'
    }
}

# Files with import issues
import_fixes = {
    "genomevault/zk_proofs/circuit_manager.py": {
        "line": 26,
        "issue": "trailing comma after equals"
    },
    "genomevault/zk_proofs/circuits/biological/multi_omics.py": {
        "line": 21,
        "issue": "trailing comma in from statement"
    },
    "genomevault/zk_proofs/circuits/biological/variant.py": {
        "line": 23,
        "issue": "trailing comma in from statement"
    },
    "genomevault/zk_proofs/circuits/implementations/variant_proof_circuit.py": {
        "line": 18,
        "issue": "trailing comma in from statement"
    },
    "genomevault/zk_proofs/examples/integration_demo.py": {
        "line": 16,
        "issue": "trailing comma after equals"
    }
}

def fix_file(filepath, line_num, pattern, replacement):
    """Fix a specific pattern in a file"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Fix the specific line
        if line_num <= len(lines):
            lines[line_num - 1] = re.sub(pattern, replacement, lines[line_num - 1])
        
        with open(filepath, 'w') as f:
            f.writelines(lines)
        
        return True
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def fix_import_issues(filepath, line_num, issue):
    """Fix import-related issues"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if line_num <= len(lines):
            line = lines[line_num - 1]
            
            if "trailing comma in from statement" in issue:
                # Remove trailing comma from from statement
                lines[line_num - 1] = re.sub(r'from\s+(.+),\s*$', r'from \1\n', line)
            elif "trailing comma after equals" in issue:
                # Remove comma after equals
                lines[line_num - 1] = re.sub(r'=\s*,', '=', line)
        
        with open(filepath, 'w') as f:
            f.writelines(lines)
        
        return True
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    print("Fixing Black syntax errors...")
    
    # Fix f-string issues
    for filepath, fix_info in fixes.items():
        if fix_file(filepath, fix_info["line"], fix_info["pattern"], fix_info["replacement"]):
            print(f"✓ Fixed f-string in {filepath}")
    
    # Fix import issues
    for filepath, fix_info in import_fixes.items():
        if fix_import_issues(filepath, fix_info["line"], fix_info["issue"]):
            print(f"✓ Fixed import in {filepath}")
    
    print("\nDone! Run 'black --check .' again to verify.")

if __name__ == "__main__":
    main()
