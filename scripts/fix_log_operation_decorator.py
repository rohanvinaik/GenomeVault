#!/usr/bin/env python3

"""Fix log_operation decorator usage across the codebase."""

import re
from pathlib import Path

def fix_log_operation_decorators(file_path):
    """Fix log_operation decorator usage in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Check if file uses the wrong pattern
    if '@performance_logger.log_operation' in content or '@logger.log_operation' in content:
        # First, ensure log_operation is imported
        if 'from genomevault.utils.logging import' in content:
            # Check if log_operation is already imported
            if 'log_operation' not in content:
                # Add log_operation to the import
                content = re.sub(
                    r'from genomevault\.utils\.logging import ([^)]+)',
                    lambda m: f"from genomevault.utils.logging import {m.group(1)}, log_operation"
                    if ', log_operation' not in m.group(1) else m.group(0),
                    content
                )
        
        # Replace the decorator usage
        content = re.sub(
            r'@(?:performance_logger|logger)\.log_operation\(["\']([^"\']+)["\']\)',
            r'@log_operation',
            content
        )
        
        # Also handle cases without arguments
        content = re.sub(
            r'@(?:performance_logger|logger)\.log_operation',
            r'@log_operation',
            content
        )
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
    
    return False

def main():
    """Fix all files with incorrect log_operation usage."""
    repo_root = Path('.')
    
    # Files known to have the issue
    files_to_fix = [
        'genomevault/zk_proofs/circuit_manager.py',
        'genomevault/zk_proofs/verifier.py',
        'genomevault/zk_proofs/prover.py',
        'genomevault/pir/server/pir_server.py',
        'genomevault/pir/server/shard_manager.py',
        'genomevault/advanced_analysis/federated_learning/client.py',
        'genomevault/advanced_analysis/federated_learning/coordinator.py',
    ]
    
    fixed_count = 0
    for file_path in files_to_fix:
        full_path = repo_root / file_path
        if full_path.exists():
            if fix_log_operation_decorators(full_path):
                print(f"✓ Fixed {file_path}")
                fixed_count += 1
            else:
                print(f"  No changes needed for {file_path}")
        else:
            print(f"✗ File not found: {file_path}")
    
    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()