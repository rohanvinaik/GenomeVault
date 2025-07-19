#!/usr/bin/env python3
"""Fix the remaining import issues in GenomeVault"""

import os
import re
import sys

def fix_imports():
    """Fix all remaining import issues"""
    
    print("🔧 Fixing remaining import issues...")
    print("=" * 50)
    
    fixes = [
        # Fix utils.hashing imports to use utils.encryption
        {
            "files": ["test_all_imports.py", "simple_test.py"],
            "old": "from utils.hashing import secure_hash",
            "new": "from utils.encryption import secure_hash"
        },
        # Fix utils.config imports
        {
            "pattern": r"from utils\.config import config\b",
            "replacement": "from utils.config import get_config\nconfig = get_config()",
            "files": ["pir/*.py", "zk_proofs/*.py", "advanced_analysis/**/*.py"]
        },
        # Fix NodeClass imports
        {
            "pattern": r"from utils\.config import.*NodeClass",
            "replacement": "from core.constants import NodeClassWeight as NodeClass",
            "files": ["blockchain/*.py", "tests/**/*.py"]
        }
    ]
    
    # Apply simple replacements
    for fix in fixes:
        if "files" in fix and "old" in fix:
            for filename in fix["files"]:
                if os.path.exists(filename):
                    print(f"Fixing {filename}...")
                    with open(filename, 'r') as f:
                        content = f.read()
                    content = content.replace(fix["old"], fix["new"])
                    with open(filename, 'w') as f:
                        f.write(content)
    
    # Apply regex replacements
    import glob
    
    # Fix config imports in PIR
    for file in glob.glob("pir/**/*.py", recursive=True):
        if os.path.isfile(file):
            print(f"Checking {file}...")
            try:
                with open(file, 'r') as f:
                    content = f.read()
                
                if "from utils.config import config" in content:
                    print(f"  Fixing config import in {file}")
                    content = content.replace(
                        "from utils.config import config",
                        "from utils.config import get_config\nconfig = get_config()"
                    )
                    with open(file, 'w') as f:
                        f.write(content)
            except Exception as e:
                print(f"  Error processing {file}: {e}")
    
    # Fix config imports in ZK proofs
    for file in glob.glob("zk_proofs/**/*.py", recursive=True):
        if os.path.isfile(file):
            try:
                with open(file, 'r') as f:
                    content = f.read()
                
                if "from utils.config import config" in content:
                    print(f"  Fixing config import in {file}")
                    content = content.replace(
                        "from utils.config import config",
                        "from utils.config import get_config\nconfig = get_config()"
                    )
                    with open(file, 'w') as f:
                        f.write(content)
            except Exception as e:
                print(f"  Error processing {file}: {e}")
    
    # Fix NodeClass imports in blockchain
    for file in glob.glob("blockchain/**/*.py", recursive=True):
        if os.path.isfile(file):
            try:
                with open(file, 'r') as f:
                    content = f.read()
                
                if "from utils.config import" in content and "NodeClass" in content:
                    print(f"  Fixing NodeClass import in {file}")
                    # Replace the import line
                    content = re.sub(
                        r"from utils\.config import.*NodeClass.*",
                        "from core.constants import NodeClassWeight as NodeClass",
                        content
                    )
                    with open(file, 'w') as f:
                        f.write(content)
            except Exception as e:
                print(f"  Error processing {file}: {e}")
    
    # Fix API imports
    for file in glob.glob("api/**/*.py", recursive=True):
        if os.path.isfile(file):
            try:
                with open(file, 'r') as f:
                    content = f.read()
                
                if "from utils.config import config" in content:
                    print(f"  Fixing config import in {file}")
                    content = content.replace(
                        "from utils.config import config",
                        "from utils.config import get_config\nconfig = get_config()"
                    )
                    with open(file, 'w') as f:
                        f.write(content)
            except Exception as e:
                print(f"  Error processing {file}: {e}")
    
    print("\n✅ Import fixes applied!")

if __name__ == "__main__":
    os.chdir("/Users/rohanvinaik/genomevault")
    fix_imports()
    
    # Now run the test again
    print("\n" + "=" * 50)
    print("Running import test again...")
    print("=" * 50)
    
    os.system("python3 test_all_imports.py")
