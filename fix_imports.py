#!/usr/bin/env python3
"""
Fix imports in GenomeVault modules
"""

import os
import re
from pathlib import Path


def fix_imports_in_file(filepath):
    """Fix relative imports in a Python file"""
    with open(filepath, "r") as f:
        content = f.read()

    # Replace relative imports with absolute imports
    replacements = [
        (r"from \.\.utils import", "from utils import"),
        (r"from \.\.core import", "from core import"),
        (r"from \.\.utils\.", "from utils."),
        (r"from \.\.core\.", "from core."),
        (r"from \.\. import", "from genomevault import"),
    ]

    modified = False
    for pattern, replacement in replacements:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            modified = True
            content = new_content

    if modified:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"✅ Fixed imports in {filepath}")
        return True
    return False


def main():
    """Fix all imports in the project"""
    print("🔧 Fixing imports in GenomeVault modules...")

    # Find all Python files in key directories
    directories = ["local_processing", "hypervector_transform", "zk_proofs", "pir", "blockchain"]

    fixed_count = 0
    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(".py") and not file.startswith("__pycache__"):
                        filepath = os.path.join(root, file)
                        if fix_imports_in_file(filepath):
                            fixed_count += 1

    print(f"\n✅ Fixed imports in {fixed_count} files")


if __name__ == "__main__":
    main()
