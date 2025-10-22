#!/usr/bin/env python3
"""
Patch NEAT library to fix the 'ValueError: low >= high' bug in generate_variants.py

This bug occurs when NEAT tries to generate a random integer in an invalid range
(where lower bound >= upper bound). The fix adds boundary checks before calling
rng.integers().

Bug location: neat/read_simulator/utils/generate_variants.py, lines 130 and 137
"""

import sys
from pathlib import Path
import shutil
import re

def find_neat_file():
    """Find the generate_variants.py file in NEAT installation."""
    # Try conda environment first
    conda_path = Path.home() / "miniconda3/envs/neat/lib/python3.10/site-packages/neat/read_simulator/utils/generate_variants.py"
    if conda_path.exists():
        return conda_path

    # Try pip installation
    import site
    for site_dir in site.getsitepackages():
        pip_path = Path(site_dir) / "neat/read_simulator/utils/generate_variants.py"
        if pip_path.exists():
            return pip_path

    return None

def backup_file(file_path: Path) -> Path:
    """Create backup of original file."""
    backup_path = file_path.with_suffix('.py.backup')
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"✅ Created backup: {backup_path}")
    else:
        print(f"ℹ️  Backup already exists: {backup_path}")
    return backup_path

def apply_patch(file_path: Path) -> bool:
    """Apply the bug fix patch to generate_variants.py"""

    print(f"\n📝 Reading file: {file_path}")
    with open(file_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'GENOMEVAULT_PATCH' in content:
        print("⚠️  File already patched!")
        return False

    # Original buggy code at line 130:
    # plus = options.rng.integers(window_start + 1, mut_region_offset[1] - 1, dtype=int)

    original_line_130 = r'(\s+)plus = options\.rng\.integers\(window_start \+ 1, mut_region_offset\[1\] - 1, dtype=int\)'

    # Fixed version with boundary check
    patched_line_130 = r'''\1# GENOMEVAULT_PATCH: Add boundary check to prevent ValueError: low >= high
\1if window_start + 1 < mut_region_offset[1] - 1:
\1    plus = options.rng.integers(window_start + 1, mut_region_offset[1] - 1, dtype=int)
\1else:
\1    # Range invalid, skip this attempt
\1    continue'''

    # Apply patch for line 130
    new_content = re.sub(original_line_130, patched_line_130, content)

    if new_content == content:
        print("❌ ERROR: Could not find line 130 pattern to patch")
        print("Looking for: plus = options.rng.integers(window_start + 1, mut_region_offset[1] - 1, dtype=int)")
        return False

    # Original buggy code at line 137:
    # minus = options.rng.integers(mut_region_offset[0], window_start - 1, dtype=int)

    original_line_137 = r'(\s+)if window_start - 1 > mut_region_offset\[0\]:\n(\s+)minus = options\.rng\.integers\(mut_region_offset\[0\], window_start - 1, dtype=int\)'

    # Fixed version
    patched_line_137 = r'''\1if window_start - 1 > mut_region_offset[0]:
\2# GENOMEVAULT_PATCH: Add boundary check
\2if mut_region_offset[0] < window_start - 1:
\2    minus = options.rng.integers(mut_region_offset[0], window_start - 1, dtype=int)
\2else:
\2    # Range invalid, skip
\2    continue'''

    # Apply patch for line 137
    new_content = re.sub(original_line_137, patched_line_137, new_content)

    # Write patched file
    print(f"✍️  Writing patched file...")
    with open(file_path, 'w') as f:
        f.write(new_content)

    print(f"✅ Patch applied successfully!")
    return True

def main():
    print("=" * 70)
    print("NEAT Bug Fix Patcher")
    print("=" * 70)
    print("\nThis script fixes the 'ValueError: low >= high' bug in NEAT's")
    print("generate_variants.py by adding boundary checks before rng.integers() calls.")
    print()

    # Find NEAT installation
    neat_file = find_neat_file()
    if not neat_file:
        print("❌ ERROR: Could not find NEAT installation")
        print("   Please ensure NEAT is installed in conda environment or via pip")
        return 1

    print(f"📍 Found NEAT file: {neat_file}")

    # Create backup
    backup_file(neat_file)

    # Apply patch
    if apply_patch(neat_file):
        print("\n" + "=" * 70)
        print("✅ PATCH SUCCESSFUL")
        print("=" * 70)
        print("\nNEAT library has been patched to fix the boundary check bug.")
        print("You can now run reference pool generation without the ValueError.")
        print(f"\nOriginal file backed up to: {neat_file.with_suffix('.py.backup')}")
        print("\nTo restore original file:")
        print(f"  cp {neat_file.with_suffix('.py.backup')} {neat_file}")
        return 0
    else:
        print("\n❌ PATCH FAILED")
        print("See error messages above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
