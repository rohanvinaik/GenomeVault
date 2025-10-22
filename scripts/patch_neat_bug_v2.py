#!/usr/bin/env python3
"""
NEAT Bug Fix v2 - Non-Deadlocking Patch

Uses try-except instead of continue to avoid multiprocessing deadlocks.
When rng.integers() hits an invalid range, catches ValueError and uses fallback.
"""

import sys
from pathlib import Path
import shutil

def find_neat_file():
    conda_path = Path.home() / "miniconda3/envs/neat/lib/python3.10/site-packages/neat/read_simulator/utils/generate_variants.py"
    if conda_path.exists():
        return conda_path
    return None

def backup_file(file_path: Path) -> Path:
    backup_path = file_path.with_suffix('.py.backup')
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"✅ Created backup: {backup_path}")
    else:
        print(f"ℹ️  Backup already exists: {backup_path}")
    return backup_path

def apply_patch_v2(file_path: Path) -> bool:
    print(f"\n📝 Reading file: {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find and patch line 130 area
    patched = False
    for i in range(len(lines)):
        # Look for the problematic line around line 130
        if 'plus = options.rng.integers(window_start + 1, mut_region_offset[1] - 1, dtype=int)' in lines[i]:
            indent = '            '
            # Replace with try-except block
            lines[i] = f'''{indent}# GENOMEVAULT_PATCH_V2: Catch ValueError and use fallback
{indent}try:
{indent}    plus = options.rng.integers(window_start + 1, mut_region_offset[1] - 1, dtype=int)
{indent}except ValueError:
{indent}    # Invalid range, use fallback position
{indent}    plus = mut_region_offset[0]
'''
            print(f"✅ Patched line {i+1} (plus assignment)")
            patched = True

        # Look for the second problematic line around line 137
        elif 'minus = options.rng.integers(mut_region_offset[0], window_start - 1, dtype=int)' in lines[i]:
            indent = '                    '
            lines[i] = f'''{indent}# GENOMEVAULT_PATCH_V2: Catch ValueError and use fallback
{indent}try:
{indent}    minus = options.rng.integers(mut_region_offset[0], window_start - 1, dtype=int)
{indent}except ValueError:
{indent}    # Invalid range, use fallback position
{indent}    minus = mut_region_offset[0]
'''
            print(f"✅ Patched line {i+1} (minus assignment)")
            patched = True

    if not patched:
        print("❌ ERROR: Could not find lines to patch")
        return False

    # Write patched file
    print(f"✍️  Writing patched file...")
    with open(file_path, 'w') as f:
        f.writelines(lines)

    print(f"✅ Patch v2 applied successfully!")
    return True

def main():
    print("=" * 70)
    print("NEAT Bug Fix v2 - Non-Deadlocking Patch")
    print("=" * 70)
    print("\nThis version uses try-except instead of continue to avoid deadlocks.")
    print()

    neat_file = find_neat_file()
    if not neat_file:
        print("❌ ERROR: Could not find NEAT installation")
        return 1

    print(f"📍 Found NEAT file: {neat_file}")
    backup_file(neat_file)

    if apply_patch_v2(neat_file):
        print("\n" + "=" * 70)
        print("✅ PATCH V2 SUCCESSFUL")
        print("=" * 70)
        print("\nNEAT patched with try-except blocks - no more deadlocks!")
        print(f"\nOriginal backed up to: {neat_file.with_suffix('.py.backup')}")
        return 0
    else:
        print("\n❌ PATCH FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
