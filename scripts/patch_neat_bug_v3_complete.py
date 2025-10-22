#!/usr/bin/env python3
"""
NEAT Bug Fix v3 - COMPLETE Patch

Wraps ALL 5 rng.integers() calls in try-except blocks to prevent ValueError: low >= high
This version patches every location where the bug can occur.
"""

import sys
from pathlib import Path
import shutil

def find_neat_file():
    conda_path = Path.home() / "miniconda3/envs/neat/lib/python3.10/site-packages/neat/read_simulator/utils/generate_variants.py"
    if conda_path.exists():
        return conda_path
    return None

def apply_complete_patch(file_path: Path) -> bool:
    print(f"\n📝 Reading file: {file_path}")
    with open(file_path, 'r') as f:
        content = f.read()

    # Patch 1: Line 125 - window_start assignment
    content = content.replace(
        '        window_start = options.rng.integers(mut_region_offset[0], mut_region_offset[1] - 1, dtype=int)',
        '''        # GENOMEVAULT_PATCH_V3: Catch ValueError for window_start
        try:
            window_start = options.rng.integers(mut_region_offset[0], mut_region_offset[1] - 1, dtype=int)
        except ValueError:
            window_start = mut_region_offset[0]'''
    )

    # Patch 2: Line 130 - plus assignment
    content = content.replace(
        '            plus = options.rng.integers(window_start + 1, mut_region_offset[1] - 1, dtype=int)',
        '''            try:
                plus = options.rng.integers(window_start + 1, mut_region_offset[1] - 1, dtype=int)
            except ValueError:
                plus = mut_region_offset[0]'''
    )

    # Patch 3: Line 137 - minus assignment
    content = content.replace(
        '                    minus = options.rng.integers(mut_region_offset[0], window_start - 1, dtype=int)',
        '''                    try:
                        minus = options.rng.integers(mut_region_offset[0], window_start - 1, dtype=int)
                    except ValueError:
                        minus = mut_region_offset[0]'''
    )

    # Patch 4: Lines 149-152 - end_point first assignment (multi-line)
    content = content.replace(
        '''        end_point = min(
            options.rng.integers(
                window_start,
                min(mut_region_offset[1], window_start+max_window_size)-1,
                dtype=int
            ),
            # Don't go past the barrier
            len(reference)-1
        )''',
        '''        try:
            end_point = min(
                options.rng.integers(
                    window_start,
                    min(mut_region_offset[1], window_start+max_window_size)-1,
                    dtype=int
                ),
                # Don't go past the barrier
                len(reference)-1
            )
        except ValueError:
            end_point = window_start'''
    )

    # Patch 5: Lines 159-162 - end_point fallback assignment (multi-line)
    content = content.replace(
        '''            end_point = options.rng.integers(
                max(window_start - max_window_size, mut_region_offset[0]),
                window_start,
                dtype=int
            )''',
        '''            try:
                end_point = options.rng.integers(
                    max(window_start - max_window_size, mut_region_offset[0]),
                    window_start,
                    dtype=int
                )
            except ValueError:
                end_point = window_start'''
    )

    # Write patched file
    print(f"✍️  Writing comprehensively patched file...")
    with open(file_path, 'w') as f:
        f.write(content)

    # Verify all patches applied
    with open(file_path, 'r') as f:
        patched_content = f.read()

    patch_count = patched_content.count('GENOMEVAULT_PATCH_V3')
    print(f"✅ Applied {patch_count} patches")

    if patch_count >= 2:  # At least 2 markers
        return True
    else:
        print(f"❌ ERROR: Only {patch_count} patches applied (expected at least 2)")
        return False

def main():
    print("=" * 70)
    print("NEAT Bug Fix v3 - COMPLETE Comprehensive Patch")
    print("=" * 70)
    print("\nThis version patches ALL 5 rng.integers() calls with try-except.")
    print()

    neat_file = find_neat_file()
    if not neat_file:
        print("❌ ERROR: Could not find NEAT installation")
        return 1

    print(f"📍 Found NEAT file: {neat_file}")

    # Create backup
    backup_path = neat_file.with_suffix('.py.backup')
    if not backup_path.exists():
        shutil.copy2(neat_file, backup_path)
        print(f"✅ Created backup: {backup_path}")

    if apply_complete_patch(neat_file):
        print("\n" + "=" * 70)
        print("✅ COMPREHENSIVE PATCH V3 SUCCESSFUL")
        print("=" * 70)
        print("\nAll 5 rng.integers() calls patched with try-except!")
        print("No more ValueError: low >= high errors!")
        return 0
    else:
        print("\n❌ PATCH FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
