#!/usr/bin/env python3
"""
NEAT Comprehensive Fix - Intelligent Variant Saturation Handling

ROOT CAUSE ANALYSIS:
After ~84 chunks, return_variants accumulates so many variants that genomic
locations become saturated. The current code:
1. Tries to place a variant
2. If location is occupied (all ploids full), increments debug counter
3. Continues trying up to 1 MILLION times
4. Then kills the worker with sys.exit(999)

This is fundamentally broken because:
- After 84 chunks, saturation is EXPECTED, not an error
- 1 million retries is wasteful and still fails
- Killing the worker hangs the entire multiprocessing pool

COMPREHENSIVE FIX:
1. Add detailed diagnostic logging to understand saturation patterns
2. Implement smart retry limit (10,000 instead of 1 million)
3. Track retry statistics per slice
4. When retry limit hit, skip remaining variants in that slice
5. Log saturation statistics for debugging
6. Return successfully placed variants (baseline genetic data)
7. Clear error counters between slices to avoid false positives
"""

import sys
from pathlib import Path
import shutil

def find_neat_file():
    conda_path = Path.home() / "miniconda3/envs/neat/lib/python3.10/site-packages/neat/read_simulator/utils/generate_variants.py"
    if conda_path.exists():
        return conda_path
    return None

def apply_comprehensive_patch(file_path: Path) -> bool:
    print(f"\n📝 Reading file: {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if any('GENOMEVAULT_COMPREHENSIVE_FIX' in line for line in lines):
        print("⚠️  Comprehensive patch already applied!")
        return False

    patches_applied = 0

    # PATCH 1: Initialize saturation tracking at beginning of slice loop (after line 212)
    for i in range(len(lines)):
        if 'debug = 0' in lines[i] and 'Begin random mutations' in lines[i-2]:
            # Insert saturation tracking initialization
            indent = '        '
            lines[i] = f'''{indent}debug = 0
{indent}# GENOMEVAULT_COMPREHENSIVE_FIX: Track saturation statistics
{indent}retry_limit = 10000  # Reasonable limit instead of 1 million
{indent}location_conflicts = 0
{indent}ploid_saturation = 0
{indent}successful_placements = 0
'''
            patches_applied += 1
            print(f"✅ Patch 1: Added saturation tracking initialization (line {i+1})")
            break

    # PATCH 2: Add diagnostic logging and smart retry handling (replace the debug counter section)
    for i in range(len(lines)):
        if 'if 0 not in composite_genotype:' in lines[i]:
            # Found the saturation detection point
            indent = '                    '
            # Replace lines i through i+5 (the debug counter and sys.exit block)
            old_section_end = i + 1
            while old_section_end < len(lines) and 'continue' not in lines[old_section_end]:
                old_section_end += 1
            old_section_end += 1  # Include the continue line

            new_section = f'''{indent}if 0 not in composite_genotype:
{indent}    # GENOMEVAULT_COMPREHENSIVE_FIX: All ploids occupied - track and handle gracefully
{indent}    ploid_saturation += 1
{indent}    debug += 1
{indent}
{indent}    if debug > retry_limit:
{indent}        # Saturation reached - log diagnostics and skip remaining variants in this slice
{indent}        _LOG.warning(
{indent}            f"Genomic saturation in slice after {{successful_placements}} placements: "
{indent}            f"{{location_conflicts}} location conflicts, {{ploid_saturation}} ploid saturations. "
{indent}            f"Skipping {{variants_to_add_in_slice}} remaining variants in this slice."
{indent}        )
{indent}        # Exit the slice loop, keeping successfully placed variants
{indent}        break
{indent}    # Try next iteration to find an open location
{indent}    continue
'''
            lines[i:old_section_end] = [new_section]
            patches_applied += 1
            print(f"✅ Patch 2: Added smart saturation handling (lines {i+1}-{old_section_end})")
            break

    # PATCH 3: Track successful placements (after line ~303 where variant is added)
    for i in range(len(lines)):
        if 'return_variants.add_variant(temp_variant)' in lines[i]:
            indent = '            '
            lines[i] = f'''{lines[i]}{indent}# GENOMEVAULT_COMPREHENSIVE_FIX: Track successful placement
{indent}successful_placements += 1
{indent}debug = 0  # Reset retry counter on success
'''
            patches_applied += 1
            print(f"✅ Patch 3: Added success tracking (line {i+1})")
            break

    # PATCH 4: Track location conflicts (after line ~258 where location conflict is detected)
    for i in range(len(lines)):
        if 'if location in return_variants:' in lines[i]:
            indent = '            '
            # Add tracking right after the if statement
            lines[i] = f'''{lines[i]}{indent}    # GENOMEVAULT_COMPREHENSIVE_FIX: Track location conflict
{indent}    location_conflicts += 1
'''
            patches_applied += 1
            print(f"✅ Patch 4: Added location conflict tracking (line {i+1})")
            break

    if patches_applied < 4:
        print(f"❌ ERROR: Only applied {patches_applied}/4 patches")
        return False

    # Write patched file
    print(f"\n✍️  Writing comprehensively patched file...")
    with open(file_path, 'w') as f:
        f.writelines(lines)

    print(f"✅ All {patches_applied} patches applied successfully!")
    return True

def main():
    print("=" * 80)
    print("NEAT Comprehensive Fix - Intelligent Variant Saturation Handling")
    print("=" * 80)
    print("\nThis patch implements:")
    print("  1. Diagnostic logging for saturation patterns")
    print("  2. Smart retry limit (10,000 instead of 1 million)")
    print("  3. Saturation statistics tracking")
    print("  4. Graceful skip of saturated slices")
    print("  5. Successful variant preservation")
    print("  6. Error counter reset between successes")
    print()

    neat_file = find_neat_file()
    if not neat_file:
        print("❌ ERROR: Could not find NEAT installation")
        return 1

    print(f"📍 Found NEAT file: {neat_file}")

    # Create backup
    backup_path = neat_file.with_suffix('.py.backup_comprehensive')
    if not backup_path.exists():
        shutil.copy2(neat_file, backup_path)
        print(f"✅ Created backup: {backup_path}")

    if apply_comprehensive_patch(neat_file):
        print("\n" + "=" * 80)
        print("✅ COMPREHENSIVE PATCH SUCCESSFUL")
        print("=" * 80)
        print("\nNEAT will now:")
        print("  • Track saturation statistics in logs")
        print("  • Gracefully skip saturated genomic regions")
        print("  • Preserve successfully placed variants")
        print("  • Continue processing without worker death")
        print("\nRestart your NEAT generation:")
        print("  pkill -f 'sample[23]'")
        print("  find /var/folders -type d -name '*tmp*' -exec rm -rf {} + 2>/dev/null")
        print("  ./benchmarks/generate_reference_pool.sh")
        return 0
    else:
        print("\n❌ PATCH FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
