#!/usr/bin/env python3
"""
NEAT Bug Fix V4 - Fix Fatal sys.exit(999) in Infinite Loop Detection

The root cause of chunk-84 deadlock:
- After 84 chunks, return_variants has accumulated many variants
- When placing new variants, overlapping locations are common
- The debug counter at line 269 hits 1,000,000 iterations
- sys.exit(999) KILLS THE WORKER PROCESS
- Multiprocessing main process hangs waiting for dead worker's result

Fix: Replace sys.exit(999) with break to gracefully exit the loop.
"""

import sys
from pathlib import Path
import shutil

def find_neat_file():
    conda_path = Path.home() / "miniconda3/envs/neat/lib/python3.10/site-packages/neat/read_simulator/utils/generate_variants.py"
    if conda_path.exists():
        return conda_path
    return None

def apply_v4_patch(file_path: Path) -> bool:
    print(f"\n📝 Reading file: {file_path}")
    with open(file_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'GENOMEVAULT_PATCH_V4' in content:
        print("⚠️  V4 patch already applied!")
        return False

    # Patch: Replace sys.exit(999) with break
    original = '''                    if debug > 1000000:
                        _LOG.error("Check this if, as it may be causing an infinite loop.")
                        sys.exit(999)'''

    patched = '''                    if debug > 1000000:
                        # GENOMEVAULT_PATCH_V4: Don't kill worker, just exit loop gracefully
                        _LOG.warning(f"Hit {debug} iterations trying to place variant - skipping this slice to avoid worker death")
                        break  # Exit inner loop, don't kill the whole worker process!'''

    if original not in content:
        print("❌ ERROR: Could not find sys.exit(999) pattern to patch")
        print("Looking for lines 270-272:")
        print(original)
        return False

    content = content.replace(original, patched)

    # Write patched file
    print(f"✍️  Writing V4 patched file...")
    with open(file_path, 'w') as f:
        f.write(content)

    # Verify patch applied
    with open(file_path, 'r') as f:
        patched_content = f.read()

    if 'GENOMEVAULT_PATCH_V4' in patched_content and 'sys.exit(999)' not in patched_content:
        print(f"✅ V4 patch applied successfully!")
        print(f"   - Removed: sys.exit(999)")
        print(f"   - Added: break statement with warning")
        return True
    else:
        print(f"❌ ERROR: Patch verification failed")
        return False

def main():
    print("=" * 70)
    print("NEAT Bug Fix V4 - Fix Fatal sys.exit(999)")
    print("=" * 70)
    print("\nThis patch fixes the chunk-84 deadlock by replacing the worker-killing")
    print("sys.exit(999) with a graceful break statement.")
    print()

    neat_file = find_neat_file()
    if not neat_file:
        print("❌ ERROR: Could not find NEAT installation")
        return 1

    print(f"📍 Found NEAT file: {neat_file}")

    # Create backup if doesn't exist
    backup_path = neat_file.with_suffix('.py.backup_v4')
    if not backup_path.exists():
        shutil.copy2(neat_file, backup_path)
        print(f"✅ Created backup: {backup_path}")

    if apply_v4_patch(neat_file):
        print("\n" + "=" * 70)
        print("✅ V4 PATCH SUCCESSFUL")
        print("=" * 70)
        print("\nFixed the chunk-84 deadlock!")
        print("Workers will no longer die when hitting difficult genomic regions.")
        print("\nNow restart your NEAT generation:")
        print("  pkill -f 'sample[23]'")
        print("  ./benchmarks/generate_reference_pool.sh")
        return 0
    else:
        print("\n❌ PATCH FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
