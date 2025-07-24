#!/usr/bin/env python3
"""
Fix Black formatting issues for GenomeVault
"""
import subprocess
import sys
from pathlib import Path


def run_black_fix():
    """Run black to automatically fix formatting issues"""
    try:
        # Change to the genomevault directory
        genomevault_dir = Path("/Users/rohanvinaik/genomevault")

        # Run black on the entire directory
        result = subprocess.run(
            ["black", "."], cwd=genomevault_dir, capture_output=True, text=True
        )

        if result.returncode == 0:
            print("✅ Black formatting completed successfully!")
            print("Files have been auto-formatted.")
            if result.stdout:
                print(f"Output: {result.stdout}")
        else:
            print("❌ Black formatting failed:")
            print(f"Error: {result.stderr}")
            return False

        return True

    except Exception as e:
        print(f"❌ Error running black: {e}")
        return False


def main():
    print("🔧 Fixing Black formatting issues...")

    if run_black_fix():
        print("\n✅ All formatting issues should now be fixed!")
        print("You can now run your CI pipeline.")
    else:
        print("\n❌ Failed to fix formatting issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
