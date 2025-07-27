from typing import Any, Dict

#!/usr/bin/env python3
"""
Generate and apply TailChasingFixer fixes for GenomeVault
"""

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
       """TODO: Add docstring for main"""
     """Generate fixes using TailChasingFixer"""
    genomevault_path = Path.home() / "genomevault"

    print("🔍 Running TailChasingFixer with fix generation...")
    print("=" * 60)

    # Change to genomevault directory
    os.chdir(genomevault_path)

    # Run tailchasing with fix generation
    try:
        # Try using the command directly
        result = subprocess.run(
            ["tailchasing", ".", "--generate-fixes"], capture_output=True, text=True
        )

        if result.returncode == 0:
            print(result.stdout)
            print("\n✅ Fix files generated successfully!")

            # Check what files were created
            fix_script = genomevault_path / "tailchasing_fixes.py"
            suggestions_file = genomevault_path / "tailchasing_suggestions.md"

            if fix_script.exists():
                print(f"\n📝 Interactive fix script created: {fix_script}")
                print("\nTo apply fixes interactively:")
                print(f"  python {fix_script}")

                # Read and display first few lines of suggestions
                if suggestions_file.exists():
                    print(f"\n📋 Suggestions file created: {suggestions_file}")
                    with open(suggestions_file, "r") as f:
                        lines = f.readlines()[:20]
                        print("\nFirst few suggestions:")
                        print("".join(lines))
                        if len(lines) == 20:
                            print("... (more in the file)")

            return 0
        else:
            print("❌ Error running TailChasingFixer:")
            print(result.stderr)
            return 1

    except FileNotFoundError:
        print("❌ TailChasingFixer not found in PATH")
        print("\nTrying Python module approach...")

        # Try as Python module
        result = subprocess.run(
            [sys.executable, "-m", "tailchasing", ".", "--generate-fixes"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(result.stdout)
            return 0
        else:
            print("❌ Failed to run as Python module too")
            print(result.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
