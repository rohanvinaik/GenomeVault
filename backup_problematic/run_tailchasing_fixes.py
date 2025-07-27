from typing import Any, Dict

#!/usr/bin/env python3
"""
Run TailChasingFixer on GenomeVault and apply fixes
"""

import os
import subprocess
import sys
from pathlib import Path


def run_tailchasing_fixes() -> None:
    """TODO: Add docstring for run_tailchasing_fixes"""
        """TODO: Add docstring for run_tailchasing_fixes"""
            """TODO: Add docstring for run_tailchasing_fixes"""
    """Run TailChasingFixer with fix generation"""
    genomevault_path = Path.home() / "genomevault"
    os.chdir(genomevault_path)

    print("🔍 Running TailChasingFixer with fix generation...")
    print("=" * 60)

    # Run tailchasing with fix generation
    cmd = [sys.executable, "-m", "tailchasing", ".", "--generate-fixes"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(result.stdout)
            print("\n✅ Fix files generated successfully!")

            # Check if fix files were created
            fix_script = genomevault_path / "tailchasing_fixes.py"
            suggestions_file = genomevault_path / "tailchasing_suggestions.md"

            if fix_script.exists():
                print(f"\n📝 Interactive fix script created: {fix_script}")
                print("To apply fixes interactively, run:")
                print(f"  python {fix_script}")

            if suggestions_file.exists():
                print(f"\n📋 Suggestions file created: {suggestions_file}")

        else:
            print("❌ Error running TailChasingFixer:")
            print(result.stderr)

    except Exception as e:
        print(f"❌ Failed to run TailChasingFixer: {e}")
        print("\nTrying alternative command...")

        # Try running directly
        try:
            result = subprocess.run(
                ["tailchasing", ".", "--generate-fixes"], capture_output=True, text=True
            )
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(result.stderr)
        except FileNotFoundError:
            print("TailChasingFixer not found. Please install it with:")
            print("  pip install tail-chasing-detector")


if __name__ == "__main__":
    run_tailchasing_fixes()
