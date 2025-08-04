#!/usr/bin/env python3
"""
Manual verification script for the genomevault fixes
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report success/failure"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout:
                print("Output:", result.stdout[:500])
        else:
            print(f"❌ FAILED: {description}")
            print("Return code:", result.returncode)
            if result.stderr:
                print("Error:", result.stderr[:500])

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {description} - {e}")
        return False


def main():
    # Change to the genomevault directory
    os.chdir("/Users/rohanvinaik/genomevault")

    print("🚀 GenomeVault Fix Verification")
    print(f"Working directory: {os.getcwd()}")

    # Test 1: Check syntax of fixed files
    tests = [
        (
            [
                "python",
                "-m",
                "py_compile",
                "genomevault/hypervector/encoding/unified_encoder.py",
            ],
            "Syntax check: unified_encoder.py",
        ),
        (
            [
                "python",
                "-m",
                "py_compile",
                "genomevault/hypervector/encoding/__init__.py",
            ],
            "Syntax check: hypervector encoding __init__.py",
        ),
        (
            [
                "python",
                "-c",
                "from genomevault.core.config import get_config, Config; print('Config import works')",
            ],
            "Import test: Config module",
        ),
        (["ruff", "--version"], "Ruff version check"),
        (["ruff", "check", ".", "--statistics"], "Ruff statistics check"),
        (
            [
                "python",
                "-c",
                "import genomevault.hypervector.encoding; print('Hypervector encoding imports work')",
            ],
            "Import test: Hypervector encoding",
        ),
    ]

    results = []
    for cmd, desc in tests:
        success = run_command(cmd, desc)
        results.append((desc, success))

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for desc, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {desc}")

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 All fixes verified successfully!")
        return 0
    else:
        print("⚠️  Some issues remain - check the failures above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
