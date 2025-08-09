#!/usr/bin/env python3
"""Simple test script to verify fixes are working."""

import sys
from pathlib import Path


def test_ruff_config():
    """Test if Ruff configuration exists and is readable."""
    config_path = Path(".ruff.toml")
    if config_path.exists():
        content = config_path.read_text()
        print("✅ .ruff.toml exists and is readable")
        if "[output]" in content and "max-violations" in content:
            print("⚠️  Config still contains [output] section")
            return False
        else:
            print("✅ Config looks good")
            return True
    else:
        print("❌ .ruff.toml not found")
        return False


def test_core_modules():
    """Test if core modules can be imported."""
    try:
        import genomevault.core.exceptions

        print("✅ genomevault.core.exceptions imports successfully")

        # Test the HypervectorError specifically
        from genomevault.core.exceptions import HypervectorError

        print("✅ HypervectorError is available")

        import genomevault.core.constants

        print("✅ genomevault.core.constants imports successfully")

        # Test specific constants
        from genomevault.core.constants import HYPERVECTOR_DIMENSIONS, MAX_VARIANTS

        print(
            f"✅ Constants loaded: HYPERVECTOR_DIMENSIONS={HYPERVECTOR_DIMENSIONS}, MAX_VARIANTS={MAX_VARIANTS}"
        )

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_basic_ruff():
    """Test if Ruff can run basic commands."""
    import subprocess

    try:
        # Test help command
        result = subprocess.run(["ruff", "--help"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ruff help command works")
        else:
            print(f"❌ Ruff help failed: {result.stderr}")
            return False

        # Test version command
        result = subprocess.run(["ruff", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Ruff version: {version}")
            if "0.4." in version:
                print("✅ Using Ruff 0.4.x - Good!")
                return True
            else:
                print(f"⚠️  Ruff version {version} may have compatibility issues")
                return False
        else:
            print(f"❌ Ruff version check failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Ruff command timed out")
        return False
    except Exception as e:
        print(f"❌ Error running Ruff: {e}")
        return False


def main():
    """Run all tests."""
    print("=== GENOMEVAULT FIX VERIFICATION ===")

    tests = [
        ("Ruff Configuration", test_ruff_config),
        ("Core Module Imports", test_core_modules),
        ("Basic Ruff Functionality", test_basic_ruff),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} error: {e}")

    print("\n=== SUMMARY ===")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! The fixes are working correctly.")
        print("\nNext steps:")
        print("1. Run: ruff check . --statistics")
        print("2. Run: pytest -q -k 'not api and not nanopore' --maxfail=3")
        print(
            "3. Commit changes: git add -A && git commit -m 'fix: resolve Ruff version conflict and add missing modules'"
        )
    else:
        print("⚠️  Some tests failed. Manual intervention may be needed.")
        print("\nTroubleshooting:")
        print("1. Check Ruff version: ruff --version")
        print("2. Verify conda environment: conda list ruff")
        print("3. Check PATH: which -a ruff")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
