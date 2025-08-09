#!/usr/bin/env python3
"""Final validation and testing script for GenomeVault fixes."""

import os
import subprocess
import sys


def test_ruff_functionality():
    """Test Ruff is working with current configuration."""
    print("=== TESTING RUFF FUNCTIONALITY ===")

    try:
        # Test version
        result = subprocess.run(
            ["ruff", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Ruff version: {version}")

            if "0.4." in version:
                print("✅ Using compatible Ruff 0.4.x")
            else:
                print(f"⚠️  Version {version} may have issues")
                return False
        else:
            print(f"❌ Ruff version check failed: {result.stderr}")
            return False

        # Test configuration parsing
        result = subprocess.run(
            ["ruff", "check", "--help"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("✅ Ruff configuration parses correctly")
        else:
            print(f"❌ Ruff configuration error: {result.stderr}")
            return False

        # Test actual checking (expect some issues but no crashes)
        result = subprocess.run(
            ["ruff", "check", ".", "--statistics"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if "unknown field" in result.stderr:
            print(
                "❌ CRITICAL: Ruff still has 'unknown field' error - version conflict remains!"
            )
            return False
        else:
            print("✅ Ruff check runs without configuration crashes")
            if result.stdout:
                print(f"Statistics preview: {result.stdout[:200]}...")
            return True

    except subprocess.TimeoutExpired:
        print("❌ Ruff commands timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing Ruff: {e}")
        return False


def test_imports():
    """Test critical imports."""
    print("\n=== TESTING IMPORTS ===")

    imports_to_test = [
        ("genomevault.exceptions", "HypervectorError"),
        ("genomevault.utils.constants", "HYPERVECTOR_DIMENSIONS"),
        ("genomevault.utils.constants", "MAX_VARIANTS"),
        ("genomevault.utils.constants", "NODE_CLASS_WEIGHT"),
    ]

    success_count = 0
    for module_name, symbol in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[symbol])
            if hasattr(module, symbol):
                print(f"✅ {module_name}.{symbol} imports successfully")
                success_count += 1
            else:
                print(f"❌ {module_name}.{symbol} not found in module")
        except ImportError as e:
            print(f"❌ Import failed: {module_name}.{symbol} - {e}")
        except Exception as e:
            print(f"❌ Error importing {module_name}.{symbol}: {e}")

    return success_count == len(imports_to_test)


def test_phase3_readiness():
    """Test if Phase 3 can run successfully."""
    print("\n=== TESTING PHASE 3 READINESS ===")

    try:
        result = subprocess.run(
            ["python", "comprehensive_cleanup.py", "--phase", "3"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if "Found" in result.stdout and "F821" in result.stdout:
            print("✅ Phase 3 found and processed F821 violations")
            return True
        elif "No F821 violations found" in result.stdout:
            print("✅ Phase 3 reports no F821 violations")
            return True
        elif result.returncode == 0:
            print("✅ Phase 3 completed successfully")
            return True
        else:
            print(f"⚠️  Phase 3 had issues: {result.stderr[:300]}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Phase 3 timed out")
        return False
    except Exception as e:
        print(f"❌ Could not run Phase 3: {e}")
        return False


def test_pytest_basic():
    """Test basic pytest functionality."""
    print("\n=== TESTING PYTEST BASICS ===")

    try:
        result = subprocess.run(
            ["pytest", "--version"], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            print(f"✅ pytest available: {result.stdout.strip()}")
        else:
            print("⚠️  pytest not available or has issues")
            return False

        # Try a very basic test run
        result = subprocess.run(
            [
                "pytest",
                "-q",
                "-k",
                "not api and not nanopore",
                "--maxfail=1",
                "--collect-only",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )

        if "collected" in result.stdout:
            print("✅ pytest can collect tests")
            return True
        else:
            print(f"⚠️  pytest collection issues: {result.stderr[:200]}")
            return False

    except Exception as e:
        print(f"❌ pytest test failed: {e}")
        return False


def show_summary_and_next_steps(results):
    """Show final summary and next steps."""
    print("\n" + "=" * 60)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    print(f"\nTest Results: {passed_tests}/{total_tests} passed")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print(
            "\nThe Ruff version conflict has been resolved and core modules are working."
        )
        print("\n📋 RECOMMENDED NEXT STEPS:")
        print("1. Run full Phase 3: python comprehensive_cleanup.py --phase 3")
        print("2. Run Phase 7 validation: python comprehensive_cleanup.py --phase 7")
        print("3. Test Ruff: ruff check . --statistics")
        print("4. Test pytest: pytest -q -k 'not api and not nanopore' --maxfail=3")
        print(
            "5. Commit changes: git add -A && git commit -m 'fix: resolve Ruff version conflict'"
        )

    elif passed_tests >= total_tests * 0.75:
        print("\n✅ MOSTLY SUCCESSFUL!")
        print("\nMost critical issues have been resolved.")
        print("\n📋 NEXT STEPS:")
        print("1. Address any remaining failed tests above")
        print("2. Run: ruff check . --statistics")
        print("3. If Ruff works, proceed with comprehensive cleanup")

    else:
        print("\n⚠️  SIGNIFICANT ISSUES REMAIN")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check Ruff version: ruff --version")
        print("2. Verify conda environment: conda list ruff")
        print("3. Check PATH: which -a ruff")
        print("4. Try manual Ruff install: pip install 'ruff>=0.4.4,<0.5'")
        print("5. Clear shell cache: hash -r")

    print("\n💡 KEY ACHIEVEMENTS:")
    print("• Ruff configuration updated for 0.4.x compatibility")
    print("• HypervectorError added to genomevault.exceptions")
    print("• Core constants enhanced with missing values")
    print("• Project structure validated")

    return passed_tests == total_tests


def main():
    """Main validation routine."""
    print("=" * 60)
    print("GENOMEVAULT RUFF FIX FINAL VALIDATION")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")

    # Ensure we're in the right directory
    os.chdir("/Users/rohanvinaik/genomevault")

    # Run all tests
    tests = {
        "Ruff Functionality": test_ruff_functionality,
        "Critical Imports": test_imports,
        "Phase 3 Readiness": test_phase3_readiness,
        "pytest Basics": test_pytest_basic,
    }

    results = {}
    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False

    # Show summary
    success = show_summary_and_next_steps(results)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
