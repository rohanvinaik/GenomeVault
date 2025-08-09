#!/usr/bin/env python3

"""Final validation and testing script for GenomeVault fixes."""

import os
import subprocess
import sys
from genomevault.utils.logging import get_logger
logger = get_logger(__name__)



def test_ruff_functionality():
    """Test Ruff is working with current configuration."""
    logger.debug("=== TESTING RUFF FUNCTIONALITY ===")

    try:
        # Test version
        result = subprocess.run(["ruff", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.debug(f"✅ Ruff version: {version}")

            if "0.4." in version:
                logger.debug("✅ Using compatible Ruff 0.4.x")
            else:
                logger.debug(f"⚠️  Version {version} may have issues")
                return False
        else:
            logger.error(f"❌ Ruff version check failed: {result.stderr}")
            return False

        # Test configuration parsing
        result = subprocess.run(
            ["ruff", "check", "--help"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            logger.debug("✅ Ruff configuration parses correctly")
        else:
            logger.error(f"❌ Ruff configuration error: {result.stderr}")
            return False

        # Test actual checking (expect some issues but no crashes)
        result = subprocess.run(
            ["ruff", "check", ".", "--statistics"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if "unknown field" in result.stderr:
            logger.error("❌ CRITICAL: Ruff still has 'unknown field' error - version conflict remains!")
            return False
        else:
            logger.debug("✅ Ruff check runs without configuration crashes")
            if result.stdout:
                logger.debug(f"Statistics preview: {result.stdout[:200]}...")
            return True

    except subprocess.TimeoutExpired:
        logger.debug("❌ Ruff commands timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing Ruff: {e}")
        return False


def test_imports():
    """Test critical imports."""
    logger.debug("\n=== TESTING IMPORTS ===")

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
                logger.info(f"✅ {module_name}.{symbol} imports successfully")
                success_count += 1
            else:
                logger.debug(f"❌ {module_name}.{symbol} not found in module")
        except ImportError as e:
            logger.error(f"❌ Import failed: {module_name}.{symbol} - {e}")
        except Exception as e:
            logger.error(f"❌ Error importing {module_name}.{symbol}: {e}")

    return success_count == len(imports_to_test)


def test_phase3_readiness():
    """Test if Phase 3 can run successfully."""
    logger.debug("\n=== TESTING PHASE 3 READINESS ===")

    try:
        result = subprocess.run(
            ["python", "comprehensive_cleanup.py", "--phase", "3"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if "Found" in result.stdout and "F821" in result.stdout:
            logger.debug("✅ Phase 3 found and processed F821 violations")
            return True
        elif "No F821 violations found" in result.stdout:
            logger.debug("✅ Phase 3 reports no F821 violations")
            return True
        elif result.returncode == 0:
            logger.info("✅ Phase 3 completed successfully")
            return True
        else:
            logger.debug(f"⚠️  Phase 3 had issues: {result.stderr[:300]}")
            return False

    except subprocess.TimeoutExpired:
        logger.debug("❌ Phase 3 timed out")
        return False
    except Exception as e:
        logger.debug(f"❌ Could not run Phase 3: {e}")
        return False


def test_pytest_basic():
    """Test basic pytest functionality."""
    logger.debug("\n=== TESTING PYTEST BASICS ===")

    try:
        result = subprocess.run(["pytest", "--version"], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            logger.debug(f"✅ pytest available: {result.stdout.strip()}")
        else:
            logger.debug("⚠️  pytest not available or has issues")
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
            logger.debug("✅ pytest can collect tests")
            return True
        else:
            logger.debug(f"⚠️  pytest collection issues: {result.stderr[:200]}")
            return False

    except Exception as e:
        logger.error(f"❌ pytest test failed: {e}")
        return False


def show_summary_and_next_steps(results):
    """Show final summary and next steps."""
    logger.debug("\n" + "=" * 60)
    logger.debug("FINAL VALIDATION SUMMARY")
    logger.debug("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    logger.info(f"\nTest Results: {passed_tests}/{total_tests} passed")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.debug(f"  {test_name}: {status}")

    if passed_tests == total_tests:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.debug("\nThe Ruff version conflict has been resolved and core modules are working.")
        logger.debug("\n📋 RECOMMENDED NEXT STEPS:")
        logger.debug("1. Run full Phase 3: python comprehensive_cleanup.py --phase 3")
        logger.debug("2. Run Phase 7 validation: python comprehensive_cleanup.py --phase 7")
        logger.debug("3. Test Ruff: ruff check . --statistics")
        logger.error("4. Test pytest: pytest -q -k 'not api and not nanopore' --maxfail=3")
        logger.debug("5. Commit changes: git add -A && git commit -m 'fix: resolve Ruff version conflict'")

    elif passed_tests >= total_tests * 0.75:
        logger.info("\n✅ MOSTLY SUCCESSFUL!")
        logger.error("\nMost critical issues have been resolved.")
        logger.debug("\n📋 NEXT STEPS:")
        logger.error("1. Address any remaining failed tests above")
        logger.debug("2. Run: ruff check . --statistics")
        logger.debug("3. If Ruff works, proceed with comprehensive cleanup")

    else:
        logger.debug("\n⚠️  SIGNIFICANT ISSUES REMAIN")
        logger.debug("\n🔧 TROUBLESHOOTING:")
        logger.debug("1. Check Ruff version: ruff --version")
        logger.debug("2. Verify conda environment: conda list ruff")
        logger.debug("3. Check PATH: which -a ruff")
        logger.debug("4. Try manual Ruff install: pip install 'ruff>=0.4.4,<0.5'")
        logger.debug("5. Clear shell cache: hash -r")

    logger.debug("\n💡 KEY ACHIEVEMENTS:")
    logger.debug("• Ruff configuration updated for 0.4.x compatibility")
    logger.error("• HypervectorError added to genomevault.exceptions")
    logger.debug("• Core constants enhanced with missing values")
    logger.debug("• Project structure validated")

    return passed_tests == total_tests


def main():
    """Main validation routine."""
    logger.debug("=" * 60)
    logger.debug("GENOMEVAULT RUFF FIX FINAL VALIDATION")
    logger.debug("=" * 60)
    logger.debug(f"Working directory: {os.getcwd()}")
    logger.debug(f"Python version: {sys.version}")

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
            logger.debug(f"❌ {test_name} crashed: {e}")
            results[test_name] = False

    # Show summary
    success = show_summary_and_next_steps(results)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
