from genomevault.utils.logging import get_logger
logger = get_logger(__name__)


#!/usr/bin/env python3
"""
Manual verification script for the genomevault fixes
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and report success/failure"""
    logger.debug(f"\n{'='*60}")
    logger.debug(f"Testing: {description}")
    logger.debug(f"Command: {' '.join(cmd)}")
    logger.debug("=" * 60)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            logger.info(f"✅ SUCCESS: {description}")
            if result.stdout:
                logger.debug("Output:", result.stdout[:500])
        else:
            logger.error(f"❌ FAILED: {description}")
            logger.debug("Return code:", result.returncode)
            if result.stderr:
                logger.error("Error:", result.stderr[:500])

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        logger.debug(f"⏰ TIMEOUT: {description}")
        return False
    except Exception as e:
        logger.error(f"💥 EXCEPTION: {description} - {e}")
        return False


def main():
    # Change to the genomevault directory
    os.chdir(".")

    logger.debug("🚀 GenomeVault Fix Verification")
    logger.debug(f"Working directory: {os.getcwd()}")

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
    logger.debug("\n" + "=" * 80)
    logger.debug("VERIFICATION SUMMARY")
    logger.debug("=" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for desc, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.debug(f"{status}: {desc}")

    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        logger.info("🎉 All fixes verified successfully!")
        return 0
    else:
        logger.error("⚠️  Some issues remain - check the failures above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
