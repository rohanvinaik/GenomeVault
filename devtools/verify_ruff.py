from genomevault.utils.logging import get_logger
logger = get_logger(__name__)


#!/usr/bin/env python3
"""
Quick verification that Ruff configuration is working.
"""

import json
import subprocess


def test_ruff_config():
    """Test Ruff configuration and F821 detection."""
    logger.debug("🔧 Testing Ruff configuration...")

    # Test 1: Check if ruff check works at all
    try:
        result = subprocess.run(
            ["ruff", "check", ".", "--quiet"],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=10,
        )
        logger.debug(f"✅ Basic ruff check works (exit code: {result.returncode})")
    except Exception as e:
        logger.error(f"❌ Basic ruff check failed: {e}")
        return False

    # Test 2: Check F821 specific detection
    try:
        result = subprocess.run(
            ["ruff", "check", ".", "--select", "F821", "--output-format", "json"],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=30,
        )

        logger.debug(f"✅ F821 check works (exit code: {result.returncode})")

        if result.stdout:
            try:
                violations = json.loads(result.stdout)
                logger.debug(f"✅ Found {len(violations)} F821 violations")

                if violations:
                    logger.debug("\\nFirst few violations:")
                    for i, v in enumerate(violations[:3]):
                        file_short = v["filename"].replace("./", "")
                        logger.debug(f"  {i+1}. {file_short}:{v['location']['row']} - {v['message']}")

                return len(violations) > 0
            except json.JSONDecodeError:
                logger.debug("❌ Could not parse JSON output")
                logger.debug(f"Raw output: {result.stdout[:200]}")
        else:
            logger.debug("✅ No F821 violations found")
            return True

    except Exception as e:
        logger.error(f"❌ F821 check failed: {e}")
        return False


if __name__ == "__main__":
    logger.debug("🚀 Verifying Ruff Configuration Fix\\n")

    success = test_ruff_config()

    if success:
        logger.debug("\\n✅ Ruff is working correctly! Phase 3 should now work.")
    else:
        logger.debug("\\n❌ Ruff still has issues. Need more debugging.")
