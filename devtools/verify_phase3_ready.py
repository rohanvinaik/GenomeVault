from genomevault.utils.logging import get_logger
logger = get_logger(__name__)


#!/usr/bin/env python3
"""
Verify that Ruff is properly installed and Phase 3 is ready to run.
"""

import json
import subprocess
from pathlib import Path


def check_ruff_installation():
    """Check if Ruff is properly installed."""
    logger.debug("🔍 Checking Ruff installation...")

    # Check version
    try:
        result = subprocess.run(["ruff", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.debug(f"✅ Ruff version: {version}")

            if "0.4." in version or "0.5." in version:
                logger.debug("✅ Version is adequate for Phase 3")
                return True
            else:
                logger.debug("❌ Version is too old for Phase 3")
                return False
        else:
            logger.debug(f"❌ Could not get Ruff version: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking Ruff: {e}")
        return False


def test_f821_detection():
    """Test F821 detection with JSON output."""
    logger.debug("\\n🧪 Testing F821 detection...")

    try:
        result = subprocess.run(
            ["ruff", "check", ".", "--select", "F821", "--output-format", "json"],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=30,
        )

        logger.debug(f"F821 check exit code: {result.returncode}")

        if result.stdout:
            try:
                violations = json.loads(result.stdout)
                logger.debug(f"✅ Found {len(violations)} F821 violations")

                if violations:
                    logger.debug("\\nSample violations:")
                    for i, v in enumerate(violations[:3]):
                        file_short = v["filename"].replace("./", "")
                        logger.debug(f"  {i+1}. {file_short}:{v['location']['row']} - {v['message']}")

                return True
            except json.JSONDecodeError:
                logger.debug("❌ Could not parse JSON output")
                logger.debug(f"Raw output: {result.stdout[:200]}")
                return False
        else:
            logger.debug("✅ No F821 violations found")
            return True

    except Exception as e:
        logger.error(f"❌ Error testing F821 detection: {e}")
        return False


def test_config_file():
    """Test that .ruff.toml is working."""
    logger.debug("\\n📝 Testing .ruff.toml configuration...")

    ruff_config = Path("./.ruff.toml")

    if not ruff_config.exists():
        logger.debug("❌ .ruff.toml not found")
        return False

    content = ruff_config.read_text()
    logger.debug(f"✅ .ruff.toml exists ({len(content)} chars)")

    # Test that the config doesn't cause errors
    try:
        result = subprocess.run(
            ["ruff", "check", ".", "--quiet"],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=10,
        )

        if "unknown field" in (result.stderr or ""):
            logger.debug("❌ Configuration has unknown fields")
            logger.error(f"Error: {result.stderr}")
            return False
        else:
            logger.debug("✅ Configuration is valid")
            return True

    except Exception as e:
        logger.error(f"❌ Error testing configuration: {e}")
        return False


def test_phase3_readiness():
    """Test if Phase 3 is ready to run."""
    logger.debug("\\n🎯 Testing Phase 3 readiness...")

    try:
        result = subprocess.run(
            ["python", "enhanced_cleanup.py", "--phase", "3", "--dry-run"],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=15,
        )

        logger.debug(f"Phase 3 dry-run exit code: {result.returncode}")

        if result.returncode == 0:
            logger.info("✅ Phase 3 dry-run successful")
            return True
        else:
            logger.error("❌ Phase 3 dry-run failed")
            if result.stderr:
                logger.error(f"Error: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"❌ Error testing Phase 3: {e}")
        return False


def main():
    logger.debug("🚀 Verifying Ruff Installation and Phase 3 Readiness\\n")

    all_good = True

    # Test 1: Ruff installation
    if not check_ruff_installation():
        all_good = False

    # Test 2: F821 detection
    if not test_f821_detection():
        all_good = False

    # Test 3: Configuration
    if not test_config_file():
        all_good = False

    # Test 4: Phase 3 readiness
    if not test_phase3_readiness():
        all_good = False

    logger.debug("\\n" + "=" * 50)
    if all_good:
        logger.info("🎉 ALL TESTS PASSED! Phase 3 is ready to run.")
        logger.debug("\\nYou can now execute:")
        logger.debug("   python enhanced_cleanup.py --phase 3")
        logger.error("\\nThis should process and fix your F821 undefined name errors!")
    else:
        logger.error("❌ Some tests failed. Please check the issues above.")
        logger.debug("\\nYou may need to:")
        logger.debug("   1. Run: python proper_ruff_upgrade.py")
        logger.debug("   2. Manually install: python -m pip install ruff>=0.4.4")
        logger.debug("   3. Check your Python environment")


if __name__ == "__main__":
    main()
