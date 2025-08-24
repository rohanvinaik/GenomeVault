from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


#!/usr/bin/env python3
"""
Quick Ruff upgrade and configuration fix.
This upgrades Ruff to 0.4.4 and updates the configuration to use max-violations.
"""

import subprocess
import sys
from pathlib import Path


def upgrade_ruff():
    """Upgrade Ruff to version 0.4.4."""
    logger.debug("🔧 Upgrading Ruff to 0.4.4...")

    # Try pip first
    try:
        result = subprocess.run(
            ["pip", "install", "--upgrade", "ruff==0.4.4"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            logger.info("✅ Ruff upgraded successfully via pip")
            return True
        else:
            logger.error(f"pip failed: {result.stderr}")
    except Exception as e:
        logger.error(f"pip error: {e}")

    # Try pip3
    try:
        result = subprocess.run(
            ["pip3", "install", "--upgrade", "ruff==0.4.4"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            logger.info("✅ Ruff upgraded successfully via pip3")
            return True
        else:
            logger.error(f"pip3 failed: {result.stderr}")
    except Exception as e:
        logger.error(f"pip3 error: {e}")

    logger.debug("❌ Could not upgrade Ruff")
    return False


def update_ruff_config():
    """Update .ruff.toml to use max-violations."""
    logger.debug("📝 Updating Ruff configuration...")

    ruff_config = Path("./.ruff.toml")

    if not ruff_config.exists():
        logger.debug("❌ .ruff.toml not found")
        return False

    content = ruff_config.read_text()

    # Add [output] section with max-violations if not present
    if "[output]" not in content:
        new_content = "[output]\\nmax-violations = 200\\n\\n" + content
        ruff_config.write_text(new_content)
        logger.debug("✅ Added [output] section with max-violations = 200")
    else:
        logger.debug("✅ [output] section already exists")

    return True


def verify_setup():
    """Verify Ruff version and configuration."""
    logger.debug("🔍 Verifying setup...")

    # Check version
    try:
        result = subprocess.run(["ruff", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.debug(f"Ruff version: {version}")

            if "0.4" in version:
                logger.debug("✅ Ruff version is adequate")
            else:
                logger.debug("❌ Ruff version may still be old")
                return False
        else:
            logger.debug("❌ Could not get Ruff version")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking Ruff version: {e}")
        return False

    # Test configuration
    try:
        result = subprocess.run(
            ["ruff", "check", ".", "--quiet"],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=10,
        )

        logger.debug(f"Ruff config test exit code: {result.returncode}")
        if "unknown field" in result.stderr:
            logger.debug("❌ Configuration still has issues")
            return False
        else:
            logger.debug("✅ Ruff configuration is working")
            return True

    except Exception as e:
        logger.error(f"❌ Error testing Ruff config: {e}")
        return False


def main():
    logger.debug("🚀 Ruff Upgrade and Configuration Fix\\n")

    # Step 1: Upgrade Ruff
    if not upgrade_ruff():
        logger.error("\\n❌ Failed to upgrade Ruff. You may need to:")
        logger.debug("   1. Check your Python environment")
        logger.debug("   2. Try: pip install --upgrade ruff==0.4.4")
        logger.debug("   3. Or: conda install -c conda-forge ruff=0.4.4")
        sys.exit(1)

    # Step 2: Update configuration
    if not update_ruff_config():
        logger.error("\\n❌ Failed to update configuration")
        sys.exit(1)

    # Step 3: Verify everything works
    if not verify_setup():
        logger.error("\\n❌ Setup verification failed")
        sys.exit(1)

    logger.info("\\n🎉 Success! Ruff is now ready for Phase 3 F821 fixing.")
    logger.debug("\\nYou can now run:")
    logger.debug("   python enhanced_cleanup.py --phase 3")


if __name__ == "__main__":
    main()
