from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


#!/usr/bin/env python3
"""
Simple test to verify GenomeVault imports
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_basic_imports():
    """Test basic imports work"""
    logger.debug("Testing basic imports...")

    try:
        # Test core config
        from core.config import get_config

        logger.debug("✅ core.config imported successfully")

        # Test creating config
        get_config()
        logger.debug("✅ Config created: node_type={config.node_type}")

        # Test exceptions

        logger.debug("✅ core.exceptions imported successfully")

        # Test utils

        logger.debug("✅ utils imported successfully")

        return True

    except Exception:
        logger.exception("Unhandled exception")
        logger.debug("❌ Import failed: {e}")
        import traceback

        traceback.print_exc()
        return False
        raise


def test_module_imports():
    """Test module imports"""
    logger.debug("\nTesting module imports...")

    modules = [
        "local_processing.sequencing",
        "local_processing.transcriptomics",
        "local_processing.epigenetics",
        "hypervector_transform.encoding",
        "zk_proofs.prover",
    ]

    success = True
    for module in modules:
        try:
            module.split(".")
            exec("from {parts[0]} import {parts[1]}")
            logger.debug("✅ {module} imported successfully")
        except Exception:
            logger.exception("Unhandled exception")
            logger.debug("❌ {module} failed: {e}")
            success = False
            raise

    return success


if __name__ == "__main__":
    logger.debug("=" * 50)
    logger.debug("GenomeVault Import Test")
    logger.debug("=" * 50)

    basic_ok = test_basic_imports()
    modules_ok = test_module_imports()

    if basic_ok and modules_ok:
        logger.debug("\n✅ All imports successful!")
        sys.exit(0)
    else:
        logger.debug("\n❌ Some imports failed")
        sys.exit(1)
