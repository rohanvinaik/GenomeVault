#!/usr/bin/env python3

"""Quick diagnostic script to check for import issues."""

import os
import sys
from genomevault.utils.logging import get_logger
logger = get_logger(__name__)


# Add the genomevault directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger.debug("Checking imports...")

# Check core imports
try:
    logger.info("✅ core.constants imported successfully")
except Exception as e:
    logger.error(f"❌ core.constants import failed: {e}")

try:
    logger.info("✅ core.config imported successfully")
except Exception as e:
    logger.error(f"❌ core.config import failed: {e}")

try:
    logger.info("✅ utils.logging imported successfully")
except Exception as e:
    logger.error(f"❌ utils.logging import failed: {e}")

try:
    logger.info("✅ blockchain.node imported successfully")
except Exception as e:
    logger.error(f"❌ blockchain.node import failed: {e}")

try:
    from genomevault.api.main import app

    logger.info("✅ api.main imported successfully")
    logger.debug(f"   App created: {app.title} v{app.version}")
except Exception as e:
    logger.error(f"❌ api.main import failed: {e}")
    import traceback

    traceback.print_exc()

logger.info("\nDiagnostic complete!")
