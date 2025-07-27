#!/usr/bin/env python3
"""Quick test to identify the exact import issue"""

import sys
import traceback
import logging

logger = logging.getLogger(__name__)


logger.info("🔍 GenomeVault Import Diagnostic")
logger.info("=" * 50)

# Test 1: Basic package structure
logger.info("\n1. Testing basic package structure...")
try:
    import genomevault

    logger.info("✓ genomevault package exists")
except Exception as e:
    logger.info("✗ genomevault package error: {e}")

# Test 2: Core config
logger.info("\n2. Testing core.config...")
try:
    from core.config import Config, get_config

    logger.info("✓ core.config imports work")
except Exception as e:
    logger.info("✗ core.config error: {e}")

# Test 3: Utils
logger.info("\n3. Testing utils...")
try:
    from utils.logging import get_logger

    logger.info("✓ utils.logging works")
except Exception as e:
    logger.info("✗ utils.logging error: {e}")

try:
    from utils.encryption import AESGCMCipher

    logger.info("✓ utils.encryption works")
except Exception as e:
    logger.info("✗ utils.encryption error: {e}")

# Test 4: Hypervector - step by step
logger.info("\n4. Testing hypervector_transform step by step...")

# 4a: Can we import the package?
try:
    import hypervector_transform

    logger.info("✓ hypervector_transform package imports")
except Exception as e:
    logger.info("✗ hypervector_transform package error: {e}")
    traceback.print_exc()

# 4b: Can we import from binding directly?
try:
    from hypervector_transform.binding import circular_bind

    logger.info("✓ circular_bind imports from binding.py")
except Exception as e:
    logger.info("✗ binding.py error: {e}")

# 4c: What about the __init__.py imports?
try:
    from hypervector_transform import circular_bind

    logger.info("✓ circular_bind imports from __init__.py")
except Exception as e:
    logger.info("✗ __init__.py re-export error: {e}")

# 4d: Check encoding
try:
    from hypervector_transform.encoding import HypervectorEncoder

    logger.info("✓ HypervectorEncoder imports correctly")
except Exception as e:
    logger.info("✗ encoding.py error: {e}")
    traceback.print_exc()

# Test 5: The specific import that was failing
logger.info("\n5. Testing the specific failing import...")
try:
    from hypervector_transform import HypervectorEncoder

    logger.info("✓ HypervectorEncoder imports from package")
except Exception as e:
    logger.info("✗ Package-level import error: {e}")

logger.info("\n" + "=" * 50)
logger.info("Diagnostic complete!")
