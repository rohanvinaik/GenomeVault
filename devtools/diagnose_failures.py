#!/usr/bin/env python3

"""Quick diagnostic to see what's failing"""

import logging
import sys
import traceback

logger = logging.getLogger(__name__)


logger.info("🔍 Quick Import Diagnostic")
logger.info("=" * 50)

# Test 1: Can we import torch?
logger.info("\n1. Testing torch import...")
try:
    import torch

    logger.info("✓ Torch version: {torch.__version__}")
except ImportError:
    logger.exception("Unhandled exception")
    logger.info("✗ Torch not installed: {e}")
    logger.info("  Install with: pip install torch")
    raise

# Test 2: Test the specific imports that were failing
logger.info("\n2. Testing specific failing imports...")

# PIR Client
logger.info("\n  Testing PIR Client...")
try:
    sys.path.insert(0, ".")

    logger.info("  ✓ PIR Client imported")
except Exception:
    logger.exception("Unhandled exception")
    logger.info("  ✗ PIR Client failed: {e}")
    # Try to see what's in the pir directory
    import os

    pir_path = "./pir"
    if os.path.exists(pir_path):
        logger.info("  PIR directory contents: {os.listdir(pir_path)}")
        client_path = os.path.join(pir_path, "client.py")
        if os.path.exists(client_path):
            logger.info("  client.py exists, checking first few lines...")
            with open(client_path) as f:
                lines = f.readlines()[:10]
                for i, line in enumerate(lines):
                    logger.info("    {i+1}: {line.rstrip()}")
    raise

# ZK Prover
logger.info("\n  Testing ZK Prover...")
try:
    logger.info("  ✓ ZK Prover imported")
except Exception:
    logger.exception("Unhandled exception")
    logger.info("  ✗ ZK Prover failed: {e}")
    traceback.print_exc()
    raise

# Test the config import pattern
logger.info("\n3. Testing config import pattern...")
try:
    from utils.config import get_config

    config = get_config()
    logger.info("  ✓ Config loaded: {type(config)}")
    logger.info("  ✓ Environment: {config.environment}")
except Exception:
    logger.exception("Unhandled exception")
    logger.info("  ✗ Config failed: {e}")
    traceback.print_exc()
    raise

# Test if we have the required dependencies
logger.info("\n4. Checking required dependencies...")
deps = ["numpy", "pydantic", "cryptography", "torch"]
for dep in deps:
    try:
        __import__(dep)
        logger.info("  ✓ {dep} installed")
    except ImportError:
        logger.exception("Unhandled exception")
        logger.info("  ✗ {dep} NOT installed")
        raise

logger.info("\n" + "=" * 50)
logger.info("Diagnostic complete!")
