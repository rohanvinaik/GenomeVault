#!/usr/bin/env python3
"""Minimal test to verify GenomeVault is working"""

try:
    # Test basic imports
    from core.config import get_config

    print("✅ Core config imported")

    from utils import get_logger

    print("✅ Utils imported")

    print("✅ Sequencing module imported")

    print("✅ Hypervector module imported")

    print("✅ ZK proofs module imported")

    # Test creating instances
    config = get_config()
    print("✅ Config created: node_type = {config.node_type}")

    logger = get_logger(__name__)
    print("✅ Logger created")

    print("\n✅ All imports successful!")

except Exception:
    print("\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
    exit(1)
