#!/usr/bin/env python3
"""Minimal test to verify GenomeVault is working"""

try:
    # Test basic imports
    from core.config import Config, get_config

    print("✅ Core config imported")

    from utils import get_logger

    print("✅ Utils imported")

    from local_processing.sequencing import SequencingProcessor

    print("✅ Sequencing module imported")

    from hypervector_transform.encoding import HypervectorEncoder

    print("✅ Hypervector module imported")

    from zk_proofs.prover import Prover

    print("✅ ZK proofs module imported")

    # Test creating instances
    config = get_config()
    print(f"✅ Config created: node_type={config.node_type}")

    logger = get_logger(__name__)
    print("✅ Logger created")

    print("\n✅ All imports successful!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
    exit(1)
