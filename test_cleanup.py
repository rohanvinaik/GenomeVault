#!/usr/bin/env python3
"""Test to verify imports are working after cleanup"""

import sys

try:
    print("Testing core imports...")
    from core.config import Config, get_config
    print("✅ Core config imported")
    
    print("\nTesting utils imports...")
    from utils import get_logger
    print("✅ Utils imported")
    
    print("\nTesting local_processing imports...")
    from local_processing.sequencing import SequencingProcessor
    print("✅ Sequencing module imported")
    
    print("\nTesting hypervector imports...")
    from hypervector_transform.encoding import HypervectorEncoder
    from hypervector_transform.binding import HypervectorBinder
    print("✅ Hypervector modules imported")
    
    print("\nTesting hypervector_transform package import...")
    import hypervector_transform
    print("✅ Package imported")
    print(f"   Available: {[x for x in dir(hypervector_transform) if not x.startswith('_')]}")
    
    print("\nTesting zk_proofs imports...")
    from zk_proofs.prover import Prover
    print("✅ ZK proofs module imported")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
