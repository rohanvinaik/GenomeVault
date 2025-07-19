#!/usr/bin/env python3
"""Final comprehensive test to verify all imports are working"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test all critical imports"""

    print("🧪 GenomeVault Import Test Suite")
    print("=" * 50)

    tests = [
        # Core imports
        ("Core config", "from core.config import Config, get_config"),
        (
            "Core exceptions",
            "from core.exceptions import GenomeVaultError, EncodingError, BindingError, CircuitError",
        ),
        # Utils imports
        ("Utils logging", "from utils.logging import get_logger"),
        ("Utils encryption", "from utils.encryption import AESGCMCipher"),
        ("Utils hashing", "from utils.encryption import secure_hash"),
        # Local processing
        ("Sequencing", "from local_processing.sequencing import SequencingProcessor"),
        # Hypervector
        ("Hypervector encoding", "from hypervector_transform.encoding import HypervectorEncoder"),
        (
            "Hypervector binding",
            "from hypervector_transform.binding import circular_bind, HypervectorBinder",
        ),
        # ZK proofs
        ("ZK Prover", "from zk_proofs.prover import Prover"),
        ("ZK Prover alt", "from zk_proofs import ZKProver"),
        # PIR
        ("PIR Client", "from pir.client import PIRClient"),
        # Blockchain
        ("Blockchain node", "from blockchain.node import BlockchainNode"),
        # API - test create_app function exists
        ("API app function", "from api.app import app"),
    ]

    passed = 0
    failed = 0

    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✅ {name}: PASSED")
            passed += 1
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️  {name}: ERROR - {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = test_imports()

    if success:
        print("\n✅ All imports working! Running pytest...")
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_simple.py", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    else:
        print("\n❌ Some imports still failing. Fix these before running pytest.")
        sys.exit(1)
