#!/usr/bin/env python3
"""Final comprehensive test for GenomeVault imports and basic functionality"""

import subprocess
import sys


def test_everything():
    """Run all tests and collect results"""

    print("🚀 GenomeVault Comprehensive Test")
    print("=" * 60)

    results = []

    # Test 1: Basic imports
    print("\n1️⃣ Testing basic imports...")
    try:
        from core.config import Config as CoreConfig
        from core.config import get_config as core_get_config
        from core.constants import CompressionTier, NodeClassWeight
        from core.exceptions import BindingError, EncodingError, GenomeVaultError

        print("✅ Core imports successful")
        results.append(("Core imports", True, None))
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        results.append(("Core imports", False, str(e)))

    # Test 2: Utils imports
    print("\n2️⃣ Testing utils imports...")
    try:
        from utils.config import Config as UtilsConfig
        from utils.config import get_config
        from utils.encryption import AESGCMCipher, secure_hash
        from utils.logging import get_logger

        # Test config structure
        config = get_config()
        assert hasattr(config, "storage"), "Config missing storage attribute"
        assert hasattr(config, "crypto"), "Config missing crypto attribute"
        assert hasattr(config, "privacy"), "Config missing privacy attribute"

        print("✅ Utils imports successful")
        results.append(("Utils imports", True, None))
    except Exception as e:
        print(f"❌ Utils imports failed: {e}")
        results.append(("Utils imports", False, str(e)))

    # Test 3: Local processing
    print("\n3️⃣ Testing local processing...")
    try:
        # Ensure config is loaded
        from utils.config import get_config

        config = get_config()

        # Create necessary directories
        config.storage.data_dir.mkdir(parents=True, exist_ok=True)
        (config.storage.data_dir / "references").mkdir(parents=True, exist_ok=True)

        from local_processing.sequencing import SequencingProcessor

        processor = SequencingProcessor()

        print("✅ Local processing imports successful")
        results.append(("Local processing", True, None))
    except Exception as e:
        print(f"❌ Local processing failed: {e}")
        results.append(("Local processing", False, str(e)))

    # Test 4: Hypervector
    print("\n4️⃣ Testing hypervector functionality...")
    try:
        import torch

        from hypervector_transform.binding import HypervectorBinder, circular_bind
        from hypervector_transform.encoding import HypervectorConfig, HypervectorEncoder

        # Create encoder
        hv_config = HypervectorConfig(dimension=1000)
        encoder = HypervectorEncoder(hv_config)

        # Test binding
        v1 = torch.randn(1000)
        v2 = torch.randn(1000)
        result = circular_bind([v1, v2])
        assert result.shape[0] == 1000

        print("✅ Hypervector functionality working")
        results.append(("Hypervector", True, None))
    except Exception as e:
        print(f"❌ Hypervector failed: {e}")
        results.append(("Hypervector", False, str(e)))

    # Test 5: ZK Proofs
    print("\n5️⃣ Testing ZK proofs...")
    try:
        from zk_proofs.prover import ZKProver

        # Just check it imports
        print("✅ ZK proofs import successful")
        results.append(("ZK Proofs", True, None))
    except Exception as e:
        print(f"❌ ZK proofs failed: {e}")
        results.append(("ZK Proofs", False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed

    for name, success, error in results:
        status = "✅ PASSED" if success else f"❌ FAILED: {error}"
        print(f"{name:20} {status}")

    print("=" * 60)
    print(f"Total: {passed} passed, {failed} failed")

    return failed == 0


def run_pytest():
    """Run pytest on simple tests"""
    print("\n\n🧪 Running pytest on tests/test_simple.py...")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_simple.py", "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


if __name__ == "__main__":
    all_tests_passed = test_everything()

    if all_tests_passed:
        print("\n✅ All import tests passed! Proceeding to pytest...")
        pytest_passed = run_pytest()
        sys.exit(0 if pytest_passed else 1)
    else:
        print("\n❌ Some tests failed. Fix these before running pytest.")
        sys.exit(1)
