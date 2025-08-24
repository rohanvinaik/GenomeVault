#!/usr/bin/env python3
"""Validate all GenomeVault module fixes."""

import sys
from typing import List, Tuple


def test_imports() -> List[Tuple[str, bool, str]]:
    """Test all required imports."""
    results = []

    # Test 1: HDC Config
    try:
        from genomevault.hdc import encode, bundle, similarity, HDCConfig, HDCEncoder

        results.append(("HDC basic imports", True, "Success"))
    except ImportError as e:
        results.append(("HDC basic imports", False, str(e)))

    # Test 2: Marketplace Enums
    try:
        from genomevault.marketplace import (
            AlgorithmStatus,
            PricingModel,
            RuntimeEnvironment,
            LicenseType,
        )

        results.append(("Marketplace enums", True, "Success"))
    except ImportError as e:
        results.append(("Marketplace enums", False, str(e)))

    # Test 3: PIR Protocol
    try:
        from genomevault.pir import PIRServer, PIRClient, PIRProtocol

        results.append(("PIR imports", True, "Success"))
    except ImportError as e:
        results.append(("PIR imports", False, str(e)))

    # Test 4: Federated Config
    try:
        from genomevault.federated import FederatedConfig

        results.append(("FederatedConfig", True, "Success"))
    except ImportError as e:
        results.append(("FederatedConfig", False, str(e)))

    # Test 5: TieredCompression
    try:
        from genomevault.compression import TieredCompression, CompressionTier

        results.append(("TieredCompression", True, "Success"))
    except ImportError as e:
        results.append(("TieredCompression", False, str(e)))

    # Test 6: WeightedVotingConsensus
    try:
        from genomevault.blockchain.consensus import WeightedVotingConsensus, Node, ResourceClass

        results.append(("WeightedVotingConsensus", True, "Success"))
    except ImportError as e:
        results.append(("WeightedVotingConsensus", False, str(e)))

    # Test 7: ThresholdService
    try:
        from genomevault.crypto import ThresholdService

        results.append(("ThresholdService", True, "Success"))
    except ImportError as e:
        results.append(("ThresholdService", False, str(e)))

    # Test 8: Additional HDC imports for config
    try:
        from genomevault.hypervector_transform.encoding import HypervectorConfig

        if hasattr(HypervectorConfig, "__init__"):
            # Check if similarity_threshold parameter exists
            config = HypervectorConfig(dimension=1000, similarity_threshold=0.85)
            results.append(("HypervectorConfig params", True, "Success"))
        else:
            results.append(("HypervectorConfig params", False, "No __init__ method"))
    except Exception as e:
        results.append(("HypervectorConfig params", False, str(e)))

    # Test 9: Check PIR equal-length records fix
    try:
        from genomevault.pir import PIRServer

        # Test with equal-length records
        records = [f"record_{i:03d}".encode() for i in range(10)]
        server = PIRServer(records)
        results.append(("PIR equal-length records", True, "Success"))
    except Exception as e:
        results.append(("PIR equal-length records", False, str(e)))

    # Test 10: Federated coordinator helper
    try:
        from genomevault.federated.coordinator import (
            create_coordinator_from_config,
            FederatedConfig,
        )

        config = FederatedConfig(min_participants=3)
        results.append(("Federated coordinator helper", True, "Success"))
    except ImportError as e:
        results.append(("Federated coordinator helper", False, str(e)))

    return results


def test_functionality() -> List[Tuple[str, bool, str]]:
    """Test basic functionality of fixed modules."""
    results = []

    # Test 1: HDC encoding
    try:
        from genomevault.hdc import HDCConfig, HDCEncoder
        import numpy as np

        config = HDCConfig(dimension=100, seed=42)
        encoder = HDCEncoder(config)
        data = np.array([1.0, 2.0, 3.0])
        encoded = encoder.encode(data)
        results.append(("HDC encoding", True, f"Encoded to shape {encoded.shape}"))
    except Exception as e:
        results.append(("HDC encoding", False, str(e)))

    # Test 2: Compression
    try:
        from genomevault.compression import TieredCompression, CompressionTier
        import numpy as np

        compressor = TieredCompression()
        data = np.random.randn(10, 10).astype(np.float32)
        compressed = compressor.compress(data, CompressionTier.MINI)
        results.append(("Compression", True, f"Compressed to {len(compressed)} bytes"))
    except Exception as e:
        results.append(("Compression", False, str(e)))

    # Test 3: Consensus
    try:
        from genomevault.blockchain.consensus import WeightedVotingConsensus, Node, ResourceClass

        consensus = WeightedVotingConsensus()
        node = Node("test_node", ResourceClass.FULL, stake=1000)
        consensus.add_node(node)
        results.append(("Consensus add_node", True, "Node added successfully"))
    except Exception as e:
        results.append(("Consensus add_node", False, str(e)))

    # Test 4: Threshold crypto
    try:
        from genomevault.crypto import ThresholdService

        service = ThresholdService(threshold=3, total_shares=5)
        shares = service.generate_distributed_key()
        results.append(("Threshold key generation", True, f"Generated {len(shares)} shares"))
    except Exception as e:
        results.append(("Threshold key generation", False, str(e)))

    return results


def main():
    """Run validation tests."""
    print("=" * 60)
    print("GenomeVault Module Validation")
    print("=" * 60)

    print("\n📦 Import Tests:")
    print("-" * 40)
    import_results = test_imports()

    import_passed = 0
    import_failed = 0

    for test_name, success, message in import_results:
        status = "✓" if success else "✗"
        print(f"{status} {test_name:<30} {message}")
        if success:
            import_passed += 1
        else:
            import_failed += 1

    print("\n🔧 Functionality Tests:")
    print("-" * 40)
    func_results = test_functionality()

    func_passed = 0
    func_failed = 0

    for test_name, success, message in func_results:
        status = "✓" if success else "✗"
        print(f"{status} {test_name:<30} {message}")
        if success:
            func_passed += 1
        else:
            func_failed += 1

    total_passed = import_passed + func_passed
    total_failed = import_failed + func_failed
    total_tests = total_passed + total_failed

    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Import Tests:       {import_passed}/{len(import_results)} passed")
    print(f"Functionality Tests: {func_passed}/{len(func_results)} passed")
    print(
        f"Overall:            {total_passed}/{total_tests} passed ({100*total_passed/total_tests:.1f}%)"
    )

    if total_failed == 0:
        print("\n✅ All validation tests passed!")
    else:
        print(f"\n⚠️  {total_failed} test(s) failed - review the issues above")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
