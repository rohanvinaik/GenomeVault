#!/usr/bin/env python3
"""Test PIR shard health monitoring and auto-ejection."""

import random
import numpy as np

# Add genomevault to path
import sys

sys.path.insert(0, "/Users/rohanvinaik/genomevault")

from genomevault.pir.servers import ShardHealth, ShardManager, ShardedPIRServer, FECEncoder


def test_shard_health():
    """Test shard health tracking."""
    print("=" * 60)
    print("Testing Shard Health Tracking")
    print("=" * 60)

    # Create a shard health tracker
    shard = ShardHealth(shard_id="test_shard_1")

    # Record some successful responses
    for i in range(10):
        response_time = random.uniform(0.1, 0.5)  # 100-500ms
        shard.record_response(response_time)

    print("After 10 successful responses:")
    print(f"  Health Score: {shard.health_score():.2f}")
    print(f"  P95 Latency: {shard.get_p95_latency():.3f}s")
    print(f"  Is Healthy: {shard.is_healthy}")

    # Record some errors
    shard.record_error()
    shard.record_error()
    print("\nAfter 2 errors:")
    print(f"  Health Score: {shard.health_score():.2f}")
    print(f"  Consecutive Failures: {shard.consecutive_failures}")
    print(f"  Is Healthy: {shard.is_healthy}")

    # Third error should trigger ejection
    shard.record_error()
    print("\nAfter 3rd consecutive error (auto-ejection):")
    print(f"  Health Score: {shard.health_score():.2f}")
    print(f"  Is Healthy: {shard.is_healthy}")

    assert not shard.is_healthy, "Shard should be ejected after 3 failures"
    print("✅ Shard health tracking works correctly")


def test_shard_manager():
    """Test shard manager with multiple shards."""
    print("\n" + "=" * 60)
    print("Testing Shard Manager")
    print("=" * 60)

    manager = ShardManager(min_shards=2)

    # Add 5 shards
    for i in range(5):
        manager.add_shard(f"shard_{i}")

    print(f"Initial state: {len(manager.get_healthy_shards())} healthy shards")

    # Simulate different health conditions
    # Shard 0: Very healthy (fast responses)
    for _ in range(20):
        manager.record_shard_response("shard_0", random.uniform(0.05, 0.1))

    # Shard 1: Somewhat healthy (normal responses)
    for _ in range(15):
        manager.record_shard_response("shard_1", random.uniform(0.2, 0.5))

    # Shard 2: Mixed (some errors)
    for _ in range(10):
        manager.record_shard_response("shard_2", random.uniform(0.5, 1.0))
    manager.record_shard_error("shard_2")
    manager.record_shard_error("shard_2")

    # Shard 3: Unhealthy (will be ejected)
    manager.record_shard_error("shard_3")
    manager.record_shard_error("shard_3")
    manager.record_shard_error("shard_3")  # Auto-ejected

    # Shard 4: Very unhealthy (will be ejected)
    for _ in range(4):
        manager.record_shard_error("shard_4")  # Auto-ejected

    # Check status
    status = manager.get_shard_status()
    print("\nShard Status:")
    for shard_id, info in status.items():
        print(f"  {shard_id}:")
        print(f"    Healthy: {info['is_healthy']}")
        print(f"    Score: {info['health_score']:.2f}")
        print(f"    Errors: {info['error_count']}")
        print(f"    P95: {info['p95_latency']:.3f}s" if info["p95_latency"] else "    P95: N/A")

    # Test shard selection
    selected = manager.select_shards(3)
    print(f"\nTop 3 shards selected: {selected}")
    assert "shard_0" in selected, "Best shard should be selected"
    assert "shard_3" not in selected, "Ejected shard should not be selected"

    # Test minimum shard recovery
    print("\nTesting minimum shard recovery...")
    healthy_before = len(manager.get_healthy_shards())

    # Eject more shards to trigger recovery
    for _ in range(3):
        manager.record_shard_error("shard_0")
    for _ in range(3):
        manager.record_shard_error("shard_1")

    # Should trigger recovery since we're below minimum
    healthy_after = len(manager.get_healthy_shards())
    print(f"  Healthy shards: {healthy_before} -> {healthy_after}")
    assert healthy_after >= manager.min_shards, "Should maintain minimum shards"

    print("✅ Shard manager works correctly")


def test_sharded_pir_server():
    """Test sharded PIR server with health monitoring."""
    print("\n" + "=" * 60)
    print("Testing Sharded PIR Server")
    print("=" * 60)

    # Create test database
    num_records = 100
    record_len = 32
    db = [bytes([i % 256] * record_len) for i in range(num_records)]

    # Create sharded server
    server = ShardedPIRServer(db, num_shards=5, use_fec=True)

    print(f"Created server with {server.num_shards} shards")
    print(f"FEC enabled: {server.fec is not None and server.fec.available}")

    # Test normal query
    mask = np.zeros(num_records, dtype=np.uint8)
    mask[42] = 1  # Query for record 42

    result = server.answer_with_sharding(mask)
    expected = db[42]
    assert result == expected, "Query result should match expected"
    print("✅ Normal query successful")

    # Simulate shard failures
    print("\nSimulating shard failures...")

    # Fail shard_2
    for _ in range(3):
        server.shard_manager.record_shard_error("shard_2")

    # Fail shard_3
    for _ in range(3):
        server.shard_manager.record_shard_error("shard_3")

    # Query should still work with remaining shards
    result = server.answer_with_sharding(mask)
    assert result == expected, "Query should work with degraded shards"
    print("✅ Query successful with 2 shards failed")

    # Get health report
    report = server.get_health_report()
    print("\nHealth Report:")
    print(f"  Healthy Shards: {report['healthy_shards']}/{report['total_shards']}")
    print(f"  FEC Enabled: {report['fec_enabled']}")

    # Show individual shard status
    print("\nIndividual Shard Status:")
    for shard_id, status in report["shard_status"].items():
        health_indicator = "✅" if status["is_healthy"] else "❌"
        print(
            f"  {health_indicator} {shard_id}: score={status['health_score']:.2f}, errors={status['error_count']}"
        )

    print("\n✅ Sharded PIR server works correctly")


def test_fec_encoder():
    """Test Forward Error Correction encoder."""
    print("\n" + "=" * 60)
    print("Testing Forward Error Correction")
    print("=" * 60)

    # Create FEC encoder
    fec = FECEncoder(k=3, m=2)  # 3 data fragments, 2 parity

    if not fec.available:
        print("⚠️  FEC not available (pyeclib not installed)")
        print("   Install with: pip install pyeclib")
        return

    # Test data
    original_data = b"This is test data for FEC encoding!"

    # Encode
    fragments = fec.encode(original_data)
    print(f"Encoded into {len(fragments)} fragments")

    # Decode with all fragments
    decoded = fec.decode(fragments)
    assert decoded == original_data, "Decoded data should match original"
    print("✅ Decoding with all fragments successful")

    # Decode with some fragments missing (simulate failures)
    available_fragments = fragments[:3]  # Only k fragments needed
    decoded = fec.decode(available_fragments)
    assert decoded == original_data, "Should decode with k fragments"
    print("✅ Decoding with minimum fragments successful")

    print("\n✅ FEC encoder works correctly")


def run_all_tests():
    """Run all PIR sharding tests."""
    print("🧬 GenomeVault PIR Sharding Tests")
    print("=" * 60)

    try:
        test_shard_health()
        test_shard_manager()
        test_sharded_pir_server()
        test_fec_encoder()

        print("\n" + "=" * 60)
        print("✅ ALL PIR SHARDING TESTS PASSED")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
