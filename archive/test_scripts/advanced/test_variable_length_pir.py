#!/usr/bin/env python3
"""Test PIR with variable length records."""

import sys
import json
import numpy as np

sys.path.insert(0, ".")


def test_variable_length_records():
    """Test PIR with variable length records."""
    from genomevault.pir.variable_length_engine import VariableLengthPIREngine

    print("Creating PIR engine...")
    engine = VariableLengthPIREngine()

    # Create variable length records of different types
    records = [
        b"short",
        b"medium length record",
        b"this is a much longer record with more data and details",
        b"x" * 100,  # 100 byte record
        "string record",  # String type
        {"type": "variant", "chr": "chr1", "pos": 12345},  # Dict type
        12345,  # Integer type
        3.14159,  # Float type
        b"another short one",
        "final record with unicode: åäö",
    ]

    print(f"Testing with {len(records)} variable length records")

    # Prepare database
    try:
        db, lengths = engine.prepare_database(records)
        print("✅ Database prepared successfully")
        print(f"   Original lengths: {lengths}")
        print(f"   Padded shape: {db.shape}")

        # Verify all records have same length after padding
        assert len(set(len(row) for row in db)) == 1, "Records not uniformly padded"
        print("✅ All records uniformly padded")

        # Get statistics
        stats = engine.get_stats(db, lengths)
        print("✅ Database stats:")
        print(f"   Records: {stats['num_records']}")
        print(f"   Padded size: {stats['padded_record_size']} bytes")
        print(
            f"   Original size range: {stats['original_sizes']['min']}-{stats['original_sizes']['max']} bytes"
        )
        print(f"   Padding overhead: {stats['padding_efficiency']['overhead_ratio']:.1%}")
        print(f"   Memory usage: {stats['memory_usage_mb']:.2f} MB")

    except Exception as e:
        print(f"❌ Database preparation failed: {e}")
        return False

    # Query and verify each record
    print("Testing record retrieval...")
    for i, original in enumerate(records):
        try:
            retrieved = engine.query(db, i)

            # Convert original to bytes for comparison
            if isinstance(original, str):
                expected = original.encode("utf-8")
            elif isinstance(original, bytes):
                expected = original
            elif isinstance(original, dict):
                expected = json.dumps(original, sort_keys=True).encode("utf-8")
            elif isinstance(original, (int, float)):
                expected = str(original).encode("utf-8")
            else:
                expected = str(original).encode("utf-8")

            if retrieved == expected:
                print(f"✅ Record {i}: {len(retrieved)} bytes retrieved correctly")
            else:
                print(f"❌ Record {i} mismatch:")
                print(f"   Expected: {expected[:50]}...")
                print(f"   Retrieved: {retrieved[:50]}...")
                return False

        except Exception as e:
            print(f"❌ Record {i} retrieval failed: {e}")
            return False

    print("✅ All records retrieved correctly")
    return True


def test_enhanced_pir_server():
    """Test enhanced PIR server with variable length records."""
    from genomevault.pir.variable_length_engine import EnhancedPIRServer

    print("\nTesting Enhanced PIR Server...")

    # Create test records
    records = [
        "genomic variant: chr1:12345:A>G",
        b"binary data here",
        {"sample_id": "S001", "variants": ["rs123", "rs456"]},
        "short",
        "x" * 200,  # Longer record
    ]

    try:
        # Create enhanced server
        server = EnhancedPIRServer(records, max_record_length=1024)
        print("✅ Enhanced PIR server created")

        # Get stats
        stats = server.get_database_stats()
        print(f"✅ Server stats: {stats['num_records']} records, {stats['memory_usage_mb']:.2f} MB")

        # Test queries
        for i in range(len(records)):
            mask = np.zeros(len(records), dtype=np.uint8)
            mask[i] = 1

            result = server.answer(mask)
            print(f"✅ Query {i}: Retrieved {len(result)} bytes")

            # Verify content
            if isinstance(records[i], str):
                expected = records[i].encode("utf-8")
            elif isinstance(records[i], bytes):
                expected = records[i]
            elif isinstance(records[i], dict):
                expected = json.dumps(records[i], sort_keys=True).encode("utf-8")
            else:
                expected = str(records[i]).encode("utf-8")

            assert result == expected, f"Query {i} mismatch"

        print("✅ All Enhanced PIR server queries successful")
        return True

    except Exception as e:
        print(f"❌ Enhanced PIR server test failed: {e}")
        return False


def test_validation():
    """Test database validation."""
    from genomevault.pir.variable_length_engine import VariableLengthPIREngine

    print("\nTesting validation...")
    engine = VariableLengthPIREngine(max_record_length=100)

    # Test valid database
    valid_records = ["short", "medium", "longer record"]
    is_valid, msg = engine.validate_database(valid_records)
    assert is_valid, f"Valid database rejected: {msg}"
    print("✅ Valid database accepted")

    # Test empty database
    is_valid, msg = engine.validate_database([])
    assert not is_valid, "Empty database should be invalid"
    print("✅ Empty database correctly rejected")

    # Test oversized record
    oversized_records = ["x" * 200]  # Exceeds max_record_length=100
    is_valid, msg = engine.validate_database(oversized_records)
    assert not is_valid, "Oversized record should be invalid"
    print("✅ Oversized record correctly rejected")

    print("✅ All validation tests passed")
    return True


def test_integration_with_existing():
    """Test integration with existing PIR infrastructure."""
    print("\nTesting integration with existing PIR...")

    try:
        from genomevault.pir.variable_length_engine import VariableLengthPIREngine
        from genomevault.pir.servers import PIRServer
        import numpy as np

        engine = VariableLengthPIREngine()

        # Create test data
        records = ["record1", "longer record 2", "short"]
        db, lengths = engine.prepare_database(records)

        # Convert to bytes for PIRServer
        byte_records = [row.tobytes() for row in db]

        # Create PIRServer with uniform length records
        server = PIRServer(byte_records)

        # Test query
        mask = np.zeros(len(records), dtype=np.uint8)
        mask[1] = 1  # Query second record

        result = server.answer(mask)
        unpadded = engine._unpad_record(result)

        expected = "longer record 2".encode("utf-8")
        assert unpadded == expected, f"Integration test failed: {unpadded} != {expected}"

        print("✅ Integration with existing PIR infrastructure successful")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all PIR variable length tests."""
    print("=" * 60)
    print("🔍 TESTING PIR VARIABLE LENGTH RECORDS")
    print("=" * 60)

    tests = [
        ("Basic Variable Length Records", test_variable_length_records),
        ("Enhanced PIR Server", test_enhanced_pir_server),
        ("Database Validation", test_validation),
        ("Integration with Existing PIR", test_integration_with_existing),
    ]

    passed = 0
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")

    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{len(tests)} PASSED")
    print("=" * 60)

    if passed == len(tests):
        print("🎉 ALL TESTS PASSED - PIR Variable Length Records Fixed!")
    else:
        print("⚠️  Some tests failed - check output above")

    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
