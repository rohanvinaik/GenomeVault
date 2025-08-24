#!/usr/bin/env python3
"""
Verification script for PIR correctness.

Tests that XOR aggregation correctly recovers the original data.
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_xor_pir_correctness():
    """Test XOR-based PIR correctness."""
    print("\n" + "="*60)
    print("XOR PIR CORRECTNESS TEST")
    print("="*60)
    
    # Create sample dataset
    dataset_size = 10
    element_size = 64  # Smaller for testing
    target_index = 4
    
    # Create dataset with known values
    dataset = []
    for i in range(dataset_size):
        data = f"Record_{i:03d}".encode('utf-8')
        # Pad to fixed size
        padded = data + b'\x00' * (element_size - len(data))
        dataset.append(padded[:element_size])
    
    print(f"Dataset size: {dataset_size}")
    print(f"Element size: {element_size} bytes")
    print(f"Target index: {target_index}")
    expected_data = dataset[target_index].rstrip(b'\x00').decode()
    print(f"Expected data: {expected_data}")
    
    # IT-PIR protocol simulation
    print("\n1. Generating IT-PIR query vectors...")
    
    # Create unit vector for target
    unit_vector = np.zeros(dataset_size, dtype=np.uint8)
    unit_vector[target_index] = 1
    
    # Generate random vectors for 2 servers
    query1 = np.random.randint(0, 2, dataset_size, dtype=np.uint8)
    query2 = (unit_vector - query1) % 2  # Ensures query1 XOR query2 = unit_vector
    
    # Verify queries XOR to unit vector
    verification = (query1 + query2) % 2
    assert np.array_equal(verification, unit_vector), "Query generation failed"
    print(f"   Query 1: {query1}")
    print(f"   Query 2: {query2}")
    print(f"   XOR:     {verification}")
    print("   ✓ Queries XOR to unit vector")
    
    # Simulate server responses
    print("\n2. Simulating server responses...")
    
    def server_response(query_vector, database):
        """Simulate PIR server response."""
        result = bytearray(element_size)
        for i, bit in enumerate(query_vector):
            if bit == 1:
                # XOR with selected record
                for j in range(element_size):
                    result[j] ^= database[i][j]
        return bytes(result)
    
    response1 = server_response(query1, dataset)
    response2 = server_response(query2, dataset)
    
    print(f"   Server 1 response: {len(response1)} bytes")
    print(f"   Server 2 response: {len(response2)} bytes")
    
    # XOR responses to recover data
    print("\n3. Aggregating responses...")
    recovered = bytearray(element_size)
    for i in range(element_size):
        recovered[i] = response1[i] ^ response2[i]
    
    recovered_data = bytes(recovered).rstrip(b'\x00')
    print(f"   Recovered: {recovered_data.decode()}")
    
    # Verify correctness
    expected = dataset[target_index].rstrip(b'\x00')
    if recovered_data == expected:
        print("   ✓ Correctly recovered target record!")
        return True
    else:
        print(f"   ✗ Mismatch!")
        print(f"     Expected: {expected}")
        print(f"     Got: {recovered_data}")
        return False


def test_multi_server_pir():
    """Test PIR with 3 servers."""
    print("\n" + "="*60)
    print("3-SERVER PIR TEST")
    print("="*60)
    
    dataset_size = 5
    target_index = 2
    
    # Simple dataset
    dataset = [f"Data_{i}".encode() for i in range(dataset_size)]
    max_len = max(len(d) for d in dataset)
    dataset = [d.ljust(max_len, b'\x00') for d in dataset]
    
    target_data = dataset[target_index].rstrip(b'\x00').decode()
    print(f"Target: index {target_index} = '{target_data}'")
    
    # Generate 3 query vectors
    unit_vector = np.zeros(dataset_size, dtype=np.uint8)
    unit_vector[target_index] = 1
    
    # Random vectors for servers 1 and 2
    q1 = np.random.randint(0, 2, dataset_size, dtype=np.uint8)
    q2 = np.random.randint(0, 2, dataset_size, dtype=np.uint8)
    
    # Server 3 query ensures XOR = unit vector
    q3 = (unit_vector - q1 - q2) % 2
    
    print(f"\nQuery vectors:")
    print(f"  Q1: {q1}")
    print(f"  Q2: {q2}")
    print(f"  Q3: {q3}")
    print(f"  XOR: {(q1 + q2 + q3) % 2}")
    
    # Server responses
    def xor_response(query, data):
        result = bytearray(max_len)
        for i, select in enumerate(query):
            if select:
                for j in range(max_len):
                    result[j] ^= data[i][j]
        return bytes(result)
    
    r1 = xor_response(q1, dataset)
    r2 = xor_response(q2, dataset)
    r3 = xor_response(q3, dataset)
    
    # XOR all responses
    final = bytearray(max_len)
    for i in range(max_len):
        final[i] = r1[i] ^ r2[i] ^ r3[i]
    
    recovered = bytes(final).rstrip(b'\x00').decode()
    expected = dataset[target_index].rstrip(b'\x00').decode()
    
    print(f"\nResult:")
    print(f"  Expected: '{expected}'")
    print(f"  Recovered: '{recovered}'")
    
    if recovered == expected:
        print("  ✓ Success!")
        return True
    else:
        print("  ✗ Failed!")
        return False


if __name__ == "__main__":
    print("\n🔬 PIR Protocol Verification Tests")
    
    # Run tests
    test1 = test_xor_pir_correctness()
    test2 = test_multi_server_pir()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"XOR PIR Correctness: {'✓ PASSED' if test1 else '✗ FAILED'}")
    print(f"3-Server PIR: {'✓ PASSED' if test2 else '✗ FAILED'}")
    
    if test1 and test2:
        print("\n✅ All verification tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)