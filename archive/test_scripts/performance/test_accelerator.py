#!/usr/bin/env python3
"""
Integration tests for GenomeVault accelerator.
"""

import numpy as np
import sys
import unittest

sys.path.insert(0, ".")

from genomevault.accelerator import Accelerator


class TestAccelerator(unittest.TestCase):
    """Test accelerator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.python_accel = Accelerator(force_python=True)
        try:
            self.rust_accel = Accelerator(force_python=False)
            self.has_rust = self.rust_accel.use_rust
        except:
            self.rust_accel = None
            self.has_rust = False

    def test_hypervector_similarity(self):
        """Test hypervector similarity computation."""
        dim = 1000
        vec1 = np.random.randn(dim).astype(np.float32)
        vec2 = np.random.randn(dim).astype(np.float32)

        # Python result
        py_result = self.python_accel.hypervector_similarity(vec1, vec2)

        # Rust result if available
        if self.has_rust:
            rust_result = self.rust_accel.hypervector_similarity(vec1, vec2)
            self.assertAlmostEqual(py_result, rust_result, places=5)

        # Test with identical vectors
        self_sim = self.python_accel.hypervector_similarity(vec1, vec1)
        self.assertAlmostEqual(self_sim, 1.0, places=5)

        # Test with orthogonal vectors
        vec3 = np.zeros(dim, dtype=np.float32)
        vec3[0] = 1.0
        vec4 = np.zeros(dim, dtype=np.float32)
        vec4[1] = 1.0
        ortho_sim = self.python_accel.hypervector_similarity(vec3, vec4)
        self.assertAlmostEqual(ortho_sim, 0.0, places=5)

    def test_batch_hypervector_similarity(self):
        """Test batch hypervector similarity."""
        n_vectors = 10
        dim = 100
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        query = np.random.randn(dim).astype(np.float32)

        # Python result
        py_result = self.python_accel.batch_hypervector_similarity(vectors, query)
        self.assertEqual(len(py_result), n_vectors)

        # Rust result if available
        if self.has_rust:
            rust_result = self.rust_accel.batch_hypervector_similarity(vectors, query)
            np.testing.assert_allclose(py_result, rust_result, rtol=1e-5)

    def test_pir_xor_mask(self):
        """Test PIR XOR mask operation."""
        size = 1000
        data = np.random.randint(0, 256, size, dtype=np.uint8)
        mask = np.random.randint(0, 256, size, dtype=np.uint8)

        # Python result
        py_result = self.python_accel.pir_xor_mask(data, mask)

        # Verify XOR properties
        # XOR with same mask twice should give original
        double_xor = self.python_accel.pir_xor_mask(py_result, mask)
        np.testing.assert_array_equal(data, double_xor)

        # Rust result if available
        if self.has_rust:
            rust_result = self.rust_accel.pir_xor_mask(data, mask)
            np.testing.assert_array_equal(py_result, rust_result)

    def test_batch_pir_query(self):
        """Test batch PIR query processing."""
        n_records = 50
        record_len = 100
        database = np.random.randint(0, 256, (n_records, record_len), dtype=np.uint8)

        # Query for single record
        query_mask = np.zeros(n_records, dtype=np.uint8)
        query_mask[10] = 1  # Select record 10

        # Python result
        py_result = self.python_accel.batch_pir_query(database, query_mask)
        np.testing.assert_array_equal(py_result, database[10])

        # Rust result if available
        if self.has_rust:
            rust_result = self.rust_accel.batch_pir_query(database, query_mask)
            np.testing.assert_array_equal(py_result, rust_result)

    def test_hamming_distance(self):
        """Test Hamming distance computation."""
        size = 1000
        vec1 = np.random.randint(0, 256, size, dtype=np.uint8)
        vec2 = np.random.randint(0, 256, size, dtype=np.uint8)

        # Python result
        py_result = self.python_accel.hamming_distance(vec1, vec2)

        # Distance with self should be 0
        self_dist = self.python_accel.hamming_distance(vec1, vec1)
        self.assertEqual(self_dist, 0)

        # Distance with complement should be maximal
        complement = 255 - vec1
        max_dist = self.python_accel.hamming_distance(vec1, complement)
        self.assertGreater(max_dist, 0)

        # Rust result if available
        if self.has_rust:
            rust_result = self.rust_accel.hamming_distance(vec1, vec2)
            self.assertEqual(py_result, rust_result)

    def test_encode_variant(self):
        """Test variant encoding."""
        dim = 1000

        # Encode same variant multiple times
        encodings = []
        for _ in range(3):
            vec = self.python_accel.encode_variant(
                chromosome=1, position=12345, ref_allele="A", alt_allele="G", dimension=dim
            )
            encodings.append(vec)

        # Should be deterministic
        for i in range(1, len(encodings)):
            np.testing.assert_array_almost_equal(encodings[0], encodings[i])

        # Should be normalized
        norm = np.linalg.norm(encodings[0])
        self.assertAlmostEqual(norm, 1.0, places=5)

        # Different variants should have different encodings
        vec2 = self.python_accel.encode_variant(
            chromosome=2, position=67890, ref_allele="C", alt_allele="T", dimension=dim
        )

        similarity = self.python_accel.hypervector_similarity(encodings[0], vec2)
        self.assertLess(abs(similarity), 0.5)  # Should be somewhat orthogonal

    def test_compression_decompression(self):
        """Test hypervector compression and decompression."""
        dim = 10000
        vector = np.random.randn(dim).astype(np.float32)

        # Compress
        compressed = self.python_accel.compress_hypervector(vector)
        expected_size = (dim + 7) // 8
        self.assertEqual(len(compressed), expected_size)

        # Decompress
        decompressed = self.python_accel.decompress_hypervector(compressed, dim)
        self.assertEqual(len(decompressed), dim)

        # Check binary preservation (sign)
        for i in range(dim):
            if vector[i] > 0:
                self.assertGreater(decompressed[i], 0)
            else:
                self.assertLess(decompressed[i], 0)

        # Rust comparison if available
        if self.has_rust:
            rust_compressed = self.rust_accel.compress_hypervector(vector)
            rust_decompressed = self.rust_accel.decompress_hypervector(rust_compressed, dim)
            np.testing.assert_array_almost_equal(decompressed, rust_decompressed)

    def test_knn_search(self):
        """Test k-nearest neighbors search."""
        n_vectors = 100
        dim = 100
        k = 5

        database = np.random.randn(n_vectors, dim).astype(np.float32)
        query = np.random.randn(dim).astype(np.float32)

        # Python result
        py_indices, py_distances = self.python_accel.knn_search(database, query, k)

        self.assertEqual(len(py_indices), k)
        self.assertEqual(len(py_distances), k)

        # Distances should be sorted (descending for similarity)
        for i in range(1, k):
            self.assertLessEqual(py_distances[i], py_distances[i - 1])

        # Rust result if available
        if self.has_rust:
            rust_indices, rust_distances = self.rust_accel.knn_search(database, query, k)
            np.testing.assert_array_equal(py_indices, rust_indices)
            np.testing.assert_allclose(py_distances, rust_distances, rtol=1e-5)

    def test_consistency_across_implementations(self):
        """Test that Python and Rust implementations give consistent results."""
        if not self.has_rust:
            self.skipTest("Rust accelerator not available")

        # Generate test data
        dim = 1000
        vec1 = np.random.randn(dim).astype(np.float32)
        vec2 = np.random.randn(dim).astype(np.float32)
        binary1 = np.random.randint(0, 256, 1000, dtype=np.uint8)
        binary2 = np.random.randint(0, 256, 1000, dtype=np.uint8)

        # Test all operations
        tests = [
            ("hypervector_similarity", lambda a: a.hypervector_similarity(vec1, vec2)),
            ("hamming_distance", lambda a: a.hamming_distance(binary1, binary2)),
            ("pir_xor_mask", lambda a: a.pir_xor_mask(binary1, binary2)),
            ("encode_variant", lambda a: a.encode_variant(1, 12345, "A", "G", 1000)),
        ]

        for name, op in tests:
            py_result = op(self.python_accel)
            rust_result = op(self.rust_accel)

            if isinstance(py_result, np.ndarray):
                np.testing.assert_allclose(
                    py_result, rust_result, rtol=1e-5, err_msg=f"Mismatch in {name}"
                )
            elif isinstance(py_result, (int, float)):
                self.assertAlmostEqual(py_result, rust_result, places=5, msg=f"Mismatch in {name}")


def run_tests():
    """Run all tests."""
    print("=" * 80)
    print("🧪 GENOMEVAULT ACCELERATOR TESTS")
    print("=" * 80)

    # Check for Rust availability
    try:
        accel = Accelerator(force_python=False)
        if accel.use_rust:
            print("✅ Testing with Rust accelerator")
        else:
            print("ℹ️  Testing Python implementation only (Rust not available)")
    except:
        print("ℹ️  Testing Python implementation only")

    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAccelerator)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} tests failed")
        print(f"❌ {len(result.errors)} tests had errors")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
