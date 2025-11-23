"""
Python interface for Rust accelerator with automatic fallback to pure Python.
"""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import Rust accelerator
try:
    import genomevault_accel

    RUST_AVAILABLE = True
    logger.info("✅ Rust accelerator loaded - using optimized implementations")
except ImportError:
    RUST_AVAILABLE = False
    logger.warning("⚠️ Rust accelerator not available - using pure Python implementations")


class Accelerator:
    """
    High-performance accelerator for GenomeVault operations.
    Automatically uses Rust implementations when available, falls back to Python.
    """

    def __init__(self, force_python: bool = False):
        """
        Initialize accelerator.

        Args:
            force_python: Force use of Python implementations even if Rust is available
        """
        self.use_rust = RUST_AVAILABLE and not force_python
        if self.use_rust:
            logger.info("Using Rust accelerator for hot paths")
        else:
            logger.info("Using Python implementations")

    def hypervector_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two hypervectors.

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            Cosine similarity between vectors
        """
        if self.use_rust:
            return genomevault_accel.fast_hypervector_similarity(
                a.astype(np.float32), b.astype(np.float32)
            )
        else:
            # Python fallback
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            return dot / (norm_a * norm_b + 1e-10)

    def batch_hypervector_similarity(self, vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
        """
        Compute similarities between multiple vectors and a query.

        Args:
            vectors: Matrix of vectors (n_vectors, dimension)
            query: Query vector (dimension,)

        Returns:
            Array of similarities
        """
        if self.use_rust:
            return genomevault_accel.batch_hypervector_similarity(
                vectors.astype(np.float32), query.astype(np.float32)
            )
        else:
            # Python fallback
            similarities = []
            for vec in vectors:
                similarities.append(self.hypervector_similarity(vec, query))
            return np.array(similarities)

    def pir_xor_mask(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply XOR mask for PIR operations.

        Args:
            data: Data to mask
            mask: XOR mask

        Returns:
            Masked data
        """
        if self.use_rust:
            return genomevault_accel.fast_pir_xor_mask(data.astype(np.uint8), mask.astype(np.uint8))
        else:
            # Python fallback
            return np.bitwise_xor(data, mask)

    def batch_pir_query(self, database: np.ndarray, query_mask: np.ndarray) -> np.ndarray:
        """
        Process batch PIR query.

        Args:
            database: Database matrix
            query_mask: Query selection mask

        Returns:
            Query result
        """
        if self.use_rust:
            return genomevault_accel.batch_pir_query(
                database.astype(np.uint8), query_mask.astype(np.uint8)
            )
        else:
            # Python fallback
            result = np.zeros(database.shape[1], dtype=np.uint8)
            for i, mask_bit in enumerate(query_mask):
                if mask_bit:
                    result ^= database[i]
            return result

    def hamming_distance(self, a: np.ndarray, b: np.ndarray) -> int:
        """
        Compute Hamming distance between binary arrays.

        Args:
            a: First binary array
            b: Second binary array

        Returns:
            Hamming distance
        """
        if self.use_rust:
            return genomevault_accel.fast_hamming_distance(a.astype(np.uint8), b.astype(np.uint8))
        else:
            # Python fallback
            return np.sum(np.bitwise_xor(a, b).astype(np.uint8))

    def batch_hamming_distance(self, vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
        """
        Compute Hamming distances for multiple vectors.

        Args:
            vectors: Matrix of binary vectors
            query: Query vector

        Returns:
            Array of distances
        """
        if self.use_rust:
            return genomevault_accel.batch_hamming_distance(
                vectors.astype(np.uint8), query.astype(np.uint8)
            )
        else:
            # Python fallback
            distances = []
            for vec in vectors:
                distances.append(self.hamming_distance(vec, query))
            return np.array(distances)

    def encode_variant(
        self,
        chromosome: int,
        position: int,
        ref_allele: str,
        alt_allele: str,
        dimension: int = 10000,
    ) -> np.ndarray:
        """
        Encode genomic variant to hypervector.

        Args:
            chromosome: Chromosome number
            position: Genomic position
            ref_allele: Reference allele
            alt_allele: Alternative allele
            dimension: Hypervector dimension

        Returns:
            Encoded hypervector
        """
        if self.use_rust:
            return genomevault_accel.fast_encode_variant(
                chromosome, position, ref_allele, alt_allele, dimension
            )
        else:
            # Python fallback using hash-based encoding
            import hashlib

            variant_str = f"chr{chromosome}:{position}:{ref_allele}>{alt_allele}"
            hash_obj = hashlib.sha256(variant_str.encode())
            hash_bytes = hash_obj.digest()

            # Generate sparse hypervector
            np.random.seed(int.from_bytes(hash_bytes[:4], "little"))
            hypervector = np.zeros(dimension, dtype=np.float32)

            # 10% sparsity
            num_active = int(dimension * 0.1)
            indices = np.random.choice(dimension, num_active, replace=False)
            hypervector[indices] = np.random.randn(num_active)

            # Normalize
            norm = np.linalg.norm(hypervector)
            if norm > 0:
                hypervector /= norm

            return hypervector

    def compress_hypervector(self, vector: np.ndarray) -> np.ndarray:
        """
        Compress hypervector to binary representation.

        Args:
            vector: Hypervector to compress

        Returns:
            Compressed binary representation
        """
        if self.use_rust:
            return genomevault_accel.compress_hypervector_binary(vector.astype(np.float32))
        else:
            # Python fallback
            binary = (vector > 0).astype(np.uint8)
            # Pack bits into bytes
            num_bytes = (len(binary) + 7) // 8
            compressed = np.zeros(num_bytes, dtype=np.uint8)

            for i, bit in enumerate(binary):
                if bit:
                    byte_idx = i // 8
                    bit_idx = i % 8
                    compressed[byte_idx] |= 1 << bit_idx

            return compressed

    def decompress_hypervector(self, compressed: np.ndarray, dimension: int) -> np.ndarray:
        """
        Decompress binary representation to hypervector.

        Args:
            compressed: Compressed binary data
            dimension: Original vector dimension

        Returns:
            Decompressed hypervector
        """
        if self.use_rust:
            return genomevault_accel.decompress_binary_hypervector(
                compressed.astype(np.uint8), dimension
            )
        else:
            # Python fallback
            vector = np.zeros(dimension, dtype=np.float32)

            for byte_idx, byte_val in enumerate(compressed):
                for bit_idx in range(8):
                    idx = byte_idx * 8 + bit_idx
                    if idx < dimension:
                        if (byte_val >> bit_idx) & 1:
                            vector[idx] = 1.0
                        else:
                            vector[idx] = -1.0

            return vector

    def knn_search(
        self, database: np.ndarray, query: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        K-nearest neighbors search in hypervector space.

        Args:
            database: Database of vectors
            query: Query vector
            k: Number of neighbors

        Returns:
            Tuple of (indices, distances)
        """
        if self.use_rust:
            return genomevault_accel.fast_knn_search(
                database.astype(np.float32), query.astype(np.float32), k
            )
        else:
            # Python fallback
            similarities = self.batch_hypervector_similarity(database, query)
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            top_k_distances = similarities[top_k_indices]
            return top_k_indices, top_k_distances

    def benchmark(self) -> dict:
        """
        Benchmark accelerator performance.

        Returns:
            Dictionary with benchmark results
        """
        import time

        results = {}

        # Test hypervector similarity
        dim = 10000
        vec1 = np.random.randn(dim).astype(np.float32)
        vec2 = np.random.randn(dim).astype(np.float32)

        start = time.perf_counter()
        for _ in range(100):
            _ = self.hypervector_similarity(vec1, vec2)
        results["hypervector_similarity_ms"] = (time.perf_counter() - start) * 10

        # Test PIR XOR
        data = np.random.randint(0, 256, 1000, dtype=np.uint8)
        mask = np.random.randint(0, 256, 1000, dtype=np.uint8)

        start = time.perf_counter()
        for _ in range(1000):
            _ = self.pir_xor_mask(data, mask)
        results["pir_xor_ms"] = time.perf_counter() - start

        # Test Hamming distance
        bin1 = np.random.randint(0, 256, 1000, dtype=np.uint8)
        bin2 = np.random.randint(0, 256, 1000, dtype=np.uint8)

        start = time.perf_counter()
        for _ in range(1000):
            _ = self.hamming_distance(bin1, bin2)
        results["hamming_distance_ms"] = time.perf_counter() - start

        results["accelerator"] = "rust" if self.use_rust else "python"

        return results


# Global accelerator instance
_accelerator = None


def get_accelerator(force_python: bool = False) -> Accelerator:
    """
    Get global accelerator instance.

    Args:
        force_python: Force use of Python implementations

    Returns:
        Accelerator instance
    """
    global _accelerator
    if _accelerator is None or force_python:
        _accelerator = Accelerator(force_python=force_python)
    return _accelerator
