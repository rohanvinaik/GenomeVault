"""
CPU Backend - Production Default

Pure NumPy implementation optimized for:
- Low latency (single-sample encoding <10ms)
- Predictable performance (no driver dependencies)
- Universal compatibility (runs anywhere)

Performance Targets:
- Single encode: <10ms
- Batch encode (100): <1s
- Similarity search (1M): <5s with FAISS indexing

Use Cases:
- Production API endpoints
- Real-time clinical queries
- Edge deployment
- Development/testing
"""

import numpy as np
import logging
from typing import Optional
import warnings

logger = logging.getLogger(__name__)


class CPUBackend:
    """
    CPU-optimized implementation using NumPy + optimized BLAS

    Relies on:
    - NumPy's vectorized operations
    - SIMD instructions (AVX-512/NEON via BLAS)
    - Cache-friendly memory access patterns
    - FAISS for large-scale similarity search
    """

    def __init__(self):
        self.name = "CPU (NumPy)"
        self.logger = logging.getLogger(__name__)

        # Pre-allocate workspace buffers for common dimensions
        self._workspace_8192 = None
        self._workspace_10000 = None

        # FAISS index cache for similarity search
        self._faiss_index: Optional[object] = None
        self._faiss_database_size = 0

        # Verify NumPy is using optimized BLAS
        self._verify_blas()

        self.logger.debug(f"Initialized {self.name}")

    def _verify_blas(self):
        """Verify NumPy is using optimized BLAS (OpenBLAS/MKL/Accelerate)"""
        try:
            config = np.__config__.show()
            # Check if using optimized BLAS
            if 'openblas' in str(config).lower():
                self.logger.debug("Using OpenBLAS for SIMD acceleration")
            elif 'mkl' in str(config).lower():
                self.logger.debug("Using Intel MKL for SIMD acceleration")
            elif 'accelerate' in str(config).lower():
                self.logger.debug("Using Apple Accelerate for SIMD acceleration")
            else:
                self.logger.warning(
                    "NumPy may not be using optimized BLAS. "
                    "Consider installing numpy with OpenBLAS or MKL for better performance."
                )
        except Exception as e:
            self.logger.debug(f"Could not verify BLAS configuration: {e}")

    def encode_single(self, variants: np.ndarray) -> np.ndarray:
        """
        Encode single genomic sample with minimal latency

        Optimizations:
        - Direct NumPy operations (no threading overhead)
        - Pre-allocated buffers
        - Cache-friendly access patterns

        Args:
            variants: (n_variants, n_features) array of genomic variants

        Returns:
            (dimension,) hypervector (binary: {0, 1} or bipolar: {-1, +1})

        Target Performance: <10ms
        """
        # Ensure float32 for SIMD efficiency
        if variants.dtype != np.float32:
            variants = variants.astype(np.float32, copy=False)

        # Simple random projection (placeholder - replace with actual HDC logic)
        dimension = 8192  # Standard dimension
        n_variants, n_features = variants.shape

        # Get or allocate workspace
        if dimension == 8192:
            if self._workspace_8192 is None or self._workspace_8192.shape != (dimension, n_features):
                self._workspace_8192 = np.random.randn(dimension, n_features).astype(np.float32)
                self._workspace_8192 /= np.sqrt(n_features)  # Normalize
            projection_matrix = self._workspace_8192
        else:
            projection_matrix = np.random.randn(dimension, n_features).astype(np.float32)
            projection_matrix /= np.sqrt(n_features)

        # Project to hyperdimensional space using optimized matrix multiply
        # This uses BLAS gemv (general matrix-vector product) internally
        hypervector = projection_matrix @ variants.flatten()

        # Binarize (threshold at 0)
        hypervector = (hypervector > 0).astype(np.float32)

        return hypervector

    def encode_batch(self, variants_batch: list[np.ndarray]) -> np.ndarray:
        """
        Encode multiple samples using vectorized operations

        Optimizations:
        - Batch operations across all samples
        - No multiprocessing (context switching overhead > benefit for <1K)
        - Vectorized threshold operations

        Args:
            variants_batch: List of (n_variants, n_features) arrays

        Returns:
            (batch_size, dimension) array of hypervectors

        Target Performance: <1s for 100 samples
        """
        batch_size = len(variants_batch)

        if batch_size == 0:
            return np.array([])

        if batch_size == 1:
            return self.encode_single(variants_batch[0]).reshape(1, -1)

        # Warn if large batch on CPU
        if batch_size > 1000:
            estimated_time_ms = batch_size * 5  # ~5ms per sample
            self.logger.warning(
                f"Encoding {batch_size} samples on CPU will take ~{estimated_time_ms}ms "
                f"({estimated_time_ms/1000:.1f}s). Consider enabling GPU acceleration "
                f"for {estimated_time_ms/100:.1f}× speedup."
            )

        # Stack all variants (assumes same shape)
        try:
            # Fast path: all same shape
            variants_stacked = np.stack([v.flatten() for v in variants_batch])
        except ValueError:
            # Slow path: different shapes, pad to max size
            max_size = max(v.size for v in variants_batch)
            variants_stacked = np.zeros((batch_size, max_size), dtype=np.float32)
            for i, v in enumerate(variants_batch):
                flat = v.flatten()
                variants_stacked[i, :flat.size] = flat

        # Batch projection
        dimension = 8192
        projection_matrix = np.random.randn(dimension, variants_stacked.shape[1]).astype(np.float32)
        projection_matrix /= np.sqrt(variants_stacked.shape[1])

        # Batch matrix multiply: (batch, features) @ (features, dimension).T
        # Uses BLAS gemm (general matrix-matrix product)
        hypervectors = variants_stacked @ projection_matrix.T

        # Batch binarization
        hypervectors = (hypervectors > 0).astype(np.float32)

        return hypervectors

    def similarity_search(
        self,
        query: np.ndarray,
        database: np.ndarray,
        top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find k most similar vectors using optimized search

        Strategy:
        - Small DB (<100K): Direct cosine similarity (fast with BLAS)
        - Large DB (>=100K): FAISS index (approximate but faster)

        Args:
            query: (dimension,) query hypervector
            database: (n_samples, dimension) database of hypervectors
            top_k: Number of top results to return

        Returns:
            (indices, similarities):
                - indices: (top_k,) array of database indices
                - similarities: (top_k,) array of similarity scores [0, 1]

        Target Performance: <5s for 1M database
        """
        n_samples = database.shape[0]

        # Normalize query and database for cosine similarity
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        database_norm = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-8)

        if n_samples < 100_000:
            # Direct computation with BLAS
            similarities = database_norm @ query_norm

            # Get top-k (negative for descending sort)
            top_indices = np.argpartition(-similarities, min(top_k, n_samples-1))[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]

            top_similarities = similarities[top_indices]

            return top_indices, top_similarities

        else:
            # Use FAISS for large databases
            return self._faiss_search(query_norm, database_norm, top_k)

    def _faiss_search(
        self,
        query: np.ndarray,
        database: np.ndarray,
        top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """FAISS-based similarity search for large databases"""
        try:
            import faiss
        except ImportError:
            self.logger.warning(
                "FAISS not available for large database search. "
                "Falling back to direct computation (slower). "
                "Install with: pip install faiss-cpu"
            )
            # Fallback to direct computation
            similarities = database @ query
            top_indices = np.argpartition(-similarities, min(top_k, len(similarities)-1))[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]
            return top_indices, similarities[top_indices]

        # Rebuild index if database changed
        if self._faiss_index is None or self._faiss_database_size != database.shape[0]:
            dimension = database.shape[1]

            # Use IndexFlatIP (inner product) for cosine similarity with normalized vectors
            self._faiss_index = faiss.IndexFlatIP(dimension)
            self._faiss_index.add(database.astype(np.float32))
            self._faiss_database_size = database.shape[0]

            self.logger.debug(f"Built FAISS index for {self._faiss_database_size} vectors")

        # Search
        query_2d = query.reshape(1, -1).astype(np.float32)
        similarities, indices = self._faiss_index.search(query_2d, top_k)

        return indices[0], similarities[0]

    def bind_vectors(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        HDC binding operation (XOR for binary, multiply for bipolar)

        Args:
            a: (dimension,) or (batch, dimension)
            b: (dimension,) or (batch, dimension)

        Returns:
            Bound vector(s) same shape as inputs
        """
        # Assume binary vectors {0, 1}
        # XOR = (a + b) mod 2 = a + b - 2*(a*b)
        return np.logical_xor(a.astype(bool), b.astype(bool)).astype(np.float32)

    def bundle_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        HDC bundling operation (majority vote)

        Args:
            vectors: (n_vectors, dimension) array

        Returns:
            (dimension,) bundled vector
        """
        # Sum across vectors and threshold at half
        summed = np.sum(vectors, axis=0)
        threshold = len(vectors) / 2.0

        return (summed > threshold).astype(np.float32)
