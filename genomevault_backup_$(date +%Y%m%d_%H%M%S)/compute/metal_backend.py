"""
Metal Backend - Apple Silicon Acceleration

MLX-based implementation optimized for:
- M1/M2/M3 chips with unified memory
- Zero-copy operations (no CPU↔GPU transfers)
- Energy efficiency

Performance Targets:
- Single encode: <1ms (10× CPU speedup)
- Batch encode (1K): <100ms (50× CPU speedup)
- Memory efficiency: Unified memory architecture

Use Cases:
- MacBook/Mac Mini/Mac Studio deployment
- Research workflows on Apple Silicon
- Development with hardware acceleration
"""

import numpy as np
import logging
from typing import Optional

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

logger = logging.getLogger(__name__)


class MetalBackend:
    """
    Apple Metal acceleration via MLX

    Advantages:
    - Unified memory: No CPU→GPU copy overhead
    - Energy efficient: Optimized for Apple Silicon
    - Fast compilation: JIT compilation of operations

    Key Performance Benefit:
    Unlike discrete GPUs (CUDA), Metal benefits even single operations
    due to zero-copy unified memory architecture
    """

    def __init__(self):
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX not available. Install with: pip install mlx\n"
                "Note: MLX only works on Apple Silicon (M1/M2/M3)"
            )

        self.name = "Metal (Apple Silicon)"
        self.logger = logging.getLogger(__name__)

        # Pre-compile common kernels
        self._compile_kernels()

        # Cache for projection matrices
        self._projection_cache = {}

        self.logger.debug(f"Initialized {self.name}")

    def _compile_kernels(self):
        """Pre-compile frequently used operations"""
        # MLX auto-compiles, but we can define reusable functions

        @mx.compile
        def random_projection(data: mx.array, proj_matrix: mx.array) -> mx.array:
            """Compiled random projection"""
            return mx.matmul(data, proj_matrix.T)

        @mx.compile
        def binarize(x: mx.array) -> mx.array:
            """Compiled binarization"""
            return (x > 0).astype(mx.float32)

        self._random_projection = random_projection
        self._binarize = binarize

    def encode_single(self, variants: np.ndarray) -> np.ndarray:
        """
        Encode single sample with Metal acceleration

        Even single encodes benefit from Metal due to:
        - No transfer overhead (unified memory)
        - Fast JIT-compiled operations
        - Optimized Metal kernels

        Args:
            variants: (n_variants, n_features) NumPy array

        Returns:
            (dimension,) NumPy array (automatically copied back)

        Target Performance: <1ms
        """
        # Convert to MLX array (zero-copy on unified memory)
        v = mx.array(variants, dtype=mx.float32)

        # Encode using Metal
        result = self._metal_encode(v)

        # Convert back to NumPy (small copy, but result is small)
        return np.array(result)

    def encode_batch(self, variants_batch: list[np.ndarray]) -> np.ndarray:
        """
        Batch encoding - THIS IS WHERE METAL SHINES

        All operations stay in Metal:
        - Binding, bundling, projection all on GPU
        - No intermediate CPU copies
        - Vectorized across entire batch

        Args:
            variants_batch: List of (n_variants, n_features) arrays

        Returns:
            (batch_size, dimension) NumPy array

        Target Performance: <100ms for 1K samples
        """
        batch_size = len(variants_batch)

        if batch_size == 0:
            return np.array([])

        if batch_size == 1:
            return self.encode_single(variants_batch[0]).reshape(1, -1)

        # Stack into batch (on CPU, then transfer once)
        try:
            variants_stacked_np = np.stack([v.flatten() for v in variants_batch])
        except ValueError:
            # Different shapes, pad to max size
            max_size = max(v.size for v in variants_batch)
            variants_stacked_np = np.zeros((batch_size, max_size), dtype=np.float32)
            for i, v in enumerate(variants_batch):
                flat = v.flatten()
                variants_stacked_np[i, :flat.size] = flat

        # Single transfer to Metal
        batch_mx = mx.array(variants_stacked_np, dtype=mx.float32)

        # Vectorized encoding (all ops on Metal)
        results = mx.vmap(self._metal_encode)(batch_mx)

        # Single transfer back
        return np.array(results)

    def _metal_encode(self, v: mx.array) -> mx.array:
        """
        Core encoding kernel - stays entirely on Metal

        Args:
            v: (n_variants, n_features) or (features,) MLX array

        Returns:
            (dimension,) MLX array
        """
        # Flatten if needed
        if v.ndim > 1:
            v = v.reshape(-1)

        dimension = 8192
        features = v.shape[0]

        # Get or create projection matrix
        cache_key = (dimension, features)
        if cache_key not in self._projection_cache:
            # Create on Metal
            proj = mx.random.normal((dimension, features), dtype=mx.float32)
            proj = proj / mx.sqrt(mx.array(features, dtype=mx.float32))
            self._projection_cache[cache_key] = proj
        else:
            proj = self._projection_cache[cache_key]

        # Project (all on Metal)
        hypervector = self._random_projection(v.reshape(1, -1), proj)

        # Binarize (all on Metal)
        hypervector = self._binarize(hypervector)

        return hypervector.reshape(-1)

    def similarity_search(
        self,
        query: np.ndarray,
        database: np.ndarray,
        top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Metal-accelerated similarity search

        For large databases, brute-force on Metal is often faster
        than approximate search on CPU

        Args:
            query: (dimension,) query vector
            database: (n_samples, dimension) database

        Returns:
            (indices, similarities)
        """
        # Transfer to Metal
        query_mx = mx.array(query, dtype=mx.float32)
        database_mx = mx.array(database, dtype=mx.float32)

        # Normalize for cosine similarity
        query_norm = query_mx / (mx.linalg.norm(query_mx) + 1e-8)
        database_norm = database_mx / (
            mx.linalg.norm(database_mx, axis=1, keepdims=True) + 1e-8
        )

        # Compute similarities (all on Metal)
        similarities = mx.matmul(database_norm, query_norm)

        # Get top-k indices
        # MLX doesn't have argpartition, use full sort
        sorted_indices = mx.argsort(-similarities)[:top_k]
        top_similarities = similarities[sorted_indices]

        # Convert back to NumPy
        return np.array(sorted_indices), np.array(top_similarities)

    def bind_vectors(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        HDC binding (XOR) on Metal

        Args:
            a: (dimension,) or (batch, dimension)
            b: (dimension,) or (batch, dimension)

        Returns:
            Bound vector(s)
        """
        a_mx = mx.array(a, dtype=mx.float32)
        b_mx = mx.array(b, dtype=mx.float32)

        # XOR for binary vectors
        result = mx.logical_xor(
            a_mx.astype(mx.bool_),
            b_mx.astype(mx.bool_)
        ).astype(mx.float32)

        return np.array(result)

    def bundle_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        HDC bundling (majority vote) on Metal

        Args:
            vectors: (n_vectors, dimension)

        Returns:
            (dimension,) bundled vector
        """
        vectors_mx = mx.array(vectors, dtype=mx.float32)

        # Sum and threshold
        summed = mx.sum(vectors_mx, axis=0)
        threshold = vectors.shape[0] / 2.0

        result = (summed > threshold).astype(mx.float32)

        return np.array(result)


# Fallback if MLX not available
if not MLX_AVAILABLE:
    class MetalBackend:
        """Fallback when MLX unavailable"""
        def __init__(self):
            raise ImportError(
                "MLX not available. Install with: pip install mlx\n"
                "Note: MLX only works on Apple Silicon (M1/M2/M3)"
            )
