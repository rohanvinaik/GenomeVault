"""
Enhanced hierarchical hypervector compression module.
Implements the advanced compression from the project specifications.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from genomevault.utils.logging import get_logger

logger = logging.getLogger(__name__)


logger = get_logger(__name__)


@dataclass
class CompressedHierarchicalVector:
    """Compressed hierarchical hypervector with multiple levels."""

    vector_id: str
    base_vector: np.ndarray | None  # 10,000-D: foundational features
    mid_vector: np.ndarray | None  # 15,000-D: integrated modality
    high_vector: np.ndarray | None  # 20,000-D: semantic concepts
    compression_metadata: dict[str, Any]

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio from base to current level."""
        if self.base_vector is not None:
            base_size = self.base_vector.nbytes
            current_size = self.get_active_vector().nbytes
            return base_size / current_size
        return 1.0

    def get_active_vector(self) -> np.ndarray:
        """Get highest level vector available."""
        if self.high_vector is not None:
            return self.high_vector
        elif self.mid_vector is not None:
            return self.mid_vector
        elif self.base_vector is not None:
            return self.base_vector
        else:
            raise ValueError("No vector data available")

    @property
    def level(self) -> str:
        """Get current compression level."""
        if self.high_vector is not None:
            return "high"
        elif self.mid_vector is not None:
            return "mid"
        elif self.base_vector is not None:
            return "base"
        else:
            return "none"


class AdvancedHierarchicalCompressor:
    """
    Advanced hierarchical compression for GenomeVault hypervectors.
    Implements the three-tier system from specifications.
    """

    # Dimension sizes for each level
    BASE_DIM = 10000  # Foundational genome features
    MID_DIM = 15000  # Integrated modality features
    HIGH_DIM = 20000  # Semantic concepts

    def __init__(
        self,
        base_dim: int = BASE_DIM,
        mid_dim: int = MID_DIM,
        high_dim: int = HIGH_DIM,
        sparsity_threshold: float = 0.01,
    ):
        """
        Initialize hierarchical compressor.

        Args:
            base_dim: Dimension of base vectors
            mid_dim: Dimension of mid-level vectors
            high_dim: Dimension of high-level vectors
            sparsity_threshold: Threshold for sparse encoding
        """
        self.base_dim = base_dim
        self.mid_dim = mid_dim
        self.high_dim = high_dim
        self.sparsity_threshold = sparsity_threshold

        # Initialize projection matrices
        self.base_to_mid_projection = self._init_projection_matrix(base_dim, mid_dim)
        self.mid_to_high_projection = self._init_projection_matrix(mid_dim, high_dim)

        # Context vectors for semantic composition
        self.modality_contexts = self._init_modality_contexts()
        self.semantic_contexts = self._init_semantic_contexts()

        logger.info(
            "AdvancedHierarchicalCompressor initialized: %sbase_dimD -> %smid_dimD -> %shigh_dimD"
        )

    def _init_projection_matrix(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Initialize random projection matrix preserving distances."""
        # Use sparse random projection for efficiency
        # Johnson-Lindenstrauss: need O(log n / ε²) dimensions

        # Sparse matrix with +1, -1, 0 values
        sparsity = 1 / np.sqrt(in_dim)

        matrix = np.random.choice(
            [-1, 0, 1],
            size=(out_dim, in_dim),
            p=[sparsity / 2, 1 - sparsity, sparsity / 2],
        )

        # Normalize for distance preservation
        row_norms = np.sqrt(np.sum(matrix != 0, axis=1, keepdims=True))
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        matrix = matrix / row_norms

        return matrix.astype(np.float32)

    def _init_modality_contexts(self) -> dict[str, np.ndarray]:
        """Initialize context vectors for different modalities."""
        contexts = {
            "genomic": np.random.randn(self.mid_dim),
            "transcriptomic": np.random.randn(self.mid_dim),
            "epigenetic": np.random.randn(self.mid_dim),
            "proteomic": np.random.randn(self.mid_dim),
            "clinical": np.random.randn(self.mid_dim),
            "structural": np.random.randn(self.mid_dim),
        }

        # Normalize context vectors
        for key in contexts:
            contexts[key] = contexts[key] / np.linalg.norm(contexts[key])

        return contexts

    def _init_semantic_contexts(self) -> dict[str, np.ndarray]:
        """Initialize context vectors for semantic concepts."""
        contexts = {
            "disease_risk": np.random.randn(self.high_dim),
            "drug_response": np.random.randn(self.high_dim),
            "ancestry": np.random.randn(self.high_dim),
            "trait_association": np.random.randn(self.high_dim),
            "pathway_activity": np.random.randn(self.high_dim),
            "structural_variant": np.random.randn(self.high_dim),
        }

        # Normalize
        for key in contexts:
            contexts[key] = contexts[key] / np.linalg.norm(contexts[key])

        return contexts

    def hierarchical_compression(
        self,
        base_vector: np.ndarray,
        modality_context: str = "genomic",
        overall_model_context: str = "disease_risk",
    ) -> CompressedHierarchicalVector:
        """
        Implement the hierarchical compression from the specification:

        def hierarchical_compression(base_vector):
            mid_vector = semantic_composition(base_vector, modality_context)
            high_vector = semantic_composition(mid_vector, overall_model_context)
            return high_vector

        Args:
            base_vector: Base-level vector (10,000-D)
            modality_context: Modality for mid-level compression
            overall_model_context: Model context for high-level compression

        Returns:
            Compressed hierarchical vector
        """
        if len(base_vector) != self.base_dim:
            raise ValueError(f"Base vector must be {self.base_dim}-dimensional")

        # Normalize base vector
        base_vector = base_vector / (np.linalg.norm(base_vector) + 1e-8)

        compression_start = time.time()

        # Base -> Mid compression using semantic composition
        mid_vector = self.semantic_composition(base_vector, modality_context, level="base_to_mid")

        # Mid -> High compression using semantic composition
        high_vector = self.semantic_composition(
            mid_vector, overall_model_context, level="mid_to_high"
        )

        compression_time = time.time() - compression_start

        # Create compressed hierarchical vector
        compressed = CompressedHierarchicalVector(
            vector_id=self._generate_vector_id(base_vector),
            base_vector=None,  # Don't store base to save space
            mid_vector=None,  # Don't store mid to save space
            high_vector=high_vector,
            compression_metadata={
                "modality_context": modality_context,
                "overall_model_context": overall_model_context,
                "compression_time": compression_time,
                "original_dim": self.base_dim,
                "final_dim": self.high_dim,
                "compression_ratio": self.base_dim * 4 / (self.high_dim * 4),  # Assuming float32
                "sparsity": {
                    "base": np.mean(np.abs(base_vector) < self.sparsity_threshold),
                    "mid": np.mean(np.abs(mid_vector) < self.sparsity_threshold),
                    "high": np.mean(np.abs(high_vector) < self.sparsity_threshold),
                },
            },
        )

        logger.info(
            "Compressed vector %scompressed.vector_id: "
            "%sself.base_dimD -> %sself.high_dimD, "
            "ratio: %scompressed.compression_metadata['compression_ratio']:.2fx"
        )

        return compressed

    def semantic_composition(self, vector: np.ndarray, context: str, level: str) -> np.ndarray:
        """
        Perform semantic composition of vector with context.

        Args:
            vector: Input vector
            context: Context name
            level: Compression level ("base_to_mid" or "mid_to_high")

        Returns:
            Semantically composed vector
        """
        if level == "base_to_mid":
            # Project to mid dimension
            projected = self.base_to_mid_projection @ vector

            # Get modality context
            if context not in self.modality_contexts:
                logger.warning("Unknown modality %scontext, using genomic")
                context = "genomic"

            context_vector = self.modality_contexts[context]

        elif level == "mid_to_high":
            # Project to high dimension
            projected = self.mid_to_high_projection @ vector

            # Get semantic context
            if context not in self.semantic_contexts:
                logger.warning("Unknown semantic context %scontext, using disease_risk")
                context = "disease_risk"

            context_vector = self.semantic_contexts[context]

        else:
            raise ValueError(f"Unknown level: {level}")

        # Perform semantic composition using circular convolution
        composed = self._circular_convolution(projected, context_vector)

        # Apply sparsification
        composed = self._sparsify_vector(composed)

        # Normalize
        composed = composed / (np.linalg.norm(composed) + 1e-8)

        return composed

    def _circular_convolution(self, vector: np.ndarray, context: np.ndarray) -> np.ndarray:
        """
        Perform circular convolution for semantic binding.
        Preserves similarity relationships while integrating context.
        """
        # FFT-based circular convolution
        fft_vector = np.fft.fft(vector)
        fft_context = np.fft.fft(context)

        # Element-wise multiplication in frequency domain
        fft_result = fft_vector * fft_context

        # Inverse FFT to get convolution result
        result = np.real(np.fft.ifft(fft_result))

        return result

    def _sparsify_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Apply sparsification to reduce storage and improve efficiency.
        """
        # Keep only values above threshold
        sparse_vector = vector.copy()
        sparse_vector[np.abs(sparse_vector) < self.sparsity_threshold] = 0

        # Renormalize non-zero values to preserve energy
        non_zero_count = np.sum(sparse_vector != 0)
        if non_zero_count > 0:
            scaling_factor = np.sqrt(len(vector) / non_zero_count)
            sparse_vector = sparse_vector * scaling_factor

        return sparse_vector

    def _generate_vector_id(self, vector: np.ndarray) -> str:
        """Generate unique ID for vector."""
        # Use hash of vector content
        vector_bytes = vector.tobytes()
        return hashlib.sha256(vector_bytes).hexdigest()[:16]

    def create_storage_optimized_vector(
        self, base_vector: np.ndarray, tier: str = "clinical"
    ) -> dict[str, Any]:
        """
        Create storage-optimized vector for specific tier.

        Implements the three tiers from specifications:
        - Mini (25KB): Mobile apps, quick lookups
        - Clinical (300KB): Clinical decision support
        - FullHDC: Full analysis, research

        Args:
            base_vector: Input base vector
            tier: Storage tier

        Returns:
            Storage-optimized representation
        """
        # Compress to high level first
        compressed = self.hierarchical_compression(base_vector)
        high_vector = compressed.high_vector

        if tier == "mini":
            # 25KB target - extreme compression
            # Use only top 1000 dimensions after PCA-like reduction
            important_dims = np.argsort(np.abs(high_vector))[-1000:]
            mini_vector = np.zeros(1000, dtype=np.float16)  # 2KB
            mini_vector[:] = high_vector[important_dims].astype(np.float16)

            return {
                "tier": "mini",
                "data": mini_vector,
                "indices": important_dims.astype(np.uint16),  # 2KB
                "size_kb": (mini_vector.nbytes + important_dims.nbytes) / 1024,
                "metadata": compressed.compression_metadata,
            }

        elif tier == "clinical":
            # 300KB target - balanced compression
            # Quantize to 8-bit and keep sparse representation
            quantized = self._quantize_vector(high_vector, bits=8)
            non_zero_indices = np.where(quantized != 0)[0]
            non_zero_values = quantized[non_zero_indices]

            return {
                "tier": "clinical",
                "indices": non_zero_indices.astype(np.uint32),
                "values": non_zero_values,
                "size_kb": (non_zero_indices.nbytes + non_zero_values.nbytes) / 1024,
                "metadata": compressed.compression_metadata,
            }

        else:  # FullHDC
            # Full representation with all levels
            full_compressed = self.hierarchical_compression(base_vector)
            full_compressed.base_vector = base_vector
            full_compressed.mid_vector = self.semantic_composition(
                base_vector,
                compressed.compression_metadata["modality_context"],
                "base_to_mid",
            )

            return {
                "tier": "fullhdc",
                "vector": full_compressed,
                "size_kb": (
                    base_vector.nbytes
                    + full_compressed.mid_vector.nbytes
                    + full_compressed.high_vector.nbytes
                )
                / 1024,
                "metadata": compressed.compression_metadata,
            }

    def _quantize_vector(self, vector: np.ndarray, bits: int) -> np.ndarray:
        """Quantize vector to specified bit depth."""
        # Normalize to [-1, 1]
        min_val = vector.min()
        max_val = vector.max()
        if max_val - min_val > 0:
            normalized = 2 * (vector - min_val) / (max_val - min_val) - 1
        else:
            normalized = np.zeros_like(vector)

        # Quantize
        levels = 2**bits
        quantized = np.round((normalized + 1) * (levels - 1) / 2)

        # Convert back to approximate original range
        dequantized = (quantized / (levels - 1) * 2 - 1) * (max_val - min_val) / 2 + (
            max_val + min_val
        ) / 2

        return dequantized.astype(np.float32)


# Example usage
if __name__ == "__main__":
    # Initialize compressor
    compressor = AdvancedHierarchicalCompressor()

    # Create example base vector with genomic features
    logger.info("Advanced Hierarchical Compression Example")
    logger.info("=" * 60)

    # Simulate genomic feature vector (sparse)
    base_features = np.random.randn(10000)
    base_features[np.random.choice(10000, 9000, replace=False)] = 0  # Make sparse

    logger.info("Base vector: %slen(base_features) dimensions")
    logger.info("Sparsity: %snp.mean(base_features == 0):.2%")
    logger.info("Size: %sbase_features.nbytes / 1024:.1f KB")

    # Test hierarchical compression
    logger.info("\n\nHierarchical Compression:")
    logger.info("-" * 60)

    compressed = compressor.hierarchical_compression(
        base_features, modality_context="genomic", overall_model_context="disease_risk"
    )

    logger.info("Vector ID: %scompressed.vector_id")
    logger.info("Compression level: %scompressed.level")
    logger.info("Final dimensions: %slen(compressed.high_vector)")
    logger.info("Compression ratio: %scompressed.compression_metadata['compression_ratio']:.2fx")
    logger.info(
        "Compression time: %scompressed.compression_metadata['compression_time'] * 1000:.1f ms"
    )
    logger.info(f"Sparsity (high): {compressed.compression_metadata['sparsity']['high']:.2%}")

    # Test storage tiers
    logger.info("\n\nStorage Optimization Tiers:")
    logger.info("-" * 60)

    for tier in ["mini", "clinical", "fullhdc"]:
        optimized = compressor.create_storage_optimized_vector(base_features, tier)

        logger.info("\n%stier.upper() Tier:")
        logger.info("  Target use case: ", end="")
        if tier == "mini":
            logger.info("Mobile apps, quick lookups")
        elif tier == "clinical":
            logger.info("Clinical decision support")
        else:
            logger.info("Full analysis, research")

        logger.info("  Actual size: %soptimized['size_kb']:.1f KB")

        if tier == "mini":
            logger.info("  Dimensions kept: %slen(optimized['data'])")
        elif tier == "clinical":
            logger.info("  Non-zero values: %slen(optimized['values'])")
            logger.info("  Compression: %s20000 / len(optimized['values']):.1fx")
