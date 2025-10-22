"""
Differential Hypervector Encoder for GenomeVault.

This module implements hyperdimensional encoding of variant differences,
projecting 95-dimensional feature vectors into high-dimensional hypervector
space (typically 10,000 dimensions) and binding with metadata using
hyperdimensional computing operations.

Section 6.2: Differential Hypervector Encoding
"""

from __future__ import annotations

import hashlib
import numpy as np
from typing import List, Optional, Dict, Any

from genomevault.differential_encoding.differences import VariantDifference
from genomevault.differential_encoding.metadata import DifferentialEncodingMetadata
from genomevault.differential_encoding.feature_vectors import (
    differences_to_feature_vector,
    TOTAL_FEATURE_DIM,
)
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class DifferentialHypervectorEncoder:
    """
    Hyperdimensional encoder for differential genomic data.

    This class encodes variant differences and their metadata into high-dimensional
    hypervectors using hyperdimensional computing (HDC) operations. The encoding
    combines:
    - Random projection of 95D feature vectors to hypervector space
    - Binding with genomic position information
    - Binding with reference genome and chromosome identifiers
    - Binding with difference type distributions

    Attributes:
        dimension: Hypervector dimension (default 10,000)
        feature_dim: Input feature dimension (95)
        projection_matrix: Random projection from feature space to hypervector space
        base_vectors: Base vectors for deterministic encodings
        chromosome_permutations: Permutation matrices for chromosome encoding
        _rng: Random number generator for reproducibility

    Example:
        >>> encoder = DifferentialHypervectorEncoder(dimension=10000, seed=42)
        >>> hypervector = encoder.encode_difference_vector(differences, metadata)
        >>> print(hypervector.shape)
        (10000,)
    """

    def __init__(
        self,
        dimension: int = 10000,
        seed: Optional[int] = None,
        feature_dim: int = TOTAL_FEATURE_DIM,
    ):
        """
        Initialize the differential hypervector encoder.

        Args:
            dimension: Dimension of hypervectors (default 10,000)
            seed: Random seed for reproducibility (default None)
            feature_dim: Dimension of input feature vectors (default 95)

        Raises:
            ValueError: If dimension is not positive or feature_dim is invalid
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")

        self.dimension = dimension
        self.feature_dim = feature_dim
        self._rng = np.random.RandomState(seed)

        # Initialize random projection matrix (95D → hypervector dimension)
        # Use orthogonal random projection for better preservation of distances
        self.projection_matrix = self._initialize_projection_matrix()

        # Initialize base vectors for deterministic encodings
        self.base_vectors = self._initialize_base_vectors()

        # Initialize chromosome permutation matrices
        self.chromosome_permutations = self._initialize_chromosome_permutations()

        logger.info(
            f"Initialized DifferentialHypervectorEncoder: "
            f"dimension={dimension}, feature_dim={feature_dim}, seed={seed}"
        )

    def _initialize_projection_matrix(self) -> np.ndarray:
        """
        Initialize random projection matrix using Gaussian distribution.

        Uses random Gaussian matrix scaled for proper projection.

        Returns:
            Projection matrix of shape (dimension, feature_dim)
        """
        # Generate random Gaussian matrix
        # Scale by 1/sqrt(feature_dim) for Johnson-Lindenstrauss property
        matrix = self._rng.randn(self.dimension, self.feature_dim).astype(np.float32)
        matrix = matrix / np.sqrt(self.feature_dim)

        return matrix

    def _initialize_base_vectors(self) -> Dict[str, np.ndarray]:
        """
        Initialize base vectors for deterministic encodings.

        Creates orthogonal base vectors for binding operations:
        - position_base: For encoding genomic positions
        - reference_base: For encoding reference genome IDs
        - chromosome_base: For encoding chromosome identifiers
        - type_base: For encoding difference types

        Returns:
            Dictionary of base vectors
        """
        base_vectors = {}

        # Create orthogonal base vectors using Gram-Schmidt
        num_bases = 4
        raw_bases = self._rng.randn(num_bases, self.dimension).astype(np.float32)

        # Gram-Schmidt orthogonalization
        orthogonal_bases = []
        for i in range(num_bases):
            vec = raw_bases[i]
            # Subtract projections onto previous vectors
            for j in range(i):
                vec -= np.dot(vec, orthogonal_bases[j]) * orthogonal_bases[j]
            # Normalize
            vec /= (np.linalg.norm(vec) + 1e-10)
            orthogonal_bases.append(vec)

        base_vectors['position_base'] = orthogonal_bases[0]
        base_vectors['reference_base'] = orthogonal_bases[1]
        base_vectors['chromosome_base'] = orthogonal_bases[2]
        base_vectors['type_base'] = orthogonal_bases[3]

        return base_vectors

    def _initialize_chromosome_permutations(self) -> Dict[str, np.ndarray]:
        """
        Initialize permutation vectors for chromosome encoding.

        Creates deterministic permutations for common chromosomes.

        Returns:
            Dictionary mapping chromosome names to permutation indices
        """
        chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
        permutations = {}

        for chrom in chromosomes:
            # Use chromosome name to seed permutation
            chrom_seed = int(hashlib.sha256(chrom.encode()).hexdigest()[:8], 16)
            chrom_rng = np.random.RandomState(chrom_seed)
            permutations[chrom] = chrom_rng.permutation(self.dimension)

        return permutations

    def encode_difference_vector(
        self,
        differences: List[VariantDifference],
        metadata: Optional[DifferentialEncodingMetadata] = None,
        representative_position: Optional[int] = None,
    ) -> np.ndarray:
        """
        Encode variant differences and metadata into a hypervector.

        This is the main encoding method that:
        1. Generates 95D feature vector from differences
        2. Projects to hypervector space via random projection
        3. Binds with position range information
        4. Binds with chromosome identifier
        5. Binds with reference genome ID (if metadata provided)
        6. Binds with difference type distribution
        7. Normalizes the final hypervector

        Args:
            differences: List of variant differences to encode
            metadata: Optional metadata for additional context
            representative_position: Optional position override (default: median)

        Returns:
            Normalized hypervector of shape (dimension,)

        Example:
            >>> differences = compute_variant_differences(exp_section, ref_section)
            >>> metadata = create_metadata_from_chunk(...)
            >>> hv = encoder.encode_difference_vector(differences, metadata)
            >>> print(hv.shape, np.linalg.norm(hv))
            (10000,) 1.0
        """
        if not differences:
            logger.warning("Empty differences list, returning zero vector")
            return np.zeros(self.dimension, dtype=np.float32)

        # 1. Generate feature vector
        feature_vector = differences_to_feature_vector(
            differences,
            representative_position=representative_position
        )

        # 2. Project to hypervector space and normalize
        hypervector = self._project_to_hypervector(feature_vector)
        hypervector = self._normalize(hypervector)

        # 3. Bind with position range
        if differences:
            positions = [d.position for d in differences]
            start_pos = min(positions)
            end_pos = max(positions)
            chromosome = differences[0].chromosome

            position_hv = self._encode_position_range(start_pos, end_pos)
            hypervector = self._bind(hypervector, position_hv)
            hypervector = self._normalize(hypervector)  # Normalize after binding

            # 4. Bind with chromosome
            chromosome_hv = self._encode_chromosome(chromosome)
            hypervector = self._bind(hypervector, chromosome_hv)
            hypervector = self._normalize(hypervector)  # Normalize after binding

        # 5. Bind with reference ID (if metadata provided)
        if metadata:
            reference_hv = self._encode_reference_id(metadata.reference_genome_id)
            hypervector = self._bind(hypervector, reference_hv)
            hypervector = self._normalize(hypervector)  # Normalize after binding

        # 6. Bind with difference types
        type_hv = self._encode_difference_types(differences)
        hypervector = self._bind(hypervector, type_hv)

        # 7. Final normalization
        hypervector = self._normalize(hypervector)

        logger.debug(
            f"Encoded {len(differences)} differences into {self.dimension}D hypervector"
        )

        return hypervector

    def _project_to_hypervector(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Project feature vector to hypervector space.

        Uses random projection matrix to map from 95D to hypervector dimension.

        Args:
            feature_vector: 95D feature vector

        Returns:
            Projected hypervector (not normalized)
        """
        if len(feature_vector) != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim}D feature vector, got {len(feature_vector)}D"
            )

        # Project: hv = projection_matrix @ feature_vector
        hypervector = self.projection_matrix @ feature_vector

        return hypervector

    def _encode_position_range(self, start_pos: int, end_pos: int) -> np.ndarray:
        """
        Encode genomic position range into hypervector.

        Uses sinusoidal encoding of center position combined with
        span information.

        Args:
            start_pos: Start position (0-based)
            end_pos: End position (exclusive)

        Returns:
            Position range hypervector
        """
        # Encode center position using sinusoidal encoding
        center_pos = (start_pos + end_pos) // 2
        span = max(end_pos - start_pos, 1)  # Ensure minimum span of 1 to avoid zero encoding

        # Create sinusoidal encoding
        indices = np.arange(self.dimension, dtype=np.float32)

        # Alternate sin and cos
        position_encoding = np.zeros(self.dimension, dtype=np.float32)
        position_encoding[0::2] = np.sin(center_pos / (10000.0 ** (2 * indices[0::2] / self.dimension)))
        position_encoding[1::2] = np.cos(center_pos / (10000.0 ** (2 * indices[1::2] / self.dimension)))

        # Modulate by span (normalized) - add small offset to avoid zero
        span_factor = np.tanh(span / 100000.0 + 0.1)  # Add offset to avoid zero for small spans
        position_encoding *= span_factor

        # Bind with position base vector
        position_hv = self._bind(self.base_vectors['position_base'], position_encoding)

        return position_hv

    def _encode_reference_id(self, reference_id: str) -> np.ndarray:
        """
        Encode reference genome ID into hypervector.

        Uses deterministic hash-based encoding for reproducibility.

        Args:
            reference_id: Reference genome identifier

        Returns:
            Reference ID hypervector
        """
        # Generate deterministic hypervector from hash
        hash_seed = int(hashlib.sha256(reference_id.encode()).hexdigest()[:8], 16)
        ref_rng = np.random.RandomState(hash_seed)

        # Generate binary hypervector
        reference_encoding = ref_rng.choice(
            [-1.0, 1.0],
            size=self.dimension,
            p=[0.5, 0.5]
        ).astype(np.float32)

        # Bind with reference base vector
        reference_hv = self._bind(self.base_vectors['reference_base'], reference_encoding)

        return reference_hv

    def _encode_chromosome(self, chromosome: str) -> np.ndarray:
        """
        Encode chromosome identifier into hypervector.

        Uses permutation-based encoding for chromosomes.

        Args:
            chromosome: Chromosome identifier (e.g., 'chr1', 'chrX')

        Returns:
            Chromosome hypervector
        """
        # Get or create permutation for this chromosome
        if chromosome not in self.chromosome_permutations:
            # Create new permutation for unknown chromosome
            chrom_seed = int(hashlib.sha256(chromosome.encode()).hexdigest()[:8], 16)
            chrom_rng = np.random.RandomState(chrom_seed)
            self.chromosome_permutations[chromosome] = chrom_rng.permutation(self.dimension)

        # Apply permutation to chromosome base vector
        permutation = self.chromosome_permutations[chromosome]
        chromosome_hv = self.base_vectors['chromosome_base'][permutation]

        return chromosome_hv

    def _encode_difference_types(self, differences: List[VariantDifference]) -> np.ndarray:
        """
        Encode difference type distribution into hypervector.

        Uses weighted bundling of type-specific vectors.

        Args:
            differences: List of variant differences

        Returns:
            Difference types hypervector
        """
        from genomevault.differential_encoding.differences import DifferenceType

        # Count difference types
        type_counts = {
            DifferenceType.NEW_MUTATION: 0,
            DifferenceType.MISSING: 0,
            DifferenceType.GENOTYPE_DIFF: 0,
        }

        for diff in differences:
            type_counts[diff.difference_type] += 1

        total = len(differences)
        if total == 0:
            return np.zeros(self.dimension, dtype=np.float32)

        # Create type-specific vectors
        type_vectors = {}
        for diff_type in DifferenceType:
            # Deterministic vector for each type
            type_seed = int(hashlib.sha256(diff_type.value.encode()).hexdigest()[:8], 16)
            type_rng = np.random.RandomState(type_seed)
            type_vectors[diff_type] = type_rng.choice(
                [-1.0, 1.0],
                size=self.dimension,
                p=[0.5, 0.5]
            ).astype(np.float32)

        # Weighted bundle
        type_hv = np.zeros(self.dimension, dtype=np.float32)
        for diff_type, count in type_counts.items():
            weight = count / total
            type_hv += weight * type_vectors[diff_type]

        # Bind with type base vector
        type_hv = self._bind(self.base_vectors['type_base'], type_hv)

        return type_hv

    def _bind(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """
        Bind two hypervectors using circular convolution (FFT-based).

        Binding creates a dissimilar vector from two similar vectors,
        implementing the multiplication-like operation in HDC.

        Args:
            hv1: First hypervector
            hv2: Second hypervector

        Returns:
            Bound hypervector (not normalized)

        Mathematical formulation:
            bind(a, b) = ifft(fft(a) ⊙ fft(b))

        where ⊙ is element-wise multiplication in frequency domain.
        """
        if len(hv1) != len(hv2):
            raise ValueError(f"Hypervector dimension mismatch: {len(hv1)} != {len(hv2)}")

        # Use FFT for efficient circular convolution
        # bind(a, b) = IFFT(FFT(a) * FFT(b))
        fft_hv1 = np.fft.fft(hv1)
        fft_hv2 = np.fft.fft(hv2)
        fft_result = fft_hv1 * fft_hv2
        bound = np.fft.ifft(fft_result).real.astype(np.float32)

        return bound

    def _bundle(self, hypervectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Bundle (superpose) multiple hypervectors.

        Bundling creates a similar vector from multiple vectors,
        implementing the addition-like operation in HDC.

        Args:
            hypervectors: List of hypervectors to bundle
            weights: Optional weights for each hypervector (default: equal weights)

        Returns:
            Bundled hypervector (normalized)

        Mathematical formulation:
            bundle(a, b, c, ...) = normalize(Σ wᵢ * vᵢ)

        where wᵢ are weights (default 1/n) and vᵢ are hypervectors.
        """
        if not hypervectors:
            return np.zeros(self.dimension, dtype=np.float32)

        if weights is None:
            weights = [1.0 / len(hypervectors)] * len(hypervectors)

        if len(weights) != len(hypervectors):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of hypervectors ({len(hypervectors)})"
            )

        # Weighted sum
        bundled = np.zeros(self.dimension, dtype=np.float32)
        for hv, weight in zip(hypervectors, weights):
            bundled += weight * hv

        # Normalize
        bundled = self._normalize(bundled)

        return bundled

    def _normalize(self, hypervector: np.ndarray) -> np.ndarray:
        """
        Normalize hypervector to unit length.

        Args:
            hypervector: Hypervector to normalize

        Returns:
            Normalized hypervector with L2 norm = 1
        """
        norm = np.linalg.norm(hypervector)
        if norm < 1e-10:
            logger.warning("Near-zero hypervector, returning zero vector")
            return np.zeros(self.dimension, dtype=np.float32)

        return hypervector / norm

    def similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """
        Compute cosine similarity between two hypervectors.

        Args:
            hv1: First hypervector
            hv2: Second hypervector

        Returns:
            Cosine similarity in range [-1, 1]

        Example:
            >>> hv1 = encoder.encode_difference_vector(diffs1, metadata1)
            >>> hv2 = encoder.encode_difference_vector(diffs2, metadata2)
            >>> sim = encoder.similarity(hv1, hv2)
            >>> print(f"Similarity: {sim:.3f}")
        """
        if len(hv1) != len(hv2):
            raise ValueError(f"Hypervector dimension mismatch: {len(hv1)} != {len(hv2)}")

        # Cosine similarity
        dot_product = np.dot(hv1, hv2)
        norm1 = np.linalg.norm(hv1)
        norm2 = np.linalg.norm(hv2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def encode_batch(
        self,
        differences_list: List[List[VariantDifference]],
        metadata_list: Optional[List[DifferentialEncodingMetadata]] = None,
    ) -> np.ndarray:
        """
        Encode multiple difference sets into hypervectors.

        Efficient batch encoding for multiple genomic regions.

        Args:
            differences_list: List of difference lists
            metadata_list: Optional list of metadata (must match length)

        Returns:
            Matrix of hypervectors, shape (n_samples, dimension)

        Example:
            >>> differences_batch = [diffs1, diffs2, diffs3]
            >>> metadata_batch = [meta1, meta2, meta3]
            >>> hv_matrix = encoder.encode_batch(differences_batch, metadata_batch)
            >>> print(hv_matrix.shape)
            (3, 10000)
        """
        if metadata_list is not None and len(metadata_list) != len(differences_list):
            raise ValueError(
                f"Number of metadata ({len(metadata_list)}) must match "
                f"number of difference sets ({len(differences_list)})"
            )

        hypervectors = []
        for i, differences in enumerate(differences_list):
            metadata = metadata_list[i] if metadata_list else None
            hv = self.encode_difference_vector(differences, metadata)
            hypervectors.append(hv)

        return np.stack(hypervectors)

    def bind(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """
        Bind two hypervectors using circular convolution (public interface).

        Binding creates a dissimilar vector from two similar vectors,
        implementing the multiplication-like operation in HDC.

        Args:
            hv1: First hypervector
            hv2: Second hypervector

        Returns:
            Bound hypervector (normalized)

        Example:
            >>> hv1 = encoder.encode_difference_vector(diffs1)
            >>> hv2 = encoder.encode_difference_vector(diffs2)
            >>> bound = encoder.bind(hv1, hv2)
            >>> print(bound.shape)
            (10000,)
        """
        bound = self._bind(hv1, hv2)
        return self._normalize(bound)

    def bundle(self, hypervectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Bundle (superpose) multiple hypervectors (public interface).

        Bundling creates a similar vector from multiple vectors,
        implementing the addition-like operation in HDC.

        Args:
            hypervectors: List of hypervectors to bundle
            weights: Optional weights for each hypervector (default: equal weights)

        Returns:
            Bundled hypervector (normalized)

        Example:
            >>> hvs = [encoder.encode_difference_vector(d) for d in diff_list]
            >>> bundled = encoder.bundle(hvs)
            >>> print(bundled.shape)
            (10000,)
        """
        return self._bundle(hypervectors, weights)

    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration.

        Returns:
            Dictionary with encoder configuration
        """
        return {
            'dimension': self.dimension,
            'feature_dim': self.feature_dim,
            'num_base_vectors': len(self.base_vectors),
            'num_chromosome_permutations': len(self.chromosome_permutations),
        }
