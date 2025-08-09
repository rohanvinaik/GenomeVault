"""
Unified hypervector encoder supporting sparse and orthogonal projections
for genomic data at 10k/15k/20k dimensions with seed control and determinism.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np

from genomevault.core.exceptions import HypervectorError, ProjectionError
from genomevault.hypervector.encoding.orthogonal_projection import OrthogonalProjection
from genomevault.hypervector.encoding.sparse_projection import SparseRandomProjection
from genomevault.hypervector.operations.binding import (
    bundle,
    circular_convolution,
    element_wise_multiply,
    permutation_binding,
    unbundle,
)

ProjectionType = Literal["sparse", "orthogonal"]
DimensionTier = Literal[10000, 15000, 20000]


class UnifiedHypervectorEncoder:
    """
    Unified encoder for genomic features supporting both sparse and orthogonal projections
    at standard dimension tiers (10k, 15k, 20k) with deterministic seed control.
    """

    def __init__(
        self,
        dimension: DimensionTier = 10000,
        projection_type: ProjectionType = "sparse",
        sparse_density: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize the unified encoder.

        Args:
            dimension: Target hypervector dimension (10000, 15000, or 20000)
            projection_type: Type of projection to use ("sparse" or "orthogonal")
            sparse_density: Density for sparse projections (only used if projection_type="sparse")
            seed: Random seed for reproducibility
        """
        if dimension not in (10000, 15000, 20000):
            raise ProjectionError(
                f"Dimension must be 10000, 15000, or 20000, got {dimension}",
                context={"dimension": dimension},
            )

        self.dimension = dimension
        self.projection_type = projection_type
        self.seed = seed
        self.sparse_density = sparse_density

        # Initialize projection
        self.projection: Union[SparseRandomProjection, OrthogonalProjection]
        if projection_type == "sparse":
            self.projection = SparseRandomProjection(
                n_components=dimension, density=sparse_density, seed=seed
            )
        elif projection_type == "orthogonal":
            self.projection = OrthogonalProjection(n_components=dimension, seed=seed)
        else:
            raise ProjectionError(
                f"Unknown projection type: {projection_type}",
                context={"projection_type": projection_type},
            )

        # Base vectors for different genomic features
        self._base_vectors: dict[str, np.ndarray] = {}
        self._fitted = False

    def fit(self, n_features: int) -> "UnifiedHypervectorEncoder":
        """
        Fit the encoder to the input feature dimension.

        Args:
            n_features: Number of input features

        Returns:
            Self for method chaining
        """
        self.projection.fit(n_features)
        self._fitted = True

        # Generate base vectors for genomic elements
        rng = np.random.default_rng(self.seed)

        # Nucleotide base vectors (orthogonal)
        for i, base in enumerate(["A", "T", "G", "C"]):
            vec = np.zeros(self.dimension)
            start = i * (self.dimension // 4)
            end = (i + 1) * (self.dimension // 4)
            vec[start:end] = 1.0
            vec = vec / np.linalg.norm(vec)
            self._base_vectors[base] = vec

        # Variant type vectors (random)
        for variant_type in ["SNP", "INS", "DEL", "DUP", "INV"]:
            vec = rng.standard_normal(self.dimension)
            vec = vec / np.linalg.norm(vec)
            self._base_vectors[variant_type] = vec

        # Chromosome vectors
        for i in range(1, 25):  # chr1-22, X, Y
            chr_name = f"chr{i}" if i <= 22 else ("chrX" if i == 23 else "chrY")
            vec = rng.standard_normal(self.dimension)
            vec = vec / np.linalg.norm(vec)
            self._base_vectors[chr_name] = vec

        return self

    def encode_genomic_features(
        self, features: np.ndarray, feature_names: list[str] | None = None
    ) -> np.ndarray:
        """
        Encode genomic features into hypervector space.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            feature_names: Optional names for features (for semantic encoding)

        Returns:
            Hypervector matrix of shape (n_samples, dimension)
        """
        if not self._fitted:
            raise HypervectorError("Encoder must be fitted before encoding")

        # Project features to hypervector space
        hypervectors = self.projection.transform(features)

        # Normalize
        norms = np.linalg.norm(hypervectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        hypervectors = hypervectors / norms

        return hypervectors

    def encode_variant(
        self,
        chromosome: str,
        position: int,
        ref: str,
        alt: str,
        variant_type: str = "SNP",
    ) -> np.ndarray:
        """
        Encode a single genomic variant.

        Args:
            chromosome: Chromosome name (e.g., "chr1", "chrX")
            position: Genomic position
            ref: Reference allele
            alt: Alternative allele
            variant_type: Type of variant (SNP, INS, DEL, etc.)

        Returns:
            Hypervector representation of the variant
        """
        if not self._fitted:
            # Auto-fit with default feature size
            self.fit(n_features=100)

        # Start with variant type vector
        if variant_type not in self._base_vectors:
            raise HypervectorError(f"Unknown variant type: {variant_type}")

        variant_vec = self._base_vectors[variant_type].copy()

        # Bind with chromosome vector
        if chromosome in self._base_vectors:
            chr_vec = self._base_vectors[chromosome]
            variant_vec = circular_convolution(variant_vec, chr_vec)

        # Encode position using permutation
        position_vec = self._encode_position(position)
        variant_vec = element_wise_multiply(variant_vec, position_vec)

        # Add reference and alternative alleles if single nucleotides
        if len(ref) == 1 and ref in self._base_vectors:
            ref_vec = self._base_vectors[ref]
            variant_vec = circular_convolution(variant_vec, ref_vec)

        if len(alt) == 1 and alt in self._base_vectors:
            alt_vec = self._base_vectors[alt]
            # Use permutation to distinguish alt from ref
            alt_vec = permutation_binding(alt_vec, shift=1)
            variant_vec = circular_convolution(variant_vec, alt_vec)

        # Normalize
        norm = np.linalg.norm(variant_vec)
        if norm > 0:
            variant_vec = variant_vec / norm

        return variant_vec

    def _encode_position(self, position: int) -> np.ndarray:
        """
        Encode genomic position as a hypervector.

        Args:
            position: Genomic position

        Returns:
            Position hypervector
        """
        # Use position as seed for deterministic encoding
        rng = np.random.default_rng(position)
        vec = rng.standard_normal(self.dimension)
        return vec / np.linalg.norm(vec)

    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode a DNA sequence into a hypervector.

        Args:
            sequence: DNA sequence string

        Returns:
            Sequence hypervector
        """
        if not self._fitted:
            self.fit(n_features=100)

        if not sequence:
            return np.zeros(self.dimension)

        # Encode each base with position binding
        vectors = []
        for i, base in enumerate(sequence.upper()):
            if base in self._base_vectors:
                base_vec = self._base_vectors[base]
                # Apply position-specific permutation
                pos_vec = permutation_binding(base_vec, shift=i)
                vectors.append(pos_vec)

        if not vectors:
            return np.zeros(self.dimension)

        # Bundle all position-encoded bases
        return bundle(vectors)

    def cross_modal_binding(
        self,
        genomic_vec: np.ndarray,
        clinical_vec: np.ndarray,
        modality_weights: dict[str, float] | None = None,
    ) -> np.ndarray:
        """
        Perform cross-modal binding between genomic and clinical data.

        Args:
            genomic_vec: Genomic hypervector
            clinical_vec: Clinical hypervector
            modality_weights: Optional weights for each modality

        Returns:
            Cross-modal bound hypervector
        """
        if modality_weights is None:
            modality_weights = {"genomic": 0.5, "clinical": 0.5}

        # Weighted combination
        w_genomic = modality_weights.get("genomic", 0.5)
        w_clinical = modality_weights.get("clinical", 0.5)

        # Normalize weights
        total_weight = w_genomic + w_clinical
        if total_weight > 0:
            w_genomic /= total_weight
            w_clinical /= total_weight

        # Weighted binding
        combined = w_genomic * genomic_vec + w_clinical * clinical_vec

        # Add interaction term using circular convolution
        interaction = circular_convolution(genomic_vec, clinical_vec)
        combined = combined + 0.1 * interaction  # Small weight for interaction

        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two hypervectors.

        Args:
            vec1: First hypervector
            vec2: Second hypervector

        Returns:
            Cosine similarity in range [-1, 1]
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def decode_components(
        self, hypervector: np.ndarray, threshold: float = 0.3
    ) -> list[tuple[str, float]]:
        """
        Attempt to decode components from a bundled hypervector.

        Args:
            hypervector: Input hypervector
            threshold: Minimum similarity threshold for component detection

        Returns:
            List of (component_name, similarity) tuples
        """
        if not self._base_vectors:
            return []

        return unbundle(hypervector, self._base_vectors, threshold)


def create_encoder(
    dimension: DimensionTier = 10000,
    projection_type: ProjectionType = "sparse",
    **kwargs,
) -> UnifiedHypervectorEncoder:
    """
    Factory function to create a hypervector encoder.

    Args:
        dimension: Target dimension (10000, 15000, or 20000)
        projection_type: Type of projection ("sparse" or "orthogonal")
        **kwargs: Additional arguments passed to encoder

    Returns:
        Configured encoder instance
    """
    return UnifiedHypervectorEncoder(
        dimension=dimension, projection_type=projection_type, **kwargs
    )
