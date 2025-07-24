"""
Hierarchical Hyperdimensional Computing (HDC) Encoder for genomic data

This module implements the core hypervector encoding functionality that transforms
processed multi-omics data into high-dimensional, privacy-preserving representations.
"""

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from genomevault.core.config import get_config
from genomevault.core.constants import OmicsType
from genomevault.core.exceptions import EncodingError
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class ProjectionType(Enum):
    """Types of projection matrices for different use cases"""

    RANDOM_GAUSSIAN = "random_gaussian"
    SPARSE_RANDOM = "sparse_random"
    LEARNED = "learned"
    ORTHOGONAL = "orthogonal"


@dataclass
class HypervectorConfig:
    """Configuration for hypervector encoding"""

    dimension: int = 10000
    projection_type: ProjectionType = ProjectionType.SPARSE_RANDOM
    sparsity: float = 0.1  # For sparse projections
    seed: Optional[int] = None
    normalize: bool = True
    quantize: bool = False
    quantization_bits: int = 8


class HypervectorEncoder:
    """
    Hierarchical hyperdimensional encoder for multi-omics data

    This encoder transforms biological features into high-dimensional vectors
    that preserve similarity relationships while providing privacy guarantees.
    """

    def __init__(self, config: Optional[HypervectorConfig] = None):
        """
        Initialize the hypervector encoder

        Args:
            config: Encoding configuration, uses defaults if None
        """
        self.config = config or HypervectorConfig()
        self.app_config = get_config()

        # Set random seed for reproducibility
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)

        # Cache for projection matrices
        self._projection_cache = {}

        # Multi-resolution dimensions
        self.resolution_dims = {"base": 10000, "mid": 15000, "high": 20000}

        logger.info(
            "Initialized HypervectorEncoder with {self.config.dimension}D vectors"
        )

    def encode(
        self,
        features: Union[np.ndarray, torch.Tensor, Dict],
        omics_type: OmicsType,
        resolution: str = "base",
    ) -> torch.Tensor:
        """
        Encode features into a hypervector

        Args:
            features: Input features (array, tensor, or dict from processing)
            omics_type: Type of omics data being encoded
            resolution: Resolution level (base, mid, high)

        Returns:
            Encoded hypervector
        """
        try:
            # Extract features if dict
            if isinstance(features, dict):
                features = self._extract_features(features, omics_type)

            # Convert to tensor
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()
            elif not isinstance(features, torch.Tensor):
                raise EncodingError("Unsupported feature type: {type(features)}")

            # Get appropriate dimension
            dimension = self.resolution_dims.get(resolution, self.config.dimension)

            # Get or create projection matrix
            projection_matrix = self._get_projection_matrix(
                len(features), dimension, omics_type
            )

            # Project to hypervector space
            hypervector = self._project(features, projection_matrix)

            # Apply post-processing
            if self.config.normalize:
                hypervector = self._normalize(hypervector)

            if self.config.quantize:
                hypervector = self._quantize(hypervector)

            logger.debug(
                "Encoded {omics_type.value} features to {dimension}D hypervector"
            )

            return hypervector

        except Exception as e:
            logger.error("Encoding error: {str(e)}")
            raise EncodingError("Failed to encode features: {str(e)}")

    def encode_multiresolution(
        self, features: Union[np.ndarray, torch.Tensor, Dict], omics_type: OmicsType
    ) -> Dict[str, torch.Tensor]:
        """
        Encode features at multiple resolution levels

        Args:
            features: Input features
            omics_type: Type of omics data

        Returns:
            Dictionary mapping resolution levels to hypervectors
        """
        multiresolution_vectors = {}

        for resolution in ["base", "mid", "high"]:
            multiresolution_vectors[resolution] = self.encode(
                features, omics_type, resolution
            )

        return multiresolution_vectors

    def _extract_features(self, data: Dict, omics_type: OmicsType) -> torch.Tensor:
        """Extract relevant features from processed data dictionary"""
        if omics_type == OmicsType.GENOMIC:
            # Extract variant features
            variants = data.get("variants", {})
            features = []

            # Add variant counts by type
            for var_type in ["snps", "indels", "cnvs"]:
                if var_type in variants:
                    features.append(len(variants[var_type]))

            # Add quality metrics
            if "quality_metrics" in data:
                qm = data["quality_metrics"]
                features.extend(
                    [
                        qm.get("mean_coverage", 0),
                        qm.get("uniformity", 0),
                        qm.get("gc_content", 0),
                    ]
                )

            # Convert to tensor
            return torch.tensor(features, dtype=torch.float32)

        elif omics_type == OmicsType.TRANSCRIPTOMIC:
            # Extract expression features
            if "expression_matrix" in data:
                expr = data["expression_matrix"]
                if isinstance(expr, pd.DataFrame):
                    return torch.from_numpy(expr.values.flatten()).float()
                else:
                    return torch.tensor(expr, dtype=torch.float32)

        elif omics_type == OmicsType.EPIGENOMIC:
            # Extract methylation features
            if "methylation_levels" in data:
                return torch.tensor(data["methylation_levels"], dtype=torch.float32)

        elif omics_type == OmicsType.PROTEOMIC:
            # Extract protein abundances
            if "protein_abundances" in data:
                return torch.tensor(data["protein_abundances"], dtype=torch.float32)

        # Default: flatten all numeric values
        features = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                features.append(value)
            elif isinstance(value, (list, np.ndarray)):
                features.extend(np.array(value).flatten())

        return torch.tensor(features, dtype=torch.float32)

    def _get_projection_matrix(
        self, input_dim: int, output_dim: int, omics_type: OmicsType
    ) -> torch.Tensor:
        """Get or create projection matrix for given dimensions"""
        # Create cache key
        cache_key = "{omics_type.value}_{input_dim}_{output_dim}"

        if cache_key in self._projection_cache:
            return self._projection_cache[cache_key]

        # Create new projection matrix
        if self.config.projection_type == ProjectionType.RANDOM_GAUSSIAN:
            matrix = self._create_gaussian_projection(input_dim, output_dim)
        elif self.config.projection_type == ProjectionType.SPARSE_RANDOM:
            matrix = self._create_sparse_projection(input_dim, output_dim)
        elif self.config.projection_type == ProjectionType.ORTHOGONAL:
            matrix = self._create_orthogonal_projection(input_dim, output_dim)
        else:
            raise EncodingError(
                "Unsupported projection type: {self.config.projection_type}"
            )

        # Cache the matrix
        self._projection_cache[cache_key] = matrix

        return matrix

    def _create_gaussian_projection(
        self, input_dim: int, output_dim: int
    ) -> torch.Tensor:
        """Create random Gaussian projection matrix"""
        # Standard random projection
        matrix = torch.randn(output_dim, input_dim) / np.sqrt(input_dim)
        return matrix

    def _create_sparse_projection(
        self, input_dim: int, output_dim: int
    ) -> torch.Tensor:
        """Create sparse random projection matrix"""
        matrix = torch.zeros(output_dim, input_dim)

        # Sparse random projection (Achlioptas, 2003)
        s = 1.0 / np.sqrt(self.config.sparsity * input_dim)

        for i in range(output_dim):
            # Number of non-zero entries
            nnz = int(input_dim * self.config.sparsity)

            # Random positions
            indices = torch.randperm(input_dim)[:nnz]

            # Random values from {-s, +s}
            values = torch.bernoulli(torch.ones(nnz) * 0.5) * 2 - 1
            values *= s

            matrix[i, indices] = values

        return matrix

    def _create_orthogonal_projection(
        self, input_dim: int, output_dim: int
    ) -> torch.Tensor:
        """Create orthogonal projection matrix using QR decomposition"""
        # For orthogonal projection, we need output_dim <= input_dim
        if output_dim > input_dim:
            # Use transpose for dimensionality expansion
            base_matrix = torch.randn(input_dim, output_dim)
            q, _ = torch.qr(base_matrix)
            return q.T
        else:
            base_matrix = torch.randn(output_dim, input_dim)
            q, _ = torch.qr(base_matrix)
            return q

    def _project(
        self, features: torch.Tensor, projection_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Project features using the projection matrix"""
        # Handle batch dimension if present
        if features.dim() > 1:
            # Flatten all but batch dimension
            batch_size = features.shape[0]
            features = features.view(batch_size, -1)

            # Batch matrix multiplication
            hypervectors = torch.matmul(features, projection_matrix.T)
            return hypervectors
        else:
            # Single vector projection
            return torch.matmul(projection_matrix, features)

    def _normalize(self, hypervector: torch.Tensor) -> torch.Tensor:
        """Normalize hypervector to unit length"""
        norm = torch.norm(hypervector, p=2, dim=-1, keepdim=True)
        return hypervector / (norm + 1e-8)

    def _quantize(self, hypervector: torch.Tensor) -> torch.Tensor:
        """Quantize hypervector to specified bit depth"""
        # Normalize to [-1, 1]
        min_val = hypervector.min()
        max_val = hypervector.max()
        normalized = 2 * (hypervector - min_val) / (max_val - min_val + 1e-8) - 1

        # Quantize
        levels = 2**self.config.quantization_bits
        quantized = torch.round((normalized + 1) * (levels - 1) / 2)

        # Scale back
        return 2 * quantized / (levels - 1) - 1

    def similarity(
        self, hv1: torch.Tensor, hv2: torch.Tensor, metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two hypervectors

        Args:
            hv1: First hypervector
            hv2: Second hypervector
            metric: Similarity metric (cosine, euclidean, hamming)

        Returns:
            Similarity score
        """
        if metric == "cosine":
            return torch.nn.functional.cosine_similarity(
                hv1.view(1, -1), hv2.view(1, -1)
            ).item()
        elif metric == "euclidean":
            return -torch.dist(hv1, hv2, p=2).item()
        elif metric == "hamming":
            # For binary hypervectors
            return (torch.sign(hv1) == torch.sign(hv2)).float().mean().item()
        else:
            raise ValueError("Unknown similarity metric: {metric}")

    def get_projection_stats(self) -> Dict:
        """Get statistics about cached projections"""
        stats = {
            "num_cached_matrices": len(self._projection_cache),
            "cache_keys": list(self._projection_cache.keys()),
            "total_parameters": sum(m.numel() for m in self._projection_cache.values()),
            "memory_mb": sum(
                m.element_size() * m.numel() / 1024**2
                for m in self._projection_cache.values()
            ),
        }
        return stats


# Convenience functions
def create_encoder(
    dimension: int = 10000, projection_type: str = "sparse_random", **kwargs
) -> HypervectorEncoder:
    """Create a hypervector encoder with specified configuration"""
    config = HypervectorConfig(
        dimension=dimension, projection_type=ProjectionType(projection_type), **kwargs
    )
    return HypervectorEncoder(config)


def encode_genomic_data(genomic_data: Dict, dimension: int = 10000) -> torch.Tensor:
    """Convenience function to encode genomic data"""
    encoder = create_encoder(dimension=dimension)
    return encoder.encode(genomic_data, OmicsType.GENOMIC)


# Import pandas only if needed
try:
    import pandas as pd
except ImportError:
    pd = None
