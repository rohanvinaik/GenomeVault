"""
Hierarchical Hyperdimensional Computing (HDC) Encoder for genomic data

This module implements the core hypervector encoding functionality that transforms
processed multi-omics data into high-dimensional, privacy-preserving representations.
"""
from typing import Dict, List, Optional, Any, Union

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

# Try to import pandas, but make it optional
try:
    import pandas as pd
except ImportError:
    pd = None

# Import our optimized Hamming LUT
try:
    from genomevault.hypervector.operations.hamming_lut import HammingLUT

    HAMMING_LUT_AVAILABLE = True
except ImportError:
    HAMMING_LUT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OmicsType(Enum):
    """Types of omics data"""

    GENOMIC = "genomic"
    TRANSCRIPTOMIC = "transcriptomic"
    EPIGENOMIC = "epigenomic"
    PROTEOMIC = "proteomic"
    CLINICAL = "clinical"


class ProjectionType(Enum):
    """Types of projection matrices for different use cases"""

    RANDOM_GAUSSIAN = "random_gaussian"
    SPARSE_RANDOM = "sparse_random"
    LEARNED = "learned"
    ORTHOGONAL = "orthogonal"


class CompressionTier(Enum):
    """Compression tiers for different use cases"""

    MINI = "mini"  # ~5,000 SNPs, ~25KB
    CLINICAL = "clinical"  # ACMG + PharmGKB variants (~120k), ~300KB
    FULL = "full"  # Full HDC 10,000-20,000D vectors, 100-200KB


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
    compression_tier: CompressionTier = CompressionTier.FULL


@dataclass
class EncodingMetrics:
    """Metrics for encoding performance"""

    encoding_time_ms: float
    memory_usage_kb: float
    dimension: int
    sparsity: float
    compression_ratio: float


class HypervectorEncoder:
    """
    Hierarchical hyperdimensional encoder for multi-omics data

    This encoder transforms biological features into high-dimensional vectors
    that preserve similarity relationships while providing privacy guarantees.
    """

    def __init__(self, config: Optional[HypervectorConfig] = None) -> None:
           """TODO: Add docstring for __init__"""
     """
        Initialize the hypervector encoder

        Args:
            config: Encoding configuration, uses defaults if None
        """
        self.config = config or HypervectorConfig()

        # Set random seed for reproducibility
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)

        # Cache for projection matrices
        self._projection_cache = {}

        # Multi-resolution dimensions based on tier
        self.tier_configs = {
            CompressionTier.MINI: {
                "dimension": 5000,
                "features": "most_studied_snps",
                "size_kb": 25,
            },
            CompressionTier.CLINICAL: {
                "dimension": 10000,
                "features": "acmg_pharmgkb_variants",
                "size_kb": 300,
            },
            CompressionTier.FULL: {
                "dimension": self.config.dimension,
                "features": "all",
                "size_kb": 200,  # Due to compression
            },
        }

        # Set encoder metadata
        self.version = "v1.0.0"
        self.fingerprint = self._generate_fingerprint()
        self.dimension = self.config.dimension

        # Initialize Hamming LUT if available
        self.hamming_lut = None
        if HAMMING_LUT_AVAILABLE:
            try:
                self.hamming_lut = HammingLUT(use_gpu=torch.cuda.is_available())
                logger.info("Initialized with optimized Hamming LUT")
            except Exception as e:
                logger.warning(f"Failed to initialize Hamming LUT: {e}")

        logger.info(f"Initialized HypervectorEncoder with {self.config.dimension}D vectors")

    def encode(
        self,
        features: Union[np.ndarray, torch.Tensor, Dict],
        omics_type: OmicsType,
        tier: Optional[CompressionTier] = None,
    ) -> torch.Tensor:
           """TODO: Add docstring for encode"""
     """
        Encode features into a hypervector

        Args:
            features: Input features (array, tensor, or dict from processing)
            omics_type: Type of omics data being encoded
            tier: Compression tier (uses config default if None)

        Returns:
            Encoded hypervector
        """
        try:
            # Use tier from config if not specified
            tier = tier or self.config.compression_tier

            # Extract features if dict
            if isinstance(features, dict):
                features = self._extract_features(features, omics_type)

            # Convert to tensor
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()
            elif not isinstance(features, torch.Tensor):
                raise ValueError(f"Unsupported feature type: {type(features)}")

            # Get appropriate dimension for tier
            dimension = self.tier_configs[tier]["dimension"]

            # Get or create projection matrix
            projection_matrix = self._get_projection_matrix(
                len(features), dimension, omics_type, tier
            )

            # Project to hypervector space
            hypervector = self._project(features, projection_matrix)

            # Apply post-processing
            if self.config.normalize:
                hypervector = self._normalize(hypervector)

            if self.config.quantize:
                hypervector = self._quantize(hypervector)

            logger.debug(f"Encoded {omics_type.value} features to {dimension}D hypervector")

            return hypervector

        except Exception as e:
            logger.error(f"Encoding error: {str(e)}")
            raise

    def encode_multiresolution(
        self, features: Union[np.ndarray, torch.Tensor, Dict], omics_type: OmicsType
    ) -> Dict[str, torch.Tensor]:
           """TODO: Add docstring for encode_multiresolution"""
     """
        Encode features at multiple resolution levels

        Args:
            features: Input features
            omics_type: Type of omics data

        Returns:
            Dictionary mapping tiers to hypervectors
        """
        multiresolution_vectors = {}

        for tier in CompressionTier:
            multiresolution_vectors[tier.value] = self.encode(features, omics_type, tier)

        return multiresolution_vectors

    def _extract_features(self, data: Dict, omics_type: OmicsType) -> torch.Tensor:
           """TODO: Add docstring for _extract_features"""
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
                    [qm.get("mean_coverage", 0), qm.get("uniformity", 0), qm.get("gc_content", 0)]
                )

            # Convert to tensor
            return torch.tensor(features, dtype=torch.float32)

        elif omics_type == OmicsType.TRANSCRIPTOMIC:
            # Extract expression features
            if "expression_matrix" in data:
                expr = data["expression_matrix"]
                if pd is not None and isinstance(expr, pd.DataFrame):
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

        elif omics_type == OmicsType.CLINICAL:
            # Extract clinical features
            features = []
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    features.append(value)
                elif isinstance(value, str):
                    # Hash categorical values
                    features.append(int(hashlib.md5(value.encode()).hexdigest()[:8], 16) % 1000)
            return torch.tensor(features, dtype=torch.float32)

        # Default: flatten all numeric values
        features = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                features.append(value)
            elif isinstance(value, (list, np.ndarray)):
                features.extend(np.array(value).flatten())

        return torch.tensor(features, dtype=torch.float32)

    def _get_projection_matrix(
        self, input_dim: int, output_dim: int, omics_type: OmicsType, tier: CompressionTier
    ) -> torch.Tensor:
           """TODO: Add docstring for _get_projection_matrix"""
     """Get or create projection matrix for given dimensions"""
        # Create cache key
        cache_key = f"{omics_type.value}_{input_dim}_{output_dim}_{tier.value}"

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
            raise ValueError(f"Unsupported projection type: {self.config.projection_type}")

        # Cache the matrix
        self._projection_cache[cache_key] = matrix

        return matrix

    def _create_gaussian_projection(self, input_dim: int, output_dim: int) -> torch.Tensor:
           """TODO: Add docstring for _create_gaussian_projection"""
     """Create random Gaussian projection matrix"""
        # Standard random projection
        matrix = torch.randn(output_dim, input_dim) / np.sqrt(input_dim)
        return matrix

    def _create_sparse_projection(self, input_dim: int, output_dim: int) -> torch.Tensor:
           """TODO: Add docstring for _create_sparse_projection"""
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

    def _create_orthogonal_projection(self, input_dim: int, output_dim: int) -> torch.Tensor:
           """TODO: Add docstring for _create_orthogonal_projection"""
     """Create orthogonal projection matrix using QR decomposition"""
        # For orthogonal projection, we need output_dim <= input_dim
        if output_dim > input_dim:
            # Use transpose for dimensionality expansion
            base_matrix = torch.randn(input_dim, output_dim)
            q, _ = torch.linalg.qr(base_matrix)
            return q.T
        else:
            base_matrix = torch.randn(output_dim, input_dim)
            q, _ = torch.linalg.qr(base_matrix)
            return q

    def _project(self, features: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
           """TODO: Add docstring for _project"""
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
           """TODO: Add docstring for _normalize"""
     """Normalize hypervector to unit length"""
        norm = torch.norm(hypervector, p=2, dim=-1, keepdim=True)
        return hypervector / (norm + 1e-8)

    def _quantize(self, hypervector: torch.Tensor) -> torch.Tensor:
           """TODO: Add docstring for _quantize"""
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

    def similarity(self, hv1: torch.Tensor, hv2: torch.Tensor, metric: str = "cosine") -> float:
           """TODO: Add docstring for similarity"""
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
            return torch.nn.functional.cosine_similarity(hv1.view(1, -1), hv2.view(1, -1)).item()
        elif metric == "euclidean":
            return -torch.dist(hv1, hv2, p=2).item()
        elif metric == "hamming":
            # Use optimized LUT if available
            if self.hamming_lut is not None:
                # Convert to binary
                hv1_binary = (torch.sign(hv1) > 0).numpy().astype(np.uint8)
                hv2_binary = (torch.sign(hv2) > 0).numpy().astype(np.uint8)

                # Pack bits
                # Pack bits and ensure size is divisible by 8 for uint64 conversion
                hv1_bits = np.packbits(hv1_binary)
                # Pad to make divisible by 8
                pad_size = (8 - len(hv1_bits) % 8) % 8
                if pad_size > 0:
                    hv1_bits = np.pad(hv1_bits, (0, pad_size), mode="constant")
                hv1_packed = hv1_bits.view(np.uint64)

                # Pack bits and ensure size is divisible by 8 for uint64 conversion
                hv2_bits = np.packbits(hv2_binary)
                # Pad to make divisible by 8
                pad_size = (8 - len(hv2_bits) % 8) % 8
                if pad_size > 0:
                    hv2_bits = np.pad(hv2_bits, (0, pad_size), mode="constant")
                hv2_packed = hv2_bits.view(np.uint64)

                # Compute distance and normalize to similarity
                distance = self.hamming_lut.distance(hv1_packed, hv2_packed)
                return 1.0 - (distance / len(hv1_binary))
            else:
                # Fallback to standard implementation
                return (torch.sign(hv1) == torch.sign(hv2)).float().mean().item()
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def batch_similarity(
        self, hvs1: torch.Tensor, hvs2: torch.Tensor, metric: str = "hamming"
    ) -> torch.Tensor:
           """TODO: Add docstring for batch_similarity"""
     """
        Compute pairwise similarities between batches of hypervectors.

        Args:
            hvs1: First batch of hypervectors (N x D)
            hvs2: Second batch of hypervectors (M x D)
            metric: Similarity metric

        Returns:
            Similarity matrix (N x M)
        """
        if metric == "hamming" and self.hamming_lut is not None:
            # Use optimized batch computation
            hvs1_binary = (torch.sign(hvs1) > 0).numpy().astype(np.uint8)
            hvs2_binary = (torch.sign(hvs2) > 0).numpy().astype(np.uint8)

            # Pack bits for each vector
            hvs1_packed = np.array([np.packbits(v).view(np.uint64) for v in hvs1_binary])
            hvs2_packed = np.array([np.packbits(v).view(np.uint64) for v in hvs2_binary])

            # Compute distances
            distances = self.hamming_lut.distance_batch(hvs1_packed, hvs2_packed)

            # Convert to similarities
            similarities = 1.0 - (distances / hvs1.shape[1])
            return torch.from_numpy(similarities).float()
        else:
            # Fallback to pairwise computation
            n, d = hvs1.shape
            m, _ = hvs2.shape
            similarities = torch.zeros(n, m)

            for i in range(n):
                for j in range(m):
                    similarities[i, j] = self.similarity(hvs1[i], hvs2[j], metric)

            return similarities

    def get_encoding_metrics(self, start_time: float, hypervector: torch.Tensor) -> EncodingMetrics:
           """TODO: Add docstring for get_encoding_metrics"""
     """Calculate encoding metrics"""
        encoding_time_ms = (datetime.now().timestamp() - start_time) * 1000
        memory_usage_kb = hypervector.element_size() * hypervector.nelement() / 1024

        # Calculate sparsity
        sparsity = (hypervector == 0).float().mean().item()

        # Compression ratio (compared to float32 representation)
        original_size = hypervector.nelement() * 4  # float32
        compressed_size = hypervector.element_size() * hypervector.nelement()
        compression_ratio = original_size / compressed_size

        return EncodingMetrics(
            encoding_time_ms=encoding_time_ms,
            memory_usage_kb=memory_usage_kb,
            dimension=hypervector.shape[-1],
            sparsity=sparsity,
            compression_ratio=compression_ratio,
        )

    def get_projection_stats(self) -> Dict:
           """TODO: Add docstring for get_projection_stats"""
     """Get statistics about cached projections"""
        stats = {
            "num_cached_matrices": len(self._projection_cache),
            "cache_keys": list(self._projection_cache.keys()),
            "total_parameters": sum(m.numel() for m in self._projection_cache.values()),
            "memory_mb": sum(
                m.element_size() * m.numel() / 1024**2 for m in self._projection_cache.values()
            ),
        }
        return stats

    def _generate_fingerprint(self) -> str:
           """TODO: Add docstring for _generate_fingerprint"""
     """Generate fingerprint for this encoder configuration"""
        config_str = json.dumps(
            {
                "dimension": self.config.dimension,
                "projection_type": self.config.projection_type.value,
                "sparsity": self.config.sparsity,
                "seed": self.config.seed,
            },
            sort_keys=True,
        )

        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# Convenience functions
def create_encoder(
    dimension: int = 10000,
    projection_type: str = "sparse_random",
    compression_tier: str = "full",
    **kwargs,
) -> HypervectorEncoder:
       """TODO: Add docstring for create_encoder"""
     """Create a hypervector encoder with specified configuration"""
    config = HypervectorConfig(
        dimension=dimension,
        projection_type=ProjectionType(projection_type),
        compression_tier=CompressionTier(compression_tier),
        **kwargs,
    )
    return HypervectorEncoder(config)


def encode_genomic_data(genomic_data: Dict, dimension: int = 10000) -> torch.Tensor:
       """TODO: Add docstring for encode_genomic_data"""
     """Convenience function to encode genomic data"""
    encoder = create_encoder(dimension=dimension)
    return encoder.encode(genomic_data, OmicsType.GENOMIC)
