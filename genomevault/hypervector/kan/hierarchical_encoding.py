"""
Enhanced Hierarchical Hypervector Encoding

Implements the multi-resolution hierarchical encoding system from the KAN-HD insights,
with domain-specific projections and adaptive dimensionality.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..encoding.genomic import GenomicEncoder
from .kan_layer import LinearKAN


class DataModality(Enum):
    """Different types of genomic data modalities"""

    GENOMIC_VARIANTS = "variants"  # SNPs, indels, structural variants
    GENE_EXPRESSION = "expression"  # RNA-seq, microarray data
    EPIGENETIC = "epigenetic"  # DNA methylation, histone modifications
    PROTEOMIC = "proteomic"  # Protein expression levels
    PHENOTYPIC = "phenotypic"  # Clinical measurements, traits
    STRUCTURAL = "structural"  # 3D protein/DNA structure data


@dataclass
class EncodingSpecification:
    """Specification for hierarchical encoding"""

    modality: DataModality
    target_dimension: int
    compression_ratio: float
    privacy_level: str  # 'public', 'sensitive', 'highly_sensitive'
    interpretability_required: bool = True


@dataclass
class MultiResolutionVector:
    """Multi-resolution hyperdimensional vector"""

    base_vector: torch.Tensor  # Base resolution (D=10,000)
    mid_vector: torch.Tensor  # Mid resolution (D=15,000)
    high_vector: torch.Tensor  # High resolution (D=20,000)
    modality: DataModality
    compression_metadata: Dict[str, Any]


class AdaptiveDimensionalityCalculator:
    """
    Calculate optimal dimensions based on data complexity and privacy requirements

    Implements the insight that different data types need different dimensions.
    """

    def __init__(self):
        # Base dimensions from KAN-HD insights
        self.base_dimensions = {
            DataModality.GENOMIC_VARIANTS: 10000,
            DataModality.GENE_EXPRESSION: 15000,
            DataModality.EPIGENETIC: 20000,
            DataModality.PROTEOMIC: 12000,
            DataModality.PHENOTYPIC: 8000,
            DataModality.STRUCTURAL: 25000,
        }

        # Privacy multipliers
        self.privacy_multipliers = {"public": 0.8, "sensitive": 1.0, "highly_sensitive": 1.5}

    def calculate_optimal_dimension(
        self,
        modality: DataModality,
        data_complexity: float,
        privacy_level: str,
        target_error: float = 0.01,
    ) -> int:
        """
        Calculate optimal dimension using Johnson-Lindenstrauss lemma

        Args:
            modality: Type of genomic data
            data_complexity: Estimated intrinsic dimensionality
            privacy_level: Required privacy level
            target_error: Target distortion (epsilon)

        Returns:
            Optimal hyperdimensional dimension
        """
        base_dim = self.base_dimensions[modality]

        # Johnson-Lindenstrauss dimension requirement
        # d >= 4 * ln(n) / (epsilon^2 / 2 - epsilon^3 / 3)
        # where n is the number of points (approximated by complexity)

        epsilon = target_error
        jl_factor = 4 * math.log(max(data_complexity, 2)) / (epsilon**2 / 2 - epsilon**3 / 3)

        # Apply modality and privacy adjustments
        privacy_mult = self.privacy_multipliers.get(privacy_level, 1.0)
        adjusted_dim = int(base_dim * privacy_mult * math.sqrt(jl_factor / base_dim))

        # Ensure dimension is within reasonable bounds
        min_dim = 1000
        max_dim = 50000

        return max(min_dim, min(max_dim, adjusted_dim))

    def estimate_data_complexity(self, data_tensor: torch.Tensor) -> float:
        """Estimate intrinsic dimensionality of data using PCA"""
        if data_tensor.numel() == 0:
            return 1.0

        # Center the data
        centered = data_tensor - data_tensor.mean(dim=0)

        # Compute SVD
        try:
            U, S, V = torch.svd(centered)

            # Estimate intrinsic dimension using 95% variance threshold
            total_variance = torch.sum(S**2)
            cumulative_variance = torch.cumsum(S**2, dim=0)
            variance_ratio = cumulative_variance / total_variance

            # Find dimension that captures 95% of variance
            intrinsic_dim = torch.sum(variance_ratio < 0.95).item() + 1

            return float(max(intrinsic_dim, 1))

        except Exception:
            # Fallback to simple heuristic
            return float(min(data_tensor.shape))


class HierarchicalHypervectorEncoder(nn.Module):
    """
    Enhanced hierarchical hypervector encoder with multi-resolution support

    Implements the full pipeline from KAN-HD insights:
    1. Feature extraction layer
    2. Domain-specific projections
    3. Multi-resolution vectors
    4. Adaptive compression
    """

    def __init__(self, base_dim: int = 10000, enable_adaptive_dim: bool = True):
        super().__init__()

        self.base_dim = base_dim
        self.enable_adaptive_dim = enable_adaptive_dim
        self.dim_calculator = AdaptiveDimensionalityCalculator()

        # Multi-modal feature extractors
        self.feature_extractors = nn.ModuleDict(
            {
                "variants": self._create_variant_extractor(),
                "expression": self._create_expression_extractor(),
                "epigenetic": self._create_epigenetic_extractor(),
                "proteomic": self._create_proteomic_extractor(),
                "phenotypic": self._create_phenotypic_extractor(),
                "structural": self._create_structural_extractor(),
            }
        )

        # Domain-specific KAN projections (learnable functions φ_{q,p})
        self.domain_projections = nn.ModuleDict(
            {
                "genomic_variants": LinearKAN(base_dim, 10000),
                "oncology": LinearKAN(base_dim, 10000),
                "rare_disease": LinearKAN(base_dim, 15000),
                "population_genetics": LinearKAN(base_dim, 20000),
                "structural_biology": LinearKAN(base_dim, 25000),
            }
        )

        # Multi-resolution generators
        self.resolution_generators = nn.ModuleDict(
            {
                "base": LinearKAN(base_dim, 10000),  # Base-level vectors
                "mid": LinearKAN(base_dim, 15000),  # Mid-level vectors
                "high": LinearKAN(base_dim, 20000),  # High-level vectors
            }
        )

        # Adaptive compression networks
        self.adaptive_compressors = nn.ModuleDict(
            {
                modality.value: self._create_adaptive_compressor(modality)
                for modality in DataModality
            }
        )

        # Interpretability extractors (for scientific discovery)
        self.interpretability_analyzer = LinearKAN(base_dim, 128)

    def _create_variant_extractor(self) -> nn.Module:
        """Create feature extractor for genomic variants"""
        return nn.Sequential(
            LinearKAN(4, 64),  # A,T,G,C encoding
            nn.BatchNorm1d(64),
            LinearKAN(64, 256),
            nn.Dropout(0.1),
            LinearKAN(256, self.base_dim),
        )

    def _create_expression_extractor(self) -> nn.Module:
        """Create feature extractor for gene expression data"""
        return nn.Sequential(
            LinearKAN(20000, 5000),  # Typical gene count
            nn.BatchNorm1d(5000),
            LinearKAN(5000, 1000),
            nn.Dropout(0.2),
            LinearKAN(1000, self.base_dim),
        )

    def _create_epigenetic_extractor(self) -> nn.Module:
        """Create feature extractor for epigenetic data"""
        return nn.Sequential(
            LinearKAN(1000, 500),  # CpG sites
            nn.BatchNorm1d(500),
            LinearKAN(500, self.base_dim),
            nn.Tanh(),  # Methylation values are bounded
        )

    def _create_proteomic_extractor(self) -> nn.Module:
        """Create feature extractor for proteomic data"""
        return nn.Sequential(
            LinearKAN(5000, 1000),  # Protein count
            nn.BatchNorm1d(1000),
            LinearKAN(1000, self.base_dim),
            nn.ReLU(),  # Protein levels are non-negative
        )

    def _create_phenotypic_extractor(self) -> nn.Module:
        """Create feature extractor for phenotypic data"""
        return nn.Sequential(
            LinearKAN(100, 256),  # Clinical measurements
            LinearKAN(256, 512),
            LinearKAN(512, self.base_dim),
        )

    def _create_structural_extractor(self) -> nn.Module:
        """Create feature extractor for 3D structural data"""
        return nn.Sequential(
            LinearKAN(3, 128),  # 3D coordinates
            LinearKAN(128, 512),
            LinearKAN(512, 1024),
            LinearKAN(1024, self.base_dim),
        )

    def _create_adaptive_compressor(self, modality: DataModality) -> nn.Module:
        """Create adaptive compressor for specific modality"""
        base_dim = self.dim_calculator.base_dimensions[modality]
        compressed_dim = base_dim // 100  # 100x compression target

        return nn.Sequential(
            LinearKAN(base_dim, base_dim // 2),
            nn.BatchNorm1d(base_dim // 2),
            LinearKAN(base_dim // 2, base_dim // 10),
            nn.Dropout(0.1),
            LinearKAN(base_dim // 10, compressed_dim),
            nn.Tanh(),  # Bounded output for stable compression
        )

    def encode_multimodal_data(
        self, data_dict: Dict[str, torch.Tensor], specifications: Dict[str, EncodingSpecification]
    ) -> Dict[str, MultiResolutionVector]:
        """
        Encode multi-modal genomic data with hierarchical resolution

        Args:
            data_dict: Dictionary mapping modality names to data tensors
            specifications: Encoding specifications for each modality

        Returns:
            Dictionary of multi-resolution vectors for each modality
        """
        encoded_vectors = {}

        for modality_name, data_tensor in data_dict.items():
            if modality_name not in specifications:
                continue

            spec = specifications[modality_name]

            # Step 1: Extract features using modality-specific extractor
            if modality_name in self.feature_extractors:
                extracted_features = self.feature_extractors[modality_name](data_tensor)
            else:
                # Default linear projection
                extracted_features = nn.Linear(data_tensor.shape[-1], self.base_dim)(data_tensor)

            # Step 2: Apply domain-specific projections
            domain_projected = self._apply_domain_projection(extracted_features, spec.modality)

            # Step 3: Generate multi-resolution vectors
            multi_res_vector = self._generate_multiresolution_vector(domain_projected, spec)

            # Step 4: Apply adaptive compression if needed
            if spec.compression_ratio > 1.0:
                multi_res_vector = self._apply_adaptive_compression(multi_res_vector, spec)

            encoded_vectors[modality_name] = multi_res_vector

        return encoded_vectors

    def _apply_domain_projection(
        self, features: torch.Tensor, modality: DataModality
    ) -> torch.Tensor:
        """Apply domain-specific KAN projection"""

        # Map modality to domain projection
        domain_mapping = {
            DataModality.GENOMIC_VARIANTS: "genomic_variants",
            DataModality.GENE_EXPRESSION: "oncology",  # Common use case
            DataModality.EPIGENETIC: "rare_disease",
            DataModality.PROTEOMIC: "population_genetics",
            DataModality.STRUCTURAL: "structural_biology",
        }

        domain_key = domain_mapping.get(modality, "genomic_variants")

        if domain_key in self.domain_projections:
            return self.domain_projections[domain_key](features)
        else:
            return features

    def _generate_multiresolution_vector(
        self, features: torch.Tensor, spec: EncodingSpecification
    ) -> MultiResolutionVector:
        """Generate multi-resolution hypervector"""

        # Generate different resolution levels
        base_vector = self.resolution_generators["base"](features)
        mid_vector = self.resolution_generators["mid"](features)
        high_vector = self.resolution_generators["high"](features)

        # Adaptive dimensionality if enabled
        optimal_dim = None
        if self.enable_adaptive_dim:
            complexity = self.dim_calculator.estimate_data_complexity(features)
            optimal_dim = self.dim_calculator.calculate_optimal_dimension(
                spec.modality, complexity, spec.privacy_level
            )

            # Adjust vectors to optimal dimension
            base_vector = self._adjust_vector_dimension(base_vector, optimal_dim)
            mid_vector = self._adjust_vector_dimension(mid_vector, optimal_dim)
            high_vector = self._adjust_vector_dimension(high_vector, optimal_dim)

        return MultiResolutionVector(
            base_vector=base_vector,
            mid_vector=mid_vector,
            high_vector=high_vector,
            modality=spec.modality,
            compression_metadata={
                "target_compression": spec.compression_ratio,
                "privacy_level": spec.privacy_level,
                "optimal_dimension": optimal_dim,
            },
        )

    def _adjust_vector_dimension(self, vector: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Adjust vector to target dimension"""
        current_dim = vector.shape[-1]

        if current_dim == target_dim:
            return vector
        elif current_dim < target_dim:
            # Pad with structured noise (not zeros for better properties)
            padding_size = target_dim - current_dim
            structured_padding = (
                torch.randn(*vector.shape[:-1], padding_size, device=vector.device) * 0.1
            )
            return torch.cat([vector, structured_padding], dim=-1)
        else:
            # Intelligent truncation using SVD
            return self._svd_truncate(vector, target_dim)

    def _svd_truncate(self, vector: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Truncate vector using SVD to preserve most important components"""
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)

        U, S, V = torch.svd(vector)

        # Keep top components
        truncated = U[:, :target_dim] @ torch.diag(S[:target_dim])

        return truncated.squeeze(0) if truncated.shape[0] == 1 else truncated

    def _apply_adaptive_compression(
        self, multi_res_vector: MultiResolutionVector, spec: EncodingSpecification
    ) -> MultiResolutionVector:
        """Apply adaptive compression based on modality"""

        modality_key = spec.modality.value
        if modality_key in self.adaptive_compressors:
            compressor = self.adaptive_compressors[modality_key]

            # Compress each resolution level
            compressed_base = compressor(multi_res_vector.base_vector.unsqueeze(0)).squeeze(0)
            compressed_mid = compressor(multi_res_vector.mid_vector.unsqueeze(0)).squeeze(0)
            compressed_high = compressor(multi_res_vector.high_vector.unsqueeze(0)).squeeze(0)

            return MultiResolutionVector(
                base_vector=compressed_base,
                mid_vector=compressed_mid,
                high_vector=compressed_high,
                modality=spec.modality,
                compression_metadata={
                    **multi_res_vector.compression_metadata,
                    "compression_applied": True,
                    "original_dimensions": {
                        "base": multi_res_vector.base_vector.shape[-1],
                        "mid": multi_res_vector.mid_vector.shape[-1],
                        "high": multi_res_vector.high_vector.shape[-1],
                    },
                },
            )

        return multi_res_vector

    def bind_multimodal_vectors(
        self, vectors: Dict[str, MultiResolutionVector], binding_strategy: str = "hierarchical"
    ) -> MultiResolutionVector:
        """
        Bind multiple modality vectors into unified representation

        Args:
            vectors: Dictionary of multi-resolution vectors
            binding_strategy: Strategy for binding ('hierarchical', 'weighted', 'attention')

        Returns:
            Bound multi-resolution vector
        """
        if not vectors:
            raise ValueError("No vectors provided for binding")

        if len(vectors) == 1:
            return list(vectors.values())[0]

        vector_list = list(vectors.values())

        if binding_strategy == "hierarchical":
            return self._hierarchical_binding(vector_list)
        elif binding_strategy == "weighted":
            return self._weighted_binding(vector_list)
        elif binding_strategy == "attention":
            return self._attention_binding(vector_list)
        else:
            raise ValueError(f"Unknown binding strategy: {binding_strategy}")

    def _hierarchical_binding(self, vectors: List[MultiResolutionVector]) -> MultiResolutionVector:
        """Hierarchical binding using circular convolution"""

        # Find common dimension for binding
        base_dims = [v.base_vector.shape[-1] for v in vectors]
        common_dim = min(base_dims)

        # Bind base vectors
        bound_base = vectors[0].base_vector[:common_dim]
        for vector in vectors[1:]:
            vec_truncated = vector.base_vector[:common_dim]
            bound_base = self._circular_convolution(bound_base, vec_truncated)

        # Bind mid vectors
        mid_dims = [v.mid_vector.shape[-1] for v in vectors]
        common_mid_dim = min(mid_dims)

        bound_mid = vectors[0].mid_vector[:common_mid_dim]
        for vector in vectors[1:]:
            vec_truncated = vector.mid_vector[:common_mid_dim]
            bound_mid = self._circular_convolution(bound_mid, vec_truncated)

        # Bind high vectors
        high_dims = [v.high_vector.shape[-1] for v in vectors]
        common_high_dim = min(high_dims)

        bound_high = vectors[0].high_vector[:common_high_dim]
        for vector in vectors[1:]:
            vec_truncated = vector.high_vector[:common_high_dim]
            bound_high = self._circular_convolution(bound_high, vec_truncated)

        # Combine metadata
        combined_metadata = {}
        for vector in vectors:
            combined_metadata.update(vector.compression_metadata)

        return MultiResolutionVector(
            base_vector=bound_base,
            mid_vector=bound_mid,
            high_vector=bound_high,
            modality=DataModality.GENOMIC_VARIANTS,  # Generic after binding
            compression_metadata=combined_metadata,
        )

    def _circular_convolution(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """Circular convolution for hypervector binding"""
        fft1 = torch.fft.fft(vec1.float())
        fft2 = torch.fft.fft(vec2.float())
        result_fft = fft1 * fft2
        result = torch.fft.ifft(result_fft).real

        # Normalize to preserve magnitude
        return result / torch.norm(result) * torch.norm(vec1)

    def _weighted_binding(self, vectors: List[MultiResolutionVector]) -> MultiResolutionVector:
        """Binding with learned attention weights"""
        # Simplified implementation - could be enhanced with attention mechanism
        weights = torch.softmax(torch.randn(len(vectors)), dim=0)

        # Weighted sum for each resolution
        base_weighted = sum(w * v.base_vector for w, v in zip(weights, vectors))
        mid_weighted = sum(w * v.mid_vector for w, v in zip(weights, vectors))
        high_weighted = sum(w * v.high_vector for w, v in zip(weights, vectors))

        return MultiResolutionVector(
            base_vector=base_weighted,
            mid_vector=mid_weighted,
            high_vector=high_weighted,
            modality=DataModality.GENOMIC_VARIANTS,
            compression_metadata={"binding_strategy": "weighted", "weights": weights.tolist()},
        )

    def _attention_binding(self, vectors: List[MultiResolutionVector]) -> MultiResolutionVector:
        """Attention-based binding (simplified version)"""
        # This could be enhanced with a proper attention mechanism
        # For now, use similarity-based weighting

        if len(vectors) < 2:
            return vectors[0]

        # Compute pairwise similarities
        similarities = []
        for i, vec_i in enumerate(vectors):
            for j, vec_j in enumerate(vectors):
                if i != j:
                    sim = torch.cosine_similarity(
                        vec_i.base_vector.unsqueeze(0), vec_j.base_vector.unsqueeze(0)
                    ).item()
                    similarities.append(sim)

        # Use average similarity as weight
        avg_sim = np.mean(similarities)
        weights = torch.softmax(torch.tensor([avg_sim] * len(vectors)), dim=0)

        # Weighted combination
        base_combined = sum(w * v.base_vector for w, v in zip(weights, vectors))
        mid_combined = sum(w * v.mid_vector for w, v in zip(weights, vectors))
        high_combined = sum(w * v.high_vector for w, v in zip(weights, vectors))

        return MultiResolutionVector(
            base_vector=base_combined,
            mid_vector=mid_combined,
            high_vector=high_combined,
            modality=DataModality.GENOMIC_VARIANTS,
            compression_metadata={"binding_strategy": "attention", "avg_similarity": avg_sim},
        )

    def extract_interpretable_patterns(
        self, multi_res_vector: MultiResolutionVector
    ) -> Dict[str, Any]:
        """
        Extract interpretable patterns from the encoded representation

        Implements the scientific interpretability insight from KAN-HD.
        """
        # Analyze base vector patterns
        base_analysis = self.interpretability_analyzer(
            multi_res_vector.base_vector.unsqueeze(0)
        ).squeeze(0)

        # Extract dominant frequencies (biological rhythms, periodic patterns)
        base_fft = torch.fft.fft(multi_res_vector.base_vector.float())
        dominant_frequencies = torch.argsort(torch.abs(base_fft), descending=True)[:10]

        # Compute sparsity (interpretability indicator)
        sparsity = (
            torch.sum(torch.abs(multi_res_vector.base_vector) < 0.1).item()
            / multi_res_vector.base_vector.numel()
        )

        # Identify cluster structure
        cluster_analysis = self._analyze_cluster_structure(multi_res_vector.base_vector)

        return {
            "dominant_frequencies": dominant_frequencies.tolist(),
            "sparsity_ratio": sparsity,
            "interpretability_score": torch.mean(torch.abs(base_analysis)).item(),
            "cluster_structure": cluster_analysis,
            "modality": multi_res_vector.modality.value,
            "vector_norms": {
                "base": torch.norm(multi_res_vector.base_vector).item(),
                "mid": torch.norm(multi_res_vector.mid_vector).item(),
                "high": torch.norm(multi_res_vector.high_vector).item(),
            },
        }

    def _analyze_cluster_structure(self, vector: torch.Tensor) -> Dict[str, Any]:
        """Analyze clustering structure in the hypervector"""
        # Simple clustering analysis using k-means approximation
        vector_reshaped = vector.reshape(-1, 1)  # noqa: F841

        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(vector) - 1):
            if vector[i] > vector[i - 1] and vector[i] > vector[i + 1] and vector[i] > 0.5:
                peaks.append(i)

        # Estimate number of clusters based on peaks
        num_clusters = len(peaks)

        return {
            "estimated_clusters": num_clusters,
            "peak_positions": peaks[:10],  # Top 10 peaks
            "clustering_score": min(num_clusters / 10.0, 1.0),  # Normalized score
        }
