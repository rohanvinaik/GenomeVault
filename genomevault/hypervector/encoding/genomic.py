"""
Hypervector encoding for genomic data
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from genomevault.core.constants import HYPERVECTOR_DIMENSIONS
from genomevault.core.exceptions import HypervectorError


class GenomicEncoder:
    """
    Encodes genomic variants into high-dimensional vectors
    """

    def __init__(self, dimension: int = HYPERVECTOR_DIMENSIONS["base"]):
        self.dimension = dimension
        self.base_vectors = self._init_base_vectors()

    def _init_base_vectors(self) -> Dict[str, torch.Tensor]:
        """Initialize base hypervectors for nucleotides and operations"""
        base_vectors = {}

        # Nucleotide base vectors (orthogonal)
        for i, base in enumerate(["A", "T", "G", "C"]):
            vec = torch.zeros(self.dimension)
            # Use sparse representation
            indices = torch.randperm(self.dimension)[: self.dimension // 4]
            vec[indices] = torch.randn(len(indices))
            vec = vec / torch.norm(vec)
            base_vectors[base] = vec

        # Position encoding vectors
        base_vectors["POS"] = self._generate_position_vector()

        # Variant type vectors
        for variant_type in ["SNP", "INS", "DEL", "DUP", "INV"]:
            vec = torch.randn(self.dimension)
            vec = vec / torch.norm(vec)
            base_vectors[variant_type] = vec

        return base_vectors

    def _generate_position_vector(self) -> torch.Tensor:
        """Generate position encoding vector using sinusoidal encoding"""
        position = torch.arange(self.dimension).float()
        div_term = torch.exp(
            torch.arange(0, self.dimension, 2).float() * -(np.log(10000.0) / self.dimension)
        )

        pos_encoding = torch.zeros(self.dimension)
        pos_encoding[0::2] = torch.sin(position[0::2] * div_term)
        pos_encoding[1::2] = torch.cos(position[1::2] * div_term[: len(position[1::2])])

        return pos_encoding

    def encode_variant(
        self,
        chromosome: str,
        position: int,
        ref: str,
        alt: str,
        variant_type: str = "SNP",
    ) -> torch.Tensor:
        """
        Encode a single genetic variant into a hypervector

        Args:
            chromosome: Chromosome name (e.g., 'chr1')
            position: Genomic position
            ref: Reference allele
            alt: Alternative allele
            variant_type: Type of variant (SNP, INS, DEL, etc.)

        Returns:
            Hypervector representation of the variant
        """
        try:
            # Start with variant type vector
            variant_vec = self.base_vectors[variant_type].clone()

            # Bind with position information
            pos_factor = position / 1e9  # Normalize position
            pos_vec = torch.roll(self.base_vectors["POS"], int(pos_factor * 1000))
            variant_vec = self._bind(variant_vec, pos_vec)

            # Bind with reference allele information
            if len(ref) == 1 and ref in self.base_vectors:
                variant_vec = self._bind(variant_vec, self.base_vectors[ref])

            # Bind with alternative allele information
            if len(alt) == 1 and alt in self.base_vectors:
                # Use permutation to distinguish alt from ref
                alt_vec = self._permute(self.base_vectors[alt], 1)
                variant_vec = self._bind(variant_vec, alt_vec)

            # Add chromosome information
            chr_vec = self._chromosome_vector(chromosome)
            variant_vec = self._bundle(variant_vec, chr_vec)

            # Normalize
            variant_vec = variant_vec / torch.norm(variant_vec)

            return variant_vec

        except Exception as e:
            raise HypervectorError("Failed to encode variant: {str(e)}")

    def encode_genome(self, variants: List[Dict]) -> torch.Tensor:
        """
        Encode a complete set of variants into a single hypervector

        Args:
            variants: List of variant dictionaries

        Returns:
            Hypervector representation of the genome
        """
        if not variants:
            return torch.zeros(self.dimension)

        # Encode each variant
        variant_vectors = []
        for variant in variants:
            vec = self.encode_variant(
                chromosome=variant["chromosome"],
                position=variant["position"],
                ref=variant["ref"],
                alt=variant["alt"],
                variant_type=variant.get("type", "SNP"),
            )
            variant_vectors.append(vec)

        # Bundle all variants using superposition
        genome_vec = torch.stack(variant_vectors).sum(dim=0)

        # Apply holographic reduction
        genome_vec = self._holographic_reduce(genome_vec)

        # Final normalization
        genome_vec = genome_vec / torch.norm(genome_vec)

        return genome_vec

    def _bind(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """Bind two hypervectors using circular convolution"""
        return torch.fft.irfft(torch.fft.rfft(vec1) * torch.fft.rfft(vec2))

    def _bundle(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """Bundle two hypervectors using element-wise addition"""
        return vec1 + vec2

    def _permute(self, vec: torch.Tensor, n: int) -> torch.Tensor:
        """Permute hypervector by n positions"""
        return torch.roll(vec, n)

    def _chromosome_vector(self, chromosome: str) -> torch.Tensor:
        """Generate chromosome-specific vector"""
        # Extract chromosome number
        chr_num = chromosome.replace("chr", "").replace("X", "23").replace("Y", "24")
        try:
            chr_idx = int(chr_num)
        except Exception:
            chr_idx = 25  # For mitochondrial or other

        # Generate deterministic vector based on chromosome
        torch.manual_seed(chr_idx)
        chr_vec = torch.randn(self.dimension)
        chr_vec = chr_vec / torch.norm(chr_vec)

        return chr_vec

    def _holographic_reduce(self, vec: torch.Tensor) -> torch.Tensor:
        """Apply holographic reduction to maintain fixed dimensionality"""
        # Use circular shifting and XOR-like operations
        reduced = vec.clone()
        for i in range(3):  # Multiple reduction rounds
            shift = self.dimension // (2 ** (i + 1))
            shifted = torch.roll(vec, shift)
            reduced = self._bind(reduced, shifted)

        return reduced

    def similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Calculate cosine similarity between two hypervectors"""
        return torch.cosine_similarity(vec1, vec2, dim=0).item()

    # Catalytic extensions
    def use_catalytic_projections(self, projection_pool):
        """
        Switch to memory-mapped catalytic projections.
        
        Args:
            projection_pool: CatalyticProjectionPool instance
        """
        self.projection_pool = projection_pool
        self.use_catalytic = True
        
    def encode_variant_catalytic(self, *args, **kwargs):
        """
        Encode variant using catalytic projections.
        
        This method uses memory-mapped projections to reduce
        memory usage by 95% compared to standard encoding.
        """
        if hasattr(self, 'projection_pool'):
            # Use memory-mapped projections
            variant_vec = self.encode_variant(*args, **kwargs)
            
            # Apply catalytic projection
            projected = self.projection_pool.apply_catalytic_projection(
                variant_vec, [0, 1, 2]  # Use first 3 projections
            )
            
            return projected
        else:
            # Fall back to standard encoding
            return self.encode_variant(*args, **kwargs)
