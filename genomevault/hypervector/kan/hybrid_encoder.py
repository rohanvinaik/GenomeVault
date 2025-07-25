"""
KAN-HD Hybrid Encoder

Implements the hybrid architecture that combines KAN compression
with hyperdimensional computing for privacy-preserving genomic storage.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..encoding.genomic import GenomicEncoder
from .compression import AdaptiveKANCompressor, KANCompressor
from .kan_layer import KANLayer, LinearKAN


class EncodingLevel(Enum):
    """Hierarchical encoding levels"""

    BASE = "base"  # 10,000D - Genomic variants
    MID = "mid"  # 15,000D - Expression data
    HIGH = "high"  # 20,000D - Epigenetic data
    FULL = "full"  # All levels combined


class KANHybridEncoder(nn.Module):
    """
    Hybrid encoder combining KAN compression with HD computing

    Architecture:
    1. KAN compression: Reduces dimensionality while learning patterns
    2. HD projection: Maps to hyperdimensional space for privacy
    3. Hierarchical binding: Combines multiple data modalities
    """

    def __init__(self, base_dim: int = 10000, compressed_dim: int = 100, use_adaptive: bool = True):
        super().__init__()

        # Initialize base components
        self.base_dim = base_dim
        self.compressed_dim = compressed_dim

        # HD encoder for privacy
        self.hd_encoder = GenomicEncoder(dimension=base_dim)

        # KAN compressor
        if use_adaptive:
            self.kan_compressor = AdaptiveKANCompressor(input_dim=base_dim)
        else:
            self.kan_compressor = KANCompressor(
                input_dim=base_dim, compressed_dim=compressed_dim, num_layers=3
            )

        # Hierarchical projections for different data types
        self.projections = nn.ModuleDict(
            {
                "genomic": self._create_projection(base_dim, base_dim),
                "expression": self._create_projection(base_dim, 15000),
                "epigenetic": self._create_projection(base_dim, 20000),
            }
        )

        # Privacy-preserving mixer
        self.privacy_mixer = self._create_privacy_mixer()

    def _create_projection(self, in_dim: int, out_dim: int) -> nn.Module:
        """Create domain-specific projection using KAN"""
        return nn.Sequential(
            LinearKAN(in_dim, (in_dim + out_dim) // 2),
            nn.Tanh(),
            LinearKAN((in_dim + out_dim) // 2, out_dim),
            nn.Tanh(),
        )

    def _create_privacy_mixer(self) -> nn.Module:
        """Create privacy-preserving mixing network"""
        return nn.Sequential(
            nn.Linear(self.compressed_dim, self.compressed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.compressed_dim * 2, self.compressed_dim),
            nn.Tanh(),
        )

    def encode_genomic_data(self, variants: List[Dict], compress: bool = True) -> torch.Tensor:
        """
        Encode genomic variants with optional compression

        Args:
            variants: List of variant dictionaries
            compress: Whether to apply KAN compression

        Returns:
            Encoded representation
        """
        # First, encode variants to hypervectors
        hd_vector = self.hd_encoder.encode_genome(variants)

        if not compress:
            return hd_vector

        # Apply KAN compression
        hd_vector_batch = hd_vector.unsqueeze(0)  # Add batch dimension

        if hasattr(self.kan_compressor, "analyze_complexity"):
            compressed, level = self.kan_compressor.encode(hd_vector_batch)
            self.last_compression_level = level
        else:
            compressed = self.kan_compressor.encode(hd_vector_batch)
            self.last_compression_level = "default"

        # Apply privacy mixing
        private_compressed = self.privacy_mixer(compressed)

        return private_compressed.squeeze(0)

    def encode_hierarchical(
        self,
        genomic_data: Optional[List[Dict]] = None,
        expression_data: Optional[torch.Tensor] = None,
        epigenetic_data: Optional[torch.Tensor] = None,
        level: EncodingLevel = EncodingLevel.FULL,
    ) -> torch.Tensor:
        """
        Hierarchical encoding of multi-modal genomic data

        Args:
            genomic_data: Variant list
            expression_data: Gene expression tensor
            epigenetic_data: Epigenetic modification tensor
            level: Encoding level to use

        Returns:
            Hierarchically encoded hypervector
        """
        encoded_components = []

        # Level 1: Genomic variants (base level)
        if genomic_data is not None and level.value in ["base", "full"]:
            genomic_hv = self.encode_genomic_data(genomic_data, compress=True)
            genomic_projected = self.projections["genomic"](genomic_hv.unsqueeze(0)).squeeze(0)
            encoded_components.append(genomic_projected)

        # Level 2: Expression data (mid level)
        if expression_data is not None and level.value in ["mid", "full"]:
            # Compress expression data
            expr_compressed = self.kan_compressor.encode(expression_data.unsqueeze(0)).squeeze(0)

            # Project to higher dimension
            expr_projected = self.projections["expression"](
                self._pad_or_truncate(expr_compressed, self.base_dim).unsqueeze(0)
            ).squeeze(0)
            encoded_components.append(expr_projected)

        # Level 3: Epigenetic data (high level)
        if epigenetic_data is not None and level.value in ["high", "full"]:
            # Compress epigenetic data
            epi_compressed = self.kan_compressor.encode(epigenetic_data.unsqueeze(0)).squeeze(0)

            # Project to highest dimension
            epi_projected = self.projections["epigenetic"](
                self._pad_or_truncate(epi_compressed, self.base_dim).unsqueeze(0)
            ).squeeze(0)
            encoded_components.append(epi_projected)

        # Combine all components
        if not encoded_components:
            return torch.zeros(self.base_dim)

        # Hierarchical binding using circular convolution
        combined = encoded_components[0]
        for component in encoded_components[1:]:
            # Ensure same dimension for binding
            component_resized = self._pad_or_truncate(component, combined.shape[0])
            combined = self._bind_vectors(combined, component_resized)

        # Final normalization
        combined = combined / torch.norm(combined)

        return combined

    def _bind_vectors(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """Bind two hypervectors using circular convolution"""
        return torch.fft.irfft(torch.fft.rfft(vec1) * torch.fft.rfft(vec2))

    def _pad_or_truncate(self, tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Pad or truncate tensor to target dimension"""
        current_dim = tensor.shape[-1]

        if current_dim == target_dim:
            return tensor
        elif current_dim < target_dim:
            # Pad with zeros
            padding = torch.zeros(target_dim - current_dim, device=tensor.device)
            return torch.cat([tensor, padding])
        else:
            # Truncate
            return tensor[:target_dim]

    def decode_genomic_data(
        self, compressed: torch.Tensor, level: Optional[str] = None
    ) -> torch.Tensor:
        """
        Decode compressed representation back to hypervector

        Args:
            compressed: Compressed representation
            level: Compression level (for adaptive compressor)

        Returns:
            Reconstructed hypervector
        """
        compressed_batch = compressed.unsqueeze(0)

        if hasattr(self.kan_compressor, "decode") and level:
            reconstructed = self.kan_compressor.decode(compressed_batch, level)
        else:
            reconstructed = self.kan_compressor.decode(compressed_batch)

        return reconstructed.squeeze(0)

    def compute_privacy_guarantee(
        self, original: torch.Tensor, encoded: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute privacy metrics for the encoding

        Returns:
            Dictionary with privacy metrics
        """
        # Information leakage (mutual information approximation)
        original_entropy = self._estimate_entropy(original)
        encoded_entropy = self._estimate_entropy(encoded)

        # Correlation analysis
        if original.shape == encoded.shape:
            correlation = torch.corrcoef(torch.stack([original.flatten(), encoded.flatten()]))[
                0, 1
            ].item()
        else:
            correlation = 0.0

        # Reconstruction difficulty (using random projections)
        num_trials = 100
        successful_reconstructions = 0

        for _ in range(num_trials):
            # Try to reconstruct with random decoder
            random_decoder = torch.randn(encoded.shape[0], original.shape[0])
            attempted_reconstruction = torch.matmul(encoded.unsqueeze(0), random_decoder.T).squeeze(
                0
            )

            # Check if reconstruction is close
            if (
                torch.cosine_similarity(
                    original.unsqueeze(0), attempted_reconstruction.unsqueeze(0)
                ).item()
                > 0.9
            ):
                successful_reconstructions += 1

        reconstruction_difficulty = 1.0 - (successful_reconstructions / num_trials)

        return {
            "information_preservation": encoded_entropy / original_entropy,
            "correlation": abs(correlation),
            "reconstruction_difficulty": reconstruction_difficulty,
            "privacy_score": reconstruction_difficulty * (1 - abs(correlation)),
        }

    def _estimate_entropy(self, tensor: torch.Tensor) -> float:
        """Estimate entropy of tensor using histogram method"""
        # Quantize to 256 levels
        quantized = torch.floor(
            (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8) * 255
        )

        # Compute histogram
        hist = torch.histc(quantized, bins=256, min=0, max=255)
        hist = hist / hist.sum()  # Normalize

        # Compute entropy
        entropy = -torch.sum(hist * torch.log2(hist + 1e-10))

        return entropy.item()

    def save_compressed(self, compressed: torch.Tensor, metadata: Dict, filepath: str):
        """Save compressed representation with metadata"""
        torch.save(
            {
                "compressed": compressed,
                "metadata": metadata,
                "compression_level": getattr(self, "last_compression_level", "default"),
                "model_state": self.state_dict(),
            },
            filepath,
        )

    def load_compressed(self, filepath: str) -> Tuple[torch.Tensor, Dict]:
        """Load compressed representation and metadata"""
        checkpoint = torch.load(filepath)

        # Restore model state if needed
        if "model_state" in checkpoint:
            self.load_state_dict(checkpoint["model_state"])

        return checkpoint["compressed"], checkpoint["metadata"]


class StreamingKANHybridEncoder(KANHybridEncoder):
    """
    Streaming variant for processing large genomic datasets

    Processes data in chunks to handle genome-scale data efficiently.
    """

    def __init__(
        self, base_dim: int = 10000, compressed_dim: int = 100, chunk_size: int = 1000000
    ):  # 1M variants per chunk
        super().__init__(base_dim, compressed_dim)
        self.chunk_size = chunk_size

        # Streaming aggregator
        self.aggregator = nn.Sequential(LinearKAN(compressed_dim * 2, compressed_dim), nn.Tanh())

    def encode_genome_streaming(
        self, variant_iterator, progress_callback: Optional[callable] = None
    ) -> torch.Tensor:
        """
        Encode genome data in streaming fashion

        Args:
            variant_iterator: Iterator over variants
            progress_callback: Optional callback for progress updates

        Returns:
            Final compressed representation
        """
        chunk_vectors = []
        variants_processed = 0
        current_chunk = []

        for variant in variant_iterator:
            current_chunk.append(variant)

            if len(current_chunk) >= self.chunk_size:
                # Process chunk
                chunk_vector = self.encode_genomic_data(current_chunk)
                chunk_vectors.append(chunk_vector)

                variants_processed += len(current_chunk)
                current_chunk = []

                if progress_callback:
                    progress_callback(variants_processed)

        # Process remaining variants
        if current_chunk:
            chunk_vector = self.encode_genomic_data(current_chunk)
            chunk_vectors.append(chunk_vector)
            variants_processed += len(current_chunk)

        # Aggregate all chunks
        if not chunk_vectors:
            return torch.zeros(self.compressed_dim)

        # Progressive aggregation to avoid memory issues
        aggregated = chunk_vectors[0]
        for chunk in chunk_vectors[1:]:
            combined = torch.cat([aggregated, chunk])
            aggregated = self.aggregator(combined.unsqueeze(0)).squeeze(0)

        return aggregated
