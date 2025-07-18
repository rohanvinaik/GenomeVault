"""
Tiered compression for hypervector representations
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json
import lzma
import pickle

from core.constants import CompressionTier, OmicsType
from core.exceptions import CompressionError


@dataclass
class CompressedHypervector:
    """Compressed hypervector representation"""
    tier: CompressionTier
    omics_type: OmicsType
    compressed_data: bytes
    metadata: Dict[str, any]
    original_dimension: int
    compressed_size: int
    
    def save(self, path):
        """Save compressed data to file"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """Load compressed data from file"""
        with open(path, 'rb') as f:
            return pickle.load(f)


class TieredCompressor:
    """
    Implements hierarchical compression for genomic hypervectors
    """
    
    def __init__(self, tier: CompressionTier = CompressionTier.CLINICAL):
        self.tier = tier
        self.compression_configs = self._get_compression_configs()
        
    def _get_compression_configs(self) -> Dict[CompressionTier, Dict]:
        """Get compression parameters for each tier"""
        return {
            CompressionTier.MINI: {
                "target_size": 25 * 1024,  # 25KB
                "num_components": 5000,     # Top 5000 SNPs
                "compression_ratio": 0.95,
                "use_quantization": True,
                "bits_per_value": 4
            },
            CompressionTier.CLINICAL: {
                "target_size": 300 * 1024,  # 300KB
                "num_components": 120000,   # ACMG + PharmGKB variants
                "compression_ratio": 0.85,
                "use_quantization": True,
                "bits_per_value": 8
            },
            CompressionTier.FULL: {
                "target_size": 200 * 1024,  # 200KB per modality
                "num_components": -1,       # Keep all components
                "compression_ratio": 0.7,
                "use_quantization": False,
                "bits_per_value": 16
            }
        }
    
    def compress(self, data: Union[torch.Tensor, Dict], 
                omics_type: OmicsType) -> CompressedHypervector:
        """
        Compress hypervector data according to tier
        
        Args:
            data: Hypervector tensor or dict of processing results
            omics_type: Type of omics data
            
        Returns:
            Compressed representation
        """
        config = self.compression_configs[self.tier]
        
        # Extract hypervector if dict
        if isinstance(data, dict):
            hypervector = data.get("hypervector")
            metadata = data
        else:
            hypervector = data
            metadata = {}
        
        if hypervector is None:
            raise CompressionError("No hypervector found in data")
        
        # Apply tier-specific compression
        if self.tier == CompressionTier.MINI:
            compressed = self._compress_mini(hypervector, config)
        elif self.tier == CompressionTier.CLINICAL:
            compressed = self._compress_clinical(hypervector, config, omics_type)
        else:  # FULL
            compressed = self._compress_full(hypervector, config)
        
        # Apply additional lossless compression
        final_compressed = lzma.compress(compressed, preset=6)
        
        return CompressedHypervector(
            tier=self.tier,
            omics_type=omics_type,
            compressed_data=final_compressed,
            metadata=metadata,
            original_dimension=len(hypervector),
            compressed_size=len(final_compressed)
        )
    
    def decompress(self, compressed: CompressedHypervector) -> torch.Tensor:
        """
        Decompress a hypervector
        """
        # Decompress lzma
        decompressed = lzma.decompress(compressed.compressed_data)
        
        # Apply tier-specific decompression
        config = self.compression_configs[compressed.tier]
        
        if compressed.tier == CompressionTier.MINI:
            hypervector = self._decompress_mini(decompressed, compressed, config)
        elif compressed.tier == CompressionTier.CLINICAL:
            hypervector = self._decompress_clinical(decompressed, compressed, config)
        else:  # FULL
            hypervector = self._decompress_full(decompressed, compressed, config)
        
        return hypervector
    
    def _compress_mini(self, hypervector: torch.Tensor, config: Dict) -> bytes:
        """
        Mini tier compression - keep only most important components
        """
        # Get top k components by magnitude
        k = config["num_components"]
        topk_values, topk_indices = torch.topk(torch.abs(hypervector), k)
        
        # Get signs of top k components
        signs = torch.sign(hypervector[topk_indices])
        
        # Quantize values
        if config["use_quantization"]:
            quantized_values = self._quantize(topk_values, config["bits_per_value"])
        else:
            quantized_values = topk_values
        
        # Pack data efficiently
        data = {
            "indices": topk_indices.numpy().astype(np.uint16),
            "values": quantized_values.numpy(),
            "signs": signs.numpy().astype(np.int8)
        }
        
        return pickle.dumps(data)
    
    def _decompress_mini(self, data: bytes, compressed: CompressedHypervector, 
                        config: Dict) -> torch.Tensor:
        """Decompress mini tier data"""
        unpacked = pickle.loads(data)
        
        # Reconstruct sparse hypervector
        hypervector = torch.zeros(compressed.original_dimension)
        
        indices = torch.from_numpy(unpacked["indices"])
        values = torch.from_numpy(unpacked["values"])
        signs = torch.from_numpy(unpacked["signs"])
        
        if config["use_quantization"]:
            values = self._dequantize(values, config["bits_per_value"])
        
        hypervector[indices] = values * signs
        
        return hypervector
    
    def _compress_clinical(self, hypervector: torch.Tensor, config: Dict,
                          omics_type: OmicsType) -> bytes:
        """
        Clinical tier compression - keep clinically relevant variants
        """
        # Get clinical variant indices based on omics type
        clinical_indices = self._get_clinical_indices(omics_type)
        
        # Extract clinical components
        clinical_values = hypervector[clinical_indices]
        
        # Apply transform coding for better compression
        transformed = self._dct_transform(clinical_values)
        
        # Quantize
        if config["use_quantization"]:
            quantized = self._quantize(transformed, config["bits_per_value"])
        else:
            quantized = transformed
        
        data = {
            "clinical_indices": clinical_indices.numpy(),
            "transformed_values": quantized.numpy(),
            "transform_type": "dct"
        }
        
        return pickle.dumps(data)
    
    def _decompress_clinical(self, data: bytes, compressed: CompressedHypervector,
                           config: Dict) -> torch.Tensor:
        """Decompress clinical tier data"""
        unpacked = pickle.loads(data)
        
        # Reconstruct hypervector
        hypervector = torch.zeros(compressed.original_dimension)
        
        clinical_indices = torch.from_numpy(unpacked["clinical_indices"])
        transformed_values = torch.from_numpy(unpacked["transformed_values"])
        
        if config["use_quantization"]:
            transformed_values = self._dequantize(transformed_values, config["bits_per_value"])
        
        # Inverse transform
        clinical_values = self._idct_transform(transformed_values)
        
        hypervector[clinical_indices] = clinical_values
        
        return hypervector
    
    def _compress_full(self, hypervector: torch.Tensor, config: Dict) -> bytes:
        """
        Full tier compression - preserve maximum information
        """
        # Apply PCA for dimensionality reduction
        reduced = self._pca_compress(hypervector, config["compression_ratio"])
        
        # Use float16 for storage efficiency
        reduced_f16 = reduced.numpy().astype(np.float16)
        
        data = {
            "reduced_vector": reduced_f16,
            "compression_method": "pca",
            "compression_ratio": config["compression_ratio"]
        }
        
        return pickle.dumps(data)
    
    def _decompress_full(self, data: bytes, compressed: CompressedHypervector,
                        config: Dict) -> torch.Tensor:
        """Decompress full tier data"""
        unpacked = pickle.loads(data)
        
        reduced_vector = torch.from_numpy(unpacked["reduced_vector"].astype(np.float32))
        
        # For full decompression, we'd need the PCA components
        # For now, return padded version
        if len(reduced_vector) < compressed.original_dimension:
            hypervector = torch.zeros(compressed.original_dimension)
            hypervector[:len(reduced_vector)] = reduced_vector
        else:
            hypervector = reduced_vector
        
        return hypervector
    
    def _quantize(self, values: torch.Tensor, bits: int) -> torch.Tensor:
        """Quantize values to specified bit depth"""
        # Normalize to [0, 1]
        min_val = values.min()
        max_val = values.max()
        normalized = (values - min_val) / (max_val - min_val + 1e-8)
        
        # Quantize
        levels = 2 ** bits - 1
        quantized = torch.round(normalized * levels)
        
        # Store scale factors
        self._quant_params = {"min": min_val, "max": max_val, "bits": bits}
        
        return quantized
    
    def _dequantize(self, quantized: torch.Tensor, bits: int) -> torch.Tensor:
        """Dequantize values"""
        levels = 2 ** bits - 1
        normalized = quantized / levels
        
        # Denormalize using stored parameters
        if hasattr(self, '_quant_params'):
            min_val = self._quant_params["min"]
            max_val = self._quant_params["max"]
            values = normalized * (max_val - min_val) + min_val
        else:
            values = normalized
        
        return values
    
    def _dct_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Apply Discrete Cosine Transform"""
        # Simple DCT implementation
        n = len(values)
        dct_matrix = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                if i == 0:
                    dct_matrix[i, j] = 1.0 / np.sqrt(n)
                else:
                    dct_matrix[i, j] = np.sqrt(2.0/n) * np.cos(np.pi * i * (j + 0.5) / n)
        
        return torch.matmul(dct_matrix, values)
    
    def _idct_transform(self, transformed: torch.Tensor) -> torch.Tensor:
        """Apply Inverse Discrete Cosine Transform"""
        # Use transpose of DCT matrix for inverse
        n = len(transformed)
        dct_matrix = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                if i == 0:
                    dct_matrix[i, j] = 1.0 / np.sqrt(n)
                else:
                    dct_matrix[i, j] = np.sqrt(2.0/n) * np.cos(np.pi * i * (j + 0.5) / n)
        
        return torch.matmul(dct_matrix.T, transformed)
    
    def _pca_compress(self, hypervector: torch.Tensor, ratio: float) -> torch.Tensor:
        """Apply PCA compression"""
        # For single vector, just truncate
        # In practice, would use actual PCA across population
        target_dim = int(len(hypervector) * ratio)
        return hypervector[:target_dim]
    
    def _get_clinical_indices(self, omics_type: OmicsType) -> torch.Tensor:
        """Get indices of clinically relevant components"""
        # In practice, these would be loaded from clinical databases
        # For now, return deterministic indices based on omics type
        
        if omics_type == OmicsType.GENOMIC:
            # ACMG genes and pharmacogenomic variants
            num_clinical = 120000
        elif omics_type == OmicsType.TRANSCRIPTOMIC:
            # Disease-associated transcripts
            num_clinical = 50000
        else:
            num_clinical = 30000
        
        # Generate deterministic clinical indices
        torch.manual_seed(hash(omics_type.value))
        indices = torch.randperm(10000)[:num_clinical]
        
        return indices
