"""
Backend Adapter for HDC Encoding

Bridges the existing HypervectorEncoder with the new unified hardware backend system.
Provides backward compatibility while enabling access to CPU/Metal/CUDA acceleration.

Usage:
    # Option 1: Direct use of backend-optimized encoder
    from genomevault.hypervector_transform.backend_adapter import BackendOptimizedEncoder
    encoder = BackendOptimizedEncoder(dimension=8192)
    hypervector = encoder.encode_single(variants)

    # Option 2: Enable in existing HypervectorEncoder
    from genomevault.hypervector_transform import HypervectorEncoder, HypervectorConfig
    config = HypervectorConfig(use_unified_backend=True)
    encoder = HypervectorEncoder(config)
"""

from __future__ import annotations

import logging
from typing import Optional, Union
from dataclasses import dataclass

import numpy as np
import torch

from genomevault.core.constants import OmicsType, HYPERVECTOR_DIMENSIONS
from genomevault.compute import (
    get_accelerator,
    get_backend,
    initialize_backend,
    ComputeBackend,
)
from genomevault.config.loader import get_config

logger = logging.getLogger(__name__)


@dataclass
class BackendEncoderConfig:
    """Configuration for backend-optimized HDC encoder"""

    dimension: int = HYPERVECTOR_DIMENSIONS
    backend: Optional[ComputeBackend] = None  # None = auto-detect
    use_config_loader: bool = True  # Use compute.yaml if available
    normalize: bool = True
    seed: Optional[int] = 42


class BackendOptimizedEncoder:
    """
    Hardware-accelerated HDC encoder using unified backend system.

    This encoder leverages the CPU/Metal/CUDA backend abstraction for
    optimal performance across different hardware configurations.

    Performance Targets:
        - CPU: <10ms single encode, <1s for 100 samples
        - Metal: <1ms single, <100ms for 1K samples
        - CUDA: ~2ms single, <150ms for 1K samples
    """

    def __init__(self, config: Optional[BackendEncoderConfig] = None):
        """
        Initialize backend-optimized encoder

        Args:
            config: Configuration for encoder. If None, uses defaults.
        """
        self.config = config or BackendEncoderConfig()

        # Initialize backend
        if self.config.use_config_loader:
            try:
                # Load configuration from compute.yaml and environment
                compute_config = get_config()
                self.backend = compute_config.initialize_backend()
                logger.info(f"Initialized backend from config: {self.backend.value}")
            except Exception as e:
                logger.warning(f"Failed to load config, falling back to auto-detect: {e}")
                self.backend = initialize_backend(ComputeBackend.AUTO)
        elif self.config.backend:
            self.backend = initialize_backend(self.config.backend)
        else:
            self.backend = initialize_backend(ComputeBackend.AUTO)

        # Get accelerator instance
        self.accelerator = get_accelerator()
        logger.info(f"Using accelerator: {self.accelerator.name}")

        # Set random seed for reproducibility
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

    def encode_single(
        self,
        variants: Union[np.ndarray, torch.Tensor, dict],
        omics_type: Optional[OmicsType] = None
    ) -> np.ndarray:
        """
        Encode single genomic sample to hypervector

        Args:
            variants: Genomic variants as numpy array, torch tensor, or dict of features
            omics_type: Type of omics data (for compatibility)

        Returns:
            Binary hypervector of configured dimension
        """
        # Handle dict input (convert to numpy array)
        if isinstance(variants, dict):
            # Convert dict values to flat array
            values = []
            for key in sorted(variants.keys()):  # Sort for consistency
                val = variants[key]
                if isinstance(val, (int, float)):
                    values.append(float(val))
                elif isinstance(val, (list, tuple)):
                    values.extend([float(v) for v in val])
                elif isinstance(val, np.ndarray):
                    values.extend(val.flatten().tolist())
            variants = np.array(values, dtype=np.float32)

        # Convert to numpy if needed
        elif isinstance(variants, torch.Tensor):
            variants = variants.detach().cpu().numpy()

        # Ensure numpy array
        if not isinstance(variants, np.ndarray):
            variants = np.array(variants, dtype=np.float32)

        # Ensure float32
        if variants.dtype != np.float32:
            variants = variants.astype(np.float32, copy=False)

        # Ensure 2D shape for backend
        if variants.ndim == 1:
            variants = variants.reshape(-1, 1)

        # Encode using hardware accelerator
        hypervector = self.accelerator.encode_single(variants)

        # Normalize if requested
        if self.config.normalize and hypervector.dtype == np.float32:
            norm = np.linalg.norm(hypervector)
            if norm > 0:
                hypervector = hypervector / norm

        return hypervector

    def encode_batch(
        self,
        variants_batch: list[Union[np.ndarray, torch.Tensor]],
        omics_type: Optional[OmicsType] = None
    ) -> np.ndarray:
        """
        Encode batch of genomic samples to hypervectors

        Args:
            variants_batch: List of genomic variants
            omics_type: Type of omics data (for compatibility)

        Returns:
            Array of binary hypervectors, shape (batch_size, dimension)
        """
        # Convert to numpy
        variants_np = []
        for v in variants_batch:
            if isinstance(v, torch.Tensor):
                variants_np.append(v.detach().cpu().numpy().astype(np.float32))
            else:
                variants_np.append(np.asarray(v, dtype=np.float32))

        # Encode batch using hardware accelerator
        hypervectors = self.accelerator.encode_batch(variants_np)

        # Normalize if requested
        if self.config.normalize and hypervectors.dtype == np.float32:
            norms = np.linalg.norm(hypervectors, axis=1, keepdims=True)
            hypervectors = np.divide(
                hypervectors,
                norms,
                out=hypervectors,
                where=(norms > 0)
            )

        return hypervectors

    def similarity_search(
        self,
        query: Union[np.ndarray, torch.Tensor],
        database: Union[np.ndarray, torch.Tensor],
        top_k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for similar hypervectors in database

        Args:
            query: Query hypervector
            database: Database of hypervectors
            top_k: Number of results to return

        Returns:
            Tuple of (indices, similarities)
        """
        # Convert to numpy
        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy()
        if isinstance(database, torch.Tensor):
            database = database.detach().cpu().numpy()

        # Ensure float32
        query = query.astype(np.float32, copy=False)
        database = database.astype(np.float32, copy=False)

        # Use accelerator for search
        return self.accelerator.similarity_search(query, database, top_k)

    def bind_vectors(
        self,
        a: Union[np.ndarray, torch.Tensor],
        b: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        HDC binding operation (XOR for binary vectors)

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            Bound hypervector
        """
        # Convert to numpy
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()

        return self.accelerator.bind_vectors(a, b)

    def bundle_vectors(
        self,
        vectors: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        HDC bundling operation (majority vote for binary vectors)

        Args:
            vectors: Array of hypervectors to bundle

        Returns:
            Bundled hypervector
        """
        # Convert to numpy
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.detach().cpu().numpy()

        return self.accelerator.bundle_vectors(vectors)

    @property
    def backend_name(self) -> str:
        """Get name of current backend"""
        return self.accelerator.name

    @property
    def backend_type(self) -> ComputeBackend:
        """Get type of current backend"""
        return self.backend


def create_backend_encoder(
    dimension: int = HYPERVECTOR_DIMENSIONS,
    backend: Optional[str] = None,
    **kwargs
) -> BackendOptimizedEncoder:
    """
    Convenience function to create backend-optimized encoder

    Args:
        dimension: Hypervector dimension
        backend: Backend preference ('cpu', 'metal', 'cuda', 'auto')
        **kwargs: Additional config parameters

    Returns:
        Configured BackendOptimizedEncoder

    Example:
        # Auto-detect backend
        encoder = create_backend_encoder(dimension=8192)

        # Force CPU backend
        encoder = create_backend_encoder(dimension=8192, backend='cpu')

        # Use Metal with custom seed
        encoder = create_backend_encoder(dimension=8192, backend='metal', seed=42)
    """
    backend_enum = None
    if backend:
        backend_map = {
            'cpu': ComputeBackend.CPU,
            'metal': ComputeBackend.METAL,
            'cuda': ComputeBackend.CUDA,
            'auto': ComputeBackend.AUTO,
        }
        backend_enum = backend_map.get(backend.lower(), ComputeBackend.AUTO)

    config = BackendEncoderConfig(
        dimension=dimension,
        backend=backend_enum,
        **kwargs
    )

    return BackendOptimizedEncoder(config)


# Compatibility aliases
HDCBackendEncoder = BackendOptimizedEncoder
create_hdc_encoder = create_backend_encoder
