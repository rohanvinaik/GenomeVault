"""
Hardware Abstraction Layer for GenomeVault

Core Principle: Default to CPU with GPU as opt-in acceleration for batch workloads.
Design for graceful degradation when GPU unavailable.

Architecture:
- Tier 1: CPU-Only (Production Default) - ZK, PIR, API, Single-Sample HDC
- Tier 2: CPU-First with GPU Fallback - Batch HDC, Similarity Search, Binding/Bundling
- Tier 3: GPU-Preferred - Bulk imports, Validation pipelines, Research analytics

Backend Selection Modes:
- Config Mode (default): Static rules from compute.yaml
- Intelligent Mode (opt-in): Dynamic data-driven selection via IntelligentBackendSelector
"""

from enum import Enum
from typing import Protocol, Optional, Union, Dict, Any
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)


class ComputeBackend(Enum):
    """Available compute backends"""
    CPU = "cpu"
    METAL = "metal"  # Apple Silicon
    CUDA = "cuda"    # NVIDIA
    AUTO = "auto"    # Detect best available


class HardwareAccelerator(Protocol):
    """
    Interface all accelerators must implement

    All methods must produce identical results across backends
    (within floating-point precision tolerances)
    """

    name: str

    def encode_single(self, variants: np.ndarray) -> np.ndarray:
        """
        Encode a single genomic sample to hypervector

        Optimized for latency, not throughput
        Use for API endpoints and real-time clinical queries

        Args:
            variants: (n_variants, n_features) array

        Returns:
            (dimension,) hypervector
        """
        ...

    def encode_batch(self, variants_batch: list[np.ndarray]) -> np.ndarray:
        """
        Encode multiple genomic samples in batch

        Optimized for throughput
        Use for bulk imports, validation pipelines, research

        Args:
            variants_batch: List of (n_variants, n_features) arrays

        Returns:
            (batch_size, dimension) array of hypervectors
        """
        ...

    def similarity_search(
        self,
        query: np.ndarray,
        database: np.ndarray,
        top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find k most similar vectors in database

        Args:
            query: (dimension,) query vector
            database: (n_samples, dimension) database vectors
            top_k: Number of results to return

        Returns:
            (indices, similarities) - both (top_k,) arrays
        """
        ...

    def bind_vectors(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        HDC binding operation (XOR for binary vectors)

        Args:
            a: (dimension,) or (batch, dimension)
            b: (dimension,) or (batch, dimension)

        Returns:
            Bound vector(s) same shape as inputs
        """
        ...

    def bundle_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        HDC bundling operation (majority vote for binary vectors)

        Args:
            vectors: (n_vectors, dimension) array

        Returns:
            (dimension,) bundled vector
        """
        ...


class ComputeBackendManager:
    """
    Thread-safe singleton that manages hardware backend selection

    Usage:
        manager = ComputeBackendManager()
        backend_type = manager.initialize(ComputeBackend.AUTO)
        accelerator = manager.get_accelerator()
        result = accelerator.encode_single(data)
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, intelligent_mode: bool = False):
        """Ensure singleton pattern"""
        # Note: For testing, allow different instances with different modes
        # In production, use the global singleton instead
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, intelligent_mode: bool = False):
        """
        Initialize manager (only once due to singleton)

        Args:
            intelligent_mode: Enable intelligent data-driven backend selection
                            False (default): Use config-based selection (backward compatible)
                            True: Analyze data characteristics to select backend
        """
        # Allow re-initialization if intelligent_mode changed (for testing)
        if hasattr(self, '_initialized') and hasattr(self, '_intelligent_mode'):
            if self._intelligent_mode == intelligent_mode:
                return  # Already initialized with same mode

        self._backend: Optional[ComputeBackend] = None
        self._accelerator: Optional[HardwareAccelerator] = None
        self._intelligent_mode = intelligent_mode
        self._intelligent_selector = None
        self._initialized = True
        self.logger = logging.getLogger(__name__)

        # Initialize intelligent selector if enabled
        if intelligent_mode:
            try:
                from genomevault.compute.intelligent_selector import IntelligentBackendSelector
                self._intelligent_selector = IntelligentBackendSelector()
                self.logger.info("Intelligent backend selection enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize intelligent selector: {e}")
                self.logger.info("Falling back to config-based selection")
                self._intelligent_mode = False

    def initialize(
        self,
        preferred: ComputeBackend = ComputeBackend.AUTO,
        force: bool = False
    ) -> ComputeBackend:
        """
        Initialize best available backend

        Args:
            preferred: Desired backend (AUTO for automatic detection)
            force: Force re-initialization even if already initialized

        Returns:
            Actual backend that was initialized

        Raises:
            RuntimeError: If preferred backend unavailable and not AUTO
        """
        # Skip if already initialized (unless force)
        if self._backend is not None and not force:
            self.logger.debug(f"Backend already initialized: {self._backend.value}")
            return self._backend

        if preferred == ComputeBackend.AUTO:
            # Detection priority: Metal > CUDA > CPU
            if self._is_metal_available():
                return self._init_metal()
            elif self._is_cuda_available():
                return self._init_cuda()
            else:
                return self._init_cpu()
        else:
            return self._init_specific(preferred)

    def get_accelerator(self) -> HardwareAccelerator:
        """
        Get current accelerator instance

        Automatically initializes with AUTO if not yet initialized

        Returns:
            Active HardwareAccelerator implementation
        """
        if self._accelerator is None:
            self.initialize(ComputeBackend.AUTO)
        return self._accelerator

    def get_backend(self) -> ComputeBackend:
        """Get current backend type"""
        if self._backend is None:
            self.initialize(ComputeBackend.AUTO)
        return self._backend

    # Backend detection

    def _is_metal_available(self) -> bool:
        """Check if Apple Metal (MLX) is available"""
        try:
            import mlx.core as mx
            # Test basic operation
            _ = mx.array([1.0, 2.0, 3.0])
            self.logger.debug("Metal backend available")
            return True
        except (ImportError, RuntimeError) as e:
            self.logger.debug(f"Metal not available: {e}")
            return False

    def _is_cuda_available(self) -> bool:
        """Check if NVIDIA CUDA (PyTorch) is available"""
        try:
            import torch
            if torch.cuda.is_available():
                self.logger.debug(f"CUDA backend available: {torch.cuda.device_count()} devices")
                return True
            else:
                self.logger.debug("PyTorch installed but CUDA unavailable")
                return False
        except ImportError as e:
            self.logger.debug(f"CUDA not available: {e}")
            return False

    # Backend initialization

    def _init_metal(self) -> ComputeBackend:
        """Initialize Apple Metal backend"""
        try:
            from genomevault.compute.metal_backend import MetalBackend
            self._accelerator = MetalBackend()
            self._backend = ComputeBackend.METAL
            self.logger.info(f"✓ Initialized {self._accelerator.name}")
            return ComputeBackend.METAL
        except Exception as e:
            self.logger.warning(f"Metal initialization failed: {e}, falling back to CPU")
            return self._init_cpu()

    def _init_cuda(self) -> ComputeBackend:
        """Initialize NVIDIA CUDA backend"""
        try:
            from genomevault.compute.cuda_backend import CUDABackend
            self._accelerator = CUDABackend()
            self._backend = ComputeBackend.CUDA
            self.logger.info(f"✓ Initialized {self._accelerator.name}")
            return ComputeBackend.CUDA
        except Exception as e:
            self.logger.warning(f"CUDA initialization failed: {e}, falling back to CPU")
            return self._init_cpu()

    def _init_cpu(self) -> ComputeBackend:
        """Initialize CPU backend (always available)"""
        from genomevault.compute.cpu_backend import CPUBackend
        self._accelerator = CPUBackend()
        self._backend = ComputeBackend.CPU
        self.logger.info(f"✓ Initialized {self._accelerator.name}")
        return ComputeBackend.CPU

    def _init_specific(self, backend: ComputeBackend) -> ComputeBackend:
        """
        Initialize specific backend, raise if unavailable

        Args:
            backend: Specific backend to initialize

        Returns:
            Initialized backend

        Raises:
            RuntimeError: If requested backend unavailable
        """
        if backend == ComputeBackend.METAL:
            if not self._is_metal_available():
                raise RuntimeError(
                    "Metal backend requested but unavailable. "
                    "Ensure running on Apple Silicon with MLX installed."
                )
            return self._init_metal()

        elif backend == ComputeBackend.CUDA:
            if not self._is_cuda_available():
                raise RuntimeError(
                    "CUDA backend requested but unavailable. "
                    "Ensure NVIDIA GPU present and PyTorch with CUDA support installed."
                )
            return self._init_cuda()

        elif backend == ComputeBackend.CPU:
            return self._init_cpu()

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def get_backend_for_operation(
        self,
        operation: str,
        data: Union[np.ndarray, list, int],
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[ComputeBackend, str]:
        """
        Select optimal backend for an operation (intelligent mode only)

        This method is only active when intelligent_mode=True.
        Otherwise, returns the configured default backend.

        Args:
            operation: Operation name ('encode', 'search', 'prove', 'retrieve')
            data: Input data or size hint
            context: Operation context (interactive, batch, latency_sensitive)

        Returns:
            Tuple of (selected_backend, reasoning)

        Example:
            >>> manager = ComputeBackendManager(intelligent_mode=True)
            >>> backend, reason = manager.get_backend_for_operation(
            ...     operation='encode',
            ...     data=my_variants,
            ...     context={'interactive': True}
            ... )
            >>> print(f"Selected {backend.value}: {reason}")
        """
        # If intelligent mode disabled, return current backend
        if not self._intelligent_mode or self._intelligent_selector is None:
            current_backend = self.get_backend()
            return current_backend, "Config-based selection (intelligent mode disabled)"

        # Use intelligent selector
        try:
            return self._intelligent_selector.select_backend_for_operation(
                operation=operation,
                data=data,
                context=context
            )
        except Exception as e:
            self.logger.error(f"Intelligent selection failed: {e}")
            # Fallback to current backend
            current_backend = self.get_backend()
            return current_backend, f"Fallback due to error: {e}"

    def reset(self):
        """Reset manager state (primarily for testing)"""
        self._backend = None
        self._accelerator = None


# Global singleton instance
_global_manager = ComputeBackendManager()


def get_accelerator() -> HardwareAccelerator:
    """
    Convenience function to get global accelerator instance

    Returns:
        Active HardwareAccelerator implementation

    Example:
        >>> from genomevault.compute.backend import get_accelerator
        >>> accelerator = get_accelerator()
        >>> result = accelerator.encode_single(variants)
    """
    return _global_manager.get_accelerator()


def get_backend() -> ComputeBackend:
    """
    Get current backend type

    Returns:
        Active ComputeBackend enum value
    """
    return _global_manager.get_backend()


def initialize_backend(preferred: ComputeBackend = ComputeBackend.AUTO) -> ComputeBackend:
    """
    Initialize compute backend explicitly

    Args:
        preferred: Desired backend (AUTO for automatic detection)

    Returns:
        Actual backend that was initialized

    Example:
        >>> from genomevault.compute.backend import initialize_backend, ComputeBackend
        >>> backend = initialize_backend(ComputeBackend.METAL)
        >>> print(f"Using {backend.value} backend")
    """
    return _global_manager.initialize(preferred)
