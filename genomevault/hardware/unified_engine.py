"""Unified hardware acceleration engine."""

import time
import numpy as np
from typing import Optional, Union, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

from genomevault.hardware.backend import (
    AcceleratorType,
    HardwareBackend,
    get_best_accelerator,
    list_available_accelerators
)
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AccelerationConfig:
    """Configuration for hardware acceleration."""
    
    dimension: int = 10000
    batch_size: int = 1024
    precision: str = "float32"  # float32, float16, bfloat16, int8
    device: Optional[AcceleratorType] = None  # Auto-detect if None
    device_id: int = 0
    memory_fraction: float = 0.8
    enable_mixed_precision: bool = True
    compile_kernels: bool = True
    
    def validate(self):
        """Validate configuration."""
        if self.dimension <= 0:
            raise ValueError(f"Dimension must be positive: {self.dimension}")
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive: {self.batch_size}")
        if not 0 < self.memory_fraction <= 1:
            raise ValueError(f"Memory fraction must be in (0, 1]: {self.memory_fraction}")


class UnifiedAccelerationEngine:
    """
    Unified hardware acceleration engine.
    
    Provides a single interface for all hardware acceleration backends,
    allowing multiple pipelines to share optimized implementations.
    """
    
    def __init__(self, config: Optional[AccelerationConfig] = None):
        """
        Initialize unified acceleration engine.
        
        Args:
            config: Acceleration configuration
        """
        self.config = config or AccelerationConfig()
        self.config.validate()
        
        # Detect and select backend
        self.backend = self._select_backend()
        self.accelerator = None
        
        # Initialize specific accelerator
        self._initialize_accelerator()
        
        # Cache for compiled kernels
        self._kernel_cache = {}
        
        logger.info(
            f"🚀 Unified Acceleration Engine Initialized\n"
            f"  Backend: {self.backend.type.value}\n"
            f"  Device: {self.backend.name}\n"
            f"  Precision: {self.config.precision}\n"
            f"  Dimension: {self.config.dimension}"
        )
    
    def _select_backend(self) -> HardwareBackend:
        """Select the best available backend."""
        if self.config.device:
            # User specified device
            backends = list_available_accelerators()
            for backend in backends:
                if backend.type == self.config.device and backend.available:
                    return backend
            logger.warning(f"Requested device {self.config.device} not available")
        
        # Auto-select best available
        return get_best_accelerator()
    
    def _initialize_accelerator(self):
        """Initialize the specific accelerator."""
        if self.backend.type == AcceleratorType.METAL:
            self._init_metal()
        elif self.backend.type == AcceleratorType.CUDA:
            self._init_cuda()
        elif self.backend.type == AcceleratorType.ROCM:
            self._init_rocm()
        elif self.backend.type == AcceleratorType.TPU:
            self._init_tpu()
        else:
            self._init_cpu()
    
    def _init_metal(self):
        """Initialize Apple Metal acceleration."""
        try:
            import mlx.core as mx
            self.mx = mx
            self.device = mx.default_device()
            
            # Set precision
            if self.config.precision == "float16":
                self.dtype = mx.float16
            elif self.config.precision == "bfloat16":
                self.dtype = mx.bfloat16
            else:
                self.dtype = mx.float32
            
            logger.debug("Metal acceleration initialized with MLX")
        except ImportError:
            logger.warning("MLX not available, falling back to CPU")
            self._init_cpu()
    
    def _init_cuda(self):
        """Initialize NVIDIA CUDA acceleration."""
        try:
            import cupy as cp
            self.cp = cp
            cp.cuda.Device(self.config.device_id).use()
            
            # Set memory pool
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(
                size=int(cp.cuda.Device().mem_info[1] * self.config.memory_fraction)
            )
            
            logger.debug("CUDA acceleration initialized with CuPy")
        except ImportError:
            try:
                import torch
                self.torch = torch
                self.device = torch.device(f"cuda:{self.config.device_id}")
                torch.cuda.set_per_process_memory_fraction(
                    self.config.memory_fraction, self.config.device_id
                )
                logger.debug("CUDA acceleration initialized with PyTorch")
            except ImportError:
                logger.warning("No CUDA library available, falling back to CPU")
                self._init_cpu()
    
    def _init_rocm(self):
        """Initialize AMD ROCm acceleration."""
        try:
            import torch
            self.torch = torch
            self.device = torch.device(f"cuda:{self.config.device_id}")  # ROCm uses CUDA API
            logger.debug("ROCm acceleration initialized")
        except ImportError:
            logger.warning("PyTorch with ROCm not available, falling back to CPU")
            self._init_cpu()
    
    def _init_tpu(self):
        """Initialize Google TPU acceleration."""
        try:
            import jax
            import jax.numpy as jnp
            self.jax = jax
            self.jnp = jnp
            logger.debug("TPU acceleration initialized with JAX")
        except ImportError:
            logger.warning("JAX not available, falling back to CPU")
            self._init_cpu()
    
    def _init_cpu(self):
        """Initialize CPU-only acceleration."""
        self.backend = HardwareBackend(
            type=AcceleratorType.CPU,
            name="CPU",
            available=True
        )
        logger.debug("Using CPU-only acceleration")
    
    def to_device(self, data: np.ndarray) -> Any:
        """
        Move data to accelerator device.
        
        Args:
            data: NumPy array
            
        Returns:
            Device-specific array
        """
        if self.backend.type == AcceleratorType.METAL and hasattr(self, 'mx'):
            return self.mx.array(data, dtype=self.dtype)
        elif self.backend.type == AcceleratorType.CUDA and hasattr(self, 'cp'):
            return self.cp.array(data, dtype=self.cp.float32)
        elif self.backend.type == AcceleratorType.CUDA and hasattr(self, 'torch'):
            return self.torch.tensor(data, device=self.device, dtype=self.torch.float32)
        elif self.backend.type == AcceleratorType.TPU and hasattr(self, 'jnp'):
            return self.jnp.array(data)
        else:
            return data
    
    def from_device(self, data: Any) -> np.ndarray:
        """
        Move data from accelerator to CPU.
        
        Args:
            data: Device-specific array
            
        Returns:
            NumPy array
        """
        if self.backend.type == AcceleratorType.METAL and hasattr(self, 'mx'):
            return np.array(data)
        elif self.backend.type == AcceleratorType.CUDA and hasattr(self, 'cp'):
            return data.get()
        elif self.backend.type == AcceleratorType.CUDA and hasattr(self, 'torch'):
            return data.cpu().numpy()
        elif self.backend.type == AcceleratorType.TPU and hasattr(self, 'jnp'):
            return np.array(data)
        else:
            return np.array(data)
    
    def matmul(self, a: Any, b: Any) -> Any:
        """
        Accelerated matrix multiplication.
        
        Args:
            a: First matrix (device array)
            b: Second matrix (device array)
            
        Returns:
            Result matrix (device array)
        """
        if self.backend.type == AcceleratorType.METAL and hasattr(self, 'mx'):
            result = self.mx.matmul(a, b)
            self.mx.eval(result)
            return result
        elif self.backend.type == AcceleratorType.CUDA and hasattr(self, 'cp'):
            return self.cp.matmul(a, b)
        elif self.backend.type == AcceleratorType.CUDA and hasattr(self, 'torch'):
            return self.torch.matmul(a, b)
        elif self.backend.type == AcceleratorType.TPU and hasattr(self, 'jnp'):
            return self.jnp.matmul(a, b)
        else:
            return np.matmul(a, b)
    
    def fft(self, data: Any, inverse: bool = False) -> Any:
        """
        Accelerated FFT.
        
        Args:
            data: Input data (device array)
            inverse: Whether to compute inverse FFT
            
        Returns:
            FFT result (device array)
        """
        if self.backend.type == AcceleratorType.METAL and hasattr(self, 'mx'):
            # MLX requires complex type for FFT
            # Check if data is real and convert to complex
            if str(data.dtype) not in ['complex64', 'complex128']:
                data = data.astype(self.mx.complex64)
            if inverse:
                result = self.mx.fft.ifft(data)
            else:
                result = self.mx.fft.fft(data)
            self.mx.eval(result)
            return result
        elif self.backend.type == AcceleratorType.CUDA and hasattr(self, 'cp'):
            if inverse:
                return self.cp.fft.ifft(data)
            else:
                return self.cp.fft.fft(data)
        elif self.backend.type == AcceleratorType.CUDA and hasattr(self, 'torch'):
            if inverse:
                return self.torch.fft.ifft(data)
            else:
                return self.torch.fft.fft(data)
        elif self.backend.type == AcceleratorType.TPU and hasattr(self, 'jnp'):
            if inverse:
                return self.jnp.fft.ifft(data)
            else:
                return self.jnp.fft.fft(data)
        else:
            if inverse:
                return np.fft.ifft(data)
            else:
                return np.fft.fft(data)
    
    def normalize(self, data: Any, axis: int = -1) -> Any:
        """
        L2 normalization.
        
        Args:
            data: Input data (device array)
            axis: Axis to normalize along
            
        Returns:
            Normalized data (device array)
        """
        if self.backend.type == AcceleratorType.METAL and hasattr(self, 'mx'):
            norm = self.mx.linalg.norm(data, axis=axis, keepdims=True)
            return data / (norm + 1e-8)
        elif self.backend.type == AcceleratorType.CUDA and hasattr(self, 'cp'):
            norm = self.cp.linalg.norm(data, axis=axis, keepdims=True)
            return data / (norm + 1e-8)
        elif self.backend.type == AcceleratorType.CUDA and hasattr(self, 'torch'):
            norm = self.torch.linalg.norm(data, dim=axis, keepdim=True)
            return data / (norm + 1e-8)
        elif self.backend.type == AcceleratorType.TPU and hasattr(self, 'jnp'):
            norm = self.jnp.linalg.norm(data, axis=axis, keepdims=True)
            return data / (norm + 1e-8)
        else:
            norm = np.linalg.norm(data, axis=axis, keepdims=True)
            return data / (norm + 1e-8)
    
    def random_projection_matrix(
        self,
        input_dim: int,
        output_dim: int,
        sparse: bool = False
    ) -> Any:
        """
        Generate random projection matrix.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            sparse: Whether to use sparse projection
            
        Returns:
            Projection matrix (device array)
        """
        if sparse:
            # Sparse random projection (±1, 0 with probabilities)
            if self.backend.type == AcceleratorType.METAL and hasattr(self, 'mx'):
                rand = self.mx.random.uniform(shape=[output_dim, input_dim], dtype=self.mx.float32)
                projection = self.mx.where(rand < 1/6, -1.0, 0.0)
                projection = self.mx.where(rand > 5/6, 1.0, projection)
                projection = projection * np.sqrt(3.0 / input_dim)
                return projection.astype(self.dtype)
            else:
                # CPU implementation
                rand = np.random.uniform(size=(output_dim, input_dim))
                projection = np.where(rand < 1/6, -1.0, 0.0)
                projection = np.where(rand > 5/6, 1.0, projection)
                projection = projection * np.sqrt(3.0 / input_dim)
                return self.to_device(projection.astype(np.float32))
        else:
            # Dense random projection
            if self.backend.type == AcceleratorType.METAL and hasattr(self, 'mx'):
                return self.mx.random.normal(
                    shape=[output_dim, input_dim],
                    dtype=self.dtype
                ) / np.sqrt(input_dim)
            elif self.backend.type == AcceleratorType.CUDA and hasattr(self, 'cp'):
                return self.cp.random.normal(
                    size=(output_dim, input_dim),
                    dtype=self.cp.float32
                ) / np.sqrt(input_dim)
            else:
                projection = np.random.randn(output_dim, input_dim) / np.sqrt(input_dim)
                return self.to_device(projection.astype(np.float32))
    
    def cosine_similarity(self, a: Any, b: Any) -> Any:
        """
        Compute cosine similarity.
        
        Args:
            a: First vector/matrix (device array)
            b: Second vector/matrix (device array)
            
        Returns:
            Similarity scores (device array)
        """
        a_norm = self.normalize(a)
        b_norm = self.normalize(b)
        
        if len(a_norm.shape) == 1 and len(b_norm.shape) == 2:
            # Query vs database
            return self.matmul(b_norm, a_norm[:, None])[:, 0]
        else:
            return self.matmul(a_norm, b_norm.T)
    
    def hamming_distance(self, a: Any, b: Any) -> Any:
        """
        Compute Hamming distance.
        
        Args:
            a: First binary vector/matrix (device array)
            b: Second binary vector/matrix (device array)
            
        Returns:
            Hamming distances (device array)
        """
        if self.backend.type == AcceleratorType.METAL and hasattr(self, 'mx'):
            # Binarize
            a_binary = (a > 0).astype(self.mx.float32)
            b_binary = (b > 0).astype(self.mx.float32)
            
            # XOR and count differences
            if len(a.shape) == 1 and len(b.shape) == 2:
                differences = self.mx.abs(b_binary - a_binary[None, :])
            else:
                differences = self.mx.abs(a_binary - b_binary)
            
            distances = self.mx.sum(differences, axis=-1)
            return distances / self.config.dimension
        
        elif self.backend.type == AcceleratorType.CUDA and hasattr(self, 'cp'):
            a_binary = (a > 0).astype(self.cp.float32)
            b_binary = (b > 0).astype(self.cp.float32)
            
            if len(a.shape) == 1 and len(b.shape) == 2:
                differences = self.cp.abs(b_binary - a_binary[None, :])
            else:
                differences = self.cp.abs(a_binary - b_binary)
            
            distances = self.cp.sum(differences, axis=-1)
            return distances / self.config.dimension
        
        else:
            # CPU fallback
            a_np = self.from_device(a)
            b_np = self.from_device(b)
            
            a_binary = (a_np > 0).astype(np.float32)
            b_binary = (b_np > 0).astype(np.float32)
            
            if len(a_np.shape) == 1 and len(b_np.shape) == 2:
                differences = np.abs(b_binary - a_binary[None, :])
            else:
                differences = np.abs(a_binary - b_binary)
            
            distances = np.sum(differences, axis=-1)
            return self.to_device(distances / self.config.dimension)
    
    def benchmark(self, operation: str = "matmul", size: int = 1000) -> Dict[str, Any]:
        """
        Benchmark specific operation.
        
        Args:
            operation: Operation to benchmark
            size: Problem size
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking {operation} on {self.backend.type.value}")
        
        results = {
            "backend": self.backend.type.value,
            "device": self.backend.name,
            "operation": operation,
            "size": size
        }
        
        if operation == "matmul":
            # Generate test matrices
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            
            a_dev = self.to_device(a)
            b_dev = self.to_device(b)
            
            # Warmup
            _ = self.matmul(a_dev[:10, :10], b_dev[:10, :10])
            
            # Benchmark
            start = time.perf_counter()
            result = self.matmul(a_dev, b_dev)
            if hasattr(self, 'mx'):
                self.mx.eval(result)
            elapsed = time.perf_counter() - start
            
            results["time_ms"] = elapsed * 1000
            results["gflops"] = (2 * size**3) / (elapsed * 1e9)
            
        elif operation == "fft":
            # Generate test data
            data = np.random.randn(size) + 1j * np.random.randn(size)
            data_dev = self.to_device(data)
            
            # Benchmark
            start = time.perf_counter()
            result = self.fft(data_dev)
            if hasattr(self, 'mx'):
                self.mx.eval(result)
            elapsed = time.perf_counter() - start
            
            results["time_ms"] = elapsed * 1000
            results["samples_per_sec"] = size / elapsed
        
        return results
    
    def get_info(self) -> Dict[str, Any]:
        """Get engine information."""
        return {
            "backend": self.backend.type.value,
            "device": self.backend.name,
            "available": self.backend.available,
            "memory_gb": self.backend.memory_gb,
            "precision": self.config.precision,
            "dimension": self.config.dimension,
            "batch_size": self.config.batch_size
        }