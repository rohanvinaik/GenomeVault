"""
Local GPU acceleration engine for hypervector operations.

Supports NVIDIA CUDA, AMD ROCm, and Intel oneAPI for local GPU acceleration.
Automatically detects available GPU hardware and optimizes accordingly.
"""

from __future__ import annotations

import time
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import warnings
import os

# Try importing GPU libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from genomevault.utils.logging import get_logger
from genomevault.core.constants import OmicsType

logger = get_logger(__name__)


class GPUBackend(Enum):
    """Available GPU backends."""
    
    CUDA = "cuda"          # NVIDIA GPUs
    ROCM = "rocm"          # AMD GPUs
    METAL = "metal"        # Apple Silicon (handled separately)
    ONEAPI = "oneapi"      # Intel GPUs
    VULKAN = "vulkan"      # Cross-platform
    CPU = "cpu"            # Fallback


@dataclass
class LocalGPUConfig:
    """Configuration for local GPU acceleration."""
    
    dimension: int = 10000
    batch_size: int = 1024
    precision: str = "float32"  # float32, float16, bfloat16, int8
    backend: Optional[GPUBackend] = None  # Auto-detect if None
    device_id: int = 0  # GPU device ID
    memory_fraction: float = 0.8  # Fraction of GPU memory to use
    enable_mixed_precision: bool = True
    enable_tensor_cores: bool = True  # For NVIDIA GPUs
    compile_kernels: bool = True  # JIT compilation
    
    def validate(self):
        """Validate configuration."""
        if self.dimension <= 0:
            raise ValueError(f"Dimension must be positive: {self.dimension}")
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive: {self.batch_size}")
        if not 0 < self.memory_fraction <= 1:
            raise ValueError(f"Memory fraction must be in (0, 1]: {self.memory_fraction}")


class LocalGPUEngine:
    """
    Local GPU acceleration engine for hypervector operations.
    
    Features:
    - Automatic GPU detection and backend selection
    - Support for NVIDIA CUDA, AMD ROCm, Intel oneAPI
    - Mixed precision training for faster computation
    - Memory-efficient batch processing
    - JIT compilation for optimized kernels
    """
    
    def __init__(self, config: Optional[LocalGPUConfig] = None):
        """
        Initialize local GPU engine.
        
        Args:
            config: GPU configuration settings
        """
        self.config = config or LocalGPUConfig()
        self.config.validate()
        
        # Detect and initialize backend
        self.backend = self._detect_backend()
        self.device = self._initialize_device()
        
        # Initialize projection matrices
        self._projection_matrices = {}
        
        # Setup precision and optimizations
        self._setup_precision()
        self._setup_optimizations()
        
        logger.info(
            f"🎮 Local GPU Acceleration Enabled\n"
            f"  Backend: {self.backend.value}\n"
            f"  Device: {self._get_device_name()}\n"
            f"  Memory: {self._get_device_memory():.1f} GB\n"
            f"  Dimension: {self.config.dimension}\n"
            f"  Precision: {self.config.precision}"
        )
    
    def _detect_backend(self) -> GPUBackend:
        """Detect available GPU backend."""
        if self.config.backend:
            return self.config.backend
        
        # Check NVIDIA CUDA
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return GPUBackend.CUDA
        
        # Check AMD ROCm
        if TORCH_AVAILABLE and hasattr(torch, 'hip') and torch.hip.is_available():
            return GPUBackend.ROCM
        
        # Check Intel oneAPI
        if TORCH_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
            return GPUBackend.ONEAPI
        
        # Check JAX with GPU support
        if JAX_AVAILABLE:
            try:
                devices = jax.devices()
                if any('gpu' in str(d).lower() for d in devices):
                    return GPUBackend.CUDA  # JAX typically uses CUDA
            except:
                pass
        
        # Fallback to CPU
        logger.warning("No GPU detected, falling back to CPU")
        return GPUBackend.CPU
    
    def _initialize_device(self):
        """Initialize compute device."""
        if self.backend == GPUBackend.CUDA:
            if TORCH_AVAILABLE:
                device = torch.device(f"cuda:{self.config.device_id}")
                # Set memory fraction
                torch.cuda.set_per_process_memory_fraction(
                    self.config.memory_fraction, 
                    self.config.device_id
                )
                return device
            elif CUPY_AVAILABLE:
                cp.cuda.Device(self.config.device_id).use()
                # Set memory pool
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(
                    size=int(cp.cuda.Device().mem_info[1] * self.config.memory_fraction)
                )
                return self.config.device_id
        
        elif self.backend == GPUBackend.ROCM:
            if TORCH_AVAILABLE:
                return torch.device(f"cuda:{self.config.device_id}")  # ROCm uses CUDA API
        
        elif self.backend == GPUBackend.ONEAPI:
            if TORCH_AVAILABLE:
                return torch.device(f"xpu:{self.config.device_id}")
        
        # CPU fallback
        if TORCH_AVAILABLE:
            return torch.device("cpu")
        return None
    
    def _get_device_name(self) -> str:
        """Get device name."""
        if self.backend == GPUBackend.CUDA and TORCH_AVAILABLE:
            return torch.cuda.get_device_name(self.config.device_id)
        elif self.backend == GPUBackend.CUDA and CUPY_AVAILABLE:
            return cp.cuda.Device(self.config.device_id).name.decode()
        elif self.backend == GPUBackend.CPU:
            return "CPU"
        return "Unknown"
    
    def _get_device_memory(self) -> float:
        """Get device memory in GB."""
        if self.backend == GPUBackend.CUDA:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return torch.cuda.get_device_properties(
                    self.config.device_id
                ).total_memory / 1e9
            elif CUPY_AVAILABLE:
                return cp.cuda.Device(self.config.device_id).mem_info[1] / 1e9
        return 0.0
    
    def _setup_precision(self):
        """Setup computation precision."""
        if TORCH_AVAILABLE:
            if self.config.precision == "float16":
                self.dtype = torch.float16
            elif self.config.precision == "bfloat16":
                self.dtype = torch.bfloat16
            elif self.config.precision == "int8":
                self.dtype = torch.int8
            else:
                self.dtype = torch.float32
        elif CUPY_AVAILABLE:
            if self.config.precision == "float16":
                self.dtype = cp.float16
            else:
                self.dtype = cp.float32
        else:
            self.dtype = np.float32
    
    def _setup_optimizations(self):
        """Setup GPU optimizations."""
        if self.backend == GPUBackend.CUDA and TORCH_AVAILABLE:
            # Enable TensorFloat-32 for A100/H100
            if self.config.enable_tensor_cores:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudnn autotuner
            torch.backends.cudnn.benchmark = True
            
            # Compile mode for PyTorch 2.0+
            if self.config.compile_kernels and hasattr(torch, 'compile'):
                self._compiled_encode = torch.compile(self._encode_torch)
            else:
                self._compiled_encode = self._encode_torch
    
    def encode_with_gpu(
        self, 
        data: Union[np.ndarray, List[float]], 
        omics_type: OmicsType = OmicsType.GENOMIC
    ) -> np.ndarray:
        """
        Encode data to hypervector using local GPU.
        
        Args:
            data: Input data (features or variants)
            omics_type: Type of omics data
            
        Returns:
            GPU-accelerated hypervector
        """
        # Convert to numpy if needed
        if isinstance(data, list):
            data = np.array(data)
        
        start = time.perf_counter()
        
        if self.backend == GPUBackend.CUDA and TORCH_AVAILABLE:
            result = self._encode_torch(data, omics_type)
        elif self.backend == GPUBackend.CUDA and CUPY_AVAILABLE:
            result = self._encode_cupy(data, omics_type)
        elif JAX_AVAILABLE:
            result = self._encode_jax(data, omics_type)
        else:
            # CPU fallback
            result = self._encode_cpu(data, omics_type)
        
        encoding_time = (time.perf_counter() - start) * 1000
        logger.debug(f"GPU encoding completed in {encoding_time:.2f}ms")
        
        return result
    
    def _encode_torch(
        self, 
        data: np.ndarray, 
        omics_type: OmicsType
    ) -> np.ndarray:
        """Encode using PyTorch."""
        # Move data to GPU
        data_gpu = torch.from_numpy(data).to(self.device, dtype=self.dtype)
        
        # Get or create projection matrix
        projection = self._get_projection_matrix_torch(omics_type, data.shape[-1])
        
        # Perform encoding
        if len(data_gpu.shape) == 1:
            # Single sample
            hypervector = torch.matmul(projection, data_gpu)
        else:
            # Batch processing
            hypervector = torch.matmul(data_gpu, projection.T)
        
        # Apply activation
        if self.config.enable_mixed_precision:
            with torch.cuda.amp.autocast():
                hypervector = torch.relu(hypervector)
        else:
            hypervector = torch.relu(hypervector)
        
        # Normalize
        hypervector = torch.nn.functional.normalize(hypervector, dim=-1)
        
        # Move back to CPU
        return hypervector.cpu().numpy()
    
    def _encode_cupy(
        self, 
        data: np.ndarray, 
        omics_type: OmicsType
    ) -> np.ndarray:
        """Encode using CuPy."""
        # Move data to GPU
        data_gpu = cp.asarray(data, dtype=self.dtype)
        
        # Get or create projection matrix
        projection = self._get_projection_matrix_cupy(omics_type, data.shape[-1])
        
        # Perform encoding
        if len(data_gpu.shape) == 1:
            hypervector = cp.dot(projection, data_gpu)
        else:
            hypervector = cp.dot(data_gpu, projection.T)
        
        # Apply activation (ReLU)
        hypervector = cp.maximum(hypervector, 0)
        
        # Normalize
        norm = cp.linalg.norm(hypervector, axis=-1, keepdims=True)
        hypervector = hypervector / (norm + 1e-8)
        
        # Move back to CPU
        return cp.asnumpy(hypervector)
    
    def _encode_jax(
        self, 
        data: np.ndarray, 
        omics_type: OmicsType
    ) -> np.ndarray:
        """Encode using JAX."""
        # Move data to GPU
        data_gpu = jnp.array(data, dtype=jnp.float32)
        
        # Get or create projection matrix
        projection = self._get_projection_matrix_jax(omics_type, data.shape[-1])
        
        # JIT compile the encoding function
        @jax.jit
        def encode_fn(proj, d):
            if len(d.shape) == 1:
                hv = jnp.dot(proj, d)
            else:
                hv = jnp.dot(d, proj.T)
            hv = jax.nn.relu(hv)
            return hv / (jnp.linalg.norm(hv, axis=-1, keepdims=True) + 1e-8)
        
        hypervector = encode_fn(projection, data_gpu)
        
        # Move back to CPU
        return np.array(hypervector)
    
    def _encode_cpu(
        self, 
        data: np.ndarray, 
        omics_type: OmicsType
    ) -> np.ndarray:
        """CPU fallback encoding."""
        # Get or create projection matrix
        projection = self._get_projection_matrix_cpu(omics_type, data.shape[-1])
        
        # Perform encoding
        if len(data.shape) == 1:
            hypervector = np.dot(projection, data)
        else:
            hypervector = np.dot(data, projection.T)
        
        # Apply activation (ReLU)
        hypervector = np.maximum(hypervector, 0)
        
        # Normalize
        norm = np.linalg.norm(hypervector, axis=-1, keepdims=True)
        hypervector = hypervector / (norm + 1e-8)
        
        return hypervector
    
    def _get_projection_matrix_torch(
        self, 
        omics_type: OmicsType, 
        input_dim: int
    ) -> torch.Tensor:
        """Get or create projection matrix for PyTorch."""
        key = f"{omics_type.value}_{input_dim}"
        
        if key not in self._projection_matrices:
            # Create new projection matrix
            if omics_type == OmicsType.GENOMIC:
                # Sparse initialization for genomic data
                projection = self._create_sparse_projection_torch(input_dim)
            else:
                # Dense initialization
                projection = torch.randn(
                    self.config.dimension, 
                    input_dim, 
                    device=self.device, 
                    dtype=self.dtype
                ) / np.sqrt(input_dim)
            
            self._projection_matrices[key] = projection
        
        return self._projection_matrices[key]
    
    def _create_sparse_projection_torch(self, input_dim: int) -> torch.Tensor:
        """Create sparse projection matrix for PyTorch."""
        # Random sparse projection (±1, 0 with probabilities)
        rand = torch.rand(
            self.config.dimension, 
            input_dim, 
            device=self.device
        )
        
        projection = torch.zeros_like(rand, dtype=self.dtype)
        projection[rand < 1/6] = -1.0
        projection[rand > 5/6] = 1.0
        
        # Scale for variance preservation
        projection *= np.sqrt(3.0 / input_dim)
        
        return projection
    
    def _get_projection_matrix_cupy(
        self, 
        omics_type: OmicsType, 
        input_dim: int
    ) -> cp.ndarray:
        """Get or create projection matrix for CuPy."""
        key = f"{omics_type.value}_{input_dim}"
        
        if key not in self._projection_matrices:
            if omics_type == OmicsType.GENOMIC:
                # Sparse initialization
                rand = cp.random.rand(self.config.dimension, input_dim)
                projection = cp.zeros_like(rand, dtype=self.dtype)
                projection[rand < 1/6] = -1.0
                projection[rand > 5/6] = 1.0
                projection *= np.sqrt(3.0 / input_dim)
            else:
                # Dense initialization
                projection = cp.random.randn(
                    self.config.dimension, 
                    input_dim, 
                    dtype=self.dtype
                ) / np.sqrt(input_dim)
            
            self._projection_matrices[key] = projection
        
        return self._projection_matrices[key]
    
    def _get_projection_matrix_jax(
        self, 
        omics_type: OmicsType, 
        input_dim: int
    ) -> jnp.ndarray:
        """Get or create projection matrix for JAX."""
        key = f"{omics_type.value}_{input_dim}"
        
        if key not in self._projection_matrices:
            # Use JAX random key
            rng_key = jax.random.PRNGKey(42)
            
            if omics_type == OmicsType.GENOMIC:
                # Sparse initialization
                rand = jax.random.uniform(
                    rng_key, 
                    shape=(self.config.dimension, input_dim)
                )
                projection = jnp.where(rand < 1/6, -1.0, 0.0)
                projection = jnp.where(rand > 5/6, 1.0, projection)
                projection *= np.sqrt(3.0 / input_dim)
            else:
                # Dense initialization
                projection = jax.random.normal(
                    rng_key,
                    shape=(self.config.dimension, input_dim)
                ) / np.sqrt(input_dim)
            
            self._projection_matrices[key] = projection
        
        return self._projection_matrices[key]
    
    def _get_projection_matrix_cpu(
        self, 
        omics_type: OmicsType, 
        input_dim: int
    ) -> np.ndarray:
        """Get or create projection matrix for CPU."""
        key = f"{omics_type.value}_{input_dim}"
        
        if key not in self._projection_matrices:
            if omics_type == OmicsType.GENOMIC:
                # Sparse initialization
                rand = np.random.rand(self.config.dimension, input_dim)
                projection = np.zeros_like(rand, dtype=np.float32)
                projection[rand < 1/6] = -1.0
                projection[rand > 5/6] = 1.0
                projection *= np.sqrt(3.0 / input_dim)
            else:
                # Dense initialization
                projection = np.random.randn(
                    self.config.dimension, 
                    input_dim
                ).astype(np.float32) / np.sqrt(input_dim)
            
            self._projection_matrices[key] = projection
        
        return self._projection_matrices[key]
    
    def compute_similarity_batch(
        self, 
        query: np.ndarray, 
        database: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute similarity between query and database vectors using GPU.
        
        Args:
            query: Query hypervector
            database: Database of hypervectors
            metric: Similarity metric (cosine, hamming, euclidean)
            
        Returns:
            Similarity scores
        """
        if self.backend == GPUBackend.CUDA and TORCH_AVAILABLE:
            return self._similarity_torch(query, database, metric)
        elif self.backend == GPUBackend.CUDA and CUPY_AVAILABLE:
            return self._similarity_cupy(query, database, metric)
        else:
            return self._similarity_cpu(query, database, metric)
    
    def _similarity_torch(
        self, 
        query: np.ndarray, 
        database: np.ndarray, 
        metric: str
    ) -> np.ndarray:
        """Compute similarity using PyTorch."""
        query_gpu = torch.from_numpy(query).to(self.device, dtype=self.dtype)
        db_gpu = torch.from_numpy(database).to(self.device, dtype=self.dtype)
        
        if metric == "cosine":
            query_norm = torch.nn.functional.normalize(query_gpu, dim=-1)
            db_norm = torch.nn.functional.normalize(db_gpu, dim=-1)
            similarities = torch.matmul(db_norm, query_norm.T)
        
        elif metric == "hamming":
            query_binary = (query_gpu > 0).float()
            db_binary = (db_gpu > 0).float()
            differences = torch.abs(db_binary - query_binary.unsqueeze(0))
            distances = torch.sum(differences, dim=1)
            similarities = 1.0 - (distances / self.config.dimension)
        
        elif metric == "euclidean":
            diff = db_gpu - query_gpu.unsqueeze(0)
            distances = torch.norm(diff, dim=1)
            similarities = 1.0 / (1.0 + distances)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarities.cpu().numpy()
    
    def _similarity_cupy(
        self, 
        query: np.ndarray, 
        database: np.ndarray, 
        metric: str
    ) -> np.ndarray:
        """Compute similarity using CuPy."""
        query_gpu = cp.asarray(query, dtype=self.dtype)
        db_gpu = cp.asarray(database, dtype=self.dtype)
        
        if metric == "cosine":
            query_norm = query_gpu / (cp.linalg.norm(query_gpu) + 1e-8)
            db_norm = db_gpu / (cp.linalg.norm(db_gpu, axis=1, keepdims=True) + 1e-8)
            similarities = cp.dot(db_norm, query_norm)
        
        elif metric == "hamming":
            query_binary = (query_gpu > 0).astype(cp.float32)
            db_binary = (db_gpu > 0).astype(cp.float32)
            differences = cp.abs(db_binary - query_binary[None, :])
            distances = cp.sum(differences, axis=1)
            similarities = 1.0 - (distances / self.config.dimension)
        
        elif metric == "euclidean":
            diff = db_gpu - query_gpu[None, :]
            distances = cp.linalg.norm(diff, axis=1)
            similarities = 1.0 / (1.0 + distances)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return cp.asnumpy(similarities)
    
    def _similarity_cpu(
        self, 
        query: np.ndarray, 
        database: np.ndarray, 
        metric: str
    ) -> np.ndarray:
        """Compute similarity using CPU."""
        if metric == "cosine":
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            db_norm = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(db_norm, query_norm)
        
        elif metric == "hamming":
            query_binary = (query > 0).astype(np.float32)
            db_binary = (database > 0).astype(np.float32)
            differences = np.abs(db_binary - query_binary[None, :])
            distances = np.sum(differences, axis=1)
            similarities = 1.0 - (distances / self.config.dimension)
        
        elif metric == "euclidean":
            diff = database - query[None, :]
            distances = np.linalg.norm(diff, axis=1)
            similarities = 1.0 / (1.0 + distances)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarities
    
    def optimize_for_inference(self, hypervector: np.ndarray) -> np.ndarray:
        """
        Optimize hypervector for inference.
        
        Args:
            hypervector: Input hypervector
            
        Returns:
            Optimized hypervector
        """
        if self.config.precision == "int8":
            # Quantize to int8
            min_val = hypervector.min()
            max_val = hypervector.max()
            scale = 127 / max(abs(min_val), abs(max_val))
            quantized = np.round(hypervector * scale).astype(np.int8)
            return quantized
        
        elif self.config.precision == "float16":
            # Convert to float16
            return hypervector.astype(np.float16)
        
        # Binary quantization for maximum compression
        return (hypervector > 0).astype(np.int8)
    
    def benchmark(self, input_dim: int = 1000, num_samples: int = 10000) -> dict:
        """
        Benchmark GPU acceleration performance.
        
        Args:
            input_dim: Input dimension
            num_samples: Number of samples to process
            
        Returns:
            Benchmark results
        """
        logger.info(f"Running GPU benchmark: {num_samples} samples, {input_dim} features")
        
        # Generate test data
        data = np.random.randn(num_samples, input_dim).astype(np.float32)
        
        # Warmup
        _ = self.encode_with_gpu(data[:10])
        
        # Benchmark encoding
        start = time.perf_counter()
        encoded = self.encode_with_gpu(data)
        encoding_time = time.perf_counter() - start
        
        # Benchmark similarity computation
        query = encoded[0]
        start = time.perf_counter()
        similarities = self.compute_similarity_batch(query, encoded)
        similarity_time = time.perf_counter() - start
        
        # Calculate metrics
        results = {
            "backend": self.backend.value,
            "device": self._get_device_name(),
            "samples": num_samples,
            "input_dim": input_dim,
            "output_dim": self.config.dimension,
            "encoding_time_ms": encoding_time * 1000,
            "similarity_time_ms": similarity_time * 1000,
            "samples_per_second": num_samples / encoding_time,
            "precision": self.config.precision,
            "memory_used_gb": self._get_memory_usage()
        }
        
        logger.info(
            f"Benchmark Results:\n"
            f"  Backend: {results['backend']}\n"
            f"  Device: {results['device']}\n"
            f"  Encoding: {results['encoding_time_ms']:.2f}ms "
            f"({results['samples_per_second']:.0f} samples/sec)\n"
            f"  Similarity: {results['similarity_time_ms']:.2f}ms\n"
            f"  Memory: {results['memory_used_gb']:.2f} GB"
        )
        
        return results
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if self.backend == GPUBackend.CUDA:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return torch.cuda.memory_allocated(self.device) / 1e9
            elif CUPY_AVAILABLE:
                mempool = cp.get_default_memory_pool()
                return mempool.used_bytes() / 1e9
        return 0.0
    
    def cleanup(self):
        """Clean up GPU resources."""
        if self.backend == GPUBackend.CUDA:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif CUPY_AVAILABLE:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
        
        self._projection_matrices.clear()
        logger.info("GPU resources cleaned up")


class MultiGPUEngine:
    """
    Multi-GPU engine for distributed processing.
    
    Supports data parallelism across multiple GPUs for
    large-scale genomic processing.
    """
    
    def __init__(self, config: Optional[LocalGPUConfig] = None):
        """Initialize multi-GPU engine."""
        self.config = config or LocalGPUConfig()
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for multi-GPU support")
        
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            logger.warning(f"Only {self.num_gpus} GPU(s) detected, using single GPU")
            self.single_gpu = LocalGPUEngine(config)
        else:
            logger.info(f"Initialized multi-GPU engine with {self.num_gpus} GPUs")
            self.single_gpu = None
    
    def encode_parallel(
        self,
        data: np.ndarray,
        omics_type: OmicsType = OmicsType.GENOMIC
    ) -> np.ndarray:
        """
        Encode data using multiple GPUs in parallel.
        
        Args:
            data: Input data (batch_size, features)
            omics_type: Type of omics data
            
        Returns:
            Encoded hypervectors
        """
        if self.single_gpu:
            return self.single_gpu.encode_with_gpu(data, omics_type)
        
        # Split data across GPUs
        batch_size = len(data)
        chunk_size = (batch_size + self.num_gpus - 1) // self.num_gpus
        
        results = []
        for gpu_id in range(self.num_gpus):
            start_idx = gpu_id * chunk_size
            end_idx = min((gpu_id + 1) * chunk_size, batch_size)
            
            if start_idx >= batch_size:
                break
            
            # Process chunk on specific GPU
            chunk = data[start_idx:end_idx]
            with torch.cuda.device(gpu_id):
                gpu_config = LocalGPUConfig(
                    dimension=self.config.dimension,
                    precision=self.config.precision,
                    device_id=gpu_id
                )
                engine = LocalGPUEngine(gpu_config)
                result = engine.encode_with_gpu(chunk, omics_type)
                results.append(result)
        
        # Concatenate results
        return np.concatenate(results, axis=0)


def demonstrate_local_gpu():
    """Demonstrate local GPU acceleration capabilities."""
    print("\n" + "="*70)
    print("  LOCAL GPU ACCELERATION DEMONSTRATION")
    print("="*70)
    
    # Initialize engine
    config = LocalGPUConfig(
        dimension=10000,
        precision="float32",
        enable_mixed_precision=True
    )
    
    try:
        engine = LocalGPUEngine(config)
    except Exception as e:
        print(f"\n⚠️  GPU initialization failed: {e}")
        print("Falling back to CPU mode...")
        config.backend = GPUBackend.CPU
        engine = LocalGPUEngine(config)
    
    # Run benchmark
    print("\nRunning performance benchmark...")
    results = engine.benchmark(input_dim=1000, num_samples=10000)
    
    print(f"\nGPU Acceleration Results:")
    print(f"  Backend: {results['backend']}")
    print(f"  Device: {results['device']}")
    print(f"  Throughput: {results['samples_per_second']:.0f} samples/sec")
    print(f"  Encoding latency: {results['encoding_time_ms']/results['samples']:.3f}ms per sample")
    
    # Compare with CPU baseline
    if results['backend'] != 'cpu':
        print("\nTesting CPU baseline for comparison...")
        cpu_config = LocalGPUConfig(
            dimension=10000,
            backend=GPUBackend.CPU
        )
        cpu_engine = LocalGPUEngine(cpu_config)
        cpu_results = cpu_engine.benchmark(input_dim=1000, num_samples=1000)
        
        speedup = cpu_results['encoding_time_ms'] / results['encoding_time_ms'] * 10
        print(f"\nSpeedup over CPU: {speedup:.1f}x")
        print(f"  CPU time: {cpu_results['encoding_time_ms']:.1f}ms (1000 samples)")
        print(f"  GPU time: {results['encoding_time_ms']/10:.1f}ms (1000 samples normalized)")
    
    # Cleanup
    engine.cleanup()
    
    print("\n✅ Local GPU acceleration successfully demonstrated!")
    print("="*70)


if __name__ == "__main__":
    demonstrate_local_gpu()