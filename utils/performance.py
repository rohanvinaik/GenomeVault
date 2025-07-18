"""
Performance optimization utilities for GenomeVault.

This module provides:
- SIMD optimizations for hypervector operations
- GPU acceleration support
- Memory-efficient data structures
- Parallel processing utilities
"""

import numpy as np
import numba
from numba import cuda, jit, prange
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Tuple, Optional, Union, Callable
import torch
import cupy as cp  # GPU arrays
from functools import lru_cache
import psutil

from ..utils.logging import get_logger, performance_logger

logger = get_logger(__name__)

# Check available acceleration
CUDA_AVAILABLE = cuda.is_available()
CPU_COUNT = mp.cpu_count()
TORCH_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    logger.info("CUDA acceleration available")
if TORCH_AVAILABLE:
    logger.info(f"PyTorch GPU available: {torch.cuda.get_device_name(0)}")

class HypervectorAccelerator:
    """Hardware-accelerated hypervector operations"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and (CUDA_AVAILABLE or TORCH_AVAILABLE)
        
        if self.use_gpu:
            if TORCH_AVAILABLE:
                self.device = torch.device('cuda')
                self.backend = 'torch'
            elif CUDA_AVAILABLE:
                self.backend = 'cupy'
            logger.info(f"Using GPU acceleration with {self.backend}")
        else:
            self.backend = 'numpy'
            logger.info("Using CPU with SIMD optimizations")
    
    @staticmethod
    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def _hamming_distance_cpu(v1: np.ndarray, v2: np.ndarray) -> int:
        """CPU-optimized Hamming distance using Numba"""
        distance = 0
        for i in prange(len(v1)):
            if v1[i] != v2[i]:
                distance += 1
        return distance
    
    def hamming_distance(self, v1: Union[np.ndarray, torch.Tensor], 
                        v2: Union[np.ndarray, torch.Tensor]) -> int:
        """Compute Hamming distance with hardware acceleration"""
        
        if self.backend == 'torch':
            if not isinstance(v1, torch.Tensor):
                v1 = torch.from_numpy(v1).to(self.device)
            if not isinstance(v2, torch.Tensor):
                v2 = torch.from_numpy(v2).to(self.device)
            
            return int((v1 != v2).sum().cpu())
            
        elif self.backend == 'cupy':
            if not isinstance(v1, cp.ndarray):
                v1 = cp.asarray(v1)
            if not isinstance(v2, cp.ndarray):
                v2 = cp.asarray(v2)
            
            return int(cp.sum(v1 != v2))
            
        else:
            return self._hamming_distance_cpu(v1, v2)
    
    @staticmethod
    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def _cosine_similarity_cpu(v1: np.ndarray, v2: np.ndarray) -> float:
        """CPU-optimized cosine similarity"""
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        
        for i in prange(len(v1)):
            dot_product += v1[i] * v2[i]
            norm1 += v1[i] * v1[i]
            norm2 += v2[i] * v2[i]
        
        return dot_product / (np.sqrt(norm1) * np.sqrt(norm2))
    
    def cosine_similarity(self, v1: Union[np.ndarray, torch.Tensor],
                         v2: Union[np.ndarray, torch.Tensor]) -> float:
        """Compute cosine similarity with hardware acceleration"""
        
        if self.backend == 'torch':
            if not isinstance(v1, torch.Tensor):
                v1 = torch.from_numpy(v1).float().to(self.device)
            if not isinstance(v2, torch.Tensor):
                v2 = torch.from_numpy(v2).float().to(self.device)
            
            cos_sim = torch.nn.functional.cosine_similarity(
                v1.unsqueeze(0), v2.unsqueeze(0)
            )
            return float(cos_sim.cpu())
            
        elif self.backend == 'cupy':
            if not isinstance(v1, cp.ndarray):
                v1 = cp.asarray(v1, dtype=cp.float32)
            if not isinstance(v2, cp.ndarray):
                v2 = cp.asarray(v2, dtype=cp.float32)
            
            dot = cp.dot(v1, v2)
            norm1 = cp.linalg.norm(v1)
            norm2 = cp.linalg.norm(v2)
            return float(dot / (norm1 * norm2))
            
        else:
            return self._cosine_similarity_cpu(v1.astype(np.float32), 
                                             v2.astype(np.float32))
    
    def batch_hamming_distance(self, vectors: List[np.ndarray], 
                              query: np.ndarray) -> np.ndarray:
        """Compute Hamming distances for batch of vectors"""
        
        if self.backend == 'torch':
            # Convert to torch tensors
            batch = torch.stack([torch.from_numpy(v) for v in vectors]).to(self.device)
            query_t = torch.from_numpy(query).to(self.device)
            
            # Broadcast and compute
            distances = (batch != query_t).sum(dim=1)
            return distances.cpu().numpy()
            
        elif self.backend == 'cupy':
            # Stack vectors
            batch = cp.stack([cp.asarray(v) for v in vectors])
            query_cp = cp.asarray(query)
            
            # Compute distances
            distances = cp.sum(batch != query_cp, axis=1)
            return cp.asnumpy(distances)
            
        else:
            # Parallel CPU computation
            distances = np.zeros(len(vectors), dtype=np.int32)
            for i, v in enumerate(vectors):
                distances[i] = self._hamming_distance_cpu(v, query)
            return distances
    
    @staticmethod
    @cuda.jit
    def _circular_convolution_kernel(a, b, result, n):
        """CUDA kernel for circular convolution"""
        idx = cuda.grid(1)
        if idx < n:
            temp = 0.0
            for j in range(n):
                temp += a[j] * b[(idx - j) % n]
            result[idx] = temp
    
    def circular_convolution(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Perform circular convolution with acceleration"""
        n = len(v1)
        
        if self.backend == 'torch':
            # Use FFT for circular convolution
            v1_t = torch.from_numpy(v1).float().to(self.device)
            v2_t = torch.from_numpy(v2).float().to(self.device)
            
            # FFT method
            fft1 = torch.fft.fft(v1_t)
            fft2 = torch.fft.fft(v2_t)
            result = torch.fft.ifft(fft1 * fft2).real
            
            return result.cpu().numpy()
            
        elif self.backend == 'cupy' and CUDA_AVAILABLE:
            # CUDA kernel implementation
            v1_gpu = cuda.to_device(v1.astype(np.float32))
            v2_gpu = cuda.to_device(v2.astype(np.float32))
            result_gpu = cuda.device_array(n, dtype=np.float32)
            
            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            
            self._circular_convolution_kernel[blocks_per_grid, threads_per_block](
                v1_gpu, v2_gpu, result_gpu, n
            )
            
            return result_gpu.copy_to_host()
            
        else:
            # NumPy FFT fallback
            return np.fft.ifft(np.fft.fft(v1) * np.fft.fft(v2)).real.astype(v1.dtype)


class MemoryEfficientStorage:
    """Memory-efficient storage for large hypervector collections"""
    
    def __init__(self, dimension: int, dtype=np.uint8):
        self.dimension = dimension
        self.dtype = dtype
        self.chunks = []
        self.chunk_size = 10000  # Vectors per chunk
        self.current_chunk = []
        
    def add_vector(self, vector: np.ndarray):
        """Add vector to storage"""
        if len(self.current_chunk) >= self.chunk_size:
            # Compress and store chunk
            compressed = self._compress_chunk(self.current_chunk)
            self.chunks.append(compressed)
            self.current_chunk = []
        
        self.current_chunk.append(vector)
    
    def get_vector(self, index: int) -> np.ndarray:
        """Retrieve vector by index"""
        chunk_idx = index // self.chunk_size
        vector_idx = index % self.chunk_size
        
        if chunk_idx < len(self.chunks):
            chunk = self._decompress_chunk(self.chunks[chunk_idx])
            return chunk[vector_idx]
        elif index - len(self.chunks) * self.chunk_size < len(self.current_chunk):
            return self.current_chunk[vector_idx]
        else:
            raise IndexError("Vector index out of range")
    
    def _compress_chunk(self, vectors: List[np.ndarray]) -> bytes:
        """Compress a chunk of vectors"""
        # Stack vectors and compress
        chunk_array = np.stack(vectors)
        return chunk_array.tobytes()  # Can use zlib for further compression
    
    def _decompress_chunk(self, compressed: bytes) -> np.ndarray:
        """Decompress a chunk of vectors"""
        # Reconstruct array
        chunk_array = np.frombuffer(compressed, dtype=self.dtype)
        return chunk_array.reshape(-1, self.dimension)


class ParallelProcessor:
    """Utilities for parallel processing of genomic data"""
    
    def __init__(self, n_workers: Optional[int] = None):
        self.n_workers = n_workers or CPU_COUNT
        self.thread_pool = ThreadPoolExecutor(max_workers=self.n_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.n_workers)
        
    def parallel_map(self, func: Callable, items: List, 
                    use_processes: bool = False) -> List:
        """Apply function to items in parallel"""
        
        pool = self.process_pool if use_processes else self.thread_pool
        
        with performance_logger.track_operation(
            f"parallel_map_{func.__name__}",
            {"n_items": len(items), "n_workers": self.n_workers}
        ):
            results = list(pool.map(func, items))
        
        return results
    
    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def _parallel_variant_processing(variants: np.ndarray, 
                                   reference: np.ndarray) -> np.ndarray:
        """Process variants in parallel using Numba"""
        n_variants = len(variants)
        results = np.zeros(n_variants, dtype=np.int32)
        
        for i in prange(n_variants):
            # Example: count differences from reference
            diff_count = 0
            for j in range(len(reference)):
                if variants[i, j] != reference[j]:
                    diff_count += 1
            results[i] = diff_count
        
        return results
    
    def process_variants_batch(self, variants: np.ndarray, 
                              reference: np.ndarray) -> np.ndarray:
        """Process batch of variants with parallel acceleration"""
        return self._parallel_variant_processing(variants, reference)
    
    def __del__(self):
        """Clean up thread pools"""
        self.thread_pool.shutdown(wait=False)
        self.process_pool.shutdown(wait=False)


class CacheOptimizer:
    """Cache optimization for frequently accessed data"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        
    @lru_cache(maxsize=1000)
    def cached_hypervector_operation(self, v1_hash: int, v2_hash: int, 
                                   operation: str) -> float:
        """Cache results of expensive hypervector operations"""
        # This would be called with actual computation
        pass
    
    def optimize_memory_layout(self, vectors: np.ndarray) -> np.ndarray:
        """Optimize memory layout for cache efficiency"""
        # Ensure C-contiguous layout for better cache performance
        if not vectors.flags['C_CONTIGUOUS']:
            vectors = np.ascontiguousarray(vectors)
        
        # Align to cache line boundaries (typically 64 bytes)
        if vectors.itemsize * vectors.shape[1] % 64 != 0:
            # Pad to align
            pad_size = (64 - (vectors.itemsize * vectors.shape[1] % 64)) // vectors.itemsize
            vectors = np.pad(vectors, ((0, 0), (0, pad_size)), mode='constant')
        
        return vectors


class ResourceMonitor:
    """Monitor and optimize resource usage"""
    
    def __init__(self):
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = self.process.memory_info()
        return {
            'rss': memory_info.rss / (1024 * 1024 * 1024),  # GB
            'vms': memory_info.vms / (1024 * 1024 * 1024),  # GB
            'percent': self.process.memory_percent()
        }
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        return self.process.cpu_percent(interval=0.1)
    
    def optimize_for_available_memory(self, required_memory_gb: float) -> bool:
        """Check if operation can proceed with available memory"""
        available_memory = psutil.virtual_memory().available / (1024**3)
        
        if available_memory < required_memory_gb * 1.2:  # 20% buffer
            logger.warning(
                "insufficient_memory",
                required_gb=required_memory_gb,
                available_gb=available_memory
            )
            # Trigger garbage collection
            import gc
            gc.collect()
            
            # Re-check
            available_memory = psutil.virtual_memory().available / (1024**3)
            
        return available_memory >= required_memory_gb


# Global instances
hypervector_accelerator = HypervectorAccelerator()
parallel_processor = ParallelProcessor()
resource_monitor = ResourceMonitor()
