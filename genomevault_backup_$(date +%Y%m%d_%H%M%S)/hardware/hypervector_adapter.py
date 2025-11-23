"""Adapter to use unified hardware acceleration for hypervector operations."""

import numpy as np
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass

from genomevault.hardware.unified_engine import UnifiedAccelerationEngine, AccelerationConfig
from genomevault.hardware.backend import AcceleratorType
from genomevault.core.constants import OmicsType
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HypervectorConfig:
    """Configuration for hypervector operations."""

    dimension: int = 10000
    batch_size: int = 1024
    precision: str = "float32"
    use_sparse: bool = True
    normalization: str = "l2"  # l2, l1, or none

    def to_acceleration_config(self) -> AccelerationConfig:
        """Convert to acceleration config."""
        return AccelerationConfig(
            dimension=self.dimension, batch_size=self.batch_size, precision=self.precision
        )


class HardwareAcceleratedHypervectorEngine:
    """
    Hypervector engine using unified hardware acceleration.

    This replaces the separate Metal and LocalGPU engines with a single
    unified implementation that automatically selects the best backend.
    """

    def __init__(self, config: Optional[HypervectorConfig] = None):
        """
        Initialize hardware-accelerated hypervector engine.

        Args:
            config: Hypervector configuration
        """
        self.config = config or HypervectorConfig()

        # Initialize unified acceleration engine
        self.engine = UnifiedAccelerationEngine(self.config.to_acceleration_config())

        # Cache for projection matrices
        self._projection_matrices = {}

        logger.info(
            f"🧬 Hardware-Accelerated Hypervector Engine\n"
            f"  Backend: {self.engine.backend.type.value}\n"
            f"  Device: {self.engine.backend.name}\n"
            f"  Dimension: {self.config.dimension}"
        )

    def encode(
        self, data: Union[np.ndarray, List[float]], omics_type: OmicsType = OmicsType.GENOMIC
    ) -> np.ndarray:
        """
        Encode data to hypervector using hardware acceleration.

        Args:
            data: Input data (features or variants)
            omics_type: Type of omics data

        Returns:
            Encoded hypervector
        """
        # Convert to numpy array
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)

        # Get or create projection matrix
        projection = self._get_projection_matrix(omics_type, data.shape[-1])

        # Move data to device
        data_dev = self.engine.to_device(data)

        # Perform encoding
        if len(data.shape) == 1:
            # Single sample
            hypervector = self.engine.matmul(projection, data_dev[:, None])[:, 0]
        else:
            # Batch processing
            hypervector = self.engine.matmul(data_dev, projection.T)

        # Apply activation and normalization
        hypervector = self._apply_activation(hypervector)

        if self.config.normalization == "l2":
            hypervector = self.engine.normalize(hypervector)

        # Move back to CPU
        return self.engine.from_device(hypervector)

    def _get_projection_matrix(self, omics_type: OmicsType, input_dim: int) -> Any:
        """
        Get or create projection matrix for encoding.

        Args:
            omics_type: Type of omics data
            input_dim: Input dimension

        Returns:
            Projection matrix (device array)
        """
        key = f"{omics_type.value}_{input_dim}"

        if key not in self._projection_matrices:
            # Create new projection matrix
            logger.debug(f"Creating projection matrix: {input_dim} -> {self.config.dimension}")

            # Use different initialization based on omics type
            if omics_type == OmicsType.GENOMIC and self.config.use_sparse:
                # Sparse initialization for genomic data
                projection = self.engine.random_projection_matrix(
                    input_dim, self.config.dimension, sparse=True
                )
            else:
                # Dense initialization for other data types
                projection = self.engine.random_projection_matrix(
                    input_dim, self.config.dimension, sparse=False
                )

            self._projection_matrices[key] = projection

        return self._projection_matrices[key]

    def _apply_activation(self, hypervector: Any) -> Any:
        """
        Apply activation function to hypervector.

        Args:
            hypervector: Input hypervector (device array)

        Returns:
            Activated hypervector (device array)
        """
        # ReLU activation for sparsity
        if self.engine.backend.type == AcceleratorType.METAL and hasattr(self.engine, "mx"):
            return self.engine.mx.maximum(hypervector, 0)
        elif self.engine.backend.type == AcceleratorType.CUDA and hasattr(self.engine, "cp"):
            return self.engine.cp.maximum(hypervector, 0)
        elif self.engine.backend.type == AcceleratorType.CUDA and hasattr(self.engine, "torch"):
            return self.engine.torch.relu(hypervector)
        else:
            return np.maximum(hypervector, 0)

    def compute_similarity(
        self, query: np.ndarray, database: np.ndarray, metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute similarity between query and database vectors.

        Args:
            query: Query hypervector
            database: Database of hypervectors
            metric: Similarity metric (cosine, hamming, euclidean)

        Returns:
            Similarity scores
        """
        # Move to device
        query_dev = self.engine.to_device(query)
        database_dev = self.engine.to_device(database)

        if metric == "cosine":
            similarities = self.engine.cosine_similarity(query_dev, database_dev)
        elif metric == "hamming":
            distances = self.engine.hamming_distance(query_dev, database_dev)
            similarities = 1.0 - distances
        elif metric == "euclidean":
            # Compute Euclidean distance
            if len(query.shape) == 1:
                diff = database_dev - query_dev[None, :]
            else:
                diff = database_dev - query_dev

            if hasattr(self.engine, "mx"):
                distances = self.engine.mx.linalg.norm(diff, axis=-1)
            elif hasattr(self.engine, "cp"):
                distances = self.engine.cp.linalg.norm(diff, axis=-1)
            else:
                distances = np.linalg.norm(self.engine.from_device(diff), axis=-1)
                distances = self.engine.to_device(distances)

            similarities = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Move back to CPU
        return self.engine.from_device(similarities)

    def optimize_for_inference(self, hypervector: np.ndarray) -> np.ndarray:
        """
        Optimize hypervector for inference (quantization, pruning).

        Args:
            hypervector: Input hypervector

        Returns:
            Optimized hypervector
        """
        # Quantize to binary for maximum compression
        return (hypervector > 0).astype(np.int8)

    def benchmark(self, input_dim: int = 1000, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Benchmark hardware acceleration performance.

        Args:
            input_dim: Input dimension
            num_samples: Number of samples to process

        Returns:
            Benchmark results
        """
        import time

        logger.info(f"Running benchmark: {num_samples} samples, {input_dim} features")

        # Generate test data
        data = np.random.randn(num_samples, input_dim).astype(np.float32)

        # Warmup
        _ = self.encode(data[:10])

        # Benchmark encoding
        start = time.perf_counter()
        encoded = self.encode(data)
        encoding_time = time.perf_counter() - start

        # Benchmark similarity computation
        query = encoded[0]
        start = time.perf_counter()
        similarities = self.compute_similarity(query, encoded)
        similarity_time = time.perf_counter() - start

        results = {
            "backend": self.engine.backend.type.value,
            "device": self.engine.backend.name,
            "samples": num_samples,
            "input_dim": input_dim,
            "output_dim": self.config.dimension,
            "encoding_time_ms": encoding_time * 1000,
            "similarity_time_ms": similarity_time * 1000,
            "samples_per_second": num_samples / encoding_time,
            "precision": self.config.precision,
        }

        logger.info(
            f"Benchmark Results:\n"
            f"  Backend: {results['backend']}\n"
            f"  Encoding: {results['encoding_time_ms']:.2f}ms "
            f"({results['samples_per_second']:.0f} samples/sec)\n"
            f"  Similarity: {results['similarity_time_ms']:.2f}ms"
        )

        return results


# Backward compatibility aliases
MetalHypervectorEngine = HardwareAcceleratedHypervectorEngine
LocalGPUEngine = HardwareAcceleratedHypervectorEngine


def get_hypervector_engine(
    backend: Optional[str] = None, config: Optional[HypervectorConfig] = None
) -> HardwareAcceleratedHypervectorEngine:
    """
    Get hypervector engine with specified backend.

    Args:
        backend: Backend to use (metal, cuda, cpu, or None for auto)
        config: Hypervector configuration

    Returns:
        Hardware-accelerated hypervector engine
    """
    if config is None:
        config = HypervectorConfig()

    if backend:
        # Map backend string to AcceleratorType
        backend_map = {
            "metal": AcceleratorType.METAL,
            "cuda": AcceleratorType.CUDA,
            "rocm": AcceleratorType.ROCM,
            "tpu": AcceleratorType.TPU,
            "cpu": AcceleratorType.CPU,
        }

        if backend.lower() in backend_map:
            accel_config = config.to_acceleration_config()
            accel_config.device = backend_map[backend.lower()]
            engine = HardwareAcceleratedHypervectorEngine(config)
            engine.engine.config.device = backend_map[backend.lower()]

    return HardwareAcceleratedHypervectorEngine(config)
