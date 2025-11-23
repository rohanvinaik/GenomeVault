"""
Performance Prediction Models for Backend Selection

Provides simple performance models to predict execution time on different backends
based on empirical benchmark data. Used by IntelligentBackendSelector to make
informed backend selection decisions.

Models are intentionally simple (linear predictions) to avoid overfitting and
maintain interpretability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

import numpy as np

from genomevault.compute.backend import ComputeBackend

logger = logging.getLogger(__name__)


class OperationClass(Enum):
    """Operation classes for performance modeling"""
    HDC_ENCODING = "hdc_encoding"
    SIMILARITY_SEARCH = "similarity_search"
    VECTOR_OPERATIONS = "vector_operations"
    UNKNOWN = "unknown"


@dataclass
class PerformanceModel:
    """
    Simple linear performance model for an operation

    Model: time = warmup + (size * per_unit_time)
    """
    operation: str
    backend: ComputeBackend
    warmup_ms: float              # Fixed overhead (ms)
    time_per_unit_ms: float       # Per sample/record time (ms)
    crossover_point: Optional[int] = None  # Size where this backend becomes optimal

    def predict(self, size: int, include_warmup: bool = True) -> float:
        """
        Predict execution time for given data size

        Args:
            size: Number of samples/records
            include_warmup: Whether to include warmup overhead

        Returns:
            Predicted time in milliseconds
        """
        warmup = self.warmup_ms if include_warmup else 0.0
        return warmup + (size * self.time_per_unit_ms)

    def is_optimal_for_size(self, size: int) -> bool:
        """Check if this backend is optimal for given size"""
        if self.crossover_point is None:
            return True  # No crossover point = always optimal
        return size >= self.crossover_point


class PerformancePredictor:
    """
    Predicts execution time for operations on different backends

    Uses empirical models from benchmarks or configuration to estimate
    CPU vs GPU performance for informed backend selection.
    """

    def __init__(self, benchmark_data: Optional[Dict[str, Any]] = None):
        """
        Initialize performance predictor

        Args:
            benchmark_data: Dict of performance models from config/benchmarks
                           If None, uses default models from GenomeVault benchmarks
        """
        self.models: Dict[str, Dict[ComputeBackend, PerformanceModel]] = {}

        if benchmark_data:
            self._load_models_from_config(benchmark_data)
        else:
            self._load_default_models()

        logger.debug(f"Performance predictor initialized with {len(self.models)} operation models")

    def _load_default_models(self):
        """Load default performance models from GenomeVault benchmarks"""

        # HDC Encoding Models (from backend migration tests)
        # CPU: ~5-10ms per sample (from test results)
        # Metal: ~0.5-1ms per sample (from MLX benchmarks)
        # CUDA: ~2ms per sample (estimated)
        self.models[OperationClass.HDC_ENCODING.value] = {
            ComputeBackend.CPU: PerformanceModel(
                operation="hdc_encoding",
                backend=ComputeBackend.CPU,
                warmup_ms=0.0,                    # No warmup
                time_per_unit_ms=5.0,             # 5ms per sample
                crossover_point=None              # Always available
            ),
            ComputeBackend.METAL: PerformanceModel(
                operation="hdc_encoding",
                backend=ComputeBackend.METAL,
                warmup_ms=5.0,                    # 5ms Metal warmup
                time_per_unit_ms=0.5,             # 0.5ms per sample (10× faster)
                crossover_point=100               # Better than CPU for >100 samples
            ),
            ComputeBackend.CUDA: PerformanceModel(
                operation="hdc_encoding",
                backend=ComputeBackend.CUDA,
                warmup_ms=10.0,                   # 10ms CUDA warmup
                time_per_unit_ms=2.0,             # 2ms per sample (2.5× faster)
                crossover_point=150               # Better than CPU for >150 samples
            ),
        }

        # Similarity Search Models
        # CPU: ~2ms per 1K database records (from test results)
        # Metal: ~0.2ms per 1K records
        # CUDA: ~0.5ms per 1K records
        self.models[OperationClass.SIMILARITY_SEARCH.value] = {
            ComputeBackend.CPU: PerformanceModel(
                operation="similarity_search",
                backend=ComputeBackend.CPU,
                warmup_ms=0.0,
                time_per_unit_ms=0.002,           # 2ms per 1K records = 0.002ms per record
                crossover_point=None
            ),
            ComputeBackend.METAL: PerformanceModel(
                operation="similarity_search",
                backend=ComputeBackend.METAL,
                warmup_ms=5.0,
                time_per_unit_ms=0.0002,          # 0.2ms per 1K = 0.0002ms per record
                crossover_point=10000             # Better for >10K records
            ),
            ComputeBackend.CUDA: PerformanceModel(
                operation="similarity_search",
                backend=ComputeBackend.CUDA,
                warmup_ms=10.0,
                time_per_unit_ms=0.0005,          # 0.5ms per 1K = 0.0005ms per record
                crossover_point=15000             # Better for >15K records
            ),
        }

    def _load_models_from_config(self, benchmark_data: Dict[str, Any]):
        """Load performance models from configuration data"""
        for operation, backends in benchmark_data.items():
            self.models[operation] = {}

            for backend_name, params in backends.items():
                backend_enum = self._parse_backend(backend_name)

                self.models[operation][backend_enum] = PerformanceModel(
                    operation=operation,
                    backend=backend_enum,
                    warmup_ms=params.get('warmup_ms', 0.0),
                    time_per_unit_ms=params.get('time_per_unit_ms', 1.0),
                    crossover_point=params.get('crossover_point')
                )

    def _parse_backend(self, backend_name: str) -> ComputeBackend:
        """Parse backend name string to enum"""
        backend_map = {
            'cpu': ComputeBackend.CPU,
            'metal': ComputeBackend.METAL,
            'cuda': ComputeBackend.CUDA,
            'auto': ComputeBackend.AUTO,
        }
        return backend_map.get(backend_name.lower(), ComputeBackend.CPU)

    def predict_time(
        self,
        operation: str,
        backend: ComputeBackend,
        size: int,
        include_warmup: bool = True
    ) -> float:
        """
        Predict execution time for operation on backend

        Args:
            operation: Operation class (hdc_encoding, similarity_search)
            backend: Target backend
            size: Data size (samples, records, etc.)
            include_warmup: Whether to include warmup overhead

        Returns:
            Predicted time in milliseconds
        """
        # Get model for operation
        operation_models = self.models.get(operation)
        if not operation_models:
            logger.warning(f"No performance model for operation: {operation}")
            # Fallback to linear assumption
            return size * 1.0

        # Get model for backend
        model = operation_models.get(backend)
        if not model:
            # Try AUTO fallback
            model = operation_models.get(ComputeBackend.AUTO)
            if not model:
                logger.warning(f"No model for backend {backend} on operation {operation}")
                return size * 1.0

        return model.predict(size, include_warmup)

    def recommend_backend(
        self,
        operation: str,
        size: int,
        available_backends: Optional[list[ComputeBackend]] = None,
        latency_target_ms: Optional[float] = None
    ) -> tuple[ComputeBackend, str]:
        """
        Recommend optimal backend based on predicted performance

        Args:
            operation: Operation class
            size: Data size
            available_backends: List of available backends (None = all)
            latency_target_ms: Target latency (if specified, prefer meeting target)

        Returns:
            Tuple of (recommended_backend, reasoning)
        """
        operation_models = self.models.get(operation)
        if not operation_models:
            return ComputeBackend.AUTO, f"No model for {operation}, using AUTO"

        # Default to all backends if not specified
        if available_backends is None:
            available_backends = [ComputeBackend.CPU, ComputeBackend.METAL, ComputeBackend.CUDA]

        # Predict time for each available backend
        predictions = {}
        for backend in available_backends:
            if backend == ComputeBackend.AUTO:
                continue  # Skip AUTO, it's a selector not a backend

            model = operation_models.get(backend)
            if model:
                predictions[backend] = model.predict(size, include_warmup=True)

        if not predictions:
            return ComputeBackend.AUTO, "No available backend models"

        # If latency target specified, prefer backend that meets it (prefer simpler)
        if latency_target_ms is not None:
            cpu_time = predictions.get(ComputeBackend.CPU, float('inf'))
            if cpu_time <= latency_target_ms:
                return ComputeBackend.CPU, f"CPU meets latency target ({cpu_time:.1f}ms ≤ {latency_target_ms}ms)"

        # Otherwise, select fastest backend
        fastest_backend = min(predictions.keys(), key=lambda b: predictions[b])
        fastest_time = predictions[fastest_backend]

        # Check if speedup is significant (>20% improvement)
        cpu_time = predictions.get(ComputeBackend.CPU, fastest_time)
        if fastest_backend != ComputeBackend.CPU:
            speedup = cpu_time / fastest_time
            if speedup < 1.2:  # <20% improvement
                return ComputeBackend.CPU, f"GPU speedup minimal ({speedup:.1f}×), prefer CPU simplicity"

        reason = f"Fastest backend: {fastest_time:.1f}ms (vs CPU: {cpu_time:.1f}ms)"
        return fastest_backend, reason

    def get_crossover_point(self, operation: str, backend: ComputeBackend) -> Optional[int]:
        """
        Get data size where backend becomes optimal

        Args:
            operation: Operation class
            backend: Backend to check

        Returns:
            Crossover point (size) or None
        """
        operation_models = self.models.get(operation)
        if not operation_models:
            return None

        model = operation_models.get(backend)
        return model.crossover_point if model else None

    def compare_backends(
        self,
        operation: str,
        size: int,
        backends: Optional[list[ComputeBackend]] = None
    ) -> Dict[ComputeBackend, float]:
        """
        Compare predicted times for multiple backends

        Args:
            operation: Operation class
            size: Data size
            backends: Backends to compare (None = all)

        Returns:
            Dict mapping backend to predicted time
        """
        if backends is None:
            backends = [ComputeBackend.CPU, ComputeBackend.METAL, ComputeBackend.CUDA]

        predictions = {}
        for backend in backends:
            predictions[backend] = self.predict_time(operation, backend, size)

        return predictions
