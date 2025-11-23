"""
Intelligent Backend Selection

Analyzes data characteristics and analysis requirements to dynamically select
optimal hardware backend (CPU/Metal/CUDA), while respecting configuration overrides.

This module provides data-driven backend selection as an alternative to the
static configuration-based approach in compute.yaml.

Usage:
    selector = IntelligentBackendSelector()
    backend = selector.select_backend_for_operation(
        operation='encode',
        data=my_data,
        context={'interactive': True, 'batch': False}
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any
from enum import Enum

import numpy as np

from genomevault.compute.backend import ComputeBackend
from genomevault.config.loader import get_config

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations for backend selection"""
    SINGLE_ENCODE = "single_encode"
    BATCH_ENCODE = "batch_encode"
    SIMILARITY_SEARCH = "similarity_search"
    ZK_PROOF = "zk_proof"
    PIR_QUERY = "pir_query"
    UNKNOWN = "unknown"


class LatencyRequirement(Enum):
    """Latency sensitivity levels"""
    REAL_TIME = "real_time"          # <100ms target, prefer CPU to avoid GPU warmup
    INTERACTIVE = "interactive"      # <1s target, GPU acceptable if faster
    BATCH = "batch"                  # >1s acceptable, optimize for throughput
    OFFLINE = "offline"              # No time constraint, maximize throughput


@dataclass
class DataProfile:
    """Characteristics of input data for backend selection"""
    size: int                        # Number of samples/records
    dimensionality: int              # Feature dimensions
    sparsity: float                  # Percentage of zeros (0.0 - 1.0)
    memory_footprint_mb: float       # Estimated memory needed
    complexity_score: float          # Estimated computation cost (0.0 - 1.0)

    @classmethod
    def from_array(cls, data: np.ndarray) -> 'DataProfile':
        """Create data profile from numpy array"""
        if data.ndim == 1:
            data = data.reshape(1, -1)

        size = data.shape[0]
        dimensionality = data.shape[1] if data.ndim > 1 else 1

        # Calculate sparsity
        sparsity = np.count_nonzero(data == 0) / data.size if data.size > 0 else 0.0

        # Estimate memory footprint
        memory_footprint_mb = data.nbytes / (1024 * 1024)

        # Complexity score based on size and dimensionality
        # High dimensional data with many samples = higher complexity
        complexity_score = min(1.0, (size * dimensionality) / 1e6)

        return cls(
            size=size,
            dimensionality=dimensionality,
            sparsity=sparsity,
            memory_footprint_mb=memory_footprint_mb,
            complexity_score=complexity_score
        )


@dataclass
class AnalysisProfile:
    """Analysis requirements for backend selection"""
    operation: OperationType
    latency_requirement: LatencyRequirement
    throughput_focused: bool         # Optimize for throughput over latency
    interactive: bool                # User waiting for results
    batch_processing: bool           # Part of larger batch pipeline
    recommended_backend: Optional[ComputeBackend] = None
    reason: str = ""


class IntelligentBackendSelector:
    """
    Intelligent backend selection based on data and analysis characteristics

    Decision Process:
    1. Check for mandatory CPU operations (ZK, PIR)
    2. Analyze data characteristics
    3. Consider analysis requirements
    4. Predict performance on available backends
    5. Select optimal backend with transparent reasoning

    Fallback: Always falls back to config-based selection on errors
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize intelligent selector

        Args:
            config: Configuration dict. If None, loads from compute.yaml
        """
        if config is None:
            try:
                full_config = get_config()
                self.config = full_config.config.get('compute', {}).get('intelligent_mode', {})
            except Exception as e:
                logger.warning(f"Failed to load intelligent mode config: {e}")
                self.config = self._default_config()
        else:
            self.config = config

        # Load thresholds from config
        thresholds = self.config.get('thresholds', {})
        self.small_data_threshold = thresholds.get('small_data_samples', 100)
        self.large_data_threshold = thresholds.get('large_data_samples', 1000)
        self.gpu_warmup_ms = thresholds.get('gpu_warmup_cost_ms', 5.0)
        self.interactive_latency_target_ms = thresholds.get('interactive_latency_target_ms', 100.0)

        # Load performance models
        self.performance_models = self.config.get('performance_models', {})

        logger.info("Intelligent backend selector initialized")
        logger.debug(f"  Thresholds: small={self.small_data_threshold}, large={self.large_data_threshold}")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if config file unavailable"""
        return {
            'enabled': False,
            'thresholds': {
                'small_data_samples': 100,
                'large_data_samples': 1000,
                'gpu_warmup_cost_ms': 5.0,
                'interactive_latency_target_ms': 100.0,
            },
            'performance_models': {
                'hdc_encoding': {
                    'cpu_time_per_sample_ms': 5.0,
                    'gpu_time_per_sample_ms': 0.5,
                    'gpu_warmup_ms': 5.0,
                    'batch_crossover_point': 100,
                },
                'similarity_search': {
                    'cpu_time_per_1k_db_ms': 2.0,
                    'gpu_time_per_1k_db_ms': 0.2,
                    'database_crossover_point': 10000,
                }
            }
        }

    def analyze_data(self, data: Union[np.ndarray, list, int]) -> DataProfile:
        """
        Analyze data characteristics

        Args:
            data: Input data (numpy array, list, or size hint as int)

        Returns:
            DataProfile with analyzed characteristics
        """
        if isinstance(data, int):
            # Size hint provided instead of actual data
            return DataProfile(
                size=data,
                dimensionality=0,  # Unknown
                sparsity=0.0,
                memory_footprint_mb=0.0,
                complexity_score=min(1.0, data / 1000.0)
            )

        if isinstance(data, list):
            data = np.array(data)

        return DataProfile.from_array(data)

    def infer_analysis_type(
        self,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AnalysisProfile:
        """
        Infer analysis requirements from operation and context

        Args:
            operation: Operation name ('encode', 'search', 'prove', 'retrieve')
            context: Additional context (batch, interactive, latency_sensitive)

        Returns:
            AnalysisProfile with inferred requirements
        """
        context = context or {}

        # Map operation to type
        op_map = {
            'encode': OperationType.SINGLE_ENCODE,
            'encode_single': OperationType.SINGLE_ENCODE,
            'encode_batch': OperationType.BATCH_ENCODE,
            'search': OperationType.SIMILARITY_SEARCH,
            'similarity': OperationType.SIMILARITY_SEARCH,
            'prove': OperationType.ZK_PROOF,
            'zk': OperationType.ZK_PROOF,
            'retrieve': OperationType.PIR_QUERY,
            'pir': OperationType.PIR_QUERY,
        }

        operation_type = op_map.get(operation.lower(), OperationType.UNKNOWN)

        # Determine latency requirement
        if context.get('latency_sensitive') or context.get('interactive'):
            latency_req = LatencyRequirement.REAL_TIME
        elif context.get('batch'):
            latency_req = LatencyRequirement.BATCH
        else:
            latency_req = LatencyRequirement.INTERACTIVE

        return AnalysisProfile(
            operation=operation_type,
            latency_requirement=latency_req,
            throughput_focused=context.get('throughput_focused', False),
            interactive=context.get('interactive', False),
            batch_processing=context.get('batch', False)
        )

    def predict_cpu_time(self, operation: OperationType, data_size: int) -> float:
        """
        Predict CPU execution time in milliseconds

        Args:
            operation: Type of operation
            data_size: Number of samples/records

        Returns:
            Predicted time in milliseconds
        """
        models = self.performance_models

        if operation == OperationType.SINGLE_ENCODE or operation == OperationType.BATCH_ENCODE:
            model = models.get('hdc_encoding', {})
            time_per_sample = model.get('cpu_time_per_sample_ms', 5.0)
            return data_size * time_per_sample

        elif operation == OperationType.SIMILARITY_SEARCH:
            model = models.get('similarity_search', {})
            time_per_1k = model.get('cpu_time_per_1k_db_ms', 2.0)
            return (data_size / 1000.0) * time_per_1k

        else:
            # Unknown operation, assume linear scaling
            return data_size * 1.0

    def predict_gpu_time(
        self,
        operation: OperationType,
        data_size: int,
        include_warmup: bool = True
    ) -> float:
        """
        Predict GPU execution time in milliseconds

        Args:
            operation: Type of operation
            data_size: Number of samples/records
            include_warmup: Whether to include GPU warmup overhead

        Returns:
            Predicted time in milliseconds
        """
        models = self.performance_models
        warmup = self.gpu_warmup_ms if include_warmup else 0.0

        if operation == OperationType.SINGLE_ENCODE or operation == OperationType.BATCH_ENCODE:
            model = models.get('hdc_encoding', {})
            time_per_sample = model.get('gpu_time_per_sample_ms', 0.5)
            return warmup + (data_size * time_per_sample)

        elif operation == OperationType.SIMILARITY_SEARCH:
            model = models.get('similarity_search', {})
            time_per_1k = model.get('gpu_time_per_1k_db_ms', 0.2)
            return warmup + ((data_size / 1000.0) * time_per_1k)

        else:
            # Unknown operation, assume linear scaling with warmup
            return warmup + (data_size * 0.1)

    def select_backend(
        self,
        data_profile: DataProfile,
        analysis_profile: AnalysisProfile,
        config_override: Optional[ComputeBackend] = None,
        available_backends: Optional[list[ComputeBackend]] = None
    ) -> tuple[ComputeBackend, str]:
        """
        Select optimal backend based on data and analysis profiles

        Args:
            data_profile: Analyzed data characteristics
            analysis_profile: Analysis requirements
            config_override: Config-mandated backend (highest priority)
            available_backends: List of available backends

        Returns:
            Tuple of (selected_backend, reasoning)
        """
        # Log input profiles
        logger.info("Intelligent Backend Selection:")
        logger.info(f"  Data: {data_profile.size} samples, "
                   f"{data_profile.sparsity:.1%} sparse, "
                   f"{data_profile.memory_footprint_mb:.1f}MB")
        logger.info(f"  Analysis: {analysis_profile.operation.value}, "
                   f"latency={analysis_profile.latency_requirement.value}")

        # 1. Config override takes absolute precedence
        if config_override:
            reason = f"Config override mandates {config_override.value}"
            logger.info(f"  → {reason}")
            return config_override, reason

        # 2. Mandatory CPU operations (ZK, PIR) - never use GPU
        if analysis_profile.operation in [OperationType.ZK_PROOF, OperationType.PIR_QUERY]:
            reason = f"{analysis_profile.operation.value} operations require CPU (algorithm design)"
            logger.info(f"  → {reason}")
            return ComputeBackend.CPU, reason

        # 3. Predict performance on CPU and GPU
        cpu_time = self.predict_cpu_time(analysis_profile.operation, data_profile.size)
        gpu_time = self.predict_gpu_time(analysis_profile.operation, data_profile.size, include_warmup=True)

        logger.info(f"  Predicted: CPU {cpu_time:.1f}ms, GPU {gpu_time:.1f}ms")

        # 4. Decision logic based on latency requirements and predicted performance

        # Real-time latency requirement - prefer CPU to avoid GPU warmup
        if analysis_profile.latency_requirement == LatencyRequirement.REAL_TIME:
            if cpu_time < self.interactive_latency_target_ms:
                reason = f"Real-time latency requirement met by CPU ({cpu_time:.1f}ms < {self.interactive_latency_target_ms}ms)"
                logger.info(f"  → CPU selected: {reason}")
                return ComputeBackend.CPU, reason
            elif gpu_time < cpu_time * 0.5:  # GPU must be 2× faster to justify warmup
                reason = f"GPU significantly faster despite warmup ({gpu_time:.1f}ms vs {cpu_time:.1f}ms)"
                logger.info(f"  → GPU selected: {reason}")
                return ComputeBackend.AUTO, reason
            else:
                reason = f"CPU preferred for real-time operation ({cpu_time:.1f}ms, predictable)"
                logger.info(f"  → CPU selected: {reason}")
                return ComputeBackend.CPU, reason

        # Small data - CPU almost always better due to GPU overhead
        if data_profile.size < self.small_data_threshold:
            reason = f"Small data ({data_profile.size} < {self.small_data_threshold}) - CPU overhead-free"
            logger.info(f"  → CPU selected: {reason}")
            return ComputeBackend.CPU, reason

        # Large data - GPU likely better if available
        if data_profile.size >= self.large_data_threshold:
            if gpu_time < cpu_time:
                reason = f"Large data ({data_profile.size} ≥ {self.large_data_threshold}) - GPU faster ({gpu_time:.1f}ms < {cpu_time:.1f}ms)"
                logger.info(f"  → GPU selected: {reason}")
                return ComputeBackend.AUTO, reason

        # Medium data - compare predicted times directly
        if gpu_time < cpu_time * 0.8:  # GPU must be 20% faster
            reason = f"GPU faster for medium data ({gpu_time:.1f}ms vs {cpu_time:.1f}ms)"
            logger.info(f"  → GPU selected: {reason}")
            return ComputeBackend.AUTO, reason
        else:
            reason = f"CPU comparable or better for medium data ({cpu_time:.1f}ms vs {gpu_time:.1f}ms)"
            logger.info(f"  → CPU selected: {reason}")
            return ComputeBackend.CPU, reason

    def select_backend_for_operation(
        self,
        operation: str,
        data: Union[np.ndarray, list, int],
        context: Optional[Dict[str, Any]] = None,
        config_override: Optional[ComputeBackend] = None
    ) -> tuple[ComputeBackend, str]:
        """
        High-level API: Select backend for an operation

        Args:
            operation: Operation name ('encode', 'search', etc.)
            data: Input data or size hint
            context: Operation context (batch, interactive, etc.)
            config_override: Config-mandated backend

        Returns:
            Tuple of (selected_backend, reasoning)

        Example:
            backend, reason = selector.select_backend_for_operation(
                operation='encode',
                data=my_variants,
                context={'interactive': True}
            )
        """
        try:
            # Analyze data and operation
            data_profile = self.analyze_data(data)
            analysis_profile = self.infer_analysis_type(operation, context)

            # Select backend
            return self.select_backend(
                data_profile=data_profile,
                analysis_profile=analysis_profile,
                config_override=config_override
            )

        except Exception as e:
            # Fallback to safe default on any error
            logger.error(f"Intelligent backend selection failed: {e}")
            logger.info("Falling back to AUTO backend")
            return ComputeBackend.AUTO, f"Fallback due to error: {e}"
