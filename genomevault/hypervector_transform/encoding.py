"""Encoding module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Union
import logging

from numpy.typing import NDArray
import numpy as np
import torch

from genomevault.core.constants import HYPERVECTOR_DIMENSIONS, OmicsType
from genomevault.core.exceptions import EncodingError, ProjectionError

logger = logging.getLogger(__name__)

# Optional differential privacy integration
try:
    from genomevault.privacy import (
        GaussianMechanism,
        PrivacyLevel,
        PrivacyAccountant,
        DifferentiallyPrivateHDC,
    )

    DP_AVAILABLE = True
except ImportError:
    DP_AVAILABLE = False

# Optional Metal acceleration
try:
    from genomevault.hypervector.metal_engine import MetalHypervectorEngine, MetalConfig
    import mlx.core as mx
    METAL_AVAILABLE = True
    print("METAL ACCELERATION DETECTED!")  # Debug print
    logger.info("Metal acceleration support detected")
except ImportError as e:
    METAL_AVAILABLE = False
    MetalHypervectorEngine = None
    MetalConfig = None
    print(f"METAL IMPORT FAILED: {e}")  # Debug print
    logger.debug(f"Metal acceleration not available: {e}")
except Exception as e:
    METAL_AVAILABLE = False
    MetalHypervectorEngine = None
    MetalConfig = None
    print(f"METAL IMPORT ERROR (non-ImportError): {e}")  # Debug print
    logger.debug(f"Metal acceleration error: {e}")

TensorLike = Union[np.ndarray, torch.Tensor]


class ProjectionType(Enum):
    """ProjectionType implementation."""

    RANDOM_GAUSSIAN = "random_gaussian"
    SPARSE_RANDOM = "sparse_random"
    ORTHOGONAL = "orthogonal"


@dataclass
class HypervectorConfig:
    """Data container for hypervectorconfig information."""

    dimension: int = HYPERVECTOR_DIMENSIONS
    projection_type: ProjectionType = ProjectionType.SPARSE_RANDOM
    sparsity: float = 0.1
    seed: Optional[int] = None
    normalize: bool = True
    quantize: bool = False
    quantization_bits: int = 8
    # Differential privacy parameters
    use_differential_privacy: bool = False
    privacy_level: Optional["PrivacyLevel"] = None
    privacy_epsilon: Optional[float] = None
    privacy_delta: Optional[float] = None
    # Metal acceleration parameters
    use_metal: Optional[bool] = None  # None = auto-detect
    metal_memory_gb: float = 20.0  # Target memory allocation for Metal


class HypervectorEncoder:
    """Minimal, correct encoder to unblock tests."""

    def __init__(self, config: Optional[HypervectorConfig] = None) -> None:
        """Initialize instance.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or HypervectorConfig()
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
        self._projection_cache: Dict[str, torch.Tensor] = {}

        # Initialize Metal acceleration if available and requested
        self.metal_engine = None
        if self.config.use_metal is None:
            # Auto-detect Metal availability
            if METAL_AVAILABLE:
                try:
                    metal_config = MetalConfig(
                        dimension=self.config.dimension,
                        max_memory_gb=self.config.metal_memory_gb,
                        use_neural_engine=True,
                        precision="float32"
                    )
                    self.metal_engine = MetalHypervectorEngine(metal_config)
                    logger.info(f"🍎 Metal acceleration auto-enabled with {self.config.metal_memory_gb}GB memory")
                except Exception as e:
                    logger.warning(f"Metal auto-detection failed: {e}")
        elif self.config.use_metal and METAL_AVAILABLE:
            # Explicitly requested Metal
            metal_config = MetalConfig(
                dimension=self.config.dimension,
                max_memory_gb=self.config.metal_memory_gb,
                use_neural_engine=True,
                precision="float32"
            )
            self.metal_engine = MetalHypervectorEngine(metal_config)
            logger.info(f"🍎 Metal acceleration enabled with {self.config.metal_memory_gb}GB memory")

        # Initialize differential privacy if requested
        self.dp_mechanism = None
        self.privacy_accountant = None

        if self.config.use_differential_privacy and DP_AVAILABLE:
            if self.config.privacy_level:
                # Use predefined privacy level
                epsilon, delta = self.config.privacy_level.value
            elif self.config.privacy_epsilon and self.config.privacy_delta:
                # Use custom privacy parameters
                epsilon = self.config.privacy_epsilon
                delta = self.config.privacy_delta
            else:
                # Default to clinical level
                epsilon, delta = 1.0, 1e-7

            # Initialize privacy accountant
            self.privacy_accountant = PrivacyAccountant(
                total_epsilon=epsilon * 100,  # Budget for 100 operations
                total_delta=delta * 100,
            )

            # Sensitivity for normalized hypervectors (max L2 distance = sqrt(2))
            sensitivity = np.sqrt(2.0)

            self.dp_mechanism = GaussianMechanism(epsilon, delta, sensitivity)
            logger.info(
                "Differential privacy enabled: ε=%.2f, δ=%.2e, σ=%.4f",
                epsilon,
                delta,
                self.dp_mechanism.sigma,
            )

        logger.info(
            "Initialized HypervectorEncoder(dim=%d, proj=%s, dp=%s)",
            self.config.dimension,
            self.config.projection_type.value,
            self.config.use_differential_privacy,
        )

    def encode(
        self,
        features: Union[TensorLike, Mapping[str, TensorLike]],
        omics_type: OmicsType,
        *,
        resolution: str = "base",
        add_dp_noise: bool = True,
    ) -> torch.Tensor:
        """Encode features into a single hypervector with optional differential privacy."""
        try:
            # Use Metal acceleration if available
            if self.metal_engine is not None:
                # Convert features to numpy for Metal
                if isinstance(features, torch.Tensor):
                    features_np = features.detach().cpu().numpy()
                elif isinstance(features, dict):
                    # Handle dict features
                    features_np = np.concatenate([
                        v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                        for v in features.values()
                    ])
                else:
                    features_np = np.array(features)
                
                # Encode with Metal
                hv_metal = self.metal_engine.encode_with_metal(features_np, omics_type)
                
                # Convert back to torch tensor
                hv_np = self.metal_engine.to_numpy(hv_metal)
                hv = torch.from_numpy(hv_np).float()
                
                # Metal already normalizes, skip additional normalization
            else:
                # Original CPU/CUDA path
                x = self._as_tensor(features)
                proj = self._get_projection_matrix(x.shape[-1], self.config.dimension, omics_type)
                hv = proj @ x.float()
                if self.config.normalize:
                    hv = self._normalize(hv)

            # Add differential privacy noise if enabled
            if self.config.use_differential_privacy and add_dp_noise and self.dp_mechanism:
                try:
                    # Convert to numpy for DP mechanism
                    hv_numpy = hv.detach().cpu().numpy()

                    # Allocate privacy budget if accountant available
                    if self.privacy_accountant:
                        params = self.privacy_accountant.allocate_budget(
                            "hdc_encoder",
                            f"encode_{omics_type.value}",
                            self.dp_mechanism.params.epsilon,
                        )
                        # Update mechanism with allocated budget
                        self.dp_mechanism.params = params

                    # Add noise
                    hv_noisy = self.dp_mechanism.add_noise(hv_numpy)

                    # Re-normalize after adding noise
                    hv_noisy = hv_noisy / (np.linalg.norm(hv_noisy) + 1e-10)

                    # Convert back to tensor
                    hv = torch.from_numpy(hv_noisy).float()

                    logger.debug(
                        "Added differential privacy noise (σ=%.4f)", self.dp_mechanism.sigma
                    )

                except Exception as e:
                    logger.warning(f"Failed to add DP noise: {e}, continuing without privacy")

            if self.config.quantize:
                hv = self._quantize(hv, bits=self.config.quantization_bits)
            return hv.view(-1)
        except ProjectionError:
            raise
        except Exception as e:
            logger.exception("Encoding failed")
            raise EncodingError(f"Failed to encode features: {e!s}") from e

    def encode_multiresolution(
        self,
        features: Union[TensorLike, Mapping[str, TensorLike]],
        omics_type: OmicsType,
        resolutions: Iterable[int] = (10_000, 15_000, 20_000),
    ) -> Dict[int, torch.Tensor]:
        """Encode the same features at multiple dimensions."""
        x = self._as_tensor(features)
        out: Dict[int, torch.Tensor] = {}
        for dim in resolutions:
            proj = self._get_projection_matrix(x.shape[-1], dim, omics_type)
            hv = proj @ x.float()
            if self.config.normalize:
                hv = self._normalize(hv)
            if self.config.quantize:
                hv = self._quantize(hv, bits=self.config.quantization_bits)
            out[dim] = hv.view(-1)
        return out

    # --- internals ---

    def _as_tensor(self, features: Union[TensorLike, Mapping[str, TensorLike]]) -> torch.Tensor:
        """as tensor.
        Args:        features: List of items.
        Returns:
            torch.Tensor"""
        if isinstance(features, Mapping):
            # deterministic order
            arrs = [np.asarray(v) for k, v in sorted(features.items())]
            x = np.concatenate([a.ravel() for a in arrs]).astype(np.float32)
            return torch.from_numpy(x.astype(np.float32, copy=False))
        if isinstance(features, np.ndarray):
            return torch.from_numpy(features.astype(np.float32).ravel())
        if isinstance(features, torch.Tensor):
            return features.view(-1).float()
        raise EncodingError(f"Unsupported feature type: {type(features)!r}")

    def _cache_key(self, input_dim: int, output_dim: int, omics_type: OmicsType) -> str:
        """ cache key.
            Args:        input_dim: Parameter value.        output_dim: Parameter value.        \
                omics_type: Parameter value.
            Returns:
                str    """
        return f"{omics_type.value}:{input_dim}->{output_dim}:{self.config.projection_type.value}"

    def _get_projection_matrix(
        self, input_dim: int, output_dim: int, omics_type: OmicsType
    ) -> torch.Tensor:
        """ get projection matrix.
            Args:        input_dim: Parameter value.        output_dim: Parameter value.        \
                omics_type: Parameter value.
            Returns:
                torch.Tensor    """
        key = self._cache_key(input_dim, output_dim, omics_type)
        if key in self._projection_cache:
            return self._projection_cache[key]

        if self.config.projection_type == ProjectionType.RANDOM_GAUSSIAN:
            mat = torch.randn(output_dim, input_dim) / np.sqrt(input_dim)
        elif self.config.projection_type == ProjectionType.SPARSE_RANDOM:
            mat = self._sparse_random(output_dim, input_dim, sparsity=self.config.sparsity)
        elif self.config.projection_type == ProjectionType.ORTHOGONAL:
            mat = self._orthogonal(output_dim, input_dim)
        else:
            raise ProjectionError(f"Unsupported projection type {self.config.projection_type}")

        self._projection_cache[key] = mat
        return mat

    def _sparse_random(self, rows: int, cols: int, *, sparsity: float) -> torch.Tensor:
        """sparse random.
        Args:        rows: List of items.        cols: List of items.
        Returns:
            torch.Tensor"""
        # Achlioptas-style: values in {-1, 0, +1}
        probs = [sparsity / 2, 1 - sparsity, sparsity / 2]
        vals: NDArray[np.float32] = np.random.choice(
            [-1.0, 0.0, 1.0], size=(rows, cols), p=probs
        ).astype(np.float32)
        mat = torch.from_numpy(vals.astype(np.float32, copy=False))
        # scale so that E[||x||] is preserved
        if sparsity > 0:
            mat = mat / np.sqrt(sparsity * cols)
        return mat

    def _orthogonal(self, rows: int, cols: int) -> torch.Tensor:
        """orthogonal.
        Args:        rows: List of items.        cols: List of items.
        Returns:
            torch.Tensor"""
        # Build via QR on a Gaussian matrix and crop/pad
        a = torch.randn(max(rows, cols), max(rows, cols))
        q, _ = torch.linalg.qr(a)
        return q[:rows, :cols].contiguous()

    def _normalize(self, hv: torch.Tensor) -> torch.Tensor:
        """normalize.
        Args:        hv: Parameter value.
        Returns:
            torch.Tensor"""
        n = torch.norm(hv, p=2).clamp_min(1e-12)
        return hv / n

    def _quantize(self, hv: torch.Tensor, *, bits: int = 8) -> torch.Tensor:
        """quantize.
        Args:        hv: Parameter value.
        Returns:
            torch.Tensor"""
        # symmetric uniform quantization to int8/intN
        scale = hv.abs().max().clamp_min(1e-8)
        q = torch.clamp(
            (hv / scale) * (2 ** (bits - 1) - 1),
            min=-(2 ** (bits - 1)),
            max=(2 ** (bits - 1) - 1),
        )
        return q.round()


# Convenience functions
def create_encoder(
    dimension: int = HYPERVECTOR_DIMENSIONS,
    projection_type: str = "sparse_random",
    **kwargs,
) -> HypervectorEncoder:
    """Create a hypervector encoder with specified configuration."""
    config = HypervectorConfig(
        dimension=dimension, projection_type=ProjectionType(projection_type), **kwargs
    )
    return HypervectorEncoder(config)


def encode_genomic_data(
    genomic_data: Union[TensorLike, Mapping[str, TensorLike]],
    dimension: int = HYPERVECTOR_DIMENSIONS,
) -> torch.Tensor:
    """Convenience function to encode genomic data."""
    encoder = create_encoder(dimension=dimension)
    return encoder.encode(genomic_data, OmicsType.GENOMIC)
