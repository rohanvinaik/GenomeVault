# genomevault/hypervector_transform/encoding.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Union

import numpy as np
import torch

from genomevault.core.constants import HYPERVECTOR_DIMENSIONS, OmicsType
from genomevault.core.exceptions import EncodingError, ProjectionError

TensorLike = Union[np.ndarray, torch.Tensor]
logger = logging.getLogger(__name__)


class ProjectionType(Enum):
    RANDOM_GAUSSIAN = "random_gaussian"
    SPARSE_RANDOM = "sparse_random"
    ORTHOGONAL = "orthogonal"


@dataclass
class HypervectorConfig:
    dimension: int = HYPERVECTOR_DIMENSIONS
    projection_type: ProjectionType = ProjectionType.SPARSE_RANDOM
    sparsity: float = 0.1
    seed: Optional[int] = None
    normalize: bool = True
    quantize: bool = False
    quantization_bits: int = 8


class HypervectorEncoder:
    """Minimal, correct encoder to unblock tests."""

    def __init__(self, config: Optional[HypervectorConfig] = None) -> None:
        self.config = config or HypervectorConfig()
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
        self._projection_cache: Dict[str, torch.Tensor] = {}
        logger.info(
            "Initialized HypervectorEncoder(dim=%d, proj=%s)",
            self.config.dimension,
            self.config.projection_type.value,
        )

    def encode(
        self,
        features: Union[TensorLike, Mapping[str, TensorLike]],
        omics_type: OmicsType,
        *,
        resolution: str = "base",
    ) -> torch.Tensor:
        """Encode features into a single hypervector."""
        try:
            x = self._as_tensor(features)
            proj = self._get_projection_matrix(
                x.shape[-1], self.config.dimension, omics_type
            )
            hv = proj @ x.float()
            if self.config.normalize:
                hv = self._normalize(hv)
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

    def _as_tensor(
        self, features: Union[TensorLike, Mapping[str, TensorLike]]
    ) -> torch.Tensor:
        if isinstance(features, Mapping):
            # deterministic order
            arrs = [np.asarray(v) for k, v in sorted(features.items())]
            x = np.concatenate([a.ravel() for a in arrs]).astype(np.float32)
            return torch.from_numpy(x)
        if isinstance(features, np.ndarray):
            return torch.from_numpy(features.astype(np.float32).ravel())
        if isinstance(features, torch.Tensor):
            return features.view(-1).float()
        raise EncodingError(f"Unsupported feature type: {type(features)!r}")

    def _cache_key(self, input_dim: int, output_dim: int, omics_type: OmicsType) -> str:
        return f"{omics_type.value}:{input_dim}->{output_dim}:{self.config.projection_type.value}"

    def _get_projection_matrix(
        self, input_dim: int, output_dim: int, omics_type: OmicsType
    ) -> torch.Tensor:
        key = self._cache_key(input_dim, output_dim, omics_type)
        if key in self._projection_cache:
            return self._projection_cache[key]

        if self.config.projection_type == ProjectionType.RANDOM_GAUSSIAN:
            mat = torch.randn(output_dim, input_dim) / np.sqrt(input_dim)
        elif self.config.projection_type == ProjectionType.SPARSE_RANDOM:
            mat = self._sparse_random(
                output_dim, input_dim, sparsity=self.config.sparsity
            )
        elif self.config.projection_type == ProjectionType.ORTHOGONAL:
            mat = self._orthogonal(output_dim, input_dim)
        else:
            raise ProjectionError(
                f"Unsupported projection type {self.config.projection_type}"
            )

        self._projection_cache[key] = mat
        return mat

    def _sparse_random(self, rows: int, cols: int, *, sparsity: float) -> torch.Tensor:
        # Achlioptas-style: values in {-1, 0, +1}
        probs = [sparsity / 2, 1 - sparsity, sparsity / 2]
        vals = np.random.choice([-1.0, 0.0, 1.0], size=(rows, cols), p=probs).astype(
            np.float32
        )
        mat = torch.from_numpy(vals)
        # scale so that E[||x||] is preserved
        if sparsity > 0:
            mat = mat / np.sqrt(sparsity * cols)
        return mat

    def _orthogonal(self, rows: int, cols: int) -> torch.Tensor:
        # Build via QR on a Gaussian matrix and crop/pad
        a = torch.randn(max(rows, cols), max(rows, cols))
        q, _ = torch.linalg.qr(a)
        return q[:rows, :cols].contiguous()

    def _normalize(self, hv: torch.Tensor) -> torch.Tensor:
        n = torch.norm(hv, p=2).clamp_min(1e-12)
        return hv / n

    def _quantize(self, hv: torch.Tensor, *, bits: int = 8) -> torch.Tensor:
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
