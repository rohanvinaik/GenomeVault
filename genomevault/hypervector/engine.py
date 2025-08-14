from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Union
import hashlib
import numpy as np

Numeric = Union[int, float, np.number]
VectorLike = Union[Sequence[Numeric], np.ndarray]


@dataclass
class HypervectorConfig:
    dim: int = 8192
    seed: int = 0
    normalize: bool = True
    binary: bool = False  # if True, return {0,1} vector; else float32


class HypervectorEngine:
    """
    Minimal, safe hypervector encoder.
    Accepts numeric features directly OR higher-level tokens already featurized.
    """

    def __init__(self, config: HypervectorConfig | None = None) -> None:
        self.config = config or HypervectorConfig()
        self._rng = np.random.default_rng(self.config.seed)
        # pre-generate a projection basis (fast & deterministic)
        self._basis = self._rng.standard_normal((256, self.config.dim), dtype=np.float32)

    def _hash_to_index(self, token: str) -> int:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=2).digest()
        return int.from_bytes(h, "little") % self._basis.shape[0]

    def encode_numeric(self, features: VectorLike) -> np.ndarray:
        """
        Projects numeric feature vector to a hypervector.
        """
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("encode_numeric expects a 1D vector")
        # simple dense projection: chunk into bytes → hash → pick basis rows
        hv = np.zeros(self.config.dim, dtype=np.float32)
        # bucket numeric features into 0..255 for stable indexing
        if len(x) == 0:
            return hv
        scaled = np.clip(((x - x.mean()) / (x.std() + 1e-6) * 32 + 128), 0, 255).astype(np.uint8)
        counts = np.bincount(scaled, minlength=256)
        # weighted sum of basis rows
        hv = counts @ self._basis
        if self.config.normalize:
            n = np.linalg.norm(hv) or 1.0
            hv = hv / n
        if self.config.binary:
            hv = (hv > 0).astype(np.uint8)
        return hv

    def encode_tokens(self, tokens: Iterable[str]) -> np.ndarray:
        """
        Token-based encoding (e.g., k-mers, categorical hashes), no raw genome exposure.
        """
        hv = np.zeros(self.config.dim, dtype=np.float32)
        for t in tokens:
            idx = self._hash_to_index(str(t))
            hv += self._basis[idx]
        if self.config.normalize:
            n = np.linalg.norm(hv) or 1.0
            hv = hv / n
        if self.config.binary:
            hv = (hv > 0).astype(np.uint8)
        return hv

    def encode(self, x: Union[VectorLike, Iterable[str]]) -> np.ndarray:
        """
        Dispatch: numeric vectors → encode_numeric, else treat as tokens.
        """
        # Heuristic: if the first element is numeric, treat as numeric vector
        if isinstance(x, np.ndarray) or (
            isinstance(x, (list, tuple)) and x and isinstance(x[0], (int, float, np.number))
        ):
            return self.encode_numeric(x)  # type: ignore[arg-type]
        return self.encode_tokens(x)  # type: ignore[arg-type]
