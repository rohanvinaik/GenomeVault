from __future__ import annotations

from uuid import uuid4

import numpy as np

from genomevault.core.constants import (
    DEFAULT_DENSITY,
    DEFAULT_SEED,
    HYPERVECTOR_DIMENSIONS,
)
from genomevault.core.exceptions import EncodingError, ProjectionError, ValidationError
from genomevault.hypervector.encoding.sparse_projection import SparseRandomProjection
from genomevault.hypervector.operations.binding import bundle as hv_bundle
from genomevault.hypervector.operations.binding import (
    circular_convolution as hv_convolve,
)
from genomevault.hypervector.operations.binding import (
    element_wise_multiply as hv_multiply,
)
from genomevault.hypervector.operations.binding import permutation_binding as hv_permute
from genomevault.hypervector.stores.in_memory import InMemoryStore


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


class HypervectorEngine:
    """Minimal engine for encoding, operating, and comparing hypervectors."""

    def __init__(self, store: InMemoryStore | None = None) -> None:
        self.store = store or InMemoryStore()

    # ---- storage helpers ----
    def _put(self, v: np.ndarray) -> str:
        vid = str(uuid4())
        self.store.put(vid, v)
        return vid

    def _get(self, vid: str) -> np.ndarray:
        v = self.store.get(vid)
        if v is None:
            raise ValidationError(f"vector_id not found: {vid}", context={"vector_id": vid})
        return v

    # ---- public API ----
    def encode(
        self,
        *,
        data: dict[str, list[float]],
        dimension: int,
        compression_tier: str = "full",
    ) -> dict:
        allowed_dims = set(HYPERVECTOR_DIMENSIONS.values())
        if int(dimension) not in allowed_dims:
            raise ProjectionError(
                "unsupported dimension",
                context={"dimension": dimension, "allowed": sorted(allowed_dims)},
            )
        if not isinstance(data, dict) or not data:
            raise EncodingError("data must be a non-empty dict[str, list[float]]")

        hv_list: list[np.ndarray] = []
        for modality, arr in data.items():
            try:
                x = np.asarray(arr, dtype=np.float64).reshape(1, -1)  # (1, n_features)
            except Exception as e:
                from genomevault.observability.logging import configure_logging

                logger = configure_logging()
                logger.exception("Unhandled exception")
                raise EncodingError(
                    "failed to parse input array", context={"modality": modality}
                ) from e
                raise RuntimeError("Unspecified error")
            n_features = int(x.shape[1])
            proj = SparseRandomProjection(
                n_components=int(dimension), density=DEFAULT_DENSITY, seed=DEFAULT_SEED
            )
            proj.fit(n_features=n_features)
            y = proj.transform(x)  # (1, D)
            hv_list.append(y[0])

        if len(hv_list) == 1:
            hv = hv_list[0]
        else:
            hv = hv_bundle(hv_list)

        sparsity = float(np.mean(hv == 0.0))
        vid = self._put(hv)
        return {
            "vector_id": vid,
            "dimension": int(dimension),
            "sparsity": sparsity,
            "compression_tier": str(compression_tier),
        }

    def operate(
        self, *, operation: str, vector_ids: list[str], parameters: dict | None = None
    ) -> dict:
        operation = (operation or "").lower()
        parameters = parameters or {}
        if operation not in {"permute", "bundle", "bind", "multiply", "convolve"}:
            raise ValidationError("unsupported operation", context={"operation": operation})

        if operation == "permute":
            if len(vector_ids) != 1:
                raise ValidationError("permute requires 1 vector id")
            shift = int(parameters.get("shift", 1))
            v = self._get(vector_ids[0])
            out = hv_permute(v, shift=shift)
            out_id = self._put(out)
            return {
                "result_vector_id": out_id,
                "operation": operation,
                "metadata": {"shift": shift},
            }

        if operation in ("bind", "multiply"):
            if len(vector_ids) != 2:
                raise ValidationError("bind/multiply requires 2 vector ids")
            a = self._get(vector_ids[0])
            b = self._get(vector_ids[1])
            out = hv_multiply(a, b)
            out_id = self._put(out)
            return {"result_vector_id": out_id, "operation": operation}

        if operation == "convolve":
            if len(vector_ids) != 2:
                raise ValidationError("convolve requires 2 vector ids")
            a = self._get(vector_ids[0])
            b = self._get(vector_ids[1])
            out = hv_convolve(a, b)
            out_id = self._put(out)
            return {"result_vector_id": out_id, "operation": operation}

        if operation == "bundle":
            if len(vector_ids) < 2:
                raise ValidationError("bundle requires >= 2 vector ids")
            vecs = [self._get(vid) for vid in vector_ids]
            out = hv_bundle(vecs)
            out_id = self._put(out)
            return {"result_vector_id": out_id, "operation": operation}

        raise ValidationError("unhandled operation", context={"operation": operation})

    def calculate_similarity(self, vector_id1: str, vector_id2: str) -> float:
        a = self._get(vector_id1)
        b = self._get(vector_id2)
        return _cosine(a, b)
