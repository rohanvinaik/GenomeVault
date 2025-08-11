from __future__ import annotations

"""Aggregator module."""
import numpy as np

from genomevault.core.exceptions import ValidationError
from genomevault.federated.models import (
    AggregateRequest,
    AggregateResponse,
    ModelUpdate,
)


def _l2_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def _clip_by_l2(x: np.ndarray, clip: float) -> np.ndarray:
    n = _l2_norm(x)
    if n <= clip or clip <= 0.0:
        return x
    return x * (clip / n)


class FedAvgAggregator:
    """Simple FedAvg with optional per-update L2 clipping.

    SECURITY NOTE: This is a minimal scaffold with *no* Byzantine-robustness, secure aggregation, or DP.
    Do not use in adversarial environments without hardening.
    """

    def __init__(self) -> None:
        """Initialize instance."""
        self._last_shape: int | None = None

    def _validate_and_prepare(
        self, updates: list[ModelUpdate], clip_norm: float | None
    ) -> tuple[list[np.ndarray], list[int]]:
        arrs: list[np.ndarray] = []
        counts: list[int] = []
        L: int | None = None
        for u in updates:
            w = np.asarray(u.weights, dtype=np.float64)
            if L is None:
                L = w.size
            if w.size != L:
                raise ValidationError(
                    "all weight vectors must have the same length",
                    context={"expected": L, "got": w.size, "client_id": u.client_id},
                )
            if clip_norm is not None and clip_norm > 0.0:
                w = _clip_by_l2(w, clip_norm)
            arrs.append(w)
            counts.append(int(u.num_examples))
        self._last_shape = int(L or 0)
        return arrs, counts

    def aggregate(self, req: AggregateRequest) -> AggregateResponse:
        """Aggregate.

        Args:
            req: Req.

        Returns:
            AggregateResponse instance.

        Raises:
            ValidationError: When operation fails.
        """
        arrs, counts = self._validate_and_prepare(req.updates, req.clip_norm)
        total_examples = int(sum(counts))
        if total_examples <= 0:
            raise ValidationError("total_examples must be positive")
        # Weighted average by num_examples
        numer = sum(w * n for w, n in zip(arrs, counts))
        denom = float(total_examples)
        agg = numer / denom
        return AggregateResponse(
            aggregated_weights=agg.tolist(),
            total_examples=total_examples,
            client_count=len(arrs),
            details={"clip_norm": req.clip_norm},
        )
