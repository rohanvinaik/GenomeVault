"""Models module."""

from __future__ import annotations

from pydantic import BaseModel, Field, validator
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""

    # Core settings
    aggregation_method: str = "federated_avg"  # federated_avg, weighted_avg, median
    min_clients: int = 2  # Minimum number of clients for aggregation
    min_participants: int = 2  # Alternative name for min_clients
    rounds: int = 10  # Number of federated learning rounds

    # Protocol settings
    protocol: Optional[Any] = None  # AggregationProtocol enum (SECAGG, FEDAVG, etc.)
    dropout_tolerance: float = 0.3  # Fraction of clients that can drop out

    # Privacy settings
    differential_privacy: bool = False  # Enable differential privacy
    differential_privacy_enabled: bool = False  # Alternative name
    epsilon: float = 1.0  # DP epsilon parameter
    delta: float = 1e-5  # DP delta parameter
    clip_norm: Optional[float] = None  # L2 norm clipping threshold

    # Security settings
    secure_aggregation: bool = False  # Enable secure aggregation

    # Training settings
    learning_rate: float = 0.01  # Client learning rate
    batch_size: int = 32  # Client batch size
    local_epochs: int = 5  # Epochs per client per round
    validation_split: float = 0.2  # Validation data split
    patience: int = 5  # Early stopping patience

    # Optimization settings
    model_compression: bool = False  # Enable model compression
    compression_rate: float = 0.5  # Compression ratio


class ModelUpdate(BaseModel):
    """ModelUpdate implementation."""

    client_id: str = Field(..., description="Unique identifier for the client")
    weights: list[float] = Field(..., description="Flattened model weights")
    num_examples: int = Field(
        ..., ge=1, description="Number of examples used to compute the update"
    )
    signature: str | None = Field(
        None, description="Optional signature (not verified in this scaffold)"
    )

    @validator("weights")
    def _non_empty(cls, v: list[float]):
        """non empty.
        Args:        v: Parameter value."""
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("weights must be a non-empty list[float]")
        return v


class AggregateRequest(BaseModel):
    """AggregateRequest implementation."""

    updates: list[ModelUpdate] = Field(..., min_items=1)
    clip_norm: float | None = Field(
        None, ge=0.0, description="Optional L2 clip per-update before averaging"
    )


class AggregateResponse(BaseModel):
    """AggregateResponse implementation."""

    aggregated_weights: list[float]
    total_examples: int
    client_count: int
    details: dict[str, Any] = {}
