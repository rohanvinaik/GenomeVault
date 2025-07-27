# genomevault/federation/aggregator.py
"""Federated aggregation with differential privacy."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
from scipy import stats


class AggregationMethod(Enum):
    """Aggregation methods for federated learning."""

    SECURE_MEAN = auto()
    TRIMMED_MEAN = auto()
    KRUM = auto()
    MEDIAN = auto()
    BYZANTINE_ROBUST = auto()


@dataclass
class ParticipantProfile:
    """Profile of federation participant."""

    participant_id: str
    institution: str
    data_size: int
    join_timestamp: float
    reputation_score: float = 1.0
    contributions: int = 0
    quality_scores: List[float] = field(default_factory=list)

    @property
    def average_quality(self) -> float:
        """Average quality score."""
        return np.mean(self.quality_scores) if self.quality_scores else 0.0


@dataclass
class DPConfig:
    """Differential privacy configuration."""

    epsilon: float = 1.0
    delta: float = 1e-5
    sensitivity: float = 1.0
    mechanism: str = "gaussian"  # 'gaussian' or 'laplace'

    def compute_noise_scale(self) -> float:
        """Compute noise scale for privacy mechanism."""
        if self.mechanism == "gaussian":
            return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        else:  # laplace
            return self.sensitivity / self.epsilon


class UpdateValidator(Protocol):
    """Protocol for update validation."""

    def validate(self, update: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Validate an update."""
        ...


class SecureAggregator:
    """Secure aggregation with robustness and privacy."""

    def __init__(
        self,
        aggregation_method: AggregationMethod = AggregationMethod.TRIMMED_MEAN,
        dp_config: Optional[DPConfig] = None,
        min_participants: int = 3,
        reputation_threshold: float = 0.7,
    ):
        self.method = aggregation_method
        self.dp_config = dp_config or DPConfig()
        self.min_participants = min_participants
        self.reputation_threshold = reputation_threshold

        # Participant tracking
        self.participants: Dict[str, ParticipantProfile] = {}

        # Aggregation state
        self.current_round = 0
        self.round_history: List[Dict[str, Any]] = []

    def register_participant(self, profile: ParticipantProfile) -> bool:
        """Register new participant."""
        if profile.participant_id in self.participants:
            return False

        self.participants[profile.participant_id] = profile
        return True

    async def aggregate_updates(
        self, updates: Dict[str, np.ndarray], metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Aggregate updates with privacy and robustness."""
        # Filter by reputation
        valid_updates = self._filter_by_reputation(updates)

        if len(valid_updates) < self.min_participants:
            raise ValueError(
                f"Insufficient participants: {len(valid_updates)} < {self.min_participants}"
            )

        # Validate updates
        if metadata:
            valid_updates = self._validate_updates(valid_updates, metadata)

        # Apply aggregation method
        if self.method == AggregationMethod.SECURE_MEAN:
            aggregated = self._secure_mean(valid_updates)
        elif self.method == AggregationMethod.TRIMMED_MEAN:
            aggregated = self._trimmed_mean(valid_updates)
        elif self.method == AggregationMethod.KRUM:
            aggregated = self._krum_aggregation(valid_updates)
        elif self.method == AggregationMethod.MEDIAN:
            aggregated = self._median_aggregation(valid_updates)
        elif self.method == AggregationMethod.BYZANTINE_ROBUST:
            aggregated = self._byzantine_robust_aggregation(valid_updates)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")

        # Add differential privacy
        if self.dp_config:
            aggregated = self._add_differential_privacy(aggregated)

        # Update participant profiles
        self._update_participant_profiles(valid_updates, aggregated)

        # Record round
        round_info = {
            "round": self.current_round,
            "method": self.method.name,
            "participants": list(valid_updates.keys()),
            "dp_applied": self.dp_config is not None,
            "timestamp": time.time(),
        }
        self.round_history.append(round_info)
        self.current_round += 1

        return aggregated, round_info

    def _filter_by_reputation(self, updates: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Filter updates by participant reputation."""
        valid = {}

        for pid, update in updates.items():
            if pid in self.participants:
                profile = self.participants[pid]
                if profile.reputation_score >= self.reputation_threshold:
                    valid[pid] = update

        return valid

    def _validate_updates(
        self, updates: Dict[str, np.ndarray], metadata: Dict[str, Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Validate updates based on metadata."""
        valid = {}

        for pid, update in updates.items():
            # meta = metadata.get(pid, {})

            # Basic validation
            if np.any(np.isnan(update)) or np.any(np.isinf(update)):
                continue

            # Check update norm
            norm = np.linalg.norm(update)
            if norm > 100:  # Suspiciously large
                continue

            valid[pid] = update

        return valid

    def _secure_mean(self, updates: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute secure mean."""
        values = list(updates.values())
        return np.mean(values, axis=0)

    def _trimmed_mean(self, updates: Dict[str, np.ndarray], trim_pct: float = 0.2) -> np.ndarray:
        """Compute trimmed mean."""
        values = np.array(list(updates.values()))
        n_trim = int(len(values) * trim_pct)

        if n_trim == 0:
            return np.mean(values, axis=0)

        # Trim along participant axis
        trimmed = stats.trim_mean(values, trim_pct, axis=0)
        return trimmed

    def _krum_aggregation(
        self, updates: Dict[str, np.ndarray], n_byzantine: Optional[int] = None
    ) -> np.ndarray:
        """Krum aggregation for Byzantine robustness."""
        pids = list(updates.keys())
        values = np.array([updates[pid] for pid in pids])
        n = len(values)

        if n_byzantine is None:
            n_byzantine = int(n * 0.2)  # Assume 20% Byzantine

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(values[i] - values[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute Krum scores
        scores = []
        k = n - n_byzantine - 2

        for i in range(n):
            # Get k nearest neighbors
            dists = distances[i]
            nearest_indices = np.argpartition(dists, k)[:k]
            score = np.sum(dists[nearest_indices])
            scores.append(score)

        # Select update with minimum score
        best_idx = np.argmin(scores)
        return values[best_idx]

    def _median_aggregation(self, updates: Dict[str, np.ndarray]) -> np.ndarray:
        """Coordinate-wise median aggregation."""
        values = np.array(list(updates.values()))
        return np.median(values, axis=0)

    def _byzantine_robust_aggregation(self, updates: Dict[str, np.ndarray]) -> np.ndarray:
        """Byzantine-robust aggregation using multiple defenses."""
        # First apply Krum to identify good subset
        krum_result = self._krum_aggregation(updates)

        # Then apply trimmed mean on updates close to Krum result
        values = list(updates.values())
        distances = [np.linalg.norm(v - krum_result) for v in values]
        threshold = np.percentile(distances, 75)

        good_updates = {
            pid: update
            for pid, update in updates.items()
            if np.linalg.norm(update - krum_result) <= threshold
        }

        # Apply trimmed mean on good subset
        return self._trimmed_mean(good_updates, trim_pct=0.1)

    def _add_differential_privacy(self, aggregated: np.ndarray) -> np.ndarray:
        """Add calibrated noise for differential privacy."""
        noise_scale = self.dp_config.compute_noise_scale()

        if self.dp_config.mechanism == "gaussian":
            noise = np.random.normal(0, noise_scale, aggregated.shape)
        else:  # laplace
            noise = np.random.laplace(0, noise_scale, aggregated.shape)

        return aggregated + noise

    def _update_participant_profiles(
        self, updates: Dict[str, np.ndarray], aggregated: np.ndarray
    ) -> None:
        """Update participant profiles based on contribution quality."""
        # Compute quality scores
        for pid, update in updates.items():
            if pid not in self.participants:
                continue

            profile = self.participants[pid]

            # Quality based on distance from aggregated
            distance = np.linalg.norm(update - aggregated)
            quality = 1.0 / (1.0 + distance)  # Higher quality for closer updates

            profile.quality_scores.append(quality)
            profile.contributions += 1

            # Update reputation (exponential moving average)
            alpha = 0.1
            profile.reputation_score = (1 - alpha) * profile.reputation_score + alpha * quality

    def get_participant_stats(self) -> Dict[str, Any]:
        """Get statistics about participants."""
        active = sum(
            1 for p in self.participants.values() if p.reputation_score >= self.reputation_threshold
        )

        return {
            "total_participants": len(self.participants),
            "active_participants": active,
            "average_reputation": np.mean([p.reputation_score for p in self.participants.values()]),
            "total_contributions": sum(p.contributions for p in self.participants.values()),
            "reputation_distribution": {
                "excellent": sum(
                    1 for p in self.participants.values() if p.reputation_score >= 0.9
                ),
                "good": sum(
                    1 for p in self.participants.values() if 0.7 <= p.reputation_score < 0.9
                ),
                "fair": sum(
                    1 for p in self.participants.values() if 0.5 <= p.reputation_score < 0.7
                ),
                "poor": sum(1 for p in self.participants.values() if p.reputation_score < 0.5),
            },
        }
