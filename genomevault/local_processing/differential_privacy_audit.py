"""
Differential Privacy Audit Trail for Model Training

This module implements privacy budget tracking and verification
for differentially private model training in GenomeVault.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class PrivacyMechanism(Enum):
    """Types of differential privacy mechanisms"""

    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    GRADIENT_CLIPPING = "gradient_clipping"
    FEDERATED_AVERAGING = "federated_averaging"


@dataclass
class PrivacyEvent:
    """Record of a single privacy-consuming event"""

    event_id: str
    timestamp: int
    mechanism: PrivacyMechanism
    epsilon_consumed: float
    delta_consumed: float
    sensitivity: float
    data_size: int
    operation: str
    metadata: dict[str, Any]


@dataclass
class PrivacyBudget:
    """Privacy budget allocation for a training session"""

    total_epsilon: float
    total_delta: float
    consumed_epsilon: float
    consumed_delta: float
    mechanism_allocations: dict[str, float]
    start_time: int
    end_time: int | None


class DifferentialPrivacyAuditor:
    """
    Tracks and verifies differential privacy budget consumption during model training.

    This auditor ensures:
    1. Privacy budget is not exceeded
    2. All privacy-consuming operations are logged
    3. Composition theorems are correctly applied
    4. Audit trail is cryptographically verifiable
    """

    def __init__(self, session_id: str, total_epsilon: float, total_delta: float = 1e-5):
        """
        Initialize privacy auditor for a training session.

        Args:
            session_id: Unique identifier for this training session
            total_epsilon: Total privacy budget (epsilon)
            total_delta: Total privacy budget (delta)
        """
        self.session_id = session_id
        self.privacy_events: list[PrivacyEvent] = []
        self.event_hashes: list[str] = []

        # Initialize budget
        self.budget = PrivacyBudget(
            total_epsilon=total_epsilon,
            total_delta=total_delta,
            consumed_epsilon=0.0,
            consumed_delta=0.0,
            mechanism_allocations={mechanism.value: 0.0 for mechanism in PrivacyMechanism},
            start_time=int(time.time()),
            end_time=None,
        )

        # Composition parameters
        self.use_advanced_composition = True
        self.composition_slack = 1e-5

        logger.info(
            "Privacy auditor initialized for session %ssession_id "
            "with budget ε=%stotal_epsilon, δ=%stotal_delta"
        )

    def log_privacy_event(
        self,
        mechanism: PrivacyMechanism,
        epsilon: float,
        delta: float,
        sensitivity: float,
        data_size: int,
        operation: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        """
        Log a privacy-consuming event and check budget.

        Args:
            mechanism: Type of privacy mechanism used
            epsilon: Privacy loss (epsilon) for this operation
            delta: Privacy loss (delta) for this operation
            sensitivity: Sensitivity of the operation
            data_size: Number of data points involved
            operation: Description of the operation
            metadata: Additional metadata

        Returns:
            Tuple of (event_id, budget_ok)
        """
        # Check if budget would be exceeded
        new_epsilon, new_delta = self._compose_privacy_loss(epsilon, delta)

        if new_epsilon > self.budget.total_epsilon:
            logger.error(
                "Privacy budget would be exceeded: "
                "%snew_epsilon:.4f > %sself.budget.total_epsilon:.4f"
            )
            return "", False

        if new_delta > self.budget.total_delta:
            logger.error(
                "Delta budget would be exceeded: %snew_delta:.6f > %sself.budget.total_delta:.6f"
            )
            return "", False

        # Create event
        event_data = f"{self.session_id}{mechanism.value}{epsilon}{delta}{operation}{time.time()}"
        event_id = hashlib.sha256(event_data.encode()).hexdigest()[:16]

        event = PrivacyEvent(
            event_id=event_id,
            timestamp=int(time.time()),
            mechanism=mechanism,
            epsilon_consumed=epsilon,
            delta_consumed=delta,
            sensitivity=sensitivity,
            data_size=data_size,
            operation=operation,
            metadata=metadata or {},
        )

        # Update budget
        self.budget.consumed_epsilon = new_epsilon
        self.budget.consumed_delta = new_delta
        self.budget.mechanism_allocations[mechanism.value] += epsilon

        # Store event
        self.privacy_events.append(event)

        # Create hash chain
        previous_hash = self.event_hashes[-1] if self.event_hashes else "0"
        event_hash = self._hash_event(event, previous_hash)
        self.event_hashes.append(event_hash)

        logger.info(
            "Privacy event %sevent_id: %soperation consumed "
            "ε=%sepsilon:.4f, δ=%sdelta:.6f "
            "(total: ε=%sself.budget.consumed_epsilon:.4f, δ=%sself.budget.consumed_delta:.6f)"
        )

        return event_id, True

    def verify_gradient_clipping(
        self, gradients: np.ndarray, clip_norm: float, noise_scale: float
    ) -> tuple[float, float]:
        """
        Verify gradient clipping and compute privacy loss.

        Args:
            gradients: Gradient values
            clip_norm: L2 norm threshold for clipping
            noise_scale: Scale of noise to be added

        Returns:
            Tuple of (epsilon, delta) consumed
        """
        # Compute actual gradient norm
        actual_norm = np.linalg.norm(gradients)

        # Verify clipping was applied
        if actual_norm > clip_norm * 1.01:  # Allow 1% tolerance
            logger.warning("Gradient norm %sactual_norm:.4f exceeds clip threshold %sclip_norm:.4f")

        # Compute sensitivity
        sensitivity = 2 * clip_norm  # For gradient clipping

        # Compute privacy loss (simplified - in practice use tighter bounds)
        epsilon = sensitivity / noise_scale
        delta = 1e-7  # Small delta for Gaussian mechanism

        # Log the event
        event_id, success = self.log_privacy_event(
            mechanism=PrivacyMechanism.GRADIENT_CLIPPING,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            data_size=gradients.shape[0],
            operation="gradient_clipping",
            metadata={
                "clip_norm": clip_norm,
                "actual_norm": float(actual_norm),
                "noise_scale": noise_scale,
            },
        )

        return epsilon, delta

    def verify_noise_addition(
        self,
        mechanism: PrivacyMechanism,
        sensitivity: float,
        noise_params: dict[str, float],
        data_size: int,
    ) -> tuple[float, float]:
        """
        Verify noise addition and compute privacy loss.

        Args:
            mechanism: Type of noise mechanism
            sensitivity: Query sensitivity
            noise_params: Parameters for noise distribution
            data_size: Size of data

        Returns:
            Tuple of (epsilon, delta) consumed
        """
        if mechanism == PrivacyMechanism.LAPLACE:
            scale = noise_params.get("scale", 1.0)
            epsilon = sensitivity / scale
            delta = 0.0

        elif mechanism == PrivacyMechanism.GAUSSIAN:
            sigma = noise_params.get("sigma", 1.0)
            delta_target = noise_params.get("delta", 1e-5)

            # Compute epsilon for Gaussian mechanism
            epsilon = sensitivity * np.sqrt(2 * np.log(1.25 / delta_target)) / sigma
            delta = delta_target

        else:
            raise ValueError(f"Unsupported noise mechanism: {mechanism}")

        # Log the event
        event_id, success = self.log_privacy_event(
            mechanism=mechanism,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            data_size=data_size,
            operation=f"{mechanism.value}_noise",
            metadata=noise_params,
        )

        return epsilon, delta

    def _compose_privacy_loss(self, new_epsilon: float, new_delta: float) -> tuple[float, float]:
        """
        Compose privacy loss using appropriate composition theorem.

        Args:
            new_epsilon: New epsilon to add
            new_delta: New delta to add

        Returns:
            Total (epsilon, delta) after composition
        """
        if self.use_advanced_composition and len(self.privacy_events) > 1:
            # Use advanced composition for tighter bounds
            k = len(self.privacy_events) + 1

            # Compute sum of squared epsilons
            eps_squared_sum = sum(e.epsilon_consumed**2 for e in self.privacy_events)
            eps_squared_sum += new_epsilon**2

            # Advanced composition bound
            total_epsilon = np.sqrt(2 * eps_squared_sum * np.log(1 / self.composition_slack))
            total_epsilon += sum(e.epsilon_consumed for e in self.privacy_events) + new_epsilon
            total_epsilon = min(
                total_epsilon, self.budget.consumed_epsilon + new_epsilon
            )  # Basic composition

            total_delta = self.budget.consumed_delta + new_delta + k * self.composition_slack

        else:
            # Basic composition
            total_epsilon = self.budget.consumed_epsilon + new_epsilon
            total_delta = self.budget.consumed_delta + new_delta

        return total_epsilon, total_delta

    def get_remaining_budget(self) -> tuple[float, float]:
        """Get remaining privacy budget"""
        remaining_epsilon = self.budget.total_epsilon - self.budget.consumed_epsilon
        remaining_delta = self.budget.total_delta - self.budget.consumed_delta

        return max(0, remaining_epsilon), max(0, remaining_delta)

    def get_mechanism_breakdown(self) -> dict[str, float]:
        """Get privacy consumption breakdown by mechanism"""
        return self.budget.mechanism_allocations.copy()

    def finalize_session(self) -> dict[str, Any]:
        """
        Finalize the privacy audit session and generate report.

        Returns:
            Final audit report
        """
        self.budget.end_time = int(time.time())

        # Compute final hash
        final_hash = self._compute_audit_hash()

        # Generate report
        report = {
            "session_id": self.session_id,
            "start_time": self.budget.start_time,
            "end_time": self.budget.end_time,
            "duration_seconds": self.budget.end_time - self.budget.start_time,
            "privacy_budget": {
                "total_epsilon": self.budget.total_epsilon,
                "total_delta": self.budget.total_delta,
                "consumed_epsilon": self.budget.consumed_epsilon,
                "consumed_delta": self.budget.consumed_delta,
                "utilization_epsilon": self.budget.consumed_epsilon / self.budget.total_epsilon,
                "utilization_delta": self.budget.consumed_delta / self.budget.total_delta,
            },
            "mechanism_breakdown": self.budget.mechanism_allocations,
            "total_events": len(self.privacy_events),
            "audit_hash": final_hash,
            "verification_chain": self.event_hashes[-10:],  # Last 10 hashes
            "summary": {
                "avg_epsilon_per_event": (
                    self.budget.consumed_epsilon / len(self.privacy_events)
                    if self.privacy_events
                    else 0
                ),
                "max_single_epsilon": (
                    max(e.epsilon_consumed for e in self.privacy_events)
                    if self.privacy_events
                    else 0
                ),
                "total_data_processed": sum(e.data_size for e in self.privacy_events),
            },
        }

        logger.info(
            "Privacy audit session %sself.session_id finalized: "
            "ε=%sself.budget.consumed_epsilon:.4f/%sself.budget.total_epsilon:.4f, "
            "δ=%sself.budget.consumed_delta:.6f/%sself.budget.total_delta:.6f"
        )

        return report

    def export_audit_trail(self, include_metadata: bool = False) -> list[dict[str, Any]]:
        """
        Export the complete audit trail.

        Args:
            include_metadata: Whether to include event metadata

        Returns:
            List of audit events
        """
        trail = []

        for i, event in enumerate(self.privacy_events):
            event_dict = {
                "index": i,
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "mechanism": event.mechanism.value,
                "epsilon": event.epsilon_consumed,
                "delta": event.delta_consumed,
                "sensitivity": event.sensitivity,
                "data_size": event.data_size,
                "operation": event.operation,
                "event_hash": self.event_hashes[i],
            }

            if include_metadata:
                event_dict["metadata"] = event.metadata

            trail.append(event_dict)

        return trail

    def verify_audit_trail(self) -> bool:
        """
        Verify the integrity of the audit trail.

        Returns:
            True if audit trail is valid
        """
        if not self.privacy_events:
            return True

        # Verify hash chain
        previous_hash = "0"

        for i, event in enumerate(self.privacy_events):
            expected_hash = self._hash_event(event, previous_hash)

            if expected_hash != self.event_hashes[i]:
                logger.error("Hash mismatch at event %si: %sevent.event_id")
                return False

            previous_hash = expected_hash

        # Verify budget calculations
        recalc_epsilon = 0.0
        recalc_delta = 0.0

        for event in self.privacy_events:
            recalc_epsilon += event.epsilon_consumed
            recalc_delta += event.delta_consumed

        # Allow small numerical error
        epsilon_error = abs(recalc_epsilon - self.budget.consumed_epsilon)
        delta_error = abs(recalc_delta - self.budget.consumed_delta)

        if epsilon_error > 1e-6 or delta_error > 1e-9:
            logger.error(
                "Budget calculation mismatch: ε error=%sepsilon_error, δ error=%sdelta_error"
            )
            return False

        logger.info("Audit trail verification successful")
        return True

    def _hash_event(self, event: PrivacyEvent, previous_hash: str) -> str:
        """Create hash of privacy event"""
        event_data = {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "mechanism": event.mechanism.value,
            "epsilon": event.epsilon_consumed,
            "delta": event.delta_consumed,
            "operation": event.operation,
            "previous_hash": previous_hash,
        }

        event_str = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()

    def _compute_audit_hash(self) -> str:
        """Compute final hash of entire audit trail"""
        audit_data = {
            "session_id": self.session_id,
            "total_events": len(self.privacy_events),
            "final_epsilon": self.budget.consumed_epsilon,
            "final_delta": self.budget.consumed_delta,
            "event_hashes": self.event_hashes,
        }

        audit_str = json.dumps(audit_data, sort_keys=True)
        return hashlib.sha256(audit_str.encode()).hexdigest()


class PrivacyAccountant:
    """
    Global privacy accountant for managing multiple training sessions.
    """

    def __init__(self):
        self.sessions: dict[str, DifferentialPrivacyAuditor] = {}
        self.global_budget = {
            "daily_epsilon": 10.0,
            "weekly_epsilon": 50.0,
            "monthly_epsilon": 150.0,
        }
        self.consumption_history: list[dict[str, Any]] = []

    def create_session(
        self, session_id: str, epsilon_budget: float, delta_budget: float = 1e-5
    ) -> DifferentialPrivacyAuditor:
        """Create a new privacy audit session"""
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")

        # Check global budget constraints
        daily_consumed = self._get_period_consumption("daily")
        if daily_consumed + epsilon_budget > self.global_budget["daily_epsilon"]:
            raise ValueError(
                f"Daily epsilon budget would be exceeded: "
                f"{daily_consumed + epsilon_budget:.2f} > {self.global_budget['daily_epsilon']}"
            )

        auditor = DifferentialPrivacyAuditor(session_id, epsilon_budget, delta_budget)
        self.sessions[session_id] = auditor

        return auditor

    def get_session(self, session_id: str) -> DifferentialPrivacyAuditor | None:
        """Get an existing session"""
        return self.sessions.get(session_id)

    def finalize_session(self, session_id: str) -> dict[str, Any]:
        """Finalize a session and update global records"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        auditor = self.sessions[session_id]
        report = auditor.finalize_session()

        # Update consumption history
        self.consumption_history.append(
            {
                "session_id": session_id,
                "timestamp": report["end_time"],
                "epsilon_consumed": report["privacy_budget"]["consumed_epsilon"],
                "delta_consumed": report["privacy_budget"]["consumed_delta"],
            }
        )

        return report

    def _get_period_consumption(self, period: str) -> float:
        """Get epsilon consumption for a time period"""
        current_time = time.time()

        if period == "daily":
            cutoff = current_time - 86400  # 24 hours
        elif period == "weekly":
            cutoff = current_time - 604800  # 7 days
        elif period == "monthly":
            cutoff = current_time - 2592000  # 30 days
        else:
            raise ValueError(f"Unknown period: {period}")

        return sum(
            record["epsilon_consumed"]
            for record in self.consumption_history
            if record["timestamp"] > cutoff
        )
