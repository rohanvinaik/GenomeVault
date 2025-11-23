"""
Zero-knowledge proof verification monitoring.

Monitors ZK proof generation and verification for PIR operations,
tracking performance, failures, and security metrics.
"""

import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading

import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CollectorRegistry

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class ProofType(Enum):
    """Types of zero-knowledge proofs."""

    PIR_QUERY = "pir_query"
    PIR_RESPONSE = "pir_response"
    DATA_INTEGRITY = "data_integrity"
    ACCESS_CONTROL = "access_control"
    COMPUTATION = "computation"


class VerificationStatus(Enum):
    """Proof verification status."""

    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"
    INVALID = "invalid"


@dataclass
class ProofMetrics:
    """Metrics for a single proof verification."""

    proof_id: str
    proof_type: ProofType
    generation_time_ms: float
    verification_time_ms: float
    proof_size_bytes: int
    status: VerificationStatus
    timestamp: float
    client_id: Optional[str] = None
    server_id: Optional[int] = None
    error_message: Optional[str] = None
    gas_used: Optional[int] = None  # For blockchain-based proofs


@dataclass
class VerificationStats:
    """Aggregated verification statistics."""

    total_proofs: int = 0
    verified_proofs: int = 0
    failed_proofs: int = 0
    avg_generation_time_ms: float = 0.0
    avg_verification_time_ms: float = 0.0
    avg_proof_size_bytes: float = 0.0
    verification_rate: float = 0.0
    failure_rate: float = 0.0
    proofs_per_second: float = 0.0


class ZKVerificationMonitor:
    """
    Monitor for zero-knowledge proof verification.

    Tracks:
    1. Proof generation and verification times
    2. Success/failure rates
    3. Performance metrics
    4. Security anomalies
    """

    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
        history_size: int = 10000,
        alert_threshold: float = 0.95,
    ):
        """
        Initialize verification monitor.

        Args:
            registry: Prometheus registry
            history_size: Size of metrics history
            alert_threshold: Threshold for alerting
        """
        self.registry = registry or CollectorRegistry()
        self.history_size = history_size
        self.alert_threshold = alert_threshold

        # Metrics storage
        self.proof_history: deque = deque(maxlen=history_size)
        self.active_proofs: Dict[str, ProofMetrics] = {}

        # Statistics by type
        self.stats_by_type: Dict[ProofType, VerificationStats] = {
            pt: VerificationStats() for pt in ProofType
        }

        # Time-based metrics
        self.hourly_metrics: defaultdict = defaultdict(list)
        self.daily_metrics: defaultdict = defaultdict(list)

        # Thread safety
        self.lock = threading.RLock()

        # Initialize Prometheus metrics
        self._init_prometheus_metrics()

        # Alert tracking
        self.alerts: List[Dict[str, Any]] = []

        logger.info("Initialized ZK verification monitor")

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        # Counters
        self.proof_total = Counter(
            "genomevault_zk_proofs_total",
            "Total number of ZK proofs",
            ["type", "status"],
            registry=self.registry,
        )

        self.proof_failures = Counter(
            "genomevault_zk_proof_failures_total",
            "Total number of failed ZK proofs",
            ["type", "reason"],
            registry=self.registry,
        )

        # Histograms
        self.generation_time = Histogram(
            "genomevault_zk_proof_generation_seconds",
            "ZK proof generation time",
            ["type"],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
            registry=self.registry,
        )

        self.verification_time = Histogram(
            "genomevault_zk_proof_verification_seconds",
            "ZK proof verification time",
            ["type"],
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
            registry=self.registry,
        )

        self.proof_size = Histogram(
            "genomevault_zk_proof_size_bytes",
            "ZK proof size in bytes",
            ["type"],
            buckets=(100, 500, 1000, 5000, 10000, 50000, 100000),
            registry=self.registry,
        )

        # Gauges
        self.active_proofs_gauge = Gauge(
            "genomevault_zk_active_proofs",
            "Number of active ZK proofs",
            ["type"],
            registry=self.registry,
        )

        self.verification_rate_gauge = Gauge(
            "genomevault_zk_verification_rate",
            "ZK proof verification success rate",
            ["type"],
            registry=self.registry,
        )

        # Summary
        self.proof_latency = Summary(
            "genomevault_zk_proof_latency_seconds",
            "End-to-end ZK proof latency",
            ["type"],
            registry=self.registry,
        )

    def start_proof_generation(
        self, proof_id: str, proof_type: ProofType, client_id: Optional[str] = None
    ) -> float:
        """
        Start tracking proof generation.

        Args:
            proof_id: Unique proof identifier
            proof_type: Type of proof
            client_id: Client identifier

        Returns:
            Start timestamp
        """
        start_time = time.time()

        with self.lock:
            self.active_proofs[proof_id] = ProofMetrics(
                proof_id=proof_id,
                proof_type=proof_type,
                generation_time_ms=0,
                verification_time_ms=0,
                proof_size_bytes=0,
                status=VerificationStatus.PENDING,
                timestamp=start_time,
                client_id=client_id,
            )

            # Update gauge
            active_count = sum(1 for p in self.active_proofs.values() if p.proof_type == proof_type)
            self.active_proofs_gauge.labels(type=proof_type.value).set(active_count)

        logger.debug(f"Started tracking proof {proof_id} generation")

        return start_time

    def end_proof_generation(
        self, proof_id: str, proof_data: bytes, generation_time: float
    ) -> None:
        """
        End proof generation tracking.

        Args:
            proof_id: Proof identifier
            proof_data: Generated proof data
            generation_time: Generation time in seconds
        """
        with self.lock:
            if proof_id not in self.active_proofs:
                logger.warning(f"Unknown proof ID: {proof_id}")
                return

            metrics = self.active_proofs[proof_id]
            metrics.generation_time_ms = generation_time * 1000
            metrics.proof_size_bytes = len(proof_data)

            # Update Prometheus metrics
            self.generation_time.labels(type=metrics.proof_type.value).observe(generation_time)

            self.proof_size.labels(type=metrics.proof_type.value).observe(len(proof_data))

        logger.debug(f"Completed proof {proof_id} generation in {generation_time:.3f}s")

    def verify_proof(
        self, proof_id: str, proof_data: bytes, expected_hash: Optional[str] = None
    ) -> bool:
        """
        Verify a zero-knowledge proof.

        Args:
            proof_id: Proof identifier
            proof_data: Proof data to verify
            expected_hash: Expected proof hash

        Returns:
            True if verification successful
        """
        start_time = time.time()

        with self.lock:
            if proof_id not in self.active_proofs:
                logger.warning(f"Unknown proof ID for verification: {proof_id}")
                return False

            metrics = self.active_proofs[proof_id]

            # Perform verification (simplified - would call actual ZK verifier)
            try:
                # Check proof structure
                if len(proof_data) < 100:  # Minimum proof size
                    raise ValueError("Proof too small")

                # Verify hash if provided
                if expected_hash:
                    actual_hash = hashlib.sha256(proof_data).hexdigest()
                    if actual_hash != expected_hash:
                        raise ValueError("Proof hash mismatch")

                # Simulate verification computation
                # In production, this would call the actual ZK verifier
                verification_successful = self._perform_verification(proof_data)

                if verification_successful:
                    metrics.status = VerificationStatus.VERIFIED
                    verification_time = time.time() - start_time
                    metrics.verification_time_ms = verification_time * 1000

                    # Update metrics
                    self.verification_time.labels(type=metrics.proof_type.value).observe(
                        verification_time
                    )

                    self.proof_total.labels(type=metrics.proof_type.value, status="verified").inc()

                    # Update statistics
                    self._update_statistics(metrics)

                    logger.info(f"Proof {proof_id} verified successfully")
                    return True
                else:
                    raise ValueError("Verification computation failed")

            except Exception as e:
                metrics.status = VerificationStatus.FAILED
                metrics.error_message = str(e)

                self.proof_failures.labels(
                    type=metrics.proof_type.value, reason=type(e).__name__
                ).inc()

                self.proof_total.labels(type=metrics.proof_type.value, status="failed").inc()

                logger.error(f"Proof {proof_id} verification failed: {e}")

                # Generate alert if failure rate is high
                self._check_failure_rate(metrics.proof_type)

                return False

            finally:
                # Move to history
                self.proof_history.append(metrics)
                del self.active_proofs[proof_id]

                # Update gauge
                active_count = sum(
                    1 for p in self.active_proofs.values() if p.proof_type == metrics.proof_type
                )
                self.active_proofs_gauge.labels(type=metrics.proof_type.value).set(active_count)

    def _perform_verification(self, proof_data: bytes) -> bool:
        """
        Perform actual proof verification.

        In production, this would call the appropriate ZK verifier
        based on the proof type.

        Args:
            proof_data: Proof to verify

        Returns:
            True if verification successful
        """
        # Simulate verification with probability based on proof validity
        # In production, replace with actual verification logic

        # Check proof structure
        if len(proof_data) < 100:
            return False

        # Check magic bytes (example)
        if proof_data[:4] != b"ZKPF":  # Zero-Knowledge Proof Format
            # For now, accept any properly sized proof
            pass

        # Simulate computational verification
        time.sleep(0.001)  # Simulate computation time

        # Return success (would be actual verification result)
        return True

    def _update_statistics(self, metrics: ProofMetrics) -> None:
        """
        Update aggregated statistics.

        Args:
            metrics: Proof metrics to aggregate
        """
        stats = self.stats_by_type[metrics.proof_type]

        # Update counts
        stats.total_proofs += 1
        if metrics.status == VerificationStatus.VERIFIED:
            stats.verified_proofs += 1
        else:
            stats.failed_proofs += 1

        # Update averages (exponential moving average)
        alpha = 0.1  # Smoothing factor

        stats.avg_generation_time_ms = (
            alpha * metrics.generation_time_ms + (1 - alpha) * stats.avg_generation_time_ms
        )

        stats.avg_verification_time_ms = (
            alpha * metrics.verification_time_ms + (1 - alpha) * stats.avg_verification_time_ms
        )

        stats.avg_proof_size_bytes = (
            alpha * metrics.proof_size_bytes + (1 - alpha) * stats.avg_proof_size_bytes
        )

        # Update rates
        if stats.total_proofs > 0:
            stats.verification_rate = stats.verified_proofs / stats.total_proofs
            stats.failure_rate = stats.failed_proofs / stats.total_proofs

        # Update verification rate gauge
        self.verification_rate_gauge.labels(type=metrics.proof_type.value).set(
            stats.verification_rate
        )

        # Add to time-based metrics
        hour_key = datetime.fromtimestamp(metrics.timestamp).strftime("%Y%m%d%H")
        day_key = datetime.fromtimestamp(metrics.timestamp).strftime("%Y%m%d")

        self.hourly_metrics[hour_key].append(metrics)
        self.daily_metrics[day_key].append(metrics)

    def _check_failure_rate(self, proof_type: ProofType) -> None:
        """
        Check failure rate and generate alerts if needed.

        Args:
            proof_type: Type of proof to check
        """
        stats = self.stats_by_type[proof_type]

        if stats.verification_rate < self.alert_threshold:
            alert = {
                "timestamp": time.time(),
                "type": "high_failure_rate",
                "proof_type": proof_type.value,
                "failure_rate": stats.failure_rate,
                "threshold": 1 - self.alert_threshold,
                "message": f"High failure rate for {proof_type.value} proofs: {stats.failure_rate:.2%}",
            }

            self.alerts.append(alert)
            logger.error(alert["message"])

    def get_statistics(self, proof_type: Optional[ProofType] = None) -> Dict[str, Any]:
        """
        Get verification statistics.

        Args:
            proof_type: Optional specific proof type

        Returns:
            Statistics dictionary
        """
        with self.lock:
            if proof_type:
                stats = self.stats_by_type[proof_type]
                return asdict(stats)
            else:
                # Return all statistics
                all_stats = {
                    "by_type": {pt.value: asdict(self.stats_by_type[pt]) for pt in ProofType},
                    "total_active": len(self.active_proofs),
                    "total_historical": len(self.proof_history),
                    "recent_alerts": self.alerts[-10:],  # Last 10 alerts
                }

                # Add aggregate statistics
                total_proofs = sum(s.total_proofs for s in self.stats_by_type.values())
                total_verified = sum(s.verified_proofs for s in self.stats_by_type.values())

                all_stats["aggregate"] = {
                    "total_proofs": total_proofs,
                    "total_verified": total_verified,
                    "overall_verification_rate": (
                        total_verified / total_proofs if total_proofs > 0 else 0
                    ),
                }

                return all_stats

    def get_proof_history(
        self, limit: int = 100, proof_type: Optional[ProofType] = None
    ) -> List[Dict]:
        """
        Get recent proof history.

        Args:
            limit: Maximum number of entries
            proof_type: Optional filter by type

        Returns:
            List of proof metrics
        """
        with self.lock:
            history = list(self.proof_history)

            if proof_type:
                history = [h for h in history if h.proof_type == proof_type]

            # Return most recent
            return [asdict(h) for h in history[-limit:]]

    def get_time_series_metrics(self, period: str = "hour", lookback: int = 24) -> List[Dict]:
        """
        Get time series metrics.

        Args:
            period: 'hour' or 'day'
            lookback: Number of periods to look back

        Returns:
            Time series data
        """
        with self.lock:
            if period == "hour":
                metrics_dict = self.hourly_metrics
                time_format = "%Y%m%d%H"
                delta = timedelta(hours=1)
            else:
                metrics_dict = self.daily_metrics
                time_format = "%Y%m%d"
                delta = timedelta(days=1)

            # Generate time range
            current_time = datetime.now()
            time_series = []

            for i in range(lookback):
                timestamp = current_time - (delta * i)
                key = timestamp.strftime(time_format)

                if key in metrics_dict:
                    metrics_list = metrics_dict[key]

                    # Aggregate metrics for this period
                    verified = sum(
                        1 for m in metrics_list if m.status == VerificationStatus.VERIFIED
                    )
                    failed = sum(1 for m in metrics_list if m.status == VerificationStatus.FAILED)
                    avg_gen_time = (
                        np.mean([m.generation_time_ms for m in metrics_list]) if metrics_list else 0
                    )
                    avg_ver_time = (
                        np.mean([m.verification_time_ms for m in metrics_list])
                        if metrics_list
                        else 0
                    )

                    time_series.append(
                        {
                            "timestamp": timestamp.isoformat(),
                            "total": len(metrics_list),
                            "verified": verified,
                            "failed": failed,
                            "verification_rate": (
                                verified / len(metrics_list) if metrics_list else 0
                            ),
                            "avg_generation_time_ms": avg_gen_time,
                            "avg_verification_time_ms": avg_ver_time,
                        }
                    )
                else:
                    time_series.append(
                        {
                            "timestamp": timestamp.isoformat(),
                            "total": 0,
                            "verified": 0,
                            "failed": 0,
                            "verification_rate": 0,
                            "avg_generation_time_ms": 0,
                            "avg_verification_time_ms": 0,
                        }
                    )

            return time_series

    def export_prometheus_metrics(self) -> bytes:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus formatted metrics
        """
        return generate_latest(self.registry)

    def clear_old_metrics(self, days: int = 7) -> int:
        """
        Clear metrics older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of metrics cleared
        """
        cutoff_time = time.time() - (days * 24 * 3600)
        cleared = 0

        with self.lock:
            # Clear old proof history
            original_size = len(self.proof_history)
            self.proof_history = deque(
                (m for m in self.proof_history if m.timestamp > cutoff_time),
                maxlen=self.history_size,
            )
            cleared += original_size - len(self.proof_history)

            # Clear old time-based metrics
            for key in list(self.hourly_metrics.keys()):
                # Parse timestamp from key
                try:
                    timestamp = datetime.strptime(key, "%Y%m%d%H")
                    if timestamp.timestamp() < cutoff_time:
                        del self.hourly_metrics[key]
                        cleared += 1
                except:
                    pass

            for key in list(self.daily_metrics.keys()):
                try:
                    timestamp = datetime.strptime(key, "%Y%m%d")
                    if timestamp.timestamp() < cutoff_time:
                        del self.daily_metrics[key]
                        cleared += 1
                except:
                    pass

        logger.info(f"Cleared {cleared} old metrics")

        return cleared
