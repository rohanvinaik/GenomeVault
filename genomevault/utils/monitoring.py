"""
Centralized monitoring and metrics collection for GenomeVault.

This module provides comprehensive monitoring capabilities including:
- Processing metrics for genomic data operations
- Hypervector operation tracking
- PIR network performance monitoring
- Blockchain metrics
- Privacy-preserving logging
"""

import time
from datetime import datetime
from functools import wraps
from typing import Any

import prometheus_client
import structlog

# Initialize structured logger
logger = structlog.get_logger(__name__)


class MetricsCollector:
    """Centralized metrics collection for GenomeVault"""

    def __init__(self):
        # Processing metrics
        self.processing_duration = Histogram(
            "genomevault_processing_duration_seconds",
            "Time spent processing genomic data",
            ["operation", "data_type"],
        )

        self.processing_errors = Counter(
            "genomevault_processing_errors_total",
            "Total processing errors",
            ["operation", "data_type", "error_type"],
        )

        # Hypervector metrics
        self.hypervector_operations = Counter(
            "genomevault_hypervector_operations_total",
            "Total hypervector operations performed",
            ["operation_type"],
        )

        self.hypervector_compression_ratio = Gauge(
            "genomevault_hypervector_compression_ratio",
            "Current compression ratio achieved",
            ["tier"],
        )

        # Zero-knowledge proof metrics
        self.proof_generation_time = Histogram(
            "genomevault_proof_generation_seconds",
            "Time to generate zero-knowledge proofs",
            ["circuit_type"],
        )

        self.proof_verification_time = Histogram(
            "genomevault_proof_verification_seconds",
            "Time to verify zero-knowledge proofs",
            ["circuit_type"],
        )

        self.proof_size_bytes = Histogram(
            "genomevault_proof_size_bytes",
            "Size of generated proofs in bytes",
            ["circuit_type"],
        )

        # PIR metrics
        self.pir_query_latency = Histogram(
            "genomevault_pir_query_latency_seconds",
            "PIR query latency",
            ["shard_count", "config_type"],
        )

        self.pir_privacy_failure_prob = Gauge(
            "genomevault_pir_privacy_failure_probability",
            "Current PIR privacy failure probability",
        )

        self.pir_server_honesty = Gauge(
            "genomevault_pir_server_honesty_score",
            "Server honesty probability",
            ["server_type"],
        )

        # Blockchain metrics
        self.block_production_time = Histogram(
            "genomevault_block_production_seconds", "Time to produce blocks"
        )

        self.node_voting_power = Gauge(
            "genomevault_node_voting_power",
            "Node voting power distribution",
            ["node_class", "signatory_status"],
        )

        self.credits_earned = Counter(
            "genomevault_credits_earned_total", "Total credits earned", ["node_type"]
        )

        self.governance_proposals = Counter(
            "genomevault_governance_proposals_total",
            "Total governance proposals",
            ["status", "committee"],
        )

        # Storage metrics
        self.storage_tier_usage = Gauge(
            "genomevault_storage_tier_bytes",
            "Storage usage by compression tier",
            ["tier"],
        )

        self.storage_operations = Counter(
            "genomevault_storage_operations_total",
            "Storage operations performed",
            ["operation", "tier"],
        )

        # Privacy metrics
        self.privacy_budget_consumed = Gauge(
            "genomevault_privacy_budget_consumed",
            "Differential privacy budget consumed",
            ["user_id"],
        )

        self.privacy_violations = Counter(
            "genomevault_privacy_violations_total",
            "Privacy violation attempts detected",
            ["violation_type"],
        )

        # System health metrics
        self.system_uptime = Gauge("genomevault_system_uptime_seconds", "System uptime in seconds")

        self.active_connections = Gauge(
            "genomevault_active_connections",
            "Number of active connections",
            ["connection_type"],
        )

        # Initialize system uptime
        self._start_time = time.time()

    def track_processing(self, operation: str, data_type: str):
        """Decorator to track processing operations"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    with self.processing_duration.labels(
                        operation=operation, data_type=data_type
                    ).time():
                        result = func(*args, **kwargs)

                    # Log successful operation
                    logger.info(
                        "processing_completed",
                        operation=operation,
                        data_type=data_type,
                        duration=time.time() - start_time,
                    )

                    return result

                except Exception as e:
                    from genomevault.observability.logging import configure_logging

                    logger = configure_logging()
                    logger.exception("Unhandled exception")
                    # Track error
                    self.processing_errors.labels(
                        operation=operation,
                        data_type=data_type,
                        error_type=type(e).__name__,
                    ).inc()

                    # Log error
                    logger.error(
                        "processing_error",
                        operation=operation,
                        data_type=data_type,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )

                    raise
                    raise

            return wrapper

        return decorator

    def track_hypervector_operation(self, operation_type: str):
        """Track hypervector operations"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()

                # Increment operation counter
                self.hypervector_operations.labels(operation_type=operation_type).inc()

                result = func(*args, **kwargs)

                # Log operation
                logger.debug(
                    "hypervector_operation",
                    operation_type=operation_type,
                    duration=time.time() - start_time,
                )

                return result

            return wrapper

        return decorator

    def track_proof_generation(self, circuit_type: str):
        """Track zero-knowledge proof generation"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.proof_generation_time.labels(circuit_type=circuit_type).time():
                    proof = func(*args, **kwargs)

                # Track proof size
                if hasattr(proof, "size_bytes"):
                    self.proof_size_bytes.labels(circuit_type=circuit_type).observe(
                        proof.size_bytes
                    )

                return proof

            return wrapper

        return decorator

    def update_pir_metrics(self, config: dict[str, Any]):
        """Update PIR-specific metrics"""
        # Calculate privacy failure probability
        k = config.get("trusted_signatures", 2)
        q = config.get("server_honesty_prob", 0.98)
        p_fail = (1 - q) ** k

        self.pir_privacy_failure_prob.set(p_fail)

        # Update server honesty scores
        if "server_type" in config:
            self.pir_server_honesty.labels(server_type=config["server_type"]).set(q)

    def update_node_metrics(self, node_class: str, signatory_status: str, voting_power: int):
        """Update blockchain node metrics"""
        self.node_voting_power.labels(node_class=node_class, signatory_status=signatory_status).set(
            voting_power
        )

    def update_storage_metrics(self, tier: str, size_bytes: int):
        """Update storage tier usage metrics"""
        self.storage_tier_usage.labels(tier=tier).set(size_bytes)

    def track_privacy_budget(self, user_id: str, epsilon_consumed: float):
        """Track differential privacy budget consumption"""
        self.privacy_budget_consumed.labels(user_id=user_id).set(epsilon_consumed)

    def record_privacy_violation(self, violation_type: str):
        """Record privacy violation attempts"""
        self.privacy_violations.labels(violation_type=violation_type).inc()

        # Also log for audit
        logger.warning(
            "privacy_violation_detected",
            violation_type=violation_type,
            timestamp=datetime.utcnow().isoformat(),
        )

    def update_system_health(self):
        """Update system health metrics"""
        self.system_uptime.set(time.time() - self._start_time)

    def export_metrics(self):
        """Export metrics in Prometheus format"""
        return prometheus_client.generate_latest()

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get a summary of current metrics"""
        return {
            "uptime_seconds": time.time() - self._start_time,
            "total_hypervector_operations": sum(self.hypervector_operations._metrics.values()),
            "current_pir_privacy_failure_prob": self.pir_privacy_failure_prob._value.get(),
            "timestamp": datetime.utcnow().isoformat(),
        }


class PrivacyAwareLogger:
    """Structured logging with privacy-aware filtering"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = structlog.get_logger(service_name)
        self.configure_logging()

    def configure_logging(self):
        """Configure structured logging with privacy filters"""
        structlog.configure(
            processors=[
                self._add_timestamp,
                self._add_service_info,
                self._filter_sensitive_data,
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def _add_timestamp(self, logger, log_method, event_dict):
        """Add timestamp to log entries"""
        event_dict["timestamp"] = datetime.utcnow().isoformat()
        return event_dict

    def _add_service_info(self, logger, log_method, event_dict):
        """Add service information to log entries"""
        event_dict["service"] = self.service_name
        return event_dict

    def _filter_sensitive_data(self, logger, log_method, event_dict):
        """Filter out sensitive genomic data from logs"""
        sensitive_keys = [
            "genome_data",
            "variants",
            "hypervector",
            "raw_sequence",
            "private_key",
            "witness",
            "proof_witness",
            "encryption_key",
            "patient_id",
            "sample_id",
            "phenotype_data",
        ]

        # Deep scan for sensitive keys
        def redact_dict(d: dict[str, Any]) -> dict[str, Any]:
            redacted = {}
            for key, value in d.items():
                if any(sk in key.lower() for sk in sensitive_keys):
                    redacted[key] = "[REDACTED]"
                elif isinstance(value, dict):
                    redacted[key] = redact_dict(value)
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    redacted[key] = [
                        redact_dict(item) if isinstance(item, dict) else item for item in value
                    ]
                else:
                    redacted[key] = value
            return redacted

        return redact_dict(event_dict)

    def log_audit_event(self, event_type: str, details: dict[str, Any]):
        """Log security audit events"""
        self.logger.info("audit_event", event_type=event_type, details=details, audit=True)

    def log_access_attempt(self, user_id: str, resource: str, action: str, success: bool):
        """Log access attempts for audit trail"""
        self.logger.info(
            "access_attempt",
            user_id=user_id,
            resource=resource,
            action=action,
            success=success,
            audit=True,
        )

    def log_privacy_event(self, event_type: str, epsilon_consumed: float, remaining_budget: float):
        """Log privacy budget consumption events"""
        self.logger.info(
            "privacy_event",
            event_type=event_type,
            epsilon_consumed=epsilon_consumed,
            remaining_budget=remaining_budget,
            privacy=True,
        )

    def log_governance_event(
        self,
        action: str,
        proposal_id: str | None = None,
        voter_id: str | None = None,
        vote: str | None = None,
    ):
        """Log governance-related events"""
        self.logger.info(
            "governance_event",
            action=action,
            proposal_id=proposal_id,
            voter_id=voter_id,
            vote=vote,
            governance=True,
        )


# Global metrics collector instance
metrics_collector = MetricsCollector()

# Convenience decorators
track_processing = metrics_collector.track_processing
track_hypervector_operation = metrics_collector.track_hypervector_operation
track_proof_generation = metrics_collector.track_proof_generation
