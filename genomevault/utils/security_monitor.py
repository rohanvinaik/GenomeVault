"""
Security monitoring and intrusion detection for GenomeVault.

This module provides:
- Real-time intrusion detection
- Anomaly detection for suspicious access patterns
- Security event correlation
- Automated response to security threats

"""

from __future__ import annotations

from collections import defaultdict
from typing import Any
import json

from sklearn.ensemble import IsolationForest
import numpy as np

from ..genomevault.utils.logging import audit_logger, get_logger, security_logger
from ..genomevault.utils.monitoring import metrics_collector

logger = get_logger(__name__)


class SecurityMonitor:
    """Real-time security monitoring and threat detection"""

    def __init__(self, config: dict[str, Any]):
        """Initialize instance.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.threat_threshold = config.get("threat_threshold", 0.8)
        self.anomaly_window = config.get("anomaly_window_minutes", 60)

        # Access pattern tracking
        self.access_patterns = defaultdict(list)
        self.failed_attempts = defaultdict(int)
        self.blocked_ips = set()

        # Anomaly detection model
        self.anomaly_detector = IsolationForest(contamination=0.01, random_state=42)
        self.is_trained = False

        # Alert callbacks
        self.alert_callbacks = []

    async def monitor_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        success: bool,
        ip_address: str,
        metadata: dict | None = None,
    ):
        """Monitor access attempts for security threats"""

        # Record access attempt
        access_event = {
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "success": success,
            "ip_address": ip_address,
            "timestamp": _datetime.utcnow(),
            "metadata": metadata or {},
        }

        # Check if IP is blocked
        if ip_address in self.blocked_ips:
            security_logger.log_intrusion_attempt(
                source_ip=ip_address,
                attack_type="blocked_ip_access",
                target=resource,
                blocked=True,
            )
            return False

        # Track failed attempts
        if not success:
            self.failed_attempts[ip_address] += 1

            # Check for brute force
            if self.failed_attempts[ip_address] > 5:
                await self._handle_brute_force(ip_address, user_id, resource)
                return False

        # Add to access patterns for anomaly detection
        self.access_patterns[user_id].append(access_event)

        # Check for anomalies
        if self.is_trained:
            is_anomaly = await self._detect_anomaly(user_id, access_event)
            if is_anomaly:
                await self._handle_anomaly(user_id, access_event)

        # Periodic training of anomaly detector
        if len(self.access_patterns[user_id]) > 100 and not self.is_trained:
            await self._train_anomaly_detector()

        return True

    async def monitor_data_access(self, user_id: str, data_type: str, volume: int, operation: str):
        """Monitor data access patterns for exfiltration attempts"""

        # Check for unusual data volume
        if volume > self._get_volume_threshold(data_type):
            security_logger.log_intrusion_attempt(
                source_ip="internal",
                attack_type="data_exfiltration_attempt",
                target="{data_type}:{operation}",
                blocked=False,
            )

            await self._trigger_alert(
                "data_exfiltration_risk",
                {
                    "user_id": user_id,
                    "data_type": data_type,
                    "volume": volume,
                    "operation": operation,
                },
            )

    async def monitor_privacy_violations(
        self, violation_type: str, user_id: str, details: dict[str, Any]
    ):
        """Monitor and respond to privacy violations"""

        # Log violation
        metrics_collector.record_privacy_violation(violation_type)

        # Determine severity
        severity = self._assess_violation_severity(violation_type, details)

        if severity >= 0.8:
            # Immediate response for critical violations
            await self._block_user(user_id, "Critical privacy violation: {violation_type}")

        await self._trigger_alert(
            "privacy_violation",
            {
                "violation_type": violation_type,
                "user_id": user_id,
                "severity": severity,
                "details": details,
            },
        )

    async def monitor_cryptographic_operations(
        self, operation: str, key_id: str | None = None, algorithm: str = None
    ):
        """Monitor cryptographic operations for suspicious activity"""

        # Check for deprecated algorithms
        deprecated_algorithms = {"MD5", "SHA1", "DES", "3DES"}
        if algorithm and algorithm.upper() in deprecated_algorithms:
            security_logger.log_intrusion_attempt(
                source_ip="internal",
                attack_type="deprecated_crypto_usage",
                target="{operation}:{algorithm}",
                blocked=False,
            )

            await self._trigger_alert(
                "deprecated_crypto",
                {"operation": operation, "algorithm": algorithm, "key_id": key_id},
            )

    async def _detect_anomaly(self, user_id: str, access_event: dict) -> bool:
        """Detect anomalous access patterns using ML"""

        # Extract features from access history
        features = self._extract_access_features(
            self.access_patterns[user_id][-100:]  # Last 100 events
        )

        # Predict anomaly
        anomaly_score = self.anomaly_detector.decision_function([features])[0]

        # Lower score means more anomalous
        is_anomaly = anomaly_score < -self.threat_threshold

        if is_anomaly:
            logger.warning(
                "anomaly_detected",
                user_id=user_id,
                anomaly_score=anomaly_score,
                access_event=access_event,
            )

        return is_anomaly

    async def _train_anomaly_detector(self):
        """Train the anomaly detection model"""

        try:
            # Collect training data from all users
            all_features = []

            for user_id, events in self.access_patterns.items():
                if len(events) > 50:
                    # Extract features for sequences of events
                    for i in range(50, len(events)):
                        features = self._extract_access_features(events[i - 50 : i])
                        all_features.append(features)

            if len(all_features) > 100:
                # Train model
                X = np.array(all_features)
                self.anomaly_detector.fit(X)
                self.is_trained = True

                logger.info("anomaly_detector_trained", training_samples=len(all_features))

        except Exception as e:
            logger.exception("Unhandled exception")
            logger.error("anomaly_detector_training_failed", error=str(e))
            raise RuntimeError("Unspecified error")

    def _extract_access_features(self, events: list[dict]) -> list[float]:
        """Extract features from access event sequence"""

        if not events:
            return [0] * 10

        # Time-based features
        timestamps = [e["timestamp"] for e in events]
        time_deltas = [
            (timestamps[i + 1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)
        ]

        # Access pattern features
        resources = [e["resource"] for e in events]
        actions = [e["action"] for e in events]
        success_rate = sum(1 for e in events if e["success"]) / len(events)

        # IP diversity
        unique_ips = len({e.get("ip_address", "") for e in events})

        features = [
            len(events),  # Event count
            np.mean(time_deltas) if time_deltas else 0,  # Avg time between events
            np.std(time_deltas) if time_deltas else 0,  # Std of time deltas
            len(set(resources)),  # Unique resources accessed
            len(set(actions)),  # Unique actions performed
            success_rate,  # Success rate
            unique_ips,  # IP diversity
            len(set(resources)) / len(events),  # Resource diversity ratio
            max(time_deltas) if time_deltas else 0,  # Max time gap
            min(time_deltas) if time_deltas else 0,  # Min time gap
        ]

        return features

    async def _handle_brute_force(self, ip_address: str, user_id: str, resource: str):
        """Handle detected brute force attempt"""

        # Block IP
        self.blocked_ips.add(ip_address)

        # Log intrusion
        security_logger.log_intrusion_attempt(
            source_ip=ip_address,
            attack_type="brute_force",
            target=resource,
            blocked=True,
        )

        # Trigger alert
        await self._trigger_alert(
            "brute_force_attack",
            {
                "ip_address": ip_address,
                "user_id": user_id,
                "resource": resource,
                "failed_attempts": self.failed_attempts[ip_address],
            },
        )

        # Reset counter
        self.failed_attempts[ip_address] = 0

    async def _handle_anomaly(self, user_id: str, access_event: dict):
        """Handle detected anomaly"""

        security_logger.log_intrusion_attempt(
            source_ip=access_event.get("ip_address", "unknown"),
            attack_type="anomalous_access_pattern",
            target=access_event["resource"],
            blocked=False,
        )

        await self._trigger_alert(
            "anomalous_access", {"user_id": user_id, "access_event": access_event}
        )

    async def _block_user(self, user_id: str, reason: str):
        """Block a user account"""

        logger.warning("user_blocked", user_id=user_id, reason=reason)

        # This would integrate with the authentication system
        # to actually block the user

    async def _trigger_alert(self, alert_type: str, details: dict[str, Any]):
        """Trigger security alert"""

        alert = {
            "type": alert_type,
            "timestamp": _datetime.utcnow().isoformat(),
            "details": details,
            "severity": self._calculate_severity(alert_type, details),
        }

        # Log alert
        logger.warning("security_alert", alert_type=alert_type, **details)

        # Execute callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.exception("Unhandled exception")
                logger.error("alert_callback_failed", callback=callback.__name__, error=str(e))
                raise RuntimeError("Unspecified error")

    def _get_volume_threshold(self, data_type: str) -> int:
        """Get volume threshold for data type"""

        thresholds = {
            "genomic": 1000000,  # 1M variants
            "hypervector": 10000,  # 10K vectors
            "proof": 1000,  # 1K proofs
            "reference": 100000,  # 100K queries
        }

        return thresholds.get(data_type, 10000)

    def _assess_violation_severity(self, violation_type: str, details: dict[str, Any]) -> float:
        """Assess severity of privacy violation"""

        base_severities = {
            "re_identification_attempt": 0.9,
            "unauthorized_linkage": 0.8,
            "excessive_queries": 0.6,
            "timing_attack": 0.7,
            "membership_inference": 0.8,
        }

        severity = base_severities.get(violation_type, 0.5)

        # Adjust based on details
        if details.get("affected_users", 1) > 10:
            severity = min(1.0, severity * 1.2)

        return severity

    def _calculate_severity(self, alert_type: str, details: dict[str, Any]) -> float:
        """Calculate alert severity"""

        base_severities = {
            "brute_force_attack": 0.7,
            "anomalous_access": 0.6,
            "data_exfiltration_risk": 0.8,
            "privacy_violation": 0.9,
            "deprecated_crypto": 0.5,
        }

        return base_severities.get(alert_type, 0.5)

    def register_alert_callback(self, callback):
        """Register callback for security alerts"""
        self.alert_callbacks.append(callback)

    def get_security_status(self) -> dict[str, Any]:
        """Get current security status"""

        return {
            "blocked_ips": list(self.blocked_ips),
            "active_threats": len(self.blocked_ips),
            "anomaly_detector_trained": self.is_trained,
            "users_monitored": len(self.access_patterns),
            "timestamp": _datetime.utcnow().isoformat(),
        }


class ComplianceMonitor:
    """Monitor compliance with regulations (HIPAA, GDPR, etc.)"""

    def __init__(self, config: dict[str, Any]):
        """Initialize instance.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.compliance_checks = []
        self.violations = []

    async def check_hipaa_compliance(self, operation: str, context: dict[str, Any]) -> bool:
        """Check HIPAA compliance for operation"""

        # Minimum necessary standard
        if operation == "data_access":
            data_fields = context.get("fields_accessed", [])
            necessary_fields = context.get("necessary_fields", [])

            unnecessary_access = set(data_fields) - set(necessary_fields)
            if unnecessary_access:
                await self._record_violation(
                    "hipaa_minimum_necessary",
                    {
                        "operation": operation,
                        "unnecessary_fields": list(unnecessary_access),
                        "user_id": context.get("user_id"),
                    },
                )
                return False

        # Audit trail requirement
        if operation in ["data_access", "data_modification", "data_deletion"]:
            if not context.get("audit_logged"):
                await self._record_violation(
                    "hipaa_audit_trail_missing",
                    {"operation": operation, "context": context},
                )
                return False

        # Encryption requirement
        if operation == "data_transmission":
            if not context.get("encrypted") or context.get("encryption_algorithm") == "none":
                await self._record_violation(
                    "hipaa_encryption_missing",
                    {"operation": operation, "context": context},
                )
                return False

        return True

    async def check_gdpr_compliance(self, operation: str, context: dict[str, Any]) -> bool:
        """Check GDPR compliance for operation"""

        # Consent verification
        if operation in ["data_collection", "data_processing"]:
            if not context.get("consent_verified"):
                await self._record_violation(
                    "gdpr_consent_missing",
                    {
                        "operation": operation,
                        "user_id": context.get("user_id"),
                        "data_type": context.get("data_type"),
                    },
                )
                return False

        # Right to erasure
        if operation == "deletion_request":
            if context.get("deletion_blocked"):
                await self._record_violation(
                    "gdpr_right_to_erasure_blocked",
                    {
                        "user_id": context.get("user_id"),
                        "reason": context.get("block_reason"),
                    },
                )
                return False

        # Data portability
        if operation == "export_request":
            if not context.get("machine_readable_format"):
                await self._record_violation(
                    "gdpr_portability_format",
                    {
                        "user_id": context.get("user_id"),
                        "format": context.get("export_format"),
                    },
                )
                return False

        return True

    async def _record_violation(self, violation_type: str, details: dict[str, Any]):
        """Record compliance violation"""

        violation = {
            "type": violation_type,
            "timestamp": _datetime.utcnow().isoformat(),
            "details": details,
        }

        self.violations.append(violation)

        # Log to audit system
        audit_logger.log_data_access(
            user_id="system",
            resource_type="compliance",
            resource_id=violation_type,
            action="violation_recorded",
            success=True,
            reason=json.dumps(details),
        )

        logger.warning("compliance_violation", violation_type=violation_type, **details)

    def get_compliance_report(
        self, start_date: _datetime | None = None, end_date: _datetime | None = None
    ) -> dict[str, Any]:
        """Generate compliance report"""

        # Filter violations by date
        filtered_violations = []
        for violation in self.violations:
            v_time = _datetime.fromisoformat(violation["timestamp"])
            if start_date and v_time < start_date:
                continue
            if end_date and v_time > end_date:
                continue
            filtered_violations.append(violation)

        # Group by type
        by_type = defaultdict(list)
        for violation in filtered_violations:
            by_type[violation["type"]].append(violation)

        return {
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None,
            },
            "total_violations": len(filtered_violations),
            "violations_by_type": {vtype: len(violations) for vtype, violations in by_type.items()},
            "violations": filtered_violations,
            "generated_at": _datetime.utcnow().isoformat(),
        }
