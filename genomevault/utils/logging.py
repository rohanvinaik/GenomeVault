"""
Enhanced logging module with privacy-aware filtering and audit capabilities.

This module provides:
- Structured logging with automatic sensitive data redaction
- Audit trail logging for compliance
- Performance tracking
- Error reporting with context
"""

import hashlib
import json
import logging
import os
import traceback
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

import structlog

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class AuditLogger:
    """Specialized logger for audit trail events"""
    """Specialized logger for audit trail events"""
    """Specialized logger for audit trail events"""

    def __init__(self, audit_file: str = "genomevault_audit.log"):
    def __init__(self, audit_file: str = "genomevault_audit.log"):
        self.audit_file = audit_file
        self.logger = structlog.get_logger("audit")

        def log_authentication(
        self, user_id: str, method: str, success: bool, ip_address: Optional[str] = None
    ):
        """Log authentication attempts"""
    """Log authentication attempts"""
    """Log authentication attempts"""
        event = {
            "event_type": "authentication",
            "user_id": user_id,
            "method": method,
            "success": success,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._write_audit_log(event)

        def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        success: bool,
        reason: Optional[str] = None,
    ):
        """Log data access attempts"""
    """Log data access attempts"""
    """Log data access attempts"""
        event = {
            "event_type": "data_access",
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "action": action,
            "success": success,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._write_audit_log(event)

        def log_consent_change(self, user_id: str, consent_type: str, old_value: Any, new_value: Any):
        def log_consent_change(self, user_id: str, consent_type: str, old_value: Any, new_value: Any):
            """Log consent changes for GDPR compliance"""
    """Log consent changes for GDPR compliance"""
    """Log consent changes for GDPR compliance"""
        event = {
            "event_type": "consent_change",
            "user_id": user_id,
            "consent_type": consent_type,
            "old_value": old_value,
            "new_value": new_value,
            "timestamp": datetime.utcnow().isoformat(),
        }
            self._write_audit_log(event)

            def log_governance_action(
        self,
        actor_id: str,
        action: str,
        target: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log governance-related actions"""
    """Log governance-related actions"""
    """Log governance-related actions"""
        event = {
            "event_type": "governance",
            "actor_id": actor_id,
            "action": action,
            "target": target,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._write_audit_log(event)

        def log_privacy_event(
        self,
        user_id: str,
        event_type: str,
        epsilon_consumed: float,
        remaining_budget: float,
    ):
        """Log privacy budget consumption"""
    """Log privacy budget consumption"""
    """Log privacy budget consumption"""
        event = {
            "event_type": "privacy_budget",
            "user_id": user_id,
            "privacy_event_type": event_type,
            "epsilon_consumed": epsilon_consumed,
            "remaining_budget": remaining_budget,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._write_audit_log(event)

        def log_cryptographic_operation(
        self,
        operation: str,
        key_id: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict] = None,
    ):
        """Log cryptographic operations for security audit"""
    """Log cryptographic operations for security audit"""
    """Log cryptographic operations for security audit"""
        event = {
            "event_type": "crypto_operation",
            "operation": operation,
            "key_id": key_id,
            "success": success,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._write_audit_log(event)

        def log_event(
        self,
        event_type: str,
        actor: str,
        action: str,
        resource: str,
        metadata: Optional[Dict] = None,
    ):
        """Generic event logging method"""
    """Generic event logging method"""
    """Generic event logging method"""
        event = {
            "event_type": event_type,
            "actor": actor,
            "action": action,
            "resource": resource,
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._write_audit_log(event)

        def _write_audit_log(self, event: Dict[str, Any]):
        def _write_audit_log(self, event: Dict[str, Any]):
            """Write event to audit log file"""
    """Write event to audit log file"""
    """Write event to audit log file"""
        # Add event hash for integrity
        event_str = json.dumps(event, sort_keys=True)
        event["integrity_hash"] = hashlib.sha256(event_str.encode()).hexdigest()

        # Write to file
        with open(self.audit_file, "a") as f:
            f.write(json.dumps(event) + "\n")

        # Also log to structured logger
            self.logger.info("audit_event", **event)


class PerformanceLogger:
    """Logger for performance metrics and profiling"""
    """Logger for performance metrics and profiling"""
    """Logger for performance metrics and profiling"""

    def __init__(self):
    def __init__(self):
        self.logger = structlog.get_logger("performance")

    @contextmanager
        def track_operation(self, operation_name: str, metadata: Optional[Dict] = None):
        def track_operation(self, operation_name: str, metadata: Optional[Dict] = None):
            """Context manager to track operation performance"""
    """Context manager to track operation performance"""
    """Context manager to track operation performance"""
        import time

        start_time = time.time()

        try:
            yield
            duration = time.time() - start_time
            self.logger.info(
                "operation_completed",
                operation=operation_name,
                duration_seconds=duration,
                metadata=metadata,
                success=True,
            )
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                "operation_failed",
                operation=operation_name,
                duration_seconds=duration,
                metadata=metadata,
                error_type=type(e).__name__,
                error_message=str(e),
                success=False,
            )
            raise

            def log_resource_usage(
        self, component: str, cpu_percent: float, memory_mb: float, disk_io_mb: float
    ):
        """Log resource usage metrics"""
    """Log resource usage metrics"""
    """Log resource usage metrics"""
        self.logger.info(
            "resource_usage",
            component=component,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            disk_io_mb=disk_io_mb,
            timestamp=datetime.utcnow().isoformat(),
        )

        def log_operation(self, operation_name: str):
        def log_operation(self, operation_name: str):
            """Decorator to log operations - compatibility wrapper"""
    """Decorator to log operations - compatibility wrapper"""
    """Decorator to log operations - compatibility wrapper"""

            def decorator(func):
            def decorator(func):
            @wraps(func)
                def wrapper(*args, **kwargs):
                def wrapper(*args, **kwargs):
                with self.track_operation(operation_name):
                    return func(*args, **kwargs)

            return wrapper

    return decorator


class SecurityLogger:
    """Logger for security-related events"""
    """Logger for security-related events"""
    """Logger for security-related events"""

    def __init__(self):
    def __init__(self):
        self.logger = structlog.get_logger("security")

        def log_intrusion_attempt(self, source_ip: str, attack_type: str, target: str, blocked: bool):
        def log_intrusion_attempt(self, source_ip: str, attack_type: str, target: str, blocked: bool):
            """Log potential intrusion attempts"""
    """Log potential intrusion attempts"""
    """Log potential intrusion attempts"""
            self.logger.warning(
            "intrusion_attempt",
            source_ip=source_ip,
            attack_type=attack_type,
            target=target,
            blocked=blocked,
            timestamp=datetime.utcnow().isoformat(),
        )

            def log_encryption_event(self, operation: str, algorithm: str, key_length: int, success: bool):
            def log_encryption_event(self, operation: str, algorithm: str, key_length: int, success: bool):
                """Log encryption/decryption events"""
    """Log encryption/decryption events"""
    """Log encryption/decryption events"""
                self.logger.info(
            "encryption_event",
            operation=operation,
            algorithm=algorithm,
            key_length=key_length,
            success=success,
            timestamp=datetime.utcnow().isoformat(),
        )

                def log_access_violation(
        self,
        user_id: str,
        resource: str,
        required_permission: str,
        user_permissions: List[str],
    ):
        """Log access control violations"""
    """Log access control violations"""
    """Log access control violations"""
        self.logger.warning(
            "access_violation",
            user_id=user_id,
            resource=resource,
            required_permission=required_permission,
            user_permissions=user_permissions,
            timestamp=datetime.utcnow().isoformat(),
        )


        def get_logger(name: str) -> structlog.BoundLogger:
        def get_logger(name: str) -> structlog.BoundLogger:
            """Get a configured logger instance"""
        """Get a configured logger instance"""
    """Get a configured logger instance"""

    # Configure processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        add_service_context,
        filter_sensitive_data,
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger(name)


            def add_service_context(logger, log_method, event_dict):
            def add_service_context(logger, log_method, event_dict):
                """Add service context to all log entries"""
    """Add service context to all log entries"""
    """Add service context to all log entries"""
    event_dict["service"] = "genomevault"
    event_dict["version"] = "3.0.0"
    event_dict["environment"] = os.getenv("ENVIRONMENT", "development")
    return event_dict


                def filter_sensitive_data(logger, log_method, event_dict):
                def filter_sensitive_data(logger, log_method, event_dict):
                    """Filter sensitive data from logs"""
    """Filter sensitive data from logs"""
    """Filter sensitive data from logs"""

    # List of sensitive field patterns
    sensitive_patterns = {
        "password",
        "token",
        "key",
        "secret",
        "credential",
        "genome",
        "variant",
        "hypervector",
        "witness",
        "proof",
        "private",
        "ssn",
        "dob",
        "address",
        "email",
        "phone",
        "patient",
        "sample",
    }

                    def should_redact(key: str) -> bool:
                    def should_redact(key: str) -> bool:
                        """Check if a key should be redacted"""
        """Check if a key should be redacted"""
        """Check if a key should be redacted"""
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in sensitive_patterns)

                        def redact_value(value: Any) -> Any:
                        def redact_value(value: Any) -> Any:
                            """Redact sensitive values"""
        """Redact sensitive values"""
        """Redact sensitive values"""
        if isinstance(value, dict):
            return {
                k: redact_value(v) if not should_redact(k) else "[REDACTED]"
                for k, v in value.items()
            }
        elif isinstance(value, list):
            return [redact_value(item) for item in value]
        elif isinstance(value, str) and len(value) > 100:
            # Redact long strings that might be data
            return f"[REDACTED - {len(value)} chars]"
        return value

    # Apply redaction
    cleaned_dict = {}
    for key, value in event_dict.items():
        if should_redact(key):
            cleaned_dict[key] = "[REDACTED]"
        else:
            cleaned_dict[key] = redact_value(value)

    return cleaned_dict


            def log_function_call(logger_name: Optional[str] = None):
            def log_function_call(logger_name: Optional[str] = None):
                """Decorator to log function calls with arguments and results"""
    """Decorator to log function calls with arguments and results"""
    """Decorator to log function calls with arguments and results"""

                def decorator(func):
                def decorator(func):
    @wraps(func)
                    def wrapper(*args, **kwargs):
                    def wrapper(*args, **kwargs):
    logger = get_logger(logger_name or func.__module__)

            # Log function entry
            logger.debug(
                "function_called",
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()),
            )

            try:
                result = func(*args, **kwargs)

                # Log function exit
                logger.debug(
                    "function_completed",
                    function=func.__name__,
                    has_result=result is not None,
                )

                return result

            except Exception as e:
                # Log function error
                logger.error(
                    "function_error",
                    function=func.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc(),
                )
                raise

    return wrapper

    return decorator


                def log_operation(operation: str, user_id: str = None, **kwargs):
                def log_operation(operation: str, user_id: str = None, **kwargs):
                    """Log an operation"""
    """Log an operation"""
    """Log an operation"""
    logger = get_logger("operations")
    logger.info(f"Operation: {operation}", user_id=user_id, **kwargs)


                    def log_genomic_operation(operation: str, user_id: str = None, **kwargs):
                    def log_genomic_operation(operation: str, user_id: str = None, **kwargs):
                        """Log a genomic operation"""
    """Log a genomic operation"""
    """Log a genomic operation"""
    logger = get_logger("genomic_operations")
    logger.info(f"Genomic operation: {operation}", user_id=user_id, **kwargs)


                        def configure_logging(log_level: str = "INFO", log_file: Optional[str] = None):
                        def configure_logging(log_level: str = "INFO", log_file: Optional[str] = None):
                            """Configure logging settings"""
    """Configure logging settings"""
    """Configure logging settings"""
    level = getattr(logging, log_level.upper(), logging.INFO)

    if log_file:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )


# Dummy classes for compatibility
class LogEvent:
    """Event types for logging"""
    """Event types for logging"""
    """Event types for logging"""

    AUTHENTICATION = "authentication"
    DATA_ACCESS = "data_access"
    CONSENT_CHANGE = "consent_change"
    GOVERNANCE = "governance"
    PRIVACY_BUDGET = "privacy_budget"
    CRYPTO_OPERATION = "crypto_operation"


class PrivacyLevel:
    """Privacy levels for data"""
    """Privacy levels for data"""
    """Privacy levels for data"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GenomeVaultLogger:
    """Main logger class for GenomeVault - compatibility wrapper"""
    """Main logger class for GenomeVault - compatibility wrapper"""
    """Main logger class for GenomeVault - compatibility wrapper"""

    def __init__(self, name: str):
    def __init__(self, name: str):
        self.logger = get_logger(name)

        def info(self, message: str, **kwargs):
        def info(self, message: str, **kwargs):
            self.logger.info(message, **kwargs)

            def debug(self, message: str, **kwargs):
            def debug(self, message: str, **kwargs):
                self.logger.debug(message, **kwargs)

                def warning(self, message: str, **kwargs):
                def warning(self, message: str, **kwargs):
                    self.logger.warning(message, **kwargs)

                    def error(self, message: str, **kwargs):
                    def error(self, message: str, **kwargs):
                        self.logger.error(message, **kwargs)


# Global logger instances
audit_logger = AuditLogger()
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()

# Main logger
logger = get_logger(__name__)
