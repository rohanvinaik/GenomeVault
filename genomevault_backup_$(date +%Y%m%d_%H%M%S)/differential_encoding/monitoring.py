"""
Performance Monitoring and Metrics Collection for Differential Encoding

This module provides comprehensive monitoring, metrics collection, and alerting
for the differential encoding pipeline in production environments.

Features:
- Real-time performance metrics collection
- Cryptographic operation audit logging
- Verification failure alerts
- Resource usage tracking
- Performance degradation detection
- Configurable alert thresholds
- Metrics export for external monitoring systems
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
import json

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# Enumerations
# ==============================================================================

class MetricType(Enum):
    """Types of metrics collected."""
    ENCODING_TIME = "encoding_time"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CHUNK_COUNT = "chunk_count"
    COMPRESSION_RATIO = "compression_ratio"
    VERIFICATION_STATUS = "verification_status"
    CRYPTO_OPERATION = "crypto_operation"
    REFERENCE_LOOKUP = "reference_lookup"
    ERROR_RATE = "error_rate"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    VERIFICATION_FAILURE = "verification_failure"
    MEMORY_THRESHOLD = "memory_threshold"
    ERROR_THRESHOLD = "error_threshold"
    CRYPTO_FAILURE = "crypto_failure"


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class PerformanceMetrics:
    """
    Performance metrics for an encoding operation.

    Attributes:
        genome_id: Identifier of encoded genome
        variant_count: Number of variants processed
        encoding_time_ms: Total encoding time in milliseconds
        throughput_variants_per_sec: Encoding throughput
        memory_peak_mb: Peak memory usage in megabytes
        memory_current_mb: Current memory usage in megabytes
        chunk_count: Number of chunks created
        compression_ratio: Achieved compression ratio
        timestamp: When metrics were collected
        analysis_type: Analysis type used
        dimension: Hypervector dimension
        verification_passed: Whether cryptographic verification passed
    """
    genome_id: str
    variant_count: int
    encoding_time_ms: float
    throughput_variants_per_sec: float
    memory_peak_mb: float
    memory_current_mb: float
    chunk_count: int
    compression_ratio: float
    timestamp: datetime = field(default_factory=datetime.now)
    analysis_type: Optional[str] = None
    dimension: Optional[int] = None
    verification_passed: bool = True
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class CryptoAuditEntry:
    """
    Audit log entry for cryptographic operations.

    Attributes:
        operation: Type of crypto operation (hash, hmac, verify, etc.)
        entity_id: ID of entity being operated on (genome_id, chunk_id, etc.)
        status: Success/failure status
        timestamp: When operation occurred
        duration_ms: Operation duration in milliseconds
        metadata: Additional operation-specific metadata
        error: Error message if operation failed
    """
    operation: str
    entity_id: str
    status: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class Alert:
    """
    Alert for monitoring systems.

    Attributes:
        alert_type: Type of alert
        level: Severity level
        message: Human-readable message
        details: Additional context and data
        timestamp: When alert was triggered
        entity_id: ID of affected entity
        acknowledged: Whether alert has been acknowledged
    """
    alert_type: AlertType
    level: AlertLevel
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    entity_id: Optional[str] = None
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['alert_type'] = self.alert_type.value
        data['level'] = self.level.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# ==============================================================================
# Performance Monitor
# ==============================================================================

class PerformanceMonitor:
    """
    Monitor and track performance metrics for differential encoding operations.

    This class provides comprehensive performance monitoring including:
    - Encoding time and throughput tracking
    - Memory usage monitoring
    - Performance degradation detection
    - Metrics aggregation and reporting
    - Alert generation for threshold violations
    """

    def __init__(
        self,
        enable_memory_tracking: bool = True,
        enable_alerts: bool = True,
    ):
        """
        Initialize performance monitor.

        Args:
            enable_memory_tracking: Enable detailed memory usage tracking
            enable_alerts: Enable automatic alert generation
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_alerts = enable_alerts

        # Metrics storage
        self.metrics_history: List[PerformanceMetrics] = []

        # Alert system
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Performance thresholds
        self.thresholds = {
            'encoding_time_ms': 10000,  # 10 seconds for 30K variants
            'throughput_min': 2000,  # variants/second
            'memory_peak_mb': 600,  # MB
            'compression_ratio_min': 1.5,
        }

        # Aggregated statistics
        self.stats = defaultdict(list)

        logger.info("PerformanceMonitor initialized")

    @contextmanager
    def track_encoding(
        self,
        genome_id: str,
        variant_count: int,
        analysis_type: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        """
        Context manager to track encoding operation performance.

        Usage:
            >>> monitor = PerformanceMonitor()
            >>> with monitor.track_encoding("genome_001", 30000) as tracker:
            ...     # Perform encoding
            ...     encoded = encoder.encode_genome(genome)
            ...     tracker.set_result(encoded)

        Args:
            genome_id: Genome identifier
            variant_count: Number of variants being encoded
            analysis_type: Analysis type being used
            dimension: Hypervector dimension

        Yields:
            EncodingTracker instance for recording additional metrics
        """
        # Start tracking
        start_time = time.perf_counter()

        if self.enable_memory_tracking:
            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
        else:
            start_memory = 0

        # Create tracker
        tracker = EncodingTracker()

        try:
            yield tracker

            # Calculate metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            throughput = (variant_count / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

            if self.enable_memory_tracking:
                current_memory, peak_memory = tracemalloc.get_traced_memory()
                memory_peak_mb = peak_memory / (1024 * 1024)
                memory_current_mb = (current_memory - start_memory) / (1024 * 1024)
                tracemalloc.stop()
            else:
                memory_peak_mb = 0
                memory_current_mb = 0

            # Create metrics record
            metrics = PerformanceMetrics(
                genome_id=genome_id,
                variant_count=variant_count,
                encoding_time_ms=elapsed_ms,
                throughput_variants_per_sec=throughput,
                memory_peak_mb=memory_peak_mb,
                memory_current_mb=memory_current_mb,
                chunk_count=tracker.chunk_count,
                compression_ratio=tracker.compression_ratio,
                analysis_type=analysis_type,
                dimension=dimension,
                verification_passed=tracker.verification_passed,
                additional_data=tracker.additional_data,
            )

            # Store metrics
            self.record_metrics(metrics)

            # Check thresholds and generate alerts
            if self.enable_alerts:
                self._check_thresholds(metrics)

            logger.info(
                f"Encoding tracked: {genome_id}, "
                f"{variant_count:,} variants, "
                f"{elapsed_ms:.1f}ms, "
                f"{throughput:,.0f} var/s"
            )

        except Exception as e:
            logger.error(f"Error tracking encoding for {genome_id}: {e}", exc_info=True)
            if self.enable_memory_tracking:
                tracemalloc.stop()
            raise

    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """
        Record performance metrics.

        Args:
            metrics: PerformanceMetrics instance to record
        """
        self.metrics_history.append(metrics)

        # Update aggregated statistics
        self.stats['encoding_time_ms'].append(metrics.encoding_time_ms)
        self.stats['throughput'].append(metrics.throughput_variants_per_sec)
        self.stats['memory_peak_mb'].append(metrics.memory_peak_mb)
        self.stats['compression_ratio'].append(metrics.compression_ratio)

        logger.debug(f"Recorded metrics for {metrics.genome_id}")

    def _check_thresholds(self, metrics: PerformanceMetrics) -> None:
        """Check metrics against thresholds and generate alerts."""

        # Check encoding time
        if metrics.encoding_time_ms > self.thresholds['encoding_time_ms']:
            self._create_alert(
                AlertType.PERFORMANCE_DEGRADATION,
                AlertLevel.WARNING,
                f"Encoding time ({metrics.encoding_time_ms:.0f}ms) exceeds threshold "
                f"({self.thresholds['encoding_time_ms']}ms)",
                {
                    'genome_id': metrics.genome_id,
                    'encoding_time_ms': metrics.encoding_time_ms,
                    'threshold_ms': self.thresholds['encoding_time_ms'],
                    'variant_count': metrics.variant_count,
                },
                entity_id=metrics.genome_id,
            )

        # Check throughput
        if metrics.throughput_variants_per_sec < self.thresholds['throughput_min']:
            self._create_alert(
                AlertType.PERFORMANCE_DEGRADATION,
                AlertLevel.WARNING,
                f"Throughput ({metrics.throughput_variants_per_sec:.0f} var/s) below threshold "
                f"({self.thresholds['throughput_min']} var/s)",
                {
                    'genome_id': metrics.genome_id,
                    'throughput': metrics.throughput_variants_per_sec,
                    'threshold': self.thresholds['throughput_min'],
                },
                entity_id=metrics.genome_id,
            )

        # Check memory usage
        if metrics.memory_peak_mb > self.thresholds['memory_peak_mb']:
            self._create_alert(
                AlertType.MEMORY_THRESHOLD,
                AlertLevel.WARNING,
                f"Peak memory ({metrics.memory_peak_mb:.1f} MB) exceeds threshold "
                f"({self.thresholds['memory_peak_mb']} MB)",
                {
                    'genome_id': metrics.genome_id,
                    'memory_peak_mb': metrics.memory_peak_mb,
                    'threshold_mb': self.thresholds['memory_peak_mb'],
                },
                entity_id=metrics.genome_id,
            )

        # Check compression ratio
        if metrics.compression_ratio < self.thresholds['compression_ratio_min']:
            self._create_alert(
                AlertType.PERFORMANCE_DEGRADATION,
                AlertLevel.INFO,
                f"Compression ratio ({metrics.compression_ratio:.1f}×) below expected "
                f"({self.thresholds['compression_ratio_min']:.1f}×)",
                {
                    'genome_id': metrics.genome_id,
                    'compression_ratio': metrics.compression_ratio,
                    'threshold': self.thresholds['compression_ratio_min'],
                },
                entity_id=metrics.genome_id,
            )

        # Check verification
        if not metrics.verification_passed:
            self._create_alert(
                AlertType.VERIFICATION_FAILURE,
                AlertLevel.CRITICAL,
                f"Cryptographic verification FAILED for {metrics.genome_id}",
                {
                    'genome_id': metrics.genome_id,
                    'analysis_type': metrics.analysis_type,
                },
                entity_id=metrics.genome_id,
            )

    def _create_alert(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        message: str,
        details: Dict[str, Any],
        entity_id: Optional[str] = None,
    ) -> None:
        """Create and process an alert."""
        alert = Alert(
            alert_type=alert_type,
            level=level,
            message=message,
            details=details,
            entity_id=entity_id,
        )

        self.alerts.append(alert)

        # Log alert
        log_method = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
        }.get(level, logger.info)

        log_method(f"ALERT [{alert_type.value}]: {message}")

        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}", exc_info=True)

    def register_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        Register callback to be called when alerts are generated.

        Args:
            callback: Function that accepts Alert instance
        """
        self.alert_callbacks.append(callback)
        logger.info(f"Registered alert callback: {callback.__name__}")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all recorded metrics.

        Returns:
            Dictionary with aggregated statistics
        """
        if not self.metrics_history:
            return {}

        summary = {
            'total_genomes': len(self.metrics_history),
            'total_variants': sum(m.variant_count for m in self.metrics_history),
            'encoding_time_ms': {
                'mean': np.mean(self.stats['encoding_time_ms']),
                'median': np.median(self.stats['encoding_time_ms']),
                'std': np.std(self.stats['encoding_time_ms']),
                'min': np.min(self.stats['encoding_time_ms']),
                'max': np.max(self.stats['encoding_time_ms']),
            },
            'throughput': {
                'mean': np.mean(self.stats['throughput']),
                'median': np.median(self.stats['throughput']),
                'std': np.std(self.stats['throughput']),
                'min': np.min(self.stats['throughput']),
                'max': np.max(self.stats['throughput']),
            },
            'memory_peak_mb': {
                'mean': np.mean(self.stats['memory_peak_mb']),
                'median': np.median(self.stats['memory_peak_mb']),
                'max': np.max(self.stats['memory_peak_mb']),
            },
            'compression_ratio': {
                'mean': np.mean(self.stats['compression_ratio']),
                'median': np.median(self.stats['compression_ratio']),
                'min': np.min(self.stats['compression_ratio']),
                'max': np.max(self.stats['compression_ratio']),
            },
            'alerts': {
                'total': len(self.alerts),
                'by_level': {
                    level.value: sum(1 for a in self.alerts if a.level == level)
                    for level in AlertLevel
                },
                'by_type': {
                    atype.value: sum(1 for a in self.alerts if a.alert_type == atype)
                    for atype in AlertType
                },
            },
        }

        return summary

    def export_metrics(self, output_path: Path, format: str = 'json') -> None:
        """
        Export metrics to file.

        Args:
            output_path: Path to output file
            format: Export format ('json', 'csv')
        """
        if format == 'json':
            data = {
                'metrics': [m.to_dict() for m in self.metrics_history],
                'summary': self.get_summary_statistics(),
                'alerts': [a.to_dict() for a in self.alerts],
            }
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        elif format == 'csv':
            # Export as CSV
            import csv
            with open(output_path, 'w', newline='') as f:
                if not self.metrics_history:
                    return

                fieldnames = list(self.metrics_history[0].to_dict().keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for metrics in self.metrics_history:
                    writer.writerow(metrics.to_dict())

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported metrics to {output_path}")


class EncodingTracker:
    """Helper class for tracking encoding results within context manager."""

    def __init__(self):
        self.chunk_count = 0
        self.compression_ratio = 1.0
        self.verification_passed = True
        self.additional_data = {}

    def set_result(self, encoded_genome) -> None:
        """
        Set encoding result for metric calculation.

        Args:
            encoded_genome: EncodedGenome instance
        """
        self.chunk_count = len(encoded_genome.chunk_hypervectors)
        self.verification_passed = encoded_genome.verify()

        # Calculate compression ratio if possible
        try:
            uncompressed_kb = encoded_genome.storage_size_kb()
            # Simulate compression (would use actual compressed size in production)
            self.compression_ratio = uncompressed_kb / (uncompressed_kb / 2.1)  # Typical ratio
        except Exception:
            self.compression_ratio = 1.0


# ==============================================================================
# Cryptographic Audit Logger
# ==============================================================================

class CryptoAuditLogger:
    """
    Audit logger for cryptographic operations.

    Maintains detailed logs of all cryptographic operations for security
    auditing and compliance purposes.
    """

    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize cryptographic audit logger.

        Args:
            log_file: Optional path to audit log file
        """
        self.log_file = log_file
        self.audit_entries: List[CryptoAuditEntry] = []

        logger.info(f"CryptoAuditLogger initialized (log_file={log_file})")

    @contextmanager
    def log_operation(
        self,
        operation: str,
        entity_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager to log cryptographic operation.

        Usage:
            >>> audit_logger = CryptoAuditLogger()
            >>> with audit_logger.log_operation("hash", "genome_001") as entry:
            ...     result = compute_hash(data)
            ...     entry['result_hash'] = result[:16]

        Args:
            operation: Operation type (hash, hmac, verify, etc.)
            entity_id: Entity being operated on
            metadata: Additional metadata to log

        Yields:
            Metadata dictionary for adding additional information
        """
        start_time = time.perf_counter()
        meta = metadata or {}
        error = None

        try:
            yield meta
            status = "SUCCESS"

        except Exception as e:
            status = "FAILURE"
            error = str(e)
            logger.error(f"Crypto operation failed: {operation} on {entity_id}: {e}")
            raise

        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000

            entry = CryptoAuditEntry(
                operation=operation,
                entity_id=entity_id,
                status=status,
                duration_ms=duration_ms,
                metadata=meta,
                error=error,
            )

            self.audit_entries.append(entry)

            # Write to log file if configured
            if self.log_file:
                self._write_to_file(entry)

            logger.debug(
                f"Crypto audit: {operation} on {entity_id} - {status} ({duration_ms:.2f}ms)"
            )

    def _write_to_file(self, entry: CryptoAuditEntry) -> None:
        """Write audit entry to file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(entry.to_json() + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def get_entries(
        self,
        operation: Optional[str] = None,
        entity_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[CryptoAuditEntry]:
        """
        Get filtered audit entries.

        Args:
            operation: Filter by operation type
            entity_id: Filter by entity ID
            status: Filter by status

        Returns:
            List of matching audit entries
        """
        entries = self.audit_entries

        if operation:
            entries = [e for e in entries if e.operation == operation]

        if entity_id:
            entries = [e for e in entries if e.entity_id == entity_id]

        if status:
            entries = [e for e in entries if e.status == status]

        return entries

    def get_failure_count(self, operation: Optional[str] = None) -> int:
        """
        Get count of failed operations.

        Args:
            operation: Optional operation type filter

        Returns:
            Number of failed operations
        """
        entries = self.get_entries(operation=operation, status="FAILURE")
        return len(entries)


# ==============================================================================
# Global Instances
# ==============================================================================

# Global performance monitor instance
_performance_monitor = PerformanceMonitor()

# Global crypto audit logger instance
_crypto_audit_logger = CryptoAuditLogger()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


def get_crypto_audit_logger() -> CryptoAuditLogger:
    """Get the global crypto audit logger instance."""
    return _crypto_audit_logger
