"""
GenomeVault Monitoring System

Implements comprehensive monitoring with Prometheus exporters, performance tracking,
alerting rules, and Grafana dashboards as specified in Section 7.2.2 and Appendix C.

Key features:
- Custom metrics for HDC, PIR, ZK proof operations
- Performance target tracking from Appendix C
- Alerting rules for latency, privacy, and system health
- Grafana dashboard configurations
"""

from __future__ import annotations

import json
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from functools import wraps

import numpy as np

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
        Info,
        Enum as PrometheusEnum,
    )
    HAS_PROMETHEUS = True
except ImportError:
    logger.warning("prometheus_client not available, using simulation mode")
    HAS_PROMETHEUS = False
    # Mock classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, amount=1): pass
        def labels(self, **kwargs): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, value): pass
        def inc(self, amount=1): pass
        def dec(self, amount=1): pass
        def labels(self, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, value): pass
        def time(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def labels(self, **kwargs): return self
    
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, value): pass
        def time(self): return self
        def labels(self, **kwargs): return self
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, value): pass
    
    class PrometheusEnum:
        def __init__(self, *args, **kwargs): pass
        def state(self, value): pass
    
    class CollectorRegistry:
        def __init__(self): pass
    
    def generate_latest(registry=None):
        return b"# No metrics available (prometheus_client not installed)"
    
    CONTENT_TYPE_LATEST = "text/plain"


class MetricType(Enum):
    """Types of metrics being tracked"""
    HDC_OPERATION = "hdc_operation"
    PIR_QUERY = "pir_query"
    ZK_PROOF = "zk_proof"
    COMPRESSION = "compression"
    PRIVACY_BUDGET = "privacy_budget"
    NODE_VOTING = "node_voting"
    STORAGE = "storage"
    LATENCY = "latency"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceTarget:
    """Performance target from Appendix C"""
    metric_name: str
    target_value: float
    unit: str
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    name: str
    severity: AlertSeverity
    condition: str
    threshold: float
    message: str
    cooldown_minutes: int = 5
    last_fired: Optional[datetime] = None
    
    def should_fire(self, current_value: float) -> bool:
        """Check if alert should fire based on condition"""
        if self.last_fired:
            cooldown_end = self.last_fired + timedelta(minutes=self.cooldown_minutes)
            if datetime.now() < cooldown_end:
                return False
        
        # Evaluate condition
        if ">" in self.condition:
            return current_value > self.threshold
        elif "<" in self.condition:
            return current_value < self.threshold
        elif ">=" in self.condition:
            return current_value >= self.threshold
        elif "<=" in self.condition:
            return current_value <= self.threshold
        elif "==" in self.condition:
            return abs(current_value - self.threshold) < 0.001
        
        return False


class PrometheusExporter:
    """Prometheus metrics exporter for GenomeVault"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize Prometheus exporter
        
        Args:
            registry: Prometheus collector registry
        """
        self.registry = registry or CollectorRegistry()
        
        # Initialize metrics
        self._init_hdc_metrics()
        self._init_pir_metrics()
        self._init_zk_metrics()
        self._init_compression_metrics()
        self._init_privacy_metrics()
        self._init_node_metrics()
        self._init_storage_metrics()
        self._init_system_metrics()
    
    def _init_hdc_metrics(self):
        """Initialize HDC operation metrics"""
        self.hdc_operations_total = Counter(
            'genomevault_hdc_operations_total',
            'Total number of HDC operations',
            ['operation_type', 'status'],
            registry=self.registry
        )
        
        self.hdc_encoding_duration = Histogram(
            'genomevault_hdc_encoding_duration_seconds',
            'HDC encoding duration in seconds',
            ['dimension', 'data_type'],
            buckets=[0.1, 0.5, 1, 5, 10, 30, 60],
            registry=self.registry
        )
        
        self.hdc_similarity_computation = Summary(
            'genomevault_hdc_similarity_computation_seconds',
            'HDC similarity computation time',
            ['metric_type'],
            registry=self.registry
        )
        
        self.hypervector_dimension = Gauge(
            'genomevault_hypervector_dimension',
            'Current hypervector dimension',
            registry=self.registry
        )
        
        self.hdc_memory_usage_bytes = Gauge(
            'genomevault_hdc_memory_usage_bytes',
            'Memory usage for HDC operations',
            registry=self.registry
        )
    
    def _init_pir_metrics(self):
        """Initialize PIR query metrics"""
        self.pir_queries_total = Counter(
            'genomevault_pir_queries_total',
            'Total number of PIR queries',
            ['server', 'status'],
            registry=self.registry
        )
        
        self.pir_query_latency = Histogram(
            'genomevault_pir_query_latency_milliseconds',
            'PIR query latency in milliseconds',
            ['server', 'query_type'],
            buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000],
            registry=self.registry
        )
        
        self.pir_server_availability = Gauge(
            'genomevault_pir_server_availability',
            'PIR server availability (0=down, 1=up)',
            ['server'],
            registry=self.registry
        )
        
        self.pir_query_size_bytes = Histogram(
            'genomevault_pir_query_size_bytes',
            'PIR query size in bytes',
            buckets=[100, 500, 1000, 5000, 10000, 50000],
            registry=self.registry
        )
        
        self.pir_response_size_bytes = Histogram(
            'genomevault_pir_response_size_bytes',
            'PIR response size in bytes',
            buckets=[100, 500, 1000, 5000, 10000, 50000, 100000],
            registry=self.registry
        )
    
    def _init_zk_metrics(self):
        """Initialize zero-knowledge proof metrics"""
        self.zk_proofs_generated = Counter(
            'genomevault_zk_proofs_generated_total',
            'Total ZK proofs generated',
            ['circuit_type', 'status'],
            registry=self.registry
        )
        
        self.zk_proof_generation_time = Histogram(
            'genomevault_zk_proof_generation_seconds',
            'ZK proof generation time in seconds',
            ['circuit_type', 'hardware'],
            buckets=[1, 5, 10, 15, 30, 60, 120],
            registry=self.registry
        )
        
        self.zk_proof_verification_time = Histogram(
            'genomevault_zk_proof_verification_milliseconds',
            'ZK proof verification time in milliseconds',
            ['circuit_type'],
            buckets=[1, 5, 10, 50, 100, 500],
            registry=self.registry
        )
        
        self.zk_circuit_constraints = Gauge(
            'genomevault_zk_circuit_constraints',
            'Number of constraints in ZK circuit',
            ['circuit_type'],
            registry=self.registry
        )
        
        self.zk_proof_size_bytes = Gauge(
            'genomevault_zk_proof_size_bytes',
            'ZK proof size in bytes',
            ['circuit_type'],
            registry=self.registry
        )
    
    def _init_compression_metrics(self):
        """Initialize compression metrics"""
        self.compression_ratio = Gauge(
            'genomevault_compression_ratio',
            'Compression ratio achieved',
            ['tier', 'data_type'],
            registry=self.registry
        )
        
        self.compression_operations = Counter(
            'genomevault_compression_operations_total',
            'Total compression operations',
            ['tier', 'operation', 'status'],
            registry=self.registry
        )
        
        self.compressed_size_bytes = Histogram(
            'genomevault_compressed_size_bytes',
            'Compressed data size in bytes',
            ['tier'],
            buckets=[1000, 10000, 100000, 1000000, 10000000, 100000000],
            registry=self.registry
        )
        
        self.compression_time = Histogram(
            'genomevault_compression_time_seconds',
            'Compression operation time',
            ['tier', 'operation'],
            buckets=[0.01, 0.1, 0.5, 1, 5, 10],
            registry=self.registry
        )
        
        self.compression_tier_usage = Gauge(
            'genomevault_compression_tier_usage',
            'Number of profiles using each compression tier',
            ['tier'],
            registry=self.registry
        )
    
    def _init_privacy_metrics(self):
        """Initialize privacy metrics"""
        self.privacy_budget_consumed = Gauge(
            'genomevault_privacy_budget_consumed',
            'Privacy budget consumed (epsilon)',
            ['user', 'operation'],
            registry=self.registry
        )
        
        self.privacy_budget_remaining = Gauge(
            'genomevault_privacy_budget_remaining',
            'Privacy budget remaining (epsilon)',
            ['user'],
            registry=self.registry
        )
        
        self.privacy_breach_probability = Gauge(
            'genomevault_privacy_breach_probability',
            'Estimated privacy breach probability',
            ['method'],
            registry=self.registry
        )
        
        self.differential_privacy_noise = Histogram(
            'genomevault_differential_privacy_noise',
            'Differential privacy noise added',
            ['mechanism'],
            buckets=[0.001, 0.01, 0.1, 1, 10],
            registry=self.registry
        )
        
        self.privacy_violations = Counter(
            'genomevault_privacy_violations_total',
            'Total privacy violation attempts detected',
            ['violation_type'],
            registry=self.registry
        )
    
    def _init_node_metrics(self):
        """Initialize blockchain node metrics"""
        self.node_voting_weight = Gauge(
            'genomevault_node_voting_weight',
            'Node voting weight in consensus',
            ['node_id', 'node_class'],
            registry=self.registry
        )
        
        self.node_participation_rate = Gauge(
            'genomevault_node_participation_rate',
            'Node participation rate in consensus',
            ['node_id'],
            registry=self.registry
        )
        
        self.consensus_rounds = Counter(
            'genomevault_consensus_rounds_total',
            'Total consensus rounds',
            ['status'],
            registry=self.registry
        )
        
        self.block_generation_time = Histogram(
            'genomevault_block_generation_seconds',
            'Block generation time',
            buckets=[1, 2, 5, 10, 30],
            registry=self.registry
        )
        
        self.credit_transactions = Counter(
            'genomevault_credit_transactions_total',
            'Total credit transactions',
            ['transaction_type'],
            registry=self.registry
        )
        
        self.credit_balance = Gauge(
            'genomevault_credit_balance',
            'Credit balance by user',
            ['user_type'],
            registry=self.registry
        )
    
    def _init_storage_metrics(self):
        """Initialize storage metrics"""
        self.storage_usage_bytes = Gauge(
            'genomevault_storage_usage_bytes',
            'Storage usage in bytes',
            ['storage_type', 'tier'],
            registry=self.registry
        )
        
        self.profile_storage_size = Histogram(
            'genomevault_profile_storage_gigabytes',
            'Profile storage size in GB',
            buckets=[1, 2, 5, 10, 20, 50, 100],
            registry=self.registry
        )
        
        self.storage_operations = Counter(
            'genomevault_storage_operations_total',
            'Total storage operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.storage_latency = Histogram(
            'genomevault_storage_latency_milliseconds',
            'Storage operation latency',
            ['operation'],
            buckets=[1, 5, 10, 50, 100, 500, 1000],
            registry=self.registry
        )
    
    def _init_system_metrics(self):
        """Initialize system-wide metrics"""
        self.system_uptime = Gauge(
            'genomevault_system_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )
        
        self.active_users = Gauge(
            'genomevault_active_users',
            'Number of active users',
            registry=self.registry
        )
        
        self.api_requests = Counter(
            'genomevault_api_requests_total',
            'Total API requests',
            ['endpoint', 'method', 'status_code'],
            registry=self.registry
        )
        
        self.api_latency = Histogram(
            'genomevault_api_latency_milliseconds',
            'API request latency',
            ['endpoint', 'method'],
            buckets=[10, 50, 100, 500, 1000, 5000],
            registry=self.registry
        )
        
        self.error_rate = Gauge(
            'genomevault_error_rate',
            'System error rate (errors per minute)',
            registry=self.registry
        )


class PerformanceMonitor:
    """Monitor performance against Appendix C targets"""
    
    def __init__(self):
        """Initialize performance monitor with targets from Appendix C"""
        self.targets = self._init_performance_targets()
        self.measurements: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.violations: Dict[str, int] = defaultdict(int)
    
    def _init_performance_targets(self) -> Dict[str, PerformanceTarget]:
        """Initialize performance targets from Appendix C"""
        return {
            "pir_query_latency": PerformanceTarget(
                metric_name="pir_query_latency",
                target_value=300,  # milliseconds (middle of 100-500ms range)
                unit="ms",
                description="PIR query latency",
                min_value=100,
                max_value=500
            ),
            "zk_proof_generation": PerformanceTarget(
                metric_name="zk_proof_generation",
                target_value=15,  # seconds (high-end hardware)
                unit="seconds",
                description="ZK proof generation time",
                max_value=15
            ),
            "hypervector_generation": PerformanceTarget(
                metric_name="hypervector_generation",
                target_value=30,  # seconds
                unit="seconds",
                description="Hypervector generation time",
                max_value=30
            ),
            "profile_storage": PerformanceTarget(
                metric_name="profile_storage",
                target_value=7.5,  # GB (middle of 5-10GB range)
                unit="GB",
                description="Storage per genomic profile",
                min_value=5,
                max_value=10
            ),
            "compression_ratio_mini": PerformanceTarget(
                metric_name="compression_ratio_mini",
                target_value=4000,  # 100MB -> 25KB
                unit="ratio",
                description="Mini tier compression ratio",
                min_value=4000
            ),
            "compression_ratio_clinical": PerformanceTarget(
                metric_name="compression_ratio_clinical",
                target_value=333,  # 100MB -> 300KB
                unit="ratio",
                description="Clinical tier compression ratio",
                min_value=333
            ),
            "compression_ratio_full": PerformanceTarget(
                metric_name="compression_ratio_full",
                target_value=13.3,  # 100MB -> 7.5MB
                unit="ratio",
                description="Full HDC tier compression ratio",
                min_value=13.3
            ),
            "api_latency_p99": PerformanceTarget(
                metric_name="api_latency_p99",
                target_value=1000,  # milliseconds
                unit="ms",
                description="API 99th percentile latency",
                max_value=1000
            ),
            "consensus_time": PerformanceTarget(
                metric_name="consensus_time",
                target_value=5,  # seconds
                unit="seconds",
                description="Blockchain consensus time",
                max_value=5
            ),
            "privacy_breach_probability": PerformanceTarget(
                metric_name="privacy_breach_probability",
                target_value=0.0001,  # 10^-4
                unit="probability",
                description="Maximum privacy breach probability",
                max_value=0.0001
            )
        }
    
    def record_measurement(self, metric_name: str, value: float) -> bool:
        """
        Record a performance measurement
        
        Args:
            metric_name: Name of the metric
            value: Measured value
            
        Returns:
            True if within target, False if violation
        """
        if metric_name not in self.targets:
            logger.warning(f"Unknown metric: {metric_name}")
            return True
        
        target = self.targets[metric_name]
        self.measurements[metric_name].append(value)
        
        # Check against target
        within_target = True
        if target.min_value is not None and value < target.min_value:
            within_target = False
        if target.max_value is not None and value > target.max_value:
            within_target = False
        
        if not within_target:
            self.violations[metric_name] += 1
            logger.warning(
                f"Performance violation: {metric_name}={value}{target.unit} "
                f"(target: {target.min_value}-{target.max_value}{target.unit})"
            )
        
        return within_target
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if metric_name not in self.measurements:
            return {}
        
        values = list(self.measurements[metric_name])
        if not values:
            return {}
        
        return {
            "current": values[-1] if values else 0,
            "mean": np.mean(values),
            "median": np.median(values),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
            "min": np.min(values),
            "max": np.max(values),
            "violations": self.violations[metric_name]
        }
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report against targets"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        for metric_name, target in self.targets.items():
            stats = self.get_statistics(metric_name)
            if stats:
                compliance = True
                if target.max_value and stats.get("p99", 0) > target.max_value:
                    compliance = False
                if target.min_value and stats.get("p99", float('inf')) < target.min_value:
                    compliance = False
                
                report["metrics"][metric_name] = {
                    "target": {
                        "value": target.target_value,
                        "min": target.min_value,
                        "max": target.max_value,
                        "unit": target.unit
                    },
                    "measured": stats,
                    "compliant": compliance
                }
        
        return report


class AlertManager:
    """Manage system alerts and notifications"""
    
    def __init__(self):
        """Initialize alert manager"""
        self.alerts = self._init_alerts()
        self.active_alerts: Set[str] = set()
        self.alert_history: List[Dict[str, Any]] = []
    
    def _init_alerts(self) -> Dict[str, Alert]:
        """Initialize alert rules"""
        return {
            "high_latency": Alert(
                alert_id="high_latency",
                name="High Latency Alert",
                severity=AlertSeverity.WARNING,
                condition="latency > 2× expected",
                threshold=2.0,  # multiplier
                message="System latency exceeds 2× expected value"
            ),
            "privacy_breach_risk": Alert(
                alert_id="privacy_breach_risk",
                name="Privacy Breach Risk",
                severity=AlertSeverity.CRITICAL,
                condition="probability > 10^-4",
                threshold=0.0001,
                message="Privacy breach probability exceeds 10^-4"
            ),
            "low_compression": Alert(
                alert_id="low_compression",
                name="Low Compression Ratio",
                severity=AlertSeverity.WARNING,
                condition="ratio < target",
                threshold=0.8,  # 80% of target
                message="Compression ratio below target"
            ),
            "voting_imbalance": Alert(
                alert_id="voting_imbalance",
                name="Node Voting Weight Imbalance",
                severity=AlertSeverity.WARNING,
                condition="gini > 0.5",
                threshold=0.5,  # Gini coefficient
                message="Node voting weight distribution is imbalanced"
            ),
            "storage_critical": Alert(
                alert_id="storage_critical",
                name="Storage Space Critical",
                severity=AlertSeverity.CRITICAL,
                condition="usage > 90%",
                threshold=0.9,
                message="Storage usage exceeds 90%"
            ),
            "zk_proof_timeout": Alert(
                alert_id="zk_proof_timeout",
                name="ZK Proof Generation Timeout",
                severity=AlertSeverity.WARNING,
                condition="time > 30s",
                threshold=30,
                message="ZK proof generation exceeds 30 seconds"
            ),
            "api_error_rate": Alert(
                alert_id="api_error_rate",
                name="High API Error Rate",
                severity=AlertSeverity.CRITICAL,
                condition="error_rate > 5%",
                threshold=0.05,
                message="API error rate exceeds 5%"
            ),
            "consensus_failure": Alert(
                alert_id="consensus_failure",
                name="Consensus Failure",
                severity=AlertSeverity.EMERGENCY,
                condition="failures > 3",
                threshold=3,
                message="Multiple consensus failures detected"
            )
        }
    
    def check_alert(self, alert_id: str, current_value: float) -> Optional[Dict[str, Any]]:
        """
        Check if an alert should fire
        
        Args:
            alert_id: Alert identifier
            current_value: Current metric value
            
        Returns:
            Alert details if fired, None otherwise
        """
        if alert_id not in self.alerts:
            return None
        
        alert = self.alerts[alert_id]
        
        if alert.should_fire(current_value):
            alert.last_fired = datetime.now()
            self.active_alerts.add(alert_id)
            
            alert_details = {
                "alert_id": alert_id,
                "name": alert.name,
                "severity": alert.severity.value,
                "message": alert.message,
                "current_value": current_value,
                "threshold": alert.threshold,
                "timestamp": datetime.now().isoformat()
            }
            
            self.alert_history.append(alert_details)
            logger.warning(f"ALERT: {alert.name} - {alert.message}")
            
            return alert_details
        
        # Clear alert if condition no longer met
        if alert_id in self.active_alerts:
            self.active_alerts.remove(alert_id)
        
        return None
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        active = []
        for alert_id in self.active_alerts:
            alert = self.alerts[alert_id]
            active.append({
                "alert_id": alert_id,
                "name": alert.name,
                "severity": alert.severity.value,
                "message": alert.message
            })
        return active


class GrafanaDashboard:
    """Grafana dashboard configuration generator"""
    
    def __init__(self):
        """Initialize dashboard generator"""
        self.dashboards = {
            "system_overview": self._create_system_overview(),
            "privacy_monitoring": self._create_privacy_dashboard(),
            "network_topology": self._create_network_dashboard(),
            "credit_economy": self._create_credit_dashboard()
        }
    
    def _create_system_overview(self) -> Dict[str, Any]:
        """Create system overview dashboard"""
        return {
            "title": "GenomeVault System Overview",
            "panels": [
                {
                    "title": "HDC Operations",
                    "type": "graph",
                    "targets": [
                        {"expr": "rate(genomevault_hdc_operations_total[5m])"},
                        {"expr": "histogram_quantile(0.99, genomevault_hdc_encoding_duration_seconds)"}
                    ]
                },
                {
                    "title": "PIR Query Latency",
                    "type": "graph",
                    "targets": [
                        {"expr": "histogram_quantile(0.5, genomevault_pir_query_latency_milliseconds)"},
                        {"expr": "histogram_quantile(0.99, genomevault_pir_query_latency_milliseconds)"}
                    ],
                    "alert": {
                        "condition": "avg() > 500",
                        "message": "PIR latency exceeds target"
                    }
                },
                {
                    "title": "ZK Proof Generation Time",
                    "type": "graph",
                    "targets": [
                        {"expr": "histogram_quantile(0.99, genomevault_zk_proof_generation_seconds)"}
                    ],
                    "thresholds": [
                        {"value": 15, "color": "yellow", "label": "Target"},
                        {"value": 30, "color": "red", "label": "Critical"}
                    ]
                },
                {
                    "title": "Compression Ratios by Tier",
                    "type": "bar",
                    "targets": [
                        {"expr": 'genomevault_compression_ratio{tier="mini"}'},
                        {"expr": 'genomevault_compression_ratio{tier="clinical"}'},
                        {"expr": 'genomevault_compression_ratio{tier="full"}'}
                    ]
                },
                {
                    "title": "Storage Usage",
                    "type": "stat",
                    "targets": [
                        {"expr": "sum(genomevault_storage_usage_bytes) / 1e9"}  # Convert to GB
                    ],
                    "unit": "GB"
                },
                {
                    "title": "API Request Rate",
                    "type": "graph",
                    "targets": [
                        {"expr": "rate(genomevault_api_requests_total[1m])"}
                    ]
                }
            ]
        }
    
    def _create_privacy_dashboard(self) -> Dict[str, Any]:
        """Create privacy monitoring dashboard"""
        return {
            "title": "Privacy Budget Monitoring",
            "panels": [
                {
                    "title": "Privacy Budget Consumption",
                    "type": "gauge",
                    "targets": [
                        {"expr": "avg(genomevault_privacy_budget_consumed)"},
                        {"expr": "avg(genomevault_privacy_budget_remaining)"}
                    ],
                    "thresholds": [
                        {"value": 0.5, "color": "green"},
                        {"value": 0.8, "color": "yellow"},
                        {"value": 0.95, "color": "red"}
                    ]
                },
                {
                    "title": "Privacy Breach Probability",
                    "type": "graph",
                    "targets": [
                        {"expr": "genomevault_privacy_breach_probability"}
                    ],
                    "alert": {
                        "condition": "max() > 0.0001",
                        "message": "Privacy breach risk exceeds threshold"
                    }
                },
                {
                    "title": "Differential Privacy Noise Distribution",
                    "type": "heatmap",
                    "targets": [
                        {"expr": "genomevault_differential_privacy_noise"}
                    ]
                },
                {
                    "title": "Privacy Violations",
                    "type": "counter",
                    "targets": [
                        {"expr": "sum(genomevault_privacy_violations_total)"}
                    ]
                }
            ]
        }
    
    def _create_network_dashboard(self) -> Dict[str, Any]:
        """Create network topology dashboard"""
        return {
            "title": "Network Topology & Consensus",
            "panels": [
                {
                    "title": "Node Voting Weights",
                    "type": "pie",
                    "targets": [
                        {"expr": "genomevault_node_voting_weight"}
                    ]
                },
                {
                    "title": "Node Participation Rate",
                    "type": "bar",
                    "targets": [
                        {"expr": "genomevault_node_participation_rate"}
                    ]
                },
                {
                    "title": "Consensus Performance",
                    "type": "graph",
                    "targets": [
                        {"expr": "histogram_quantile(0.99, genomevault_block_generation_seconds)"},
                        {"expr": "rate(genomevault_consensus_rounds_total[5m])"}
                    ]
                },
                {
                    "title": "Network Topology Map",
                    "type": "nodeGraph",
                    "targets": [
                        {"expr": "genomevault_node_voting_weight"}
                    ]
                }
            ]
        }
    
    def _create_credit_dashboard(self) -> Dict[str, Any]:
        """Create credit economy dashboard"""
        return {
            "title": "Credit Economy Statistics",
            "panels": [
                {
                    "title": "Credit Transactions",
                    "type": "graph",
                    "targets": [
                        {"expr": "rate(genomevault_credit_transactions_total[5m])"}
                    ]
                },
                {
                    "title": "Credit Balance Distribution",
                    "type": "histogram",
                    "targets": [
                        {"expr": "genomevault_credit_balance"}
                    ]
                },
                {
                    "title": "Transaction Types",
                    "type": "pie",
                    "targets": [
                        {"expr": 'sum by(transaction_type) (genomevault_credit_transactions_total)'}
                    ]
                },
                {
                    "title": "Credit Flow",
                    "type": "sankey",
                    "targets": [
                        {"expr": "genomevault_credit_transactions_total"}
                    ]
                }
            ]
        }
    
    def export_dashboard(self, dashboard_name: str) -> str:
        """Export dashboard configuration as JSON"""
        if dashboard_name not in self.dashboards:
            return "{}"
        
        return json.dumps(self.dashboards[dashboard_name], indent=2)
    
    def get_all_dashboards(self) -> List[str]:
        """Get list of all available dashboards"""
        return list(self.dashboards.keys())


class MonitoringSystem:
    """Main monitoring system orchestrator"""
    
    def __init__(self):
        """Initialize monitoring system"""
        self.exporter = PrometheusExporter()
        self.performance_monitor = PerformanceMonitor()
        self.alert_manager = AlertManager()
        self.dashboards = GrafanaDashboard()
        
        # Start time for uptime tracking
        self.start_time = time.time()
        
        # Simulated metrics for demonstration
        self._start_metric_simulation()
        
        logger.info("Monitoring system initialized")
    
    def _start_metric_simulation(self):
        """Start background thread to simulate metrics"""
        def simulate():
            while True:
                # Simulate HDC operations
                self.exporter.hdc_operations_total.labels(
                    operation_type="encode", status="success"
                ).inc()
                self.exporter.hdc_encoding_duration.labels(
                    dimension="10000", data_type="genomic"
                ).observe(np.random.uniform(5, 35))
                
                # Simulate PIR queries
                latency = np.random.uniform(80, 520)
                self.exporter.pir_query_latency.labels(
                    server="server1", query_type="standard"
                ).observe(latency)
                self.performance_monitor.record_measurement("pir_query_latency", latency)
                
                # Simulate ZK proofs
                proof_time = np.random.uniform(10, 20)
                self.exporter.zk_proof_generation_time.labels(
                    circuit_type="variant", hardware="gpu"
                ).observe(proof_time)
                self.performance_monitor.record_measurement("zk_proof_generation", proof_time)
                
                # Simulate compression
                compression_ratio = np.random.uniform(3000, 5000)
                self.exporter.compression_ratio.labels(
                    tier="mini", data_type="genomic"
                ).set(compression_ratio)
                self.performance_monitor.record_measurement("compression_ratio_mini", compression_ratio)
                
                # Check alerts
                self.alert_manager.check_alert("high_latency", latency / 300)  # Compare to target
                self.alert_manager.check_alert("privacy_breach_risk", np.random.uniform(0, 0.0002))
                
                # Update system uptime
                self.exporter.system_uptime.set(time.time() - self.start_time)
                
                time.sleep(1)  # Update every second
        
        thread = threading.Thread(target=simulate, daemon=True)
        thread.start()
    
    def record_hdc_operation(
        self,
        operation_type: str,
        dimension: int,
        duration: float,
        success: bool = True
    ):
        """Record HDC operation metrics"""
        status = "success" if success else "failure"
        self.exporter.hdc_operations_total.labels(
            operation_type=operation_type, status=status
        ).inc()
        
        self.exporter.hdc_encoding_duration.labels(
            dimension=str(dimension), data_type="mixed"
        ).observe(duration)
        
        self.exporter.hypervector_dimension.set(dimension)
        
        # Check performance
        if operation_type == "encode":
            self.performance_monitor.record_measurement("hypervector_generation", duration)
    
    def record_pir_query(
        self,
        server: str,
        latency_ms: float,
        query_size: int,
        response_size: int,
        success: bool = True
    ):
        """Record PIR query metrics"""
        status = "success" if success else "failure"
        self.exporter.pir_queries_total.labels(server=server, status=status).inc()
        
        self.exporter.pir_query_latency.labels(
            server=server, query_type="standard"
        ).observe(latency_ms)
        
        self.exporter.pir_query_size_bytes.observe(query_size)
        self.exporter.pir_response_size_bytes.observe(response_size)
        
        # Check performance and alerts
        self.performance_monitor.record_measurement("pir_query_latency", latency_ms)
        self.alert_manager.check_alert("high_latency", latency_ms / 300)
    
    def record_zk_proof(
        self,
        circuit_type: str,
        generation_time: float,
        verification_time: float,
        proof_size: int,
        constraints: int
    ):
        """Record ZK proof metrics"""
        self.exporter.zk_proofs_generated.labels(
            circuit_type=circuit_type, status="success"
        ).inc()
        
        self.exporter.zk_proof_generation_time.labels(
            circuit_type=circuit_type, hardware="cpu"
        ).observe(generation_time)
        
        self.exporter.zk_proof_verification_time.labels(
            circuit_type=circuit_type
        ).observe(verification_time * 1000)  # Convert to ms
        
        self.exporter.zk_proof_size_bytes.labels(circuit_type=circuit_type).set(proof_size)
        self.exporter.zk_circuit_constraints.labels(circuit_type=circuit_type).set(constraints)
        
        # Check performance
        self.performance_monitor.record_measurement("zk_proof_generation", generation_time)
        self.alert_manager.check_alert("zk_proof_timeout", generation_time)
    
    def record_compression(
        self,
        tier: str,
        original_size: int,
        compressed_size: int,
        compression_time: float
    ):
        """Record compression metrics"""
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        self.exporter.compression_ratio.labels(
            tier=tier, data_type="genomic"
        ).set(ratio)
        
        self.exporter.compression_operations.labels(
            tier=tier, operation="compress", status="success"
        ).inc()
        
        self.exporter.compressed_size_bytes.labels(tier=tier).observe(compressed_size)
        self.exporter.compression_time.labels(
            tier=tier, operation="compress"
        ).observe(compression_time)
        
        # Check performance
        metric_name = f"compression_ratio_{tier}"
        if metric_name in self.performance_monitor.targets:
            self.performance_monitor.record_measurement(metric_name, ratio)
            self.alert_manager.check_alert("low_compression", ratio / 4000)  # Compare to target
    
    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format"""
        if HAS_PROMETHEUS:
            return generate_latest(self.exporter.registry)
        else:
            # Generate mock metrics
            metrics = []
            metrics.append("# HELP genomevault_hdc_operations_total Total HDC operations")
            metrics.append("# TYPE genomevault_hdc_operations_total counter")
            metrics.append('genomevault_hdc_operations_total{operation_type="encode",status="success"} 100')
            
            metrics.append("# HELP genomevault_pir_query_latency_milliseconds PIR query latency")
            metrics.append("# TYPE genomevault_pir_query_latency_milliseconds histogram")
            metrics.append('genomevault_pir_query_latency_milliseconds_bucket{le="100"} 20')
            metrics.append('genomevault_pir_query_latency_milliseconds_bucket{le="500"} 95')
            metrics.append('genomevault_pir_query_latency_milliseconds_bucket{le="+Inf"} 100')
            
            return "\n".join(metrics).encode('utf-8')
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring system status"""
        return {
            "uptime_seconds": time.time() - self.start_time,
            "active_alerts": self.alert_manager.get_active_alerts(),
            "performance_compliance": self.performance_monitor.get_compliance_report(),
            "dashboards": self.dashboards.get_all_dashboards(),
            "metrics_endpoint": "/metrics"
        }


# Decorator for automatic metric collection
def monitor_performance(metric_type: str):
    """Decorator to automatically monitor function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metric based on type
                if metric_type == "hdc":
                    monitoring.record_hdc_operation(
                        func.__name__, 10000, duration, True
                    )
                elif metric_type == "pir":
                    monitoring.record_pir_query(
                        "default", duration * 1000, 1000, 5000, True
                    )
                elif metric_type == "zk":
                    monitoring.record_zk_proof(
                        "standard", duration, 0.01, 1000, 10000
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Function {func.__name__} failed: {e}")
                raise
        return wrapper
    return decorator


# Global monitoring instance
monitoring = MonitoringSystem()


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("GENOMEVAULT MONITORING SYSTEM")
    print("=" * 70)
    
    # Simulate some operations
    print("\nSimulating operations...")
    
    # Record HDC operation
    monitoring.record_hdc_operation("encode", 10000, 25.3)
    print("  ✓ Recorded HDC encoding: 25.3s")
    
    # Record PIR query
    monitoring.record_pir_query("server1", 250, 1024, 4096)
    print("  ✓ Recorded PIR query: 250ms")
    
    # Record ZK proof
    monitoring.record_zk_proof("variant", 14.5, 0.015, 2048, 50000)
    print("  ✓ Recorded ZK proof: 14.5s generation")
    
    # Record compression
    monitoring.record_compression("mini", 100_000_000, 25_000, 2.5)
    print("  ✓ Recorded compression: ratio 4000:1")
    
    # Export metrics
    print("\n" + "=" * 70)
    print("PROMETHEUS METRICS")
    print("=" * 70)
    metrics = monitoring.export_metrics()
    print(metrics.decode('utf-8')[:500] + "...")
    
    # Check system status
    print("\n" + "=" * 70)
    print("SYSTEM STATUS")
    print("=" * 70)
    status = monitoring.get_status()
    print(f"Uptime: {status['uptime_seconds']:.1f} seconds")
    print(f"Active alerts: {len(status['active_alerts'])}")
    print(f"Available dashboards: {', '.join(status['dashboards'])}")
    
    # Export Grafana dashboard
    print("\n" + "=" * 70)
    print("GRAFANA DASHBOARD EXPORT")
    print("=" * 70)
    dashboard_json = monitoring.dashboards.export_dashboard("system_overview")
    print(dashboard_json[:500] + "...")
    
    print("\n✅ Monitoring system operational!")