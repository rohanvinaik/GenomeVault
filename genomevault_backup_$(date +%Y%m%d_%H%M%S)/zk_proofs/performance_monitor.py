"""Real-time performance monitoring for ZK proof generation."""

import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np
from collections import deque, defaultdict


@dataclass
class PerformanceMetric:
    """Single performance measurement."""

    timestamp: float
    circuit_type: str
    operation: str  # witness, proof, verify
    duration_ms: float
    input_size: int
    memory_mb: float
    cache_hit: bool = False
    device: str = "cpu"
    success: bool = True
    error: Optional[str] = None


@dataclass
class CircuitStats:
    """Statistics for a specific circuit."""

    circuit_type: str
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0

    avg_witness_ms: float = 0
    p95_witness_ms: float = 0
    min_witness_ms: float = float("inf")
    max_witness_ms: float = 0

    cache_hits: int = 0
    cache_hit_rate: float = 0

    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    hourly_throughput: deque = field(default_factory=lambda: deque(maxlen=24))


class PerformanceMonitor:
    """Monitor and track ZK proof generation performance."""

    def __init__(self, log_dir: Path = Path("zk_performance_logs")):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)

        self.metrics: List[PerformanceMetric] = []
        self.circuit_stats: Dict[str, CircuitStats] = {}

        # Real-time tracking
        self.current_hour_operations = defaultdict(int)
        self.last_hour_reset = time.time()

        # Alerts
        self.alert_thresholds = {
            "witness_ms": 5.0,  # Alert if witness > 5ms
            "error_rate": 0.01,  # Alert if error rate > 1%
            "cache_hit_rate": 0.5,  # Alert if cache hit < 50%
        }
        self.alerts: List[Dict] = []

    def record_operation(
        self,
        circuit_type: str,
        operation: str,
        duration_ms: float,
        input_size: int,
        memory_mb: float = 0,
        cache_hit: bool = False,
        device: str = "cpu",
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record a single operation."""

        metric = PerformanceMetric(
            timestamp=time.time(),
            circuit_type=circuit_type,
            operation=operation,
            duration_ms=duration_ms,
            input_size=input_size,
            memory_mb=memory_mb,
            cache_hit=cache_hit,
            device=device,
            success=success,
            error=error,
        )

        self.metrics.append(metric)
        self._update_stats(metric)
        self._check_alerts(metric)

        # Persist to disk periodically
        if len(self.metrics) % 100 == 0:
            self._persist_metrics()

    def _update_stats(self, metric: PerformanceMetric) -> None:
        """Update circuit statistics."""

        if metric.circuit_type not in self.circuit_stats:
            self.circuit_stats[metric.circuit_type] = CircuitStats(circuit_type=metric.circuit_type)

        stats = self.circuit_stats[metric.circuit_type]
        stats.total_operations += 1

        if metric.success:
            stats.successful_operations += 1

            if metric.operation == "witness":
                stats.recent_latencies.append(metric.duration_ms)

                # Update statistics
                latencies = list(stats.recent_latencies)
                stats.avg_witness_ms = np.mean(latencies)
                stats.p95_witness_ms = np.percentile(latencies, 95) if latencies else 0
                stats.min_witness_ms = min(stats.min_witness_ms, metric.duration_ms)
                stats.max_witness_ms = max(stats.max_witness_ms, metric.duration_ms)

            if metric.cache_hit:
                stats.cache_hits += 1

            stats.cache_hit_rate = stats.cache_hits / stats.total_operations
        else:
            stats.failed_operations += 1

        # Update hourly throughput
        self.current_hour_operations[metric.circuit_type] += 1

        # Reset hourly counter if needed
        if time.time() - self.last_hour_reset > 3600:
            for circuit, count in self.current_hour_operations.items():
                if circuit in self.circuit_stats:
                    self.circuit_stats[circuit].hourly_throughput.append(count)
            self.current_hour_operations.clear()
            self.last_hour_reset = time.time()

    def _check_alerts(self, metric: PerformanceMetric) -> None:
        """Check for performance alerts."""

        # High latency alert
        if (
            metric.operation == "witness"
            and metric.duration_ms > self.alert_thresholds["witness_ms"]
        ):
            self.alerts.append(
                {
                    "type": "high_latency",
                    "circuit": metric.circuit_type,
                    "value": metric.duration_ms,
                    "threshold": self.alert_thresholds["witness_ms"],
                    "timestamp": metric.timestamp,
                }
            )

        # Error rate alert
        if metric.circuit_type in self.circuit_stats:
            stats = self.circuit_stats[metric.circuit_type]
            error_rate = stats.failed_operations / max(1, stats.total_operations)

            if error_rate > self.alert_thresholds["error_rate"]:
                self.alerts.append(
                    {
                        "type": "high_error_rate",
                        "circuit": metric.circuit_type,
                        "value": error_rate,
                        "threshold": self.alert_thresholds["error_rate"],
                        "timestamp": time.time(),
                    }
                )

        # Low cache hit rate alert
        if metric.circuit_type in self.circuit_stats:
            stats = self.circuit_stats[metric.circuit_type]

            if (
                stats.total_operations > 100
                and stats.cache_hit_rate < self.alert_thresholds["cache_hit_rate"]
            ):
                self.alerts.append(
                    {
                        "type": "low_cache_hit_rate",
                        "circuit": metric.circuit_type,
                        "value": stats.cache_hit_rate,
                        "threshold": self.alert_thresholds["cache_hit_rate"],
                        "timestamp": time.time(),
                    }
                )

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard."""

        # Overall statistics
        total_ops = sum(s.total_operations for s in self.circuit_stats.values())
        total_success = sum(s.successful_operations for s in self.circuit_stats.values())
        total_cache_hits = sum(s.cache_hits for s in self.circuit_stats.values())

        # Circuit-specific data
        circuit_data = {}
        for circuit, stats in self.circuit_stats.items():
            circuit_data[circuit] = {
                "total_operations": stats.total_operations,
                "success_rate": stats.successful_operations / max(1, stats.total_operations),
                "avg_witness_ms": stats.avg_witness_ms,
                "p95_witness_ms": stats.p95_witness_ms,
                "cache_hit_rate": stats.cache_hit_rate,
                "recent_throughput": sum(stats.hourly_throughput)
                / max(1, len(stats.hourly_throughput)),
            }

        # Recent metrics for graphs
        recent_metrics = self.metrics[-1000:]  # Last 1000 operations

        return {
            "summary": {
                "total_operations": total_ops,
                "success_rate": total_success / max(1, total_ops),
                "overall_cache_hit_rate": total_cache_hits / max(1, total_ops),
                "circuits_tracked": len(self.circuit_stats),
                "active_alerts": len(self.alerts),
            },
            "circuits": circuit_data,
            "recent_metrics": [asdict(m) for m in recent_metrics[-100:]],
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "timestamp": time.time(),
        }

    def _persist_metrics(self) -> None:
        """Save metrics to disk."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"metrics_{timestamp}.json"

        data = {
            "metrics": [asdict(m) for m in self.metrics[-1000:]],
            "stats": {circuit: asdict(stats) for circuit, stats in self.circuit_stats.items()},
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def generate_report(self) -> str:
        """Generate performance report."""

        report = ["# ZK Performance Report", ""]
        report.append(f"Generated: {datetime.now()}")
        report.append("")

        # Summary
        data = self.get_dashboard_data()
        summary = data["summary"]

        report.append("## Summary")
        report.append(f"- Total Operations: {summary['total_operations']:,}")
        report.append(f"- Success Rate: {summary['success_rate']:.2%}")
        report.append(f"- Cache Hit Rate: {summary['overall_cache_hit_rate']:.2%}")
        report.append(f"- Active Alerts: {summary['active_alerts']}")
        report.append("")

        # Circuit details
        report.append("## Circuit Performance")
        report.append("")
        report.append("| Circuit | Ops | Success | Avg (ms) | P95 (ms) | Cache Hit |")
        report.append("|---------|-----|---------|----------|----------|-----------|")

        for circuit, stats in data["circuits"].items():
            report.append(
                f"| {circuit} | {stats['total_operations']} | "
                f"{stats['success_rate']:.1%} | {stats['avg_witness_ms']:.2f} | "
                f"{stats['p95_witness_ms']:.2f} | {stats['cache_hit_rate']:.1%} |"
            )

        # Alerts
        if self.alerts:
            report.append("")
            report.append("## Recent Alerts")
            for alert in self.alerts[-5:]:
                report.append(
                    f"- {alert['type']}: {alert['circuit']} "
                    f"({alert['value']:.2f} > {alert['threshold']:.2f})"
                )

        return "\n".join(report)


# Global monitor instance
_monitor = None


def get_monitor() -> PerformanceMonitor:
    """Get global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor
