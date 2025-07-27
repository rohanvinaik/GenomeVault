"""
Metrics collection framework for GenomeVault.

Provides real-time metrics collection and reporting for performance monitoring,
benchmarking, and system health tracking.
"""

import json
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class MetricsCollector:
    """
    Thread-safe metrics collector for performance monitoring.

    Collects and aggregates metrics across the system including:
    - Proof generation/verification times
    - Network latencies
    - Compression ratios
    - Error rates
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for global metrics collection."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize metrics collector."""
        if not hasattr(self, "initialized"):
            self.metrics = defaultdict(list)
            self.start_time = time.time()
            self.lock = threading.Lock()
            self.initialized = True

    def record(
        self, metric_name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None
    ):
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Numeric value to record
            unit: Unit of measurement
            tags: Optional tags for filtering/grouping
        """
        with self.lock:
            self.metrics[metric_name].append(
                {"value": value, "unit": unit, "timestamp": time.time(), "tags": tags or {}}
            )

    def record_duration(
        self,
        metric_name: str,
        start_time: float,
        unit: str = "ms",
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Record a duration metric.

        Args:
            metric_name: Name of the metric
            start_time: Start time from time.time()
            unit: Unit of measurement (default: ms)
            tags: Optional tags
        """
        duration = time.time() - start_time

        if unit == "ms":
            duration *= 1000
        elif unit == "us":
            duration *= 1000000

        self.record(metric_name, duration, unit, tags)

    def get_metric(self, metric_name: str, since: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get all recorded values for a metric.

        Args:
            metric_name: Name of the metric
            since: Optional timestamp to filter results

        Returns:
            List of metric records
        """
        with self.lock:
            records = self.metrics.get(metric_name, [])

            if since:
                records = [r for r in records if r["timestamp"] >= since]

            return records.copy()

    def get_summary(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistical summary of metrics.

        Args:
            metric_name: Optional specific metric (None for all)

        Returns:
            Dictionary with statistical summaries
        """
        with self.lock:
            if metric_name:
                metrics_to_summarize = {metric_name: self.metrics.get(metric_name, [])}
            else:
                metrics_to_summarize = dict(self.metrics)

            summary = {}

            for name, records in metrics_to_summarize.items():
                if not records:
                    continue

                values = [r["value"] for r in records]

                summary[name] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99),
                    "unit": records[0].get("unit", ""),
                    "latest": values[-1],
                    "latest_timestamp": records[-1]["timestamp"],
                }

            return summary

    def get_time_series(self, metric_name: str, interval: int = 60) -> List[Dict[str, Any]]:
        """
        Get time series data for a metric.

        Args:
            metric_name: Name of the metric
            interval: Aggregation interval in seconds

        Returns:
            List of time-bucketed aggregations
        """
        with self.lock:
            records = self.metrics.get(metric_name, [])

            if not records:
                return []

            # Bucket by interval
            buckets = defaultdict(list)

            for record in records:
                bucket = int(record["timestamp"] // interval) * interval
                buckets[bucket].append(record["value"])

            # Aggregate each bucket
            time_series = []

            for timestamp, values in sorted(buckets.items()):
                time_series.append(
                    {
                        "timestamp": timestamp,
                        "count": len(values),
                        "mean": np.mean(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "sum": np.sum(values),
                    }
                )

            return time_series

    def export_json(self, output_path: str, include_raw: bool = False):
        """
        Export metrics to JSON file.

        Args:
            output_path: Path to write JSON file
            include_raw: Include raw metric records
        """
        with self.lock:
            export_data = {
                "export_time": datetime.now().isoformat(),
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "duration_seconds": time.time() - self.start_time,
                "summary": self.get_summary(),
            }

            if include_raw:
                export_data["raw_metrics"] = dict(self.metrics)

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

    def clear(self, metric_name: Optional[str] = None):
        """
        Clear metrics.

        Args:
            metric_name: Optional specific metric to clear (None for all)
        """
        with self.lock:
            if metric_name:
                self.metrics.pop(metric_name, None)
            else:
                self.metrics.clear()

    def compare_claimed_vs_measured(self, claims: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare claimed metrics against measured values.

        Args:
            claims: Dictionary of claimed metric values

        Returns:
            Comparison results with ratios
        """
        summary = self.get_summary()
        comparisons = {}

        for metric_name, claimed_value in claims.items():
            if metric_name in summary:
                measured = summary[metric_name]

                comparisons[metric_name] = {
                    "claimed": claimed_value,
                    "measured_mean": measured["mean"],
                    "measured_median": measured["median"],
                    "measured_p95": measured["p95"],
                    "ratio_mean": measured["mean"] / claimed_value,
                    "ratio_median": measured["median"] / claimed_value,
                    "within_claim": measured["mean"] <= claimed_value,
                    "unit": measured["unit"],
                }
            else:
                comparisons[metric_name] = {
                    "claimed": claimed_value,
                    "measured": "NOT_MEASURED",
                    "ratio": None,
                }

        return comparisons


class MetricsContext:
    """Context manager for timing operations."""

    def __init__(self, metric_name: str, unit: str = "ms", tags: Optional[Dict[str, str]] = None):
        """
        Initialize metrics context.

        Args:
            metric_name: Name of the metric to record
            unit: Time unit (ms, s, us)
            tags: Optional tags
        """
        self.metric_name = metric_name
        self.unit = unit
        self.tags = tags
        self.start_time = None
        self.collector = MetricsCollector()

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record metric."""
        if self.start_time:
            self.collector.record_duration(self.metric_name, self.start_time, self.unit, self.tags)


def metrics_decorator(metric_name: str, unit: str = "ms"):
    """
    Decorator for timing function execution.

    Args:
        metric_name: Name of the metric
        unit: Time unit
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with MetricsContext(metric_name, unit):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Global instance
metrics = MetricsCollector()


# Example usage
if __name__ == "__main__":
    # Example 1: Direct recording
    metrics.record("proof_size", 384, "bytes")
    metrics.record("verification_time", 25.5, "ms")

    # Example 2: Context manager
    with MetricsContext("proof_generation_time"):
        time.sleep(0.1)  # Simulate work

    # Example 3: Decorator
    @metrics_decorator("function_execution_time")
    def example_function():
        time.sleep(0.05)
        return "done"

    example_function()

    # Get summary
    print("Metrics Summary:")
    print(json.dumps(metrics.get_summary(), indent=2))

    # Compare against claims
    claims = {"proof_size": 384, "verification_time": 25}

    print("\nClaimed vs Measured:")
    print(json.dumps(metrics.compare_claimed_vs_measured(claims), indent=2))

    # Export to file
    metrics.export_json("metrics_report.json", include_raw=True)
