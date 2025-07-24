"""
GenomeVault Metrics Collection Framework
Provides real-time metrics collection and analysis for performance monitoring.
"""

import time
import json
import numpy as np
from collections import defaultdict
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path


class MetricsCollector:
    """Collects and aggregates performance metrics across the system."""
    
    def __init__(self, output_dir: str = "./metrics"):
        self.metrics = defaultdict(list)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def record(self, metric_name: str, value: float, unit: str = "", metadata: Dict[str, Any] = None):
        """Record a metric value with optional metadata."""
        entry = {
            "value": value,
            "unit": unit,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.metrics[metric_name].append(entry)
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return TimedOperation(self, operation_name)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get statistical summary of all metrics."""
        summary = {}
        
        for name, values in self.metrics.items():
            if not values:
                continue
                
            vals = [m["value"] for m in values]
            summary[name] = {
                "mean": np.mean(vals),
                "std": np.std(vals),
                "min": min(vals),
                "max": max(vals),
                "count": len(vals),
                "unit": values[0].get("unit", ""),
                "percentiles": {
                    "p50": np.percentile(vals, 50),
                    "p90": np.percentile(vals, 90),
                    "p95": np.percentile(vals, 95),
                    "p99": np.percentile(vals, 99)
                }
            }
        
        return summary
    
    def export_json(self, filename: Optional[str] = None) -> str:
        """Export metrics to JSON file."""
        if filename is None:
            filename = f"metrics_{self.session_id}.json"
        
        filepath = self.output_dir / filename
        
        data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "raw_metrics": dict(self.metrics)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return str(filepath)
    
    def export_csv(self, metric_name: str, filename: Optional[str] = None) -> str:
        """Export specific metric to CSV."""
        import csv
        
        if filename is None:
            filename = f"{metric_name}_{self.session_id}.csv"
        
        filepath = self.output_dir / filename
        
        if metric_name not in self.metrics:
            raise ValueError(f"Metric '{metric_name}' not found")
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "value", "unit", "metadata"])
            
            for entry in self.metrics[metric_name]:
                writer.writerow([
                    entry["timestamp"],
                    entry["value"],
                    entry["unit"],
                    json.dumps(entry.get("metadata", {}))
                ])
        
        return str(filepath)
    
    def compare_metrics(self, metric1: str, metric2: str) -> Dict[str, Any]:
        """Compare two metrics statistically."""
        if metric1 not in self.metrics or metric2 not in self.metrics:
            raise ValueError("Both metrics must exist")
        
        vals1 = [m["value"] for m in self.metrics[metric1]]
        vals2 = [m["value"] for m in self.metrics[metric2]]
        
        # Perform t-test if scipy is available
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(vals1, vals2)
            statistical_test = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            }
        except ImportError:
            statistical_test = None
        
        return {
            "metric1": {
                "name": metric1,
                "mean": np.mean(vals1),
                "std": np.std(vals1)
            },
            "metric2": {
                "name": metric2,
                "mean": np.mean(vals2),
                "std": np.std(vals2)
            },
            "difference": {
                "mean": np.mean(vals1) - np.mean(vals2),
                "percentage": ((np.mean(vals1) - np.mean(vals2)) / np.mean(vals2)) * 100
            },
            "statistical_test": statistical_test
        }


class TimedOperation:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, operation_name: str):
        self.collector = collector
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.time() - self.start_time) * 1000  # Convert to ms
        self.collector.record(f"{self.operation_name}_time", duration, "ms")


# Global metrics instance
global_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return global_metrics
