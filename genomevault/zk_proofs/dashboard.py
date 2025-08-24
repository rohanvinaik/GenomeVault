"""Dashboard visualization for ZK proof performance monitoring."""

import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from genomevault.zk_proofs.performance_monitor import get_monitor, PerformanceMonitor
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceDashboard:
    """Terminal-based performance dashboard."""

    def __init__(self, monitor: Optional[PerformanceMonitor] = None):
        """Initialize dashboard."""
        self.monitor = monitor or get_monitor()
        self.refresh_interval = 1.0  # seconds

    def render_summary(self, data: Dict[str, Any]) -> List[str]:
        """Render summary section."""
        summary = data["summary"]
        lines = []

        lines.append("=" * 80)
        lines.append("  ZK PROOF PERFORMANCE DASHBOARD")
        lines.append("=" * 80)
        lines.append("")

        # Summary stats
        lines.append("📊 SUMMARY")
        lines.append(f"  Total Operations: {summary['total_operations']:,}")
        lines.append(f"  Success Rate:     {summary['success_rate']:.1%}")
        lines.append(f"  Cache Hit Rate:   {summary['overall_cache_hit_rate']:.1%}")
        lines.append(f"  Active Circuits:  {summary['circuits_tracked']}")

        # Alerts
        if summary["active_alerts"] > 0:
            lines.append(f"  ⚠️  Active Alerts:  {summary['active_alerts']}")

        lines.append("")
        return lines

    def render_circuits(self, data: Dict[str, Any]) -> List[str]:
        """Render circuit performance table."""
        lines = []

        lines.append("⚡ CIRCUIT PERFORMANCE")
        lines.append("")
        lines.append("  Circuit               │ Ops    │ Success │ Avg(ms) │ P95(ms) │ Cache")
        lines.append("  ──────────────────────┼────────┼─────────┼─────────┼─────────┼───────")

        for circuit, stats in data["circuits"].items():
            lines.append(
                f"  {circuit:20s} │ {stats['total_operations']:6d} │ "
                f"{stats['success_rate']*100:6.1f}% │ {stats['avg_witness_ms']:7.2f} │ "
                f"{stats['p95_witness_ms']:7.2f} │ {stats['cache_hit_rate']*100:5.1f}%"
            )

        lines.append("")
        return lines

    def render_latency_graph(self, data: Dict[str, Any]) -> List[str]:
        """Render ASCII latency graph."""
        lines = []

        # Get recent witness operations
        recent_metrics = data.get("recent_metrics", [])
        witness_metrics = [
            m for m in recent_metrics if m["operation"] == "witness" and m["success"]
        ]

        if not witness_metrics:
            return lines

        lines.append("📈 WITNESS LATENCY (last 20 operations)")
        lines.append("")

        # Get last 20 latencies
        latencies = [m["duration_ms"] for m in witness_metrics[-20:]]

        if latencies:
            max_lat = max(latencies)
            min_lat = min(latencies)

            # Create ASCII bars
            height = 10
            width = len(latencies)

            for h in range(height, 0, -1):
                threshold = min_lat + (max_lat - min_lat) * (h / height)
                row = "  "
                for lat in latencies:
                    if lat >= threshold:
                        row += "█"
                    else:
                        row += " "

                if h == height:
                    row += f" {max_lat:.1f}ms"
                elif h == 1:
                    row += f" {min_lat:.1f}ms"

                lines.append(row)

            lines.append("  " + "─" * width)
            lines.append("")

        return lines

    def render_alerts(self, data: Dict[str, Any]) -> List[str]:
        """Render recent alerts."""
        lines = []

        alerts = data.get("alerts", [])
        if not alerts:
            return lines

        lines.append("⚠️  RECENT ALERTS")
        lines.append("")

        for alert in alerts[-5:]:
            timestamp = datetime.fromtimestamp(alert["timestamp"]).strftime("%H:%M:%S")

            if alert["type"] == "high_latency":
                lines.append(
                    f"  [{timestamp}] High latency: {alert['circuit']} "
                    f"({alert['value']:.1f}ms > {alert['threshold']:.1f}ms)"
                )
            elif alert["type"] == "high_error_rate":
                lines.append(
                    f"  [{timestamp}] High errors: {alert['circuit']} "
                    f"({alert['value']*100:.1f}% > {alert['threshold']*100:.1f}%)"
                )
            elif alert["type"] == "low_cache_hit_rate":
                lines.append(
                    f"  [{timestamp}] Low cache hits: {alert['circuit']} "
                    f"({alert['value']*100:.1f}% < {alert['threshold']*100:.1f}%)"
                )

        lines.append("")
        return lines

    def render_device_usage(self, data: Dict[str, Any]) -> List[str]:
        """Render device usage statistics."""
        lines = []

        recent_metrics = data.get("recent_metrics", [])
        if not recent_metrics:
            return lines

        # Count device usage
        device_counts = {}
        for metric in recent_metrics:
            device = metric.get("device", "cpu")
            device_counts[device] = device_counts.get(device, 0) + 1

        lines.append("🖥️  DEVICE USAGE")
        lines.append("")

        total = sum(device_counts.values())
        for device, count in sorted(device_counts.items()):
            percentage = (count / total) * 100 if total > 0 else 0
            bar_length = int(percentage / 2)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            lines.append(f"  {device:8s} │ {bar} │ {percentage:5.1f}%")

        lines.append("")
        return lines

    def render(self) -> str:
        """Render complete dashboard."""
        data = self.monitor.get_dashboard_data()

        lines = []
        lines.extend(self.render_summary(data))
        lines.extend(self.render_circuits(data))
        lines.extend(self.render_latency_graph(data))
        lines.extend(self.render_device_usage(data))
        lines.extend(self.render_alerts(data))

        # Footer
        lines.append("─" * 80)
        lines.append(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)

    def print_dashboard(self):
        """Print dashboard to console."""
        import os

        # Clear screen
        os.system("clear" if os.name == "posix" else "cls")

        # Print dashboard
        print(self.render())

    def run_interactive(self, duration: Optional[int] = None):
        """Run interactive dashboard."""
        import signal
        import sys

        running = True

        def signal_handler(sig, frame):
            nonlocal running
            running = False
            print("\n\nDashboard stopped.")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        start_time = time.time()

        print("Starting performance dashboard... (Press Ctrl+C to stop)")
        time.sleep(1)

        while running:
            self.print_dashboard()

            if duration and (time.time() - start_time) > duration:
                break

            time.sleep(self.refresh_interval)


class HTMLDashboard:
    """Generate HTML dashboard for web viewing."""

    def __init__(self, monitor: Optional[PerformanceMonitor] = None):
        """Initialize HTML dashboard."""
        self.monitor = monitor or get_monitor()

    def generate_html(self) -> str:
        """Generate HTML dashboard."""
        data = self.monitor.get_dashboard_data()

        html = (
            """
<!DOCTYPE html>
<html>
<head>
    <title>ZK Performance Dashboard</title>
    <style>
        body {
            font-family: 'Courier New', monospace;
            background: #1e1e1e;
            color: #00ff00;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-box {
            background: #333;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #00ff00;
        }
        .stat-label {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #444;
        }
        th { background: #333; }
        .alert {
            background: #ff3333;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .graph {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        canvas { width: 100%; height: 200px; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 ZK Performance Dashboard</h1>
            <p>Real-time monitoring of zero-knowledge proof generation</p>
        </div>

        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">"""
            + f"{data['summary']['total_operations']:,}"
            + """</div>
                <div class="stat-label">Total Operations</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">"""
            + f"{data['summary']['success_rate']:.1%}"
            + """</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">"""
            + f"{data['summary']['overall_cache_hit_rate']:.1%}"
            + """</div>
                <div class="stat-label">Cache Hit Rate</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">"""
            + f"{data['summary']['active_alerts']}"
            + """</div>
                <div class="stat-label">Active Alerts</div>
            </div>
        </div>

        <div class="metric-card">
            <h2>Circuit Performance</h2>
            <table>
                <thead>
                    <tr>
                        <th>Circuit</th>
                        <th>Operations</th>
                        <th>Success Rate</th>
                        <th>Avg Latency</th>
                        <th>P95 Latency</th>
                        <th>Cache Hits</th>
                    </tr>
                </thead>
                <tbody>"""
        )

        for circuit, stats in data["circuits"].items():
            html += f"""
                    <tr>
                        <td>{circuit}</td>
                        <td>{stats['total_operations']}</td>
                        <td>{stats['success_rate']:.1%}</td>
                        <td>{stats['avg_witness_ms']:.2f}ms</td>
                        <td>{stats['p95_witness_ms']:.2f}ms</td>
                        <td>{stats['cache_hit_rate']:.1%}</td>
                    </tr>"""

        html += (
            """
                </tbody>
            </table>
        </div>

        <div class="graph">
            <h2>Latency Trend</h2>
            <canvas id="latencyChart"></canvas>
        </div>

        <script>
            // Latency chart
            const ctx = document.getElementById('latencyChart').getContext('2d');
            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ["""
            + ",".join([f"'{i}'" for i in range(20)])
            + """],
                    datasets: [{
                        label: 'Witness Latency (ms)',
                        data: ["""
            + ",".join(
                [
                    str(m["duration_ms"])
                    for m in data["recent_metrics"][-20:]
                    if m["operation"] == "witness"
                ]
            )
            + """],
                        borderColor: '#00ff00',
                        backgroundColor: 'rgba(0, 255, 0, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: '#00ff00' } }
                    },
                    scales: {
                        y: { ticks: { color: '#888' }, grid: { color: '#444' } },
                        x: { ticks: { color: '#888' }, grid: { color: '#444' } }
                    }
                }
            });

            // Auto-refresh
            setTimeout(() => location.reload(), 5000);
        </script>
    </div>
</body>
</html>
        """
        )

        return html

    def save_html(self, filepath: Path = Path("zk_dashboard.html")):
        """Save HTML dashboard to file."""
        html = self.generate_html()
        with open(filepath, "w") as f:
            f.write(html)
        logger.info(f"Dashboard saved to {filepath}")


def run_dashboard(mode: str = "terminal", duration: Optional[int] = None):
    """
    Run performance dashboard.

    Args:
        mode: Dashboard mode ('terminal', 'html', or 'both')
        duration: Duration to run in seconds (None for infinite)
    """
    monitor = get_monitor()

    if mode == "terminal":
        dashboard = PerformanceDashboard(monitor)
        dashboard.run_interactive(duration)

    elif mode == "html":
        dashboard = HTMLDashboard(monitor)
        dashboard.save_html()
        print("HTML dashboard saved to zk_dashboard.html")

    elif mode == "both":
        # Generate HTML
        html_dashboard = HTMLDashboard(monitor)
        html_dashboard.save_html()

        # Run terminal dashboard
        terminal_dashboard = PerformanceDashboard(monitor)
        terminal_dashboard.run_interactive(duration)

    else:
        raise ValueError(f"Unknown mode: {mode}")
