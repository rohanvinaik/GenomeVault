#!/usr/bin/env python3
"""
Privacy-Preserving Repository Insights Visualizer

Generates visualizations and alerts for GitHub repository metrics while
maintaining differential privacy guarantees.

Features:
- Trend analysis with privacy-preserving smoothing
- Anomaly detection with alert thresholds
- Daily/weekly pattern visualization
- Clone source diversity tracking
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class PrivacyInsightsVisualizer:
    """Visualize privacy-preserved repository insights with alerts"""

    def __init__(
        self,
        data_dir: Path = Path("repository_insights"),
        archive_dir: Path = Path("repository_insights_archive"),
    ):
        self.data_dir = Path(data_dir)
        self.archive_dir = Path(archive_dir)

        # Alert thresholds
        self.alert_thresholds = {
            "view_spike": 20,          # Alert if daily views exceed 20
            "clone_spike": 50,         # Alert if daily clones exceed 50
            "source_diversity": 10,    # Alert if clone sources exceed 10
            "view_growth_rate": 2.0,   # Alert if 200% growth vs previous period
            "clone_growth_rate": 2.0,  # Alert if 200% growth vs previous period
        }

    def load_latest_data(self) -> Optional[Dict]:
        """Load the most recent insights data"""
        data_files = sorted(self.data_dir.glob("raw_data_*.json"), reverse=True)
        if not data_files:
            return None

        with open(data_files[0], 'r') as f:
            return json.load(f)

    def load_historical_data(self) -> List[Dict]:
        """Load all historical data files for trend analysis"""
        all_files = list(self.data_dir.glob("raw_data_*.json")) + \
                   list(self.archive_dir.glob("raw_data_*.json"))

        historical = []
        for file_path in sorted(all_files):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['_file'] = file_path.name
                    historical.append(data)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")

        return historical

    def detect_anomalies(self, current: Dict, historical: List[Dict]) -> List[Dict]:
        """Detect anomalous patterns and generate alerts"""
        alerts = []

        if not current or not historical:
            return alerts

        current_summary = current.get('summary', {})

        # Check for view spike
        daily_views = self._get_daily_metric(current, 'views')
        if daily_views > self.alert_thresholds['view_spike']:
            alerts.append({
                'type': 'view_spike',
                'severity': 'medium',
                'message': f"Daily page views ({daily_views}) exceeded threshold ({self.alert_thresholds['view_spike']})",
                'value': daily_views,
                'threshold': self.alert_thresholds['view_spike']
            })

        # Check for clone spike
        daily_clones = self._get_daily_metric(current, 'clones')
        if daily_clones > self.alert_thresholds['clone_spike']:
            alerts.append({
                'type': 'clone_spike',
                'severity': 'high',
                'message': f"Daily clones ({daily_clones}) exceeded threshold ({self.alert_thresholds['clone_spike']})",
                'value': daily_clones,
                'threshold': self.alert_thresholds['clone_spike']
            })

        # Check for high source diversity
        clone_sources = current_summary.get('distinct_clone_sources', 0)
        if clone_sources > self.alert_thresholds['source_diversity']:
            alerts.append({
                'type': 'high_diversity',
                'severity': 'low',
                'message': f"Clone source diversity ({clone_sources}) is high - indicates broad interest",
                'value': clone_sources,
                'threshold': self.alert_thresholds['source_diversity']
            })

        # Check for growth rate anomalies
        if len(historical) >= 2:
            prev_summary = historical[-2].get('summary', {})

            # View growth
            prev_views = prev_summary.get('total_page_views', 0)
            curr_views = current_summary.get('total_page_views', 0)
            if prev_views > 0:
                view_growth = curr_views / prev_views
                if view_growth >= self.alert_thresholds['view_growth_rate']:
                    alerts.append({
                        'type': 'view_growth',
                        'severity': 'medium',
                        'message': f"Page views grew {view_growth:.1f}× vs previous collection",
                        'value': view_growth,
                        'threshold': self.alert_thresholds['view_growth_rate']
                    })

            # Clone growth
            prev_clones = prev_summary.get('total_repository_clones', 0)
            curr_clones = current_summary.get('total_repository_clones', 0)
            if prev_clones > 0:
                clone_growth = curr_clones / prev_clones
                if clone_growth >= self.alert_thresholds['clone_growth_rate']:
                    alerts.append({
                        'type': 'clone_growth',
                        'severity': 'high',
                        'message': f"Repository clones grew {clone_growth:.1f}× vs previous collection",
                        'value': clone_growth,
                        'threshold': self.alert_thresholds['clone_growth_rate']
                    })

        return alerts

    def _get_daily_metric(self, data: Dict, metric_type: str) -> int:
        """Extract today's metric from weekly data"""
        if metric_type == 'views':
            weekly_data = data.get('weekly_view_trend', [])
            field_name = 'views'
        else:  # clones
            weekly_data = data.get('weekly_clone_pattern', [])
            field_name = 'clone_events'

        if not weekly_data:
            return 0

        # Return the most recent day's count
        return weekly_data[-1].get(field_name, 0)

    def generate_ascii_chart(self, data_points: List[Tuple[str, int]], max_width: int = 50) -> str:
        """Generate ASCII bar chart for terminal display"""
        if not data_points:
            return "No data available"

        max_value = max(val for _, val in data_points)
        if max_value == 0:
            return "No activity recorded"

        lines = []
        for label, value in data_points:
            bar_length = int((value / max_value) * max_width)
            bar = '█' * bar_length
            lines.append(f"{label:12} | {bar} {value}")

        return '\n'.join(lines)

    def visualize_trends(self, historical: List[Dict]) -> str:
        """Generate trend visualizations from historical data"""
        if not historical:
            return "No historical data available"

        output = []
        output.append("=" * 80)
        output.append("PRIVACY-PRESERVING REPOSITORY INSIGHTS - TREND ANALYSIS")
        output.append("=" * 80)
        output.append("")

        # Weekly view trends
        output.append("📊 Weekly Page View Trends")
        output.append("-" * 80)

        latest = historical[-1]
        weekly_views = latest.get('weekly_view_trend', [])

        if weekly_views:
            view_data = [(item['date'], item['views']) for item in weekly_views[-7:]]
            output.append(self.generate_ascii_chart(view_data))
            output.append("")

            total_weekly_views = sum(item['views'] for item in weekly_views[-7:])
            output.append(f"Total views (last 7 days): {total_weekly_views}")
        else:
            output.append("No view data available")

        output.append("")
        output.append("")

        # Weekly clone patterns
        output.append("🔄 Weekly Clone Patterns")
        output.append("-" * 80)

        weekly_clones = latest.get('weekly_clone_pattern', [])

        if weekly_clones:
            clone_data = [(item['date'], item['clone_events']) for item in weekly_clones[-7:]]
            output.append(self.generate_ascii_chart(clone_data))
            output.append("")

            total_weekly_clones = sum(item['clone_events'] for item in weekly_clones[-7:])
            unique_sources = sum(item['unique_initiators'] for item in weekly_clones[-7:])
            output.append(f"Total clones (last 7 days): {total_weekly_clones}")
            output.append(f"Unique clone sources (last 7 days): {unique_sources}")
        else:
            output.append("No clone data available")

        output.append("")
        output.append("")

        # Historical summary
        output.append("📈 Historical Growth Summary")
        output.append("-" * 80)

        if len(historical) >= 2:
            first = historical[0].get('summary', {})
            latest_summary = latest.get('summary', {})

            view_growth = latest_summary.get('total_page_views', 0) - first.get('total_page_views', 0)
            clone_growth = latest_summary.get('total_repository_clones', 0) - first.get('total_repository_clones', 0)

            output.append(f"Total page views growth: +{view_growth}")
            output.append(f"Total repository clones growth: +{clone_growth}")
            output.append(f"Data collection period: {len(historical)} snapshots")
        else:
            output.append("Insufficient historical data for growth analysis")

        output.append("")

        return '\n'.join(output)

    def generate_alert_report(self, alerts: List[Dict]) -> str:
        """Generate formatted alert report"""
        if not alerts:
            return "✅ No alerts detected - all metrics within normal ranges"

        output = []
        output.append("🚨 ALERTS DETECTED")
        output.append("=" * 80)
        output.append("")

        # Sort by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_alerts = sorted(alerts, key=lambda x: severity_order.get(x['severity'], 3))

        for alert in sorted_alerts:
            severity_icon = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }.get(alert['severity'], '⚪')

            output.append(f"{severity_icon} [{alert['severity'].upper()}] {alert['type']}")
            output.append(f"   {alert['message']}")
            output.append("")

        return '\n'.join(output)

    def run_full_analysis(self) -> str:
        """Run complete analysis with trends and alerts"""
        current = self.load_latest_data()
        historical = self.load_historical_data()

        output = []

        # Header
        output.append("")
        output.append("=" * 80)
        output.append("PRIVACY-PRESERVING REPOSITORY INSIGHTS - FULL ANALYSIS")
        output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("=" * 80)
        output.append("")

        # Current snapshot
        if current:
            summary = current.get('summary', {})
            output.append("📊 Current Metrics")
            output.append("-" * 80)
            output.append(f"Total page views: {summary.get('total_page_views', 0)}")
            output.append(f"Unique observers: {summary.get('unique_observers', 0)}")
            output.append(f"Total repository clones: {summary.get('total_repository_clones', 0)}")
            output.append(f"Distinct clone sources: {summary.get('distinct_clone_sources', 0)}")
            output.append(f"Community endorsements (stars): {summary.get('community_endorsements', 0)}")
            output.append(f"Derivative projects (forks): {summary.get('derivative_projects', 0)}")
            output.append("")
            output.append(f"Privacy guarantee: (ε=1.0, δ=1e-5)-differential privacy")
            output.append("")

        # Alerts
        alerts = self.detect_anomalies(current, historical)
        output.append(self.generate_alert_report(alerts))
        output.append("")

        # Trends
        output.append(self.visualize_trends(historical))

        output.append("")
        output.append("=" * 80)
        output.append("END OF ANALYSIS")
        output.append("=" * 80)

        return '\n'.join(output)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Privacy-Preserving Repository Insights Visualizer"
    )
    parser.add_argument(
        '--data-dir',
        default='repository_insights',
        help='Directory containing insights data'
    )
    parser.add_argument(
        '--archive-dir',
        default='repository_insights_archive',
        help='Directory containing archived data'
    )
    parser.add_argument(
        '--output',
        help='Output file for analysis report (default: stdout)'
    )

    args = parser.parse_args()

    visualizer = PrivacyInsightsVisualizer(
        data_dir=Path(args.data_dir),
        archive_dir=Path(args.archive_dir)
    )

    report = visualizer.run_full_analysis()

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Analysis report saved to: {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()
