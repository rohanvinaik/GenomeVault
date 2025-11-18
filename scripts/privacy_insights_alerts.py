#!/usr/bin/env python3
"""
Privacy-Preserving Repository Insights Alert System

Monitors privacy-preserved repository metrics and sends alerts when
anomalous patterns are detected.

Features:
- Real-time anomaly detection
- Configurable alert thresholds
- Alert history tracking
- Notification system (console, file, optional email)
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class AlertSystem:
    """Automated alert system for privacy-preserved repository insights"""

    def __init__(
        self,
        data_dir: Path = Path("repository_insights"),
        alert_log: Path = Path("repository_insights/alerts.jsonl"),
        check_interval: int = 3600,  # 1 hour
    ):
        self.data_dir = Path(data_dir)
        self.alert_log = Path(alert_log)
        self.check_interval = check_interval

        # Create alert log directory if needed
        self.alert_log.parent.mkdir(parents=True, exist_ok=True)

        # Alert thresholds (configurable)
        self.thresholds = {
            "view_spike_daily": 20,        # Daily views
            "clone_spike_daily": 50,       # Daily clones
            "source_diversity_high": 15,   # Distinct clone sources
            "view_growth_rate": 2.0,       # 2× growth
            "clone_growth_rate": 2.0,      # 2× growth
            "sustained_high_activity": 3,  # Days of high activity
        }

        # Track alert state
        self.previous_data: Optional[Dict] = None
        self.alert_history: List[Dict] = []
        self.consecutive_high_activity_days = 0

    def load_latest_data(self) -> Optional[Dict]:
        """Load the most recent insights data"""
        data_files = sorted(self.data_dir.glob("raw_data_*.json"), reverse=True)
        if not data_files:
            return None

        with open(data_files[0], 'r') as f:
            return json.load(f)

    def check_view_spike(self, data: Dict) -> Optional[Dict]:
        """Check for unusual spike in page views"""
        weekly_views = data.get('weekly_view_trend', [])
        if not weekly_views:
            return None

        # Get most recent day's views
        today_views = weekly_views[-1].get('views', 0)

        if today_views > self.thresholds['view_spike_daily']:
            return {
                'type': 'view_spike',
                'severity': 'medium',
                'message': f"Page views today ({today_views}) exceeded threshold ({self.thresholds['view_spike_daily']})",
                'metric': 'page_views',
                'value': today_views,
                'threshold': self.thresholds['view_spike_daily'],
                'timestamp': datetime.now().isoformat()
            }

        return None

    def check_clone_spike(self, data: Dict) -> Optional[Dict]:
        """Check for unusual spike in repository clones"""
        weekly_clones = data.get('weekly_clone_pattern', [])
        if not weekly_clones:
            return None

        # Get most recent day's clones
        today_clones = weekly_clones[-1].get('clone_events', 0)

        if today_clones > self.thresholds['clone_spike_daily']:
            return {
                'type': 'clone_spike',
                'severity': 'high',
                'message': f"Repository clones today ({today_clones}) exceeded threshold ({self.thresholds['clone_spike_daily']})",
                'metric': 'repository_clones',
                'value': today_clones,
                'threshold': self.thresholds['clone_spike_daily'],
                'timestamp': datetime.now().isoformat()
            }

        return None

    def check_source_diversity(self, data: Dict) -> Optional[Dict]:
        """Check for high clone source diversity"""
        summary = data.get('summary', {})
        distinct_sources = summary.get('distinct_clone_sources', 0)

        if distinct_sources > self.thresholds['source_diversity_high']:
            return {
                'type': 'high_source_diversity',
                'severity': 'low',
                'message': f"Clone source diversity ({distinct_sources}) indicates broad interest",
                'metric': 'clone_sources',
                'value': distinct_sources,
                'threshold': self.thresholds['source_diversity_high'],
                'timestamp': datetime.now().isoformat()
            }

        return None

    def check_growth_rate(self, current: Dict, previous: Dict) -> List[Dict]:
        """Check for anomalous growth rates"""
        alerts = []

        if not previous:
            return alerts

        curr_summary = current.get('summary', {})
        prev_summary = previous.get('summary', {})

        # Check view growth
        curr_views = curr_summary.get('total_page_views', 0)
        prev_views = prev_summary.get('total_page_views', 0)

        if prev_views > 0:
            view_growth = curr_views / prev_views
            if view_growth >= self.thresholds['view_growth_rate']:
                alerts.append({
                    'type': 'view_growth_anomaly',
                    'severity': 'medium',
                    'message': f"Page views grew {view_growth:.1f}× since last check",
                    'metric': 'view_growth',
                    'value': view_growth,
                    'threshold': self.thresholds['view_growth_rate'],
                    'timestamp': datetime.now().isoformat()
                })

        # Check clone growth
        curr_clones = curr_summary.get('total_repository_clones', 0)
        prev_clones = prev_summary.get('total_repository_clones', 0)

        if prev_clones > 0:
            clone_growth = curr_clones / prev_clones
            if clone_growth >= self.thresholds['clone_growth_rate']:
                alerts.append({
                    'type': 'clone_growth_anomaly',
                    'severity': 'high',
                    'message': f"Repository clones grew {clone_growth:.1f}× since last check",
                    'metric': 'clone_growth',
                    'value': clone_growth,
                    'threshold': self.thresholds['clone_growth_rate'],
                    'timestamp': datetime.now().isoformat()
                })

        return alerts

    def check_sustained_activity(self, data: Dict) -> Optional[Dict]:
        """Check for sustained high activity over multiple days"""
        weekly_clones = data.get('weekly_clone_pattern', [])
        if not weekly_clones or len(weekly_clones) < 3:
            return None

        # Check last 3 days
        recent_activity = [day.get('clone_events', 0) for day in weekly_clones[-3:]]
        avg_activity = sum(recent_activity) / len(recent_activity)

        if avg_activity > self.thresholds['clone_spike_daily'] / 2:
            self.consecutive_high_activity_days += 1
        else:
            self.consecutive_high_activity_days = 0

        if self.consecutive_high_activity_days >= self.thresholds['sustained_high_activity']:
            return {
                'type': 'sustained_high_activity',
                'severity': 'medium',
                'message': f"Sustained high activity for {self.consecutive_high_activity_days} days (avg {avg_activity:.0f} clones/day)",
                'metric': 'sustained_activity',
                'value': self.consecutive_high_activity_days,
                'threshold': self.thresholds['sustained_high_activity'],
                'timestamp': datetime.now().isoformat()
            }

        return None

    def run_checks(self) -> List[Dict]:
        """Run all alert checks"""
        current_data = self.load_latest_data()

        if not current_data:
            return []

        alerts = []

        # Individual checks
        alert = self.check_view_spike(current_data)
        if alert:
            alerts.append(alert)

        alert = self.check_clone_spike(current_data)
        if alert:
            alerts.append(alert)

        alert = self.check_source_diversity(current_data)
        if alert:
            alerts.append(alert)

        alert = self.check_sustained_activity(current_data)
        if alert:
            alerts.append(alert)

        # Growth rate checks (requires previous data)
        if self.previous_data:
            growth_alerts = self.check_growth_rate(current_data, self.previous_data)
            alerts.extend(growth_alerts)

        # Update previous data
        self.previous_data = current_data

        return alerts

    def log_alert(self, alert: Dict):
        """Log alert to file"""
        with open(self.alert_log, 'a') as f:
            f.write(json.dumps(alert) + '\n')

        self.alert_history.append(alert)

    def display_alert(self, alert: Dict):
        """Display alert to console with formatting"""
        severity_icons = {
            'high': '🔴',
            'medium': '🟡',
            'low': '🟢'
        }

        icon = severity_icons.get(alert['severity'], '⚪')
        timestamp = alert.get('timestamp', 'unknown')

        print(f"\n{icon} ALERT [{alert['severity'].upper()}] - {timestamp}")
        print(f"   Type: {alert['type']}")
        print(f"   {alert['message']}")
        print()

    def run_continuous(self):
        """Run alert system continuously"""
        print("=" * 80)
        print("Privacy-Preserving Repository Insights - Alert System")
        print("=" * 80)
        print(f"Check interval: {self.check_interval} seconds")
        print(f"Alert log: {self.alert_log}")
        print()
        print("Configured thresholds:")
        for key, value in self.thresholds.items():
            print(f"  - {key}: {value}")
        print()
        print("Press Ctrl+C to stop")
        print("=" * 80)
        print()

        try:
            while True:
                alerts = self.run_checks()

                if alerts:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {len(alerts)} alert(s) detected")
                    for alert in alerts:
                        self.log_alert(alert)
                        self.display_alert(alert)
                else:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No alerts - all metrics within normal ranges ✅")

                print(f"Next check in {self.check_interval} seconds...")
                print()
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n")
            print("=" * 80)
            print("Alert system stopped")
            print(f"Total alerts generated: {len(self.alert_history)}")
            print("=" * 80)

    def get_alert_summary(self) -> str:
        """Generate summary of recent alerts"""
        if not self.alert_log.exists():
            return "No alerts recorded"

        # Load all alerts
        alerts = []
        with open(self.alert_log, 'r') as f:
            for line in f:
                alerts.append(json.loads(line.strip()))

        if not alerts:
            return "No alerts recorded"

        # Summary by type
        alert_counts = {}
        for alert in alerts:
            alert_type = alert['type']
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1

        output = []
        output.append("=" * 80)
        output.append("ALERT SUMMARY")
        output.append("=" * 80)
        output.append(f"Total alerts: {len(alerts)}")
        output.append(f"First alert: {alerts[0]['timestamp']}")
        output.append(f"Latest alert: {alerts[-1]['timestamp']}")
        output.append("")
        output.append("Alerts by type:")
        for alert_type, count in sorted(alert_counts.items(), key=lambda x: x[1], reverse=True):
            output.append(f"  - {alert_type}: {count}")
        output.append("")

        return '\n'.join(output)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Privacy-Preserving Repository Insights Alert System"
    )
    parser.add_argument(
        '--data-dir',
        default='repository_insights',
        help='Directory containing insights data'
    )
    parser.add_argument(
        '--alert-log',
        default='repository_insights/alerts.jsonl',
        help='Alert log file'
    )
    parser.add_argument(
        '--check-interval',
        type=int,
        default=3600,
        help='Check interval in seconds (default: 3600 = 1 hour)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show alert summary and exit'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run checks once and exit'
    )

    args = parser.parse_args()

    alert_system = AlertSystem(
        data_dir=Path(args.data_dir),
        alert_log=Path(args.alert_log),
        check_interval=args.check_interval
    )

    if args.summary:
        print(alert_system.get_alert_summary())
    elif args.once:
        alerts = alert_system.run_checks()
        if alerts:
            print(f"{len(alerts)} alert(s) detected:")
            for alert in alerts:
                alert_system.display_alert(alert)
                alert_system.log_alert(alert)
        else:
            print("✅ No alerts - all metrics within normal ranges")
    else:
        alert_system.run_continuous()


if __name__ == '__main__':
    main()
