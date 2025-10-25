#!/usr/bin/env python3
"""
GitHub Visitor & Traffic Monitor
Dual tracking system with both native GitHub API and external badge counters
Similar to genomic download monitoring system
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd):
    """Run shell command and return output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def get_github_traffic():
    """Get traffic data from GitHub API"""
    views = run_command("gh api repos/rohanvinaik/GenomeVault/traffic/views")
    clones = run_command("gh api repos/rohanvinaik/GenomeVault/traffic/clones")
    referrers = run_command("gh api repos/rohanvinaik/GenomeVault/traffic/popular/referrers")
    paths = run_command("gh api repos/rohanvinaik/GenomeVault/traffic/popular/paths")
    repo_stats = run_command("gh api repos/rohanvinaik/GenomeVault")

    data = {
        "timestamp": datetime.now().isoformat(),
        "views": json.loads(views) if views else {},
        "clones": json.loads(clones) if clones else {},
        "referrers": json.loads(referrers) if referrers else [],
        "popular_paths": json.loads(paths) if paths else [],
        "repo_stats": json.loads(repo_stats) if repo_stats else {}
    }

    return data


def save_traffic_history(data):
    """Save traffic data to history file"""
    history_file = Path("traffic_history/visitor_data.jsonl")
    history_file.parent.mkdir(exist_ok=True)

    with open(history_file, "a") as f:
        f.write(json.dumps(data) + "\n")


def print_traffic_summary(data):
    """Print formatted traffic summary"""
    views = data.get("views", {})
    clones = data.get("clones", {})
    repo = data.get("repo_stats", {})

    print("\n" + "="*60)
    print(f"  GitHub Traffic Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    print("\n📊 Repository Stats:")
    print(f"  Stars: {repo.get('stargazers_count', 0)}")
    print(f"  Forks: {repo.get('forks_count', 0)}")
    print(f"  Watchers: {repo.get('subscribers_count', 0)}")
    print(f"  Open Issues: {repo.get('open_issues_count', 0)}")

    print("\n👁️  Views (Last 14 Days):")
    print(f"  Total Views: {views.get('count', 0)}")
    print(f"  Unique Visitors: {views.get('uniques', 0)}")

    if views.get("views"):
        print("\n  Daily Breakdown:")
        for day in views["views"][-7:]:  # Last 7 days
            date = day["timestamp"][:10]
            print(f"    {date}: {day['count']} views ({day['uniques']} unique)")

    print("\n📥 Clones (Last 14 Days):")
    print(f"  Total Clones: {clones.get('count', 0)}")
    print(f"  Unique Cloners: {clones.get('uniques', 0)}")

    if clones.get("clones"):
        print("\n  Recent Activity:")
        for day in clones["clones"][-7:]:  # Last 7 days
            date = day["timestamp"][:10]
            print(f"    {date}: {day['count']} clones ({day['uniques']} unique)")

    print("\n🔗 Top Referrers:")
    if data.get("referrers"):
        for ref in data["referrers"][:5]:
            print(f"  {ref['referrer']}: {ref['count']} views ({ref['uniques']} unique)")
    else:
        print("  No referrer data (all direct traffic)")

    print("\n📄 Popular Pages:")
    if data.get("popular_paths"):
        for path in data["popular_paths"][:5]:
            print(f"  {path['path']}: {path['count']} views ({path['uniques']} unique)")
    else:
        print("  No page data available")

    print("\n" + "="*60)
    print(f"  History saved to: traffic_history/visitor_data.jsonl")
    print("="*60 + "\n")


def watch_traffic(interval=3600):
    """Watch traffic in real-time"""
    import time

    print("🔄 Starting visitor traffic monitor...")
    print(f"   Refreshing every {interval} seconds (Ctrl+C to stop)\n")

    try:
        while True:
            data = get_github_traffic()
            save_traffic_history(data)
            print_traffic_summary(data)

            if interval > 0:
                print(f"Waiting {interval} seconds before next update...")
                time.sleep(interval)
            else:
                break
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
        sys.exit(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Monitor GitHub visitor traffic")
    parser.add_argument("--watch", action="store_true", help="Watch in real-time")
    parser.add_argument("--interval", type=int, default=3600, help="Update interval in seconds (default: 3600 = 1 hour)")

    args = parser.parse_args()

    if args.watch:
        watch_traffic(args.interval)
    else:
        data = get_github_traffic()
        save_traffic_history(data)
        print_traffic_summary(data)


if __name__ == "__main__":
    main()
