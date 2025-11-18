#!/usr/bin/env python3
"""
Genomic Data Download Monitor

Real-time monitoring of genomic data downloads with:
- Download progress tracking
- Speed and ETA calculations
- Disk space monitoring
- Process resource usage
- Live graphical display

Usage:
    python scripts/monitor_genomic_downloads.py
    python scripts/monitor_genomic_downloads.py --watch  # Auto-refresh every 5s
    python scripts/monitor_genomic_downloads.py --state data/download_state.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class DownloadMonitor:
    """Monitor genomic data downloads with real-time updates."""

    def __init__(self, state_file: Path):
        self.state_file = Path(state_file)
        self.last_update = None

    def load_state(self) -> Optional[Dict]:
        """Load download state from JSON file."""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except:
            return None

    def get_process_info(self, process_name: str) -> Optional[Dict]:
        """Get information about a running process."""
        try:
            result = subprocess.run(
                f"ps aux | grep '{process_name}' | grep -v grep | grep -v monitor",
                shell=True,
                capture_output=True,
                text=True
            )

            if not result.stdout.strip():
                return None

            lines = result.stdout.strip().split('\n')
            if not lines:
                return None

            fields = lines[0].split()
            return {
                'pid': int(fields[1]),
                'cpu_percent': float(fields[2]),
                'mem_percent': float(fields[3]),
                'state': fields[7],
                'running': True
            }
        except:
            return None

    def get_active_downloads(self) -> List[Dict]:
        """Get information about active download processes."""
        downloads = []
        tools = ['fasterq-dump', 'prefetch', 'pigz']

        for tool in tools:
            try:
                result = subprocess.run(
                    f"ps aux | grep '{tool}' | grep -v grep",
                    shell=True,
                    capture_output=True,
                    text=True
                )

                if result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        fields = line.split()
                        if len(fields) >= 11:
                            cmd = ' '.join(fields[10:])
                            if len(cmd) > 80:
                                cmd = cmd[:77] + "..."

                            downloads.append({
                                'tool': tool,
                                'pid': int(fields[1]),
                                'cpu_percent': float(fields[2]),
                                'mem_percent': float(fields[3]),
                                'state': fields[7],
                                'command': cmd
                            })
            except:
                pass

        return downloads

    def get_disk_space(self, path: str = '.') -> Dict:
        """Get disk space information."""
        try:
            result = subprocess.run(
                ['df', '-BG', path],
                capture_output=True,
                text=True
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                fields = lines[1].split()
                total_gb = float(fields[1].replace('G', ''))
                used_gb = float(fields[2].replace('G', ''))
                available_gb = float(fields[3].replace('G', ''))
                percent_used = float(fields[4].replace('%', ''))

                return {
                    'total_gb': total_gb,
                    'used_gb': used_gb,
                    'available_gb': available_gb,
                    'percent_used': percent_used
                }
        except:
            pass

        # Default return for both exception and insufficient df output
        return {'total_gb': 0, 'used_gb': 0, 'available_gb': 0, 'percent_used': 0}

    def get_download_storage_info(self) -> Dict:
        """Get storage usage across all download stages."""
        storage = {
            'sra_cache_gb': 0.0,
            'intermediate_fastq_gb': 0.0,
            'compressed_fastq_gb': 0.0,
            'total_gb': 0.0
        }

        try:
            # Check SRA cache (~/.ncbi/public/sra/)
            sra_cache = Path.home() / '.ncbi' / 'public' / 'sra'
            if sra_cache.exists():
                result = subprocess.run(
                    ['du', '-sg', str(sra_cache)],
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    storage['sra_cache_gb'] = float(result.stdout.split()[0])
        except:
            pass

        try:
            # Also check for local SRA downloads (ERR*/ERR*.sra.tmp files)
            result = subprocess.run(
                ['find', '.', '-maxdepth', '2', '-name', '*.sra.tmp', '-type', 'f'],
                capture_output=True,
                text=True
            )
            if result.stdout:
                for sra_file in result.stdout.strip().split('\n'):
                    if sra_file:
                        try:
                            size = Path(sra_file).stat().st_size / (1024**3)
                            storage['sra_cache_gb'] += size
                        except:
                            pass
        except:
            pass

        try:
            # Check output directory for intermediate FASTQ and compressed files
            output_dir = Path('data/downloaded/fastq')
            if output_dir.exists():
                # Find uncompressed FASTQ files
                result = subprocess.run(
                    ['find', str(output_dir), '-name', '*.fastq', '-type', 'f'],
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    for fastq_file in result.stdout.strip().split('\n'):
                        if fastq_file:
                            try:
                                size = Path(fastq_file).stat().st_size / (1024**3)
                                storage['intermediate_fastq_gb'] += size
                            except:
                                pass

                # Find compressed FASTQ files
                result = subprocess.run(
                    ['find', str(output_dir), '-name', '*.fastq.gz', '-type', 'f'],
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    for gz_file in result.stdout.strip().split('\n'):
                        if gz_file:
                            try:
                                size = Path(gz_file).stat().st_size / (1024**3)
                                storage['compressed_fastq_gb'] += size
                            except:
                                pass
        except:
            pass

        storage['total_gb'] = (
            storage['sra_cache_gb'] +
            storage['intermediate_fastq_gb'] +
            storage['compressed_fastq_gb']
        )

        return storage

    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def format_size(self, gb: float) -> str:
        """Format size in GB."""
        if gb < 1:
            return f"{gb*1000:.0f} MB"
        return f"{gb:.2f} GB"

    def create_progress_bar(self, percent: float, width: int = 50) -> str:
        """Create a visual progress bar."""
        filled = int(width * percent / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {percent:.1f}%"

    def display_dashboard(self, state: Dict, downloads: List[Dict], disk: Dict, refresh_interval: int = 5):
        """Display comprehensive download dashboard."""
        # Clear screen
        os.system('clear' if os.name != 'nt' else 'cls')

        # Header
        print("╔" + "═" * 98 + "╗")
        print("║" + " " * 25 + "🧬 GENOMEVAULT DATA DOWNLOAD MONITOR 🧬" + " " * 33 + "║")
        print("╚" + "═" * 98 + "╝")
        print()

        # Overall Status
        print("┌─ 📊 OVERALL STATUS " + "─" * 77 + "┐")

        start_time = datetime.fromisoformat(state['start_time'])
        elapsed = (datetime.now() - start_time).total_seconds()

        total_samples = len(state['samples'])
        completed = sum(1 for s in state['samples'].values() if s['status'] == 'completed')
        downloading = sum(1 for s in state['samples'].values() if s['status'] == 'downloading')
        failed = sum(1 for s in state['samples'].values() if s['status'] == 'failed')
        queued = sum(1 for s in state['samples'].values() if s['status'] == 'queued')

        completion_percent = (completed / total_samples * 100) if total_samples > 0 else 0

        print(f"│  Pipeline Status: {state['status'].upper():<20} │ Elapsed Time: {self.format_duration(elapsed):<15} │")
        print(f"│  Total Samples:   {total_samples:<20} │ Downloaded:   {self.format_size(state['total_downloaded_gb']):<15} │")
        print(f"│")
        print(f"│  {self.create_progress_bar(completion_percent, width=92)}")
        print(f"│")
        print(f"│  ✅ Completed: {completed:<3}  │  ⏳ Downloading: {downloading:<3}  │  ❌ Failed: {failed:<3}  │  ⏸️  Queued: {queued:<3}")
        print("└" + "─" * 98 + "┘")
        print()

        # Disk Space with download storage breakdown
        print("┌─ 💾 DISK SPACE " + "─" * 81 + "┐")
        disk_bar = self.create_progress_bar(disk['percent_used'], width=70)
        print(f"│  {disk_bar}  │")
        print(f"│  Available: {self.format_size(disk['available_gb']):<15} │ Used: {self.format_size(disk['used_gb']):<15} │ Total: {self.format_size(disk['total_gb']):<15}  │")

        # Download storage breakdown
        storage = self.get_download_storage_info()
        if storage['total_gb'] > 0:
            print("│" + " " * 98 + "│")
            print(f"│  📊 Download Storage: {self.format_size(storage['total_gb']):<12}  │  " +
                  f"SRA: {self.format_size(storage['sra_cache_gb']):<12}  │  " +
                  f"FASTQ: {self.format_size(storage['intermediate_fastq_gb']):<12}  │  " +
                  f"Compressed: {self.format_size(storage['compressed_fastq_gb']):<12}  │")
        print("└" + "─" * 98 + "┘")
        print()

        # Active Downloads
        if downloads:
            print("┌─ 🔄 ACTIVE DOWNLOAD PROCESSES " + "─" * 66 + "┐")
            for dl in downloads[:5]:  # Show up to 5 active processes
                tool_emoji = {'fasterq-dump': '📥', 'prefetch': '⬇️', 'pigz': '🗜️'}.get(dl['tool'], '⚙️')
                print(f"│  {tool_emoji} {dl['tool']:<15} │ PID: {dl['pid']:<8} │ CPU: {dl['cpu_percent']:>5.1f}% │ MEM: {dl['mem_percent']:>5.1f}%  │")
                print(f"│     {dl['command']:<93}│")
            print("└" + "─" * 98 + "┘")
            print()

        # Sample Progress
        print("┌─ 📦 SAMPLE DOWNLOAD STATUS " + "─" * 69 + "┐")

        # Show recent samples (last 10)
        recent_samples = sorted(
            state['samples'].values(),
            key=lambda x: x.get('start_time', '') or '',
            reverse=True
        )[:10]

        for sample in recent_samples:
            acc = sample['accession']
            status = sample['status']
            size_gb = sample.get('size_gb', 0.0)

            # Status emoji and color
            status_map = {
                'completed': ('✅', 'Completed'),
                'downloading': ('⏳', 'Downloading'),
                'failed': ('❌', 'Failed'),
                'queued': ('⏸️ ', 'Queued')
            }
            emoji, status_text = status_map.get(status, ('❓', status.capitalize()))

            # Calculate duration
            duration = ""
            if sample.get('start_time'):
                start = datetime.fromisoformat(sample['start_time'])
                if sample.get('end_time'):
                    end = datetime.fromisoformat(sample['end_time'])
                    duration = self.format_duration((end - start).total_seconds())
                else:
                    duration = self.format_duration((datetime.now() - start).total_seconds())

            # Format line
            pool = sample.get('pool', 'unknown')[:10]
            sample_type = sample.get('sample_type', 'ref')[:5]

            print(f"│  {emoji} {acc:<15} │ {pool:<10} │ {sample_type:<5} │ {status_text:<12} │ {self.format_size(size_gb):<10} │ {duration:<8}  │")

        if len(state['samples']) > 10:
            remaining = len(state['samples']) - 10
            print(f"│  ... and {remaining} more samples")

        print("└" + "─" * 98 + "┘")
        print()

        # Footer
        if refresh_interval > 0:
            print(f"⏱️  Auto-refreshing every {refresh_interval} seconds... (Press Ctrl+C to stop)")
            print(f"📄 State file: {self.state_file}")
        else:
            print(f"📄 State file: {self.state_file}")
            print("💡 Run with --watch for auto-refresh")

    def run(self, watch: bool = False, interval: int = 5):
        """Run the monitor."""
        try:
            while True:
                # Load current state
                state = self.load_state()

                if state is None:
                    print("❌ No download state file found.")
                    print(f"Expected location: {self.state_file}")
                    print("\nStart a download first:")
                    print("  python scripts/download_genomic_data_automated.py --pool european --samples 3")
                    sys.exit(1)

                # Get active downloads
                downloads = self.get_active_downloads()

                # Get disk space
                disk = self.get_disk_space('data')

                # Display dashboard
                self.display_dashboard(state, downloads, disk, interval if watch else 0)

                # Break if not watching
                if not watch:
                    break

                # Wait for next refresh
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n👋 Monitor stopped")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Monitor genomic data downloads in real-time'
    )
    parser.add_argument(
        '--state',
        default='data/download_state.json',
        help='Path to download state JSON file'
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Auto-refresh the display'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Refresh interval in seconds (default: 5)'
    )

    args = parser.parse_args()

    monitor = DownloadMonitor(state_file=Path(args.state))
    monitor.run(watch=args.watch, interval=args.interval)


if __name__ == '__main__':
    main()
