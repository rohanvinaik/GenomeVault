#!/usr/bin/env python3
"""
k=3 GDiff Benchmark Monitor - Repurposed from graphical_pipeline_tracker.py

Real-time monitoring of whole-genome differential encoding with:
- Chromosome-level progress tracking
- Worker status and resource usage
- Progress bars and color-coded status
- Time estimates and completion predictions
- Live log tailing

Usage:
    python scripts/monitor_k3_gdiff_benchmark.py
    python scripts/monitor_k3_gdiff_benchmark.py --watch  # Auto-refresh every 30s
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GRAY = '\033[90m'

def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')

def draw_progress_bar(percentage, width=40, label=""):
    """Draw a colorful progress bar."""
    filled = int(width * percentage / 100)
    empty = width - filled

    # Color based on progress
    if percentage < 30:
        color = Colors.FAIL
    elif percentage < 70:
        color = Colors.WARNING
    else:
        color = Colors.OKGREEN

    bar = f"{color}{'█' * filled}{Colors.GRAY}{'░' * empty}{Colors.ENDC}"
    return f"{label} [{bar}] {percentage:.1f}%"

def draw_box(title, content, width=80):
    """Draw a box around content."""
    lines = content.split('\n')

    result = f"{Colors.BOLD}╔{'═' * (width - 2)}╗{Colors.ENDC}\n"
    result += f"{Colors.BOLD}║{Colors.OKBLUE} {title.center(width - 4)} {Colors.BOLD}║{Colors.ENDC}\n"
    result += f"{Colors.BOLD}╠{'═' * (width - 2)}╣{Colors.ENDC}\n"

    for line in lines:
        padding = width - len(line) - 4
        result += f"{Colors.BOLD}║{Colors.ENDC} {line}{' ' * padding} {Colors.BOLD}║{Colors.ENDC}\n"

    result += f"{Colors.BOLD}╚{'═' * (width - 2)}╝{Colors.ENDC}"
    return result

def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m {int(seconds%60)}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"

def format_size(bytes):
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} TB"

class K3BenchmarkMonitor:
    """Monitor k=3 whole genome GDiff benchmark."""

    CHROMOSOMES = [
        'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
        'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
        'chr20', 'chr21', 'chr22', 'chrX', 'chrY'
    ]

    def __init__(self):
        self.output_dir = Path("benchmark_results/k3_whole_genome_benchmark")
        self.log_file = Path("benchmark_results/k3_whole_genome_benchmark_DEBUG.log")
        self.process_name = "run_k3_whole_genome_benchmark.py"
        self.start_time = None

    def get_process_info(self) -> Optional[Dict]:
        """Get information about the running benchmark process."""
        try:
            result = subprocess.run(
                f"ps aux | grep '{self.process_name}' | grep -v grep",
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
                'etime': fields[9],
                'running': True
            }
        except Exception:
            return None

    def get_worker_info(self) -> List[Dict]:
        """Get information about worker processes."""
        workers = []
        try:
            proc_info = self.get_process_info()
            if not proc_info:
                return workers

            parent_pid = proc_info['pid']

            # Find child processes (workers)
            result = subprocess.run(
                f"ps aux | grep -E 'multiprocessing.*spawn_main' | grep -v grep",
                shell=True,
                capture_output=True,
                text=True
            )

            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                fields = line.split()
                if len(fields) < 10:
                    continue

                workers.append({
                    'pid': int(fields[1]),
                    'cpu_percent': float(fields[2]),
                    'mem_percent': float(fields[3]),
                    'time': fields[9],
                })

        except Exception:
            pass

        return workers

    def parse_log_progress(self) -> Dict:
        """Parse log file for chromosome processing progress."""
        progress = {
            'chromosomes_started': [],
            'chromosomes_completed': [],
            'current_stage': 'Unknown',
            'last_update': None,
        }

        if not self.log_file.exists():
            return progress

        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                # Check for worker starts
                match = re.search(r'\[WORKER (chr\w+_consensus)\] Starting', line)
                if match:
                    chrom = match.group(1).replace('_consensus', '')
                    if chrom not in progress['chromosomes_started']:
                        progress['chromosomes_started'].append(chrom)

                # Check for stage updates
                if 'Computing GDiff Differential Encoding' in line:
                    progress['current_stage'] = 'Computing Differential Encoding'
                elif 'Saving GDiff document' in line:
                    progress['current_stage'] = 'Saving GDiff Document'
                elif 'COMPLETE' in line:
                    progress['current_stage'] = 'Complete'

            # Get last modification time
            progress['last_update'] = datetime.fromtimestamp(self.log_file.stat().st_mtime)

        except Exception:
            pass

        return progress

    def get_output_file_size(self) -> int:
        """Get size of output GDiff file if it exists."""
        output_file = self.output_dir / "experimental.gdiff.gz"
        if output_file.exists():
            return output_file.stat().st_size
        return 0

    def get_start_time(self) -> Optional[datetime]:
        """Get benchmark start time from log file."""
        if self.log_file.exists():
            return datetime.fromtimestamp(self.log_file.stat().st_ctime)
        return None

    def display_status(self):
        """Display comprehensive benchmark status."""
        clear_screen()

        # Header
        print(draw_box(
            "k=3 WHOLE GENOME GDIFF BENCHMARK MONITOR 🔬",
            f"Real-time monitoring of differential encoding pipeline\n"
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            width=90
        ))
        print()

        # Process information
        proc_info = self.get_process_info()
        if not proc_info:
            print(f"{Colors.FAIL}❌ Benchmark not running!{Colors.ENDC}")
            return

        start_time = self.get_start_time()
        if start_time:
            runtime = (datetime.now() - start_time).total_seconds()
            runtime_str = format_time(runtime)
        else:
            runtime = 0
            runtime_str = "Unknown"

        # Main process info
        content = (
            f"PID: {proc_info['pid']}\n"
            f"State: {Colors.OKGREEN}{proc_info['state']}{Colors.ENDC}\n"
            f"Runtime: {Colors.OKCYAN}{runtime_str}{Colors.ENDC}\n"
            f"CPU: {proc_info['cpu_percent']}% | RAM: {proc_info['mem_percent']}%"
        )
        print(draw_box("MAIN PROCESS", content, width=90))
        print()

        # Worker processes
        workers = self.get_worker_info()
        if workers:
            worker_lines = []
            total_cpu = 0
            for i, worker in enumerate(workers, 1):
                cpu_time_parts = worker['time'].split(':')
                if len(cpu_time_parts) == 2:
                    cpu_minutes = int(cpu_time_parts[0])
                    cpu_seconds = float(cpu_time_parts[1])
                    total_cpu_time = cpu_minutes * 60 + cpu_seconds
                else:
                    total_cpu_time = 0

                total_cpu += worker['cpu_percent']

                worker_lines.append(
                    f"Worker {i} [PID {worker['pid']}]: "
                    f"CPU={Colors.OKGREEN}{worker['cpu_percent']}%{Colors.ENDC} | "
                    f"MEM={worker['mem_percent']}% | "
                    f"TIME={Colors.OKCYAN}{worker['time']}{Colors.ENDC}"
                )

            worker_lines.append(f"\nTotal CPU Usage: {Colors.BOLD}{total_cpu:.1f}%{Colors.ENDC}")
            print(draw_box("WORKER PROCESSES", '\n'.join(worker_lines), width=90))
            print()

        # Progress tracking
        progress = self.parse_log_progress()

        # Chromosome progress
        total_chroms = len(self.CHROMOSOMES)
        started_count = len(progress['chromosomes_started'])

        # Estimate progress (very rough - workers don't report completion)
        progress_pct = min(95, (started_count / total_chroms) * 100)

        chrom_content = (
            f"Total Chromosomes: {total_chroms}\n"
            f"Started: {Colors.OKGREEN}{started_count}{Colors.ENDC}\n"
            f"Processing: {', '.join(progress['chromosomes_started'][:5])}"
            + ("..." if len(progress['chromosomes_started']) > 5 else "") +
            f"\n\n{draw_progress_bar(progress_pct, width=60, label='Overall')}"
        )
        print(draw_box("CHROMOSOME PROGRESS", chrom_content, width=90))
        print()

        # Output file tracking
        output_size = self.get_output_file_size()
        expected_size = 150 * 1024 * 1024  # 150 MB

        if output_size > 0:
            size_pct = min(100, (output_size / expected_size) * 100)
            output_content = (
                f"Output File: experimental.gdiff.gz\n"
                f"Current Size: {Colors.OKCYAN}{format_size(output_size)}{Colors.ENDC}\n"
                f"Expected Size: ~{format_size(expected_size)}\n"
                f"\n{draw_progress_bar(size_pct, width=60, label='File Size')}"
            )
        else:
            output_content = (
                f"Output File: {Colors.WARNING}Not created yet{Colors.ENDC}\n"
                f"Expected Size: ~{format_size(expected_size)}\n"
                f"Status: Workers are processing chromosomes..."
            )
        print(draw_box("OUTPUT FILE", output_content, width=90))
        print()

        # Time estimates
        if runtime > 0 and started_count > 0:
            # Very rough estimate
            time_per_chrom = runtime / started_count
            remaining_chroms = total_chroms - started_count
            estimated_remaining = time_per_chrom * remaining_chroms

            eta = datetime.now() + timedelta(seconds=estimated_remaining)

            time_content = (
                f"Current Stage: {Colors.OKCYAN}{progress['current_stage']}{Colors.ENDC}\n"
                f"Runtime: {runtime_str}\n"
                f"Est. Remaining: {Colors.WARNING}{format_time(estimated_remaining)}{Colors.ENDC}\n"
                f"Est. Completion: {eta.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"\n{Colors.GRAY}Note: Estimate is rough - actual time varies per chromosome{Colors.ENDC}"
            )
        else:
            time_content = (
                f"Current Stage: {Colors.OKCYAN}{progress['current_stage']}{Colors.ENDC}\n"
                f"Runtime: {runtime_str}\n"
                f"Est. Remaining: {Colors.WARNING}Calculating...{Colors.ENDC}\n"
                f"\n{Colors.GRAY}Processing initial chromosomes to calculate estimate{Colors.ENDC}"
            )
        print(draw_box("TIME ESTIMATES", time_content, width=90))
        print()

        # Recent log output
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            recent_lines = [line.strip() for line in lines[-5:] if line.strip()]
            log_content = '\n'.join(recent_lines)
        except Exception:
            log_content = "Unable to read log file"

        print(draw_box("RECENT LOG OUTPUT", log_content, width=90))
        print()

        # Instructions
        print(f"{Colors.GRAY}Press Ctrl+C to exit monitor{Colors.ENDC}")
        print(f"{Colors.GRAY}Log file: {self.log_file}{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(description="Monitor k=3 GDiff Benchmark")
    parser.add_argument('--watch', action='store_true', help='Auto-refresh every 30 seconds')
    parser.add_argument('--interval', type=int, default=30, help='Refresh interval in seconds')
    args = parser.parse_args()

    monitor = K3BenchmarkMonitor()

    try:
        if args.watch:
            while True:
                monitor.display_status()
                time.sleep(args.interval)
        else:
            monitor.display_status()
    except KeyboardInterrupt:
        print(f"\n{Colors.OKGREEN}Monitor stopped by user{Colors.ENDC}")
        sys.exit(0)

if __name__ == "__main__":
    main()
