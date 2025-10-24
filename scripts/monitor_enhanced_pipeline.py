#!/usr/bin/env python3
"""
Enhanced Privacy Pipeline Monitor

Real-time monitoring of the 4-layer privacy pipeline with:
- Process status and resource usage
- Layer-by-layer progress tracking
- File output monitoring
- Time estimates
- Live log tailing

Usage:
    python scripts/monitor_enhanced_pipeline.py
    python scripts/monitor_enhanced_pipeline.py --watch  # Auto-refresh every 10s
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PipelineMonitor:
    """Monitor the enhanced privacy pipeline execution."""

    def __init__(self, output_dir: Path, process_name: str = "run_enhanced_privacy_pipeline"):
        self.output_dir = Path(output_dir)
        self.process_name = process_name
        self.start_time = None

        # Expected layers
        self.layers = {
            'layer1_consensus': {
                'name': 'Layer 1: Superposition Consensus',
                'files': ['consensus.fa', 'superposition_consensus.fa'],
                'size_estimate': 50_000_000,  # 50 MB
            },
            'layer2_reference_pool': {
                'name': 'Layer 2: Rolling Reference Pool',
                'files': ['ref1.sorted.bam', 'ref2.sorted.bam', 'ref3.sorted.bam',
                         'ref1.vcf.gz', 'ref2.vcf.gz', 'ref3.vcf.gz'],
                'size_estimate': 5_000_000_000,  # 5 GB per BAM
            },
            'layer3_query_alignment': {
                'name': 'Layer 3: Query Alignment',
                'files': ['query.sorted.bam', 'query.vcf.gz'],
                'size_estimate': 5_000_000_000,  # 5 GB
            },
            'layer4_genomevault': {
                'name': 'Layer 4: GenomeVault Core',
                'files': ['differential_encoding.bin', 'hypervector.npy',
                         'zk_proof.json', 'pir_query.json'],
                'size_estimate': 100_000_000,  # 100 MB
            }
        }

    def get_process_info(self) -> Optional[Dict]:
        """Get information about the running pipeline process."""
        try:
            # Find process by name
            result = subprocess.run(
                f"ps aux | grep '{self.process_name}' | grep -v grep",
                shell=True,
                capture_output=True,
                text=True
            )

            if not result.stdout.strip():
                return None

            # Parse ps output
            lines = result.stdout.strip().split('\n')
            if not lines:
                return None

            # Get first matching process
            fields = lines[0].split()

            return {
                'pid': int(fields[1]),
                'cpu_percent': float(fields[2]),
                'mem_percent': float(fields[3]),
                'state': fields[7],
                'running': True
            }
        except Exception as e:
            return None

    def get_subprocess_info(self) -> List[Dict]:
        """Get information about alignment subprocesses (minimap2, samtools, bcftools)."""
        subprocesses = []
        tools = ['minimap2', 'samtools', 'bcftools', 'bwa', 'bowtie2']

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
                            # Parse command to get short description
                            cmd = ' '.join(fields[10:])
                            # Truncate long paths
                            if len(cmd) > 100:
                                cmd = cmd[:97] + "..."

                            # Get detailed timing info (elapsed time, CPU time)
                            pid = int(fields[1])
                            try:
                                time_result = subprocess.run(
                                    f"ps -p {pid} -o etime,cputime",
                                    shell=True,
                                    capture_output=True,
                                    text=True
                                )
                                time_lines = time_result.stdout.strip().split('\n')
                                if len(time_lines) > 1:
                                    time_fields = time_lines[1].split()
                                    elapsed_time = time_fields[0] if len(time_fields) > 0 else "0:00"
                                    cpu_time = time_fields[1] if len(time_fields) > 1 else "0:00"
                                else:
                                    elapsed_time = fields[9]
                                    cpu_time = fields[9]
                            except:
                                elapsed_time = fields[9]
                                cpu_time = fields[9]

                            subprocesses.append({
                                'tool': tool,
                                'pid': pid,
                                'cpu_percent': float(fields[2]),
                                'mem_percent': float(fields[3]),
                                'mem_gb': float(fields[3]) / 100 * self._get_total_memory_gb(),
                                'state': fields[7],
                                'elapsed_time': elapsed_time,
                                'cpu_time': cpu_time,
                                'command': cmd
                            })
            except:
                continue

        return subprocesses

    def _get_total_memory_gb(self) -> float:
        """Get total system memory in GB."""
        try:
            result = subprocess.run(
                "sysctl hw.memsize | awk '{print $2}'",
                shell=True,
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                bytes_mem = int(result.stdout.strip())
                return bytes_mem / (1024**3)
        except:
            pass
        return 64.0  # Default fallback

    def get_layer_progress(self, layer_dir: str) -> Dict:
        """Get progress for a specific layer."""
        layer_path = self.output_dir / layer_dir
        layer_info = self.layers[layer_dir]

        if not layer_path.exists():
            return {
                'status': 'pending',
                'files_found': 0,
                'files_expected': len(layer_info['files']),
                'size_bytes': 0,
                'progress_pct': 0.0,
                'temp_files': []
            }

        # Check which files exist
        files_found = []
        total_size = 0

        for filename in layer_info['files']:
            file_path = layer_path / filename
            if file_path.exists():
                files_found.append(filename)
                total_size += file_path.stat().st_size

        # Check for temporary files (indicates work in progress)
        temp_files = []
        try:
            for file_path in layer_path.glob('*.tmp.*'):
                temp_files.append({
                    'name': file_path.name,
                    'size': file_path.stat().st_size
                })
        except:
            pass

        # Determine status
        if len(files_found) == 0 and len(temp_files) == 0:
            status = 'pending'
        elif len(temp_files) > 0:
            status = 'in_progress'
        elif len(files_found) == len(layer_info['files']):
            status = 'complete'
        else:
            status = 'in_progress'

        # Calculate progress percentage
        progress_pct = (len(files_found) / len(layer_info['files'])) * 100

        return {
            'status': status,
            'files_found': len(files_found),
            'files_expected': len(layer_info['files']),
            'files': files_found,
            'size_bytes': total_size,
            'progress_pct': progress_pct,
            'temp_files': temp_files
        }

    def get_latest_log_lines(self, n: int = 10) -> List[str]:
        """Get the latest N lines from pipeline logs."""
        # Try to find log files
        log_patterns = [
            self.output_dir / 'pipeline.log',
            self.output_dir / '*.log',
        ]

        for pattern in log_patterns:
            if isinstance(pattern, Path) and pattern.exists():
                try:
                    with open(pattern, 'r') as f:
                        lines = f.readlines()
                        return lines[-n:] if lines else []
                except:
                    pass

        return []

    def estimate_time_remaining(self, process_info: Dict) -> Optional[timedelta]:
        """Estimate time remaining based on current progress."""
        # Get overall progress
        total_progress = 0
        total_layers = len(self.layers)

        for layer_dir in self.layers.keys():
            progress = self.get_layer_progress(layer_dir)
            total_progress += progress['progress_pct']

        overall_pct = total_progress / (total_layers * 100)

        if overall_pct < 0.01:  # Less than 1% complete
            return None

        # Estimate based on Layer 1 completion time (335s)
        # and typical scaling for whole-genome alignment
        layer1_time = 335  # seconds
        estimated_total = layer1_time + (60 * 60 * 1.5)  # ~1.5 hours for alignment

        elapsed = time.time() - (process_info.get('start_time', time.time()))
        if elapsed < 1:
            return None

        estimated_remaining = (estimated_total / overall_pct) - elapsed
        return timedelta(seconds=max(0, estimated_remaining))

    def format_bytes(self, bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024:
                return f"{bytes:.1f} {unit}"
            bytes /= 1024
        return f"{bytes:.1f} PB"

    def format_duration(self, seconds: float) -> str:
        """Format duration in seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def print_status(self, clear_screen: bool = True):
        """Print comprehensive pipeline status."""
        if clear_screen and sys.stdout.isatty():
            os.system('clear' if os.name != 'nt' else 'cls')

        print("=" * 80)
        print("ENHANCED PRIVACY PIPELINE MONITOR")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output Directory: {self.output_dir}")
        print()

        # Process status
        process = self.get_process_info()

        if process:
            print("🟢 PIPELINE PROCESS: RUNNING")
            print(f"  PID:        {process['pid']}")
            print(f"  CPU:        {process['cpu_percent']:.1f}%")
            print(f"  Memory:     {process['mem_percent']:.1f}%")
            print(f"  State:      {process['state']}")
        else:
            print("🔴 PIPELINE PROCESS: NOT RUNNING")
            print("  (Pipeline may have completed or failed)")

        print()

        # Subprocess status (the real workers)
        subprocesses = self.get_subprocess_info()
        if subprocesses:
            print("=" * 80)
            print("🔥 ACTIVE ALIGNMENT SUBPROCESSES (Doing the Heavy Lifting)")
            print("=" * 80)

            # Calculate totals
            total_cpu = sum(p['cpu_percent'] for p in subprocesses)
            total_mem = sum(p['mem_gb'] for p in subprocesses)

            print(f"\n  Total CPU Usage:    {total_cpu:.1f}% ({len(subprocesses)} processes)")
            print(f"  Total Memory:       {total_mem:.2f} GB")
            print()

            for proc in subprocesses:
                icon = "⚡" if proc['cpu_percent'] > 100 else "🔄"
                print(f"{icon} {proc['tool'].upper():<12} PID {proc['pid']:<7} "
                      f"CPU: {proc['cpu_percent']:>6.1f}%  "
                      f"Mem: {proc['mem_gb']:>5.2f}GB")
                print(f"   └─ Elapsed: {proc['elapsed_time']:<12} "
                      f"CPU Time: {proc['cpu_time']:<12} "
                      f"(~{proc['cpu_percent']/100:.1f}× parallelization)")

            print()
        else:
            print("⏸️  No active alignment subprocesses (waiting or between stages)")

        print()
        print("=" * 80)
        print("LAYER PROGRESS")
        print("=" * 80)

        # Layer-by-layer status
        overall_progress = 0
        for layer_dir, layer_info in self.layers.items():
            progress = self.get_layer_progress(layer_dir)
            overall_progress += progress['progress_pct']

            # Status icon
            if progress['status'] == 'complete':
                icon = '✅'
            elif progress['status'] == 'in_progress':
                icon = '🔄'
            else:
                icon = '⏳'

            print(f"\n{icon} {layer_info['name']}")
            print(f"  Status:     {progress['status'].upper()}")
            print(f"  Progress:   {progress['progress_pct']:.1f}% ({progress['files_found']}/{progress['files_expected']} files)")
            print(f"  Size:       {self.format_bytes(progress['size_bytes'])}")

            if progress.get('files'):
                print(f"  Files:      {', '.join(progress['files'][:3])}")
                if len(progress['files']) > 3:
                    print(f"              ... and {len(progress['files']) - 3} more")

            # Show temporary files (work in progress)
            if progress.get('temp_files'):
                num_temp = len(progress['temp_files'])
                print(f"  Temp Files: {num_temp} files being created")

                # Estimate progress based on typical temp file count
                # samtools typically creates ~30-40 temp files for whole genome
                if num_temp > 0:
                    estimated_total_temp = 35  # typical for whole genome
                    estimated_progress = min(95, (num_temp / estimated_total_temp) * 100)
                    print(f"  Estimated:  ~{estimated_progress:.0f}% complete (based on temp file count)")

                for temp in progress['temp_files'][:3]:
                    print(f"              - {temp['name']}: {self.format_bytes(temp['size'])}")
                if len(progress['temp_files']) > 3:
                    print(f"              ... and {len(progress['temp_files']) - 3} more")

        print()
        print("=" * 80)
        print(f"OVERALL PROGRESS: {overall_progress / (len(self.layers) * 100) * 100:.1f}%")
        print("=" * 80)

        # Progress bar
        bar_length = 60
        filled = int((overall_progress / (len(self.layers) * 100)) * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"[{bar}]")

        # Time estimate
        if process:
            time_remaining = self.estimate_time_remaining(process)
            if time_remaining:
                print(f"\nEstimated time remaining: {time_remaining}")

        print()

        # Latest outputs
        log_lines = self.get_latest_log_lines(5)
        if log_lines:
            print("=" * 80)
            print("LATEST LOG OUTPUT")
            print("=" * 80)
            for line in log_lines:
                print(line.rstrip())

        print()
        print("=" * 80)
        print("Commands:")
        print("  Ctrl+C:           Exit monitor")
        print("  --watch:          Auto-refresh every 10 seconds")
        print("  --tail-logs:      Tail full logs")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Monitor the Enhanced Privacy Pipeline execution'
    )
    parser.add_argument(
        '--output',
        default='benchmark_results/enhanced_privacy_pipeline',
        help='Pipeline output directory (default: benchmark_results/enhanced_privacy_pipeline)'
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Auto-refresh every 10 seconds'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Refresh interval in seconds (default: 10)'
    )
    parser.add_argument(
        '--no-clear',
        action='store_true',
        help='Do not clear screen between updates'
    )

    args = parser.parse_args()

    monitor = PipelineMonitor(
        output_dir=Path(args.output),
        process_name='run_enhanced_privacy_pipeline'
    )

    try:
        if args.watch:
            print("Starting watch mode (Ctrl+C to exit)...")
            while True:
                monitor.print_status(clear_screen=not args.no_clear)
                time.sleep(args.interval)
        else:
            monitor.print_status(clear_screen=False)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)


if __name__ == '__main__':
    main()
