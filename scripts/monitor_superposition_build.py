#!/usr/bin/env python3
"""
Monitor Superposition Genome Build Progress

Tracks the iterative superposition consensus building process with visual progress bars.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Process to monitor
BUILD_PID = 90762

# Output directory
OUTPUT_DIR = Path("benchmark_results/superposition_iter1")

# Expected output files
EXPECTED_FILES = {
    "consensus.fa": "Consensus FASTA",
    "consensus.fa.fai": "FASTA Index",
    "metadata.json": "Build Metadata",
    "statistics.json": "Statistics",
}

def get_process_info(pid):
    """Get process information from ps."""
    try:
        # Get process info
        cmd = f"ps aux | grep {pid} | grep -v grep"
        output = os.popen(cmd).read().strip()

        if not output:
            return None

        parts = output.split()
        if len(parts) < 10:
            return None

        return {
            "pid": parts[1],
            "cpu": parts[2],
            "mem": parts[3],
            "state": parts[7],
            "elapsed": parts[9],
            "command": " ".join(parts[10:])
        }
    except:
        return None

def get_file_size(filepath):
    """Get file size in bytes."""
    try:
        return os.path.getsize(filepath)
    except:
        return 0

def format_size(bytes_size):
    """Format bytes to human readable."""
    if bytes_size == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def draw_progress_bar(percentage, width=40):
    """Draw a progress bar."""
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percentage:6.2f}%"

def parse_elapsed_time(elapsed_str):
    """Parse elapsed time string to seconds."""
    try:
        parts = elapsed_str.split(':')
        if len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except:
        pass
    return 0

def estimate_completion(elapsed_seconds, current_progress):
    """Estimate time to completion based on current progress."""
    if current_progress == 0:
        return "Unknown"

    total_estimated = (elapsed_seconds / current_progress) * 100
    remaining = total_estimated - elapsed_seconds

    if remaining < 60:
        return f"{int(remaining)}s"
    elif remaining < 3600:
        return f"{int(remaining/60)}m {int(remaining%60)}s"
    else:
        hours = int(remaining / 3600)
        mins = int((remaining % 3600) / 60)
        return f"{hours}h {mins}m"

def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')

def monitor_build(watch=False, interval=10):
    """Monitor superposition build progress."""

    try:
        while True:
            clear_screen()

            print("=" * 80)
            print("SUPERPOSITION GENOME BUILD MONITOR")
            print("=" * 80)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Iteration: 1 of 3")
            print(f"References: hg38, hg19, CHM13")
            print(f"Output: {OUTPUT_DIR}")
            print("=" * 80)
            print()

            # Get process info
            proc_info = get_process_info(BUILD_PID)

            if proc_info is None:
                print("⚠️  BUILD PROCESS NOT RUNNING")
                print(f"   PID {BUILD_PID} not found")
                print()
                print("Checking output directory for results...")
                print()
            else:
                print("PROCESS STATUS:")
                print(f"  PID:     {proc_info['pid']}")
                print(f"  State:   {proc_info['state']}")
                print(f"  CPU:     {proc_info['cpu']}%")
                print(f"  Memory:  {proc_info['mem']}%")
                print(f"  Elapsed: {proc_info['elapsed']}")
                print()

                # Estimate progress based on elapsed time
                elapsed_seconds = parse_elapsed_time(proc_info['elapsed'])

                # Rough estimate: 30-90 min for full genome, assume 60 min average
                estimated_total = 60 * 60  # 60 minutes
                progress_pct = min((elapsed_seconds / estimated_total) * 100, 99.0)

                progress_bar = draw_progress_bar(progress_pct, width=50)
                print(f"ESTIMATED PROGRESS:")
                print(f"  {progress_bar}")
                print(f"  Time elapsed: {int(elapsed_seconds/60)}m {int(elapsed_seconds%60)}s")

                if progress_pct < 99:
                    eta = estimate_completion(elapsed_seconds, progress_pct)
                    print(f"  ETA: ~{eta} (rough estimate)")
                else:
                    print(f"  ETA: Finalizing output files...")
                print()

            # Check output files
            print("OUTPUT FILES:")
            files_found = 0
            total_size = 0

            for filename, description in EXPECTED_FILES.items():
                filepath = OUTPUT_DIR / filename
                size = get_file_size(filepath)

                if size > 0:
                    status = "✓"
                    files_found += 1
                    total_size += size
                    size_str = format_size(size)
                else:
                    status = "⏳"
                    size_str = "Pending"

                print(f"  {status} {filename:<25} {description:<20} {size_str:>10}")

            print()
            print(f"  Files created: {files_found}/{len(EXPECTED_FILES)}")
            print(f"  Total size: {format_size(total_size)}")
            print()

            # Check for any other files in output directory
            if OUTPUT_DIR.exists():
                all_files = list(OUTPUT_DIR.glob("*"))
                if all_files:
                    other_files = [f for f in all_files if f.name not in EXPECTED_FILES]
                    if other_files:
                        print("  Additional files:")
                        for f in other_files[:5]:  # Show first 5
                            size = get_file_size(f)
                            print(f"    - {f.name} ({format_size(size)})")
                        if len(other_files) > 5:
                            print(f"    ... and {len(other_files)-5} more")
                        print()

            # Memory usage estimate
            if proc_info:
                mem_pct = float(proc_info['mem'])
                # System has ~65.5 GB RAM
                mem_gb = (mem_pct / 100) * 65.5
                print(f"MEMORY USAGE: {mem_gb:.1f} GB ({mem_pct}%)")
                print()

            print("=" * 80)

            if not proc_info and files_found == len(EXPECTED_FILES):
                print("✓ BUILD COMPLETE!")
                print()
                print("Next step: Iteration 2")
                print("  Merge GRCh38_no_alt, hs37d5, GRCh38_full into iteration 1 output")
                break
            elif not proc_info and files_found == 0:
                print("✗ BUILD FAILED - No output files created")
                print()
                print("Process stopped without producing output.")
                print("Check logs or try alternative approach:")
                print("  - Use single reference (hg38) instead of superposition")
                print("  - Reduce memory usage with chromosome-by-chromosome processing")
                break

            if watch:
                print(f"Refreshing every {interval} seconds... (Ctrl+C to stop)")
                time.sleep(interval)
            else:
                break

    except KeyboardInterrupt:
        print("\n\n✓ Monitor stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor superposition genome build")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval (seconds)")
    parser.add_argument("--pid", type=int, default=BUILD_PID, help="Process ID to monitor")

    args = parser.parse_args()

    if args.pid != BUILD_PID:
        BUILD_PID = args.pid

    monitor_build(watch=args.watch, interval=args.interval)
