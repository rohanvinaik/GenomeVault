#!/usr/bin/env python3
"""
Monitor Sequential Superposition Genome Build

Tracks the three-phase sequential chromosome-by-chromosome build process:
1. File format conversion (gzip → bgzip)
2. Chromosome consensus building (24 chromosomes × 7 references)
3. Final assembly
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Build configuration
OUTPUT_DIR = Path("benchmark_results/superposition_7refs_sequential")
BGZIP_DIR = OUTPUT_DIR / "bgzip_refs"
TEMP_DIR = OUTPUT_DIR / "temp"
LOG_FILE = Path("logs/superposition_sequential_build_fixed.log")

# Expected files
EXPECTED_GENOMES = [
    "hg38.fa.gz",
    "hg19.fa.gz",
    "chm13v2.0.fa.gz",
    "GRCh38_no_alt.fa.gz",
    "hs37d5.fa.gz",
    "GRCh38_full_analysis_set.fa.gz",
    "hg18.fa.gz"
]

# Human chromosomes
CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

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
    return f"[{bar}] {percentage:5.1f}%"

def get_build_process():
    """Find the build process."""
    try:
        cmd = "ps aux | grep build_superposition_sequential_chromosomes.sh | grep -v grep"
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
            "state": parts[7]
        }
    except:
        return None

def get_conversion_process():
    """Find active gunzip/bgzip processes."""
    try:
        cmd = "ps aux | grep -E '(gunzip|bgzip)' | grep -v grep | head -2"
        output = os.popen(cmd).read().strip()
        if not output:
            return None

        processes = []
        for line in output.split('\n'):
            parts = line.split()
            if len(parts) >= 11:
                processes.append({
                    "pid": parts[1],
                    "cpu": parts[2],
                    "mem": parts[3],
                    "cmd": " ".join(parts[10:14])
                })
        return processes if processes else None
    except:
        return None

def check_conversion_status():
    """Check which genomes have been converted to bgzip."""
    converted = []
    for genome in EXPECTED_GENOMES:
        bgzip_file = BGZIP_DIR / genome
        fai_file = BGZIP_DIR / f"{genome}.fai"

        if bgzip_file.exists() and fai_file.exists():
            size = get_file_size(bgzip_file)
            converted.append((genome, size, True))
        elif bgzip_file.exists():
            size = get_file_size(bgzip_file)
            converted.append((genome, size, False))
        else:
            converted.append((genome, 0, False))

    return converted

def check_chromosome_status():
    """Check which chromosomes have consensus built."""
    completed = []
    if TEMP_DIR.exists():
        for chr_name in CHROMOSOMES:
            consensus_file = TEMP_DIR / chr_name / f"consensus_{chr_name}.fa.gz"
            if consensus_file.exists():
                size = get_file_size(consensus_file)
                completed.append((chr_name, size))

    return completed

def get_latest_log_lines(n=10):
    """Get last N lines from log file."""
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE) as f:
                lines = f.readlines()
                return lines[-n:] if lines else []
    except:
        pass
    return []

def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')

def monitor_build(watch=False, interval=15):
    """Monitor the sequential superposition build."""

    try:
        while True:
            clear_screen()

            print("=" * 80)
            print("SEQUENTIAL SUPERPOSITION BUILD MONITOR")
            print("=" * 80)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Output: {OUTPUT_DIR}")
            print("=" * 80)
            print()

            # Get process status
            build_proc = get_build_process()
            conversion_procs = get_conversion_process()

            if build_proc:
                print("BUILD PROCESS:")
                print(f"  PID:    {build_proc['pid']}")
                print(f"  State:  {build_proc['state']}")
                print(f"  CPU:    {build_proc['cpu']}%")
                print(f"  Memory: {build_proc['mem']}%")
                print()

            # Phase 1: Conversion Status
            print("PHASE 1: FILE FORMAT CONVERSION (gzip → bgzip)")
            print("-" * 80)

            converted = check_conversion_status()
            complete_count = sum(1 for _, _, complete in converted if complete)
            conversion_pct = (complete_count / len(EXPECTED_GENOMES)) * 100

            print(f"Progress: {draw_progress_bar(conversion_pct, width=50)}")
            print(f"Completed: {complete_count}/{len(EXPECTED_GENOMES)} genomes")
            print()

            for genome, size, complete in converted:
                if complete:
                    status = "✓"
                    size_str = format_size(size)
                elif size > 0:
                    status = "⏳"
                    size_str = f"{format_size(size)} (in progress)"
                else:
                    status = "⏸️"
                    size_str = "Pending"

                print(f"  {status} {genome:<35} {size_str:>15}")

            print()

            if conversion_procs:
                print("Active conversion processes:")
                for proc in conversion_procs:
                    print(f"  PID {proc['pid']}: {proc['cpu']}% CPU - {proc['cmd']}")
                print()

            # Phase 2: Chromosome Consensus Building
            print("PHASE 2: CHROMOSOME CONSENSUS BUILDING")
            print("-" * 80)

            chr_completed = check_chromosome_status()
            chr_pct = (len(chr_completed) / len(CHROMOSOMES)) * 100

            print(f"Progress: {draw_progress_bar(chr_pct, width=50)}")
            print(f"Completed: {len(chr_completed)}/{len(CHROMOSOMES)} chromosomes")

            if chr_completed:
                print()
                print("Recent completions:")
                for chr_name, size in chr_completed[-5:]:
                    print(f"  ✓ {chr_name:<10} {format_size(size):>12}")

            print()

            # Phase 3: Final Assembly
            print("PHASE 3: FINAL ASSEMBLY")
            print("-" * 80)

            final_consensus = OUTPUT_DIR / "superposition_consensus.fa.gz"
            final_index = OUTPUT_DIR / "superposition_consensus.fa.gz.fai"
            metadata = OUTPUT_DIR / "metadata.json"

            if final_consensus.exists() and final_index.exists():
                final_size = get_file_size(final_consensus)
                print(f"✓ superposition_consensus.fa.gz   {format_size(final_size):>12}")
                print(f"✓ superposition_consensus.fa.gz.fai")
                if metadata.exists():
                    print(f"✓ metadata.json")
                print()
                print("=" * 80)
                print("✓ BUILD COMPLETE!")
                print("=" * 80)
                break
            else:
                print("⏳ Waiting for final assembly...")

            print()

            # Recent log activity
            print("RECENT LOG ACTIVITY:")
            print("-" * 80)
            recent_logs = get_latest_log_lines(5)
            if recent_logs:
                for line in recent_logs:
                    print(f"  {line.rstrip()}")
            else:
                print("  No log file found")

            print()
            print("=" * 80)

            # Check if process is still running
            if not build_proc and complete_count < len(EXPECTED_GENOMES):
                print("⚠️  BUILD PROCESS STOPPED")
                print()
                print("Process stopped during conversion phase.")
                print(f"Completed: {complete_count}/{len(EXPECTED_GENOMES)} genomes")
                break
            elif not build_proc and chr_pct < 100:
                print("⚠️  BUILD PROCESS STOPPED")
                print()
                print("Process stopped during chromosome consensus phase.")
                print(f"Completed: {len(chr_completed)}/{len(CHROMOSOMES)} chromosomes")
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

    parser = argparse.ArgumentParser(description="Monitor sequential superposition build")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor")
    parser.add_argument("--interval", type=int, default=15, help="Refresh interval (seconds)")

    args = parser.parse_args()

    monitor_build(watch=args.watch, interval=args.interval)
