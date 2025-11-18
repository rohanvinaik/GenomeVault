#!/usr/bin/env python3
"""
Monitor REFERENCE genome downloads with visual progress tracking
"""

import os
import time
import sys
from datetime import datetime
from pathlib import Path

# Expected REFERENCE genome downloads
EXPECTED_REFERENCES = {
    "hg38.fa.gz": 938 * 1024 * 1024,  # 938 MB
    "hg19.fa.gz": 905 * 1024 * 1024,  # 905 MB
    "chm13v2.0.fa.gz": 936 * 1024 * 1024,  # 936 MB
    "GRCh38_no_alt.fa.gz": 833 * 1024 * 1024,  # 833 MB
    "hs37d5.fa.gz": 851 * 1024 * 1024,  # 851 MB
    "GRCh38_full_analysis_set.fa.gz": 849 * 1024 * 1024,  # 849 MB
    "chm13v2.0_plus_hg38y.fa.gz": 1000 * 1024 * 1024,  # ~1 GB
    "hg18.fa.gz": 930 * 1024 * 1024,  # 930 MB
}

REF_DIR = Path("data/reference_genomes")

def get_file_size(filepath):
    """Get file size in bytes"""
    try:
        return os.path.getsize(filepath)
    except:
        return 0

def format_size(bytes_size):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def draw_progress_bar(percentage, width=40):
    """Draw a progress bar"""
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percentage:6.2f}%"

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')

def monitor_downloads(watch=False, interval=5):
    """Monitor REFERENCE genome downloads"""

    try:
        while True:
            clear_screen()

            print("=" * 80)
            print("REFERENCE GENOME DOWNLOAD MONITOR")
            print("=" * 80)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Location: {REF_DIR}")
            print("=" * 80)
            print()

            # Check each expected reference
            total_expected = 0
            total_downloaded = 0
            completed_count = 0

            for ref_name, expected_size in EXPECTED_REFERENCES.items():
                filepath = REF_DIR / ref_name
                current_size = get_file_size(filepath)

                total_expected += expected_size
                total_downloaded += current_size

                if current_size >= expected_size * 0.95:  # Consider 95%+ as complete
                    status = "✓ COMPLETE"
                    completed_count += 1
                    percentage = 100.0
                elif current_size > 0:
                    status = "⬇ DOWNLOADING"
                    percentage = (current_size / expected_size) * 100
                else:
                    status = "⏳ PENDING"
                    percentage = 0.0

                # Draw progress bar
                progress = draw_progress_bar(percentage, width=30)
                size_str = f"{format_size(current_size):>10} / {format_size(expected_size):<10}"

                print(f"{ref_name:<35} {status:<15}")
                print(f"  {progress}  {size_str}")
                print()

            # Overall progress
            print("=" * 80)
            overall_percentage = (total_downloaded / total_expected) * 100 if total_expected > 0 else 0
            overall_progress = draw_progress_bar(overall_percentage, width=50)

            print(f"\nOVERALL PROGRESS: {completed_count}/{len(EXPECTED_REFERENCES)} files complete")
            print(f"{overall_progress}")
            print(f"Total: {format_size(total_downloaded)} / {format_size(total_expected)}")
            print()

            # Check if directory exists
            if REF_DIR.exists():
                actual_size = sum(f.stat().st_size for f in REF_DIR.glob("*.fa.gz") if f.is_file())
                print(f"Actual directory size: {format_size(actual_size)}")
            else:
                print(f"Directory {REF_DIR} does not exist yet")

            print()
            print("=" * 80)

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

    parser = argparse.ArgumentParser(description="Monitor REFERENCE genome downloads")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval (seconds)")

    args = parser.parse_args()

    monitor_downloads(watch=args.watch, interval=args.interval)
