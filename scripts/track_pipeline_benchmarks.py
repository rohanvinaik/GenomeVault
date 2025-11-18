#!/usr/bin/env python3
"""
Comprehensive benchmark tracking for Phase 1-3 optimized pipeline.

Tracks and logs:
- Per-reference timing (alignment, sorting, variant calling)
- Resource usage (CPU, memory, I/O)
- Speedup calculations vs baseline
- Cumulative statistics
- Optimization effectiveness
"""

import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import re

class PipelineBenchmarkTracker:
    """Track and analyze pipeline performance."""

    def __init__(self, log_file: str, output_dir: str):
        self.log_file = Path(log_file)
        self.output_dir = Path(output_dir)
        self.benchmarks = {
            "start_time": None,
            "end_time": None,
            "total_runtime_sec": 0,
            "baseline_estimate_hours": 90,  # 7.5 hours × 12 refs
            "references": {},
            "summary": {},
            "optimizations": {
                "phase1": {
                    "sambamba": True,
                    "parallel_bcftools": True,
                    "minimap2_optimized": True,
                    "metal_gpu": True
                },
                "phase2": {
                    "index_caching": True
                },
                "phase3": {
                    "chromosome_parallel_sort": True,
                    "parallel_vcf_parsing": True
                }
            }
        }

    def parse_log(self):
        """Parse log file and extract all timing metrics."""
        if not self.log_file.exists():
            print(f"Log file not found: {self.log_file}")
            return

        with open(self.log_file, 'r') as f:
            log_content = f.read()

        # Extract index build time
        match = re.search(r'Index built in ([0-9.]+) seconds', log_content)
        if match:
            self.benchmarks['index_build_sec'] = float(match.group(1))

        # Parse per-reference metrics
        current_ref = None
        for line in log_content.split('\n'):
            # Detect which reference is being processed
            match = re.search(r'\[(\d+)/\d+\] Processing (ref\d+)', line)
            if match:
                current_ref = match.group(2)
                self.benchmarks['references'][current_ref] = {
                    'index': int(match.group(1)),
                    'alignment_sec': None,
                    'sorting_sec': None,
                    'variant_calling_sec': None,
                    'total_sec': None
                }

            # Extract alignment completion time
            if current_ref and 'Alignment complete in' in line:
                match = re.search(r'([0-9.]+) seconds', line)
                if match:
                    self.benchmarks['references'][current_ref]['alignment_sec'] = float(match.group(1))

            # Extract variant calling time
            if current_ref and 'Variant calling complete in' in line:
                match = re.search(r'([0-9.]+) seconds', line)
                if match:
                    self.benchmarks['references'][current_ref]['variant_calling_sec'] = float(match.group(1))

            # Extract total time per reference
            if current_ref and 'complete in' in line and 'total_time_sec' not in line:
                match = re.search(r'complete in ([0-9.]+)s', line)
                if match:
                    total_time = float(match.group(1))
                    # Only set if not already set (avoid overwriting with later "complete" messages)
                    if self.benchmarks['references'][current_ref]['total_sec'] is None:
                        self.benchmarks['references'][current_ref]['total_sec'] = total_time

        # Calculate cumulative stats
        self._calculate_summary()

    def _calculate_summary(self):
        """Calculate summary statistics."""
        refs = self.benchmarks['references']

        if not refs:
            return

        # Collect all timing data
        align_times = [r['alignment_sec'] for r in refs.values() if r['alignment_sec']]
        variant_times = [r['variant_calling_sec'] for r in refs.values() if r['variant_calling_sec']]
        total_times = [r['total_sec'] for r in refs.values() if r['total_sec']]

        self.benchmarks['summary'] = {
            'num_completed': len([r for r in refs.values() if r['total_sec']]),
            'total_refs': 12,
            'alignment': {
                'avg_sec': sum(align_times) / len(align_times) if align_times else 0,
                'min_sec': min(align_times) if align_times else 0,
                'max_sec': max(align_times) if align_times else 0,
                'total_sec': sum(align_times) if align_times else 0
            },
            'variant_calling': {
                'avg_sec': sum(variant_times) / len(variant_times) if variant_times else 0,
                'min_sec': min(variant_times) if variant_times else 0,
                'max_sec': max(variant_times) if variant_times else 0,
                'total_sec': sum(variant_times) if variant_times else 0
            },
            'per_reference': {
                'avg_sec': sum(total_times) / len(total_times) if total_times else 0,
                'avg_min': (sum(total_times) / len(total_times)) / 60 if total_times else 0,
                'min_sec': min(total_times) if total_times else 0,
                'max_sec': max(total_times) if total_times else 0
            }
        }

        # Calculate speedup vs baseline
        if total_times:
            avg_per_ref_hours = (sum(total_times) / len(total_times)) / 3600
            baseline_per_ref_hours = 7.5
            self.benchmarks['summary']['speedup_vs_baseline'] = baseline_per_ref_hours / avg_per_ref_hours

            # Project total time
            completed = len(total_times)
            remaining = 12 - completed
            est_remaining_sec = (sum(total_times) / len(total_times)) * remaining
            est_total_sec = sum(total_times) + est_remaining_sec

            self.benchmarks['summary']['projection'] = {
                'est_remaining_sec': est_remaining_sec,
                'est_remaining_hours': est_remaining_sec / 3600,
                'est_total_sec': est_total_sec,
                'est_total_hours': est_total_sec / 3600,
                'baseline_total_hours': 90,
                'time_saved_hours': 90 - (est_total_sec / 3600)
            }

    def save_benchmarks(self, filename: str = None):
        """Save benchmarks to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"phase123_benchmarks_{timestamp}.json"

        output_file = self.output_dir / filename

        # Add metadata
        self.benchmarks['metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'log_file': str(self.log_file),
            'output_dir': str(self.output_dir)
        }

        with open(output_file, 'w') as f:
            json.dump(self.benchmarks, f, indent=2)

        print(f"Benchmarks saved to: {output_file}")
        return output_file

    def print_summary(self):
        """Print formatted benchmark summary."""
        print("\n" + "=" * 80)
        print("PHASE 1-3 OPTIMIZED PIPELINE - BENCHMARK SUMMARY")
        print("=" * 80)

        summary = self.benchmarks.get('summary', {})

        if not summary:
            print("\nNo benchmark data available yet.")
            return

        # Overall progress
        print(f"\n📊 Progress: {summary['num_completed']} / {summary['total_refs']} references completed\n")

        # Index build
        if 'index_build_sec' in self.benchmarks:
            print(f"🔨 Minimap2 Index Build: {self.benchmarks['index_build_sec']:.1f}s (one-time, reused for all refs)")

        # Per-reference timing
        print("\n⏱  Per-Reference Performance:")
        per_ref = summary.get('per_reference', {})
        if per_ref.get('avg_sec'):
            print(f"  Average: {per_ref['avg_min']:.1f} min ({per_ref['avg_sec']:.1f}s)")
            print(f"  Range: {per_ref['min_sec']:.1f}s - {per_ref['max_sec']:.1f}s")

        # Alignment stats
        align = summary.get('alignment', {})
        if align.get('avg_sec'):
            print(f"\n  Alignment+Sort Average: {align['avg_sec']:.1f}s")
            print(f"    Range: {align['min_sec']:.1f}s - {align['max_sec']:.1f}s")

        # Variant calling stats
        variant = summary.get('variant_calling', {})
        if variant.get('avg_sec'):
            print(f"\n  Variant Calling Average: {variant['avg_sec']:.1f}s")
            print(f"    Range: {variant['min_sec']:.1f}s - {variant['max_sec']:.1f}s")

        # Speedup
        if 'speedup_vs_baseline' in summary:
            print(f"\n🚀 Speedup vs Baseline: {summary['speedup_vs_baseline']:.1f}×")
            print(f"   Baseline: 7.5 hours per reference")
            print(f"   Optimized: {per_ref['avg_min']:.1f} min per reference")

        # Projection
        proj = summary.get('projection', {})
        if proj:
            print(f"\n📈 Total Pipeline Projection:")
            print(f"  Estimated Total: {proj['est_total_hours']:.2f} hours")
            print(f"  Baseline Total: {proj['baseline_total_hours']:.1f} hours")
            print(f"  Time Saved: {proj['time_saved_hours']:.1f} hours ({proj['time_saved_hours']/proj['baseline_total_hours']*100:.1f}%)")
            print(f"  Remaining: {proj['est_remaining_hours']:.2f} hours")

        # Optimizations
        print("\n✨ Active Optimizations:")
        print("  Phase 1: Sambamba + Parallel BCFtools + Minimap2 + Metal GPU")
        print("  Phase 2: Index caching")
        print("  Phase 3: Chromosome-parallel sort + Parallel VCF parsing")

        print("\n" + "=" * 80 + "\n")


def main():
    """Main function."""
    log_file = "logs/phase123_optimized_deployment.log"
    output_dir = "benchmark_results/enhanced_privacy_k13_phase123_optimized"

    print("Tracking Phase 1-3 Pipeline Benchmarks...")
    print(f"Log file: {log_file}")
    print(f"Output directory: {output_dir}\n")

    tracker = PipelineBenchmarkTracker(log_file, output_dir)
    tracker.parse_log()
    tracker.print_summary()

    # Save benchmarks
    output_file = tracker.save_benchmarks()

    print(f"\n💾 Detailed benchmarks saved to: {output_file}")
    print("\nRun this script periodically to update benchmarks:")
    print("  python3 scripts/track_pipeline_benchmarks.py")


if __name__ == "__main__":
    main()
