"""
Analyze batch timing patterns to identify fast vs slow chunks.

Purpose: Identify genomic features that correlate with encoding speed.

Author: Claude Code
Date: November 21, 2025
"""

import re
import json
from pathlib import Path
from typing import List, Dict
import numpy as np


def parse_encoder_log(log_path: str) -> List[Dict]:
    """
    Parse encoder log to extract batch timing information.

    Args:
        log_path: Path to encoder log file

    Returns:
        List of batch info dictionaries
    """
    batches = []

    with open(log_path, 'r') as f:
        for line in f:
            # Match batch completion lines
            # Example: "✓ Batch complete: 5,000 chunks in 15.6s (320.2 chunks/s)"
            match = re.search(r'Batch complete: ([\d,]+) chunks in ([\d.]+)s \(([\d.]+) chunks/s\)', line)
            if match:
                num_chunks = int(match.group(1).replace(',', ''))
                time_s = float(match.group(2))
                chunks_per_s = float(match.group(3))

                batches.append({
                    'num_chunks': num_chunks,
                    'time_s': time_s,
                    'chunks_per_s': chunks_per_s,
                })

            # Match batch start lines to get chunk ranges
            # Example: "Batch 2/675: chunks 5,000 - 9,999"
            match = re.search(r'Batch (\d+)/(\d+): chunks ([\d,]+) - ([\d,]+)', line)
            if match:
                batch_num = int(match.group(1))
                start_chunk = int(match.group(3).replace(',', ''))
                end_chunk = int(match.group(4).replace(',', ''))

                # Add to the previous batch info (batch completes before next one starts)
                if batches and 'batch_num' not in batches[-1]:
                    batches[-1]['batch_num'] = batch_num - 1
                    batches[-1]['start_chunk'] = start_chunk - num_chunks
                    batches[-1]['end_chunk'] = start_chunk - 1

    return batches


def analyze_timing_distribution(batches: List[Dict]) -> Dict:
    """
    Analyze the distribution of batch timings.

    Args:
        batches: List of batch info dictionaries

    Returns:
        Statistics dictionary
    """
    if not batches:
        return {}

    times = [b['time_s'] for b in batches]
    speeds = [b['chunks_per_s'] for b in batches]

    stats = {
        'num_batches': len(batches),
        'time': {
            'min': np.min(times),
            'median': np.median(times),
            'mean': np.mean(times),
            'max': np.max(times),
            'std': np.std(times),
        },
        'speed': {
            'min': np.min(speeds),
            'median': np.median(speeds),
            'mean': np.mean(speeds),
            'max': np.max(speeds),
            'std': np.std(speeds),
        },
    }

    # Identify fast vs slow batches (using median as threshold)
    median_time = stats['time']['median']
    fast_batches = [b for b in batches if b['time_s'] < median_time]
    slow_batches = [b for b in batches if b['time_s'] >= median_time]

    stats['fast_batches'] = {
        'count': len(fast_batches),
        'avg_time': np.mean([b['time_s'] for b in fast_batches]) if fast_batches else 0,
        'avg_speed': np.mean([b['chunks_per_s'] for b in fast_batches]) if fast_batches else 0,
    }

    stats['slow_batches'] = {
        'count': len(slow_batches),
        'avg_time': np.mean([b['time_s'] for b in slow_batches]) if slow_batches else 0,
        'avg_speed': np.mean([b['chunks_per_s'] for b in slow_batches]) if slow_batches else 0,
    }

    return stats


def identify_outliers(batches: List[Dict], threshold_sigma: float = 2.0) -> Dict:
    """
    Identify outlier batches (very fast or very slow).

    Args:
        batches: List of batch info dictionaries
        threshold_sigma: Number of standard deviations for outlier detection

    Returns:
        Dictionary with fast and slow outliers
    """
    if not batches:
        return {'fast': [], 'slow': []}

    times = np.array([b['time_s'] for b in batches])
    mean_time = np.mean(times)
    std_time = np.std(times)

    fast_outliers = []
    slow_outliers = []

    for batch in batches:
        z_score = (batch['time_s'] - mean_time) / std_time

        if z_score < -threshold_sigma:  # Much faster than average
            fast_outliers.append({
                **batch,
                'z_score': z_score,
                'speedup': mean_time / batch['time_s'],
            })
        elif z_score > threshold_sigma:  # Much slower than average
            slow_outliers.append({
                **batch,
                'z_score': z_score,
                'slowdown': batch['time_s'] / mean_time,
            })

    return {
        'fast': sorted(fast_outliers, key=lambda x: x['z_score']),
        'slow': sorted(slow_outliers, key=lambda x: x['z_score'], reverse=True),
    }


def print_analysis(batches: List[Dict], stats: Dict, outliers: Dict):
    """Print formatted analysis results."""
    print("=" * 80)
    print("BATCH TIMING ANALYSIS")
    print("=" * 80)
    print()

    print(f"Total batches analyzed: {stats['num_batches']}")
    print()

    print("Batch timing distribution:")
    print(f"  Min:    {stats['time']['min']:>6.2f}s  ({stats['speed']['max']:>6.1f} chunks/s)")
    print(f"  Median: {stats['time']['median']:>6.2f}s  ({stats['speed']['median']:>6.1f} chunks/s)")
    print(f"  Mean:   {stats['time']['mean']:>6.2f}s  ({stats['speed']['mean']:>6.1f} chunks/s)")
    print(f"  Max:    {stats['time']['max']:>6.2f}s  ({stats['speed']['min']:>6.1f} chunks/s)")
    print(f"  StdDev: {stats['time']['std']:>6.2f}s")
    print()

    print(f"Fast batches (< median): {stats['fast_batches']['count']}")
    print(f"  Avg time: {stats['fast_batches']['avg_time']:.2f}s")
    print(f"  Avg speed: {stats['fast_batches']['avg_speed']:.1f} chunks/s")
    print()

    print(f"Slow batches (≥ median): {stats['slow_batches']['count']}")
    print(f"  Avg time: {stats['slow_batches']['avg_time']:.2f}s")
    print(f"  Avg speed: {stats['slow_batches']['avg_speed']:.1f} chunks/s")
    print()

    # Print outliers
    print("=" * 80)
    print("OUTLIERS (> 2σ from mean)")
    print("=" * 80)
    print()

    if outliers['fast']:
        print(f"Fast outliers ({len(outliers['fast'])}):")
        for i, batch in enumerate(outliers['fast'][:5], 1):  # Top 5
            print(f"  {i}. Batch {batch.get('batch_num', '?')}: "
                  f"{batch['time_s']:.2f}s ({batch['chunks_per_s']:.1f} chunks/s) "
                  f"- {batch['speedup']:.2f}× faster than average")
        print()

    if outliers['slow']:
        print(f"Slow outliers ({len(outliers['slow'])}):")
        for i, batch in enumerate(outliers['slow'][:5], 1):  # Top 5
            print(f"  {i}. Batch {batch.get('batch_num', '?')}: "
                  f"{batch['time_s']:.2f}s ({batch['chunks_per_s']:.1f} chunks/s) "
                  f"- {batch['slowdown']:.2f}× slower than average")
        print()

    print("=" * 80)
    print()


if __name__ == '__main__':
    # Parse encoder log
    log_path = '/tmp/encoder_CORRECTED.log'

    print("Parsing encoder log...")
    batches = parse_encoder_log(log_path)

    if not batches:
        print("No batches found in log file. Encoder may still be initializing.")
        exit(0)

    print(f"Found {len(batches)} completed batches")
    print()

    # Analyze timing distribution
    stats = analyze_timing_distribution(batches)

    # Identify outliers
    outliers = identify_outliers(batches, threshold_sigma=2.0)

    # Print analysis
    print_analysis(batches, stats, outliers)

    # Save results for later correlation with genomic features
    output_path = Path('genomevault/hdv_validation/hdc_experimentation/output/batch_timing_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'batches': batches,
            'stats': stats,
            'outliers': outliers,
        }, f, indent=2)

    print(f"Results saved to: {output_path}")
    print()
    print("Next steps:")
    print("  1. Correlate fast/slow batches with chromosome boundaries")
    print("  2. Check variant density in fast vs slow chunks")
    print("  3. Analyze GC content patterns")
    print("  4. Identify repetitive regions (Alu, LINE, SINE)")
