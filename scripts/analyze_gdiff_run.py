#!/usr/bin/env python3
"""
GDiff Pipeline Log Analyzer

Extracts timing, variant counts, and performance patterns from pipeline logs.
For post-run analysis and pattern discovery.
"""

import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics


def parse_log(log_file):
    """Parse GDiff pipeline log and extract metrics"""

    data = {
        'start_time': None,
        'end_time': None,
        'chunks': [],
        'chromosomes': defaultdict(lambda: {'variants': 0, 'chunks': 0, 'times': []}),
        'total_variants': 0,
        'quality_metrics': {},
        'config': {},
        'errors': []
    }

    with open(log_file, 'r') as f:
        for line in f:
            # Extract timestamp
            ts_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),', line)
            if ts_match:
                timestamp = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S')
                if not data['start_time']:
                    data['start_time'] = timestamp
                data['end_time'] = timestamp

            # Configuration
            if 'k-anonymity:' in line:
                data['config']['k_anonymity'] = int(re.search(r'k-anonymity: (\d+)', line).group(1))
            if 'Chunk size:' in line and 'MB' in line:
                data['config']['chunk_size_mb'] = float(re.search(r'Chunk size: ([\d.]+) MB', line).group(1))
            if 'Total chunks:' in line:
                data['config']['total_chunks'] = int(re.search(r'Total chunks: (\d+)', line).group(1))
            if 'workers' in line.lower() and 'parallel' in line:
                match = re.search(r'(\d+) workers', line)
                if match:
                    data['config']['workers'] = int(match.group(1))

            # Quality metrics
            if 'Q_input =' in line:
                data['quality_metrics']['q_input'] = float(re.search(r'Q_input = ([\d.]+)', line).group(1))
            if 'Average Q-score:' in line:
                data['quality_metrics']['avg_qscore'] = float(re.search(r'Average Q-score: ([\d.]+)', line).group(1))
            if 'Q30 fraction:' in line:
                data['quality_metrics']['q30_pct'] = float(re.search(r'Q30 fraction: ([\d.]+)%', line).group(1))

            # Chunk processing (with variant counts)
            chunk_match = re.search(r'✓ (chr\w+):\d+-\d+: ([\d,]+) variants \[(\d+)/(\d+)\]', line)
            if chunk_match:
                chrom = chunk_match.group(1)
                variants = int(chunk_match.group(2).replace(',', ''))
                chunk_num = int(chunk_match.group(3))

                data['chunks'].append({
                    'chromosome': chrom,
                    'variants': variants,
                    'chunk_num': chunk_num,
                    'timestamp': timestamp
                })

                data['chromosomes'][chrom]['variants'] += variants
                data['chromosomes'][chrom]['chunks'] += 1
                data['total_variants'] += variants

            # Chromosome completion
            if 'Processing complete:' in line:
                match = re.search(r'(chr\w+).*: ([\d,]+) variants', line)
                if match:
                    chrom = match.group(1)
                    variants = int(match.group(2).replace(',', ''))
                    data['chromosomes'][chrom]['total_variants'] = variants

            # Errors
            if 'ERROR' in line or 'CRITICAL' in line:
                data['errors'].append(line.strip())

    return data


def calculate_chunk_times(data):
    """Calculate processing time per chunk"""
    if len(data['chunks']) < 2:
        return []

    times = []
    for i in range(1, len(data['chunks'])):
        prev = data['chunks'][i-1]
        curr = data['chunks'][i]

        time_diff = (curr['timestamp'] - prev['timestamp']).total_seconds()
        if 0 < time_diff < 300:  # Ignore outliers (5min+)
            times.append(time_diff)
            data['chromosomes'][curr['chromosome']]['times'].append(time_diff)

    return times


def generate_report(data):
    """Generate human-readable analysis report"""

    report = []
    report.append("=" * 80)
    report.append("GDiff Pipeline Run Analysis")
    report.append("=" * 80)
    report.append("")

    # Runtime
    if data['start_time'] and data['end_time']:
        runtime = data['end_time'] - data['start_time']
        report.append(f"Run Time: {runtime}")
        report.append(f"  Started:  {data['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"  Finished: {data['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

    # Configuration
    report.append("Configuration:")
    for key, val in data['config'].items():
        report.append(f"  {key.replace('_', ' ').title()}: {val}")
    report.append("")

    # Quality Metrics
    if data['quality_metrics']:
        report.append("Input Quality:")
        for key, val in data['quality_metrics'].items():
            report.append(f"  {key.replace('_', ' ').title()}: {val}")
        report.append("")

    # Overall Stats
    report.append("Overall Statistics:")
    report.append(f"  Total Variants: {data['total_variants']:,}")
    report.append(f"  Chunks Processed: {len(data['chunks'])}")
    if data['config'].get('total_chunks'):
        progress = len(data['chunks']) / data['config']['total_chunks'] * 100
        report.append(f"  Progress: {progress:.1f}%")
    report.append("")

    # Chunk timing analysis
    chunk_times = calculate_chunk_times(data)
    if chunk_times:
        report.append("Chunk Processing Performance:")
        report.append(f"  Mean time per chunk: {statistics.mean(chunk_times):.1f}s")
        report.append(f"  Median time: {statistics.median(chunk_times):.1f}s")
        report.append(f"  Min time: {min(chunk_times):.1f}s")
        report.append(f"  Max time: {max(chunk_times):.1f}s")
        report.append(f"  Std dev: {statistics.stdev(chunk_times):.1f}s")
        report.append("")

    # Chromosome breakdown
    report.append("Chromosome Breakdown:")
    report.append(f"{'Chromosome':<20} {'Variants':>12} {'Chunks':>8} {'Avg Time':>10}")
    report.append("-" * 52)

    for chrom in sorted(data['chromosomes'].keys(), key=lambda x: (x.replace('chr', '').replace('X', '23').replace('Y', '24').zfill(2))):
        stats = data['chromosomes'][chrom]
        variants = stats['variants']
        chunks = stats['chunks']
        avg_time = statistics.mean(stats['times']) if stats['times'] else 0

        report.append(f"{chrom:<20} {variants:>12,} {chunks:>8} {avg_time:>9.1f}s")

    report.append("")

    # Interesting patterns
    report.append("Biological/Performance Patterns:")

    # Find high-variant chromosomes
    sorted_chroms = sorted(data['chromosomes'].items(), key=lambda x: x[1]['variants'], reverse=True)
    if sorted_chroms:
        report.append(f"\n  Highest variant density:")
        for chrom, stats in sorted_chroms[:5]:
            if stats['chunks'] > 0:
                density = stats['variants'] / stats['chunks']
                report.append(f"    {chrom}: {density:,.0f} variants/chunk")

    # Find slow chromosomes
    slow_chroms = [(c, statistics.mean(s['times'])) for c, s in data['chromosomes'].items() if s['times']]
    if slow_chroms:
        slow_chroms.sort(key=lambda x: x[1], reverse=True)
        report.append(f"\n  Slowest processing (potential bottlenecks):")
        for chrom, avg_time in slow_chroms[:5]:
            report.append(f"    {chrom}: {avg_time:.1f}s/chunk")

    # Errors
    if data['errors']:
        report.append("\n" + "=" * 80)
        report.append(f"Errors Detected: {len(data['errors'])}")
        report.append("=" * 80)
        for err in data['errors'][:10]:  # First 10 errors
            report.append(f"  {err}")
        if len(data['errors']) > 10:
            report.append(f"  ... and {len(data['errors']) - 10} more")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_gdiff_run.py <log_file>")
        print("Example: python analyze_gdiff_run.py gdiff_sgrs_pipeline.log")
        sys.exit(1)

    log_file = Path(sys.argv[1])
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    print(f"Analyzing {log_file}...")
    data = parse_log(log_file)
    report = generate_report(data)

    # Save report
    output_file = log_file.with_suffix('.analysis.txt')
    with open(output_file, 'w') as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to: {output_file}")


if __name__ == '__main__':
    main()
