#!/usr/bin/env python3
"""
Enhanced graphical terminal tracker for chromosome-parallel pipeline.

Features:
- Smart parent process detection (backtracks from workers)
- Chromosome-level progress tracking
- Real-time progress bars
- Color-coded status
- Detailed stage detection
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import re

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

def find_parent_from_workers():
    """
    Smart detection: find pipeline parent process by backtracking from worker PIDs.
    Falls back to direct pipeline script detection if no workers found.

    Returns:
        (pid, script_name) or (None, None)
    """
    try:
        # Get all processes
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )

        # Check for GDiff encoding pipeline first
        for line in result.stdout.split('\n'):
            if 'run_k12_gdiff_pipeline.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    return pid, 'run_k12_gdiff_pipeline.py'

        # Find worker processes
        worker_pids = []
        for line in result.stdout.split('\n'):
            if any(proc in line for proc in ['minimap2', 'sambamba', 'bcftools', 'samtools']):
                if 'grep' not in line:
                    parts = line.split()
                    if len(parts) > 1:
                        worker_pids.append(parts[1])

        # If no workers found, directly search for pipeline scripts
        # (happens during Python-only phases like pysam partitioning)
        if not worker_pids:
            pipeline_scripts = [
                'continue_ref2_to_ref12',
                'deploy_phase123',
                'deploy_optimized',
                'recover_ref2_from_sorting'
            ]

            for line in result.stdout.split('\n'):
                if 'python' in line and any(script in line for script in pipeline_scripts):
                    if 'grep' not in line:
                        parts = line.split()
                        if len(parts) > 1:
                            pid = parts[1]
                            # Extract script name from command
                            for part in parts:
                                if any(script in part for script in pipeline_scripts):
                                    script_name = part.split('/')[-1]
                                    return pid, script_name

            return None, None

        # Get parent PIDs using ps -o ppid
        for worker_pid in worker_pids:
            try:
                result = subprocess.run(
                    ["ps", "-o", "ppid=", "-p", worker_pid],
                    capture_output=True,
                    text=True
                )
                ppid = result.stdout.strip()

                if ppid:
                    # Check if this PPID is a Python pipeline process
                    result = subprocess.run(
                        ["ps", "-p", ppid, "-o", "command="],
                        capture_output=True,
                        text=True
                    )
                    command = result.stdout.strip()

                    if 'python' in command and any(script in command for script in [
                        'continue_ref2_to_ref12',
                        'deploy_phase123',
                        'deploy_optimized',
                        'recover_ref2_from_sorting',
                        'recover_ref2_from_partitioned_bams'
                    ]):
                        # Extract script name
                        script_name = command.split('/')[-1].split()[0]
                        return ppid, script_name

                    # Try parent's parent (for bash wrappers)
                    result = subprocess.run(
                        ["ps", "-o", "ppid=", "-p", ppid],
                        capture_output=True,
                        text=True
                    )
                    gppid = result.stdout.strip()

                    if gppid:
                        result = subprocess.run(
                            ["ps", "-p", gppid, "-o", "command="],
                            capture_output=True,
                            text=True
                        )
                        command = result.stdout.strip()

                        if 'python' in command and any(script in command for script in [
                            'continue_ref2_to_ref12',
                            'deploy_phase123',
                            'deploy_optimized',
                            'recover_ref2_from_sorting',
                            'recover_ref2_from_partitioned_bams'
                        ]):
                            script_name = command.split('/')[-1].split()[0]
                            return gppid, script_name

            except:
                continue

        return None, None

    except Exception as e:
        return None, None

def detect_current_stage():
    """
    Detect current pipeline stage and reference being processed.

    Returns:
        dict with stage info
    """
    stage_info = {
        'stage': 'unknown',
        'reference': None,
        'chromosome_progress': None,
        'details': '',
        'gdiff_info': None  # NEW: GDiff encoding specific info
    }

    try:
        # Check for GDiff encoding log first (both old and new formats)
        gdiff_logs = list(Path('.').glob('k11_gdiff_encoding_*.log')) + \
                     list(Path('.').glob('k11_LIGHTWEIGHT_*.log')) + \
                     list(Path('.').glob('k11_STREAMING_*.log'))
        if gdiff_logs:
            # Get the most recent log
            gdiff_log = sorted(gdiff_logs, key=lambda p: p.stat().st_mtime, reverse=True)[0]

            try:
                with open(gdiff_log, 'r') as f:
                    log_lines = f.readlines()
                    recent = ''.join(log_lines[-50:])  # Get last 50 lines

                    # Parse GDiff encoding progress
                    gdiff_info = {
                        'template_loaded': False,
                        'template_variants': 0,
                        'current_chromosome': None,
                        'current_region': None,
                        'current_guide': None,
                        'total_variants': 0,
                        'chromosomes_completed': []
                    }

                    # Check for template loading
                    match = re.search(r'Template loaded: ([\d,]+) variant sites', recent)
                    if match:
                        gdiff_info['template_loaded'] = True
                        gdiff_info['template_variants'] = int(match.group(1).replace(',', ''))

                    # Check for current chromosome
                    match = re.search(r'Processing (chr\d+_consensus|chrX_consensus|chrY_consensus)', recent)
                    if match:
                        gdiff_info['current_chromosome'] = match.group(1)
                        stage_info['stage'] = 'gdiff_encoding'

                        # Check for current region and guide
                        region_match = re.search(r'Region (\d+)-(\d+)MB: using guide (\d+)', recent)
                        if region_match:
                            start_mb = int(region_match.group(1))
                            end_mb = int(region_match.group(2))
                            guide = int(region_match.group(3))
                            gdiff_info['current_region'] = f"{start_mb}-{end_mb}MB"
                            gdiff_info['current_guide'] = guide
                            stage_info['details'] = f"Encoding {gdiff_info['current_chromosome']} region {start_mb}-{end_mb}MB (guide {guide})"
                        else:
                            stage_info['details'] = f"Encoding {gdiff_info['current_chromosome']}"

                    # Check for variants found
                    variant_matches = re.findall(r'Found ([\d,]+) variants', recent)
                    if variant_matches:
                        # Sum up all variants found
                        total = sum(int(v.replace(',', '')) for v in variant_matches)
                        gdiff_info['total_variants'] = total

                    stage_info['gdiff_info'] = gdiff_info

                    # If we detected GDiff encoding, return early
                    if stage_info['stage'] == 'gdiff_encoding':
                        return stage_info
            except Exception as e:
                pass

        # Check what processes are running
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )

        # Analyze active processes
        has_minimap2 = False
        has_sambamba = False
        has_samtools_sort = False
        has_samtools_index = False
        has_bcftools_mpileup = False
        has_bcftools_call = False
        sambamba_procs = []
        samtools_sort_procs = []

        for line in result.stdout.split('\n'):
            if 'minimap2' in line and 'grep' not in line:
                has_minimap2 = True
                # Extract reference from command
                if 'ref' in line:
                    match = re.search(r'ref(\d+)', line)
                    if match:
                        stage_info['reference'] = f"ref{match.group(1)}"

            if 'sambamba' in line and 'grep' not in line:
                has_sambamba = True
                sambamba_procs.append(line)
                # Try to extract chromosome
                match = re.search(r'(chr\d+|chrX|chrY|chrM)', line)
                if match:
                    chr_name = match.group(1)
                    if stage_info['chromosome_progress'] is None:
                        stage_info['chromosome_progress'] = []
                    stage_info['chromosome_progress'].append(chr_name)

            if 'samtools sort' in line and 'grep' not in line:
                has_samtools_sort = True
                samtools_sort_procs.append(line)
                # Try to extract chromosome
                match = re.search(r'(chr\d+|chrX|chrY|chrM)', line)
                if match:
                    chr_name = match.group(1)
                    if stage_info['chromosome_progress'] is None:
                        stage_info['chromosome_progress'] = []
                    stage_info['chromosome_progress'].append(chr_name)

            if 'samtools index' in line and 'grep' not in line:
                has_samtools_index = True
                # Extract reference from filename
                match = re.search(r'(ref\d+)\.sorted\.bam', line)
                if match:
                    stage_info['reference'] = match.group(1)

            if 'bcftools mpileup' in line and 'grep' not in line:
                has_bcftools_mpileup = True

            if 'bcftools call' in line and 'grep' not in line:
                has_bcftools_call = True

        # Determine stage
        if has_minimap2:
            stage_info['stage'] = 'alignment'
            stage_info['details'] = 'Aligning FASTQ to consensus reference'

        elif has_sambamba and sambamba_procs:
            num_chr = len(stage_info['chromosome_progress']) if stage_info['chromosome_progress'] else 0
            if num_chr > 0:
                stage_info['stage'] = 'chromosome_sorting'
                stage_info['details'] = f'Sorting {num_chr} chromosomes in parallel'
            else:
                stage_info['stage'] = 'sorting'
                stage_info['details'] = 'Sorting BAM file'

        elif has_samtools_sort and samtools_sort_procs:
            num_chr = len(stage_info['chromosome_progress']) if stage_info['chromosome_progress'] else 0
            if num_chr > 0:
                stage_info['stage'] = 'chromosome_sorting'
                stage_info['details'] = f'Sorting {num_chr} chromosomes in parallel (samtools)'
            else:
                stage_info['stage'] = 'sorting'
                stage_info['details'] = 'Sorting BAM file (samtools)'

        elif has_samtools_index:
            stage_info['stage'] = 'bam_indexing'
            stage_info['details'] = 'Indexing sorted BAM file'

        elif has_bcftools_mpileup or has_bcftools_call:
            stage_info['stage'] = 'variant_calling'
            stage_info['details'] = 'Calling variants with BCFtools'

        # Check logs for more details (try multiple log files)
        log_files = [
            Path("logs/ref2_to_ref12_chromosome_parallel.log"),
            Path("logs/ref2_recovery.log"),
            Path("logs/ref2_recovery_sorting.log")
        ]

        for log_file in log_files:
            if not log_file.exists():
                continue
            with open(log_file, 'r') as f:
                log_lines = f.readlines()
                # Get last 30 lines for progress
                recent = ''.join(log_lines[-30:])
                # Get first 50 lines for reference detection
                beginning = ''.join(log_lines[:50])

                # Extract current reference (check both beginning and recent)
                # Look for patterns like "Processing ref2" or "ref2.unsorted.bam"
                match = re.search(r'Processing (ref\d+)', beginning + recent)
                if not match:
                    match = re.search(r'(ref\d+)\.', beginning + recent)
                if match:
                    stage_info['reference'] = match.group(1)

                # Detect streaming partition with progress
                # Check for either the initial message or ongoing progress updates
                if 'Streaming through BAM to partition' in recent or 'Processed' in recent and 'reads...' in recent:
                    # Extract read count from progress updates
                    matches = re.findall(r'Processed ([0-9,]+) reads', recent)
                    if matches:
                        # Get the highest count
                        counts = [int(m.replace(',', '')) for m in matches]
                        max_count = max(counts)
                        stage_info['stage'] = 'partitioning'
                        stage_info['details'] = f'Streaming partition: {max_count/1e6:.1f}M reads processed'
                        # Infer reference from log file name or earlier in log
                        if 'ref2' in str(log_file) or 'ref2' in recent:
                            stage_info['reference'] = 'ref2'
                    elif 'Streaming through BAM to partition' in recent:
                        stage_info['stage'] = 'partitioning'
                        stage_info['details'] = 'Partitioning BAM by chromosome (streaming)'

                # Detect partitioning complete
                if 'Partitioning complete' in recent:
                    match = re.search(r'Partitioning complete: ([0-9,]+) total reads', recent)
                    if match:
                        total = int(match.group(1).replace(',', ''))
                        stage_info['details'] = f'Partitioning complete: {total/1e6:.1f}M reads'

                # Detect merging/concatenation
                if 'Concatenat' in recent or 'Merging' in recent:
                    stage_info['stage'] = 'merging'
                    stage_info['details'] = 'Merging sorted chromosomes'

                # Detect indexing
                if 'Indexing' in recent:
                    stage_info['stage'] = 'indexing'
                    stage_info['details'] = 'Indexing BAM/VCF files'

            break  # Stop after finding first valid log file

    except Exception as e:
        pass

    return stage_info

def get_pipeline_status():
    """Get comprehensive pipeline status."""
    status = {
        'running': False,
        'pid': None,
        'script_name': None,
        'cpu': 0,
        'memory': 0,
        'runtime': '0:00:00',
        'completed_refs': 0,
        'total_refs': 12,
        'current_ref': None,
        'stage_info': {},
        'active_processes': [],
        'chromosome_details': [],
        'progress_details': {},  # Detailed progress info
        'system_memory': {},  # NEW: System-wide memory stats for memory-safe pipelines
        'memory_warnings': []  # NEW: Memory safety warnings
    }

    # Smart parent detection
    pid, script_name = find_parent_from_workers()
    if pid:
        status['running'] = True
        status['pid'] = pid
        status['script_name'] = script_name

        # Get runtime
        try:
            result = subprocess.run(
                ["ps", "-p", pid, "-o", "etime="],
                capture_output=True,
                text=True
            )
            status['runtime'] = result.stdout.strip()
        except:
            pass

    # Detect current stage
    status['stage_info'] = detect_current_stage()
    if status['stage_info']['reference']:
        status['current_ref'] = status['stage_info']['reference']

    # Get resource usage from ALL worker processes
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )

        total_cpu = 0
        total_mem = 0

        for line in result.stdout.split('\n'):
            # Check for both external tools AND multiprocessing workers
            is_worker = (any(proc in line for proc in ['minimap2', 'sambamba', 'bcftools', 'samtools', 'pigz']) or
                        'multiprocessing.spawn' in line)

            if is_worker and 'grep' not in line and 'graphical_pipeline_tracker' not in line:
                parts = line.split()
                try:
                    cpu = float(parts[2])
                    mem = float(parts[3])
                    total_cpu += cpu
                    total_mem += mem

                    # Track active processes
                    if 'multiprocessing.spawn' in line:
                        # GDiff multiprocessing worker
                        proc_name = 'gdiff_worker'
                    else:
                        proc_name = parts[10].split('/')[-1]

                    if proc_name in ['minimap2', 'sambamba', 'bcftools', 'samtools', 'gdiff_worker']:
                        status['active_processes'].append({
                            'name': proc_name,
                            'pid': parts[1],
                            'cpu': cpu,
                            'mem': mem
                        })

                        # Track chromosome details for sambamba
                        if 'sambamba' in line:
                            match = re.search(r'(chr\d+|chrX|chrY|chrM)', line)
                            if match:
                                status['chromosome_details'].append({
                                    'chromosome': match.group(1),
                                    'cpu': cpu,
                                    'mem': mem
                                })

                except:
                    pass

        # Also track the parent pipeline process itself (for Python-only phases like partitioning)
        if status['pid']:
            try:
                result = subprocess.run(
                    ["ps", "-p", status['pid'], "-o", "%cpu=,%mem=,comm="],
                    capture_output=True,
                    text=True
                )
                if result.stdout.strip():
                    parts = result.stdout.strip().split()
                    if len(parts) >= 2:
                        parent_cpu = float(parts[0])
                        parent_mem = float(parts[1])
                        total_cpu += parent_cpu
                        total_mem += parent_mem

                        # Add to active processes if it's consuming significant resources
                        if parent_cpu > 1.0:
                            status['active_processes'].append({
                                'name': 'python3 (pipeline)',
                                'pid': status['pid'],
                                'cpu': parent_cpu,
                                'mem': parent_mem
                            })
            except:
                pass

        status['cpu'] = total_cpu
        status['memory'] = total_mem

    except Exception as e:
        pass

    # Count completed references
    layer2_dir = Path("benchmark_results/enhanced_privacy_k13_phase123_optimized/layer2_reference_pool")
    if layer2_dir.exists():
        fasta_files = list(layer2_dir.glob("*.fa.gz"))
        # Count actual files, not symlinks (or count symlinks as complete)
        status['completed_refs'] = len(fasta_files)

    # Read detailed progress from progress files and compute time estimates
    output_dir = Path("benchmark_results/enhanced_privacy_k13_phase123_optimized")
    if output_dir.exists():
        import json
        import time as time_module

        for progress_file in output_dir.glob(".progress_ref*.json"):
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    ref_name = progress_data['sample']

                    # Calculate elapsed time for current stage
                    elapsed_time = time_module.time() - progress_data.get('timestamp', time_module.time())

                    status['progress_details'][ref_name] = {
                        'stage': progress_data.get('stage', 'unknown'),
                        'details': progress_data.get('details', ''),
                        'timestamp': progress_data.get('timestamp', 0),
                        'elapsed_seconds': elapsed_time
                    }
            except:
                pass

        # Estimate completion time based on stage durations
        status['time_estimates'] = estimate_completion_times(status, output_dir)

    return status

def estimate_completion_times(status, output_dir):
    """Estimate time remaining based on historical data and current progress."""
    import time
    estimates = {
        'alignment_avg_sec': 0,
        'sorting_avg_sec': 0,
        'fasta_extraction_avg_sec': 0,
        'total_per_ref_avg_sec': 0,
        'estimated_remaining_sec': 0,
        'estimated_completion_time': None
    }

    # Check for benchmark/metrics files with historical timing data
    metrics_file = output_dir / "layer2_metrics.json"
    if metrics_file.exists():
        try:
            import json
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                if 'samples' in metrics and metrics['samples']:
                    # Calculate averages from completed samples
                    completed = [s for s in metrics['samples'] if s.get('total_time_sec', 0) > 0]
                    if completed:
                        estimates['alignment_avg_sec'] = sum(s.get('alignment_time_sec', 0) for s in completed) / len(completed)
                        estimates['fasta_extraction_avg_sec'] = sum(s.get('fasta_extraction_time_sec', 0) for s in completed) / len(completed)
                        estimates['total_per_ref_avg_sec'] = sum(s.get('total_time_sec', 0) for s in completed) / len(completed)
        except:
            pass

    # Fallback: estimate based on ref4 actual times (with improved cooling)
    if estimates['total_per_ref_avg_sec'] == 0:
        # Realistic estimates from ref4 (whole-genome with optimized cooling)
        estimates['alignment_avg_sec'] = 13000  # ~3.6 hours alignment (ref4: 3h 36min)
        estimates['sorting_avg_sec'] = 840     # ~14 min sorting (samtools fallback)
        estimates['fasta_extraction_avg_sec'] = 1860  # ~31 min extraction (ref4: 31min)
        estimates['total_per_ref_avg_sec'] = 15700  # ~4.4 hours total per ref

    # Calculate remaining time
    refs_remaining = status['total_refs'] - status['completed_refs']

    # If currently processing a ref, estimate its remaining time
    current_ref_remaining = 0
    if status['progress_details']:
        for ref_name, details in status['progress_details'].items():
            stage = details['stage']
            elapsed = details['elapsed_seconds']

            if stage == 'alignment':
                # Try to estimate based on file size if available
                try:
                    import subprocess
                    result = subprocess.run(
                        ["lsof", f"benchmark_results/enhanced_privacy_k13_phase123_optimized/tmp/{ref_name}.aligned.sam"],
                        capture_output=True,
                        text=True
                    )
                    if result.stdout:
                        for line in result.stdout.split('\n'):
                            if 'minimap2' in line:
                                parts = line.split()
                                if len(parts) > 6:
                                    current_bytes = int(parts[6])
                                    current_gb = current_bytes / (1024**3)
                                    # Assume ~310 GB total (from ref4)
                                    progress_pct = min(current_gb / 310.0, 0.99)
                                    remaining_alignment = estimates['alignment_avg_sec'] * (1 - progress_pct)
                                    current_ref_remaining = max(0, remaining_alignment) + \
                                                          estimates['sorting_avg_sec'] + \
                                                          estimates['fasta_extraction_avg_sec']
                                    break
                    else:
                        # Fallback to time-based estimate
                        current_ref_remaining = max(0, estimates['alignment_avg_sec'] - elapsed) + \
                                              estimates['sorting_avg_sec'] + \
                                              estimates['fasta_extraction_avg_sec']
                except:
                    # Fallback to time-based estimate
                    current_ref_remaining = max(0, estimates['alignment_avg_sec'] - elapsed) + \
                                          estimates['sorting_avg_sec'] + \
                                          estimates['fasta_extraction_avg_sec']
            elif stage == 'sorting':
                current_ref_remaining = max(0, estimates['sorting_avg_sec'] - elapsed) + \
                                      estimates['fasta_extraction_avg_sec']
            elif stage == 'fasta_extraction':
                # Try to estimate based on file size
                try:
                    fasta_path = Path(f"benchmark_results/enhanced_privacy_k13_phase123_optimized/layer2_reference_pool/{ref_name}.fa.gz")
                    if fasta_path.exists():
                        current_mb = fasta_path.stat().st_size / (1024**2)
                        # Assume ~828 MB target
                        progress_pct = min(current_mb / 828.0, 0.99)
                        current_ref_remaining = max(0, estimates['fasta_extraction_avg_sec'] * (1 - progress_pct))
                    else:
                        current_ref_remaining = max(0, estimates['fasta_extraction_avg_sec'] - elapsed)
                except:
                    current_ref_remaining = max(0, estimates['fasta_extraction_avg_sec'] - elapsed)

    # Total remaining = current ref + remaining refs
    estimates['estimated_remaining_sec'] = current_ref_remaining + \
                                          (refs_remaining - 1) * estimates['total_per_ref_avg_sec']

    # Calculate completion time
    if estimates['estimated_remaining_sec'] > 0:
        from datetime import datetime, timedelta
        estimates['estimated_completion_time'] = datetime.now() + timedelta(seconds=estimates['estimated_remaining_sec'])

    return estimates

def draw_reference_grid(completed, total):
    """Draw a grid showing reference completion status."""
    result = ""
    refs_per_row = 6

    for i in range(total):
        ref_num = i + 1
        if ref_num <= completed:
            symbol = f"{Colors.OKGREEN}✅{Colors.ENDC}"
        else:
            symbol = f"{Colors.GRAY}⬜{Colors.ENDC}"

        result += f"{symbol} ref{ref_num:2d}  "

        if (i + 1) % refs_per_row == 0:
            result += "\n"

    return result.strip()

def format_stage_display(stage_info):
    """Format stage information for display."""
    stage = stage_info['stage']
    details = stage_info['details']

    # Stage emoji
    stage_emoji = {
        'alignment': '🔗',
        'partitioning': '✂️',
        'chromosome_sorting': '⚡',
        'sorting': '📊',
        'merging': '🔀',
        'indexing': '📇',
        'variant_calling': '🧬',
        'gdiff_encoding': '🔐',  # NEW: GDiff encoding
        'unknown': '❓'
    }

    emoji = stage_emoji.get(stage, '❓')
    stage_name = stage.replace('_', ' ').title()

    # Color based on stage
    if stage == 'chromosome_sorting':
        color = Colors.OKGREEN
    elif stage in ['alignment', 'variant_calling']:
        color = Colors.OKCYAN
    elif stage == 'gdiff_encoding':
        color = Colors.OKGREEN  # NEW: Green for encoding
    elif stage in ['partitioning', 'merging']:
        color = Colors.WARNING
    else:
        color = Colors.ENDC

    return f"{emoji} {color}{stage_name}{Colors.ENDC}: {details}"

def main():
    """Main tracker loop."""
    print(f"{Colors.BOLD}Starting Enhanced Pipeline Tracker...{Colors.ENDC}")
    print(f"{Colors.GRAY}Press Ctrl+C to exit{Colors.ENDC}\n")
    time.sleep(1)

    try:
        while True:
            clear_screen()

            # Get status
            status = get_pipeline_status()

            # Header
            print(f"{Colors.BOLD}{'═' * 80}{Colors.ENDC}")

            # Dynamic title based on pipeline type
            if status['script_name'] == 'run_k12_gdiff_pipeline.py':
                title = "GenomeVault k=11 GDiff Privacy-Preserving Encoding Tracker"
            else:
                title = "GenomeVault k=13 Chromosome-Parallel Pipeline Tracker"

            print(f"{Colors.BOLD}{Colors.HEADER}   {title}   {Colors.ENDC}")
            print(f"{Colors.BOLD}{'═' * 80}{Colors.ENDC}\n")

            # Pipeline Status
            if status['running']:
                status_text = f"{Colors.OKGREEN}RUNNING{Colors.ENDC}"
                pid_text = f"PID: {status['pid']}"
                script_text = f"Script: {status['script_name']}" if status['script_name'] else ""
            else:
                status_text = f"{Colors.FAIL}STOPPED{Colors.ENDC}"
                pid_text = "No active pipeline detected"
                script_text = ""

            print(f"{Colors.BOLD}Status:{Colors.ENDC} {status_text}  |  {pid_text}  {script_text}")
            print(f"{Colors.BOLD}Runtime:{Colors.ENDC} {status['runtime']}")
            print()

            # Current Stage
            if status['stage_info']['reference']:
                print(f"{Colors.BOLD}Current Reference:{Colors.ENDC} {Colors.OKCYAN}{status['stage_info']['reference']}{Colors.ENDC}")

            if status['stage_info']['stage'] != 'unknown':
                print(f"{Colors.BOLD}Stage:{Colors.ENDC} {format_stage_display(status['stage_info'])}")
                print()

            # GDiff Encoding Progress (if applicable)
            if status['stage_info'].get('gdiff_info'):
                gdiff = status['stage_info']['gdiff_info']
                print(f"{Colors.BOLD}GDiff Encoding Progress:{Colors.ENDC}")

                if gdiff['template_loaded']:
                    print(f"  ✓ Template: {Colors.OKGREEN}{gdiff['template_variants']:,}{Colors.ENDC} variant sites indexed")

                if gdiff['current_chromosome']:
                    print(f"  Current: {Colors.OKCYAN}{gdiff['current_chromosome']}{Colors.ENDC}", end="")
                    if gdiff['current_region']:
                        print(f" {Colors.GRAY}({gdiff['current_region']}){Colors.ENDC}", end="")
                    if gdiff['current_guide']:
                        print(f" {Colors.WARNING}[Guide {gdiff['current_guide']}]{Colors.ENDC}", end="")
                    print()

                if gdiff['total_variants'] > 0:
                    print(f"  Variants found: {Colors.OKGREEN}{gdiff['total_variants']:,}{Colors.ENDC}")

                print()

            # Detailed intra-alignment progress (NEW)
            if status['progress_details']:
                current_refs = [ref for ref in status['progress_details'] if status['progress_details'][ref]['stage'] != 'complete']
                if current_refs:
                    print(f"{Colors.BOLD}Detailed Progress:{Colors.ENDC}")
                    for ref in sorted(current_refs):
                        progress = status['progress_details'][ref]
                        stage_color = {
                            'alignment': Colors.OKCYAN,
                            'sorting': Colors.WARNING,
                            'indexing': Colors.OKBLUE,
                            'fasta_extraction': Colors.OKGREEN
                        }.get(progress['stage'], Colors.ENDC)

                        # Show elapsed time for current stage
                        elapsed_min = int(progress.get('elapsed_seconds', 0) / 60)
                        elapsed_sec = int(progress.get('elapsed_seconds', 0) % 60)

                        print(f"  {Colors.BOLD}{ref}:{Colors.ENDC} {stage_color}{progress['stage']}{Colors.ENDC} ({elapsed_min}m {elapsed_sec}s)")
                        print(f"    {Colors.GRAY}{progress['details']}{Colors.ENDC}")
                    print()

            # Chromosome-level progress (if applicable)
            if status['chromosome_details']:
                print(f"{Colors.BOLD}Chromosome-Level Sorting Progress:{Colors.ENDC}")
                for chr_detail in status['chromosome_details'][:10]:  # Show first 10
                    chr_name = chr_detail['chromosome']
                    cpu = chr_detail['cpu']
                    bar = draw_progress_bar(min(cpu * 10, 100), width=30, label=f"  {chr_name:5s}")
                    print(f"{bar}  {cpu:.1f}% CPU")

                if len(status['chromosome_details']) > 10:
                    print(f"  {Colors.GRAY}... and {len(status['chromosome_details']) - 10} more chromosomes{Colors.ENDC}")
                print()

            # Resource Usage
            print(f"{Colors.BOLD}Resource Usage:{Colors.ENDC}")
            cpu_bar = draw_progress_bar(min(status['cpu'] / 10, 100), label="CPU    ")
            mem_bar = draw_progress_bar(status['memory'], label="Memory ")
            print(f"{cpu_bar}  {status['cpu']:.1f}%")
            print(f"{mem_bar}  {status['memory']:.1f}%")
            print()

            # Active Processes
            if status['active_processes']:
                print(f"{Colors.BOLD}Active Workers:{Colors.ENDC}")
                for proc in status['active_processes'][:8]:  # Show first 8
                    print(f"  {Colors.OKCYAN}{proc['name']:12s}{Colors.ENDC}  CPU: {proc['cpu']:6.1f}%  Mem: {proc['mem']:4.1f}%  PID: {proc['pid']}")
                print()

            # Reference Progress
            print(f"{Colors.BOLD}Reference Pool Progress:{Colors.ENDC} {status['completed_refs']}/{status['total_refs']} complete")
            overall_pct = (status['completed_refs'] / status['total_refs']) * 100
            print(draw_progress_bar(overall_pct, width=60, label="Overall"))
            print()
            print(draw_reference_grid(status['completed_refs'], status['total_refs']))
            print()

            # Detailed Time Estimates
            if status['running'] and 'time_estimates' in status:
                estimates = status['time_estimates']

                print(f"{Colors.BOLD}Time Estimates:{Colors.ENDC}")

                # Show average times per stage
                if estimates.get('alignment_avg_sec', 0) > 0:
                    align_min = int(estimates['alignment_avg_sec'] / 60)
                    sort_min = int(estimates['sorting_avg_sec'] / 60)
                    extract_min = int(estimates['fasta_extraction_avg_sec'] / 60)
                    total_min = int(estimates['total_per_ref_avg_sec'] / 60)

                    print(f"  {Colors.GRAY}Avg per reference: {total_min}m (align: {align_min}m, sort: {sort_min}m, extract: {extract_min}m){Colors.ENDC}")

                # Show estimated remaining time
                if estimates.get('estimated_remaining_sec', 0) > 0:
                    remaining_hours = estimates['estimated_remaining_sec'] / 3600
                    remaining_mins = (estimates['estimated_remaining_sec'] % 3600) / 60

                    # Format completion time
                    if estimates.get('estimated_completion_time'):
                        completion_str = estimates['estimated_completion_time'].strftime('%Y-%m-%d %H:%M:%S')
                        completion_day = estimates['estimated_completion_time'].strftime('%a')

                        print(f"  {Colors.BOLD}Estimated Completion:{Colors.ENDC} {Colors.OKGREEN}{completion_str} ({completion_day}){Colors.ENDC}")
                        print(f"  {Colors.GRAY}(~{remaining_hours:.1f} hours / {remaining_mins:.0f} minutes remaining){Colors.ENDC}")

                print()

            print(f"\n{Colors.GRAY}Last updated: {datetime.now().strftime('%H:%M:%S')}  |  Press Ctrl+C to exit{Colors.ENDC}")
            print(f"{Colors.BOLD}{'═' * 80}{Colors.ENDC}")

            # Update every 30 seconds
            time.sleep(30)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.BOLD}Tracker stopped.{Colors.ENDC}")
        sys.exit(0)

if __name__ == "__main__":
    main()
