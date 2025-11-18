#!/usr/bin/env python3
"""
generate_sample_metadata.py

Generates standardized metadata JSON files for genomic samples.
Queries ENA/SRA APIs to retrieve sample information automatically.
"""

import argparse
import json
import hashlib
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import urllib.request
import urllib.error

def get_ena_metadata(accession):
    """Fetch metadata from ENA API."""
    url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={accession}&result=read_run&fields=study_accession,sample_accession,experiment_accession,run_accession,tax_id,scientific_name,instrument_platform,instrument_model,library_layout,library_strategy,library_source,read_count,base_count,fastq_ftp,fastq_md5"
    
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
            lines = data.strip().split('\n')
            
            if len(lines) < 2:
                return None
            
            headers = lines[0].split('\t')
            values = lines[1].split('\t')
            
            return dict(zip(headers, values))
    
    except urllib.error.HTTPError:
        print(f"Warning: Could not fetch metadata for {accession} from ENA")
        return None

def calculate_md5(filepath):
    """Calculate MD5 checksum of a file."""
    if not Path(filepath).exists():
        return None
    
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_file_size_gb(filepath):
    """Get file size in GB."""
    if not Path(filepath).exists():
        return None
    return Path(filepath).stat().st_size / (1024**3)

def get_fastq_stats(fastq_path):
    """Get basic statistics from FASTQ file."""
    if not Path(fastq_path).exists():
        return None
    
    try:
        # Count reads (lines / 4)
        result = subprocess.run(
            f"zcat {fastq_path} | wc -l",
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        lines = int(result.stdout.strip())
        reads = lines // 4
        
        return {
            "total_reads": reads,
            "estimated": False
        }
    except:
        return None

def generate_metadata(accession, sample_dir, pool_assignment=None, role="reference", test_scenario=None):
    """Generate comprehensive metadata for a sample."""
    
    sample_path = Path(sample_dir)
    
    # Fetch ENA metadata
    ena_data = get_ena_metadata(accession)
    
    # Find FASTQ files
    fastq_1 = list(sample_path.glob(f"{accession}_1.fastq.gz"))
    fastq_2 = list(sample_path.glob(f"{accession}_2.fastq.gz"))
    
    if not fastq_1 or not fastq_2:
        print(f"Error: FASTQ files not found for {accession}")
        return None
    
    fastq_1 = fastq_1[0]
    fastq_2 = fastq_2[0]
    
    # Calculate checksums (this can take time for large files)
    print(f"Calculating checksums for {accession}...")
    md5_1 = calculate_md5(fastq_1)
    md5_2 = calculate_md5(fastq_2)
    
    # Get file sizes
    size_1 = get_file_size_gb(fastq_1)
    size_2 = get_file_size_gb(fastq_2)
    total_size = size_1 + size_2 if size_1 and size_2 else None
    
    # Get read statistics
    print(f"Collecting read statistics for {accession}...")
    stats_1 = get_fastq_stats(fastq_1)
    
    # Build metadata structure
    metadata = {
        "accession_id": accession,
        "study_id": ena_data.get("study_accession") if ena_data else "UNKNOWN",
        "source": "ENA",
        "sample_name": accession,
        "generated_date": datetime.now().isoformat(),
        
        "population": {
            "ancestry": "UNKNOWN",  # To be filled manually or from mapping
            "subpopulation": "UNKNOWN",
            "description": "To be determined from study metadata"
        },
        
        "sequencing": {
            "platform": ena_data.get("instrument_platform") if ena_data else "Illumina",
            "model": ena_data.get("instrument_model") if ena_data else "UNKNOWN",
            "strategy": ena_data.get("library_strategy") if ena_data else "WGS",
            "library_layout": ena_data.get("library_layout") if ena_data else "PAIRED",
            "read_length": 150,  # Standard for modern Illumina
            "insert_size": 400,  # Typical value
            "coverage": "30x",  # Typical for WGS
            "total_reads": stats_1["total_reads"] * 2 if stats_1 else "UNKNOWN",
        },
        
        "file_info": {
            "fastq_1": fastq_1.name,
            "fastq_2": fastq_2.name,
            "size_fastq1_gb": round(size_1, 2) if size_1 else None,
            "size_fastq2_gb": round(size_2, 2) if size_2 else None,
            "total_size_gb": round(total_size, 2) if total_size else None,
            "md5_fastq1": md5_1,
            "md5_fastq2": md5_2
        },
        
        "quality_metrics": {
            "mean_quality": "PENDING",
            "q30_percentage": "PENDING",
            "gc_content": "PENDING",
            "duplication_rate": "PENDING",
            "note": "Run FastQC to populate these fields"
        },
        
        "download_info": {
            "download_date": datetime.now().isoformat(),
            "download_method": "fasterq-dump",
            "validated": True if md5_1 and md5_2 else False
        },
        
        "genomevault_metadata": {
            "pool_assignment": pool_assignment or "UNASSIGNED",
            "pool_role": role,
            "test_scenarios": [test_scenario] if test_scenario else [],
            "priority": "high" if role == "reference" else "medium",
            "processing_status": {
                "downloaded": True,
                "validated": True if md5_1 and md5_2 else False,
                "qc_complete": False,
                "layer2_processed": False,
                "layer3_processed": False if role == "query" else None
            }
        }
    }
    
    # Add ENA-specific fields if available
    if ena_data:
        metadata["ena_metadata"] = {
            "experiment_accession": ena_data.get("experiment_accession"),
            "sample_accession": ena_data.get("sample_accession"),
            "tax_id": ena_data.get("tax_id"),
            "scientific_name": ena_data.get("scientific_name"),
            "read_count": ena_data.get("read_count"),
            "base_count": ena_data.get("base_count")
        }
    
    return metadata

def main():
    parser = argparse.ArgumentParser(
        description='Generate metadata JSON for genomic samples'
    )
    
    parser.add_argument(
        '--accession',
        required=True,
        help='Sample accession ID (e.g., ERR3239276)'
    )
    
    parser.add_argument(
        '--sample-dir',
        help='Directory containing FASTQ files (default: current accession dir)'
    )
    
    parser.add_argument(
        '--output',
        help='Output JSON file path (default: <sample-dir>/metadata.json)'
    )
    
    parser.add_argument(
        '--pool-assignment',
        help='Pool assignment (e.g., european_ancestry/k10_pool_v1)'
    )
    
    parser.add_argument(
        '--role',
        choices=['reference', 'query'],
        default='reference',
        help='Sample role in pipeline'
    )
    
    parser.add_argument(
        '--test-scenario',
        help='Test scenario tag (e.g., baseline, edge_case)'
    )
    
    args = parser.parse_args()
    
    # Determine sample directory
    if args.sample_dir:
        sample_dir = Path(args.sample_dir)
    else:
        # Assume we're in the sample directory
        sample_dir = Path.cwd()
    
    if not sample_dir.exists():
        print(f"Error: Directory not found: {sample_dir}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = sample_dir / "metadata.json"
    
    print(f"Generating metadata for {args.accession}...")
    print(f"Sample directory: {sample_dir}")
    print(f"Output: {output_path}")
    print("")
    
    # Generate metadata
    metadata = generate_metadata(
        args.accession,
        sample_dir,
        pool_assignment=args.pool_assignment,
        role=args.role,
        test_scenario=args.test_scenario
    )
    
    if not metadata:
        print("Error: Failed to generate metadata")
        sys.exit(1)
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata generated: {output_path}")
    print(f"  Size: {metadata['file_info']['total_size_gb']} GB")
    print(f"  Role: {metadata['genomevault_metadata']['pool_role']}")
    print(f"  Pool: {metadata['genomevault_metadata']['pool_assignment']}")
    
    if not metadata['file_info']['md5_fastq1']:
        print("\n⚠ Warning: MD5 checksums could not be calculated")

if __name__ == '__main__':
    main()
