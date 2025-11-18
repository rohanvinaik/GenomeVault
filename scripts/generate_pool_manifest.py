#!/usr/bin/env python3
"""
generate_pool_manifest.py

Generates a comprehensive manifest for a reference pool by aggregating
metadata from all constituent samples.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import statistics

def load_sample_metadata(sample_dir: Path) -> Dict:
    """Load metadata.json from a sample directory."""
    metadata_file = sample_dir / "metadata.json"
    
    if not metadata_file.exists():
        print(f"Warning: No metadata found for {sample_dir.name}")
        return None
    
    with open(metadata_file) as f:
        return json.load(f)

def aggregate_pool_statistics(samples_metadata: List[Dict]) -> Dict:
    """Calculate aggregate statistics for the pool."""
    
    total_size = sum(
        s['file_info']['total_size_gb'] 
        for s in samples_metadata 
        if s['file_info'].get('total_size_gb')
    )
    
    # Extract quality metrics (if available)
    qualities = [
        s['quality_metrics'].get('mean_quality')
        for s in samples_metadata
        if isinstance(s['quality_metrics'].get('mean_quality'), (int, float))
    ]
    
    avg_quality = statistics.mean(qualities) if qualities else None
    
    # Extract coverage information
    coverages = [
        s['sequencing'].get('coverage')
        for s in samples_metadata
        if s['sequencing'].get('coverage')
    ]
    
    return {
        "total_samples": len(samples_metadata),
        "total_size_gb": round(total_size, 2),
        "avg_coverage": coverages[0] if coverages and len(set(coverages)) == 1 else "mixed",
        "avg_quality": round(avg_quality, 2) if avg_quality else "PENDING",
        "platforms": list(set(
            s['sequencing']['platform'] 
            for s in samples_metadata
        )),
        "models": list(set(
            s['sequencing'].get('model', 'UNKNOWN')
            for s in samples_metadata
        ))
    }

def determine_ancestry_from_path(pool_path: Path) -> tuple:
    """Determine ancestry group from directory path."""
    path_str = str(pool_path)
    
    if 'european_ancestry' in path_str:
        return 'European', ['CEU', 'GBR']
    elif 'east_asian_ancestry' in path_str:
        return 'East Asian', ['CHB', 'CHS', 'JPT']
    elif 'african_ancestry' in path_str:
        return 'African', ['YRI', 'LWK', 'GWD']
    elif 'south_asian_ancestry' in path_str:
        return 'South Asian', ['GIH', 'PJL', 'ITU']
    else:
        return 'UNKNOWN', []

def generate_pool_manifest(pool_id: str, pool_dir: Path, k_anonymity: int = 10) -> Dict:
    """Generate a comprehensive pool manifest."""
    
    # Find all sample subdirectories
    sample_dirs = [d for d in pool_dir.iterdir() if d.is_dir()]
    
    if not sample_dirs:
        print(f"Error: No sample directories found in {pool_dir}")
        return None
    
    print(f"Found {len(sample_dirs)} samples in pool")
    
    # Load all sample metadata
    samples_metadata = []
    for sample_dir in sorted(sample_dirs):
        metadata = load_sample_metadata(sample_dir)
        if metadata:
            samples_metadata.append(metadata)
        else:
            print(f"Warning: Skipping {sample_dir.name} (no metadata)")
    
    if not samples_metadata:
        print("Error: No valid sample metadata found")
        return None
    
    # Determine ancestry
    ancestry, subpopulations = determine_ancestry_from_path(pool_dir)
    
    # Aggregate statistics
    stats = aggregate_pool_statistics(samples_metadata)
    
    # Check processing status
    all_downloaded = all(
        s['genomevault_metadata']['processing_status']['downloaded']
        for s in samples_metadata
    )
    
    all_validated = all(
        s['genomevault_metadata']['processing_status']['validated']
        for s in samples_metadata
    )
    
    any_layer2_processed = any(
        s['genomevault_metadata']['processing_status'].get('layer2_processed', False)
        for s in samples_metadata
    )
    
    all_layer2_processed = all(
        s['genomevault_metadata']['processing_status'].get('layer2_processed', False)
        for s in samples_metadata
    )
    
    # Build manifest
    manifest = {
        "pool_id": pool_id,
        "pool_version": "1.0",
        "creation_date": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        
        "pool_config": {
            "k_anonymity": k_anonymity,
            "ancestry_group": ancestry,
            "subpopulations": subpopulations,
            "consensus_threshold": 0.95,
            "target_samples": k_anonymity,
            "actual_samples": len(samples_metadata)
        },
        
        "samples": [
            {
                "accession": s['accession_id'],
                "role": s['genomevault_metadata']['pool_role'],
                "priority": i + 1,
                "quality_tier": "high",  # Could be determined from QC metrics
                "size_gb": s['file_info'].get('total_size_gb'),
                "processing_status": s['genomevault_metadata']['processing_status']
            }
            for i, s in enumerate(samples_metadata)
        ],
        
        "statistics": stats,
        
        "processing_status": {
            "all_downloaded": all_downloaded,
            "all_validated": all_validated,
            "layer1_consensus": "pending",
            "layer2_alignment": "in_progress" if any_layer2_processed else "pending",
            "layer2_variant_calling": "pending",
            "ready_for_queries": all_layer2_processed
        },
        
        "security_metrics": {
            "initial_entropy": 260.0,
            "entropy_threshold": 128.0,
            "queries_processed": 0,
            "max_queries_before_rotation": 18,
            "last_rotation": None,
            "entropy_decay_rate": 14.4
        },
        
        "validation": {
            "manifest_generated": datetime.now().isoformat(),
            "samples_validated": all_validated,
            "integrity_check": "passed" if all_validated else "pending"
        }
    }
    
    return manifest

def main():
    parser = argparse.ArgumentParser(
        description='Generate manifest JSON for a reference pool'
    )
    
    parser.add_argument(
        '--pool-id',
        required=True,
        help='Pool identifier (e.g., european_ancestry_k10_pool_v1)'
    )
    
    parser.add_argument(
        '--pool-dir',
        required=True,
        help='Directory containing pool samples'
    )
    
    parser.add_argument(
        '--output',
        help='Output JSON file path (default: <pool-dir>/pool_manifest.json)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='k-anonymity value for this pool (default: 10)'
    )
    
    args = parser.parse_args()
    
    pool_dir = Path(args.pool_dir)
    
    if not pool_dir.exists():
        print(f"Error: Pool directory not found: {pool_dir}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = pool_dir / "pool_manifest.json"
    
    print(f"Generating pool manifest for {args.pool_id}...")
    print(f"Pool directory: {pool_dir}")
    print(f"k-anonymity: {args.k}")
    print("")
    
    # Generate manifest
    manifest = generate_pool_manifest(args.pool_id, pool_dir, k_anonymity=args.k)
    
    if not manifest:
        print("Error: Failed to generate manifest")
        sys.exit(1)
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Pool manifest generated: {output_path}")
    print("")
    print("Pool Summary:")
    print(f"  Pool ID: {manifest['pool_id']}")
    print(f"  Ancestry: {manifest['pool_config']['ancestry_group']}")
    print(f"  Samples: {manifest['statistics']['total_samples']}")
    print(f"  Total size: {manifest['statistics']['total_size_gb']} GB")
    print(f"  k-anonymity: {manifest['pool_config']['k_anonymity']}")
    print(f"  Ready for queries: {manifest['processing_status']['ready_for_queries']}")
    print("")
    
    if not manifest['processing_status']['all_validated']:
        print("⚠ Warning: Not all samples have been validated")
    
    if manifest['pool_config']['actual_samples'] != manifest['pool_config']['target_samples']:
        print(f"⚠ Warning: Pool has {manifest['pool_config']['actual_samples']} samples, target is {manifest['pool_config']['target_samples']}")

if __name__ == '__main__':
    main()
