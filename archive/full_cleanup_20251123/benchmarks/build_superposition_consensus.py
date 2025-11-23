#!/usr/bin/env python3
"""
Build Superposition Consensus from Multiple Reference Genomes

Extends Byzantine consensus with graph-based superposition support, representing
multiple valid alignment paths for variable genomic regions.

Features:
- Conserved regions (95-99% agreement) → single path
- Variable regions (structural variants, common indels) → multiple paths
- Population variant integration (gnomAD, 1000 Genomes)
- Export to variation graph formats (VG, GFA, multi-FASTA)

Performance Target:
- 95-99% of genome uses single path (fast alignment)
- 1-5% uses multiple paths (population-aware)
- Total size: ~1.2GB for whole genome (1.2× single reference)

Usage:
    # Build from hg38, hg19, and T2T-CHM13
    python benchmarks/build_superposition_consensus.py \
        --references data/hg38.fa.gz data/hg19.fa.gz data/chm13v2.0.fa.gz \
        --output data/consensus_superposition/ \
        --chromosomes chr22 \
        --conservation-threshold 0.95 \
        --threads 8

    # Include population variants from gnomAD
    python benchmarks/build_superposition_consensus.py \
        --references data/hg38.fa.gz data/hg19.fa.gz data/chm13v2.0.fa.gz \
        --population-variants data/gnomad.v3.1.2.vcf.gz \
        --output data/consensus_superposition/ \
        --chromosomes chr22 \
        --population-frequency 0.01 \
        --threads 8

    # Quick test with chr22 only
    python benchmarks/build_superposition_consensus.py \
        --references data/hg38.fa.gz data/hg19.fa.gz \
        --output data/consensus_test/ \
        --chromosomes chr22 \
        --threads 4

Examples:
    # Example 1: Basic superposition consensus (no population data)
    python benchmarks/build_superposition_consensus.py \
        --references data/reference_genomes/hg38.fa.gz \
                     data/reference_genomes/hg19.fa.gz \
                     data/reference_genomes/chm13v2.0.fa.gz \
        --output benchmark_results/superposition_consensus/ \
        --chromosomes chr22 \
        --conservation-threshold 0.95 \
        --threads 8

    # Example 2: Full pipeline with population variants
    python benchmarks/build_superposition_consensus.py \
        --references data/reference_genomes/hg38.fa.gz \
                     data/reference_genomes/hg19.fa.gz \
                     data/reference_genomes/chm13v2.0.fa.gz \
        --population-variants data/gnomad/gnomad.genomes.v3.1.2.sites.chr22.vcf.gz \
        --output benchmark_results/superposition_consensus_full/ \
        --chromosomes chr22 \
        --conservation-threshold 0.95 \
        --population-frequency 0.01 \
        --threads 8

    # Example 3: Research mode (lower conservation threshold)
    python benchmarks/build_superposition_consensus.py \
        --references data/reference_genomes/hg38.fa.gz \
                     data/reference_genomes/hg19.fa.gz \
        --output benchmark_results/superposition_research/ \
        --chromosomes chr22 \
        --conservation-threshold 0.90 \
        --population-frequency 0.05 \
        --threads 4
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.reference import build_superposition_consensus

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_inputs(args):
    """Validate input files and arguments."""
    logger.info("Validating inputs...")

    # Check reference files exist
    for ref in args.references:
        ref_path = Path(ref)
        if not ref_path.exists():
            logger.error(f"Reference file not found: {ref}")
            return False
        logger.info(f"  ✓ Found reference: {ref_path.name}")

    # Check population VCF if provided
    if args.population_variants:
        pop_path = Path(args.population_variants)
        if not pop_path.exists():
            logger.error(f"Population VCF not found: {args.population_variants}")
            return False
        logger.info(f"  ✓ Found population VCF: {pop_path.name}")

    # Validate thresholds
    if not 0.0 <= args.conservation_threshold <= 1.0:
        logger.error(f"Conservation threshold must be in [0, 1]: {args.conservation_threshold}")
        return False

    if not 0.0 <= args.population_frequency <= 1.0:
        logger.error(f"Population frequency must be in [0, 1]: {args.population_frequency}")
        return False

    logger.info("  ✓ All inputs valid")
    return True


def print_output_summary(output_files: dict, output_dir: Path):
    """Print summary of output files."""
    logger.info("")
    logger.info("="*80)
    logger.info("OUTPUT FILES")
    logger.info("="*80)

    for file_type, file_path in sorted(output_files.items()):
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"  {file_type:25s} {file_path.name:40s} ({size_mb:.2f} MB)")

    # Calculate total size
    total_size_mb = sum(
        f.stat().st_size / (1024 * 1024)
        for f in output_files.values()
        if f.exists()
    )
    logger.info(f"  {'Total size':25s} {total_size_mb:.2f} MB")

    logger.info("")
    logger.info(f"All output files saved to: {output_dir}")
    logger.info("="*80)


def analyze_results(output_files: dict):
    """Analyze and report superposition consensus results."""
    logger.info("")
    logger.info("="*80)
    logger.info("SUPERPOSITION CONSENSUS ANALYSIS")
    logger.info("="*80)

    # Load statistics
    stats_file = output_files.get('stats_json')
    if stats_file and stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        summary = stats.get('summary', {})
        super_stats = stats.get('superposition_stats', {})

        logger.info("")
        logger.info("Structure:")
        logger.info(f"  Conserved regions:        {super_stats.get('conserved_regions', 0):,}")
        logger.info(f"  Variable regions:         {super_stats.get('variable_regions', 0):,}")
        logger.info(f"  Total paths:              {super_stats.get('total_paths', 0):,}")

        logger.info("")
        logger.info("Genome Coverage:")
        logger.info(f"  Conserved bases:          {super_stats.get('conserved_bases', 0):,}")
        logger.info(f"  Variable bases:           {super_stats.get('variable_bases', 0):,}")
        logger.info(f"  Conservation rate:        {summary.get('conservation_rate', 0):.2f}%")

        logger.info("")
        logger.info("Paths:")
        logger.info(f"  Avg paths per variable:   {summary.get('avg_paths_per_variable_region', 0):.2f}")

        logger.info("")
        logger.info("Population Variants:")
        logger.info(f"  Loaded from database:     {super_stats.get('population_variants_loaded', 0):,}")
        logger.info(f"  Used in paths:            {super_stats.get('population_variants_used', 0):,}")

    # Load path information
    paths_file = output_files.get('paths_json')
    if paths_file and paths_file.exists():
        with open(paths_file, 'r') as f:
            paths_data = json.load(f)

        total_nodes = sum(len(nodes) for nodes in paths_data.values())
        logger.info("")
        logger.info(f"Total superposition nodes: {total_nodes:,}")

        # Find interesting examples
        multi_path_nodes = []
        for chrom, nodes in paths_data.items():
            for node in nodes:
                if not node['is_conserved'] and len(node['paths']) > 1:
                    multi_path_nodes.append((chrom, node))

        if multi_path_nodes:
            logger.info("")
            logger.info("Example Variable Regions:")
            for i, (chrom, node) in enumerate(multi_path_nodes[:3], 1):
                logger.info(f"  {i}. {chrom}:{node['position']}-{node['end_position']} "
                           f"({len(node['paths'])} paths)")
                for j, path in enumerate(node['paths'][:3], 1):
                    logger.info(f"      Path {j}: freq={path['population_frequency']:.4f}, "
                               f"conf={path['confidence']:.4f}, "
                               f"len={len(path['sequence'])}bp")

    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Build superposition consensus reference with graph-based support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic superposition consensus
  python benchmarks/build_superposition_consensus.py \\
      --references data/hg38.fa.gz data/hg19.fa.gz data/chm13v2.0.fa.gz \\
      --output data/consensus_superposition/ \\
      --chromosomes chr22

  # With population variants
  python benchmarks/build_superposition_consensus.py \\
      --references data/hg38.fa.gz data/hg19.fa.gz data/chm13v2.0.fa.gz \\
      --population-variants data/gnomad.v3.1.2.vcf.gz \\
      --output data/consensus_superposition/ \\
      --chromosomes chr22 \\
      --population-frequency 0.01

Output Files:
  consensus_linear.fa       Linear consensus (conserved regions)
  superposition_paths.json  Alternative path metadata
  conserved_regions.bed     95-99% conserved regions
  variable_regions.bed      1-5% variable regions
  path_statistics.json      Comprehensive statistics
  consensus.vg              Variation graph (if enabled)
        """
    )

    parser.add_argument(
        '--references',
        nargs='+',
        required=True,
        help='Paths to reference FASTA files (.fa or .fa.gz)'
    )
    parser.add_argument(
        '--population-variants',
        help='Path to population variant VCF (gnomAD, 1000 Genomes)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for superposition consensus'
    )
    parser.add_argument(
        '--chromosomes',
        nargs='+',
        help='Specific chromosomes to process (default: all)'
    )
    parser.add_argument(
        '--conservation-threshold',
        type=float,
        default=0.95,
        help='Minimum agreement for conserved region (default: 0.95)'
    )
    parser.add_argument(
        '--population-frequency',
        type=float,
        default=0.01,
        help='Minimum population allele frequency to include (default: 0.01)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=8,
        help='Number of threads (default: 8)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=100,
        help='Window size for conservation analysis in bp (default: 100)'
    )
    parser.add_argument(
        '--no-graph',
        action='store_true',
        help='Disable variation graph export'
    )

    args = parser.parse_args()

    # Validate inputs
    if not validate_inputs(args):
        return 1

    # Convert paths
    references = [Path(r) for r in args.references]
    output_dir = Path(args.output)
    population_vcf = Path(args.population_variants) if args.population_variants else None

    # Print configuration
    logger.info("")
    logger.info("="*80)
    logger.info("SUPERPOSITION CONSENSUS CONFIGURATION")
    logger.info("="*80)
    logger.info(f"  References:              {len(references)}")
    for ref in references:
        logger.info(f"    - {ref.name}")
    logger.info(f"  Population VCF:          {population_vcf.name if population_vcf else 'None'}")
    logger.info(f"  Output directory:        {output_dir}")
    logger.info(f"  Chromosomes:             {', '.join(args.chromosomes) if args.chromosomes else 'all'}")
    logger.info(f"  Conservation threshold:  {args.conservation_threshold}")
    logger.info(f"  Population frequency:    {args.population_frequency}")
    logger.info(f"  Window size:             {args.window_size} bp")
    logger.info(f"  Threads:                 {args.threads}")
    logger.info(f"  Export variation graph:  {not args.no_graph}")
    logger.info("="*80)
    logger.info("")

    # Build superposition consensus
    start_time = time.time()

    try:
        output_files = build_superposition_consensus(
            references=references,
            output_dir=output_dir,
            population_vcf=population_vcf,
            conservation_threshold=args.conservation_threshold,
            population_variant_threshold=args.population_frequency,
            chromosomes=args.chromosomes,
            threads=args.threads
        )

        elapsed_time = time.time() - start_time

        # Print results
        logger.info("")
        logger.info("="*80)
        logger.info("SUPERPOSITION CONSENSUS COMPLETE")
        logger.info("="*80)
        logger.info(f"  Total time:              {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        logger.info("="*80)

        # Print output summary
        print_output_summary(output_files, output_dir)

        # Analyze results
        analyze_results(output_files)

        logger.info("")
        logger.info("✓ Superposition consensus building complete!")

        return 0

    except Exception as e:
        logger.error(f"Error building superposition consensus: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
