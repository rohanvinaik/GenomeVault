#!/usr/bin/env python3
"""
GDiff Template Creation CLI

Command-line tool to build GDiff templates from public genomic databases.

Usage:
    python scripts/create_gdiff_template.py create \
        --gnomad data/gnomad/gnomad.genomes.v4.0.sites.vcf.gz \
        --dbsnp data/dbsnp/dbsnp_156.vcf.gz \
        --clinvar data/clinvar/clinvar_20231028.vcf.gz \
        --output data/templates/gdiff_template_GRCh38.json.gz \
        --reference-build GRCh38

    python scripts/create_gdiff_template.py stats \
        --template data/templates/gdiff_template_GRCh38.json.gz

    python scripts/create_gdiff_template.py lookup \
        --template data/templates/gdiff_template_GRCh38.json.gz \
        --chrom chr1 --pos 12345 --ref A --alt G
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add genomevault to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.differential_encoding.gdiff.template import TemplateBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_create(args):
    """
    Create a new GDiff template from public databases.
    """
    logger.info("=" * 80)
    logger.info("GDIFF TEMPLATE CREATION")
    logger.info("=" * 80)

    # Initialize builder
    builder = TemplateBuilder(
        reference_build=args.reference_build,
        output_dir=Path(args.output).parent,
    )

    # Load public databases
    logger.info("\nStep 1: Loading public databases...")
    builder.load_public_databases(
        gnomad_path=Path(args.gnomad) if args.gnomad else None,
        dbsnp_path=Path(args.dbsnp) if args.dbsnp else None,
        clinvar_path=Path(args.clinvar) if args.clinvar else None,
    )

    # Create sparse template
    logger.info("\nStep 2: Creating sparse template...")
    builder.create_sparse_template()

    # Build index
    logger.info("\nStep 3: Building index...")
    builder.build_index(index_type="hash")

    # Get statistics
    stats = builder.get_statistics()
    logger.info("\nTemplate Statistics:")
    logger.info(f"  Total variants: {stats['total_variants']:,}")
    logger.info(f"  Common variants (AF>0.01): {stats['common_variants']:,}")
    logger.info(f"  Rare variants (0<AF<0.01): {stats['rare_variants']:,}")
    logger.info(f"  Novel variants (AF=0): {stats['novel_variants']:,}")
    logger.info(f"  Clinical variants: {stats['clinical_variants']:,}")
    logger.info(f"  With database IDs: {stats['with_database_ids']:,}")
    logger.info(f"  Databases loaded: {', '.join(stats['databases_loaded'])}")

    # Save template
    logger.info("\nStep 4: Saving template...")
    output_path = builder.save_template(
        output_path=Path(args.output),
        compress=args.compress,
    )

    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Template saved to: {output_path}")
    logger.info("=" * 80)


def cmd_stats(args):
    """
    Display statistics for an existing template.
    """
    logger.info("=" * 80)
    logger.info("GDIFF TEMPLATE STATISTICS")
    logger.info("=" * 80)

    # Load template
    builder = TemplateBuilder()
    builder.load_template(Path(args.template))

    # Get statistics
    stats = builder.get_statistics()

    print("\nTemplate Statistics:")
    print(f"  Total variants: {stats['total_variants']:,}")
    print(f"  Common variants (AF>0.01): {stats['common_variants']:,}")
    print(f"  Rare variants (0<AF<0.01): {stats['rare_variants']:,}")
    print(f"  Novel variants (AF=0): {stats['novel_variants']:,}")
    print(f"  Clinical variants: {stats['clinical_variants']:,}")
    print(f"  With database IDs: {stats['with_database_ids']:,}")
    print(f"  Databases loaded: {', '.join(stats['databases_loaded'])}")


def cmd_lookup(args):
    """
    Lookup a specific variant in the template.
    """
    logger.info("=" * 80)
    logger.info("GDIFF TEMPLATE VARIANT LOOKUP")
    logger.info("=" * 80)

    # Load template
    builder = TemplateBuilder()
    builder.load_template(Path(args.template))

    # Lookup variant
    logger.info(f"\nLooking up: {args.chrom}:{args.pos} {args.ref}>{args.alt}")

    result = builder.lookup_variant(
        chrom=args.chrom,
        pos=args.pos,
        ref=args.ref,
        alt=args.alt,
    )

    if result is None:
        print("\n✗ Variant NOT FOUND in template (novel variant)")
        print("  Classification: novel")
        print("  Allele frequency: 0.0")
        print("  Database ID: None")
        print("  Clinical significance: None")
    else:
        print("\n✓ Variant FOUND in template")
        print(f"  Classification: {result.variant_class}")
        print(f"  Allele frequency: {result.allele_frequency:.6f}")
        print(f"  Database ID: {result.database_id or 'None'}")
        print(f"  Clinical significance: {result.clinical_significance or 'None'}")

        if result.population_frequencies:
            print("  Population frequencies:")
            for pop, freq in result.population_frequencies.items():
                print(f"    {pop}: {freq:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Create and query GDiff templates from public genomic databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Create template
  python scripts/create_gdiff_template.py create \\
      --gnomad data/gnomad/gnomad.genomes.v4.0.sites.vcf.gz \\
      --dbsnp data/dbsnp/dbsnp_156.vcf.gz \\
      --clinvar data/clinvar/clinvar_20231028.vcf.gz \\
      --output data/templates/gdiff_template_GRCh38.json.gz \\
      --reference-build GRCh38

  # View statistics
  python scripts/create_gdiff_template.py stats \\
      --template data/templates/gdiff_template_GRCh38.json.gz

  # Lookup variant
  python scripts/create_gdiff_template.py lookup \\
      --template data/templates/gdiff_template_GRCh38.json.gz \\
      --chrom chr1 --pos 12345 --ref A --alt G
        """
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Create command
    create_parser = subparsers.add_parser(
        'create',
        help='Create a new GDiff template'
    )
    create_parser.add_argument(
        '--gnomad',
        type=str,
        help='Path to gnomAD VCF file (v4.0)'
    )
    create_parser.add_argument(
        '--dbsnp',
        type=str,
        help='Path to dbSNP VCF file (build 156)'
    )
    create_parser.add_argument(
        '--clinvar',
        type=str,
        help='Path to ClinVar VCF file'
    )
    create_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output template file path (.json.gz)'
    )
    create_parser.add_argument(
        '--reference-build',
        type=str,
        default='GRCh38',
        choices=['GRCh38', 'GRCh37', 'hg38', 'hg19'],
        help='Reference genome build (default: GRCh38)'
    )
    create_parser.add_argument(
        '--compress',
        action='store_true',
        default=True,
        help='Compress template with gzip (default: True)'
    )
    create_parser.set_defaults(func=cmd_create)

    # Stats command
    stats_parser = subparsers.add_parser(
        'stats',
        help='Display statistics for existing template'
    )
    stats_parser.add_argument(
        '--template',
        type=str,
        required=True,
        help='Path to template file'
    )
    stats_parser.set_defaults(func=cmd_stats)

    # Lookup command
    lookup_parser = subparsers.add_parser(
        'lookup',
        help='Lookup a variant in the template'
    )
    lookup_parser.add_argument(
        '--template',
        type=str,
        required=True,
        help='Path to template file'
    )
    lookup_parser.add_argument(
        '--chrom',
        type=str,
        required=True,
        help='Chromosome (e.g., chr1)'
    )
    lookup_parser.add_argument(
        '--pos',
        type=int,
        required=True,
        help='Position (1-based)'
    )
    lookup_parser.add_argument(
        '--ref',
        type=str,
        required=True,
        help='Reference allele'
    )
    lookup_parser.add_argument(
        '--alt',
        type=str,
        required=True,
        help='Alternate allele'
    )
    lookup_parser.set_defaults(func=cmd_lookup)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
