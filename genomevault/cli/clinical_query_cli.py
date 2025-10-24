"""
genomevault/cli/clinical_query_cli.py

Command-line interface for clinical variant queries
"""

import click
import json
from pathlib import Path
from typing import Optional

from genomevault.clinical_db.database import ClinicalSNPDatabase


@click.group()
@click.option('--db-path', default='data/clinical_snps_v1.0.0.json.gz',
              help='Path to clinical SNP database')
@click.pass_context
def clinical(ctx, db_path):
    """Clinical variant query commands"""
    ctx.ensure_object(dict)
    
    # Load database
    if not Path(db_path).exists():
        click.echo(f"Error: Database not found at {db_path}", err=True)
        click.echo("Run 'python -m genomevault.clinical_db.data_acquisition' to build database", err=True)
        ctx.exit(1)
    
    try:
        ctx.obj['db'] = ClinicalSNPDatabase(db_path)
        ctx.obj['db_path'] = db_path
    except Exception as e:
        click.echo(f"Error loading database: {e}", err=True)
        ctx.exit(1)


@clinical.command()
@click.pass_context
def stats(ctx):
    """Show database statistics"""
    db = ctx.obj['db']
    stats = db.get_statistics()
    
    click.echo("\n" + "=" * 60)
    click.echo("CLINICAL SNP DATABASE STATISTICS")
    click.echo("=" * 60)
    
    for key, value in stats.items():
        click.echo(f"{key:30s}: {value}")
    
    click.echo("=" * 60 + "\n")


@clinical.command()
@click.option('--chr', required=True, help='Chromosome (e.g., chr11)')
@click.option('--pos', required=True, type=int, help='Position (e.g., 5227002)')
@click.option('--ref', help='Reference allele (optional)')
@click.option('--alt', help='Alternate allele (optional)')
@click.option('--format', type=click.Choice(['json', 'detailed']), 
              default='detailed', help='Output format')
@click.pass_context
def query_position(ctx, chr, pos, ref, alt, format):
    """Query clinical significance at a genomic position"""
    db = ctx.obj['db']
    results = db.query_position(chr, pos)
    
    if not results:
        click.echo(f"No clinical variants found at {chr}:{pos}")
        return
    
    # Filter by alleles if provided
    if ref and alt:
        results = [
            s for s in results 
            if s.ref_allele == ref and alt in s.alt_alleles
        ]
        
        if not results:
            click.echo(f"No variants matching {ref}>{alt} at {chr}:{pos}")
            return
    
    # Format output
    if format == 'json':
        output = [s.to_dict() for s in results]
        click.echo(json.dumps(output, indent=2))
    
    else:  # detailed
        for i, snp in enumerate(results, 1):
            click.echo(f"\n{'=' * 60}")
            click.echo(f"VARIANT {i}/{len(results)}")
            click.echo('=' * 60)
            click.echo(f"SNP ID:                 {snp.snp_id}")
            click.echo(f"Position:               {snp.chromosome}:{snp.position}")
            click.echo(f"Gene:                   {snp.gene or 'Unknown'}")
            click.echo(f"Alleles:                {snp.ref_allele} → {', '.join(snp.alt_alleles)}")
            click.echo(f"Clinical Significance:  {snp.clinical_significance}")
            
            if snp.conditions:
                click.echo(f"\nConditions:")
                for cond in snp.conditions:
                    click.echo(f"  • {cond.name}")
                    if cond.omim_id:
                        click.echo(f"    OMIM: {cond.omim_id}")
                    if cond.inheritance:
                        click.echo(f"    Inheritance: {cond.inheritance}")
            
            if snp.clinical_annotations:
                click.echo(f"\nReview Status:          {snp.clinical_annotations.review_status}")
                click.echo(f"Stars:                  {'⭐' * snp.clinical_annotations.stars}")
            
            if snp.sources:
                click.echo(f"\nSources:")
                for source, id in snp.sources.items():
                    if id:
                        click.echo(f"  {source:20s}: {id}")


@clinical.command()
@click.argument('gene-symbol')
@click.option('--pathogenic-only', is_flag=True, help='Only show pathogenic variants')
@click.option('--format', type=click.Choice(['json', 'summary']), 
              default='summary', help='Output format')
@click.pass_context
def query_gene(ctx, gene_symbol, pathogenic_only, format):
    """Query all clinical variants in a gene"""
    db = ctx.obj['db']
    results = db.query_gene(gene_symbol)
    
    if not results:
        click.echo(f"No clinical variants found for gene {gene_symbol}")
        return
    
    # Filter pathogenic if requested
    if pathogenic_only:
        results = [s for s in results if s.is_pathogenic()]
        if not results:
            click.echo(f"No pathogenic variants found in {gene_symbol}")
            return
    
    # Format output
    if format == 'json':
        output = [s.to_dict() for s in results]
        click.echo(json.dumps(output, indent=2))
    
    else:  # summary
        click.echo(f"\n{'=' * 60}")
        click.echo(f"CLINICAL VARIANTS IN {gene_symbol}")
        click.echo('=' * 60)
        click.echo(f"Total variants:         {len(results)}")
        
        pathogenic = [s for s in results if s.is_pathogenic()]
        benign = [s for s in results if 'benign' in s.clinical_significance.lower()]
        vus = [s for s in results if 'uncertain' in s.clinical_significance.lower()]
        
        click.echo(f"Pathogenic:             {len(pathogenic)}")
        click.echo(f"Benign:                 {len(benign)}")
        click.echo(f"VUS:                    {len(vus)}")
        
        if results:
            click.echo(f"\nChromosome:             {results[0].chromosome}")
        
        if pathogenic:
            click.echo("\nTop Pathogenic Variants:")
            for snp in pathogenic[:5]:
                stars = '⭐' * (snp.clinical_annotations.stars if snp.clinical_annotations else 0)
                click.echo(f"  • {snp.snp_id} ({snp.chromosome}:{snp.position}) {stars}")
                if snp.conditions:
                    click.echo(f"    {snp.conditions[0].name}")


@clinical.command()
@click.argument('rs-id')
@click.pass_context
def query_rsid(ctx, rs_id):
    """Query specific SNP by dbSNP ID"""
    db = ctx.obj['db']
    snp = db.query_rsid(rs_id)
    
    if not snp:
        click.echo(f"SNP {rs_id} not found in database")
        return
    
    click.echo(f"\n{'=' * 60}")
    click.echo(f"SNP DETAILS: {rs_id}")
    click.echo('=' * 60)
    click.echo(f"Position:               {snp.chromosome}:{snp.position}")
    click.echo(f"Gene:                   {snp.gene or 'Unknown'}")
    click.echo(f"Alleles:                {snp.ref_allele} → {', '.join(snp.alt_alleles)}")
    click.echo(f"Clinical Significance:  {snp.clinical_significance}")
    
    if snp.conditions:
        click.echo(f"\nConditions:")
        for cond in snp.conditions:
            click.echo(f"  • {cond.name}")
    
    if snp.clinical_annotations:
        click.echo(f"\nReview Status:          {snp.clinical_annotations.review_status}")
        click.echo(f"Stars:                  {'⭐' * snp.clinical_annotations.stars}")


# Main entry point
if __name__ == '__main__':
    clinical()
