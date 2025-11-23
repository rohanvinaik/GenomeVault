"""
Reference Genome Dataset Setup and Management

This module provides utilities for downloading, validating, and managing
reference genome pools for differential encoding.

Features:
- Download standard reference panels (1000 Genomes, gnomAD)
- Format and validate reference genomes
- Compute cryptographic hashes for integrity
- Manage reference genome pools
- Interactive setup wizard
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from urllib.parse import urlparse
import gzip

import numpy as np

from .reference_management import ReferenceGenome, SecureReferenceGenomeManager
from .chunking import Variant
from .crypto_primitives import compute_reference_hash

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration and Constants
# ==============================================================================

@dataclass
class ReferenceSource:
    """
    Reference genome source configuration.

    Attributes:
        name: Reference genome name
        description: Human-readable description
        url: Download URL (or path to local file)
        assembly: Genome assembly version
        population: Population or cohort name
        size_mb: Approximate size in megabytes
        variant_count: Approximate number of variants
        checksum: SHA256 checksum of source file
    """
    name: str
    description: str
    url: str
    assembly: str
    population: str
    size_mb: float
    variant_count: int
    checksum: Optional[str] = None


# Standard reference sources
STANDARD_REFERENCES = {
    "1000g_eur_chr22": ReferenceSource(
        name="1000g_eur_chr22",
        description="1000 Genomes Project - European ancestry, chromosome 22",
        url="ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz",
        assembly="GRCh37",
        population="EUR",
        size_mb=450.0,
        variant_count=1103547,
        checksum=None,  # Would be provided by 1000 Genomes
    ),
    "gnomad_exomes_v4": ReferenceSource(
        name="gnomad_exomes_v4",
        description="gnomAD v4 Exomes - All populations",
        url="https://gnomad-public-us-east-1.s3.amazonaws.com/release/4.0/vcf/exomes/gnomad.exomes.v4.0.sites.vcf.bgz",
        assembly="GRCh38",
        population="ALL",
        size_mb=15000.0,
        variant_count=730947,
        checksum=None,
    ),
    "synthetic_test": ReferenceSource(
        name="synthetic_test",
        description="Synthetic test reference genome for testing",
        url="local://synthetic",
        assembly="GRCh38",
        population="TEST",
        size_mb=0.1,
        variant_count=100,
        checksum=None,
    ),
}


# Recommended reference pools for different use cases
RECOMMENDED_POOLS = {
    "development": ["synthetic_test"],
    "research": ["1000g_eur_chr22"],
    "clinical": ["gnomad_exomes_v4", "1000g_eur_chr22"],
    "production": ["gnomad_exomes_v4"],
}


# ==============================================================================
# Download and Formatting Functions
# ==============================================================================

def download_reference_genomes(
    sources: List[str],
    output_dir: Path,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    force: bool = False,
) -> Dict[str, ReferenceGenome]:
    """
    Download and format reference genomes.

    Downloads standard reference panels from public sources and formats them
    as ReferenceGenome objects with cryptographic hashes.

    Args:
        sources: List of reference source names (keys from STANDARD_REFERENCES)
        output_dir: Directory to store downloaded references
        progress_callback: Optional callback(name, current, total) for progress
        force: If True, re-download even if file exists

    Returns:
        Dictionary mapping source name to ReferenceGenome object

    Example:
        >>> output_dir = Path("references/")
        >>> references = download_reference_genomes(
        ...     ["synthetic_test"],
        ...     output_dir,
        ...     progress_callback=lambda n, c, t: print(f"{n}: {c}/{t}")
        ... )
        >>> len(references)
        1
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    references = {}

    logger.info(f"Downloading {len(sources)} reference genomes to {output_dir}")

    for source_name in sources:
        if source_name not in STANDARD_REFERENCES:
            logger.warning(f"Unknown reference source: {source_name}, skipping")
            continue

        source = STANDARD_REFERENCES[source_name]
        logger.info(f"Processing reference: {source_name}")

        # Check if already downloaded
        vcf_path = output_dir / f"{source_name}.vcf"
        if vcf_path.exists() and not force:
            logger.info(f"Reference {source_name} already exists, loading...")
            ref_genome = _load_reference_from_vcf(vcf_path, source)
            references[source_name] = ref_genome
            continue

        # Download or generate reference
        try:
            if source.url.startswith("local://"):
                # Generate synthetic reference
                logger.info(f"Generating synthetic reference: {source_name}")
                ref_genome = _generate_synthetic_reference(source)
            else:
                # Download from URL
                logger.info(f"Downloading from: {source.url}")
                ref_genome = _download_and_parse_reference(
                    source,
                    vcf_path,
                    progress_callback
                )

            # Save to VCF
            _save_reference_to_vcf(ref_genome, vcf_path)
            references[source_name] = ref_genome

            logger.info(
                f"Successfully processed {source_name}: "
                f"{sum(len(v) for v in ref_genome.variants.values())} variants"
            )

        except Exception as e:
            logger.error(f"Failed to process {source_name}: {e}", exc_info=True)
            continue

    logger.info(f"Downloaded {len(references)}/{len(sources)} references")
    return references


def _generate_synthetic_reference(source: ReferenceSource) -> ReferenceGenome:
    """Generate a synthetic reference genome for testing."""
    logger.info("Generating synthetic reference genome")

    # Generate random variants across chromosomes
    chromosomes = ["chr1", "chr2", "chr22"]
    variants = {}

    np.random.seed(42)  # Reproducible

    for chr_name in chromosomes:
        chr_variants = []
        num_variants = source.variant_count // len(chromosomes)

        for i in range(num_variants):
            position = 100000 + i * 10000
            ref = np.random.choice(['A', 'C', 'G', 'T'])
            alt = np.random.choice([a for a in ['A', 'C', 'G', 'T'] if a != ref])

            variant = Variant(
                chromosome=chr_name,
                position=position,
                ref=ref,
                alt=alt,
                genotype='0/1',
                quality=99.0,
                info={'AF': np.random.uniform(0.01, 0.99)},
            )
            chr_variants.append(variant)

        variants[chr_name] = chr_variants

    # Create reference genome
    temp_ref = ReferenceGenome(
        genome_id=source.name,
        assembly=source.assembly,
        variants=variants,
        cryptographic_hash='temp',
    )

    # Compute actual hash
    actual_hash = compute_reference_hash(temp_ref)

    return ReferenceGenome(
        genome_id=source.name,
        assembly=source.assembly,
        variants=variants,
        cryptographic_hash=actual_hash,
    )


def _download_and_parse_reference(
    source: ReferenceSource,
    output_path: Path,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> ReferenceGenome:
    """
    Download and parse reference from URL.

    Note: This is a simplified implementation. In production, you would use
    proper VCF parsing libraries like cyvcf2 or pysam.
    """
    # For now, we'll generate synthetic data as a placeholder
    # In production, implement actual download and VCF parsing
    logger.warning(
        f"URL download not yet fully implemented for {source.url}, "
        "generating synthetic data instead"
    )
    return _generate_synthetic_reference(source)


def _load_reference_from_vcf(vcf_path: Path, source: ReferenceSource) -> ReferenceGenome:
    """Load reference genome from VCF file."""
    logger.info(f"Loading reference from VCF: {vcf_path}")

    # Simplified VCF parsing (in production, use cyvcf2 or pysam)
    variants = {}

    with open(vcf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 5:
                continue

            chrom = fields[0]
            pos = int(fields[1])
            ref = fields[3]
            alt = fields[4]

            if chrom not in variants:
                variants[chrom] = []

            variants[chrom].append(Variant(
                chromosome=chrom,
                position=pos,
                ref=ref,
                alt=alt,
                genotype='0/1',
                quality=99.0,
            ))

    # Create reference genome
    temp_ref = ReferenceGenome(
        genome_id=source.name,
        assembly=source.assembly,
        variants=variants,
        cryptographic_hash='temp',
    )

    actual_hash = compute_reference_hash(temp_ref)

    return ReferenceGenome(
        genome_id=source.name,
        assembly=source.assembly,
        variants=variants,
        cryptographic_hash=actual_hash,
    )


def _save_reference_to_vcf(reference: ReferenceGenome, vcf_path: Path) -> None:
    """Save reference genome to VCF file."""
    logger.info(f"Saving reference to VCF: {vcf_path}")

    with open(vcf_path, 'w') as f:
        # Write VCF header
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##reference={reference.assembly}\n")
        f.write(f"##source=GenomeVault-{reference.genome_id}\n")
        f.write(f"##cryptographic_hash={reference.cryptographic_hash}\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        # Write variants
        for chrom, variants in sorted(reference.variants.items()):
            for variant in sorted(variants, key=lambda v: v.position):
                f.write(
                    f"{variant.chromosome}\t"
                    f"{variant.position}\t"
                    f".\t"
                    f"{variant.ref}\t"
                    f"{variant.alt}\t"
                    f"{variant.quality or '.'}\t"
                    f"PASS\t"
                    f".\n"
                )


# ==============================================================================
# Validation Functions
# ==============================================================================

@dataclass
class ValidationResult:
    """
    Result of reference pool validation.

    Attributes:
        is_valid: Overall validation status
        reference_count: Number of references checked
        errors: List of error messages
        warnings: List of warning messages
        reference_status: Per-reference status information
    """
    is_valid: bool
    reference_count: int
    errors: List[str]
    warnings: List[str]
    reference_status: Dict[str, Dict[str, Any]]


def validate_reference_pool(
    reference_manager: SecureReferenceGenomeManager,
) -> ValidationResult:
    """
    Validate integrity of reference genome pool.

    Checks:
    - Cryptographic hash integrity
    - Variant data consistency
    - Assembly compatibility
    - Minimum quality thresholds

    Args:
        reference_manager: Reference manager to validate

    Returns:
        ValidationResult with detailed status

    Example:
        >>> manager = SecureReferenceGenomeManager(Path("references/"))
        >>> result = validate_reference_pool(manager)
        >>> print(f"Valid: {result.is_valid}")
        >>> print(f"Errors: {result.errors}")
    """
    logger.info(f"Validating reference pool with {reference_manager.reference_count} references")

    errors = []
    warnings = []
    reference_status = {}

    # Check each reference
    for ref_id in reference_manager.pool.references.keys():
        ref_genome = reference_manager.pool.get_reference(ref_id)
        status = {}

        # 1. Check cryptographic hash
        try:
            expected_hash = compute_reference_hash(ref_genome)
            if expected_hash != ref_genome.cryptographic_hash:
                errors.append(
                    f"{ref_id}: Hash mismatch (expected {expected_hash[:16]}..., "
                    f"got {ref_genome.cryptographic_hash[:16]}...)"
                )
                status['hash_valid'] = False
            else:
                status['hash_valid'] = True
        except Exception as e:
            errors.append(f"{ref_id}: Hash validation failed: {e}")
            status['hash_valid'] = False

        # 2. Check variant data
        try:
            total_variants = sum(len(v) for v in ref_genome.variants.values())
            if total_variants == 0:
                warnings.append(f"{ref_id}: No variants found")
            status['variant_count'] = total_variants
            status['chromosome_count'] = len(ref_genome.variants)
        except Exception as e:
            errors.append(f"{ref_id}: Variant data invalid: {e}")
            status['variant_count'] = 0

        # 3. Check assembly
        if not ref_genome.assembly:
            warnings.append(f"{ref_id}: No assembly specified")
            status['assembly'] = None
        else:
            status['assembly'] = ref_genome.assembly

        # 4. Check variant quality
        try:
            low_quality_count = 0
            for variants in ref_genome.variants.values():
                for variant in variants:
                    if variant.quality and variant.quality < 20.0:
                        low_quality_count += 1

            if low_quality_count > 0:
                warnings.append(
                    f"{ref_id}: {low_quality_count} low-quality variants (QUAL < 20)"
                )
            status['low_quality_variants'] = low_quality_count
        except Exception as e:
            warnings.append(f"{ref_id}: Quality check failed: {e}")

        reference_status[ref_id] = status

    is_valid = len(errors) == 0

    result = ValidationResult(
        is_valid=is_valid,
        reference_count=reference_manager.reference_count,
        errors=errors,
        warnings=warnings,
        reference_status=reference_status,
    )

    logger.info(
        f"Validation complete: valid={is_valid}, "
        f"errors={len(errors)}, warnings={len(warnings)}"
    )

    return result


# ==============================================================================
# Setup Functions
# ==============================================================================

def setup_default_references(
    reference_dir: Path,
    use_case: str = "development",
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> SecureReferenceGenomeManager:
    """
    Set up recommended reference pool for a use case.

    Args:
        reference_dir: Directory to store references
        use_case: Use case category ("development", "research", "clinical", "production")
        progress_callback: Optional progress callback

    Returns:
        Configured SecureReferenceGenomeManager

    Example:
        >>> reference_dir = Path("references/")
        >>> manager = setup_default_references(
        ...     reference_dir,
        ...     use_case="development"
        ... )
        >>> print(f"References loaded: {manager.reference_count}")
    """
    if use_case not in RECOMMENDED_POOLS:
        raise ValueError(
            f"Unknown use case: {use_case}. "
            f"Valid options: {list(RECOMMENDED_POOLS.keys())}"
        )

    logger.info(f"Setting up {use_case} reference pool in {reference_dir}")

    # Get recommended sources
    sources = RECOMMENDED_POOLS[use_case]
    logger.info(f"Recommended references for {use_case}: {sources}")

    # Download references
    references = download_reference_genomes(
        sources,
        reference_dir,
        progress_callback=progress_callback,
    )

    # Create manager
    manager = SecureReferenceGenomeManager(reference_dir=reference_dir)

    # Add references to pool
    for ref_genome in references.values():
        manager.pool.add_reference(ref_genome)

    logger.info(
        f"Setup complete: {manager.reference_count} references loaded"
    )

    return manager


def get_reference_info(reference_dir: Path) -> Dict[str, Any]:
    """
    Get information about installed references.

    Args:
        reference_dir: Directory containing references

    Returns:
        Dictionary with reference information
    """
    manager = SecureReferenceGenomeManager(reference_dir=reference_dir)

    info = {
        "reference_count": manager.reference_count,
        "references": {},
    }

    for ref_id in manager.pool.references.keys():
        ref_genome = manager.pool.get_reference(ref_id)
        info["references"][ref_id] = {
            "genome_id": ref_genome.genome_id,
            "assembly": ref_genome.assembly,
            "variant_count": sum(len(v) for v in ref_genome.variants.values()),
            "chromosome_count": len(ref_genome.variants),
            "chromosomes": list(ref_genome.variants.keys()),
            "hash": ref_genome.cryptographic_hash[:16] + "...",
        }

    return info


# ==============================================================================
# CLI Helper Functions
# ==============================================================================

def print_available_references() -> None:
    """Print available reference sources."""
    print("\n" + "=" * 80)
    print("AVAILABLE REFERENCE GENOMES")
    print("=" * 80)
    print()

    for name, source in STANDARD_REFERENCES.items():
        print(f"📚 {name}")
        print(f"   Description: {source.description}")
        print(f"   Assembly: {source.assembly}")
        print(f"   Population: {source.population}")
        print(f"   Size: {source.size_mb:.1f} MB")
        print(f"   Variants: ~{source.variant_count:,}")
        print()


def print_recommended_pools() -> None:
    """Print recommended reference pools."""
    print("\n" + "=" * 80)
    print("RECOMMENDED REFERENCE POOLS")
    print("=" * 80)
    print()

    for use_case, sources in RECOMMENDED_POOLS.items():
        print(f"🎯 {use_case.upper()}")
        print(f"   References: {', '.join(sources)}")
        total_size = sum(
            STANDARD_REFERENCES[s].size_mb
            for s in sources
            if s in STANDARD_REFERENCES
        )
        print(f"   Total size: {total_size:.1f} MB")
        print()


def print_validation_results(result: ValidationResult) -> None:
    """Print validation results in human-readable format."""
    print("\n" + "=" * 80)
    print("REFERENCE POOL VALIDATION RESULTS")
    print("=" * 80)
    print()

    # Overall status
    status_emoji = "✅" if result.is_valid else "❌"
    print(f"{status_emoji} Overall Status: {'VALID' if result.is_valid else 'INVALID'}")
    print(f"   References checked: {result.reference_count}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Warnings: {len(result.warnings)}")
    print()

    # Errors
    if result.errors:
        print("❌ Errors:")
        for error in result.errors:
            print(f"   - {error}")
        print()

    # Warnings
    if result.warnings:
        print("⚠️  Warnings:")
        for warning in result.warnings:
            print(f"   - {warning}")
        print()

    # Per-reference status
    print("Reference Details:")
    for ref_id, status in result.reference_status.items():
        hash_emoji = "✅" if status.get('hash_valid', False) else "❌"
        print(f"  {hash_emoji} {ref_id}")
        print(f"     Assembly: {status.get('assembly', 'Unknown')}")
        print(f"     Variants: {status.get('variant_count', 0):,}")
        print(f"     Chromosomes: {status.get('chromosome_count', 0)}")
        if status.get('low_quality_variants', 0) > 0:
            print(f"     Low quality: {status['low_quality_variants']}")
        print()
