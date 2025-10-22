# Reference Genome Setup Guide

**Last Updated**: 2025-10-19
**Status**: Production Ready

## Overview

This guide explains how to download, validate, and manage reference genome pools for differential encoding in GenomeVault. Reference genomes are essential for the differential encoding system, which stores experimental genomes as cryptographically verified differences from randomly selected reference sections.

## Quick Start

### Interactive Setup Wizard

The easiest way to get started is using the interactive setup wizard:

```bash
python scripts/genomevault_setup_references.py
```

This will guide you through:
1. Choosing a reference directory
2. Selecting a use case (development, research, clinical, production)
3. Downloading references with progress bars
4. Validating integrity

### Quick Setup for Development

```bash
python scripts/genomevault_setup_references.py --use-case development
```

This downloads synthetic test data (~0.1 MB) for immediate testing.

### Quick Setup for Production

```bash
python scripts/genomevault_setup_references.py --use-case production
```

This downloads gnomAD v4 exomes (~15 GB) for production use.

## Reference Sources

### Standard References

GenomeVault provides several curated reference sources:

| Reference | Description | Assembly | Size | Variants |
|-----------|-------------|----------|------|----------|
| `synthetic_test` | Synthetic test data | GRCh38 | 0.1 MB | ~100 |
| `1000g_eur_chr22` | 1000 Genomes EUR chr22 | GRCh37 | 450 MB | ~1.1M |
| `gnomad_exomes_v4` | gnomAD v4 Exomes | GRCh38 | 15 GB | ~730K |

### Recommended Pools

Different use cases require different reference pools:

**Development** (`development`):
- References: `synthetic_test`
- Size: ~0.1 MB
- Use case: Quick testing, CI/CD, development

**Research** (`research`):
- References: `1000g_eur_chr22`
- Size: ~450 MB
- Use case: Population genetics research, ancestry studies

**Clinical** (`clinical`):
- References: `gnomad_exomes_v4`, `1000g_eur_chr22`
- Size: ~15.5 GB
- Use case: Clinical diagnostics, variant interpretation

**Production** (`production`):
- References: `gnomad_exomes_v4`
- Size: ~15 GB
- Use case: Production deployments, large-scale analysis

## Installation Methods

### Method 1: Interactive Wizard

```bash
python scripts/genomevault_setup_references.py
```

Follow the prompts to:
1. Choose reference directory
2. Select use case or custom references
3. Confirm download
4. Validate integrity

### Method 2: Command-Line Arguments

```bash
# Development setup
python scripts/genomevault_setup_references.py --use-case development

# Research setup
python scripts/genomevault_setup_references.py --use-case research

# Clinical setup
python scripts/genomevault_setup_references.py --use-case clinical

# Production setup
python scripts/genomevault_setup_references.py --use-case production

# Custom references
python scripts/genomevault_setup_references.py --custom synthetic_test 1000g_eur_chr22

# Custom directory
python scripts/genomevault_setup_references.py \
    --ref-dir /data/genomevault/references \
    --use-case production
```

### Method 3: Programmatic Setup

```python
from pathlib import Path
from genomevault.differential_encoding import (
    setup_default_references,
    download_reference_genomes,
    validate_reference_pool,
)

# Setup recommended references for a use case
reference_dir = Path("references/")
manager = setup_default_references(
    reference_dir,
    use_case="development",
)

print(f"Loaded {manager.reference_count} references")

# Or download specific references
references = download_reference_genomes(
    sources=["synthetic_test", "1000g_eur_chr22"],
    output_dir=reference_dir,
)

# Validate
result = validate_reference_pool(manager)
print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
```

## Validation

### Automatic Validation

Validation happens automatically after download. It checks:
- ✅ Cryptographic hash integrity (SHA-256)
- ✅ Variant data consistency
- ✅ Assembly compatibility
- ✅ Quality thresholds

### Manual Validation

```bash
# Validate existing references
python scripts/genomevault_setup_references.py --validate

# Validate specific directory
python scripts/genomevault_setup_references.py \
    --ref-dir /data/genomevault/references \
    --validate
```

### Programmatic Validation

```python
from pathlib import Path
from genomevault.differential_encoding import (
    SecureReferenceGenomeManager,
    validate_reference_pool,
)

# Load references
manager = SecureReferenceGenomeManager(Path("references/"))

# Validate
result = validate_reference_pool(manager)

if result.is_valid:
    print("✅ All references valid")
else:
    print(f"❌ Validation failed with {len(result.errors)} errors:")
    for error in result.errors:
        print(f"  - {error}")

# Check per-reference status
for ref_id, status in result.reference_status.items():
    print(f"{ref_id}:")
    print(f"  Hash valid: {status['hash_valid']}")
    print(f"  Variants: {status['variant_count']:,}")
    print(f"  Chromosomes: {status['chromosome_count']}")
```

## Management

### List Installed References

```bash
python scripts/genomevault_setup_references.py --list
```

Output:
```
INSTALLED REFERENCES
================================================================================

Reference directory: /Users/user/.genomevault/references

Total references: 1

📚 synthetic_test
   Assembly: GRCh38
   Variants: 99
   Chromosomes: chr1, chr2, chr22
   Hash: 1a2b3c4d5e6f7g8h...
```

### Get Reference Info

```python
from pathlib import Path
from genomevault.differential_encoding import get_reference_info

info = get_reference_info(Path("references/"))

print(f"Total references: {info['reference_count']}")
for ref_id, ref_info in info["references"].items():
    print(f"\n{ref_id}:")
    print(f"  Assembly: {ref_info['assembly']}")
    print(f"  Variants: {ref_info['variant_count']:,}")
    print(f"  Chromosomes: {', '.join(ref_info['chromosomes'])}")
```

### Remove References

```bash
# Remove all references
rm -rf ~/.genomevault/references

# Remove specific reference
rm ~/.genomevault/references/synthetic_test.vcf
```

### Update References

```bash
# Re-download with force flag
python scripts/genomevault_setup_references.py \
    --use-case development \
    --force
```

## Usage with Differential Encoding

Once references are set up, use them with differential encoding:

```python
from pathlib import Path
from genomevault.hypervector_transform import UnifiedGenomicEncoder, EncodingMode
from genomevault.differential_encoding import AnalysisType, Genome, Variant

# Create encoder with reference directory
encoder = UnifiedGenomicEncoder(
    mode=EncodingMode.DIFFERENTIAL,
    reference_dir=Path("references/"),
    dimension=10000,
)

print(f"Encoder initialized with {encoder.reference_manager.reference_count} references")

# Encode a genome
genome = Genome(
    genome_id="patient_001",
    assembly="GRCh38",
    chromosomes={
        "chr1": [
            Variant(chromosome="chr1", position=100000, ref="A", alt="G"),
            # ... more variants
        ]
    }
)

encoded = encoder.encode_genome(
    genome=genome,
    analysis_type=AnalysisType.SLIDING_WINDOW,
    bundle_chunks=True,
)

print(f"Encoded {encoded.genome_id}: {len(encoded.chunk_hypervectors)} chunks")
```

## Advanced Topics

### Custom Reference Sources

Add your own reference sources:

```python
from genomevault.differential_encoding import ReferenceSource, STANDARD_REFERENCES

# Define custom source
custom_source = ReferenceSource(
    name="custom_reference",
    description="Custom population reference",
    url="https://example.com/reference.vcf.gz",
    assembly="GRCh38",
    population="CUSTOM",
    size_mb=500.0,
    variant_count=1000000,
    checksum="abc123...",
)

# Add to standard references
STANDARD_REFERENCES["custom_reference"] = custom_source

# Download
from genomevault.differential_encoding import download_reference_genomes
references = download_reference_genomes(
    ["custom_reference"],
    Path("references/"),
)
```

### VCF Parsing

For production use with real VCF files, integrate with `cyvcf2` or `pysam`:

```python
from cyvcf2 import VCF
from genomevault.differential_encoding import Variant, ReferenceGenome
from genomevault.differential_encoding.crypto_primitives import compute_reference_hash

def load_vcf_as_reference(vcf_path: str, genome_id: str, assembly: str) -> ReferenceGenome:
    """Load VCF file as ReferenceGenome."""
    variants = {}

    vcf = VCF(vcf_path)
    for variant in vcf:
        chrom = variant.CHROM
        if chrom not in variants:
            variants[chrom] = []

        variants[chrom].append(Variant(
            chromosome=chrom,
            position=variant.POS,
            ref=variant.REF,
            alt=variant.ALT[0] if variant.ALT else ".",
            genotype=variant.gt_types[0] if variant.gt_types else None,
            quality=variant.QUAL,
            info=dict(variant.INFO),
        ))

    # Create reference with hash
    temp_ref = ReferenceGenome(
        genome_id=genome_id,
        assembly=assembly,
        variants=variants,
        cryptographic_hash="temp",
    )

    actual_hash = compute_reference_hash(temp_ref)

    return ReferenceGenome(
        genome_id=genome_id,
        assembly=assembly,
        variants=variants,
        cryptographic_hash=actual_hash,
    )
```

### Distributed Storage

For large reference pools, use distributed storage:

```python
from pathlib import Path
from genomevault.differential_encoding import SecureReferenceGenomeManager

# Option 1: Network file system
reference_dir = Path("/mnt/shared/genomevault/references")

# Option 2: Object storage (with local cache)
reference_dir = Path("/var/cache/genomevault/references")
# Sync from S3/GCS periodically

# Create manager
manager = SecureReferenceGenomeManager(reference_dir=reference_dir)
```

### Reference Pool Optimization

For large-scale deployments:

```python
from genomevault.differential_encoding import SecureReferenceGenomeManager
from pathlib import Path

# Load references
manager = SecureReferenceGenomeManager(Path("references/"))

# Get statistics
print(f"Total references: {manager.reference_count}")
print(f"Available assemblies: {manager.pool.get_available_assemblies()}")

# Pre-load references for specific assemblies
# (Reduces startup time for encoding operations)
for ref in manager.pool.references.values():
    if ref.assembly == "GRCh38":
        # Pre-compute indices, cache data, etc.
        pass
```

## Troubleshooting

### Issue: Download Fails

**Symptom**: Download fails with network error

**Solution**:
```bash
# Check internet connection
ping -c 3 ftp.1000genomes.ebi.ac.uk

# Try with verbose logging
python scripts/genomevault_setup_references.py \
    --use-case development \
    --verbose

# Use alternative source or mirror
```

### Issue: Validation Fails

**Symptom**: Hash validation fails after download

**Solution**:
```bash
# Re-download with force flag
python scripts/genomevault_setup_references.py \
    --use-case development \
    --force

# Manually verify checksum
sha256sum ~/.genomevault/references/synthetic_test.vcf
```

### Issue: Out of Disk Space

**Symptom**: Download fails due to insufficient disk space

**Solution**:
```bash
# Check disk space
df -h ~/.genomevault/references

# Use alternative directory with more space
python scripts/genomevault_setup_references.py \
    --ref-dir /data/genomevault/references \
    --use-case development

# Clean up old references
rm -rf ~/.genomevault/references/old_*
```

### Issue: Slow Downloads

**Symptom**: Downloads are very slow

**Solution**:
```bash
# Use development setup for testing (fastest)
python scripts/genomevault_setup_references.py --use-case development

# Download in background
nohup python scripts/genomevault_setup_references.py \
    --use-case production &

# Use parallel downloads (if supported)
# Multiple references download in parallel automatically
```

## Best Practices

### Development

```bash
# Use synthetic data for fast iteration
python scripts/genomevault_setup_references.py --use-case development

# Validate after changes
python scripts/genomevault_setup_references.py --validate
```

### Production

```bash
# Use production-grade references
python scripts/genomevault_setup_references.py \
    --ref-dir /opt/genomevault/references \
    --use-case production

# Set up monitoring
# Monitor disk usage: du -sh /opt/genomevault/references
# Monitor reference count: python scripts/genomevault_setup_references.py --list

# Regular validation
crontab -e
# Add: 0 2 * * 0 python /path/to/genomevault_setup_references.py --validate
```

### Continuous Integration

```yaml
# .github/workflows/test.yml
- name: Setup references
  run: |
    python scripts/genomevault_setup_references.py --use-case development
    python scripts/genomevault_setup_references.py --validate
```

## API Reference

### Functions

#### `download_reference_genomes()`

```python
def download_reference_genomes(
    sources: List[str],
    output_dir: Path,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    force: bool = False,
) -> Dict[str, ReferenceGenome]:
    """Download and format reference genomes."""
```

#### `validate_reference_pool()`

```python
def validate_reference_pool(
    reference_manager: SecureReferenceGenomeManager,
) -> ValidationResult:
    """Validate integrity of reference genome pool."""
```

#### `setup_default_references()`

```python
def setup_default_references(
    reference_dir: Path,
    use_case: str = "development",
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> SecureReferenceGenomeManager:
    """Set up recommended reference pool for a use case."""
```

#### `get_reference_info()`

```python
def get_reference_info(reference_dir: Path) -> Dict[str, Any]:
    """Get information about installed references."""
```

### Data Classes

#### `ReferenceSource`

```python
@dataclass
class ReferenceSource:
    name: str
    description: str
    url: str
    assembly: str
    population: str
    size_mb: float
    variant_count: int
    checksum: Optional[str] = None
```

#### `ValidationResult`

```python
@dataclass
class ValidationResult:
    is_valid: bool
    reference_count: int
    errors: List[str]
    warnings: List[str]
    reference_status: Dict[str, Dict[str, Any]]
```

## Support

- **Examples**: `examples/reference_setup_demo.py`
- **Tests**: `tests/differential_encoding/test_reference_setup.py`
- **API Docs**: `genomevault/differential_encoding/reference_setup.py`

## See Also

- [Differential Encoding Guide](migration_differential_encoding.md)
- [Unified Encoding Interface](DIFFERENTIAL_ENCODING_REFACTOR_SUMMARY.md)
- [API Reference](../genomevault/differential_encoding/README.md)
