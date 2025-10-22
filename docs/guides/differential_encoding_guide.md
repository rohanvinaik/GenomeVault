# Differential Encoding Guide

**Version**: 1.0.0
**Last Updated**: 2025-01-19
**Status**: Production Ready

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Analysis Types](#analysis-types)
4. [Chunking Strategies](#chunking-strategies)
5. [Reference Genome Selection](#reference-genome-selection)
6. [Quick Start](#quick-start)
7. [Advanced Usage](#advanced-usage)
8. [Performance Optimization](#performance-optimization)
9. [Security Considerations](#security-considerations)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### What is Differential Encoding?

Differential encoding is a cryptographically secure method for compressing and storing genomic data by encoding experimental genomes as **differences from reference genomes** rather than storing raw variant data. This approach provides:

- **95%+ compression** compared to raw VCF files
- **Cryptographic verification** using HMAC-SHA256 binding
- **Privacy preservation** through randomized reference selection
- **Fast querying** via hyperdimensional vector similarity
- **Deterministic encoding** with seed-based reproducibility

### How It Works

```
┌─────────────────┐
│ Experimental    │
│ Genome (VCF)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ Chunking        │────▶│ Reference Pool   │
│ (Analysis Type) │     │ (Random Selection)│
└────────┬────────┘     └─────────┬────────┘
         │                        │
         ▼                        ▼
┌─────────────────────────────────────┐
│ Compute Variant Differences         │
│ (New mutations, missing variants,   │
│  genotype changes)                  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Feature Vector Generation           │
│ (384D: position, genotype, quality, │
│  functional impact, composition)    │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Hypervector Encoding                │
│ (10K-100K dimensions, unit norm)    │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Cryptographic Binding + Metadata    │
│ (HMAC, SHA256, chunk IDs)           │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Storage (Compressed, Verified)      │
│ EncodedGenome + Metadata            │
└─────────────────────────────────────┘
```

### Key Benefits

| Feature | Benefit |
|---------|---------|
| **Compression** | 95%+ reduction in storage size |
| **Security** | Cryptographic binding prevents tampering |
| **Privacy** | Randomized reference selection prevents inference |
| **Speed** | Fast similarity-based queries via hypervectors |
| **Flexibility** | Multiple analysis types for different use cases |
| **Reproducibility** | Deterministic encoding with same seed |

---

## Core Concepts

### 1. Chunks

Genomes are divided into **chunks** based on the selected analysis type. Each chunk contains:
- A contiguous genomic region (chromosome, start, end)
- Variants within that region
- A randomly selected reference genome section
- Computed variant differences
- Feature vector (384D) and hypervector (10K-100K D)
- Cryptographic metadata (chunk ID, reference hash, binding)

### 2. Variant Differences

For each chunk, we compute **three types of differences** between experimental and reference:

```python
# New mutations: Present in experimental, absent in reference
new_mutations = experimental_variants - reference_variants

# Missing variants: Present in reference, absent in experimental
missing_variants = reference_variants - experimental_variants

# Genotype differences: Different genotypes for same variant
genotype_diffs = variants_with_different_genotypes
```

### 3. Feature Vectors

Each chunk's differences are encoded into a **384-dimensional feature vector**:

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Difference Types | 10 | Distribution of difference types (new, missing, genotype) |
| Position Encoding | 128 | Sinusoidal encoding of genomic positions |
| Allele Composition | 64 | Nucleotide composition (A, C, G, T) in variants |
| Genotype Distribution | 64 | Distribution of genotypes (0/0, 0/1, 1/1, etc.) |
| Functional Impact | 64 | VEP/SnpEff impact scores (HIGH, MODERATE, LOW, MODIFIER) |
| Quality Metrics | 54 | Variant quality statistics (mean, median, std, percentiles) |

### 4. Hypervector Encoding

Feature vectors are projected into high-dimensional space (10K-100K dimensions) using:
- **Random Gaussian projection** for feature expansion
- **Unit normalization** for consistent similarity metrics
- **Bundling** across chunks for genome-level representation

### 5. Cryptographic Binding

Each chunk is cryptographically bound to its data:

```python
chunk_id = HMAC-SHA256(chunk_data, master_seed)
reference_hash = SHA256(reference_genome_content)
binding = HMAC-SHA256(chunk_data || reference_data, chunk_seed)
```

This ensures:
- **Integrity**: Detects any tampering with chunk data
- **Authenticity**: Verifies chunk originated from valid encoding
- **Traceability**: Links chunk to specific reference genome

---

## Analysis Types

GenomeVault supports **7 analysis types** optimized for different use cases:

### 1. SLIDING_WINDOW

**Best for**: General-purpose analysis, uniform coverage

```python
analysis_type = AnalysisType.SLIDING_WINDOW
```

**Characteristics**:
- Fixed-size windows (default: 1 Mb)
- Overlapping regions (default: 100 kb overlap)
- Uniform chunk distribution
- Good for genome-wide scans

**When to use**:
- Whole-genome sequencing (WGS) analysis
- Uniform coverage required
- No specific biological features to target
- Baseline comparison across genomes

**Parameters**:
- `window_size`: Chunk size (default: 1,000,000 bp)
- `overlap`: Overlap between chunks (default: 100,000 bp)

### 2. GENE_REGION

**Best for**: Functional genomics, exome sequencing

```python
analysis_type = AnalysisType.GENE_REGION
```

**Characteristics**:
- Chunks aligned to gene boundaries
- Variable chunk sizes based on gene length
- Includes regulatory regions (promoters, enhancers)
- Functional annotation preserved

**When to use**:
- Exome or targeted sequencing
- Gene-level differential expression
- Functional variant analysis
- Clinical diagnostics (disease genes)

**Parameters**:
- Requires gene annotation (GTF/GFF3)
- Includes upstream/downstream regions
- Groups small adjacent genes

### 3. VARIANT_DENSITY

**Best for**: Hotspot analysis, somatic mutations

```python
analysis_type = AnalysisType.VARIANT_DENSITY
```

**Characteristics**:
- Adaptive chunk sizes based on variant density
- Small chunks in high-density regions
- Large chunks in low-density regions
- Optimizes information per chunk

**When to use**:
- Cancer genomes (somatic mutations)
- Hypermutable regions
- Variant hotspot analysis
- Regions with variable mutation rates

**Parameters**:
- `min_variants_per_chunk`: Minimum variants (default: 10)
- `max_chunk_size`: Maximum size (default: 5 Mb)
- `density_threshold`: Variants per kb threshold

### 4. FUNCTIONAL_REGIONS

**Best for**: Clinical variants, pathogenic mutations

```python
analysis_type = AnalysisType.FUNCTIONAL_REGIONS
```

**Characteristics**:
- Prioritizes coding regions
- Includes splice sites, UTRs
- Separates intergenic regions
- Functional impact-based chunking

**When to use**:
- Clinical diagnostics
- Pathogenic variant detection
- Pharmacogenomics
- ACMG/AMP variant classification

**Parameters**:
- Requires functional annotation
- Prioritizes high-impact regions
- Separates regulatory vs. coding

### 5. CHROMOSOMAL

**Best for**: Structural variation, CNV analysis

```python
analysis_type = AnalysisType.CHROMOSOMAL
```

**Characteristics**:
- Entire chromosomes as chunks
- Preserves long-range structure
- Large chunk sizes
- Good for structural variants

**When to use**:
- Copy number variation (CNV) analysis
- Chromosomal rearrangements
- Large structural variants
- Cytogenetic analysis

**Parameters**:
- One chunk per chromosome
- Includes telomeres and centromeres
- Preserves chromosome-level features

### 6. CUSTOM_INTERVALS

**Best for**: Targeted regions, custom panels

```python
analysis_type = AnalysisType.CUSTOM_INTERVALS
```

**Characteristics**:
- User-defined genomic intervals
- BED file input
- Flexible chunk boundaries
- Custom region prioritization

**When to use**:
- Targeted gene panels (e.g., cancer panels)
- Custom capture regions
- Specific research regions
- Validation studies

**Parameters**:
- Requires BED file with intervals
- Supports interval names/annotations
- Merges overlapping intervals

### 7. POPULATION_STRATIFIED

**Best for**: Population genetics, ancestry analysis

```python
analysis_type = AnalysisType.POPULATION_STRATIFIED
```

**Characteristics**:
- Chunks based on population-specific variants
- Uses reference genomes from target population
- Optimized for ancestry differences
- Captures population structure

**When to use**:
- Population genetics studies
- Ancestry inference
- Population-specific variant analysis
- Admixture mapping

**Parameters**:
- Requires population-specific references
- Uses population allele frequencies
- Stratifies by ancestry markers

---

## Chunking Strategies

### Strategy Configuration

Each analysis type has a corresponding chunking strategy defined in `STRATEGY_CONFIGS`:

```python
from genomevault.differential_encoding import STRATEGY_CONFIGS, AnalysisType

# Get strategy for analysis type
strategy = STRATEGY_CONFIGS[AnalysisType.SLIDING_WINDOW]

print(f"Window size: {strategy.window_size}")
print(f"Overlap: {strategy.overlap}")
print(f"Features: {strategy.features}")
```

### Custom Strategy

Create custom chunking strategies:

```python
from genomevault.differential_encoding import ChunkingStrategy, GenomicFeature

custom_strategy = ChunkingStrategy(
    name="custom_exome",
    window_size=50000,  # 50 kb windows
    overlap=5000,       # 5 kb overlap
    features=[
        GenomicFeature.CODING,
        GenomicFeature.SPLICE_SITE,
        GenomicFeature.UTR_5,
        GenomicFeature.UTR_3,
    ],
    min_variants=5,
    max_chunk_size=100000,
)

# Use with encoder
encoded = encoder.encode_genome(
    genome=genome,
    analysis_type=AnalysisType.CUSTOM_INTERVALS,
    custom_strategy=custom_strategy,
)
```

### Choosing the Right Strategy

| Use Case | Recommended Analysis Type | Rationale |
|----------|--------------------------|-----------|
| Whole-genome sequencing | SLIDING_WINDOW | Uniform coverage, no bias |
| Exome sequencing | GENE_REGION or FUNCTIONAL_REGIONS | Focuses on coding regions |
| Cancer genomes | VARIANT_DENSITY | Handles mutation hotspots |
| Clinical diagnostics | FUNCTIONAL_REGIONS | Prioritizes pathogenic variants |
| CNV analysis | CHROMOSOMAL | Preserves large-scale structure |
| Gene panels | CUSTOM_INTERVALS | Targets specific genes |
| Population studies | POPULATION_STRATIFIED | Captures ancestry differences |

---

## Reference Genome Selection

### Reference Pool Setup

Differential encoding requires a pool of reference genomes. See [Reference Genome Setup Guide](reference_genome_setup.md) for detailed instructions.

**Quick setup**:

```bash
# Development (synthetic data)
python scripts/genomevault_setup_references.py --use-case development

# Production (gnomAD)
python scripts/genomevault_setup_references.py --use-case production
```

### Selection Strategy

For each chunk, a reference genome is **randomly selected** using:

```python
reference_seed = derive_seed(master_seed || chunk_boundaries)
reference_genome = random_choice(pool, seed=reference_seed)
```

**Key properties**:
- **Deterministic**: Same seed → same reference
- **Unpredictable**: Different chunks → different references (likely)
- **Secure**: Cryptographic seed derivation (HMAC-SHA256)

### Reference Requirements

Good reference genomes should:
1. **Match assembly** (GRCh37 vs. GRCh38)
2. **High quality** (quality scores > 30)
3. **Population diversity** (multiple ancestries)
4. **Large variant sets** (>100K variants)
5. **Well-annotated** (functional consequences)

### Recommended References

| Reference | Assembly | Variants | Use Case |
|-----------|----------|----------|----------|
| gnomAD v4 Exomes | GRCh38 | ~730K | Production, clinical |
| 1000 Genomes Phase 3 | GRCh37 | ~1.1M | Research, population |
| UK Biobank | GRCh38 | ~800K | Large cohorts |
| TOPMed | GRCh38 | ~400K | Cardiovascular |

### Custom References

Load custom reference genomes:

```python
from genomevault.differential_encoding import (
    ReferenceGenome,
    Variant,
    compute_reference_hash,
)

# Create reference from VCF
variants = {
    "chr1": [
        Variant(chromosome="chr1", position=100000, ref="A", alt="G"),
        # ... more variants
    ],
    # ... more chromosomes
}

temp_ref = ReferenceGenome(
    genome_id="custom_ref_001",
    assembly="GRCh38",
    variants=variants,
    cryptographic_hash="temp",
)

# Compute hash
actual_hash = compute_reference_hash(temp_ref)

reference = ReferenceGenome(
    genome_id="custom_ref_001",
    assembly="GRCh38",
    variants=variants,
    cryptographic_hash=actual_hash,
)

# Add to manager
manager.pool.add_reference(reference)
```

---

## Quick Start

### 1. Setup References

```bash
python scripts/genomevault_setup_references.py --use-case development
```

### 2. Encode a Genome

```python
from pathlib import Path
from genomevault.hypervector_transform import UnifiedGenomicEncoder, EncodingMode
from genomevault.differential_encoding import AnalysisType, Genome, Variant

# Create encoder
encoder = UnifiedGenomicEncoder(
    mode=EncodingMode.DIFFERENTIAL,
    reference_dir=Path("references/"),
    dimension=10000,
    seed=42,
)

# Create genome
genome = Genome(
    genome_id="patient_001",
    assembly="GRCh38",
    chromosomes={
        "chr1": [
            Variant(chromosome="chr1", position=100000, ref="A", alt="G", genotype="0/1"),
            # ... more variants
        ]
    }
)

# Encode
encoded = encoder.encode_genome(
    genome=genome,
    analysis_type=AnalysisType.SLIDING_WINDOW,
    bundle_chunks=True,
)

print(f"Encoded {encoded.genome_id}")
print(f"  Chunks: {len(encoded.chunk_hypervectors)}")
print(f"  Storage: {encoded.storage_size_kb():.2f} KB")
print(f"  Verified: {encoded.verify()}")
```

### 3. Save Encoded Genome

```python
# Save with compression
save_path = Path("encoded_genomes/patient_001.enc.gz")
compressed_bytes = encoded.save(save_path, compress=True)

print(f"Saved to {save_path}")
print(f"Compression ratio: {encoded.storage_size_kb() / (compressed_bytes / 1024):.1f}x")
```

### 4. Load and Query

```python
from genomevault.differential_encoding import EncodedGenome, DifferentialGenomeQuery

# Load
loaded = EncodedGenome.load(save_path)

# Create query interface
query = DifferentialGenomeQuery(
    reference_manager=encoder.reference_manager,
    hv_encoder=encoder.differential_encoder.hypervector_encoder,
)

# Query region
result = query.query_region(loaded, "chr1", 50000, 150000)

print(f"Query result:")
print(f"  Variants: {result.variant_count}")
print(f"  Chunks: {result.chunks_used}")
```

---

## Advanced Usage

### Multiple Analysis Types

```python
# Encode with different strategies
for analysis_type in [AnalysisType.SLIDING_WINDOW, AnalysisType.GENE_REGION]:
    encoded = encoder.encode_genome(
        genome=genome,
        analysis_type=analysis_type,
        bundle_chunks=True,
    )

    save_path = Path(f"encoded/{genome.genome_id}_{analysis_type.value}.enc.gz")
    encoded.save(save_path, compress=True)
```

### Batch Processing

```python
from pathlib import Path

# Process multiple genomes
genome_files = Path("vcf_files/").glob("*.vcf")

for vcf_path in genome_files:
    # Load genome from VCF (requires VCF parser)
    genome = load_genome_from_vcf(vcf_path)

    # Encode
    encoded = encoder.encode_genome(genome, AnalysisType.GENE_REGION)

    # Save
    save_path = Path(f"encoded/{genome.genome_id}.enc.gz")
    encoded.save(save_path)
```

### Custom Chunking

```python
from genomevault.differential_encoding import ChunkingStrategy, GenomicFeature

# Define custom strategy
strategy = ChunkingStrategy(
    name="targeted_panel",
    window_size=10000,
    overlap=1000,
    features=[GenomicFeature.CODING, GenomicFeature.SPLICE_SITE],
    min_variants=3,
)

# Encode with custom strategy
encoded = encoder.encode_genome(
    genome=genome,
    analysis_type=AnalysisType.CUSTOM_INTERVALS,
    custom_strategy=strategy,
)
```

---

## Performance Optimization

### 1. Dimension Selection

| Dimension | Speed | Accuracy | Storage | Use Case |
|-----------|-------|----------|---------|----------|
| 1,000 | Fast | 90-95% | Small | Quick screening |
| 10,000 | Medium | 95-98% | Medium | Production default |
| 50,000 | Slow | 98-99% | Large | High-precision |
| 100,000 | Very slow | 99%+ | Very large | Research/validation |

```python
# Optimize for speed
encoder = UnifiedGenomicEncoder(dimension=1000, seed=42)

# Optimize for accuracy
encoder = UnifiedGenomicEncoder(dimension=50000, seed=42)
```

### 2. Compression

Always use compression for storage:

```python
# With compression (recommended)
encoded.save(path, compress=True)  # 2-5x smaller

# Without compression
encoded.save(path, compress=False)
```

### 3. Batch Operations

```python
# Inefficient: Encode chunks one at a time
for chunk in chunks:
    encoded_chunk = encode_single(chunk)

# Efficient: Batch encode
encoded_chunks = encoder.encode_batch(chunks)
```

### 4. Reference Pool Size

| Pool Size | Selection Diversity | Storage | Query Speed |
|-----------|-------------------|---------|-------------|
| 1 reference | Low | Minimal | Fast |
| 10 references | Medium | Small | Fast |
| 100 references | High | Medium | Medium |
| 1000+ references | Very high | Large | Slower |

**Recommendation**: 10-100 references for production

---

## Security Considerations

### Cryptographic Guarantees

Differential encoding provides:

1. **Integrity**: HMAC-SHA256 binding detects tampering
2. **Authenticity**: Chunk IDs verify origin
3. **Unpredictability**: Random reference selection prevents inference
4. **Determinism**: Same seed → same encoding (reproducibility)

### Best Practices

```python
# 1. Use strong seeds
import secrets
master_seed = secrets.token_bytes(32)

# 2. Verify after loading
loaded = EncodedGenome.load(path)
assert loaded.verify(), "Verification failed!"

# 3. Secure reference storage
reference_dir = Path("/secure/references")
reference_dir.chmod(0o700)  # Owner read/write/execute only

# 4. Audit trail
import logging
logging.info(f"Encoded {genome.genome_id} with seed {master_seed.hex()}")
```

### Threat Model

**Protects against**:
- Data tampering (integrity check)
- Unauthorized modification (cryptographic binding)
- Reference inference (randomized selection)
- Replay attacks (unique chunk IDs)

**Does NOT protect against**:
- Physical access to encrypted data
- Compromise of master seed
- Side-channel attacks during encoding

---

## Troubleshooting

### Common Issues

#### 1. Reference Not Found

```
Error: No references available for assembly GRCh38
```

**Solution**:
```bash
python scripts/genomevault_setup_references.py --use-case development
```

#### 2. Verification Failure

```
Error: Cryptographic verification failed
```

**Solution**:
- Check for file corruption
- Verify master seed matches
- Re-encode from source

#### 3. Memory Error

```
MemoryError: Cannot allocate array
```

**Solution**:
- Reduce hypervector dimension
- Use smaller chunk sizes
- Process chromosomes separately

#### 4. Assembly Mismatch

```
Error: Genome assembly GRCh37 does not match reference GRCh38
```

**Solution**:
- Liftover genome to correct assembly
- Download references for GRCh37
- Ensure consistent assemblies

### Performance Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| Slow encoding | Large dimension | Reduce to 10K |
| High memory | Too many chunks | Use larger chunk sizes |
| Poor compression | Small genome | Use more references |
| Slow queries | No indexing | Build query indices |

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Encoding will now show detailed progress
encoded = encoder.encode_genome(genome, AnalysisType.SLIDING_WINDOW)
```

---

## Further Reading

- [API Reference](api_reference_differential.md) - Complete API documentation
- [Reference Genome Setup](reference_genome_setup.md) - Detailed setup guide
- [Architecture Overview](architecture/differential_encoding_architecture.md) - System design
- [Examples](../examples/) - Code examples
  - [Basic Example](../examples/differential_encoding_basic.py)
  - [Advanced Example](../examples/differential_encoding_advanced.py)
- [Migration Guide](migration_differential_encoding.md) - Migrating from legacy encoding

---

## Appendix

### Analysis Type Comparison

| Analysis Type | Chunk Size | Coverage | Best For | Complexity |
|---------------|-----------|----------|----------|------------|
| SLIDING_WINDOW | Fixed (1 Mb) | Uniform | WGS | Low |
| GENE_REGION | Variable | Gene-based | Exomes | Medium |
| VARIANT_DENSITY | Adaptive | Density-based | Hotspots | Medium |
| FUNCTIONAL_REGIONS | Variable | Function-based | Clinical | High |
| CHROMOSOMAL | Whole chr | Chromosome | CNV | Low |
| CUSTOM_INTERVALS | User-defined | Custom | Panels | Low |
| POPULATION_STRATIFIED | Adaptive | Population | Ancestry | High |

### Feature Vector Components

| Component | Formula | Range |
|-----------|---------|-------|
| Position Encoding | sin/cos(pos/10000^(2i/dim)) | [-1, 1] |
| Allele Composition | count(nucleotide) / total | [0, 1] |
| Genotype Distribution | count(genotype) / total | [0, 1] |
| Functional Impact | score(impact_level) | [0, 1] |
| Quality Metrics | normalize(quality_stats) | [0, 1] |

---

**For support, see**: [GenomeVault Documentation](../README.md)
