# FASTQ Integration Guide

## Overview

The GenomeVault Enhanced Differential Encoding Pipeline supports **direct FASTQ input** with automatic region detection and privacy-preserving multi-reference extraction.

This integration bridges the gap between raw sequencing data (FASTQ) and differential encoding by:
1. **Aligning FASTQ reads** to reference genome
2. **Identifying covered genomic regions** from alignment
3. **Extracting those regions from ALL references** (k-anonymity)
4. **Differential encoding** with random reference selection
5. **Hypervector generation** for privacy-preserving storage/query

## Architecture

```
┌─────────────┐
│ FASTQ Input │
│  (R1 + R2)  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ FASTQProcessor      │
│ - Minimap2 align    │
│ - samtools depth    │
│ - bcftools call     │
└──────┬──────────────┘
       │
       ▼
┌──────────────────────┐
│ GenomicRegion        │
│ chr22:10M-10.5M      │
│ coverage=30×         │
└──────┬───────────────┘
       │
       ▼
┌─────────────────────────────┐
│ MultiReferenceExtractor     │
│ Extract chr22:10M-10.5M     │
│ from ALL 3 references:      │
│   ref1: chr22:10M-10.5M     │
│   ref2: chr22:10M-10.5M     │
│   ref3: chr22:10M-10.5M     │
└──────┬──────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ DifferentialGenomicEncoder   │
│ Randomly select 1 of 3 refs  │
│ Encode differences            │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ Hypervectors + Metadata      │
│ k=3 anonymity guarantee      │
│ 264× compression             │
└──────────────────────────────┘
```

## Quick Start

### Installation Requirements

```bash
# Install bioinformatics tools
conda install -c bioconda minimap2 samtools bcftools

# Or via homebrew on macOS
brew install minimap2 samtools bcftools
```

### Basic Usage

```python
from genomevault.differential_encoding.enhanced_pipeline import (
    create_enhanced_pipeline
)
from pathlib import Path

# Setup
reference_genome = Path("data/reference/GRCh38_chr22.fa")
reference_pool = Path("data/references/")  # Contains ref1/, ref2/, ref3/

# Create pipeline
pipeline = create_enhanced_pipeline(
    reference_genome=reference_genome,
    reference_pool_dir=reference_pool,
    dimension=8192,  # Hypervector dimension
)

# Process paired-end FASTQ
result = pipeline.encode_file(
    input_file=Path("sample_r1.fastq.gz"),
    input_file_r2=Path("sample_r2.fastq.gz"),
    output_dir=Path("output/"),
)

# Result contains:
# - result.hypervectors: List of chunk hypervectors
# - result.metadata: Cryptographic metadata for each chunk
# - result.bundled_hypervector: Single genome-level hypervector
# - result.statistics: Encoding statistics
```

## Supported Input Formats

The enhanced pipeline **automatically detects** input format:

| Format | Extension | Use Case | Region Info |
|--------|-----------|----------|-------------|
| **FASTQ** | `.fastq`, `.fq`, `.fastq.gz` | Raw sequencing reads | Auto-detected via alignment |
| **VCF** | `.vcf`, `.vcf.gz` | Pre-called variants | From VCF coordinates |
| **BAM** | `.bam`, `.sam` | Pre-aligned reads | From alignment |

### FASTQ Input (Primary Focus)

```python
# Paired-end FASTQ
result = pipeline.encode_file(
    input_file=Path("sample_r1.fastq.gz"),
    input_file_r2=Path("sample_r2.fastq.gz"),
    output_dir=Path("output/"),
)

# Single-end FASTQ
result = pipeline.encode_file(
    input_file=Path("sample.fastq.gz"),
    output_dir=Path("output/"),
)
```

**Workflow:**
1. Align reads to reference genome (minimap2)
2. Identify covered regions (samtools depth)
3. Call variants (bcftools, optional)
4. Extract regions from all references
5. Differential encode

**Output:**
- `output/aligned.sorted.bam` - Alignment file
- `output/variants.vcf.gz` - Called variants (if enabled)
- Hypervectors + metadata (returned in result)

### VCF Input (Backward Compatible)

```python
# VCF with pre-called variants
result = pipeline.encode_file(
    input_file=Path("sample_variants.vcf.gz")
)
```

**Workflow:**
1. Load variants from VCF
2. Use existing differential encoding pipeline
3. Skip alignment (faster)

**Use when:** You already have variant calls and don't need FASTQ processing.

### BAM Input

```python
# Pre-aligned BAM file
result = pipeline.encode_file(
    input_file=Path("sample.bam"),
    output_dir=Path("output/"),
)
```

**Workflow:**
1. Identify regions from existing alignment
2. Skip re-alignment
3. Extract regions from all references
4. Differential encode

**Use when:** You have existing alignments but need region extraction.

## Privacy Guarantees

### k-Anonymity via Multi-Reference Extraction

The system maintains **k-anonymity** where k = number of reference genomes:

```python
from genomevault.differential_encoding.reference_management import (
    SecureReferenceGenomeManager
)
from genomevault.differential_encoding.region_extractor import (
    MultiReferenceExtractor
)

# Load reference pool (3 genomes)
ref_manager = SecureReferenceGenomeManager(Path("data/references/"))
# ref_manager.reference_count = 3

# Extract same region from ALL references
extractor = MultiReferenceExtractor(ref_manager)
multi_ref_region = extractor.extract_region(genomic_region)

# multi_ref_region.num_references = 3
# multi_ref_region.reference_sections = {
#     "ref1": GenomeSection(chr22:10M-10.5M),
#     "ref2": GenomeSection(chr22:10M-10.5M),
#     "ref3": GenomeSection(chr22:10M-10.5M),
# }

# Differential encoder randomly selects 1 of 3
# Attacker cannot determine which was used: 1/3 probability
# All references have SAME region → perfect anonymity set
```

**Key Properties:**
- ✅ Same genomic region extracted from ALL references
- ✅ Random selection during encoding (cryptographically secure)
- ✅ No information leaked about which reference was used
- ✅ k=3 standard, configurable to any k≥2

## Complete Examples

### Example 1: Targeted Sequencing (BRCA1 Gene)

```python
from genomevault.differential_encoding.enhanced_pipeline import (
    create_enhanced_pipeline
)
from pathlib import Path

# BRCA1 targeted sequencing panel
# Only chr17:43044295-43125483 sequenced

pipeline = create_enhanced_pipeline(
    reference_genome=Path("data/reference/GRCh38_chr17.fa"),
    reference_pool_dir=Path("data/references/"),
    dimension=8192,
)

result = pipeline.encode_file(
    input_file=Path("BRCA1_patient_r1.fastq.gz"),
    input_file_r2=Path("BRCA1_patient_r2.fastq.gz"),
    output_dir=Path("output/brca1_encoding/"),
)

# System will:
# 1. Align reads → identify chr17:43044295-43125483
# 2. Extract chr17:43044295-43125483 from all 3 references
# 3. Encode with random reference
# 4. Return hypervectors

print(f"Encoded {len(result.hypervectors)} chunks")
print(f"Region: {result.metadata[0].chromosome}:{result.metadata[0].start_position}")
```

### Example 2: Whole Genome Sequencing

```python
# WGS with 30× coverage
# Multiple chromosomes sequenced

pipeline = create_enhanced_pipeline(
    reference_genome=Path("data/reference/GRCh38.fa"),  # Full genome
    reference_pool_dir=Path("data/references/"),
    dimension=8192,
)

result = pipeline.encode_file(
    input_file=Path("WGS_sample_r1.fastq.gz"),
    input_file_r2=Path("WGS_sample_r2.fastq.gz"),
    output_dir=Path("output/wgs_encoding/"),
)

# System will:
# 1. Align reads across all chromosomes
# 2. Identify multiple covered regions (chr1, chr2, ..., chr22, chrX, chrY)
# 3. Extract ALL regions from ALL references
# 4. Encode each region independently
# 5. Bundle into single genome hypervector

print(f"Chromosomes: {result.statistics['chromosomes']}")
print(f"Total chunks: {len(result.hypervectors)}")
print(f"Bundled: {len(result.bundled_hypervector)} D")
```

### Example 3: Unknown Sample (Identify Then Encode)

```python
# Sample with unknown genomic content
# Need to identify what regions are present

pipeline = create_enhanced_pipeline(
    reference_genome=Path("data/reference/GRCh38.fa"),
    reference_pool_dir=Path("data/references/"),
    dimension=8192,
)

# Process FASTQ → auto-detect regions
result = pipeline.encode_file(
    input_file=Path("unknown_sample.fastq.gz"),
    output_dir=Path("output/unknown_encoding/"),
)

# Inspect what was found
for meta in result.metadata:
    print(f"Found region: {meta.chromosome}:{meta.start_position}-{meta.end_position}")
    print(f"  Coverage: {meta.difference_counts['total']} differences")
    print(f"  Reference used: {meta.reference_genome_id} (k=3 anonymity)")
```

## Component Details

### 1. FASTQProcessor

**Location:** `genomevault/differential_encoding/fastq_processor.py`

**Purpose:** Process FASTQ files and identify genomic regions.

**Key Methods:**
```python
from genomevault.differential_encoding.fastq_processor import (
    FASTQProcessor, create_default_processor
)

# Create processor
processor = FASTQProcessor(
    reference_genome=Path("ref.fa"),
    aligner="minimap2",       # or "bwa"
    min_coverage=5.0,         # Minimum 5× coverage
    min_confidence=0.7,       # 70% confidence threshold
    threads=4,
)

# Or use defaults
processor = create_default_processor(Path("ref.fa"))

# Process FASTQ
alignment_result = processor.process_fastq(
    fastq_r1=Path("r1.fastq.gz"),
    fastq_r2=Path("r2.fastq.gz"),  # Optional
    output_dir=Path("output/"),
)

# alignment_result contains:
# - regions: List[GenomicRegion]
# - alignment_file: Path to BAM
# - vcf_file: Optional Path to VCF
# - stats: Alignment statistics
```

**Configuration:**
- `aligner`: "minimap2" (fast) or "bwa" (accurate)
- `min_coverage`: Minimum depth to consider a region (default 5×)
- `min_confidence`: Confidence threshold 0.0-1.0 (default 0.7)
- `threads`: Parallel threads for alignment (default 4)

### 2. MultiReferenceExtractor

**Location:** `genomevault/differential_encoding/region_extractor.py`

**Purpose:** Extract same region from all references (k-anonymity).

**Key Methods:**
```python
from genomevault.differential_encoding.region_extractor import (
    MultiReferenceExtractor
)
from genomevault.differential_encoding.reference_management import (
    SecureReferenceGenomeManager
)

# Load references
ref_manager = SecureReferenceGenomeManager(Path("data/references/"))

# Create extractor
extractor = MultiReferenceExtractor(ref_manager)

# Extract region from all references
multi_ref_region = extractor.extract_region(
    region=genomic_region,
    reference_ids=None,  # None = all references
)

# multi_ref_region contains:
# - chromosome, start, end: Region coordinates
# - reference_sections: Dict[ref_id → GenomeSection]
# - reference_sequences: Dict[ref_id → sequence string]
# - num_references: Count of references
```

**Privacy:**
- Extracts IDENTICAL coordinates from each reference
- Ensures all references have same region → perfect anonymity set
- Random selection happens during encoding (not extraction)

### 3. EnhancedDifferentialEncodingPipeline

**Location:** `genomevault/differential_encoding/enhanced_pipeline.py`

**Purpose:** Complete pipeline with automatic format detection.

**Key Methods:**
```python
from genomevault.differential_encoding.enhanced_pipeline import (
    EnhancedDifferentialEncodingPipeline,
    create_enhanced_pipeline,
)

# Create pipeline (manual)
pipeline = EnhancedDifferentialEncodingPipeline(
    reference_genome=Path("ref.fa"),
    reference_manager=ref_manager,
    dimension=8192,
    enable_fastq=True,
)

# Or use factory (recommended)
pipeline = create_enhanced_pipeline(
    reference_genome=Path("ref.fa"),
    reference_pool_dir=Path("data/references/"),
    dimension=8192,
)

# Encode any format (auto-detect)
result = pipeline.encode_file(
    input_file=Path("sample.fastq.gz"),  # or .vcf.gz, .bam
    input_file_r2=None,  # Optional for paired-end FASTQ
    output_dir=None,  # Optional output directory
)
```

**Format Detection:**
- `.fastq`, `.fq`, `.fastq.gz`: FASTQ processing path
- `.vcf`, `.vcf.gz`: VCF processing path
- `.bam`, `.sam`: BAM processing path
- Automatic based on file extension

## Performance Characteristics

### FASTQ Processing

| Step | Time (30× coverage, chr22) | Tool |
|------|----------------------------|------|
| Alignment | 2-5 minutes | minimap2 |
| Region detection | 10-30 seconds | samtools depth |
| Variant calling | 1-3 minutes | bcftools |
| Region extraction | 5-15 seconds | pysam |
| Differential encoding | 30-90 seconds | GenomeVault |
| **Total** | **~5-10 minutes** | |

**Scaling:**
- Linear with read count: 2× reads = 2× time
- Linear with coverage: 60× = ~2× time vs 30×
- Parallelizable: Use more threads for alignment
- Dominant: Alignment and variant calling (90% of time)

### Compression Ratios

From experimental benchmarks (see `benchmark_results/differential_encoding/`):

- **Differential encoding**: 11× compression (FASTQ → differences)
- **Hypervector encoding**: 24× compression (differences → HDC)
- **Total**: 264× compression (11× × 24×)

**Example:**
- Input FASTQ: 2.4 GB (10M reads, paired-end)
- After differential: 218 MB
- After hypervectors: 9.1 MB
- **Compression: 264×**

**Quality:**
- No information loss for variant positions
- Perfect reconstruction of differences
- Semantic similarity preserved in hypervectors

## Troubleshooting

### Common Issues

#### 1. `minimap2: command not found`

**Solution:**
```bash
conda install -c bioconda minimap2
# or
brew install minimap2
```

#### 2. `RuntimeError: FASTQ processing not available`

**Cause:** Alignment tools not installed.

**Solution:**
```bash
conda install -c bioconda minimap2 samtools bcftools
```

Verify:
```bash
minimap2 --version
samtools --version
bcftools --version
```

#### 3. `No genomic regions detected from FASTQ input`

**Cause:** Low coverage or alignment failed.

**Solutions:**
- Check FASTQ quality: `zcat sample.fastq.gz | head -8`
- Verify reference genome: `head data/reference/ref.fa`
- Lower min_coverage: `FASTQProcessor(min_coverage=1.0, ...)`
- Check alignment output: `samtools flagstat output/aligned.sorted.bam`

#### 4. `ValueError: Failed to extract region from any reference`

**Cause:** Reference pool doesn't have the identified chromosome.

**Solutions:**
- Ensure reference pool has same chromosomes as reference genome
- Check chromosome naming: "chr22" vs "22"
- Verify reference pool structure:
  ```bash
  ls data/references/
  # Should show: ref1/ ref2/ ref3/ ...
  ```

#### 5. Slow alignment performance

**Solutions:**
- Increase threads: `FASTQProcessor(threads=8, ...)`
- Use minimap2 instead of BWA (faster for long reads)
- Subsample reads for testing:
  ```bash
  seqtk sample -s100 sample.fastq.gz 100000 > subset.fastq
  ```

## Reference Pool Requirements

### Minimum Requirements

- **Number of references**: ≥2 (recommended ≥3 for k=3 anonymity)
- **Same assembly**: All references must use same assembly (e.g., GRCh38)
- **Same chromosomes**: References should cover same chromosomes
- **Variant diversity**: References should have realistic variant differences

### Generating Reference Pool

Use the automated script:

```bash
./benchmarks/generate_reference_pool.sh
```

This creates:
- 3 reference genomes (ref1, ref2, ref3)
- 1 query genome (for testing)
- Each with 10K SNPs, 2K indels, 20 CNVs, 3 inversions
- Realistic 30× FASTQ coverage

**Location:** `benchmark_results/differential_encoding_samples/`

**Structure:**
```
differential_encoding_samples/
├── references/
│   ├── ref1/
│   │   ├── reference.fa
│   │   ├── variants.vcf
│   │   └── neat_sim_r1.fastq.gz, neat_sim_r2.fastq.gz
│   ├── ref2/
│   └── ref3/
└── query/
    └── (same structure)
```

**Usage:**
```python
ref_manager = SecureReferenceGenomeManager(
    Path("benchmark_results/differential_encoding_samples/references")
)
```

## Advanced Topics

### Custom Alignment Parameters

```python
from genomevault.differential_encoding.fastq_processor import FASTQProcessor

processor = FASTQProcessor(
    reference_genome=Path("ref.fa"),
    aligner="minimap2",
    min_coverage=10.0,  # Higher coverage threshold
    min_confidence=0.9,  # Higher confidence
    threads=8,  # More parallelism
)

# Custom processing
alignment_result = processor.process_fastq(
    fastq_r1=Path("high_quality_r1.fastq.gz"),
    fastq_r2=Path("high_quality_r2.fastq.gz"),
    output_dir=Path("output/custom/"),
)
```

### Multiple Region Extraction

```python
from genomevault.differential_encoding.region_extractor import (
    MultiReferenceExtractor
)

extractor = MultiReferenceExtractor(ref_manager)

# Extract multiple regions at once
regions = [region1, region2, region3]
multi_ref_regions = extractor.extract_multiple_regions(regions)

# Process each region
for multi_ref_region in multi_ref_regions:
    print(f"Region: {multi_ref_region.chromosome}:{multi_ref_region.start}")
    print(f"References: {multi_ref_region.num_references}")
```

### Custom Chunking Strategy

```python
from genomevault.differential_encoding.chunking import AnalysisType

# Different analysis types use different chunking strategies
result = encoder.encode_experimental_genome(
    experimental_genome=genome,
    analysis_type=AnalysisType.GENE_REGION,  # Gene-based chunks
    # or AnalysisType.SLIDING_WINDOW,  # Fixed-size windows
    # or AnalysisType.WHOLE_CHROMOSOME,  # Chromosome-level
    bundle_chunks=True,
)
```

## See Also

- **Examples:** `examples/fastq_to_differential_encoding_example.py`
- **API Reference:** `docs/api_reference_differential.md`
- **Architecture:** `docs/architecture/differential_encoding_architecture.md`
- **Sequence Alignment:** `docs/ALIGNMENT_README.md`
- **Reference Setup:** `docs/reference_genome_setup.md`

## Support

For issues or questions:
- Open an issue on GitHub
- Refer to `CLAUDE.md` for project navigation
- Check `docs/differential_encoding_guide.md` for detailed documentation
