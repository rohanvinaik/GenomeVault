# Sequence Alignment System - Integration Guide

## Overview

The GenomeVault Sequence Alignment System provides low-compute, fuzzy-tolerant alignment for identifying which reference genome(s) best match an input sequence. Unlike traditional read aligners (BWA, Minimap2), this system is optimized for variant-based genomic data (VCF format) and supports multi-reference consensus.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Query Variants                     │
│                     (VCF format data)                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              K-mer Index (Pre-screening)                     │
│  • Fast candidate selection                                  │
│  • Hash-based lookup                                         │
│  • O(n) complexity where n = # variants                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            Variant-Level Alignment                           │
│  • Detailed scoring (SNPs, indels, genotypes)                │
│  • Fuzzy position matching                                   │
│  • Weighted scoring by variant type                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│          Multi-Reference Consensus                           │
│  • Compare against N references (default N=3)                │
│  • Majority voting                                           │
│  • Ambiguity detection                                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            Output: Alignment Result                          │
│  • Primary reference ID                                      │
│  • Confidence score                                          │
│  • Detailed alignment metrics                                │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from pathlib import Path
from genomevault.differential_encoding.reference_management import (
    SecureReferenceGenomeManager,
    GenomeSection,
    Variant,
)
from genomevault.differential_encoding.sequence_alignment import (
    create_default_aligner,
    AlignmentStrategy,
)

# 1. Initialize reference manager with your reference genomes
reference_dir = Path("/path/to/references")
ref_manager = SecureReferenceGenomeManager(reference_dir)

# 2. Create aligner with default settings
aligner = create_default_aligner(
    reference_manager=ref_manager,
    strategy=AlignmentStrategy.HYBRID,  # K-mer + variant scoring
    num_references=3,  # Use 3 references for consensus
)

# 3. Create query section from your variants
query_variants = [
    Variant(
        chromosome="chr1",
        position=12345,
        ref="A",
        alt="G",
        genotype="0/1",
    ),
    # ... more variants
]

query_section = GenomeSection(
    chromosome="chr1",
    start_position=0,
    end_position=1000000,
    variants=query_variants,
)

# 4. Align and get results
result = aligner.align(query_section)

# 5. Use results
print(f"Best match: {result.primary_reference}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Ambiguous: {result.ambiguous}")

if not result.ambiguous:
    # High confidence - use primary reference
    reference_id = result.primary_reference
else:
    # Ambiguous - may need manual review
    print(f"Secondary matches: {result.secondary_references}")
```

## Alignment Strategies

### 1. K-mer Only (Fastest)

Use for initial screening or when speed is critical:

```python
aligner = create_default_aligner(
    reference_manager=ref_manager,
    strategy=AlignmentStrategy.KMER_ONLY,
)

result = aligner.align(query_section, fast_mode=True)
```

**Pros:**
- Very fast (O(n) where n = # variants)
- Low memory usage
- Good for large-scale screening

**Cons:**
- Lower accuracy
- No detailed variant information
- Higher false positive rate

**Use when:**
- Processing thousands of samples
- Initial filtering step
- Low variant density

### 2. Variant Scoring (Most Accurate)

Use for detailed analysis:

```python
aligner = create_default_aligner(
    reference_manager=ref_manager,
    strategy=AlignmentStrategy.VARIANT_SCORING,
)

result = aligner.align(query_section)
```

**Pros:**
- High accuracy
- Detailed metrics (SNPs, indels, genotypes)
- Fuzzy matching for indels

**Cons:**
- Slower than k-mer only
- Higher memory for dense variants

**Use when:**
- Final assignment needed
- Quality control critical
- Detailed reporting required

### 3. Hybrid (Recommended)

Combines k-mer pre-screening with variant scoring:

```python
aligner = create_default_aligner(
    reference_manager=ref_manager,
    strategy=AlignmentStrategy.HYBRID,  # Default
)

result = aligner.align(query_section)
```

**Pros:**
- Balance of speed and accuracy
- Fast candidate selection
- Detailed scoring for top matches

**Cons:**
- Slightly more complex

**Use when:**
- General purpose alignment
- Production workflows
- Need both speed and accuracy

### 4. Consensus (Most Robust)

Multi-reference consensus with voting:

```python
aligner = create_default_aligner(
    reference_manager=ref_manager,
    strategy=AlignmentStrategy.CONSENSUS,
    num_references=5,  # Use more references
    consensus_threshold=0.7,  # 70% agreement required
)

result = aligner.align(query_section)
```

**Pros:**
- Handles admixture/hybrid genomes
- Detects ambiguous cases
- Most robust to errors

**Cons:**
- Slowest option
- Requires multiple references

**Use when:**
- Admixed populations
- Reference genome unclear
- Quality control critical

## Integration with Differential Encoding Pipeline

### Option 1: Pre-alignment Before Encoding

Identify reference before encoding:

```python
from genomevault.differential_encoding.pipeline import DifferentialGenomicEncoder
from genomevault.differential_encoding.chunking import AnalysisType, Genome

# 1. Align query to identify best reference
aligner = create_default_aligner(ref_manager)
query_section = # ... create from your VCF data
result = aligner.align(query_section)

# 2. Create Genome object with identified reference
experimental_genome = Genome(
    genome_id="SAMPLE001",
    variants_by_chromosome={
        "chr1": query_section.variants,
        # ... other chromosomes
    },
    metadata={
        "reference_genome": result.primary_reference,
        "alignment_confidence": result.confidence,
    }
)

# 3. Encode using standard pipeline
encoder = DifferentialGenomicEncoder(
    reference_manager=ref_manager,
    hypervector_encoder=hypervector_encoder,
)

encoding_result = encoder.encode_experimental_genome(
    experimental_genome=experimental_genome,
    analysis_type=AnalysisType.SLIDING_WINDOW,
)
```

### Option 2: Chunk-Level Alignment

Align each chunk independently:

```python
from genomevault.differential_encoding.chunking import CryptographicChunker

# 1. Chunk genome
chunker = CryptographicChunker(strategy=strategy)
chunks = chunker.chunk_genome(experimental_genome, AnalysisType.SLIDING_WINDOW)

# 2. Align each chunk
chunk_alignments = {}
for chunk in chunks:
    chunk_section = GenomeSection(
        chromosome=chunk.chromosome,
        start_position=chunk.start_position,
        end_position=chunk.end_position,
        variants=chunk.variants,
    )
    
    result = aligner.align(chunk_section)
    chunk_alignments[chunk.chunk_id] = result

# 3. Determine overall best reference by majority vote
overall_reference = aligner.majority_vote(chunk_alignments)

# 4. Encode with determined reference
# ...
```

### Option 3: Automatic Alignment in Pipeline

Modify the pipeline to auto-detect references:

```python
class AlignmentAwareDifferentialEncoder(DifferentialGenomicEncoder):
    """Extended encoder with automatic reference alignment."""
    
    def __init__(self, *args, alignment_strategy=AlignmentStrategy.HYBRID, **kwargs):
        super().__init__(*args, **kwargs)
        self.aligner = create_default_aligner(
            reference_manager=self.reference_manager,
            strategy=alignment_strategy,
        )
    
    def _encode_chunk(self, chunk, experimental_genome, analysis_type, master_seed):
        """Override to include alignment step."""
        
        # Create section from chunk
        chunk_section = GenomeSection(
            chromosome=chunk.chromosome,
            start_position=chunk.start_position,
            end_position=chunk.end_position,
            variants=chunk.variants,
        )
        
        # Align to find best reference
        alignment_result = self.aligner.align(chunk_section)
        
        # Get the aligned reference genome
        reference_genome = self.reference_manager.pool.get_reference(
            alignment_result.primary_reference
        )
        
        # Continue with original encoding logic...
        # (extract reference section, compute differences, encode)
        
        return hypervector, metadata
```

## Performance Tuning

### K-mer Length Selection

Default k=31 is optimal for human genome. Adjust based on:

```python
# For bacterial genomes (smaller, less repetitive)
kmer_index = KmerIndex(k=21)

# For large genomes with high repetition
kmer_index = KmerIndex(k=41)

# For very diverse/divergent sequences
kmer_index = KmerIndex(k=15)  # Shorter for more flexibility
```

### Variant Scoring Weights

Tune alignment scoring based on your data:

```python
from genomevault.differential_encoding.sequence_alignment import VariantAligner

# Prioritize SNPs over indels
aligner = MultiReferenceAligner(
    reference_manager=ref_manager,
    variant_aligner=VariantAligner(
        snp_weight=1.0,
        indel_weight=0.5,  # Lower weight for indels
        genotype_weight=0.1,  # Minimal genotype consideration
        position_tolerance=5,  # Tight position matching
    ),
)

# More lenient for noisy data
aligner = MultiReferenceAligner(
    reference_manager=ref_manager,
    variant_aligner=VariantAligner(
        snp_weight=0.8,
        indel_weight=0.8,
        genotype_weight=0.5,  # Consider genotypes more
        position_tolerance=20,  # Allow position shifts
    ),
)
```

### Consensus Threshold

Adjust consensus requirements:

```python
# Strict consensus (high confidence only)
aligner = MultiReferenceAligner(
    reference_manager=ref_manager,
    consensus_threshold=0.8,  # 80% agreement required
)

# Lenient consensus (accept more ambiguity)
aligner = MultiReferenceAligner(
    reference_manager=ref_manager,
    consensus_threshold=0.5,  # 50% agreement
)
```

## Handling Ambiguous Results

When `result.ambiguous == True`:

```python
result = aligner.align(query_section)

if result.ambiguous:
    print("Ambiguous alignment detected")
    print(f"Primary: {result.primary_reference} (score: {result.consensus_score:.2f})")
    print(f"Secondary: {result.secondary_references}")
    
    # Strategy 1: Use multiple references
    if len(result.secondary_references) >= 2:
        # Encode against all good matches
        for ref_id in [result.primary_reference] + result.secondary_references[:2]:
            # ... encode with each reference
            pass
    
    # Strategy 2: Request manual review
    if result.confidence < 0.5:
        print("Low confidence - manual review recommended")
        # Flag for expert review
    
    # Strategy 3: Use mosaic/admixed approach
    # Split genome into regions and use different references
    chunk_results = aligner.align_genome(genome_sections)
    for chunk_id, chunk_result in chunk_results.items():
        # Each chunk gets its best reference
        pass
```

## Advanced Usage

### Whole Genome Alignment

```python
from genomevault.differential_encoding.reference_management import Variant

# Load VCF data for entire genome
genome_variants = load_vcf_file("sample.vcf.gz")

# Create sections by chromosome
genome_sections = []
for chromosome, variants in genome_variants.items():
    section = GenomeSection(
        chromosome=chromosome,
        start_position=0,
        end_position=get_chromosome_length(chromosome),
        variants=variants,
    )
    genome_sections.append(section)

# Align all sections
aligner = create_default_aligner(ref_manager, num_references=3)
chunk_results = aligner.align_genome(genome_sections)

# Get overall consensus
overall_reference = aligner.majority_vote(chunk_results)

# Analyze per-chromosome results
for section in genome_sections:
    chunk_id = f"{section.chromosome}:{section.start_position}-{section.end_position}"
    result = chunk_results[chunk_id]
    print(f"{section.chromosome}: {result.primary_reference} "
          f"(confidence: {result.confidence:.2%})")
```

### Custom Scoring Function

```python
class CustomVariantAligner(VariantAligner):
    """Custom aligner with domain-specific scoring."""
    
    def align_section(self, query_section, reference_section, reference_id):
        # Start with base score
        score = super().align_section(query_section, reference_section, reference_id)
        
        # Add custom logic
        # Example: Boost score for specific genomic regions
        if query_section.chromosome == "chr6":  # HLA region
            score.overall_score *= 1.2  # Boost score
        
        # Example: Penalize if too many indels
        if score.indel_mismatches > 10:
            score.overall_score *= 0.8
        
        return score

# Use custom aligner
aligner = MultiReferenceAligner(
    reference_manager=ref_manager,
    variant_aligner=CustomVariantAligner(),
)
```

## Benchmarking and Validation

### Test Alignment Accuracy

```python
import time

# Test with known reference
known_reference = "GRCh38"
test_variants = extract_variants_from_reference(known_reference, "chr1", 0, 1000000)

query_section = GenomeSection("chr1", 0, 1000000, test_variants)

# Benchmark
start = time.time()
result = aligner.align(query_section)
elapsed = time.time() - start

print(f"Alignment time: {elapsed:.2f}s")
print(f"Correct: {result.primary_reference == known_reference}")
print(f"Score: {result.alignment_scores[result.primary_reference].overall_score:.3f}")
```

### Measure Performance

```python
# Test scaling with variant count
for n_variants in [10, 100, 1000, 10000]:
    variants = sample_variants[:n_variants]
    section = GenomeSection("chr1", 0, 10000000, variants)
    
    start = time.time()
    result = aligner.align(section)
    elapsed = time.time() - start
    
    print(f"{n_variants} variants: {elapsed:.3f}s "
          f"({n_variants/elapsed:.0f} variants/s)")
```

## Troubleshooting

### Low Alignment Scores

If all alignment scores are low:

1. Check variant quality filtering
2. Verify correct chromosome naming (chr1 vs 1)
3. Ensure reference genomes are appropriate for sample population
4. Consider increasing k-mer flexibility (lower k value)

```python
# Diagnostic: Check k-mer matches
kmer_index = aligner.kmer_index
match_rates = kmer_index.query_variants(query_variants)
print("K-mer match rates:", match_rates)

if not match_rates or max(match_rates.values()) < 0.1:
    print("Very low k-mer matches - check reference compatibility")
```

### High Ambiguity

If many alignments are ambiguous:

1. Increase number of references
2. Lower consensus threshold
3. Use chunk-level alignment instead of whole genome
4. Check if population is admixed

```python
# Try with more references
aligner = create_default_aligner(
    ref_manager,
    num_references=5,  # More references
    consensus_threshold=0.5,  # Lower threshold
)
```

### Performance Issues

If alignment is too slow:

1. Use k-mer only strategy for pre-screening
2. Reduce number of references
3. Implement parallel chunk processing

```python
# Fast pre-screening
fast_aligner = create_default_aligner(
    ref_manager,
    strategy=AlignmentStrategy.KMER_ONLY,
)

# Then detailed alignment on top candidates
candidates = fast_aligner._select_candidate_references(query_variants, top_k=3)

# ... detailed alignment only on candidates
```

## Best Practices

1. **Reference Selection**: Use references from the same population as your samples
2. **Quality Filtering**: Filter low-quality variants before alignment
3. **Chunking Strategy**: For whole genomes, align chromosome-by-chromosome
4. **Validation**: Always validate with known samples first
5. **Ambiguity Handling**: Have a workflow for ambiguous cases
6. **Documentation**: Log alignment results for reproducibility

## Reference

- Implementation based on modern alignment algorithms:
  - Minimap2 (Li, 2018)
  - K-mer indexing (KmerKeys, LAVA)
  - Consensus approaches (RNA-seq multi-mappers)

- Optimized for variant-based data (VCF format)
- Low computational requirements (no full sequence alignment needed)
- Handles population variation (SNPs, indels, genotypes)
