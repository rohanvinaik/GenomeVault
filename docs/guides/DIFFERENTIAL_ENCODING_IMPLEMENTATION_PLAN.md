# Differential Encoding Implementation Plan

## Executive Summary

The current GenomeVault hypervector encoding implementation does NOT match the theoretical framework of cryptographically secure differential encoding. This document outlines the required changes to implement the correct architecture.

## Theoretical Framework (Target State)

### Core Principle
Store genomic data as **cryptographically secure differences** from randomly selected reference genomes, using random chunking to ensure privacy and security.

### Key Components

1. **Multiple Reference Genomes**: Maintain a pool of cryptographically verified reference genomes
2. **Random Sectioning**: Chunk experimental genomes using cryptographically secure randomization
3. **Random Reference Selection**: For each chunk, randomly select a reference genome
4. **Differential Encoding**: Encode only the delta (experimental - reference)
5. **Location Metadata**: Include cryptographically secure information about:
   - Which reference genome was used for each section
   - Position/association information
   - Chunking parameters
6. **Dynamic Adaptation**: Adjust chunking strategy based on analysis type

## Current State Analysis

### What Exists
- ✅ Reference genome type definitions (`genomevault/types.py:ReferenceGenome`)
- ✅ Reference data manager for PIR (`genomevault/pir/reference_data/manager.py`)
- ✅ Variant encoding infrastructure (`genomevault/hypervector/encoding/genomic.py`)
- ✅ Positional encoding (`genomevault/hypervector/positional.py`)

### What's Missing
- ❌ Differential encoding (encoding experimental - reference)
- ❌ Cryptographically secure random chunking
- ❌ Reference genome pool management for encoding
- ❌ Random reference selection mechanism
- ❌ Integration between reference manager and hypervector encoder
- ❌ Analysis-type-specific encoding strategies

## Implementation Phases

### Phase 1: Reference Genome Management (Week 1-2)

**Goal**: Create a cryptographically secure reference genome manager for encoding

**Files to Create**:
```
genomevault/hypervector/reference_manager.py
genomevault/hypervector/reference_loader.py
tests/test_reference_manager.py
```

**Key Classes**:

```python
@dataclass
class ReferenceGenomeData:
    """Complete reference genome with variants"""
    genome_id: str
    assembly: str  # GRCh38, CHM13, etc.
    variants: Dict[str, List[Variant]]  # chromosome -> variants
    cryptographic_hash: str  # SHA-256 of genome
    provenance: Dict[str, Any]  # Source, date, etc.

class SecureReferenceGenomeManager:
    """Manages multiple reference genomes with cryptographic verification"""

    def __init__(self, reference_dir: Path):
        self.references: Dict[str, ReferenceGenomeData] = {}
        self.load_reference_genomes()
        self.verify_cryptographic_hashes()

    def get_random_reference(self,
                            seed: bytes,
                            exclude: Optional[List[str]] = None) -> ReferenceGenomeData:
        """Cryptographically secure random reference selection"""
        pass

    def get_reference_section(self,
                             genome_id: str,
                             chromosome: str,
                             start: int,
                             end: int) -> GenomeSection:
        """Get a section of a reference genome"""
        pass
```

**Deliverables**:
- [ ] Reference genome manager with cryptographic verification
- [ ] Reference genome loader (supports VCF, FASTA)
- [ ] Unit tests with synthetic reference genomes
- [ ] Documentation for adding new reference genomes

### Phase 2: Cryptographic Chunking (Week 3-4)

**Goal**: Implement cryptographically secure random genome sectioning

**Files to Create**:
```
genomevault/hypervector/chunking.py
genomevault/hypervector/chunking_strategies.py
tests/test_chunking.py
```

**Key Classes**:

```python
@dataclass
class GenomeChunk:
    """A chunk of experimental genome with metadata"""
    chromosome: str
    start_position: int
    end_position: int
    variants: List[Variant]
    chunk_id: bytes  # Cryptographic chunk identifier
    chunking_seed: bytes  # Seed used for this chunking

@dataclass
class ChunkingStrategy:
    """Strategy for chunking genomes"""
    strategy_type: str  # 'single_snp', 'window', 'whole_chromosome'
    chunk_size: Optional[int]  # For fixed-size windows
    overlap: int  # Overlap between chunks
    randomization: bool  # Use random chunk boundaries

class CryptographicChunker:
    """Cryptographically secure genome chunker"""

    def __init__(self, strategy: ChunkingStrategy):
        self.strategy = strategy
        self.crypto_rng = secrets.SystemRandom()

    def chunk_genome(self,
                     variants: List[Variant],
                     analysis_type: AnalysisType,
                     seed: Optional[bytes] = None) -> List[GenomeChunk]:
        """
        Chunk genome using cryptographically secure randomization

        Args:
            variants: Experimental genome variants
            analysis_type: Type of analysis (affects chunking)
            seed: Optional seed for deterministic chunking

        Returns:
            List of genome chunks with cryptographic IDs
        """
        if seed is None:
            seed = secrets.token_bytes(32)

        # Select strategy based on analysis type
        strategy = self._select_strategy(analysis_type)

        # Generate random chunk boundaries
        chunks = self._generate_chunks(variants, strategy, seed)

        # Assign cryptographic IDs
        for chunk in chunks:
            chunk.chunk_id = self._compute_chunk_id(chunk, seed)

        return chunks

    def _select_strategy(self, analysis_type: AnalysisType) -> ChunkingStrategy:
        """Select chunking strategy based on analysis type"""
        if analysis_type == AnalysisType.SINGLE_SNP_QUERY:
            return ChunkingStrategy(
                strategy_type='single_snp',
                chunk_size=1,
                overlap=0,
                randomization=False
            )
        elif analysis_type == AnalysisType.REGION_QUERY:
            return ChunkingStrategy(
                strategy_type='window',
                chunk_size=10000,  # 10kb windows
                overlap=1000,
                randomization=True
            )
        elif analysis_type == AnalysisType.WHOLE_GENOME:
            return ChunkingStrategy(
                strategy_type='whole_chromosome',
                chunk_size=None,
                overlap=0,
                randomization=True
            )
```

**Deliverables**:
- [ ] Cryptographic chunking implementation
- [ ] Multiple chunking strategies (single SNP, window, chromosome)
- [ ] Analysis-type-specific strategy selection
- [ ] Unit tests for deterministic reproducibility
- [ ] Security analysis documentation

### Phase 3: Differential Encoding (Week 5-6)

**Goal**: Implement encoding of (experimental - reference) differences

**Files to Modify/Create**:
```
genomevault/hypervector/differential_encoder.py  (NEW)
genomevault/hypervector/encoding/genomic.py      (MODIFY)
genomevault/hypervector_transform/encoding.py    (MODIFY)
tests/test_differential_encoding.py              (NEW)
```

**Key Classes**:

```python
@dataclass
class DifferentialEncodingMetadata:
    """Metadata about differential encoding"""
    reference_genome_id: str
    reference_selection_seed: bytes
    chunk_boundaries: List[Tuple[int, int]]
    encoding_timestamp: float
    analysis_type: AnalysisType

class DifferentialEncoder:
    """Encodes genomic differences using hypervectors"""

    def __init__(self,
                 reference_manager: SecureReferenceGenomeManager,
                 chunker: CryptographicChunker,
                 hv_config: HypervectorConfig):
        self.reference_manager = reference_manager
        self.chunker = chunker
        self.hv_encoder = HypervectorEncoder(hv_config)

    def encode_experimental_genome(self,
                                   experimental_variants: List[Variant],
                                   analysis_type: AnalysisType,
                                   seed: Optional[bytes] = None) -> Tuple[torch.Tensor, DifferentialEncodingMetadata]:
        """
        Encode experimental genome as differences from reference

        Process:
        1. Chunk experimental genome using cryptographic chunking
        2. For each chunk:
           a. Randomly select a reference genome (cryptographically secure)
           b. Extract matching reference section
           c. Compute difference: experimental - reference
           d. Encode the difference into hypervector
        3. Bundle all chunk hypervectors
        4. Return bundled hypervector + metadata
        """
        if seed is None:
            seed = secrets.token_bytes(32)

        # Step 1: Chunk experimental genome
        chunks = self.chunker.chunk_genome(experimental_variants, analysis_type, seed)

        chunk_hypervectors = []
        metadata = DifferentialEncodingMetadata(
            reference_genome_id="",  # Will be set
            reference_selection_seed=seed,
            chunk_boundaries=[],
            encoding_timestamp=time.time(),
            analysis_type=analysis_type
        )

        for chunk in chunks:
            # Step 2a: Randomly select reference genome for this chunk
            ref_genome = self.reference_manager.get_random_reference(
                seed=chunk.chunk_id,
                exclude=None
            )

            # Step 2b: Extract matching reference section
            ref_section = self.reference_manager.get_reference_section(
                genome_id=ref_genome.genome_id,
                chromosome=chunk.chromosome,
                start=chunk.start_position,
                end=chunk.end_position
            )

            # Step 2c: Compute difference
            diff_variants = self._compute_variant_difference(
                experimental=chunk.variants,
                reference=ref_section.variants
            )

            # Step 2d: Encode the difference
            chunk_hv = self._encode_difference(
                diff_variants=diff_variants,
                chunk=chunk,
                reference_id=ref_genome.genome_id
            )

            chunk_hypervectors.append(chunk_hv)
            metadata.chunk_boundaries.append((chunk.start_position, chunk.end_position))

        # Step 3: Bundle all chunks
        final_hypervector = self._bundle_chunks(chunk_hypervectors)

        return final_hypervector, metadata

    def _compute_variant_difference(self,
                                   experimental: List[Variant],
                                   reference: List[Variant]) -> List[VariantDifference]:
        """
        Compute differences between experimental and reference variants

        Returns list of differences:
        - Variants present in experimental but not reference (new mutations)
        - Variants present in reference but not experimental (reference-specific)
        - Variants with different genotypes
        """
        exp_dict = {(v.position, v.alt): v for v in experimental}
        ref_dict = {(v.position, v.alt): v for v in reference}

        differences = []

        # New mutations in experimental
        for key, exp_var in exp_dict.items():
            if key not in ref_dict:
                differences.append(VariantDifference(
                    type='new_mutation',
                    position=exp_var.position,
                    experimental_allele=exp_var.alt,
                    reference_allele=None
                ))

        # Reference-specific variants
        for key, ref_var in ref_dict.items():
            if key not in exp_dict:
                differences.append(VariantDifference(
                    type='reference_specific',
                    position=ref_var.position,
                    experimental_allele=None,
                    reference_allele=ref_var.alt
                ))

        # Genotype differences
        for key in set(exp_dict.keys()) & set(ref_dict.keys()):
            exp_var = exp_dict[key]
            ref_var = ref_dict[key]
            if exp_var.genotype != ref_var.genotype:
                differences.append(VariantDifference(
                    type='genotype_diff',
                    position=exp_var.position,
                    experimental_allele=exp_var.alt,
                    reference_allele=ref_var.alt,
                    exp_genotype=exp_var.genotype,
                    ref_genotype=ref_var.genotype
                ))

        return differences

    def _encode_difference(self,
                          diff_variants: List[VariantDifference],
                          chunk: GenomeChunk,
                          reference_id: str) -> torch.Tensor:
        """
        Encode variant differences into hypervector

        Encoding includes:
        - Difference type (new mutation, reference-specific, genotype diff)
        - Position information
        - Allele information
        - Reference genome ID (cryptographically bound)
        """
        # Create feature vector from differences
        features = self._differences_to_features(diff_variants)

        # Encode with standard hypervector encoder
        hv_diff = self.hv_encoder.encode(features, OmicsType.GENOMIC)

        # Bind with reference ID (cryptographic association)
        ref_id_vector = self._encode_reference_id(reference_id)
        hv_bound = self._bind_vectors(hv_diff, ref_id_vector)

        # Bind with position information
        pos_vector = self._encode_position_range(
            chunk.chromosome,
            chunk.start_position,
            chunk.end_position
        )
        hv_final = self._bind_vectors(hv_bound, pos_vector)

        return hv_final

    def _bind_vectors(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """Bind two hypervectors using circular convolution"""
        return torch.fft.irfft(torch.fft.rfft(v1) * torch.fft.rfft(v2))

    def _bundle_chunks(self, chunk_hvs: List[torch.Tensor]) -> torch.Tensor:
        """Bundle chunk hypervectors using superposition"""
        bundled = torch.stack(chunk_hvs).sum(dim=0)
        return bundled / torch.norm(bundled)
```

**Deliverables**:
- [ ] Differential encoder implementation
- [ ] Variant difference computation
- [ ] Cryptographic binding of reference IDs
- [ ] Metadata tracking for provenance
- [ ] Unit tests with known differences
- [ ] Integration tests with real reference genomes

### Phase 4: Integration and Testing (Week 7-8)

**Goal**: Integrate differential encoding into main pipeline and validate

**Files to Modify**:
```
genomevault/api/routers/hdc.py
genomevault/cli/main.py
genomevault/hypervector_transform/encoding.py
```

**Tasks**:

1. **Update API Endpoints**:
```python
@router.post("/encode_differential")
async def encode_differential(
    variants: List[Variant],
    analysis_type: AnalysisType,
    reference_pool: Optional[List[str]] = None
) -> DifferentialEncodingResponse:
    """Encode genome using differential encoding"""
    pass
```

2. **Update CLI**:
```bash
genomevault hdc encode-differential \
    --vcf experimental.vcf \
    --analysis-type single-snp \
    --reference-pool GRCh38,CHM13 \
    --output encoded.json
```

3. **Comprehensive Testing**:
   - [ ] Unit tests for each component
   - [ ] Integration tests for complete pipeline
   - [ ] Security tests (verify randomization, cryptographic properties)
   - [ ] Performance benchmarks
   - [ ] Reproducibility tests (same seed = same encoding)

4. **Documentation**:
   - [ ] API documentation
   - [ ] User guide for differential encoding
   - [ ] Security model documentation
   - [ ] Examples and tutorials

### Phase 5: Paper Updates (Week 9)

**Goal**: Update academic paper to accurately reflect differential encoding

**Sections to Update**:

1. **Section 3.2.2 - Genomic Encoding Algorithm** (lines 161-186):
   - Replace with differential encoding algorithm
   - Add reference genome management
   - Add cryptographic chunking steps

2. **New Section 3.2.3 - Differential Encoding Security**:
   - Cryptographic properties of random chunking
   - Reference selection randomization
   - Information-theoretic analysis of differential encoding

3. **Section 4.1 - Update Performance Results**:
   - Benchmark differential encoding speed
   - Measure security properties
   - Compare storage efficiency

4. **Section 5.3 - Update Limitations**:
   - Requirements for reference genome pool
   - Chunking strategy selection trade-offs

## Implementation Checklist

### Phase 1: Reference Management
- [ ] SecureReferenceGenomeManager class
- [ ] Reference genome loader (VCF/FASTA)
- [ ] Cryptographic hash verification
- [ ] Random reference selection (cryptographically secure)
- [ ] Unit tests
- [ ] Documentation

### Phase 2: Chunking
- [ ] CryptographicChunker class
- [ ] ChunkingStrategy implementations
- [ ] Analysis-type-specific strategy selection
- [ ] Chunk ID generation (cryptographic)
- [ ] Unit tests
- [ ] Security analysis

### Phase 3: Differential Encoding
- [ ] DifferentialEncoder class
- [ ] Variant difference computation
- [ ] Difference encoding to hypervectors
- [ ] Reference ID binding
- [ ] Metadata tracking
- [ ] Integration tests

### Phase 4: Integration
- [ ] API endpoint updates
- [ ] CLI command updates
- [ ] E2E testing
- [ ] Performance benchmarks
- [ ] Documentation

### Phase 5: Paper
- [ ] Update algorithm description
- [ ] Add security analysis section
- [ ] Update performance results
- [ ] Update limitations section
- [ ] Revise abstract/conclusions

## Security Considerations

1. **Cryptographic Randomness**: Use `secrets` module for all random selection
2. **Reference Integrity**: SHA-256 hashes for all reference genomes
3. **Chunk ID Generation**: HMAC-based chunk identifiers
4. **Side-Channel Resistance**: Constant-time operations where possible
5. **Audit Trail**: Log all reference selections and chunking operations

## Performance Targets

- **Encoding Speed**: <10ms per genome (including chunking and differential computation)
- **Memory Usage**: <2GB for 10 reference genomes
- **Storage**: <2KB per encoded genome (including metadata)
- **Reproducibility**: 100% identical encodings for same seed

## Open Questions

1. **Reference Genome Selection**: How many reference genomes in the pool?
   - Proposed: Start with 5 (GRCh38, GRCh37, CHM13, HG002, HG003)

2. **Chunking Granularity**: Default chunk size for different analysis types?
   - Single SNP: 1 variant
   - Region query: 10kb windows with 1kb overlap
   - Whole genome: Per-chromosome

3. **Backward Compatibility**: Support migration from current encoding?
   - Proposed: Add encoding version field, maintain both encoders

4. **Performance vs Security**: Trade-offs in randomization level?
   - Proposed: Configurable randomization strength parameter

## Timeline

- **Week 1-2**: Phase 1 (Reference Management)
- **Week 3-4**: Phase 2 (Chunking)
- **Week 5-6**: Phase 3 (Differential Encoding)
- **Week 7-8**: Phase 4 (Integration & Testing)
- **Week 9**: Phase 5 (Paper Updates)

**Total**: 9 weeks for complete implementation and validation
