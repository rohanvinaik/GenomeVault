"""
Cryptographic Chunking Strategies for Differential Encoding.

This module implements analysis-type-specific genomic chunking with:
1. Multiple analysis types (single SNP, gene region, sliding window, etc.)
2. Configurable chunking strategies
3. Cryptographically secure random boundary generation
4. Feature-aware chunking (genes, exons, regulatory elements)
5. Deterministic chunking for reproducibility

Key Components:
- AnalysisType: Enum of supported analysis types
- ChunkingStrategy: Configuration for chunking parameters
- STRATEGY_CONFIGS: Pre-configured strategies for each analysis type
- GenomeChunk: Chunk with cryptographic metadata
- CryptographicChunker: Main chunking engine

Security Features:
- Cryptographically secure random boundaries
- Deterministic chunking (same seed → same chunks)
- Chunk ID generation with collision resistance
- Variant count constraints for quality control
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import hashlib
import hmac

from genomevault.differential_encoding.crypto_primitives import (
    CryptoRNG,
    compute_chunk_id,
)
from genomevault.differential_encoding.reference_management import (
    Variant,
    GenomeSection,
)
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class AnalysisType(Enum):
    """
    Type of genomic analysis to perform.

    Each analysis type has different chunking requirements:
    - SINGLE_SNP_QUERY: Small windows around specific variants
    - GENE_REGION: Gene-based chunking with flanking regions
    - SLIDING_WINDOW: Fixed-size overlapping windows
    - WHOLE_CHROMOSOME: Large chunks for chromosome-wide analysis
    - STRUCTURAL_VARIANT: Optimized for large structural variations
    - HAPLOTYPE_PHASE: Preserves linkage disequilibrium structure
    - GWAS_ASSOCIATION: Association study-optimized chunking

    Example:
        >>> analysis = AnalysisType.SINGLE_SNP_QUERY
        >>> strategy = STRATEGY_CONFIGS[analysis]
        >>> print(strategy.chunk_size)
        1000
    """

    SINGLE_SNP_QUERY = "single_snp"           # Query specific variant
    GENE_REGION = "gene_region"               # Gene-level analysis
    SLIDING_WINDOW = "sliding_window"         # Window-based scan
    WHOLE_CHROMOSOME = "whole_chromosome"     # Chromosome-wide
    STRUCTURAL_VARIANT = "structural_variant" # Large variants
    HAPLOTYPE_PHASE = "haplotype_phase"       # Phasing analysis
    GWAS_ASSOCIATION = "gwas_association"     # Association studies


@dataclass
class ChunkingStrategy:
    """
    Strategy for dividing genome into sections.

    Defines parameters for how to chunk a genome based on analysis type.

    Attributes:
        strategy_type: Type of analysis
        chunk_size: Base pairs per chunk (None = dynamic sizing)
        overlap: Overlap between adjacent chunks (bp)
        min_variants: Minimum variants required per chunk
        max_variants: Maximum variants allowed per chunk
        randomize_boundaries: Use cryptographic random boundaries
        respect_features: Align chunks to genomic features (genes, etc.)

    Example:
        >>> strategy = ChunkingStrategy(
        ...     strategy_type=AnalysisType.SLIDING_WINDOW,
        ...     chunk_size=100000,
        ...     overlap=10000,
        ...     min_variants=50,
        ...     max_variants=5000,
        ...     randomize_boundaries=True,
        ...     respect_features=False
        ... )
    """

    strategy_type: AnalysisType
    chunk_size: Optional[int]      # Base pairs per chunk (None = dynamic)
    overlap: int                   # Overlap between adjacent chunks (bp)
    min_variants: int              # Minimum variants per chunk
    max_variants: int              # Maximum variants per chunk
    randomize_boundaries: bool     # Use random boundaries
    respect_features: bool         # Align to genes/features

    def __str__(self) -> str:
        return (
            f"{self.strategy_type.value}(size={self.chunk_size}, "
            f"overlap={self.overlap})"
        )


# Predefined strategies for common analysis types
STRATEGY_CONFIGS: Dict[AnalysisType, ChunkingStrategy] = {
    AnalysisType.SINGLE_SNP_QUERY: ChunkingStrategy(
        strategy_type=AnalysisType.SINGLE_SNP_QUERY,
        chunk_size=1000,              # 1kb windows around each SNP
        overlap=500,                  # 500bp overlap for context
        min_variants=1,
        max_variants=50,
        randomize_boundaries=True,
        respect_features=False
    ),

    AnalysisType.GENE_REGION: ChunkingStrategy(
        strategy_type=AnalysisType.GENE_REGION,
        chunk_size=None,              # Dynamic based on gene size
        overlap=5000,                 # 5kb flanking regions
        min_variants=1,
        max_variants=10000,
        randomize_boundaries=False,   # Respect gene boundaries
        respect_features=True
    ),

    AnalysisType.SLIDING_WINDOW: ChunkingStrategy(
        strategy_type=AnalysisType.SLIDING_WINDOW,
        chunk_size=100000,            # 100kb windows
        overlap=10000,                # 10kb overlap
        min_variants=50,
        max_variants=5000,
        randomize_boundaries=True,
        respect_features=False
    ),

    AnalysisType.WHOLE_CHROMOSOME: ChunkingStrategy(
        strategy_type=AnalysisType.WHOLE_CHROMOSOME,
        chunk_size=5000000,           # 5 Mb chunks
        overlap=500000,               # 500kb overlap
        min_variants=1000,
        max_variants=100000,
        randomize_boundaries=True,
        respect_features=False
    ),

    AnalysisType.STRUCTURAL_VARIANT: ChunkingStrategy(
        strategy_type=AnalysisType.STRUCTURAL_VARIANT,
        chunk_size=1000000,           # 1 Mb chunks for SVs
        overlap=100000,               # 100kb overlap
        min_variants=10,
        max_variants=1000,
        randomize_boundaries=True,
        respect_features=False
    ),

    AnalysisType.HAPLOTYPE_PHASE: ChunkingStrategy(
        strategy_type=AnalysisType.HAPLOTYPE_PHASE,
        chunk_size=50000,             # 50kb for haplotype blocks
        overlap=25000,                # 25kb overlap
        min_variants=10,
        max_variants=1000,
        randomize_boundaries=False,   # Preserve LD structure
        respect_features=False
    ),

    AnalysisType.GWAS_ASSOCIATION: ChunkingStrategy(
        strategy_type=AnalysisType.GWAS_ASSOCIATION,
        chunk_size=250000,            # 250kb for GWAS
        overlap=50000,                # 50kb overlap
        min_variants=100,
        max_variants=10000,
        randomize_boundaries=True,
        respect_features=False
    ),
}


@dataclass
class GenomicFeature:
    """
    Genomic feature for feature-aware chunking.

    Represents genes, exons, regulatory elements, etc.

    Attributes:
        feature_id: Unique identifier
        feature_type: Type (gene, exon, regulatory, etc.)
        chromosome: Chromosome name
        start: Start position
        end: End position
        name: Feature name (e.g., gene symbol)
        strand: Strand orientation (+/-)

    Example:
        >>> feature = GenomicFeature(
        ...     feature_id="ENSG00000139618",
        ...     feature_type="gene",
        ...     chromosome="chr13",
        ...     start=32889617,
        ...     end=32973809,
        ...     name="BRCA2",
        ...     strand="+"
        ... )
    """

    feature_id: str
    feature_type: str
    chromosome: str
    start: int
    end: int
    name: str
    strand: str = "+"

    @property
    def length(self) -> int:
        """Get feature length in base pairs."""
        return self.end - self.start


@dataclass
class Genome:
    """
    Complete genome with variants across all chromosomes.

    Represents a full experimental or reference genome for chunking.

    Attributes:
        genome_id: Unique genome identifier
        assembly: Reference assembly (e.g., GRCh38, GRCh37)
        chromosomes: Dict mapping chromosome names to variant lists
        metadata: Optional additional metadata

    Properties:
        chromosome_names: List of chromosome names
        total_variants: Total number of variants across all chromosomes

    Example:
        >>> genome = Genome(
        ...     genome_id="HG001",
        ...     assembly="GRCh38",
        ...     chromosomes={
        ...         "chr1": [variant1, variant2, ...],
        ...         "chr2": [variant3, variant4, ...],
        ...     }
        ... )
        >>> print(f"Genome has {genome.total_variants} variants")
    """

    genome_id: str
    assembly: str
    chromosomes: Dict[str, List[Variant]]
    metadata: Dict[str, any] = field(default_factory=dict)

    @property
    def chromosome_names(self) -> List[str]:
        """Get sorted list of chromosome names."""
        return sorted(self.chromosomes.keys())

    @property
    def total_variants(self) -> int:
        """Get total number of variants across all chromosomes."""
        return sum(len(variants) for variants in self.chromosomes.values())

    def get_chromosome_section(
        self,
        chromosome: str,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> GenomeSection:
        """
        Get a GenomeSection for a specific chromosome region.

        Args:
            chromosome: Chromosome name
            start: Start position (None = chromosome start)
            end: End position (None = chromosome end)

        Returns:
            GenomeSection for the specified region

        Raises:
            ValueError: If chromosome not found
        """
        if chromosome not in self.chromosomes:
            raise ValueError(f"Chromosome {chromosome} not found in genome")

        variants = self.chromosomes[chromosome]

        # Handle empty chromosome
        if not variants:
            if start is not None and end is not None:
                return GenomeSection(
                    chromosome=chromosome,
                    start_position=start,
                    end_position=end,
                    variants=[]
                )
            else:
                # Cannot create section without bounds for empty chromosome
                # Return a minimal valid section (1bp at position 0)
                return GenomeSection(
                    chromosome=chromosome,
                    start_position=0,
                    end_position=1,
                    variants=[]
                )

        # Non-empty chromosome
        sorted_variants = sorted(variants, key=lambda v: v.position)
        section_start = start or sorted_variants[0].position
        section_end = end or sorted_variants[-1].position + 1

        # Filter variants to region
        if start is not None or end is not None:
            filtered_variants = [
                v for v in variants
                if (start is None or v.position >= start) and
                   (end is None or v.position < end)
            ]
        else:
            filtered_variants = variants

        return GenomeSection(
            chromosome=chromosome,
            start_position=section_start,
            end_position=section_end,
            variants=filtered_variants
        )

    def __str__(self) -> str:
        return (
            f"Genome({self.genome_id}, {self.assembly}, "
            f"{len(self.chromosomes)} chromosomes, {self.total_variants:,} variants)"
        )


@dataclass
class GenomeChunk:
    """
    Genomic chunk with cryptographic metadata.

    Represents a contiguous section of genome created by chunking strategy.

    Attributes:
        chromosome: Chromosome identifier
        start_position: Start position (inclusive)
        end_position: End position (exclusive)
        variants: List of variants in chunk
        chunk_id: Cryptographic identifier (32 bytes)
        chunking_seed: Seed used for this chunk
        feature_id: Optional associated feature ID
        feature_name: Optional associated feature name

    Properties:
        length: Genomic length in base pairs
        variant_count: Number of variants in chunk

    Example:
        >>> chunk = GenomeChunk(
        ...     chromosome="chr1",
        ...     start_position=100000,
        ...     end_position=200000,
        ...     variants=[variant1, variant2],
        ...     chunk_id=b"...",
        ...     chunking_seed=b"..."
        ... )
        >>> print(f"Chunk length: {chunk.length:,} bp")
        >>> print(f"Variants: {chunk.variant_count}")
    """

    chromosome: str
    start_position: int
    end_position: int
    variants: List[Variant]
    chunk_id: Optional[bytes] = None           # Cryptographic identifier
    chunking_seed: Optional[bytes] = None      # Seed used for this chunk

    # Optional feature association
    feature_id: Optional[str] = None
    feature_name: Optional[str] = None

    def __len__(self) -> int:
        """Get chunk length in base pairs."""
        return self.end_position - self.start_position

    @property
    def length(self) -> int:
        """Get chunk length in base pairs."""
        return self.end_position - self.start_position

    @property
    def variant_count(self) -> int:
        """Get number of variants in chunk."""
        return len(self.variants)

    def __str__(self) -> str:
        feature_str = f" ({self.feature_name})" if self.feature_name else ""
        return (
            f"GenomeChunk({self.chromosome}:{self.start_position}-{self.end_position}, "
            f"{self.variant_count} variants{feature_str})"
        )


class CryptographicChunker:
    """
    Cryptographically secure genome chunker.

    Chunks genomes using analysis-type-specific strategies with:
    - Deterministic chunking (same seed → same chunks)
    - Cryptographic random boundaries
    - Variant count constraints
    - Feature-aware chunking

    Attributes:
        strategy: Chunking strategy to use
        crypto_rng: Cryptographic RNG for random boundaries

    Example:
        >>> from genomevault.differential_encoding import CryptoRNG, AnalysisType
        >>> rng = CryptoRNG()
        >>> strategy = STRATEGY_CONFIGS[AnalysisType.SLIDING_WINDOW]
        >>> chunker = CryptographicChunker(strategy, rng)
        >>> chunks = chunker.chunk_genome_section(
        ...     section,
        ...     master_seed=rng.derive_seed(b"chunking")
        ... )
    """

    def __init__(
        self,
        strategy: ChunkingStrategy,
        crypto_rng: CryptoRNG
    ):
        """
        Initialize chunker.

        Args:
            strategy: Chunking strategy
            crypto_rng: Cryptographic RNG instance
        """
        self.strategy = strategy
        self.crypto_rng = crypto_rng

    def chunk_genome(
        self,
        genome: Genome,
        analysis_type: AnalysisType,
        master_seed: bytes,
        feature_annotations: Optional[Dict[str, List[GenomicFeature]]] = None
    ) -> List[GenomeChunk]:
        """
        Chunk entire genome across all chromosomes.

        This is the main entry point for genome-wide chunking. It:
        1. Selects the appropriate ChunkingStrategy for the analysis type
        2. Processes each chromosome independently
        3. Returns all chunks with cryptographic IDs

        Args:
            genome: Complete genome to chunk
            analysis_type: Type of analysis (determines chunking strategy)
            master_seed: Master seed for deterministic chunking
            feature_annotations: Optional dict mapping chromosome -> features

        Returns:
            List of all chunks across all chromosomes, sorted by chromosome and position

        Example:
            >>> genome = Genome(
            ...     genome_id="HG001",
            ...     assembly="GRCh38",
            ...     chromosomes={"chr1": variants1, "chr2": variants2}
            ... )
            >>> chunker = CryptographicChunker(strategy, rng)
            >>> chunks = chunker.chunk_genome(
            ...     genome,
            ...     AnalysisType.SLIDING_WINDOW,
            ...     master_seed=rng.derive_seed(b"experiment_1")
            ... )
            >>> print(f"Created {len(chunks)} chunks across {len(genome.chromosomes)} chromosomes")

        Notes:
            - Each chromosome is chunked independently
            - Chunk IDs are unique across the entire genome
            - Chunking is deterministic: same inputs → same outputs
            - Feature annotations are filtered per chromosome
        """
        # Select strategy for analysis type
        strategy = STRATEGY_CONFIGS.get(analysis_type)
        if strategy is None:
            raise ValueError(f"No strategy configured for analysis type: {analysis_type}")

        # Store original strategy and temporarily use the selected one
        original_strategy = self.strategy
        self.strategy = strategy

        all_chunks = []

        try:
            # Process each chromosome
            for chromosome in sorted(genome.chromosome_names):
                # Get chromosome section
                section = genome.get_chromosome_section(chromosome)

                # Skip empty chromosomes
                if section.variant_count == 0:
                    logger.debug(f"Skipping empty chromosome: {chromosome}")
                    continue

                # Get features for this chromosome if provided
                chromosome_features = None
                if feature_annotations and chromosome in feature_annotations:
                    chromosome_features = feature_annotations[chromosome]

                # Derive chromosome-specific seed for determinism
                chromosome_seed = hmac.new(
                    master_seed,
                    f"chromosome_{chromosome}".encode('utf-8'),
                    hashlib.sha256
                ).digest()

                # Chunk this chromosome
                chromosome_chunks = self.chunk_genome_section(
                    section,
                    chromosome_seed,
                    features=chromosome_features
                )

                all_chunks.extend(chromosome_chunks)

                logger.info(
                    f"Chunked {chromosome}: {len(chromosome_chunks)} chunks, "
                    f"{section.variant_count:,} variants"
                )

        finally:
            # Restore original strategy
            self.strategy = original_strategy

        logger.info(
            f"Genome chunking complete: {len(all_chunks)} total chunks "
            f"across {len(genome.chromosomes)} chromosomes using {analysis_type.value} strategy"
        )

        return all_chunks

    def chunk_genome_section(
        self,
        section: GenomeSection,
        master_seed: bytes,
        features: Optional[List[GenomicFeature]] = None
    ) -> List[GenomeChunk]:
        """
        Chunk a genome section using configured strategy.

        Args:
            section: Genome section to chunk
            master_seed: Master seed for deterministic chunking
            features: Optional genomic features for feature-aware chunking

        Returns:
            List of chunks with cryptographic IDs

        Example:
            >>> section = GenomeSection("chr1", 100000, 500000, variants)
            >>> chunks = chunker.chunk_genome_section(
            ...     section,
            ...     master_seed=seed,
            ...     features=genes
            ... )
            >>> print(f"Created {len(chunks)} chunks")
        """
        if self.strategy.respect_features and features:
            # Filter features to this section
            section_features = [
                f for f in features
                if f.chromosome == section.chromosome
                and f.start < section.end_position
                and f.end > section.start_position
            ]
            chunks = self._chunk_by_features(
                section,
                section_features,
                master_seed
            )
        else:
            chunks = self._chunk_by_windows(
                section,
                master_seed
            )

        # Assign cryptographic IDs (deterministically from master_seed)
        for i, chunk in enumerate(chunks):
            # Use HMAC directly on master_seed to ensure determinism
            context = f"{section.chromosome}_chunk_{i}".encode('utf-8')
            chunk_seed = hmac.new(
                master_seed,
                context,
                hashlib.sha256
            ).digest()
            chunk.chunk_id = compute_chunk_id(chunk, chunk_seed)
            chunk.chunking_seed = chunk_seed

        logger.debug(
            f"Created {len(chunks)} chunks for {section} using "
            f"{self.strategy.strategy_type.value} strategy"
        )

        return chunks

    def _chunk_by_windows(
        self,
        section: GenomeSection,
        master_seed: bytes
    ) -> List[GenomeChunk]:
        """
        Chunk by sliding windows.

        Args:
            section: Genome section
            master_seed: Master seed for randomization

        Returns:
            List of window-based chunks
        """
        if not section.variants:
            return []

        chunks = []
        variants = sorted(section.variants, key=lambda v: v.position)

        # Create deterministic RNG from master_seed for this chunking operation
        local_rng = CryptoRNG(master_seed=master_seed)

        # Calculate boundary jitter if randomization enabled
        if self.strategy.randomize_boundaries and self.strategy.chunk_size:
            jitter_seed = local_rng.derive_seed(
                f"{section.chromosome}_jitter".encode()
            )
            boundary_jitter = local_rng.random_int(
                0,
                self.strategy.chunk_size // 10,  # Up to 10% jitter
                jitter_seed
            )
        else:
            boundary_jitter = 0

        current_pos = section.start_position
        chunk_idx = 0

        while current_pos < section.end_position:
            # Determine chunk boundaries
            chunk_start = current_pos
            chunk_end = min(
                current_pos + self.strategy.chunk_size,
                section.end_position
            ) if self.strategy.chunk_size else section.end_position

            # Apply randomization if enabled
            if self.strategy.randomize_boundaries and boundary_jitter > 0:
                jitter_seed = local_rng.derive_seed(
                    f"{section.chromosome}_{chunk_idx}".encode()
                )
                jitter = local_rng.random_int(
                    -boundary_jitter,
                    boundary_jitter + 1,
                    jitter_seed
                )
                chunk_end = min(chunk_end + jitter, section.end_position)
                chunk_end = max(chunk_end, chunk_start + 1)  # Ensure positive length

            # Extract variants in this chunk
            chunk_variants = [
                v for v in variants
                if chunk_start <= v.position < chunk_end
            ]

            # Apply variant count constraints
            if len(chunk_variants) < self.strategy.min_variants:
                # Try to extend chunk to include more variants
                chunk_end = self._extend_to_min_variants(
                    variants,
                    chunk_start,
                    chunk_end,
                    section.end_position
                )
                chunk_variants = [
                    v for v in variants
                    if chunk_start <= v.position < chunk_end
                ]

            if len(chunk_variants) > self.strategy.max_variants:
                # Truncate to max variants
                chunk_variants = chunk_variants[:self.strategy.max_variants]
                if chunk_variants:
                    chunk_end = chunk_variants[-1].position + 1

            # Create chunk if has variants
            if chunk_variants:
                chunks.append(GenomeChunk(
                    chromosome=section.chromosome,
                    start_position=chunk_start,
                    end_position=chunk_end,
                    variants=chunk_variants,
                    chunk_id=None,  # Set later
                    chunking_seed=None
                ))

            # Move to next chunk (with overlap)
            # CRITICAL: Guarantee forward progress to prevent infinite loops
            if self.strategy.chunk_size:
                next_pos = chunk_end - self.strategy.overlap
                
                # Ensure we always advance by at least 1 base pair
                # This prevents infinite loops when chunk_end is adjusted
                # or when overlap is large relative to chunk size
                if next_pos <= current_pos:
                    # Force minimum advancement: 1% of chunk_size or 1000bp, whichever is larger
                    min_advance = max(self.strategy.chunk_size // 100, 1000)
                    next_pos = current_pos + min_advance
                    logger.debug(
                        f"Forced advancement from {current_pos} to {next_pos} "
                        f"(chunk_end={chunk_end}, overlap={self.strategy.overlap})"
                    )
                
                current_pos = min(next_pos, section.end_position)
            else:
                break  # Dynamic sizing, done after one chunk

            chunk_idx += 1

            # Safety: prevent infinite loop (should rarely trigger now)
            if chunk_idx > 100000:
                logger.error(
                    f"Chunking exceeded 100k iterations at position {current_pos}/{section.end_position}. "
                    f"This indicates a bug in the chunking logic. "
                    f"Last chunk: start={chunk_start}, end={chunk_end}, "
                    f"overlap={self.strategy.overlap}, chunk_size={self.strategy.chunk_size}"
                )
                break

        return chunks

    def _chunk_by_features(
        self,
        section: GenomeSection,
        features: List[GenomicFeature],
        master_seed: bytes
    ) -> List[GenomeChunk]:
        """
        Chunk by genomic features (genes, exons, regulatory elements).

        Args:
            section: Genome section
            features: Genomic features
            master_seed: Master seed

        Returns:
            List of feature-based chunks
        """
        chunks = []
        variants = sorted(section.variants, key=lambda v: v.position)

        for feature in features:
            # Define chunk boundaries with flanking regions
            chunk_start = max(
                section.start_position,
                feature.start - self.strategy.overlap
            )
            chunk_end = min(
                section.end_position,
                feature.end + self.strategy.overlap
            )

            # Extract variants in feature region
            chunk_variants = [
                v for v in variants
                if chunk_start <= v.position < chunk_end
            ]

            # Only create chunk if meets minimum variant requirement
            if len(chunk_variants) >= self.strategy.min_variants:
                # Truncate if exceeds max
                if len(chunk_variants) > self.strategy.max_variants:
                    chunk_variants = chunk_variants[:self.strategy.max_variants]
                    if chunk_variants:
                        chunk_end = min(chunk_end, chunk_variants[-1].position + 1)

                chunks.append(GenomeChunk(
                    chromosome=section.chromosome,
                    start_position=chunk_start,
                    end_position=chunk_end,
                    variants=chunk_variants,
                    chunk_id=None,  # Set later
                    chunking_seed=None,
                    feature_id=feature.feature_id,
                    feature_name=feature.name
                ))

        return chunks

    def _extend_to_min_variants(
        self,
        variants: List[Variant],
        chunk_start: int,
        chunk_end: int,
        max_end: int
    ) -> int:
        """
        Extend chunk boundary to include minimum variants.

        Args:
            variants: Sorted list of variants
            chunk_start: Chunk start position
            chunk_end: Current chunk end position
            max_end: Maximum allowed end position

        Returns:
            Extended chunk end position
        """
        # Count current variants
        current_count = sum(
            1 for v in variants
            if chunk_start <= v.position < chunk_end
        )

        if current_count >= self.strategy.min_variants:
            return chunk_end

        # Find variants after current end
        later_variants = [
            v for v in variants
            if v.position >= chunk_end
        ]

        if not later_variants:
            return max_end

        # Extend to include enough variants
        needed = self.strategy.min_variants - current_count
        if len(later_variants) >= needed:
            # Extend to position of nth variant
            new_end = later_variants[needed - 1].position + 1
            return min(new_end, max_end)
        else:
            # Include all remaining variants
            return max_end


def get_strategy_for_analysis(analysis_type: AnalysisType) -> ChunkingStrategy:
    """
    Get pre-configured strategy for analysis type.

    Args:
        analysis_type: Type of analysis

    Returns:
        Chunking strategy

    Raises:
        ValueError: If analysis type not in STRATEGY_CONFIGS

    Example:
        >>> strategy = get_strategy_for_analysis(AnalysisType.GENE_REGION)
        >>> print(f"Chunk size: {strategy.chunk_size}")
        >>> print(f"Respects features: {strategy.respect_features}")
    """
    if analysis_type not in STRATEGY_CONFIGS:
        raise ValueError(
            f"No strategy configured for {analysis_type}. "
            f"Available: {list(STRATEGY_CONFIGS.keys())}"
        )

    return STRATEGY_CONFIGS[analysis_type]
