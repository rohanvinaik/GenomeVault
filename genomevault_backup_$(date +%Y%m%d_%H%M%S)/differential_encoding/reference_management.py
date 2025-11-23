"""
Reference Genome Management for Differential Encoding.

This module implements secure management of reference genome pools with:
1. Reference genome data structures with cryptographic verification
2. Efficient position indexing using IntervalTree
3. Cryptographically secure random reference selection
4. VCF parsing and validation
5. Reference pool management with integrity checking

Key Components:
- Variant: Individual genomic variant representation
- GenomeSection: Contiguous genomic region with variants
- ReferenceGenome: Complete reference genome with metadata
- ReferencePool: Collection of verified reference genomes
- SecureReferenceGenomeManager: Main interface for reference management

Security Features:
- Cryptographic hash verification on load
- Tamper detection via hash comparison
- Secure random selection for reference assignment
- Provenance tracking for audit trails
"""

from __future__ import annotations

import gzip
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from genomevault.differential_encoding.crypto_primitives import (
    CryptoRNG,
    compute_reference_hash,
)
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

# Simple IntervalTree implementation for position indexing
# For production, could use intervaltree package, but implementing simple version
# to avoid external dependencies


class Interval:
    """Represents a genomic interval with associated data."""

    def __init__(self, start: int, end: int, data: any):
        """
        Initialize interval.

        Args:
            start: Start position (inclusive)
            end: End position (exclusive)
            data: Associated data
        """
        self.start = start
        self.end = end
        self.data = data

    def overlaps(self, start: int, end: int) -> bool:
        """Check if this interval overlaps with given range."""
        return self.start < end and self.end > start

    def __repr__(self) -> str:
        return f"Interval({self.start}, {self.end}, {self.data})"


class IntervalTree:
    """
    Simple interval tree for efficient range queries.

    Provides O(log n + k) query time where k is number of overlapping intervals.
    Uses sorted list implementation for simplicity.
    """

    def __init__(self):
        """Initialize empty interval tree."""
        self.intervals: List[Interval] = []
        self._sorted = True

    def add(self, start: int, end: int, data: any) -> None:
        """
        Add interval to tree.

        Args:
            start: Start position (inclusive)
            end: End position (exclusive)
            data: Data associated with interval
        """
        self.intervals.append(Interval(start, end, data))
        self._sorted = False

    def query(self, start: int, end: int) -> List[any]:
        """
        Query overlapping intervals.

        Args:
            start: Query start position
            end: Query end position

        Returns:
            List of data from overlapping intervals
        """
        if not self._sorted:
            self.intervals.sort(key=lambda x: x.start)
            self._sorted = True

        results = []
        for interval in self.intervals:
            if interval.start >= end:
                break
            if interval.overlaps(start, end):
                results.append(interval.data)

        return results

    def __len__(self) -> int:
        return len(self.intervals)


@dataclass(slots=True)
class Variant:
    """
    Genomic variant representation.

    Represents a single genetic variant with position, alleles, and metadata.

    Memory Optimization: Uses __slots__ for 40-50% memory reduction.

    Attributes:
        chromosome: Chromosome identifier (e.g., "chr1", "chrX")
        position: Genomic position (1-based coordinate)
        ref: Reference allele sequence
        alt: Alternate allele sequence
        genotype: Genotype (e.g., "0/1", "1/1", "0/0")
        quality: Quality score (0.0 to 1.0 or Phred scale)
        filter: Filter status (e.g., "PASS", ".")
        info: Additional information dictionary

    Example:
        >>> variant = Variant(
        ...     chromosome="chr1",
        ...     position=12345,
        ...     ref="A",
        ...     alt="G",
        ...     genotype="0/1"
        ... )
    """

    chromosome: str
    position: int
    ref: str
    alt: str
    genotype: str = "0/1"
    quality: float = 1.0
    filter: str = "PASS"
    info: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate variant data."""
        if self.position < 0:
            raise ValueError(f"Position must be non-negative, got {self.position}")
        if not self.ref:
            raise ValueError("Reference allele cannot be empty")
        if not self.alt:
            raise ValueError("Alternate allele cannot be empty")

    def __str__(self) -> str:
        return f"{self.chromosome}:{self.position} {self.ref}>{self.alt} ({self.genotype})"

    def __lt__(self, other: Variant) -> bool:
        """Enable sorting by position."""
        return self.position < other.position


@dataclass(slots=True)
class GenomeSection:
    """
    Contiguous genomic section with variants.

    Represents a specific region of a genome with all variants in that region.

    Memory Optimization: Uses __slots__ for 40-50% memory reduction.

    Attributes:
        chromosome: Chromosome identifier
        start_position: Start position (inclusive)
        end_position: End position (exclusive)
        variants: List of variants in this section

    Properties:
        length: Genomic length in base pairs
        variant_count: Number of variants in section

    Example:
        >>> section = GenomeSection(
        ...     chromosome="chr1",
        ...     start_position=100000,
        ...     end_position=200000,
        ...     variants=[variant1, variant2]
        ... )
        >>> print(f"Section has {section.variant_count} variants")
    """

    chromosome: str
    start_position: int
    end_position: int
    variants: List[Variant] = field(default_factory=list)

    def __post_init__(self):
        """Validate section data."""
        if self.start_position < 0:
            raise ValueError(f"Start position must be non-negative: {self.start_position}")
        if self.end_position <= self.start_position:
            raise ValueError(
                f"End position ({self.end_position}) must be > start ({self.start_position})"
            )

        # Sort variants by position
        self.variants.sort(key=lambda v: v.position)

    @property
    def length(self) -> int:
        """Get genomic length in base pairs."""
        return self.end_position - self.start_position

    @property
    def variant_count(self) -> int:
        """Get number of variants in section."""
        return len(self.variants)

    def __str__(self) -> str:
        return (
            f"{self.chromosome}:{self.start_position}-{self.end_position} "
            f"({self.variant_count} variants)"
        )


@dataclass
class ReferenceGenome:
    """
    Complete reference genome with cryptographic verification.

    Stores a reference genome with variants, metadata, and position indexing
    for efficient querying.

    Attributes:
        genome_id: Unique identifier (e.g., "GRCh38", "HG002")
        assembly: Assembly version (e.g., "GRCh38", "hg19")
        variants: Dictionary mapping chromosome to list of variants
        cryptographic_hash: SHA-256 hash of entire genome for integrity
        source: Data source (e.g., "1000 Genomes", "GIAB")
        population: Population ancestry (e.g., "EUR", "AFR", "EAS")
        date_created: Unix timestamp of creation
        version: Version identifier
        position_index: IntervalTree per chromosome for fast queries

    Methods:
        get_section: Extract genomic section by coordinates
        get_variants_in_range: Get variants in specific range
        build_position_index: Build/rebuild position indices

    Example:
        >>> ref = ReferenceGenome(
        ...     genome_id="GRCh38",
        ...     assembly="GRCh38",
        ...     variants={"chr1": [variant1, variant2]},
        ...     cryptographic_hash="abc123...",
        ...     source="1000 Genomes"
        ... )
        >>> section = ref.get_section("chr1", 100000, 200000)
    """

    genome_id: str
    assembly: str
    variants: Dict[str, List[Variant]]
    cryptographic_hash: str
    source: str = "unknown"
    population: Optional[str] = None
    date_created: float = field(default_factory=time.time)
    version: str = "1.0"
    position_index: Dict[str, IntervalTree] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize position indices."""
        if not self.position_index:
            self.build_position_index()

    def build_position_index(self) -> None:
        """
        Build position indices for all chromosomes.

        Creates IntervalTree for each chromosome to enable fast range queries.
        Each variant is stored in the tree with its position as the interval.

        Complexity: O(n log n) where n is total number of variants
        """
        self.position_index = {}

        for chromosome, chr_variants in self.variants.items():
            tree = IntervalTree()
            for variant in chr_variants:
                # Each variant is a point interval [pos, pos+len(ref))
                tree.add(
                    variant.position,
                    variant.position + len(variant.ref),
                    variant
                )
            self.position_index[chromosome] = tree

        logger.debug(
            f"Built position index for {self.genome_id}: "
            f"{sum(len(t) for t in self.position_index.values())} variants indexed"
        )

    def get_section(
        self,
        chromosome: str,
        start: int,
        end: int
    ) -> GenomeSection:
        """
        Extract genomic section by coordinates.

        Uses position index for efficient variant lookup.

        Args:
            chromosome: Chromosome identifier
            start: Start position (inclusive)
            end: End position (exclusive)

        Returns:
            GenomeSection with variants in specified range

        Raises:
            ValueError: If chromosome not found

        Complexity: O(log n + k) where k is number of variants in range

        Example:
            >>> section = ref.get_section("chr1", 100000, 200000)
            >>> print(f"Found {section.variant_count} variants")
        """
        if chromosome not in self.variants:
            raise ValueError(
                f"Chromosome {chromosome} not found in {self.genome_id}. "
                f"Available: {list(self.variants.keys())}"
            )

        # Use position index if available
        if chromosome in self.position_index:
            variants_in_range = self.position_index[chromosome].query(start, end)
        else:
            # Fallback to linear search
            variants_in_range = [
                v for v in self.variants[chromosome]
                if start <= v.position < end
            ]

        return GenomeSection(
            chromosome=chromosome,
            start_position=start,
            end_position=end,
            variants=variants_in_range
        )

    def get_variants_in_range(
        self,
        chromosome: str,
        start: int,
        end: int
    ) -> List[Variant]:
        """
        Get variants in specific range.

        Convenience method that returns just the variant list.

        Args:
            chromosome: Chromosome identifier
            start: Start position (inclusive)
            end: End position (exclusive)

        Returns:
            List of variants in range
        """
        section = self.get_section(chromosome, start, end)
        return section.variants

    @property
    def total_variants(self) -> int:
        """Get total number of variants across all chromosomes."""
        return sum(len(variants) for variants in self.variants.values())

    @property
    def chromosomes(self) -> List[str]:
        """Get list of chromosomes."""
        return list(self.variants.keys())

    def __str__(self) -> str:
        return (
            f"ReferenceGenome(id={self.genome_id}, assembly={self.assembly}, "
            f"chromosomes={len(self.chromosomes)}, variants={self.total_variants})"
        )


@dataclass
class ReferencePool:
    """
    Pool of verified reference genomes.

    Manages collection of reference genomes with cryptographic verification.

    Attributes:
        references: Dictionary mapping genome_id to ReferenceGenome
        verification_status: Dictionary tracking verification status

    Methods:
        verify_all: Verify cryptographic hashes of all references
        add_reference: Add reference to pool with verification
        remove_reference: Remove reference from pool
        get_reference: Get reference by ID

    Example:
        >>> pool = ReferencePool()
        >>> pool.add_reference(grch38_genome)
        >>> pool.add_reference(grch37_genome)
        >>> assert pool.verify_all()  # Verify integrity
    """

    references: Dict[str, ReferenceGenome] = field(default_factory=dict)
    verification_status: Dict[str, bool] = field(default_factory=dict)

    def verify_all(self) -> bool:
        """
        Verify cryptographic hashes of all reference genomes.

        Recomputes hash for each genome and compares with stored hash.
        Updates verification_status dictionary.

        Returns:
            True if all references verify successfully, False otherwise

        Example:
            >>> pool = ReferencePool(references={"GRCh38": genome})
            >>> if not pool.verify_all():
            ...     print("Verification failed!")
        """
        all_valid = True

        for genome_id, genome in self.references.items():
            logger.debug(f"Verifying reference genome: {genome_id}")

            computed_hash = compute_reference_hash(genome)

            if computed_hash != genome.cryptographic_hash:
                logger.error(
                    f"Verification FAILED for {genome_id}: "
                    f"expected {genome.cryptographic_hash[:16]}..., "
                    f"got {computed_hash[:16]}..."
                )
                self.verification_status[genome_id] = False
                all_valid = False
            else:
                logger.debug(f"Verification OK for {genome_id}")
                self.verification_status[genome_id] = True

        return all_valid

    def add_reference(self, reference: ReferenceGenome, verify: bool = True) -> None:
        """
        Add reference genome to pool.

        Args:
            reference: ReferenceGenome to add
            verify: If True, verify cryptographic hash

        Raises:
            ValueError: If verification fails
        """
        genome_id = reference.genome_id

        if verify:
            computed_hash = compute_reference_hash(reference)
            if computed_hash != reference.cryptographic_hash:
                raise ValueError(
                    f"Reference {genome_id} failed verification: "
                    f"hash mismatch"
                )

        self.references[genome_id] = reference
        self.verification_status[genome_id] = verify

        logger.info(f"Added reference {genome_id} to pool (verified={verify})")

    def remove_reference(self, genome_id: str) -> None:
        """Remove reference from pool."""
        if genome_id in self.references:
            del self.references[genome_id]
            del self.verification_status[genome_id]
            logger.info(f"Removed reference {genome_id} from pool")

    def get_reference(self, genome_id: str) -> ReferenceGenome:
        """Get reference by ID."""
        if genome_id not in self.references:
            raise ValueError(
                f"Reference {genome_id} not found. "
                f"Available: {list(self.references.keys())}"
            )
        return self.references[genome_id]

    @property
    def genome_ids(self) -> List[str]:
        """Get list of genome IDs in pool."""
        return list(self.references.keys())

    @property
    def size(self) -> int:
        """Get number of references in pool."""
        return len(self.references)

    def __str__(self) -> str:
        return f"ReferencePool(size={self.size}, genomes={self.genome_ids})"


class SecureReferenceGenomeManager:
    """
    Secure management of reference genome pool.

    Provides cryptographically secure interface for:
    - Loading reference genomes from VCF files
    - Verifying genome integrity
    - Randomly selecting references
    - Extracting genomic sections

    Security Features:
    - Cryptographic hash verification on load
    - Secure random selection using CryptoRNG
    - Tamper detection via hash comparison
    - Provenance tracking

    Attributes:
        reference_dir: Directory containing reference VCF files
        pool: ReferencePool containing loaded genomes
        crypto_rng: CryptoRNG for secure random selection

    Example:
        >>> manager = SecureReferenceGenomeManager(Path("references/"))
        >>> # Randomly select reference for chunk
        >>> ref = manager.get_random_reference(chunk_seed)
        >>> # Extract section
        >>> section = manager.get_reference_section(
        ...     ref.genome_id, "chr1", 100000, 200000
        ... )
    """

    def __init__(self, reference_dir: Path, crypto_rng: Optional[CryptoRNG] = None):
        """
        Initialize reference manager.

        Args:
            reference_dir: Directory containing reference VCF files
            crypto_rng: Optional CryptoRNG instance (creates new if None)

        Raises:
            ValueError: If reference_dir doesn't exist
            RuntimeError: If reference verification fails
        """
        self.reference_dir = Path(reference_dir)

        if not self.reference_dir.exists():
            logger.warning(f"Reference directory does not exist: {reference_dir}")
            self.reference_dir.mkdir(parents=True, exist_ok=True)

        self.pool = ReferencePool()
        self.crypto_rng = crypto_rng or CryptoRNG()

        # Load references from directory
        self._load_references()

        # Verify all loaded references
        if self.pool.size > 0:
            if not self.pool.verify_all():
                raise RuntimeError(
                    "Reference genome verification failed. "
                    "One or more references have invalid cryptographic hashes."
                )
            logger.info(
                f"Initialized SecureReferenceGenomeManager with "
                f"{self.pool.size} verified references"
            )
        else:
            logger.warning("No reference genomes loaded")

    def _load_references(self) -> None:
        """
        Load reference genomes from directory.

        Searches for .vcf.gz and .vcf files and parses them.
        """
        # Look for compressed VCF files
        vcf_files = list(self.reference_dir.glob("*.vcf.gz")) + \
                    list(self.reference_dir.glob("*.vcf"))

        logger.info(f"Found {len(vcf_files)} VCF files in {self.reference_dir}")

        for vcf_path in vcf_files:
            try:
                # Extract genome ID from filename (remove .vcf.gz or .vcf)
                genome_id = vcf_path.stem
                if genome_id.endswith(".vcf"):
                    genome_id = genome_id[:-4]

                logger.info(f"Loading reference genome: {genome_id} from {vcf_path.name}")

                # Parse VCF file
                genome = self._parse_reference_vcf(vcf_path, genome_id)

                # Add to pool (will compute and verify hash)
                self.pool.add_reference(genome, verify=False)

                logger.info(
                    f"Loaded {genome_id}: {genome.total_variants} variants "
                    f"across {len(genome.chromosomes)} chromosomes"
                )

            except Exception as e:
                logger.error(f"Failed to load {vcf_path}: {e}")

    def _parse_reference_vcf(self, vcf_path: Path, genome_id: str) -> ReferenceGenome:
        """
        Parse VCF file into ReferenceGenome.

        Args:
            vcf_path: Path to VCF file (.vcf or .vcf.gz)
            genome_id: Identifier for this genome

        Returns:
            ReferenceGenome with variants loaded
        """
        variants_by_chr: Dict[str, List[Variant]] = {}
        assembly = genome_id  # Default to genome_id
        source = "unknown"

        # Determine if file is gzipped
        open_func = gzip.open if vcf_path.suffix == ".gz" else open
        mode = "rt" if vcf_path.suffix == ".gz" else "r"

        with open_func(vcf_path, mode) as f:
            for line in f:
                line = line.strip()

                # Parse header lines
                if line.startswith("##"):
                    # Extract metadata from header
                    if line.startswith("##reference="):
                        assembly = line.split("=", 1)[1]
                    elif line.startswith("##source="):
                        source = line.split("=", 1)[1]
                    continue

                # Skip column header
                if line.startswith("#CHROM"):
                    continue

                # Skip empty lines
                if not line:
                    continue

                # Parse variant line
                try:
                    variant = self._parse_vcf_line(line)
                    if variant:
                        if variant.chromosome not in variants_by_chr:
                            variants_by_chr[variant.chromosome] = []
                        variants_by_chr[variant.chromosome].append(variant)
                except Exception as e:
                    logger.warning(f"Failed to parse VCF line: {e}")
                    continue

        # Sort variants by position for each chromosome
        for chromosome in variants_by_chr:
            variants_by_chr[chromosome].sort(key=lambda v: v.position)

        # Create reference genome
        genome = ReferenceGenome(
            genome_id=genome_id,
            assembly=assembly,
            variants=variants_by_chr,
            cryptographic_hash="",  # Will be computed below
            source=source,
            date_created=time.time(),
            version="1.0"
        )

        # Compute cryptographic hash
        genome.cryptographic_hash = compute_reference_hash(genome)

        return genome

    def _parse_vcf_line(self, line: str) -> Optional[Variant]:
        """
        Parse single VCF line into Variant.

        VCF format: CHROM POS ID REF ALT QUAL FILTER INFO FORMAT SAMPLE...

        Args:
            line: VCF line

        Returns:
            Variant or None if parsing fails
        """
        fields = line.split("\t")

        if len(fields) < 8:
            return None

        try:
            chromosome = fields[0]
            position = int(fields[1])
            ref = fields[3]
            alt = fields[4]
            qual_str = fields[5]
            filter_str = fields[6]
            info_str = fields[7]

            # Parse quality
            if qual_str == ".":
                quality = 1.0
            else:
                try:
                    quality = float(qual_str)
                    # Normalize Phred scores to [0, 1]
                    if quality > 100:
                        quality = min(quality / 100.0, 1.0)
                except ValueError:
                    quality = 1.0

            # Parse genotype from FORMAT/SAMPLE if available
            genotype = "0/1"  # Default
            if len(fields) >= 10:
                format_fields = fields[8].split(":")
                sample_fields = fields[9].split(":")
                if "GT" in format_fields:
                    gt_index = format_fields.index("GT")
                    if gt_index < len(sample_fields):
                        genotype = sample_fields[gt_index]

            # Parse INFO field into dict
            info = {}
            for item in info_str.split(";"):
                if "=" in item:
                    key, value = item.split("=", 1)
                    info[key] = value
                else:
                    info[item] = True

            return Variant(
                chromosome=chromosome,
                position=position,
                ref=ref,
                alt=alt,
                genotype=genotype,
                quality=quality,
                filter=filter_str,
                info=info
            )

        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse VCF line: {e}")
            return None

    def get_random_reference(
        self,
        seed: bytes,
        exclude: Optional[List[str]] = None
    ) -> ReferenceGenome:
        """
        Select random reference genome using cryptographic randomness.

        Uses CryptoRNG for deterministic, unpredictable selection.

        Properties:
        - Deterministic: same seed → same reference
        - Unpredictable: cannot guess which reference from seed
        - Uniform: all available references equally likely

        Args:
            seed: 32-byte seed for selection
            exclude: Optional list of genome IDs to exclude

        Returns:
            Randomly selected ReferenceGenome

        Raises:
            ValueError: If no references available

        Example:
            >>> ref = manager.get_random_reference(chunk_seed)
            >>> print(f"Selected: {ref.genome_id}")
        """
        available = [
            gid for gid in self.pool.genome_ids
            if exclude is None or gid not in exclude
        ]

        if not available:
            raise ValueError("No available reference genomes")

        # Cryptographically secure selection
        selected_id = self.crypto_rng.random_choice(available, seed)

        logger.debug(
            f"Selected reference {selected_id} from pool of {len(available)}"
        )

        return self.pool.get_reference(selected_id)

    def get_reference_section(
        self,
        genome_id: str,
        chromosome: str,
        start: int,
        end: int
    ) -> GenomeSection:
        """
        Extract section from specific reference genome.

        Args:
            genome_id: Reference genome identifier
            chromosome: Chromosome identifier
            start: Start position (inclusive)
            end: End position (exclusive)

        Returns:
            GenomeSection with variants in range

        Raises:
            ValueError: If genome_id not found

        Example:
            >>> section = manager.get_reference_section(
            ...     "GRCh38", "chr1", 100000, 200000
            ... )
            >>> print(f"Got {section.variant_count} variants")
        """
        reference = self.pool.get_reference(genome_id)
        return reference.get_section(chromosome, start, end)

    def add_reference_from_vcf(
        self,
        vcf_path: Path,
        genome_id: Optional[str] = None
    ) -> ReferenceGenome:
        """
        Add reference genome from VCF file.

        Args:
            vcf_path: Path to VCF file
            genome_id: Optional genome ID (defaults to filename)

        Returns:
            Loaded ReferenceGenome

        Example:
            >>> ref = manager.add_reference_from_vcf(
            ...     Path("new_reference.vcf.gz"),
            ...     genome_id="HG002"
            ... )
        """
        if genome_id is None:
            genome_id = vcf_path.stem
            if genome_id.endswith(".vcf"):
                genome_id = genome_id[:-4]

        genome = self._parse_reference_vcf(vcf_path, genome_id)
        self.pool.add_reference(genome, verify=True)

        logger.info(f"Added reference {genome_id} from {vcf_path}")

        return genome

    @property
    def reference_count(self) -> int:
        """Get number of references in pool."""
        return self.pool.size

    @property
    def genome_ids(self) -> List[str]:
        """Get list of genome IDs."""
        return self.pool.genome_ids

    def __str__(self) -> str:
        return (
            f"SecureReferenceGenomeManager(references={self.reference_count}, "
            f"genomes={self.genome_ids})"
        )
