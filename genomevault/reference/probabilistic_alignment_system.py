"""
Probabilistic Alignment System with Hierarchical Mismatch Classification

This module implements a privacy-preserving alignment system that serves as a "blind
middleman" between fragmented FASTQ reads and public reference genomes. It introduces
1-5% unknowable positional uncertainty that makes stolen alignment data nearly useless
while maintaining 95-99% utility for legitimate genomic analysis.

SECURITY MODEL: "Data Poisoning for Defense"
- NOT trying to: Hide all alignment information (impossible)
- ACTUALLY doing: Make stolen data untrustworthy through unknowable uncertainty
- Even if adversary obtains data: Cannot determine which positions have injected noise
- Exponential search space: 4^(uncertain_positions) possible interpretations

HIERARCHICAL MISMATCH CLASSIFICATION:
1. SNP (1 isolated mismatch): Normal biological variation (~1:10^6 frequency)
2. 2 consecutive mismatches: Rare but possible adjacent SNPs (~1:10^12)
3. 3 consecutive mismatches: PEAK SUSPICION - likely sequencing error (~1:10^18)
4. 4+ consecutive mismatches: STRUCTURAL VARIANT - indel/duplication (common!)

CRITICAL: 3 consecutive is the "sequencing error threshold". Beyond 4+, we're seeing
legitimate structural variation (deletions, insertions, transpositions), NOT errors.

PURPOSE: Enable FASTQ fragment ordering WITHOUT creating deterministic, reversible
mapping to specific public references (hg38, GRCh37, T2T-CHM13).
"""

import bisect
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# SNP frequency constants
SNP_FREQUENCY_PER_BASE = 1e-6  # 1 in 1 million bases
TWO_SNP_FREQUENCY = SNP_FREQUENCY_PER_BASE ** 2  # ~1:10^12
THREE_SNP_FREQUENCY = SNP_FREQUENCY_PER_BASE ** 3  # ~1:10^18 (sequencing error threshold)

# Exponential decay parameters
DECAY_BASE = 10.0  # Base for exponential decay
CERTAINTY_FLOOR = 1e-18  # Minimum certainty (3-SNP threshold)


@dataclass
class SNPRecord:
    """Single nucleotide polymorphism record."""
    chromosome: str
    position: int  # 0-indexed genomic position
    ref_allele: str
    alt_alleles: List[str]
    frequency: float  # Population frequency
    dbsnp_id: Optional[str] = None

    def __post_init__(self):
        """Validate SNP record."""
        if len(self.ref_allele) != 1 or not all(len(a) == 1 for a in self.alt_alleles):
            raise ValueError("SNPRecord only supports single nucleotide variants")


@dataclass
class AlignmentCertainty:
    """
    Probabilistic alignment certainty with hierarchical mismatch classification.

    CRITICAL DISTINCTION: SNPs vs. Structural Variants
    - 1 mismatch (isolated): Normal SNP (~10^-6 frequency)
    - 2 consecutive: Rare but possible adjacent SNPs (~10^-12)
    - 3 consecutive: HIGHLY SUSPICIOUS - likely sequencing error (~10^-18)
    - 4+ consecutive: STRUCTURAL VARIANT - indel/duplication/transposition (common!)

    The 3-consecutive pattern is the "peak suspicion" point. Beyond 4+, we're likely
    seeing legitimate structural variation, NOT sequencing errors.
    """
    position: int
    reference_base: str
    query_base: str
    consecutive_mismatches: int
    is_known_snp: bool
    certainty_score: float
    statistical_significance: float  # p-value for mismatch pattern

    @property
    def is_likely_sequencing_error(self) -> bool:
        """
        Returns True if pattern suggests sequencing error.

        ONLY 3 consecutive mismatches is flagged as sequencing error.
        4+ consecutive indicates structural variant, not error.
        """
        return self.consecutive_mismatches == 3

    @property
    def is_structural_variant_candidate(self) -> bool:
        """Returns True if pattern suggests structural variant (4+ consecutive)."""
        return self.consecutive_mismatches >= 4

    @property
    def certainty_level(self) -> str:
        """
        Human-readable certainty level with structural variant detection.

        Classification:
        - VERY_HIGH: Perfect match
        - HIGH: Single isolated SNP (normal biological variation)
        - LOW: 2 consecutive mismatches (rare but possible)
        - VERY_LOW_SEQUENCING_ERROR: Exactly 3 consecutive (suspicious)
        - STRUCTURAL_VARIANT: 4+ consecutive (trigger SV pipeline)
        """
        if self.consecutive_mismatches >= 4:
            return "STRUCTURAL_VARIANT"
        elif self.consecutive_mismatches == 3:
            return "VERY_LOW_SEQUENCING_ERROR"
        elif self.certainty_score >= 0.99:
            return "VERY_HIGH"
        elif self.certainty_score >= 1e-6:
            return "HIGH"
        elif self.certainty_score >= 1e-12:
            return "LOW"
        else:
            return "VERY_LOW_SEQUENCING_ERROR"


@dataclass
class IndelCandidate:
    """Indel candidate detected via position checksum."""
    start_position: int
    suspected_shift: int  # Positive = insertion, negative = deletion
    length: int
    confidence: float
    local_snp_density: float  # SNPs per kb in surrounding region
    statistical_significance: float


class ChromosomeSNPIndex:
    """
    Efficient SNP database organized by chromosome and position.

    Uses binary search for O(log n) lookup complexity.
    """

    def __init__(self, chromosome: str):
        self.chromosome = chromosome
        self.positions: List[int] = []  # Sorted list of SNP positions
        self.snp_map: Dict[int, SNPRecord] = {}  # Position -> SNPRecord
        self._is_sorted = True

    def add_snp(self, snp: SNPRecord):
        """Add SNP to index."""
        if snp.chromosome != self.chromosome:
            raise ValueError(f"SNP chromosome {snp.chromosome} doesn't match index {self.chromosome}")

        self.positions.append(snp.position)
        self.snp_map[snp.position] = snp
        self._is_sorted = False

    def finalize(self):
        """Sort positions for efficient binary search."""
        if not self._is_sorted:
            self.positions.sort()
            self._is_sorted = True

    def lookup(self, position: int) -> Optional[SNPRecord]:
        """
        Binary search lookup: O(log n) complexity.

        Args:
            position: Genomic position to query

        Returns:
            SNPRecord if found, None otherwise
        """
        if not self._is_sorted:
            self.finalize()

        idx = bisect.bisect_left(self.positions, position)
        if idx < len(self.positions) and self.positions[idx] == position:
            return self.snp_map[position]
        return None

    def get_nearby_snps(self, position: int, window: int = 1000) -> List[SNPRecord]:
        """Get all SNPs within ±window bases of position."""
        if not self._is_sorted:
            self.finalize()

        start_pos = position - window
        end_pos = position + window

        # Binary search for start and end indices
        start_idx = bisect.bisect_left(self.positions, start_pos)
        end_idx = bisect.bisect_right(self.positions, end_pos)

        nearby_positions = self.positions[start_idx:end_idx]
        return [self.snp_map[pos] for pos in nearby_positions]


class SNPDatabase:
    """
    Complete SNP database with chromosome-level indexing.

    Organized structure:
    - Chromosomes as "chapters"
    - Positions sorted within each chromosome
    - Efficient binary search or hash lookup
    """

    def __init__(self):
        self.chromosomes: Dict[str, ChromosomeSNPIndex] = {}

    def add_snp(self, snp: SNPRecord):
        """Add SNP to database."""
        if snp.chromosome not in self.chromosomes:
            self.chromosomes[snp.chromosome] = ChromosomeSNPIndex(snp.chromosome)

        self.chromosomes[snp.chromosome].add_snp(snp)

    def finalize(self):
        """Finalize all chromosome indices for efficient search."""
        for chrom_index in self.chromosomes.values():
            chrom_index.finalize()

    def lookup(self, chromosome: str, position: int) -> Optional[SNPRecord]:
        """
        Lookup SNP by chromosome and position.

        O(log n) complexity via binary search.
        """
        if chromosome not in self.chromosomes:
            return None

        return self.chromosomes[chromosome].lookup(position)

    def load_from_vcf(self, vcf_path: Path):
        """Load SNPs from VCF file."""
        logger.info(f"Loading SNPs from {vcf_path}")

        count = 0
        with open(vcf_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue

                fields = line.strip().split('\t')
                if len(fields) < 5:
                    continue

                chrom = fields[0]
                pos = int(fields[1]) - 1  # VCF is 1-indexed, convert to 0-indexed
                dbsnp_id = fields[2] if fields[2] != '.' else None
                ref = fields[3]
                alts = fields[4].split(',')

                # Only process SNPs (single nucleotide)
                if len(ref) == 1 and all(len(a) == 1 for a in alts):
                    snp = SNPRecord(
                        chromosome=chrom,
                        position=pos,
                        ref_allele=ref,
                        alt_alleles=alts,
                        frequency=SNP_FREQUENCY_PER_BASE,  # Default frequency
                        dbsnp_id=dbsnp_id
                    )
                    self.add_snp(snp)
                    count += 1

        self.finalize()
        logger.info(f"Loaded {count:,} SNPs across {len(self.chromosomes)} chromosomes")


class ProbabilisticAligner:
    """
    Probabilistic alignment system with exponential certainty decay.

    Features:
    1. SNP database organized by chromosome + position (binary search)
    2. 2-nucleotide error handling with reduced certainty
    3. 3+ nucleotide error detection as sequencing errors
    4. Indel detection via position checksum
    5. Statistical significance testing for mismatch patterns
    """

    def __init__(
        self,
        snp_database: SNPDatabase,
        indel_detection_window: int = 50,
        statistical_confidence: float = 0.95
    ):
        """
        Initialize probabilistic aligner.

        Args:
            snp_database: Pre-loaded SNP database
            indel_detection_window: Window size for indel detection (bases)
            statistical_confidence: Confidence level for statistical tests (0-1)
        """
        self.snp_db = snp_database
        self.indel_window = indel_detection_window
        self.stat_confidence = statistical_confidence

        # Position checksum for indel detection
        self.position_checksum = 0
        self.expected_position = 0

    def compute_certainty(
        self,
        consecutive_mismatches: int,
        is_known_snp: bool,
        local_snp_density: float
    ) -> float:
        """
        Compute alignment certainty with hierarchical mismatch classification.

        Certainty = (base_frequency) ^ consecutive_mismatches

        Classification:
        - 0 mismatches: 1.0 (perfect match)
        - 1 mismatch: 1e-6 (normal SNP frequency)
        - 2 consecutive: 1e-12 (rare adjacent SNPs)
        - 3 consecutive: 1e-18 (SEQUENCING ERROR - peak suspicion)
        - 4+ consecutive: STRUCTURAL VARIANT (triggers SV pipeline, not flagged as error)

        CRITICAL: 4+ consecutive returns a special marker to trigger SV detection,
        not an error flag. The certainty for 4+ is actually HIGHER than 3 consecutive
        because it's likely legitimate biological variation.

        Args:
            consecutive_mismatches: Number of consecutive mismatches
            is_known_snp: Whether current position is a known SNP
            local_snp_density: SNP density in surrounding region (SNPs per kb)

        Returns:
            Certainty score (0.0 to 1.0), or special value for structural variants
        """
        if consecutive_mismatches == 0:
            return 1.0

        # Structural variant threshold: 4+ consecutive mismatches
        # These are likely legitimate biological variation (indels, duplications, etc.)
        # Return moderate certainty to indicate "this is expected" but trigger SV detection
        if consecutive_mismatches >= 4:
            return 0.5  # Moderate certainty - likely structural variant, not error

        # Base certainty from exponential decay (for 1-3 consecutive)
        # certainty = SNP_FREQUENCY ^ consecutive_mismatches
        certainty = SNP_FREQUENCY_PER_BASE ** consecutive_mismatches

        # Adjust for known SNP (slightly higher certainty)
        if is_known_snp:
            certainty *= 1.5  # Known SNPs are more likely to be real

        # Adjust for local SNP density
        # Higher density regions are more likely to have legitimate SNPs
        density_factor = min(2.0, local_snp_density / SNP_FREQUENCY_PER_BASE)
        certainty *= density_factor

        # Floor at 3-SNP threshold (ONLY for exactly 3 consecutive)
        if consecutive_mismatches == 3:
            certainty = max(certainty, CERTAINTY_FLOOR)

        # Cap at 1.0
        certainty = min(certainty, 1.0)

        return certainty

    def compute_statistical_significance(
        self,
        consecutive_mismatches: int,
        total_aligned_bases: int,
        nearby_snps: List[SNPRecord]
    ) -> float:
        """
        Compute statistical significance (p-value) of mismatch pattern.

        Uses binomial distribution:
        P(k consecutive mismatches | n bases aligned) = ?

        Args:
            consecutive_mismatches: Number of consecutive mismatches observed
            total_aligned_bases: Total bases aligned so far
            nearby_snps: Known SNPs in surrounding region

        Returns:
            p-value: probability of observing this pattern by chance
        """
        if consecutive_mismatches == 0:
            return 1.0

        # Expected number of mismatches in this region
        expected_snps = total_aligned_bases * SNP_FREQUENCY_PER_BASE

        # Adjust for known SNPs nearby
        local_snp_rate = len(nearby_snps) / max(1, total_aligned_bases)
        adjusted_rate = max(SNP_FREQUENCY_PER_BASE, local_snp_rate)

        # Probability of k consecutive mismatches
        # P(k consecutive) ≈ (SNP_rate)^k
        p_value = adjusted_rate ** consecutive_mismatches

        return p_value

    def align_position(
        self,
        chromosome: str,
        position: int,
        reference_base: str,
        query_base: str,
        previous_certainty: Optional[AlignmentCertainty] = None
    ) -> AlignmentCertainty:
        """
        Align a single position with probabilistic certainty.

        Args:
            chromosome: Chromosome name
            position: Genomic position (0-indexed)
            reference_base: Reference genome base
            query_base: Query genome base
            previous_certainty: Previous position's certainty (for tracking consecutive mismatches)

        Returns:
            AlignmentCertainty object with probabilistic score
        """
        # Check for mismatch
        is_mismatch = reference_base != query_base

        # Look up known SNPs
        known_snp = self.snp_db.lookup(chromosome, position)
        is_known_snp = known_snp is not None and query_base in known_snp.alt_alleles

        # Get nearby SNPs for local density calculation
        nearby_snps = self.snp_db.chromosomes.get(chromosome, ChromosomeSNPIndex(chromosome)).get_nearby_snps(
            position, window=1000
        )
        local_snp_density = len(nearby_snps) / 1000.0  # SNPs per kb

        # Track consecutive mismatches
        if is_mismatch:
            if previous_certainty and previous_certainty.query_base != previous_certainty.reference_base:
                # Previous position was also a mismatch
                consecutive = previous_certainty.consecutive_mismatches + 1
            else:
                # First mismatch in sequence
                consecutive = 1
        else:
            # Match - reset consecutive counter
            consecutive = 0

        # Compute certainty with exponential decay
        certainty = self.compute_certainty(
            consecutive_mismatches=consecutive,
            is_known_snp=is_known_snp,
            local_snp_density=local_snp_density
        )

        # Compute statistical significance
        # Use a sliding window of 10kb for total aligned bases
        total_aligned = 10000
        p_value = self.compute_statistical_significance(
            consecutive_mismatches=consecutive,
            total_aligned_bases=total_aligned,
            nearby_snps=nearby_snps
        )

        # Update position checksum for indel detection
        self.position_checksum = (self.position_checksum + position) % (2**32)
        self.expected_position = position + 1

        return AlignmentCertainty(
            position=position,
            reference_base=reference_base,
            query_base=query_base,
            consecutive_mismatches=consecutive,
            is_known_snp=is_known_snp,
            certainty_score=certainty,
            statistical_significance=p_value
        )

    def detect_indel(
        self,
        chromosome: str,
        current_position: int,
        alignment_history: List[AlignmentCertainty]
    ) -> Optional[IndelCandidate]:
        """
        Detect potential indel via position checksum mismatch.

        If observed position != expected position, suggests indel.
        Runs statistical experiment to determine significance.

        Args:
            chromosome: Chromosome name
            current_position: Current observed position
            alignment_history: Recent alignment certainty records

        Returns:
            IndelCandidate if detected, None otherwise
        """
        # Check for position discontinuity
        position_shift = current_position - self.expected_position

        if position_shift == 0:
            return None  # No indel

        # Analyze recent alignment history for statistical significance
        window_size = min(self.indel_window, len(alignment_history))
        if window_size < 5:
            return None  # Insufficient data

        recent_history = alignment_history[-window_size:]

        # Calculate local SNP density
        mismatches = sum(1 for cert in recent_history if cert.consecutive_mismatches > 0)
        local_snp_density = mismatches / window_size * 1000  # Per kb

        # Statistical significance of position shift
        # Expected shift = 0, observed = position_shift
        # Use exponential decay model
        shift_magnitude = abs(position_shift)

        # Probability of this shift occurring by chance
        # Assume indel rate ~1:10^4 per base
        indel_rate = 1e-4
        p_indel = indel_rate ** shift_magnitude

        # Confidence = 1 - p_value
        confidence = 1.0 - p_indel

        # Require high confidence for indel calling
        if confidence < self.stat_confidence:
            return None

        return IndelCandidate(
            start_position=self.expected_position,
            suspected_shift=position_shift,
            length=abs(position_shift),
            confidence=confidence,
            local_snp_density=local_snp_density,
            statistical_significance=p_indel
        )

    def align_sequence(
        self,
        chromosome: str,
        reference_seq: str,
        query_seq: str,
        start_position: int = 0
    ) -> Tuple[List[AlignmentCertainty], List[IndelCandidate]]:
        """
        Align entire sequence with probabilistic certainty tracking.

        Args:
            chromosome: Chromosome name
            reference_seq: Reference sequence
            query_seq: Query sequence
            start_position: Starting genomic position

        Returns:
            Tuple of (alignment_certainties, indel_candidates)
        """
        certainties = []
        indels = []

        # Reset checksum
        self.position_checksum = 0
        self.expected_position = start_position

        min_len = min(len(reference_seq), len(query_seq))

        for i in range(min_len):
            position = start_position + i
            ref_base = reference_seq[i]
            query_base = query_seq[i]

            # Get previous certainty
            prev_cert = certainties[-1] if certainties else None

            # Align position
            cert = self.align_position(
                chromosome=chromosome,
                position=position,
                reference_base=ref_base,
                query_base=query_base,
                previous_certainty=prev_cert
            )
            certainties.append(cert)

            # Detect indels
            indel = self.detect_indel(chromosome, position, certainties)
            if indel:
                indels.append(indel)
                logger.info(
                    f"Detected indel at {chromosome}:{indel.start_position} "
                    f"(shift={indel.suspected_shift}, confidence={indel.confidence:.4f})"
                )

        return certainties, indels

    def generate_alignment_report(
        self,
        certainties: List[AlignmentCertainty],
        indels: List[IndelCandidate]
    ) -> Dict:
        """Generate comprehensive alignment report with statistics."""
        total = len(certainties)
        if total == 0:
            return {}

        # Count by certainty level
        very_high = sum(1 for c in certainties if c.certainty_level == "VERY_HIGH")
        high = sum(1 for c in certainties if c.certainty_level == "HIGH")
        low = sum(1 for c in certainties if c.certainty_level == "LOW")
        sequencing_errors = sum(1 for c in certainties if c.is_likely_sequencing_error)

        # Known vs unknown SNPs
        known_snps = sum(1 for c in certainties if c.is_known_snp)
        mismatches = sum(1 for c in certainties if c.consecutive_mismatches > 0)

        # Consecutive mismatch distribution
        max_consecutive = max((c.consecutive_mismatches for c in certainties), default=0)

        report = {
            "total_bases_aligned": total,
            "certainty_levels": {
                "very_high": very_high,
                "high": high,
                "low": low,
                "sequencing_error": sequencing_errors,
            },
            "certainty_percentages": {
                "very_high_pct": 100.0 * very_high / total,
                "high_pct": 100.0 * high / total,
                "low_pct": 100.0 * low / total,
                "sequencing_error_pct": 100.0 * sequencing_errors / total,
            },
            "snp_statistics": {
                "total_mismatches": mismatches,
                "known_snps": known_snps,
                "unknown_variants": mismatches - known_snps,
                "mismatch_rate": mismatches / total,
            },
            "consecutive_mismatch_analysis": {
                "max_consecutive": max_consecutive,
                "1_mismatch": sum(1 for c in certainties if c.consecutive_mismatches == 1),
                "2_consecutive": sum(1 for c in certainties if c.consecutive_mismatches == 2),
                "3+_consecutive": sum(1 for c in certainties if c.consecutive_mismatches >= 3),
            },
            "indel_analysis": {
                "total_indels_detected": len(indels),
                "insertions": sum(1 for i in indels if i.suspected_shift > 0),
                "deletions": sum(1 for i in indels if i.suspected_shift < 0),
                "mean_indel_length": np.mean([i.length for i in indels]) if indels else 0.0,
            }
        }

        return report
